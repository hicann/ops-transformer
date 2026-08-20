/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "test_mla_prolog_v3_tiling_common.h"

// 950的MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR场景正常case，BS合轴; cacheMode = PA_BSND; T = 33
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test68)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{33, 7168}, {33, 7168}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                     // token_x 0
            {{{7168, 1536}, {7168, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 1536}, {1536, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},         // weight_uq_qr 2
            {{{8, 128, 512}, {8, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                        // weight_uk 3
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                                      // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                        // rmsnorm_gamma_ckv 6
            {{{33, 64}, {33, 64}}, ge::DT_BF16, ge::FORMAT_ND},                                  // rope_sin 7
            {{{33, 64}, {33, 64}}, ge::DT_BF16, ge::FORMAT_ND},                                  // rope_cos 8
            {{{254, 128, 1, 512}, {254, 128, 1, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},     // kv_cache 9
            {{{254, 128, 1, 64}, {254, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                // kr_cache 10
            {{{33}, {33}}, ge::DT_INT64, ge::FORMAT_ND},                                         // cache_index 11
            {{{33, 224}, {33, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                         // dequant_scale_x 12
            {{{1536, 224}, {1536, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // dequant_scale_w_dq 13
            {{{1536, 48}, {1536, 48}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},   // dequant_scale_w_uq_qr 14
            {{{576, 224}, {576, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},   // dequant_scale_w_dkv_kr 15
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // actual_seq_len 19
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // k_nope_clip_alpha 20
        },
        {
            {{{33, 8, 512}, {33, 8, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},             // query
            {{{33, 8, 64}, {33, 8, 64}}, ge::DT_BF16, ge::FORMAT_ND},                        // query_rope
            {{{254, 128, 1, 512}, {254, 128, 1, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // kv_cache
            {{{254, 128, 1, 64}, {254, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},            // kr_cache
            {{{33, 8, 1}, {33, 8, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                               // query_norm
            {{{0}, {0}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                                 // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.5632081204399622)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.9777927042497497)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3933729;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 950的MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR场景正常case，BS合轴; cacheMode = PA_NZ; T = 128
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test69)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{128, 7168}, {128, 7168}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                   // token_x 0
            {{{7168, 1536}, {7168, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 768}, {1536, 768}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},           // weight_uq_qr 2
            {{{4, 128, 512}, {4, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                        // weight_uk 3
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                                      // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                        // rmsnorm_gamma_ckv 6
            {{{128, 64}, {128, 64}}, ge::DT_BF16, ge::FORMAT_ND},                                // rope_sin 7
            {{{128, 64}, {128, 64}}, ge::DT_BF16, ge::FORMAT_ND},                                // rope_cos 8
            {{{440, 128, 1, 512}, {440, 128, 1, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},     // kv_cache 9
            {{{440, 128, 1, 64}, {440, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                // kr_cache 10
            {{{128}, {128}}, ge::DT_INT64, ge::FORMAT_ND},                                       // cache_index 11
            {{{128, 224}, {128, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                       // dequant_scale_x 12
            {{{1536, 224}, {1536, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // dequant_scale_w_dq 13
            {{{768, 48}, {768, 48}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},     // dequant_scale_w_uq_qr 14
            {{{576, 224}, {576, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},   // dequant_scale_w_dkv_kr 15
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // actual_seq_len 19
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // k_nope_clip_alpha 20
        },
        {
            {{{128, 4, 512}, {128, 4, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},           // query
            {{{128, 4, 64}, {128, 4, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // query_rope
            {{{440, 128, 1, 512}, {440, 128, 1, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // kv_cache
            {{{440, 128, 1, 64}, {440, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},            // kr_cache
            {{{128, 4, 1}, {128, 4, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                               // query_norm
            {{{0}, {0}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                                 // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.2867307068127365)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.4978298916484819)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_NZ")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3933730;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 950的MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR, query_norm_flag = true场景正常case，cache_mode：PA_BSND
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test70)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 1, 7168}, {8, 1, 7168}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                 // token_x 0
            {{{7168, 1536}, {7168, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 24576}, {1536, 24576}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},       // weight_uq_qr 2
            {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                    // weight_uk 3
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                                      // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                        // rmsnorm_gamma_ckv 6
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                              // rope_sin 7
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                              // rope_cos 8
            {{{1, 16, 1, 512}, {1, 16, 1, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},           // kv_cache 9
            {{{1, 16, 1, 64}, {1, 16, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // kr_cache 10
            {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},                                     // cache_index 11
            {{{8, 224}, {8, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                           // dequant_scale_x 12
            {{{1536, 224}, {1536, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // dequant_scale_w_dq 13
            {{{24576, 48}, {24576, 48}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // dequant_scale_w_uq_qr 14
            {{{576, 224}, {576, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},   // dequant_scale_w_dkv_kr 15
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // actual_seq_len 19
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // k_nope_clip_alpha 20
        },
        {
            {{{8, 1, 128, 512}, {8, 1, 128, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // query
            {{{8, 1, 128, 64}, {8, 1, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},            // query_rope
            {{{1, 16, 1, 512}, {1, 16, 1, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},   // kv_cache
            {{{1, 16, 1, 64}, {1, 16, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},              // kr_cache
            {{{8, 128, 1}, {8, 128, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // dequant_scale_q_nope
            {{{8, 1, 1536}, {8, 1, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},         // query_norm
            {{{8, 48}, {8, 48}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                     // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0022971812167027)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00235037235057241)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3933729;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 950的MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR, query_norm_flag = true场景异常case，cache_mode：PA_BSND, query_norm dtype
// must be DT_FLOAT8_E4M3FN, actually is DT_BF16
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test71)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 1, 7168}, {8, 1, 7168}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                 // token_x 0
            {{{7168, 1536}, {7168, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 24576}, {1536, 24576}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},       // weight_uq_qr 2
            {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                    // weight_uk 3
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                                      // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                        // rmsnorm_gamma_ckv 6
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                              // rope_sin 7
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                              // rope_cos 8
            {{{1, 16, 1, 512}, {1, 16, 1, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},           // kv_cache 9
            {{{1, 16, 1, 64}, {1, 16, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // kr_cache 10
            {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},                                     // cache_index 11
            {{{8, 224}, {8, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                           // dequant_scale_x 12
            {{{1536, 224}, {1536, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // dequant_scale_w_dq 13
            {{{24576, 48}, {24576, 48}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // dequant_scale_w_uq_qr 14
            {{{576, 224}, {576, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},   // dequant_scale_w_dkv_kr 15
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // actual_seq_len 19
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // k_nope_clip_alpha 20
        },
        {
            {{{8, 1, 128, 512}, {8, 1, 128, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // query
            {{{8, 1, 128, 64}, {8, 1, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},            // query_rope
            {{{1, 16, 1, 512}, {1, 16, 1, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},   // kv_cache
            {{{1, 16, 1, 64}, {1, 16, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},              // kr_cache
            {{{8, 128, 1}, {8, 128, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // dequant_scale_q_nope
            {{{8, 1, 1536}, {8, 1, 1536}}, ge::DT_BF16, ge::FORMAT_ND},                  // query_norm
            {{{8, 48}, {8, 48}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                     // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0022971812167027)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00235037235057241)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3933729;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 950的MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR, query_norm_flag = true场景异常case，cache_mode：PA_BSND,
// dequant_scale_q_norm dtype must be DT_FLOAT8_E8M0, actually is DT_BF16
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test72)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 1, 7168}, {8, 1, 7168}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                 // token_x 0
            {{{7168, 1536}, {7168, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 24576}, {1536, 24576}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},       // weight_uq_qr 2
            {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                    // weight_uk 3
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                                      // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                        // rmsnorm_gamma_ckv 6
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                              // rope_sin 7
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                              // rope_cos 8
            {{{1, 16, 1, 512}, {1, 16, 1, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},           // kv_cache 9
            {{{1, 16, 1, 64}, {1, 16, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // kr_cache 10
            {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},                                     // cache_index 11
            {{{8, 224}, {8, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                           // dequant_scale_x 12
            {{{1536, 224}, {1536, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // dequant_scale_w_dq 13
            {{{24576, 48}, {24576, 48}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // dequant_scale_w_uq_qr 14
            {{{576, 224}, {576, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},   // dequant_scale_w_dkv_kr 15
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // actual_seq_len 19
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // k_nope_clip_alpha 20
        },
        {
            {{{8, 1, 128, 512}, {8, 1, 128, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // query
            {{{8, 1, 128, 64}, {8, 1, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},            // query_rope
            {{{1, 16, 1, 512}, {1, 16, 1, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},   // kv_cache
            {{{1, 16, 1, 64}, {1, 16, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},              // kr_cache
            {{{8, 128, 1}, {8, 128, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // dequant_scale_q_nope
            {{{8, 1, 1536}, {8, 1, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},         // query_norm
            {{{8, 48}, {8, 48}}, ge::DT_BF16, ge::FORMAT_ND},                            // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0022971812167027)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00235037235057241)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3933729;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 950的MXFP8_FULL_QUANT_KV_QUANT_PER_TILE场景正常case，cache_mode：PA_BSND
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test73)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 1, 7168}, {8, 1, 7168}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                 // token_x 0
            {{{7168, 1536}, {7168, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 24576}, {1536, 24576}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},       // weight_uq_qr 2
            {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                    // weight_uk 3
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                                      // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                        // rmsnorm_gamma_ckv 6
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                              // rope_sin 7
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                              // rope_cos 8
            {{{1, 16, 1, 656}, {1, 16, 1, 656}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},           // kv_cache 9
            {{{0}, {0}}, ge::DT_BF16, ge::FORMAT_ND},                                            // kr_cache 10
            {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},                                     // cache_index 11
            {{{8, 224}, {8, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                           // dequant_scale_x 12
            {{{1536, 224}, {1536, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // dequant_scale_w_dq 13
            {{{24576, 48}, {24576, 48}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // dequant_scale_w_uq_qr 14
            {{{576, 224}, {576, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},   // dequant_scale_w_dkv_kr 15
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // actual_seq_len 19
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // k_nope_clip_alpha 20
        },
        {
            {{{8, 1, 128, 512}, {8, 1, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},        // query
            {{{8, 1, 128, 64}, {8, 1, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},          // query_rope
            {{{1, 16, 1, 656}, {1, 16, 1, 656}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // kv_cache
            {{{0}, {0}}, ge::DT_BF16, ge::FORMAT_ND},                                  // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                         // query_norm
            {{{0}, {0}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                           // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0022971812167027)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00235037235057241)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3933793;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 950的MXFP8_FULL_QUANT_KV_QUANT_PER_TILE场景异常case，Only support cacheMode PA_BSND, actually is PA_NZ
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test74)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 1, 7168}, {8, 1, 7168}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                 // token_x 0
            {{{7168, 1536}, {7168, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 24576}, {1536, 24576}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},       // weight_uq_qr 2
            {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                    // weight_uk 3
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                                      // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                        // rmsnorm_gamma_ckv 6
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                              // rope_sin 7
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                              // rope_cos 8
            {{{1, 16, 1, 656}, {1, 16, 1, 656}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},           // kv_cache 9
            {{{0}, {0}}, ge::DT_BF16, ge::FORMAT_ND},                                            // kr_cache 10
            {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},                                     // cache_index 11
            {{{8, 224}, {8, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                           // dequant_scale_x 12
            {{{1536, 224}, {1536, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // dequant_scale_w_dq 13
            {{{24576, 48}, {24576, 48}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // dequant_scale_w_uq_qr 14
            {{{576, 224}, {576, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},   // dequant_scale_w_dkv_kr 15
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // actual_seq_len 19
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // k_nope_clip_alpha 20
        },
        {
            {{{8, 1, 128, 512}, {8, 1, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},        // query
            {{{8, 1, 128, 64}, {8, 1, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},          // query_rope
            {{{1, 16, 1, 656}, {1, 16, 1, 656}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // kv_cache
            {{{0}, {0}}, ge::DT_BF16, ge::FORMAT_ND},                                  // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                         // query_norm
            {{{0}, {0}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                           // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0022971812167027)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00235037235057241)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_NZ")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3933729;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 950的MXFP8_FULL_QUANT_KV_QUANT_PER_TILE场景异常case，When weightQuantMode == 3, kvQuantMode must be within {0, 1, 3},
// actually is 2.
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test75)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 1, 7168}, {8, 1, 7168}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                 // token_x 0
            {{{7168, 1536}, {7168, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 24576}, {1536, 24576}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},       // weight_uq_qr 2
            {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                    // weight_uk 3
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                                      // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                        // rmsnorm_gamma_ckv 6
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                              // rope_sin 7
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                              // rope_cos 8
            {{{1, 16, 1, 656}, {1, 16, 1, 656}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},           // kv_cache 9
            {{{0}, {0}}, ge::DT_BF16, ge::FORMAT_ND},                                            // kr_cache 10
            {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},                                     // cache_index 11
            {{{8, 224}, {8, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                           // dequant_scale_x 12
            {{{1536, 224}, {1536, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // dequant_scale_w_dq 13
            {{{24576, 48}, {24576, 48}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // dequant_scale_w_uq_qr 14
            {{{576, 224}, {576, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},   // dequant_scale_w_dkv_kr 15
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // actual_seq_len 19
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // k_nope_clip_alpha 20
        },
        {
            {{{8, 1, 128, 512}, {8, 1, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},        // query
            {{{8, 1, 128, 64}, {8, 1, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},          // query_rope
            {{{1, 16, 1, 656}, {1, 16, 1, 656}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // kv_cache
            {{{0}, {0}}, ge::DT_BF16, ge::FORMAT_ND},                                  // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                         // query_norm
            {{{0}, {0}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                           // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0022971812167027)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00235037235057241)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_NZ")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3933729;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 950的MXFP8_FULL_QUANT_KV_QUANT_PER_TILE场景异常case，The ckvkrRepoMode expected 1, but got 0. The quantScaleRepoMode
// expected 1, but got 0.
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test76)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 1, 7168}, {8, 1, 7168}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                 // token_x 0
            {{{7168, 1536}, {7168, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 24576}, {1536, 24576}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},       // weight_uq_qr 2
            {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                    // weight_uk 3
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                                      // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                        // rmsnorm_gamma_ckv 6
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                              // rope_sin 7
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                              // rope_cos 8
            {{{1, 16, 1, 656}, {1, 16, 1, 656}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},           // kv_cache 9
            {{{0}, {0}}, ge::DT_BF16, ge::FORMAT_ND},                                            // kr_cache 10
            {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},                                     // cache_index 11
            {{{8, 224}, {8, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                           // dequant_scale_x 12
            {{{1536, 224}, {1536, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // dequant_scale_w_dq 13
            {{{24576, 48}, {24576, 48}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // dequant_scale_w_uq_qr 14
            {{{576, 224}, {576, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},   // dequant_scale_w_dkv_kr 15
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // actual_seq_len 19
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // k_nope_clip_alpha 20
        },
        {
            {{{8, 1, 128, 512}, {8, 1, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},        // query
            {{{8, 1, 128, 64}, {8, 1, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},          // query_rope
            {{{1, 16, 1, 656}, {1, 16, 1, 656}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // kv_cache
            {{{0}, {0}}, ge::DT_BF16, ge::FORMAT_ND},                                  // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                         // query_norm
            {{{0}, {0}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                           // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0022971812167027)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00235037235057241)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3933729;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 950的MXFP8_FULL_QUANT_KV_QUANT_PER_TILE场景异常case，The queryQuantMode expected 0, but got 1.
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test77)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 1, 7168}, {8, 1, 7168}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                 // token_x 0
            {{{7168, 1536}, {7168, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 24576}, {1536, 24576}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},       // weight_uq_qr 2
            {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                    // weight_uk 3
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                                      // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                        // rmsnorm_gamma_ckv 6
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                              // rope_sin 7
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                              // rope_cos 8
            {{{1, 16, 1, 656}, {1, 16, 1, 656}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},           // kv_cache 9
            {{{0}, {0}}, ge::DT_BF16, ge::FORMAT_ND},                                            // kr_cache 10
            {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},                                     // cache_index 11
            {{{8, 224}, {8, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                           // dequant_scale_x 12
            {{{1536, 224}, {1536, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // dequant_scale_w_dq 13
            {{{24576, 48}, {24576, 48}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // dequant_scale_w_uq_qr 14
            {{{576, 224}, {576, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},   // dequant_scale_w_dkv_kr 15
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // actual_seq_len 19
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // k_nope_clip_alpha 20
        },
        {
            {{{8, 1, 128, 512}, {8, 1, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},        // query
            {{{8, 1, 128, 64}, {8, 1, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},          // query_rope
            {{{1, 16, 1, 656}, {1, 16, 1, 656}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // kv_cache
            {{{0}, {0}}, ge::DT_BF16, ge::FORMAT_ND},                                  // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                         // query_norm
            {{{0}, {0}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                           // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0022971812167027)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00235037235057241)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3933729;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 950的MXFP8_FULL_QUANT_KV_QUANT_PER_TILE场景异常case，DtileSize allows only 656, got 512.
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test78)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 1, 7168}, {8, 1, 7168}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                 // token_x 0
            {{{7168, 1536}, {7168, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 24576}, {1536, 24576}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},       // weight_uq_qr 2
            {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                    // weight_uk 3
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                                      // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                        // rmsnorm_gamma_ckv 6
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                              // rope_sin 7
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                              // rope_cos 8
            {{{1, 16, 1, 512}, {1, 16, 1, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},           // kv_cache 9
            {{{0}, {0}}, ge::DT_BF16, ge::FORMAT_ND},                                            // kr_cache 10
            {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},                                     // cache_index 11
            {{{8, 224}, {8, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                           // dequant_scale_x 12
            {{{1536, 224}, {1536, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // dequant_scale_w_dq 13
            {{{24576, 48}, {24576, 48}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // dequant_scale_w_uq_qr 14
            {{{576, 224}, {576, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},   // dequant_scale_w_dkv_kr 15
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // actual_seq_len 19
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // k_nope_clip_alpha 20
        },
        {
            {{{8, 1, 128, 512}, {8, 1, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},        // query
            {{{8, 1, 128, 64}, {8, 1, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},          // query_rope
            {{{1, 16, 1, 512}, {1, 16, 1, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // kv_cache
            {{{0}, {0}}, ge::DT_BF16, ge::FORMAT_ND},                                  // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                         // query_norm
            {{{0}, {0}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                           // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0022971812167027)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00235037235057241)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3933729;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 950的MXFP8_FULL_QUANT_KV_QUANT_PER_TILE场景正常case
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test79)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 1, 7168}, {8, 1, 7168}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                 // token_x 0
            {{{7168, 1536}, {7168, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 24576}, {1536, 24576}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},       // weight_uq_qr 2
            {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                    // weight_uk 3
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                                      // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                        // rmsnorm_gamma_ckv 6
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                              // rope_sin 7
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                              // rope_cos 8
            {{{1, 16, 1, 656}, {1, 16, 1, 656}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},           // kv_cache 9
            {{{0, 16, 1}, {0, 16, 1}}, ge::DT_BF16, ge::FORMAT_ND},                              // kr_cache 10
            {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},                                     // cache_index 11
            {{{8, 224}, {8, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                           // dequant_scale_x 12
            {{{1536, 224}, {1536, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // dequant_scale_w_dq 13
            {{{24576, 48}, {24576, 48}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // dequant_scale_w_uq_qr 14
            {{{576, 224}, {576, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},   // dequant_scale_w_dkv_kr 15
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // actual_seq_len 19
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // k_nope_clip_alpha 20
        },
        {
            {{{8, 1, 128, 512}, {8, 1, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},        // query
            {{{8, 1, 128, 64}, {8, 1, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},          // query_rope
            {{{1, 16, 1, 656}, {1, 16, 1, 656}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // kv_cache
            {{{0, 16, 1}, {0, 16, 1}}, ge::DT_BF16, ge::FORMAT_ND},                    // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                         // query_norm
            {{{0}, {0}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                           // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0022971812167027)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00235037235057241)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3933793;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 950的MXFP8_FULL_QUANT_KV_QUANT_PER_TILE场景异常case，When weightQuantMode == 0, kvQuantMode must be within {0},
// actually is 3.
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test80)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 1, 7168}, {8, 1, 7168}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                 // token_x 0
            {{{7168, 1536}, {7168, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 24576}, {1536, 24576}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},       // weight_uq_qr 2
            {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                    // weight_uk 3
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                                      // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                        // rmsnorm_gamma_ckv 6
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                              // rope_sin 7
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                              // rope_cos 8
            {{{1, 16, 1, 656}, {1, 16, 1, 656}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},           // kv_cache 9
            {{{0}, {0}}, ge::DT_BF16, ge::FORMAT_ND},                                            // kr_cache 10
            {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},                                     // cache_index 11
            {{{8, 224}, {8, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                           // dequant_scale_x 12
            {{{1536, 224}, {1536, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // dequant_scale_w_dq 13
            {{{24576, 48}, {24576, 48}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // dequant_scale_w_uq_qr 14
            {{{576, 224}, {576, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},   // dequant_scale_w_dkv_kr 15
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // actual_seq_len 19
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // k_nope_clip_alpha 20
        },
        {
            {{{8, 1, 128, 512}, {8, 1, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},        // query
            {{{8, 1, 128, 64}, {8, 1, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},          // query_rope
            {{{1, 16, 1, 656}, {1, 16, 1, 656}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // kv_cache
            {{{0}, {0}}, ge::DT_BF16, ge::FORMAT_ND},                                  // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                         // query_norm
            {{{0}, {0}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                           // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0022971812167027)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00235037235057241)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3933729;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 非量化 950的场景正常case：qc_qr_scale = 0.4, kc_scale = 0.5
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test81)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{1, 129, 7168}, {1, 129, 7168}}, ge::DT_BF16, ge::FORMAT_ND},                         // token_x
            {{{7168, 1536}, {7168, 1536}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},                     // weight_dq
            {{{1536, 8 * (128 + 64)}, {1536, 8 * (128 + 64)}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ}, // weight_uq_qr
            {{{8, 128, 512}, {8, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                           // weight_uk
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},             // weight_dkv_kr
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                                         // rmsnorm_gamma_cq
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                           // rmsnorm_gamma_ckv
            {{{1, 129, 64}, {1, 129, 64}}, ge::DT_BF16, ge::FORMAT_ND},                             // rope_sin
            {{{1, 129, 64}, {1, 129, 64}}, ge::DT_BF16, ge::FORMAT_ND},                             // rope_cos
            {{{2, 128, 1, 512}, {2, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},                     // kv_cache
            {{{2, 128, 1, 64}, {2, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                       // kr_cache
            {{{1, 129}, {1, 129}}, ge::DT_INT64, ge::FORMAT_ND},                                    // cache_index
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                                // dequant_scale_x
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dequant_scale_w_dq
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dequant_scale_w_uq_qr
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dequant_scale_w_dkv_kr
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // quant_scale_ckv
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // quant_scale_ckr
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // smooth_scales_cq
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // actual_seq_len
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // k_nope_clip_alpha
        },
        {
            {{{1, 129, 8, 512}, {1, 129, 8, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // query
            {{{1, 129, 8, 64}, {1, 129, 8, 64}}, ge::DT_BF16, ge::FORMAT_ND},   // query_rope
            {{{2, 128, 1, 512}, {2, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // kv_cache
            {{{2, 128, 1, 64}, {2, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},   // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                          // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_BF16, ge::FORMAT_ND},                           // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                           // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_NZ")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.4f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.5f)},
        },
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3932178;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 非量化 950的场景正常case cache_mode:PA_BSND BSH
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test82)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 1, 7168}, {8, 1, 7168}}, ge::DT_BF16, ge::FORMAT_ND},                               // token_x
            {{{7168, 1536}, {7168, 1536}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},                       // weight_dq
            {{{1536, 32 * (128 + 64)}, {1536, 32 * (128 + 64)}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ}, // weight_uq_qr
            {{{32, 128, 512}, {32, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                           // weight_uk
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},               // weight_dkv_kr
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                       // rmsnorm_gamma_cq
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                         // rmsnorm_gamma_ckv
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},               // rope_sin
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},               // rope_cos
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},   // kr_cache
            {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},                      // cache_index
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // dequant_scale_x
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // dequant_scale_w_dq
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // dequant_scale_w_uq_qr
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // dequant_scale_w_dkv_kr
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // quant_scale_ckv
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // quant_scale_ckr
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // smooth_scales_cq
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                              // actual_seq_len
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // k_nope_clip_alpha
        },
        {
            {{{8, 1, 32, 512}, {8, 1, 32, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // query
            {{{8, 1, 32, 64}, {8, 1, 32, 64}}, ge::DT_BF16, ge::FORMAT_ND},       // query_rope
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},   // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_BF16, ge::FORMAT_ND},                             // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                             // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3932177;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 非量化 950的场景正常case cache_mode:PA_NZ TND
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test83)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_BF16, ge::FORMAT_ND},                                     // token_x
            {{{7168, 1536}, {7168, 1536}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},                       // weight_dq
            {{{1536, 32 * (128 + 64)}, {1536, 32 * (128 + 64)}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ}, // weight_uq_qr
            {{{32, 128, 512}, {32, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                           // weight_uk
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},               // weight_dkv_kr
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                       // rmsnorm_gamma_cq
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                         // rmsnorm_gamma_ckv
            {{{8, 64}, {8, 64}}, ge::DT_BF16, ge::FORMAT_ND},                     // rope_sin
            {{{8, 64}, {8, 64}}, ge::DT_BF16, ge::FORMAT_ND},                     // rope_cos
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},   // kr_cache
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND},                            // cache_index
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // dequant_scale_x
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // dequant_scale_w_dq
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // dequant_scale_w_uq_qr
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // dequant_scale_w_dkv_kr
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // quant_scale_ckv
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // quant_scale_ckr
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // smooth_scales_cq
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                              // actual_seq_len
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // k_nope_clip_alpha
        },
        {
            {{{8, 32, 512}, {8, 32, 512}}, ge::DT_BF16, ge::FORMAT_ND},           // query
            {{{8, 32, 64}, {8, 32, 64}}, ge::DT_BF16, ge::FORMAT_ND},             // query_rope
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},   // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_BF16, ge::FORMAT_ND},                             // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                             // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_NZ")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3932178;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 非量化 950的场景正常case cache_mode:PA_BSND TND
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test84)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_BF16, ge::FORMAT_ND},                                     // token_x
            {{{7168, 1536}, {7168, 1536}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},                       // weight_dq
            {{{1536, 32 * (128 + 64)}, {1536, 32 * (128 + 64)}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ}, // weight_uq_qr
            {{{32, 128, 512}, {32, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                           // weight_uk
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},               // weight_dkv_kr
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                       // rmsnorm_gamma_cq
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                         // rmsnorm_gamma_ckv
            {{{8, 64}, {8, 64}}, ge::DT_BF16, ge::FORMAT_ND},                     // rope_sin
            {{{8, 64}, {8, 64}}, ge::DT_BF16, ge::FORMAT_ND},                     // rope_cos
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},   // kr_cache
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND},                            // cache_index
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // dequant_scale_x
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // dequant_scale_w_dq
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // dequant_scale_w_uq_qr
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // dequant_scale_w_dkv_kr
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // quant_scale_ckv
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // quant_scale_ckr
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // smooth_scales_cq
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                              // actual_seq_len
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // k_nope_clip_alpha
        },
        {
            {{{8, 32, 512}, {8, 32, 512}}, ge::DT_BF16, ge::FORMAT_ND},           // query
            {{{8, 32, 64}, {8, 32, 64}}, ge::DT_BF16, ge::FORMAT_ND},             // query_rope
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},   // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_BF16, ge::FORMAT_ND},                             // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                             // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3932177;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 非量化 950的场景正常case cache_mode:PA_NZ BSH
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test85)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 1, 7168}, {8, 1, 7168}}, ge::DT_BF16, ge::FORMAT_ND},                               // token_x
            {{{7168, 1536}, {7168, 1536}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},                       // weight_dq
            {{{1536, 32 * (128 + 64)}, {1536, 32 * (128 + 64)}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ}, // weight_uq_qr
            {{{32, 128, 512}, {32, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                           // weight_uk
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},               // weight_dkv_kr
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                       // rmsnorm_gamma_cq
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                         // rmsnorm_gamma_ckv
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},               // rope_sin
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},               // rope_cos
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},   // kr_cache
            {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},                      // cache_index
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // dequant_scale_x
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // dequant_scale_w_dq
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // dequant_scale_w_uq_qr
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // dequant_scale_w_dkv_kr
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // quant_scale_ckv
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // quant_scale_ckr
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // smooth_scales_cq
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                              // actual_seq_len
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // k_nope_clip_alpha
        },
        {
            {{{8, 1, 32, 512}, {8, 1, 32, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // query
            {{{8, 1, 32, 64}, {8, 1, 32, 64}}, ge::DT_BF16, ge::FORMAT_ND},       // query_rope
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},   // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_BF16, ge::FORMAT_ND},                             // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                             // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_NZ")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3932178;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 950的MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR场景正常case，PA_NZ TND
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test86)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{33, 7168}, {33, 7168}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                     // token_x 0
            {{{7168, 1536}, {7168, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 1536}, {1536, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},         // weight_uq_qr 2
            {{{8, 128, 512}, {8, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                        // weight_uk 3
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                                      // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                        // rmsnorm_gamma_ckv 6
            {{{33, 64}, {33, 64}}, ge::DT_BF16, ge::FORMAT_ND},                                  // rope_sin 7
            {{{33, 64}, {33, 64}}, ge::DT_BF16, ge::FORMAT_ND},                                  // rope_cos 8
            {{{254, 128, 1, 512}, {254, 128, 1, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},     // kv_cache 9
            {{{254, 128, 1, 64}, {254, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                // kr_cache 10
            {{{33}, {33}}, ge::DT_INT64, ge::FORMAT_ND},                                         // cache_index 11
            {{{33, 224}, {33, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                         // dequant_scale_x 12
            {{{1536, 224}, {1536, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // dequant_scale_w_dq 13
            {{{1536, 48}, {1536, 48}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},   // dequant_scale_w_uq_qr 14
            {{{576, 224}, {576, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},   // dequant_scale_w_dkv_kr 15
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // actual_seq_len 19
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // k_nope_clip_alpha 20
        },
        {
            {{{33, 8, 512}, {33, 8, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},             // query
            {{{33, 8, 64}, {33, 8, 64}}, ge::DT_BF16, ge::FORMAT_ND},                        // query_rope
            {{{254, 128, 1, 512}, {254, 128, 1, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // kv_cache
            {{{254, 128, 1, 64}, {254, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},            // kr_cache
            {{{33, 8, 1}, {33, 8, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                               // query_norm
            {{{0}, {0}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                                 // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0022971812167027)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00235037235057241)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_NZ")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3933730;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 950的MXFP8_FULL_QUANT_KV_NO_QUANT场景异常场景case，可选输入(dequant_scale_x dequant_scale_w_dq dequant_scale_w_uq_qr
// dequant_scale_w_dkv_kr) dtype传错
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test87)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{33, 7168}, {33, 7168}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                     // token_x 0
            {{{7168, 1536}, {7168, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 1536}, {1536, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},         // weight_uq_qr 2
            {{{8, 128, 512}, {8, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                        // weight_uk 3
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                                      // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                        // rmsnorm_gamma_ckv 6
            {{{33, 64}, {33, 64}}, ge::DT_BF16, ge::FORMAT_ND},                                  // rope_sin 7
            {{{33, 64}, {33, 64}}, ge::DT_BF16, ge::FORMAT_ND},                                  // rope_cos 8
            {{{254, 128, 1, 512}, {254, 128, 1, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},     // kv_cache 9
            {{{254, 128, 1, 64}, {254, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                // kr_cache 10
            {{{33}, {33}}, ge::DT_INT64, ge::FORMAT_ND},                                         // cache_index 11
            {{{33, 224}, {33, 224}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale_x 12
            {{{1536, 224}, {1536, 224}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // dequant_scale_w_dq 13
            {{{1536, 48}, {1536, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // dequant_scale_w_uq_qr 14
            {{{576, 224}, {576, 224}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // dequant_scale_w_dkv_kr 15
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                     // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                     // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                     // actual_seq_len 19
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                     // k_nope_clip_alpha 20
        },
        {
            {{{33, 8, 512}, {33, 8, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},             // query
            {{{33, 8, 64}, {33, 8, 64}}, ge::DT_BF16, ge::FORMAT_ND},                        // query_rope
            {{{254, 128, 1, 512}, {254, 128, 1, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // kv_cache
            {{{254, 128, 1, 64}, {254, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},            // kr_cache
            {{{33, 8, 1}, {33, 8, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                               // query_norm
            {{{0}, {0}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                                 // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0022971812167027)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00235037235057241)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3933730;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 950的MXFP8_FULL_QUANT_KV_NO_QUANT场景异常场景case，可选输入(dequant_scale_x dequant_scale_w_dq dequant_scale_w_uq_qr
// dequant_scale_w_dkv_kr) shape传错
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test88)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{33, 1, 7168}, {33, 1, 7168}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},               // token_x 0
            {{{7168, 1536}, {7168, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 1536}, {1536, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},         // weight_uq_qr 2
            {{{8, 128, 512}, {8, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                        // weight_uk 3
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                                      // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                        // rmsnorm_gamma_ckv 6
            {{{33, 1, 64}, {33, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                            // rope_sin 7
            {{{33, 1, 64}, {33, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                            // rope_cos 8
            {{{254, 128, 1, 512}, {254, 128, 1, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},     // kv_cache 9
            {{{254, 128, 1, 64}, {254, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                // kr_cache 10
            {{{33, 1}, {33, 1}}, ge::DT_INT64, ge::FORMAT_ND},                                   // cache_index 11
            {{{33, 7168}, {33, 7168}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                       // dequant_scale_x 12
            {{{1536, 22}, {1536, 22}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // dequant_scale_w_dq 13
            {{{1536, 4}, {1536, 4}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},   // dequant_scale_w_uq_qr 14
            {{{576, 22}, {576, 22}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},   // dequant_scale_w_dkv_kr 15
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                     // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                       // actual_seq_len 19
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // k_nope_clip_alpha 20
        },
        {
            {{{33, 1, 8, 512}, {33, 1, 8, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},       // query
            {{{33, 1, 8, 64}, {33, 1, 8, 64}}, ge::DT_BF16, ge::FORMAT_ND},                  // query_rope
            {{{254, 128, 1, 512}, {254, 128, 1, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // kv_cache
            {{{254, 128, 1, 64}, {254, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},            // kr_cache
            {{{33, 8, 1}, {33, 8, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                               // query_norm
            {{{0}, {0}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                                 // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0022971812167027)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00235037235057241)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3933730;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 950的MXFP8_FULL_QUANT_KV_NO_QUANT场景异常场景，可选输入SmoothScale不支持
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test89)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{33, 7168}, {33, 7168}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                     // token_x 0
            {{{7168, 1536}, {7168, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 1536}, {1536, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},         // weight_uq_qr 2
            {{{8, 128, 512}, {8, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                        // weight_uk 3
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                                      // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                        // rmsnorm_gamma_ckv 6
            {{{33, 64}, {33, 64}}, ge::DT_BF16, ge::FORMAT_ND},                                  // rope_sin 7
            {{{33, 64}, {33, 64}}, ge::DT_BF16, ge::FORMAT_ND},                                  // rope_cos 8
            {{{254, 128, 1, 512}, {254, 128, 1, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},     // kv_cache 9
            {{{254, 128, 1, 64}, {254, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                // kr_cache 10
            {{{33}, {33}}, ge::DT_INT64, ge::FORMAT_ND},                                         // cache_index 11
            {{{33, 224}, {33, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                         // dequant_scale_x 12
            {{{1536, 224}, {1536, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // dequant_scale_w_dq 13
            {{{1536, 48}, {1536, 48}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},   // dequant_scale_w_uq_qr 14
            {{{576, 224}, {576, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},   // dequant_scale_w_dkv_kr 15
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckr 17
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // actual_seq_len 19
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // k_nope_clip_alpha 20
        },
        {
            {{{33, 8, 512}, {33, 8, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},             // query
            {{{33, 8, 64}, {33, 8, 64}}, ge::DT_BF16, ge::FORMAT_ND},                        // query_rope
            {{{254, 128, 1, 512}, {254, 128, 1, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // kv_cache
            {{{254, 128, 1, 64}, {254, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},            // kr_cache
            {{{33, 8, 1}, {33, 8, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                               // query_norm
            {{{0}, {0}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                                 // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0022971812167027)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00235037235057241)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3933730;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 950的MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR场景异常case，可选输入(dequant_scale_x dequant_scale_w_dq
// dequant_scale_w_uq_qr dequant_scale_w_dkv_kr) dtype传错
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test90)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{33, 7168}, {33, 7168}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                     // token_x 0
            {{{7168, 1536}, {7168, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 1536}, {1536, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},         // weight_uq_qr 2
            {{{8, 128, 512}, {8, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                        // weight_uk 3
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                                      // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                        // rmsnorm_gamma_ckv 6
            {{{33, 64}, {33, 64}}, ge::DT_BF16, ge::FORMAT_ND},                                  // rope_sin 7
            {{{33, 64}, {33, 64}}, ge::DT_BF16, ge::FORMAT_ND},                                  // rope_cos 8
            {{{254, 128, 1, 512}, {254, 128, 1, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},     // kv_cache 9
            {{{254, 128, 1, 64}, {254, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                // kr_cache 10
            {{{33}, {33}}, ge::DT_INT64, ge::FORMAT_ND},                                         // cache_index 11
            {{{33, 224}, {33, 224}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale_x 12
            {{{1536, 224}, {1536, 224}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // dequant_scale_w_dq 13
            {{{1536, 48}, {1536, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // dequant_scale_w_uq_qr 14
            {{{576, 224}, {576, 224}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // dequant_scale_w_dkv_kr 15
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                     // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                     // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                     // actual_seq_len 19
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                     // k_nope_clip_alpha 20
        },
        {
            {{{33, 8, 512}, {33, 8, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},             // query
            {{{33, 8, 64}, {33, 8, 64}}, ge::DT_BF16, ge::FORMAT_ND},                        // query_rope
            {{{254, 128, 1, 512}, {254, 128, 1, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // kv_cache
            {{{254, 128, 1, 64}, {254, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},            // kr_cache
            {{{33, 8, 1}, {33, 8, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                               // query_norm
            {{{0}, {0}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                                 // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0022971812167027)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00235037235057241)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3933730;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 950的MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR场景异常场景case，可选输入(dequant_scale_x) shape传错
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test91)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{33, 1, 7168}, {33, 1, 7168}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},               // token_x 0
            {{{7168, 1536}, {7168, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 1536}, {1536, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},         // weight_uq_qr 2
            {{{8, 128, 512}, {8, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                        // weight_uk 3
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                                      // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                        // rmsnorm_gamma_ckv 6
            {{{33, 1, 64}, {33, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                            // rope_sin 7
            {{{33, 1, 64}, {33, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                            // rope_cos 8
            {{{254, 128, 1, 512}, {254, 128, 1, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},     // kv_cache 9
            {{{254, 128, 1, 64}, {254, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                // kr_cache 10
            {{{33, 1}, {33, 1}}, ge::DT_INT64, ge::FORMAT_ND},                                   // cache_index 11
            {{{33, 7168}, {33, 7168}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                       // dequant_scale_x 12
            {{{1536, 224}, {1536, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // dequant_scale_w_dq 13
            {{{1536, 48}, {1536, 48}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},   // dequant_scale_w_uq_qr 14
            {{{576, 224}, {576, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},   // dequant_scale_w_dkv_kr 15
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // actual_seq_len 19
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // k_nope_clip_alpha 20
        },
        {
            {{{33, 1, 8, 512}, {33, 1, 8, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},       // query
            {{{33, 1, 8, 64}, {33, 1, 8, 64}}, ge::DT_BF16, ge::FORMAT_ND},                  // query_rope
            {{{254, 128, 1, 512}, {254, 128, 1, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // kv_cache
            {{{254, 128, 1, 64}, {254, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},            // kr_cache
            {{{33, 8, 1}, {33, 8, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                               // query_norm
            {{{0}, {0}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                                 // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0022971812167027)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00235037235057241)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3933730;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 950的MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR场景异常场景case，krCache dtype传错
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test92)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{33, 1, 7168}, {33, 1, 7168}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},               // token_x 0
            {{{7168, 1536}, {7168, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 1536}, {1536, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},         // weight_uq_qr 2
            {{{8, 128, 512}, {8, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                        // weight_uk 3
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                                      // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                        // rmsnorm_gamma_ckv 6
            {{{33, 1, 64}, {33, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                            // rope_sin 7
            {{{33, 1, 64}, {33, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                            // rope_cos 8
            {{{254, 128, 1, 512}, {254, 128, 1, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},     // kv_cache 9
            {{{254, 128, 1, 64}, {254, 128, 1, 64}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},       // kr_cache 10
            {{{33, 1}, {33, 1}}, ge::DT_INT64, ge::FORMAT_ND},                                   // cache_index 11
            {{{33, 224}, {33, 224}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                       // dequant_scale_x 12
            {{{1536, 224}, {1536, 224}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // dequant_scale_w_dq 13
            {{{1536, 48}, {1536, 48}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},   // dequant_scale_w_uq_qr 14
            {{{576, 224}, {576, 224}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},   // dequant_scale_w_dkv_kr 15
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                           // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                           // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                           // actual_seq_len 19
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                           // k_nope_clip_alpha 20
        },
        {
            {{{33, 1, 8, 512}, {33, 1, 8, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},       // query
            {{{33, 1, 8, 64}, {33, 1, 8, 64}}, ge::DT_BF16, ge::FORMAT_ND},                  // query_rope
            {{{254, 128, 1, 512}, {254, 128, 1, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // kv_cache
            {{{254, 128, 1, 64}, {254, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},            // kr_cache
            {{{33, 8, 1}, {33, 8, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                               // query_norm
            {{{0}, {0}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                                 // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0022971812167027)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00235037235057241)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3933730;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 非量化 950的场景异常case token_x dtype=DT_INT8
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test93)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_INT8, ge::FORMAT_ND},                                     // token_x
            {{{7168, 1536}, {7168, 1536}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},                       // weight_dq
            {{{1536, 32 * (128 + 64)}, {1536, 32 * (128 + 64)}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ}, // weight_uq_qr
            {{{32, 128, 512}, {32, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                           // weight_uk
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},               // weight_dkv_kr
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                       // rmsnorm_gamma_cq
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                         // rmsnorm_gamma_ckv
            {{{8, 64}, {8, 64}}, ge::DT_BF16, ge::FORMAT_ND},                     // rope_sin
            {{{8, 64}, {8, 64}}, ge::DT_BF16, ge::FORMAT_ND},                     // rope_cos
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND}, // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},   // kr_cache
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND},                            // cache_index
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // dequant_scale_x
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // dequant_scale_w_dq
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // dequant_scale_w_uq_qr
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // dequant_scale_w_dkv_kr
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // quant_scale_ckv
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // quant_scale_ckr
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // smooth_scales_cq
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                              // actual_seq_len
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // k_nope_clip_alpha
        },
        {
            {{{8, 32, 512}, {8, 32, 512}}, ge::DT_BF16, ge::FORMAT_ND},           // query
            {{{8, 32, 64}, {8, 32, 64}}, ge::DT_BF16, ge::FORMAT_ND},             // query_rope
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},   // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_BF16, ge::FORMAT_ND},                             // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                             // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_NZ")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3932178;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 950的MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR场景正常case，NZ格式
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test94)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 1, 7168}, {8, 1, 7168}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                   // token_x 0
            {{{7168, 1536}, {7168, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},           // weight_dq 1
            {{{1536, 24576}, {1536, 24576}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},         // weight_uq_qr 2
            {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                      // weight_uk 3
            {{{18, 448, 16, 32}, {18, 448, 16, 32}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                                        // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                              // rmsnorm_gamma_ckv 6
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                    // rope_sin 7
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                    // rope_cos 8
            {{{1, 16, 1, 512}, {1, 16, 1, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // kv_cache 9
            {{{1, 16, 1, 64}, {1, 16, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},            // kr_cache 10
            {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},                           // cache_index 11
            {{{8, 224}, {8, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                 // dequant_scale_x 12
            {{{1536, 224}, {1536, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},           // dequant_scale_w_dq 13
            {{{24576, 48}, {24576, 48}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},           // dequant_scale_w_uq_qr 14
            {{{576, 224}, {576, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},             // dequant_scale_w_dkv_kr 15
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                   // actual_seq_len 19
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // k_nope_clip_alpha 20
        },
        {
            {{{8, 1, 128, 512}, {8, 1, 128, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // query
            {{{8, 1, 128, 64}, {8, 1, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},            // query_rope
            {{{1, 16, 1, 512}, {1, 16, 1, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},   // kv_cache
            {{{1, 16, 1, 64}, {1, 16, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},              // kr_cache
            {{{8, 128, 1}, {8, 128, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                           // query_norm
            {{{0}, {0}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                             // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0022971812167027)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00235037235057241)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_NZ")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3933730;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 非量化 950的场景异常case weight_quant_mode取值错误
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test95)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_INT8, ge::FORMAT_ND},                                     // token_x
            {{{7168, 1536}, {7168, 1536}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},                       // weight_dq
            {{{1536, 32 * (128 + 64)}, {1536, 32 * (128 + 64)}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ}, // weight_uq_qr
            {{{32, 128, 512}, {32, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                           // weight_uk
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},               // weight_dkv_kr
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                       // rmsnorm_gamma_cq
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                         // rmsnorm_gamma_ckv
            {{{8, 64}, {8, 64}}, ge::DT_BF16, ge::FORMAT_ND},                     // rope_sin
            {{{8, 64}, {8, 64}}, ge::DT_BF16, ge::FORMAT_ND},                     // rope_cos
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND}, // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},   // kr_cache
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND},                            // cache_index
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // dequant_scale_x
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // dequant_scale_w_dq
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // dequant_scale_w_uq_qr
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // dequant_scale_w_dkv_kr
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // quant_scale_ckv
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // quant_scale_ckr
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // smooth_scales_cq
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                              // actual_seq_len
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // k_nope_clip_alpha
        },
        {
            {{{8, 32, 512}, {8, 32, 512}}, ge::DT_BF16, ge::FORMAT_ND},           // query
            {{{8, 32, 64}, {8, 32, 64}}, ge::DT_BF16, ge::FORMAT_ND},             // query_rope
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},   // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_BF16, ge::FORMAT_ND},                             // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                             // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3932178;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 950的MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR场景正常case，weight_dq weight_uq_qr NZ格式
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test96)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 1, 7168}, {8, 1, 7168}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                   // token_x 0
            {{{48, 448, 16, 32}, {48, 448, 16, 32}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ}, // weight_dq 1
            {{{768, 96, 16, 32}, {768, 96, 16, 32}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ}, // weight_uq_qr 2
            {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                      // weight_uk 3
            {{{18, 448, 16, 32}, {18, 448, 16, 32}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                                        // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                              // rmsnorm_gamma_ckv 6
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                    // rope_sin 7
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                    // rope_cos 8
            {{{1, 16, 1, 512}, {1, 16, 1, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // kv_cache 9
            {{{1, 16, 1, 64}, {1, 16, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},            // kr_cache 10
            {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},                           // cache_index 11
            {{{8, 224}, {8, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                 // dequant_scale_x 12
            {{{1536, 224}, {1536, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},           // dequant_scale_w_dq 13
            {{{24576, 48}, {24576, 48}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},           // dequant_scale_w_uq_qr 14
            {{{576, 224}, {576, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},             // dequant_scale_w_dkv_kr 15
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                   // actual_seq_len 19
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // k_nope_clip_alpha 20
        },
        {
            {{{8, 1, 128, 512}, {8, 1, 128, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // query
            {{{8, 1, 128, 64}, {8, 1, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},            // query_rope
            {{{1, 16, 1, 512}, {1, 16, 1, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},   // kv_cache
            {{{1, 16, 1, 64}, {1, 16, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},              // kr_cache
            {{{8, 128, 1}, {8, 128, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                           // query_norm
            {{{0}, {0}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                             // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0022971812167027)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00235037235057241)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_NZ")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3933730;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 950的MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR场景异常case，cache_index expected not null, got null.
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test97)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 1, 7168}, {8, 1, 7168}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                   // token_x 0
            {{{48, 448, 16, 32}, {48, 448, 16, 32}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ}, // weight_dq 1
            {{{768, 96, 16, 32}, {768, 96, 16, 32}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ}, // weight_uq_qr 2
            {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                      // weight_uk 3
            {{{18, 448, 16, 32}, {18, 448, 16, 32}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                                        // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                              // rmsnorm_gamma_ckv 6
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                    // rope_sin 7
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                    // rope_cos 8
            {{{1, 16, 1, 512}, {1, 16, 1, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // kv_cache 9
            {{{1, 16, 1, 64}, {1, 16, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},            // kr_cache 10
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                   // cache_index 11
            {{{8, 224}, {8, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                 // dequant_scale_x 12
            {{{1536, 224}, {1536, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},           // dequant_scale_w_dq 13
            {{{24576, 48}, {24576, 48}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},           // dequant_scale_w_uq_qr 14
            {{{576, 224}, {576, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},             // dequant_scale_w_dkv_kr 15
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                   // actual_seq_len 19
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // k_nope_clip_alpha 20
        },
        {
            {{{8, 1, 128, 512}, {8, 1, 128, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // query
            {{{8, 1, 128, 64}, {8, 1, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},            // query_rope
            {{{1, 16, 1, 512}, {1, 16, 1, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},   // kv_cache
            {{{1, 16, 1, 64}, {1, 16, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},              // kr_cache
            {{{8, 128, 1}, {8, 128, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                           // query_norm
            {{{0}, {0}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                             // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0022971812167027)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00235037235057241)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_NZ")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3933730;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 950的MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR场景异常case，kvCache dim num supports only [4], but got 3.
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test98)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 1, 7168}, {8, 1, 7168}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                   // token_x 0
            {{{48, 448, 16, 32}, {48, 448, 16, 32}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ}, // weight_dq 1
            {{{768, 96, 16, 32}, {768, 96, 16, 32}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ}, // weight_uq_qr 2
            {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                      // weight_uk 3
            {{{18, 448, 16, 32}, {18, 448, 16, 32}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                                        // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                        // rmsnorm_gamma_ckv 6
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},              // rope_sin 7
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},              // rope_cos 8
            {{{1, 16, 512}, {1, 16, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // kv_cache 9
            {{{1, 16, 1, 64}, {1, 16, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},      // kr_cache 10
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                             // cache_index 11
            {{{8, 224}, {8, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},           // dequant_scale_x 12
            {{{1536, 224}, {1536, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},     // dequant_scale_w_dq 13
            {{{24576, 48}, {24576, 48}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},     // dequant_scale_w_uq_qr 14
            {{{576, 224}, {576, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},       // dequant_scale_w_dkv_kr 15
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                           // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                             // actual_seq_len 19
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // k_nope_clip_alpha 20
        },
        {
            {{{8, 1, 128, 512}, {8, 1, 128, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // query
            {{{8, 1, 128, 64}, {8, 1, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},            // query_rope
            {{{1, 16, 1, 512}, {1, 16, 1, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},   // kv_cache
            {{{1, 16, 1, 64}, {1, 16, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},              // kr_cache
            {{{8, 128, 1}, {8, 128, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                           // query_norm
            {{{0}, {0}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                             // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0022971812167027)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00235037235057241)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_NZ")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3933730;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 950的MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR场景异常case，kv_cache_quant_mode传错
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test99)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{33, 7168}, {33, 7168}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                     // token_x 0
            {{{7168, 1536}, {7168, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 1536}, {1536, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},         // weight_uq_qr 2
            {{{8, 128, 512}, {8, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                        // weight_uk 3
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                                      // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                        // rmsnorm_gamma_ckv 6
            {{{33, 64}, {33, 64}}, ge::DT_BF16, ge::FORMAT_ND},                                  // rope_sin 7
            {{{33, 64}, {33, 64}}, ge::DT_BF16, ge::FORMAT_ND},                                  // rope_cos 8
            {{{254, 128, 1, 512}, {254, 128, 1, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},     // kv_cache 9
            {{{254, 128, 1, 64}, {254, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                // kr_cache 10
            {{{33}, {33}}, ge::DT_INT64, ge::FORMAT_ND},                                         // cache_index 11
            {{{33, 224}, {33, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                         // dequant_scale_x 12
            {{{1536, 224}, {1536, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // dequant_scale_w_dq 13
            {{{1536, 48}, {1536, 48}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},   // dequant_scale_w_uq_qr 14
            {{{576, 224}, {576, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},   // dequant_scale_w_dkv_kr 15
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // actual_seq_len 19
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // k_nope_clip_alpha 20
        },
        {
            {{{33, 8, 512}, {33, 8, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},             // query
            {{{33, 8, 64}, {33, 8, 64}}, ge::DT_BF16, ge::FORMAT_ND},                        // query_rope
            {{{254, 128, 1, 512}, {254, 128, 1, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // kv_cache
            {{{254, 128, 1, 64}, {254, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},            // kr_cache
            {{{33, 8, 1}, {33, 8, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                               // query_norm
            {{{0}, {0}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                                 // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0022971812167027)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00235037235057241)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3933730;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 950的MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR场景正常，cache_mode：TND；hiddensize：6144；
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test100)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 6144}, {8, 6144}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                       // token_x 0
            {{{6144, 1536}, {6144, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 24576}, {1536, 24576}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},       // weight_uq_qr 2
            {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                    // weight_uk 3
            {{{6144, 512 + 64}, {6144, 512 + 64}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                                      // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                        // rmsnorm_gamma_ckv 6
            {{{8, 64}, {8, 64}}, ge::DT_BF16, ge::FORMAT_ND},                                    // rope_sin 7
            {{{8, 64}, {8, 64}}, ge::DT_BF16, ge::FORMAT_ND},                                    // rope_cos 8
            {{{8, 1, 512}, {8, 1, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                   // kv_cache 9
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                              // kr_cache 10
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                             // cache_index 11
            {{{8, 192}, {8, 192}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                           // dequant_scale_x 12
            {{{1536, 192}, {1536, 192}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // dequant_scale_w_dq 13
            {{{24576, 48}, {24576, 48}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // dequant_scale_w_uq_qr 14
            {{{576, 192}, {576, 192}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},   // dequant_scale_w_dkv_kr 15
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // actual_seq_len 19
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // k_nope_clip_alpha 20
        },
        {
            {{{8, 128, 512}, {8, 128, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // query
            {{{8, 128, 64}, {8, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},            // query_rope
            {{{8, 1, 512}, {8, 1, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},     // kv_cache
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                // kr_cache
            {{{8, 128, 1}, {8, 128, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},             // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                     // query_norm
            {{{0}, {0}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                       // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0022971812167027)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00235037235057241)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3933728;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 950的MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR场景正常case，cache_mode：BSND；hiddensize：5120；
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test101)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 1, 5120}, {8, 1, 5120}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                 // token_x 0
            {{{5120, 1536}, {5120, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 24576}, {1536, 24576}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},       // weight_uq_qr 2
            {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                    // weight_uk 3
            {{{5120, 512 + 64}, {5120, 512 + 64}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                                      // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                        // rmsnorm_gamma_ckv 6
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                              // rope_sin 7
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                              // rope_cos 8
            {{{8, 1, 1, 512}, {8, 1, 1, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},             // kv_cache 9
            {{{8, 1, 1, 64}, {8, 1, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                        // kr_cache 10
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                             // cache_index 11
            {{{8, 160}, {8, 160}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                           // dequant_scale_x 12
            {{{1536, 160}, {1536, 160}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // dequant_scale_w_dq 13
            {{{24576, 48}, {24576, 48}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // dequant_scale_w_uq_qr 14
            {{{576, 160}, {576, 160}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},   // dequant_scale_w_dkv_kr 15
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // actual_seq_len 19
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // k_nope_clip_alpha 20
        },
        {
            {{{8, 1, 128, 512}, {8, 1, 128, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // query
            {{{8, 1, 128, 64}, {8, 1, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},            // query_rope
            {{{8, 1, 1, 512}, {8, 1, 1, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},     // kv_cache
            {{{8, 1, 1, 64}, {8, 1, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                // kr_cache
            {{{8, 128, 1}, {8, 128, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                           // query_norm
            {{{0}, {0}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                             // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0022971812167027)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00235037235057241)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3933728;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 950的MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR场景正常case，cache_mode：PA_BLK_BSND；hiddensize：7168；BlockSize:16
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test102)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 1, 7168}, {8, 1, 7168}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                 // token_x 0
            {{{7168, 1536}, {7168, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 24576}, {1536, 24576}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},       // weight_uq_qr 2
            {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                    // weight_uk 3
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                                      // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                        // rmsnorm_gamma_ckv 6
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                              // rope_sin 7
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                              // rope_cos 8
            {{{1, 16, 1, 512}, {1, 16, 1, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},           // kv_cache 9
            {{{1, 16, 1, 64}, {1, 16, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // kr_cache 10
            {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},                                     // cache_index 11
            {{{8, 224}, {8, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                           // dequant_scale_x 12
            {{{1536, 224}, {1536, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // dequant_scale_w_dq 13
            {{{24576, 48}, {24576, 48}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // dequant_scale_w_uq_qr 14
            {{{576, 224}, {576, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},   // dequant_scale_w_dkv_kr 15
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // actual_seq_len 19
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // k_nope_clip_alpha 20
        },
        {
            {{{8, 1, 128, 512}, {8, 1, 128, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // query
            {{{8, 1, 128, 64}, {8, 1, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},            // query_rope
            {{{1, 16, 1, 512}, {1, 16, 1, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},   // kv_cache
            {{{1, 16, 1, 64}, {1, 16, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},              // kr_cache
            {{{8, 128, 1}, {8, 128, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                           // query_norm
            {{{0}, {0}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                             // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0022971812167027)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00235037235057241)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BLK_BSND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3933731;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 950的MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR场景正常case，cache_mode：PA_BSND；hiddensize：2048；BlockSize:64
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test103)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 1, 2048}, {8, 1, 2048}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                 // token_x 0
            {{{2048, 1536}, {2048, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 24576}, {1536, 24576}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},       // weight_uq_qr 2
            {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                    // weight_uk 3
            {{{2048, 512 + 64}, {2048, 512 + 64}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                                      // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                        // rmsnorm_gamma_ckv 6
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                              // rope_sin 7
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                              // rope_cos 8
            {{{1, 64, 1, 512}, {1, 64, 1, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},           // kv_cache 9
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // kr_cache 10
            {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},                                     // cache_index 11
            {{{8, 644}, {8, 64}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                            // dequant_scale_x 12
            {{{1536, 64}, {1536, 64}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},   // dequant_scale_w_dq 13
            {{{24576, 48}, {24576, 48}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // dequant_scale_w_uq_qr 14
            {{{576, 64}, {576, 64}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},     // dequant_scale_w_dkv_kr 15
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // actual_seq_len 19
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // k_nope_clip_alpha 20
        },
        {
            {{{8, 1, 128, 512}, {8, 1, 128, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // query
            {{{8, 1, 128, 64}, {8, 1, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},            // query_rope
            {{{1, 64, 1, 512}, {1, 64, 1, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},   // kv_cache
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},              // kr_cache
            {{{8, 128, 1}, {8, 128, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                           // query_norm
            {{{0}, {0}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                             // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0022971812167027)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00235037235057241)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3933729;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 950的MXFP8_FULL_QUANT_KV_NOQUANT_PER_TENSOR场景正常case，cache_mode：PA_BSND ；hiddensize：7680；BlockSize:112
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test104)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 1, 7680}, {8, 1, 7680}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                 // token_x 0
            {{{7680, 1536}, {7680, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 24576}, {1536, 24576}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},       // weight_uq_qr 2
            {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                    // weight_uk 3
            {{{7680, 512 + 64}, {7680, 512 + 64}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                                      // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                        // rmsnorm_gamma_ckv 6
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                              // rope_sin 7
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                              // rope_cos 8
            {{{1, 112, 1, 512}, {1, 112, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},                  // kv_cache 9
            {{{1, 112, 1, 64}, {1, 112, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                    // kr_cache 10
            {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},                                     // cache_index 11
            {{{8, 240}, {8, 240}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                           // dequant_scale_x 12
            {{{1536, 240}, {1536, 240}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // dequant_scale_w_dq 13
            {{{24576, 48}, {24576, 48}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // dequant_scale_w_uq_qr 14
            {{{576, 240}, {576, 240}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},   // dequant_scale_w_dkv_kr 15
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // actual_seq_len 19
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // k_nope_clip_alpha 20
        },
        {
            {{{8, 1, 128, 512}, {8, 1, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // query
            {{{8, 1, 128, 64}, {8, 1, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},   // query_rope
            {{{1, 112, 1, 512}, {1, 112, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // kv_cache
            {{{1, 112, 1, 64}, {1, 112, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},   // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                          // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                  // query_norm
            {{{0}, {0}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                    // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0022971812167027)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00235037235057241)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3933665;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 950的MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR场景正常case，cache_mode：PA_BLK_NZ ；hiddensize：1024；BlockSize:32
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test105)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 1, 1024}, {8, 1, 1024}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                 // token_x 0
            {{{1024, 1536}, {1024, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 24576}, {1536, 24576}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},       // weight_uq_qr 2
            {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                    // weight_uk 3
            {{{1024, 512 + 64}, {1024, 512 + 64}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                                      // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                        // rmsnorm_gamma_ckv 6
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                              // rope_sin 7
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                              // rope_cos 8
            {{{1, 32, 1, 512}, {1, 32, 1, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},           // kv_cache 9
            {{{1, 32, 1, 64}, {1, 32, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // kr_cache 10
            {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},                                     // cache_index 11
            {{{8, 32}, {8, 32}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                             // dequant_scale_x 12
            {{{1536, 32}, {1536, 32}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},   // dequant_scale_w_dq 13
            {{{24576, 48}, {24576, 48}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // dequant_scale_w_uq_qr 14
            {{{576, 32}, {576, 32}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},     // dequant_scale_w_dkv_kr 15
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // actual_seq_len 19
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // k_nope_clip_alpha 20
        },
        {
            {{{8, 1, 128, 512}, {8, 1, 128, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // query
            {{{8, 1, 128, 64}, {8, 1, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},            // query_rope
            {{{1, 32, 1, 512}, {1, 32, 1, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},   // kv_cache
            {{{1, 32, 1, 64}, {1, 32, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},              // kr_cache
            {{{8, 128, 1}, {8, 128, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                           // query_norm
            {{{0}, {0}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                             // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0022971812167027)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00235037235057241)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BLK_NZ")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3933732;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 950的MXFP8_FULL_QUANT_KV_NOQUANT_PER_TENSOR场景正常case，cache_mode：PA_BLK_NZ ；hiddensize：6144；BlockSize:96
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test106)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 1, 6144}, {8, 1, 6144}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                 // token_x 0
            {{{6144, 1536}, {6144, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 24576}, {1536, 24576}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},       // weight_uq_qr 2
            {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                    // weight_uk 3
            {{{6144, 512 + 64}, {6144, 512 + 64}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                                      // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                        // rmsnorm_gamma_ckv 6
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                              // rope_sin 7
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                              // rope_cos 8
            {{{1, 96, 1, 512}, {1, 96, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},                    // kv_cache 9
            {{{1, 96, 1, 64}, {1, 96, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // kr_cache 10
            {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},                                     // cache_index 11
            {{{8, 192}, {8, 192}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                           // dequant_scale_x 12
            {{{1536, 192}, {1536, 192}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // dequant_scale_w_dq 13
            {{{24576, 48}, {24576, 48}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // dequant_scale_w_uq_qr 14
            {{{576, 192}, {576, 192}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},   // dequant_scale_w_dkv_kr 15
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // actual_seq_len 19
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // k_nope_clip_alpha 20
        },
        {
            {{{8, 1, 128, 512}, {8, 1, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // query
            {{{8, 1, 128, 64}, {8, 1, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},   // query_rope
            {{{1, 96, 1, 512}, {1, 96, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},   // kv_cache
            {{{1, 96, 1, 64}, {1, 96, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},     // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                          // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                  // query_norm
            {{{0}, {0}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                    // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0022971812167027)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00235037235057241)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BLK_NZ")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3933668;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 950的MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR场景正常case，cache_mode：PA_NZ；hiddensize：3072；BlockSize:48；
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test107)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 1, 3072}, {8, 1, 3072}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                 // token_x 0
            {{{3072, 1536}, {3072, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 24576}, {1536, 24576}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},       // weight_uq_qr 2
            {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                    // weight_uk 3
            {{{3072, 512 + 64}, {3072, 512 + 64}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                                      // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                        // rmsnorm_gamma_ckv 6
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                              // rope_sin 7
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                              // rope_cos 8
            {{{1, 48, 1, 512}, {1, 48, 1, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},           // kv_cache 9
            {{{1, 48, 1, 64}, {1, 48, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // kr_cache 10
            {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},                                     // cache_index 11
            {{{8, 96}, {8, 96}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                             // dequant_scale_x 12
            {{{1536, 96}, {1536, 96}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},   // dequant_scale_w_dq 13
            {{{24576, 48}, {24576, 48}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // dequant_scale_w_uq_qr 14
            {{{576, 96}, {576, 96}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},     // dequant_scale_w_dkv_kr 15
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // actual_seq_len 19
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // k_nope_clip_alpha 20
        },
        {
            {{{8, 1, 128, 512}, {8, 1, 128, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // query
            {{{8, 1, 128, 64}, {8, 1, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},            // query_rope
            {{{1, 48, 1, 512}, {1, 48, 1, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},   // kv_cache
            {{{1, 48, 1, 64}, {1, 48, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},              // kr_cache
            {{{8, 128, 1}, {8, 128, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                           // query_norm
            {{{0}, {0}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                             // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0022971812167027)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00235037235057241)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_NZ")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3933730;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}
// 950的MXFP8_FULL_QUANT_KV_QUANT_PER_TILE场景正常case，cache_mode：PA_BSND；hiddensize：4096；BlockSize:80
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test108)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 1, 7168}, {8, 1, 7168}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                 // token_x 0
            {{{7168, 1536}, {7168, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 24576}, {1536, 24576}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},       // weight_uq_qr 2
            {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                    // weight_uk 3
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                                      // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                        // rmsnorm_gamma_ckv 6
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                              // rope_sin 7
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                              // rope_cos 8
            {{{1, 80, 1, 656}, {1, 80, 1, 656}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},           // kv_cache 9
            {{{0, 80, 1}, {0, 80, 1}}, ge::DT_BF16, ge::FORMAT_ND},                              // kr_cache 10
            {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},                                     // cache_index 11
            {{{8, 224}, {8, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                           // dequant_scale_x 12
            {{{1536, 224}, {1536, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // dequant_scale_w_dq 13
            {{{24576, 48}, {24576, 48}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // dequant_scale_w_uq_qr 14
            {{{576, 224}, {576, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},   // dequant_scale_w_dkv_kr 15
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // actual_seq_len 19
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // k_nope_clip_alpha 20
        },
        {
            {{{8, 1, 128, 512}, {8, 1, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},        // query
            {{{8, 1, 128, 64}, {8, 1, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},          // query_rope
            {{{1, 80, 1, 656}, {1, 80, 1, 656}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // kv_cache
            {{{0, 80, 1}, {0, 80, 1}}, ge::DT_BF16, ge::FORMAT_ND},                    // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                         // query_norm
            {{{0}, {0}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                           // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0022971812167027)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00235037235057241)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3933793;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 950的MXFP8_FULL_QUANT_KV_QUANT_PER_TILE场景异常case，cache_mode：PA_NZ；hiddensize：8192；BlockSize:144，Not support
// both cacheMode {PA_NZ, PA_BLK_BSND, PA_BLK_NZ} and pertile effective.
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test109)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 1, 8192}, {8, 1, 8192}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                 // token_x 0
            {{{8192, 1536}, {8192, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 24576}, {1536, 24576}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},       // weight_uq_qr 2
            {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                    // weight_uk 3
            {{{8192, 512 + 64}, {8192, 512 + 64}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                                      // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                        // rmsnorm_gamma_ckv 6
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                              // rope_sin 7
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                              // rope_cos 8
            {{{1, 144, 1, 64}, {1, 144, 1, 64}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},           // kv_cache 9
            {{{0}, {0}}, ge::DT_BF16, ge::FORMAT_ND},                                            // kr_cache 10
            {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},                                     // cache_index 11
            {{{8, 256}, {8, 256}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                           // dequant_scale_x 12
            {{{1536, 256}, {1536, 256}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // dequant_scale_w_dq 13
            {{{24576, 48}, {24576, 48}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // dequant_scale_w_uq_qr 14
            {{{576, 256}, {576, 256}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},   // dequant_scale_w_dkv_kr 15
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // actual_seq_len 19
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // k_nope_clip_alpha 20
        },
        {
            {{{8, 1, 128, 512}, {8, 1, 128, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // query
            {{{8, 1, 128, 64}, {8, 1, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},            // query_rope
            {{{1, 144, 1, 64}, {1, 144, 1, 64}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},   // kv_cache
            {{{0}, {0}}, ge::DT_BF16, ge::FORMAT_ND},                                    // kr_cache
            {{{8, 128, 1}, {8, 128, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                           // query_norm
            {{{0}, {0}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                             // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0022971812167027)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00235037235057241)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_NZ")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3933730;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 950的MXFP8_FULL_QUANT_KV_QUANT_PER_TILE场景异常case，cache_mode：PA_BLK_BSND；hiddensize：8192；BlockSize:144，Not
// support both cacheMode {PA_NZ, PA_BLK_BSND, PA_BLK_NZ} and pertile effective.
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test110)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 1, 8192}, {8, 1, 8192}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                 // token_x 0
            {{{8192, 1536}, {8192, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 24576}, {1536, 24576}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},       // weight_uq_qr 2
            {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                    // weight_uk 3
            {{{8192, 512 + 64}, {8192, 512 + 64}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                                      // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                        // rmsnorm_gamma_ckv 6
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                              // rope_sin 7
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                              // rope_cos 8
            {{{1, 144, 1, 64}, {1, 144, 1, 64}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},           // kv_cache 9
            {{{0}, {0}}, ge::DT_BF16, ge::FORMAT_ND},                                            // kr_cache 10
            {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},                                     // cache_index 11
            {{{8, 256}, {8, 256}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                           // dequant_scale_x 12
            {{{1536, 256}, {1536, 256}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // dequant_scale_w_dq 13
            {{{24576, 48}, {24576, 48}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // dequant_scale_w_uq_qr 14
            {{{576, 256}, {576, 256}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},   // dequant_scale_w_dkv_kr 15
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // actual_seq_len 19
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // k_nope_clip_alpha 20
        },
        {
            {{{8, 1, 128, 512}, {8, 1, 128, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // query
            {{{8, 1, 128, 64}, {8, 1, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},            // query_rope
            {{{1, 144, 1, 64}, {1, 144, 1, 64}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},   // kv_cache
            {{{0}, {0}}, ge::DT_BF16, ge::FORMAT_ND},                                    // kr_cache
            {{{8, 128, 1}, {8, 128, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                           // query_norm
            {{{0}, {0}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                             // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0022971812167027)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00235037235057241)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_NZ")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3933730;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 950的MXFP8_FULL_QUANT_KV_QUANT_PER_TILE场景异常case，cache_mode：PA_BLK_NZ；hiddensize：8192；BlockSize:144，Not
// support both cacheMode {PA_NZ, PA_BLK_BSND, PA_BLK_NZ} and pertile effective.
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test111)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 1, 8192}, {8, 1, 8192}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                 // token_x 0
            {{{8192, 1536}, {8192, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 24576}, {1536, 24576}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},       // weight_uq_qr 2
            {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                    // weight_uk 3
            {{{8192, 512 + 64}, {8192, 512 + 64}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                                      // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                        // rmsnorm_gamma_ckv 6
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                              // rope_sin 7
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                              // rope_cos 8
            {{{1, 144, 1, 64}, {1, 144, 1, 64}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},           // kv_cache 9
            {{{0}, {0}}, ge::DT_BF16, ge::FORMAT_ND},                                            // kr_cache 10
            {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},                                     // cache_index 11
            {{{8, 256}, {8, 256}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                           // dequant_scale_x 12
            {{{1536, 256}, {1536, 256}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // dequant_scale_w_dq 13
            {{{24576, 48}, {24576, 48}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // dequant_scale_w_uq_qr 14
            {{{576, 256}, {576, 256}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},   // dequant_scale_w_dkv_kr 15
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // actual_seq_len 19
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // k_nope_clip_alpha 20
        },
        {
            {{{8, 1, 128, 512}, {8, 1, 128, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // query
            {{{8, 1, 128, 64}, {8, 1, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},            // query_rope
            {{{1, 144, 1, 64}, {1, 144, 1, 64}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},   // kv_cache
            {{{0}, {0}}, ge::DT_BF16, ge::FORMAT_ND},                                    // kr_cache
            {{{8, 128, 1}, {8, 128, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                           // query_norm
            {{{0}, {0}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                             // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0022971812167027)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00235037235057241)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BLK_NZ")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3933730;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 950的MXFP8_FULL_QUANT_KV_QUANT_PER_TILE场景异常case，cache_mode：PA_BSND；hiddensize：4096；BlockSize:72，BlockSize
// must be within [16，1024] and a multiple of 16, got 72.
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test112)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 1, 4096}, {8, 1, 4096}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                 // token_x 0
            {{{4096, 1536}, {4096, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 24576}, {1536, 24576}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},       // weight_uq_qr 2
            {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                    // weight_uk 3
            {{{4096, 512 + 64}, {4096, 512 + 64}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                                      // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                        // rmsnorm_gamma_ckv 6
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                              // rope_sin 7
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                              // rope_cos 8
            {{{1, 72, 1, 64}, {1, 72, 1, 64}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},             // kv_cache 9
            {{{0}, {0}}, ge::DT_BF16, ge::FORMAT_ND},                                            // kr_cache 10
            {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},                                     // cache_index 11
            {{{8, 128}, {8, 128}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                           // dequant_scale_x 12
            {{{1536, 128}, {1536, 128}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // dequant_scale_w_dq 13
            {{{24576, 48}, {24576, 48}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // dequant_scale_w_uq_qr 14
            {{{576, 128}, {576, 128}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},   // dequant_scale_w_dkv_kr 15
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // actual_seq_len 19
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // k_nope_clip_alpha 20
        },
        {
            {{{8, 1, 128, 512}, {8, 1, 128, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // query
            {{{8, 1, 128, 64}, {8, 1, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},            // query_rope
            {{{1, 72, 1, 64}, {1, 72, 1, 64}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},     // kv_cache
            {{{0}, {0}}, ge::DT_BF16, ge::FORMAT_ND},                                    // kr_cache
            {{{8, 128, 1}, {8, 128, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                           // query_norm
            {{{0}, {0}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                             // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0022971812167027)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00235037235057241)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3933730;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 950的MXFP8_FULL_QUANT_KV_QUANT_PER_TILE场景异常case，cache_mode：PA_BSND；hiddensize：4096；BlockSize:1534，BlockSize
// must be within [16，1024] and a multiple of 16, got 1534.
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test113)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 1, 4096}, {8, 1, 4096}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                 // token_x 0
            {{{4096, 1536}, {4096, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 24576}, {1536, 24576}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},       // weight_uq_qr 2
            {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                    // weight_uk 3
            {{{4096, 512 + 64}, {4096, 512 + 64}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                                      // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                        // rmsnorm_gamma_ckv 6
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                              // rope_sin 7
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                              // rope_cos 8
            {{{1, 1534, 1, 64}, {1, 1534, 1, 64}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},         // kv_cache 9
            {{{0}, {0}}, ge::DT_BF16, ge::FORMAT_ND},                                            // kr_cache 10
            {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},                                     // cache_index 11
            {{{8, 128}, {8, 128}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                           // dequant_scale_x 12
            {{{1536, 128}, {1536, 128}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // dequant_scale_w_dq 13
            {{{24576, 48}, {24576, 48}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // dequant_scale_w_uq_qr 14
            {{{576, 128}, {576, 128}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},   // dequant_scale_w_dkv_kr 15
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // actual_seq_len 19
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // k_nope_clip_alpha 20
        },
        {
            {{{8, 1, 128, 512}, {8, 1, 128, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // query
            {{{8, 1, 128, 64}, {8, 1, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},            // query_rope
            {{{1, 1534, 1, 64}, {1, 1534, 1, 64}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // kv_cache
            {{{0}, {0}}, ge::DT_BF16, ge::FORMAT_ND},                                    // kr_cache
            {{{8, 128, 1}, {8, 128, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                           // query_norm
            {{{0}, {0}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                             // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0022971812167027)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00235037235057241)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3933730;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 950的MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR场景正常case，cache_mode：PA_BLK_BSND；hiddensize：7168；BlockSize:16
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test114)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 1, 7168}, {8, 1, 7168}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                 // token_x 0
            {{{7168, 1536}, {7168, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 24576}, {1536, 24576}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},       // weight_uq_qr 2
            {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                    // weight_uk 3
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                                      // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                        // rmsnorm_gamma_ckv 6
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                              // rope_sin 7
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                              // rope_cos 8
            {{{1, 16, 1, 512}, {1, 16, 1, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},           // kv_cache 9
            {{{1, 16, 1, 64}, {1, 16, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // kr_cache 10
            {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},                                     // cache_index 11
            {{{8, 224}, {8, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                           // dequant_scale_x 12
            {{{1536, 224}, {1536, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // dequant_scale_w_dq 13
            {{{24576, 48}, {24576, 48}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // dequant_scale_w_uq_qr 14
            {{{576, 224}, {576, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},   // dequant_scale_w_dkv_kr 15
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // actual_seq_len 19
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // k_nope_clip_alpha 20
        },
        {
            {{{8, 1, 128, 512}, {8, 1, 128, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // query
            {{{8, 1, 128, 64}, {8, 1, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},            // query_rope
            {{{1, 16, 1, 512}, {1, 16, 1, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},   // kv_cache
            {{{1, 16, 1, 64}, {1, 16, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},              // kr_cache
            {{{8, 128, 1}, {8, 128, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                           // query_norm
            {{{0}, {0}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                             // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0022971812167027)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00235037235057241)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BLK_BSND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3933731;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 950的半量化kv量化场景正常case
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test115)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 1, 7168}, {8, 1, 7168}}, ge::DT_BF16, ge::FORMAT_ND},                               // token_x
            {{{7168, 1536}, {7168, 1536}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},                       // weight_dq
            {{{1536, 32 * (128 + 64)}, {1536, 32 * (128 + 64)}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // weight_uq_qr
            {{{32, 128, 512}, {32, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                           // weight_uk
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},               // weight_dkv_kr
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                              // rmsnorm_gamma_cq
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                // rmsnorm_gamma_ckv
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // rope_sin
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // rope_cos
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND},        // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_INT8, ge::FORMAT_ND},          // kr_cache
            {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},                             // cache_index
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // dequant_scale_x
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // dequant_scale_w_dq
            {{{1, 32 * (128 + 64)}, {1, 32 * (128 + 64)}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dequant_scale_w_uq_qr
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                      // dequant_scale_w_dkv_kr
            {{{1, 512}, {1, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckv
            {{{1, 64}, {1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},                           // quant_scale_ckr
            {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // smooth_scales_cq
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                     // actual_seq_len
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // k_nope_clip_alpha
        },
        {
            {{{8, 1, 32, 512}, {8, 1, 32, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // query
            {{{8, 1, 32, 64}, {8, 1, 32, 64}}, ge::DT_BF16, ge::FORMAT_ND},       // query_rope
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND}, // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_INT8, ge::FORMAT_ND},   // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                             // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                             // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3933345;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 950的半量化kv量化场景异常case：WeightQuantMode must be within {0, 1, 2}, actually is 3.
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test116)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 1, 7168}, {8, 1, 7168}}, ge::DT_BF16, ge::FORMAT_ND},                               // token_x
            {{{7168, 1536}, {7168, 1536}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},                       // weight_dq
            {{{1536, 32 * (128 + 64)}, {1536, 32 * (128 + 64)}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // weight_uq_qr
            {{{32, 128, 512}, {32, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                           // weight_uk
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},               // weight_dkv_kr
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                              // rmsnorm_gamma_cq
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                // rmsnorm_gamma_ckv
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // rope_sin
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // rope_cos
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND},        // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_INT8, ge::FORMAT_ND},          // kr_cache
            {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},                             // cache_index
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // dequant_scale_x
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // dequant_scale_w_dq
            {{{1, 32 * (128 + 64)}, {1, 32 * (128 + 64)}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dequant_scale_w_uq_qr
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                      // dequant_scale_w_dkv_kr
            {{{1, 512}, {1, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckv
            {{{1, 64}, {1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},                           // quant_scale_ckr
            {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // smooth_scales_cq
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                     // actual_seq_len
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // k_nope_clip_alpha
        },
        {
            {{{8, 1, 32, 512}, {8, 1, 32, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // query
            {{{8, 1, 32, 64}, {8, 1, 32, 64}}, ge::DT_BF16, ge::FORMAT_ND},       // query_rope
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND}, // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_INT8, ge::FORMAT_ND},   // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                             // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                             // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3933345;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 950的半量化kv量化场景异常case：When weightQuantMode == 1, kvQuantMode must be within {0, 2, 3}, actually is 1.
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test117)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 1, 7168}, {8, 1, 7168}}, ge::DT_BF16, ge::FORMAT_ND},                               // token_x
            {{{7168, 1536}, {7168, 1536}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},                       // weight_dq
            {{{1536, 32 * (128 + 64)}, {1536, 32 * (128 + 64)}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // weight_uq_qr
            {{{32, 128, 512}, {32, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                           // weight_uk
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},               // weight_dkv_kr
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                              // rmsnorm_gamma_cq
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                // rmsnorm_gamma_ckv
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // rope_sin
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // rope_cos
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND},        // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_INT8, ge::FORMAT_ND},          // kr_cache
            {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},                             // cache_index
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // dequant_scale_x
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // dequant_scale_w_dq
            {{{1, 32 * (128 + 64)}, {1, 32 * (128 + 64)}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dequant_scale_w_uq_qr
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                      // dequant_scale_w_dkv_kr
            {{{1, 512}, {1, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckv
            {{{1, 64}, {1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},                           // quant_scale_ckr
            {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // smooth_scales_cq
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                     // actual_seq_len
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // k_nope_clip_alpha
        },
        {
            {{{8, 1, 32, 512}, {8, 1, 32, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // query
            {{{8, 1, 32, 64}, {8, 1, 32, 64}}, ge::DT_BF16, ge::FORMAT_ND},       // query_rope
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND}, // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_INT8, ge::FORMAT_ND},   // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                             // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                             // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3933345;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 950的半量化kv量化场景异常case：Only support cacheMode {BSND, TND, PA_BSND, PA_NZ, PA_BLK_BSND, PA_BLK_NZ}, actually
// is ABCD.
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test118)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 1, 7168}, {8, 1, 7168}}, ge::DT_BF16, ge::FORMAT_ND},                               // token_x
            {{{7168, 1536}, {7168, 1536}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},                       // weight_dq
            {{{1536, 32 * (128 + 64)}, {1536, 32 * (128 + 64)}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // weight_uq_qr
            {{{32, 128, 512}, {32, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                           // weight_uk
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},               // weight_dkv_kr
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                              // rmsnorm_gamma_cq
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                // rmsnorm_gamma_ckv
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // rope_sin
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // rope_cos
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND},        // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_INT8, ge::FORMAT_ND},          // kr_cache
            {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},                             // cache_index
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // dequant_scale_x
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // dequant_scale_w_dq
            {{{1, 32 * (128 + 64)}, {1, 32 * (128 + 64)}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dequant_scale_w_uq_qr
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                      // dequant_scale_w_dkv_kr
            {{{1, 512}, {1, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckv
            {{{1, 64}, {1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},                           // quant_scale_ckr
            {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // smooth_scales_cq
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                     // actual_seq_len
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // k_nope_clip_alpha
        },
        {
            {{{8, 1, 32, 512}, {8, 1, 32, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // query
            {{{8, 1, 32, 64}, {8, 1, 32, 64}}, ge::DT_BF16, ge::FORMAT_ND},       // query_rope
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND}, // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_INT8, ge::FORMAT_ND},   // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                             // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                             // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ABCD")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3933345;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 950的半量化kv量化场景异常case：The ckvkrRepoMode expected 0, but got 1.
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test119)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 1, 7168}, {8, 1, 7168}}, ge::DT_BF16, ge::FORMAT_ND},                               // token_x
            {{{7168, 1536}, {7168, 1536}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},                       // weight_dq
            {{{1536, 32 * (128 + 64)}, {1536, 32 * (128 + 64)}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // weight_uq_qr
            {{{32, 128, 512}, {32, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                           // weight_uk
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},               // weight_dkv_kr
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                              // rmsnorm_gamma_cq
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                // rmsnorm_gamma_ckv
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // rope_sin
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // rope_cos
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND},        // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_INT8, ge::FORMAT_ND},          // kr_cache
            {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},                             // cache_index
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // dequant_scale_x
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // dequant_scale_w_dq
            {{{1, 32 * (128 + 64)}, {1, 32 * (128 + 64)}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dequant_scale_w_uq_qr
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                      // dequant_scale_w_dkv_kr
            {{{1, 512}, {1, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckv
            {{{1, 64}, {1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},                           // quant_scale_ckr
            {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // smooth_scales_cq
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                     // actual_seq_len
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // k_nope_clip_alpha
        },
        {
            {{{8, 1, 32, 512}, {8, 1, 32, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // query
            {{{8, 1, 32, 64}, {8, 1, 32, 64}}, ge::DT_BF16, ge::FORMAT_ND},       // query_rope
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND}, // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_INT8, ge::FORMAT_ND},   // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                             // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                             // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3933345;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 950的半量化kv量化场景异常case：The quantScaleRepoMode expected 0, but got 1.
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test120)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 1, 7168}, {8, 1, 7168}}, ge::DT_BF16, ge::FORMAT_ND},                               // token_x
            {{{7168, 1536}, {7168, 1536}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},                       // weight_dq
            {{{1536, 32 * (128 + 64)}, {1536, 32 * (128 + 64)}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // weight_uq_qr
            {{{32, 128, 512}, {32, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                           // weight_uk
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},               // weight_dkv_kr
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                              // rmsnorm_gamma_cq
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                // rmsnorm_gamma_ckv
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // rope_sin
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // rope_cos
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND},        // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_INT8, ge::FORMAT_ND},          // kr_cache
            {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},                             // cache_index
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // dequant_scale_x
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // dequant_scale_w_dq
            {{{1, 32 * (128 + 64)}, {1, 32 * (128 + 64)}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dequant_scale_w_uq_qr
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                      // dequant_scale_w_dkv_kr
            {{{1, 512}, {1, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckv
            {{{1, 64}, {1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},                           // quant_scale_ckr
            {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // smooth_scales_cq
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                     // actual_seq_len
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // k_nope_clip_alpha
        },
        {
            {{{8, 1, 32, 512}, {8, 1, 32, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // query
            {{{8, 1, 32, 64}, {8, 1, 32, 64}}, ge::DT_BF16, ge::FORMAT_ND},       // query_rope
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND}, // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_INT8, ge::FORMAT_ND},   // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                             // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                             // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3933345;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 950的半量化kv量化场景异常case：The queryQuantMode expected 0, but got 1.
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test121)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 1, 7168}, {8, 1, 7168}}, ge::DT_BF16, ge::FORMAT_ND},                               // token_x
            {{{7168, 1536}, {7168, 1536}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},                       // weight_dq
            {{{1536, 32 * (128 + 64)}, {1536, 32 * (128 + 64)}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // weight_uq_qr
            {{{32, 128, 512}, {32, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                           // weight_uk
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},               // weight_dkv_kr
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                              // rmsnorm_gamma_cq
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                // rmsnorm_gamma_ckv
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // rope_sin
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // rope_cos
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND},        // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_INT8, ge::FORMAT_ND},          // kr_cache
            {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},                             // cache_index
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // dequant_scale_x
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // dequant_scale_w_dq
            {{{1, 32 * (128 + 64)}, {1, 32 * (128 + 64)}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dequant_scale_w_uq_qr
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                      // dequant_scale_w_dkv_kr
            {{{1, 512}, {1, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckv
            {{{1, 64}, {1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},                           // quant_scale_ckr
            {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // smooth_scales_cq
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                     // actual_seq_len
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // k_nope_clip_alpha
        },
        {
            {{{8, 1, 32, 512}, {8, 1, 32, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // query
            {{{8, 1, 32, 64}, {8, 1, 32, 64}}, ge::DT_BF16, ge::FORMAT_ND},       // query_rope
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND}, // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_INT8, ge::FORMAT_ND},   // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                             // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                             // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3933345;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 950的半量化kv量化场景异常case：Get rmsnormEpsilonCq is nullptr
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test122)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 1, 7168}, {8, 1, 7168}}, ge::DT_BF16, ge::FORMAT_ND},                               // token_x
            {{{7168, 1536}, {7168, 1536}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},                       // weight_dq
            {{{1536, 32 * (128 + 64)}, {1536, 32 * (128 + 64)}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // weight_uq_qr
            {{{32, 128, 512}, {32, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                           // weight_uk
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},               // weight_dkv_kr
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                              // rmsnorm_gamma_cq
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                // rmsnorm_gamma_ckv
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // rope_sin
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // rope_cos
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND},        // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_INT8, ge::FORMAT_ND},          // kr_cache
            {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},                             // cache_index
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // dequant_scale_x
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // dequant_scale_w_dq
            {{{1, 32 * (128 + 64)}, {1, 32 * (128 + 64)}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dequant_scale_w_uq_qr
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                      // dequant_scale_w_dkv_kr
            {{{1, 512}, {1, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckv
            {{{1, 64}, {1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},                           // quant_scale_ckr
            {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // smooth_scales_cq
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                     // actual_seq_len
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // k_nope_clip_alpha
        },
        {
            {{{8, 1, 32, 512}, {8, 1, 32, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // query
            {{{8, 1, 32, 64}, {8, 1, 32, 64}}, ge::DT_BF16, ge::FORMAT_ND},       // query_rope
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND}, // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_INT8, ge::FORMAT_ND},   // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                             // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                             // dequant_scale_q_norm
        },
        {}, &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3933345;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 950的半量化kv量化场景异常case：Get rmsnormEpsilonCkv is nullptr
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test123)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 1, 7168}, {8, 1, 7168}}, ge::DT_BF16, ge::FORMAT_ND},                               // token_x
            {{{7168, 1536}, {7168, 1536}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},                       // weight_dq
            {{{1536, 32 * (128 + 64)}, {1536, 32 * (128 + 64)}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // weight_uq_qr
            {{{32, 128, 512}, {32, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                           // weight_uk
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},               // weight_dkv_kr
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                              // rmsnorm_gamma_cq
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                // rmsnorm_gamma_ckv
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // rope_sin
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // rope_cos
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND},        // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_INT8, ge::FORMAT_ND},          // kr_cache
            {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},                             // cache_index
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // dequant_scale_x
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // dequant_scale_w_dq
            {{{1, 32 * (128 + 64)}, {1, 32 * (128 + 64)}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dequant_scale_w_uq_qr
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                      // dequant_scale_w_dkv_kr
            {{{1, 512}, {1, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckv
            {{{1, 64}, {1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},                           // quant_scale_ckr
            {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // smooth_scales_cq
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                     // actual_seq_len
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // k_nope_clip_alpha
        },
        {
            {{{8, 1, 32, 512}, {8, 1, 32, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // query
            {{{8, 1, 32, 64}, {8, 1, 32, 64}}, ge::DT_BF16, ge::FORMAT_ND},       // query_rope
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND}, // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_INT8, ge::FORMAT_ND},   // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                             // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                             // dequant_scale_q_norm
        },
        {{"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)}}, &compileInfo, "Ascend950",
        MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3933345;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 950的半量化kv量化场景异常case：Get queryNormFlag is nullptr.
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test124)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 1, 7168}, {8, 1, 7168}}, ge::DT_BF16, ge::FORMAT_ND},                               // token_x
            {{{7168, 1536}, {7168, 1536}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},                       // weight_dq
            {{{1536, 32 * (128 + 64)}, {1536, 32 * (128 + 64)}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // weight_uq_qr
            {{{32, 128, 512}, {32, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                           // weight_uk
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},               // weight_dkv_kr
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                              // rmsnorm_gamma_cq
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                // rmsnorm_gamma_ckv
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // rope_sin
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // rope_cos
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND},        // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_INT8, ge::FORMAT_ND},          // kr_cache
            {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},                             // cache_index
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // dequant_scale_x
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // dequant_scale_w_dq
            {{{1, 32 * (128 + 64)}, {1, 32 * (128 + 64)}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dequant_scale_w_uq_qr
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                      // dequant_scale_w_dkv_kr
            {{{1, 512}, {1, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckv
            {{{1, 64}, {1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},                           // quant_scale_ckr
            {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // smooth_scales_cq
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                     // actual_seq_len
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // k_nope_clip_alpha
        },
        {
            {{{8, 1, 32, 512}, {8, 1, 32, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // query
            {{{8, 1, 32, 64}, {8, 1, 32, 64}}, ge::DT_BF16, ge::FORMAT_ND},       // query_rope
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND}, // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_INT8, ge::FORMAT_ND},   // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                             // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                             // dequant_scale_q_norm
        },
        {{"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
         {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
         {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")}},
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3933345;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 950的半量化kv量化场景异常case：Get kvQuantMode is nullptr.
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test125)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 1, 7168}, {8, 1, 7168}}, ge::DT_BF16, ge::FORMAT_ND},                               // token_x
            {{{7168, 1536}, {7168, 1536}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},                       // weight_dq
            {{{1536, 32 * (128 + 64)}, {1536, 32 * (128 + 64)}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // weight_uq_qr
            {{{32, 128, 512}, {32, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                           // weight_uk
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},               // weight_dkv_kr
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                              // rmsnorm_gamma_cq
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                // rmsnorm_gamma_ckv
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // rope_sin
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // rope_cos
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND},        // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_INT8, ge::FORMAT_ND},          // kr_cache
            {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},                             // cache_index
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // dequant_scale_x
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // dequant_scale_w_dq
            {{{1, 32 * (128 + 64)}, {1, 32 * (128 + 64)}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dequant_scale_w_uq_qr
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                      // dequant_scale_w_dkv_kr
            {{{1, 512}, {1, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckv
            {{{1, 64}, {1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},                           // quant_scale_ckr
            {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // smooth_scales_cq
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                     // actual_seq_len
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // k_nope_clip_alpha
        },
        {
            {{{8, 1, 32, 512}, {8, 1, 32, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // query
            {{{8, 1, 32, 64}, {8, 1, 32, 64}}, ge::DT_BF16, ge::FORMAT_ND},       // query_rope
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND}, // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_INT8, ge::FORMAT_ND},   // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                             // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                             // dequant_scale_q_norm
        },
        {{"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
         {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
         {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
         {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3933345;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 950的半量化kv量化场景异常case：Get queryQuantMode is nullptr.
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test126)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 1, 7168}, {8, 1, 7168}}, ge::DT_BF16, ge::FORMAT_ND},                               // token_x
            {{{7168, 1536}, {7168, 1536}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},                       // weight_dq
            {{{1536, 32 * (128 + 64)}, {1536, 32 * (128 + 64)}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // weight_uq_qr
            {{{32, 128, 512}, {32, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                           // weight_uk
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},               // weight_dkv_kr
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                              // rmsnorm_gamma_cq
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                // rmsnorm_gamma_ckv
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // rope_sin
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // rope_cos
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND},        // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_INT8, ge::FORMAT_ND},          // kr_cache
            {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},                             // cache_index
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // dequant_scale_x
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // dequant_scale_w_dq
            {{{1, 32 * (128 + 64)}, {1, 32 * (128 + 64)}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dequant_scale_w_uq_qr
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                      // dequant_scale_w_dkv_kr
            {{{1, 512}, {1, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckv
            {{{1, 64}, {1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},                           // quant_scale_ckr
            {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // smooth_scales_cq
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                     // actual_seq_len
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // k_nope_clip_alpha
        },
        {
            {{{8, 1, 32, 512}, {8, 1, 32, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // query
            {{{8, 1, 32, 64}, {8, 1, 32, 64}}, ge::DT_BF16, ge::FORMAT_ND},       // query_rope
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND}, // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_INT8, ge::FORMAT_ND},   // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                             // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                             // dequant_scale_q_norm
        },
        {{"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
         {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
         {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
         {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3933345;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 950的半量化kv量化场景异常case：Get ckvkrRepoMode is nullptr.
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test127)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 1, 7168}, {8, 1, 7168}}, ge::DT_BF16, ge::FORMAT_ND},                               // token_x
            {{{7168, 1536}, {7168, 1536}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},                       // weight_dq
            {{{1536, 32 * (128 + 64)}, {1536, 32 * (128 + 64)}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // weight_uq_qr
            {{{32, 128, 512}, {32, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                           // weight_uk
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},               // weight_dkv_kr
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                              // rmsnorm_gamma_cq
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                // rmsnorm_gamma_ckv
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // rope_sin
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // rope_cos
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND},        // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_INT8, ge::FORMAT_ND},          // kr_cache
            {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},                             // cache_index
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // dequant_scale_x
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // dequant_scale_w_dq
            {{{1, 32 * (128 + 64)}, {1, 32 * (128 + 64)}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dequant_scale_w_uq_qr
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                      // dequant_scale_w_dkv_kr
            {{{1, 512}, {1, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckv
            {{{1, 64}, {1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},                           // quant_scale_ckr
            {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // smooth_scales_cq
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                     // actual_seq_len
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // k_nope_clip_alpha
        },
        {
            {{{8, 1, 32, 512}, {8, 1, 32, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // query
            {{{8, 1, 32, 64}, {8, 1, 32, 64}}, ge::DT_BF16, ge::FORMAT_ND},       // query_rope
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND}, // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_INT8, ge::FORMAT_ND},   // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                             // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                             // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3933345;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 950的半量化kv量化场景异常case：Get quantScaleRepoMode is nullptr.
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test128)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 1, 7168}, {8, 1, 7168}}, ge::DT_BF16, ge::FORMAT_ND},                               // token_x
            {{{7168, 1536}, {7168, 1536}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},                       // weight_dq
            {{{1536, 32 * (128 + 64)}, {1536, 32 * (128 + 64)}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // weight_uq_qr
            {{{32, 128, 512}, {32, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                           // weight_uk
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},               // weight_dkv_kr
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                              // rmsnorm_gamma_cq
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                // rmsnorm_gamma_ckv
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // rope_sin
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // rope_cos
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND},        // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_INT8, ge::FORMAT_ND},          // kr_cache
            {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},                             // cache_index
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // dequant_scale_x
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // dequant_scale_w_dq
            {{{1, 32 * (128 + 64)}, {1, 32 * (128 + 64)}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dequant_scale_w_uq_qr
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                      // dequant_scale_w_dkv_kr
            {{{1, 512}, {1, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckv
            {{{1, 64}, {1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},                           // quant_scale_ckr
            {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // smooth_scales_cq
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                     // actual_seq_len
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // k_nope_clip_alpha
        },
        {
            {{{8, 1, 32, 512}, {8, 1, 32, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // query
            {{{8, 1, 32, 64}, {8, 1, 32, 64}}, ge::DT_BF16, ge::FORMAT_ND},       // query_rope
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND}, // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_INT8, ge::FORMAT_ND},   // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                             // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                             // dequant_scale_q_norm
        },
        {{"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
         {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
         {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
         {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3933345;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 950的半量化kv量化场景异常case：Get tileSize is nullptr.
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test129)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 1, 7168}, {8, 1, 7168}}, ge::DT_BF16, ge::FORMAT_ND},                               // token_x
            {{{7168, 1536}, {7168, 1536}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},                       // weight_dq
            {{{1536, 32 * (128 + 64)}, {1536, 32 * (128 + 64)}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // weight_uq_qr
            {{{32, 128, 512}, {32, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                           // weight_uk
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},               // weight_dkv_kr
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                              // rmsnorm_gamma_cq
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                // rmsnorm_gamma_ckv
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // rope_sin
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // rope_cos
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND},        // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_INT8, ge::FORMAT_ND},          // kr_cache
            {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},                             // cache_index
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // dequant_scale_x
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // dequant_scale_w_dq
            {{{1, 32 * (128 + 64)}, {1, 32 * (128 + 64)}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dequant_scale_w_uq_qr
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                      // dequant_scale_w_dkv_kr
            {{{1, 512}, {1, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckv
            {{{1, 64}, {1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},                           // quant_scale_ckr
            {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // smooth_scales_cq
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                     // actual_seq_len
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // k_nope_clip_alpha
        },
        {
            {{{8, 1, 32, 512}, {8, 1, 32, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // query
            {{{8, 1, 32, 64}, {8, 1, 32, 64}}, ge::DT_BF16, ge::FORMAT_ND},       // query_rope
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND}, // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_INT8, ge::FORMAT_ND},   // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                             // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                             // dequant_scale_q_norm
        },
        {{"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
         {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
         {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
         {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3933345;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 950的半量化kv量化场景异常case：Get qcQrScale is nullptr.
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test130)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 1, 7168}, {8, 1, 7168}}, ge::DT_BF16, ge::FORMAT_ND},                               // token_x
            {{{7168, 1536}, {7168, 1536}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},                       // weight_dq
            {{{1536, 32 * (128 + 64)}, {1536, 32 * (128 + 64)}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // weight_uq_qr
            {{{32, 128, 512}, {32, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                           // weight_uk
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},               // weight_dkv_kr
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                              // rmsnorm_gamma_cq
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                // rmsnorm_gamma_ckv
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // rope_sin
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // rope_cos
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND},        // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_INT8, ge::FORMAT_ND},          // kr_cache
            {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},                             // cache_index
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // dequant_scale_x
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // dequant_scale_w_dq
            {{{1, 32 * (128 + 64)}, {1, 32 * (128 + 64)}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dequant_scale_w_uq_qr
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                      // dequant_scale_w_dkv_kr
            {{{1, 512}, {1, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckv
            {{{1, 64}, {1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},                           // quant_scale_ckr
            {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // smooth_scales_cq
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                     // actual_seq_len
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // k_nope_clip_alpha
        },
        {
            {{{8, 1, 32, 512}, {8, 1, 32, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // query
            {{{8, 1, 32, 64}, {8, 1, 32, 64}}, ge::DT_BF16, ge::FORMAT_ND},       // query_rope
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND}, // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_INT8, ge::FORMAT_ND},   // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                             // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                             // dequant_scale_q_norm
        },
        {{"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
         {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
         {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
         {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}},
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3933345;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 950的半量化kv量化场景异常case：Get kcScale is nullptr.
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test131)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 1, 7168}, {8, 1, 7168}}, ge::DT_BF16, ge::FORMAT_ND},                               // token_x
            {{{7168, 1536}, {7168, 1536}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},                       // weight_dq
            {{{1536, 32 * (128 + 64)}, {1536, 32 * (128 + 64)}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // weight_uq_qr
            {{{32, 128, 512}, {32, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                           // weight_uk
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},               // weight_dkv_kr
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                              // rmsnorm_gamma_cq
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                // rmsnorm_gamma_ckv
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // rope_sin
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // rope_cos
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND},        // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_INT8, ge::FORMAT_ND},          // kr_cache
            {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},                             // cache_index
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // dequant_scale_x
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // dequant_scale_w_dq
            {{{1, 32 * (128 + 64)}, {1, 32 * (128 + 64)}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dequant_scale_w_uq_qr
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                      // dequant_scale_w_dkv_kr
            {{{1, 512}, {1, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckv
            {{{1, 64}, {1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},                           // quant_scale_ckr
            {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // smooth_scales_cq
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                     // actual_seq_len
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // k_nope_clip_alpha
        },
        {
            {{{8, 1, 32, 512}, {8, 1, 32, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // query
            {{{8, 1, 32, 64}, {8, 1, 32, 64}}, ge::DT_BF16, ge::FORMAT_ND},       // query_rope
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND}, // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_INT8, ge::FORMAT_ND},   // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                             // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                             // dequant_scale_q_norm
        },
        {{"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
         {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
         {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
         {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
         {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)}},
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3933345;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 950的半量化kv量化场景异常case：Get cacheMode is nullptr
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test132)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 1, 7168}, {8, 1, 7168}}, ge::DT_BF16, ge::FORMAT_ND},                               // token_x
            {{{7168, 1536}, {7168, 1536}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},                       // weight_dq
            {{{1536, 32 * (128 + 64)}, {1536, 32 * (128 + 64)}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // weight_uq_qr
            {{{32, 128, 512}, {32, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                           // weight_uk
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},               // weight_dkv_kr
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                              // rmsnorm_gamma_cq
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                // rmsnorm_gamma_ckv
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // rope_sin
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // rope_cos
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND},        // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_INT8, ge::FORMAT_ND},          // kr_cache
            {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},                             // cache_index
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // dequant_scale_x
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // dequant_scale_w_dq
            {{{1, 32 * (128 + 64)}, {1, 32 * (128 + 64)}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dequant_scale_w_uq_qr
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                      // dequant_scale_w_dkv_kr
            {{{1, 512}, {1, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckv
            {{{1, 64}, {1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},                           // quant_scale_ckr
            {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // smooth_scales_cq
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                     // actual_seq_len
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // k_nope_clip_alpha
        },
        {
            {{{8, 1, 32, 512}, {8, 1, 32, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // query
            {{{8, 1, 32, 64}, {8, 1, 32, 64}}, ge::DT_BF16, ge::FORMAT_ND},       // query_rope
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND}, // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_INT8, ge::FORMAT_ND},   // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                             // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                             // dequant_scale_q_norm
        },
        {{"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
         {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)}},
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3933345;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}
