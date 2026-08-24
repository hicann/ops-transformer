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

// 全量化kvcache pertensor量化场景正常case
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test0)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
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
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND},        // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},          // kr_cache
            {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},                             // cache_index
            {{{8 * 1, 1}, {8 * 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},                     // dequant_scale_x
            {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // dequant_scale_w_dq
            {{{1, 32 * (128 + 64)}, {1, 32 * (128 + 64)}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dequant_scale_w_uq_qr
            {{{1, 512 + 64}, {1, 512 + 64}}, ge::DT_FLOAT, ge::FORMAT_ND},               // dequant_scale_w_dkv_kr
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // quant_scale_ckv
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // quant_scale_ckr
            {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // smooth_scales_cq
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                     // actual_seq_len
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // k_nope_clip_alpha
        },
        {
            {{{8, 1, 32, 512}, {8, 1, 32, 512}}, ge::DT_INT8, ge::FORMAT_ND},     // query
            {{{8, 1, 32, 64}, {8, 1, 32, 64}}, ge::DT_BF16, ge::FORMAT_ND},       // query_rope
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND}, // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},   // kr_cache
            {{{8 * 1, 32, 1}, {8 * 1, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},      // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                             // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                             // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend910_B3", MlaPrologV3_tiling_A2SocInfo, 4096);
    int64_t expectTilingKey = 3933473;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 全量化kvcache pertensor量化场景异常case：When weightQuantMode == 2, kvQuantMode must be within {0, 1, 3}, actually
// is 2.
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test1)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
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
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND},        // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},          // kr_cache
            {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},                             // cache_index
            {{{8 * 1, 1}, {8 * 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},                     // dequant_scale_x
            {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // dequant_scale_w_dq
            {{{1, 32 * (128 + 64)}, {1, 32 * (128 + 64)}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dequant_scale_w_uq_qr
            {{{1, 512 + 64}, {1, 512 + 64}}, ge::DT_FLOAT, ge::FORMAT_ND},               // dequant_scale_w_dkv_kr
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // quant_scale_ckv
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // quant_scale_ckr
            {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // smooth_scales_cq
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                     // actual_seq_len
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // k_nope_clip_alpha
        },
        {
            {{{8, 1, 32, 512}, {8, 1, 32, 512}}, ge::DT_INT8, ge::FORMAT_ND},     // query
            {{{8, 1, 32, 64}, {8, 1, 32, 64}}, ge::DT_BF16, ge::FORMAT_ND},       // query_rope
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND}, // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},   // kr_cache
            {{{8 * 1, 32, 1}, {8 * 1, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},      // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                             // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                             // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend910_B3", MlaPrologV3_tiling_A2SocInfo, 4096);
    int64_t expectTilingKey = 3933473;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 全量化kvcache pertensor量化场景异常case：WeightQuantMode must be within {0, 1, 2，3}, actually is 5.
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test2)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
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
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND},        // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},          // kr_cache
            {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},                             // cache_index
            {{{8 * 1, 1}, {8 * 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},                     // dequant_scale_x
            {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // dequant_scale_w_dq
            {{{1, 32 * (128 + 64)}, {1, 32 * (128 + 64)}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dequant_scale_w_uq_qr
            {{{1, 512 + 64}, {1, 512 + 64}}, ge::DT_FLOAT, ge::FORMAT_ND},               // dequant_scale_w_dkv_kr
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // quant_scale_ckv
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // quant_scale_ckr
            {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // smooth_scales_cq
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                     // actual_seq_len
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                                      // k_nope_clip_alpha
        },
        {
            {{{8, 1, 32, 512}, {8, 1, 32, 512}}, ge::DT_INT8, ge::FORMAT_ND},     // query
            {{{8, 1, 32, 64}, {8, 1, 32, 64}}, ge::DT_BF16, ge::FORMAT_ND},       // query_rope
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND}, // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},   // kr_cache
            {{{8 * 1, 32, 1}, {8 * 1, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},      // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                             // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                             // dequant_scale_q_norm
        },
        {{"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
         {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
         {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
         {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
         {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)}},
        &compileInfo, "Ascend910_B3", MlaPrologV3_tiling_A2SocInfo, 4096);
    int64_t expectTilingKey = 3933473;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 全量化kvcache非量化场景正常case
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test3)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 1, 6144}, {8, 1, 6144}}, ge::DT_INT8, ge::FORMAT_ND},                               // token_x
            {{{6144, 1536}, {6144, 1536}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},                       // weight_dq
            {{{1536, 32 * (128 + 64)}, {1536, 32 * (128 + 64)}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // weight_uq_qr
            {{{32, 128, 512}, {32, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                           // weight_uk
            {{{6144, 512 + 64}, {6144, 512 + 64}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},               // weight_dkv_kr
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                              // rmsnorm_gamma_cq
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                // rmsnorm_gamma_ckv
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // rope_sin
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // rope_cos
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},        // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},          // kr_cache
            {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},                             // cache_index
            {{{8 * 1, 1}, {8 * 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},                     // dequant_scale_x
            {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // dequant_scale_w_dq
            {{{1, 32 * (128 + 64)}, {1, 32 * (128 + 64)}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dequant_scale_w_uq_qr
            {{{1, 512 + 64}, {1, 512 + 64}}, ge::DT_FLOAT, ge::FORMAT_ND},               // dequant_scale_w_dkv_kr
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // quant_scale_ckv
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // quant_scale_ckr
            {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // smooth_scales_cq
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                     // actual_seq_len
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // k_nope_clip_alpha
        },
        {
            {{{8, 1, 32, 512}, {8, 1, 32, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // query
            {{{8, 1, 32, 64}, {8, 1, 32, 64}}, ge::DT_BF16, ge::FORMAT_ND},       // query_rope
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},   // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                             // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                             // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend910_B3", MlaPrologV3_tiling_A2SocInfo, 4096);
    int64_t expectTilingKey = 3933409;
    string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// 全量化kvcache pertensor量化 NZ
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test4)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 1, 7168}, {8, 1, 7168}}, ge::DT_INT8, ge::FORMAT_ND}, // token_x
            {{{1536 / 32, 7168 / 16, 16, 32}, {1536 / 32, 7168 / 16, 16, 32}},
             ge::DT_INT8,
             ge::FORMAT_FRACTAL_NZ}, // weight_dq
            {{{128 + 64, 1536 / 16, 16, 32}, {128 + 64, 1536 / 16, 16, 32}},
             ge::DT_INT8,
             ge::FORMAT_FRACTAL_NZ},                                        // weight_uq_qr
            {{{32, 128, 512}, {32, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // weight_uk
            {{{(512 + 64) / 32, 7168 / 16, 16, 32}, {(512 + 64) / 32, 7168 / 16, 16, 32}},
             ge::DT_INT8,
             ge::FORMAT_FRACTAL_NZ},                                                     // weight_dkv_kr
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                              // rmsnorm_gamma_cq
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                // rmsnorm_gamma_ckv
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // rope_sin
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // rope_cos
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND},        // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},          // kr_cache
            {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},                             // cache_index
            {{{8 * 1, 1}, {8 * 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},                     // dequant_scale_x
            {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // dequant_scale_w_dq
            {{{1, 32 * (128 + 64)}, {1, 32 * (128 + 64)}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dequant_scale_w_uq_qr
            {{{1, 512 + 64}, {1, 512 + 64}}, ge::DT_FLOAT, ge::FORMAT_ND},               // dequant_scale_w_dkv_kr
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // quant_scale_ckv
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // quant_scale_ckr
            {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // smooth_scales_cq
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                     // actual_seq_len
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // k_nope_clip_alpha
        },
        {
            {{{8, 1, 32, 512}, {8, 1, 32, 512}}, ge::DT_INT8, ge::FORMAT_ND},     // query
            {{{8, 1, 32, 64}, {8, 1, 32, 64}}, ge::DT_BF16, ge::FORMAT_ND},       // query_rope
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND}, // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},   // kr_cache
            {{{8 * 1, 32, 1}, {8 * 1, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},      // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                             // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                             // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BLK_NZ")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend910_B3", MlaPrologV3_tiling_A2SocInfo, 4096);
    int64_t expectTilingKey = 3933476;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 非PA非量化 TND
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test5)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{16, 7168}, {16, 7168}}, ge::DT_BF16, ge::FORMAT_ND},                     // token_x 0
            {{{7168, 1536}, {7168, 1536}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 24576}, {1536, 24576}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},       // weight_uq_qr 2
            {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},           // weight_uk 3
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                             // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                               // rmsnorm_gamma_ckv 6
            {{{16, 64}, {16, 64}}, ge::DT_BF16, ge::FORMAT_ND},                         // rope_sin 7
            {{{16, 64}, {16, 64}}, ge::DT_BF16, ge::FORMAT_ND},                         // rope_cos 8
            {{{16, 1, 512}, {16, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},                 // kv_cache 9
            {{{16, 1, 64}, {16, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                   // kr_cache 10
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                    // cache_index 11
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // dequant_scale_x 12
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // dequant_scale_w_dq 13
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // dequant_scale_w_uq_qr 14
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // dequant_scale_w_dkv_kr 15
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                    // actual_seq_len 19
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // k_nope_clip_alpha 20
        },
        {
            {{{16, 128, 512}, {16, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // query
            {{{16, 128, 64}, {16, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},   // query_rope
            {{{16, 1, 512}, {16, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // kv_cache
            {{{16, 1, 64}, {16, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},       // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                      // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_BF16, ge::FORMAT_ND},                       // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                       // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.00317766385216254)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00488810249669485)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
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
        &compileInfo, "Ascend910_B3", MlaPrologV3_tiling_A2SocInfo, 4096);
    int64_t expectTilingKey = 3932176;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 非PA半量化 BSND qnorm_flag==true /mlaprolog_L0_1_128_1_11776_11776_128_BSND_1_000035
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test6)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{1, 11776, 7168}, {1, 11776, 7168}}, ge::DT_BF16, ge::FORMAT_ND},         // token_x 0
            {{{7168, 1536}, {7168, 1536}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 24576}, {1536, 24576}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},       // weight_uq_qr 2
            {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},           // weight_uk 3
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                             // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                               // rmsnorm_gamma_ckv 6
            {{{1, 11776, 64}, {1, 11776, 64}}, ge::DT_BF16, ge::FORMAT_ND},             // rope_sin 7
            {{{1, 11776, 64}, {1, 11776, 64}}, ge::DT_BF16, ge::FORMAT_ND},             // rope_cos 8
            {{{1, 11776, 1, 512}, {1, 11776, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND},     // kv_cache 9
            {{{1, 11776, 1, 64}, {1, 11776, 1, 64}}, ge::DT_INT8, ge::FORMAT_ND},       // kr_cache 10
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                    // cache_index 11
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // dequant_scale_x 12
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // dequant_scale_w_dq 13
            {{{1, 24576}, {1, 24576}}, ge::DT_FLOAT, ge::FORMAT_ND},                    // dequant_scale_w_uq_qr 14
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // dequant_scale_w_dkv_kr 15
            {{{1, 512}, {1, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},                        // quant_scale_ckv 16
            {{{1, 64}, {1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},                          // quant_scale_ckr 17
            {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},                      // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                    // actual_seq_len 19
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // k_nope_clip_alpha 20
        },
        {
            {{{1, 11776, 128, 512}, {1, 11776, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // query
            {{{1, 11776, 128, 64}, {1, 11776, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},   // query_rope
            {{{1, 11776, 1, 512}, {1, 11776, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND},     // kv_cache
            {{{1, 11776, 1, 64}, {1, 11776, 1, 64}}, ge::DT_INT8, ge::FORMAT_ND},       // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                                  // dequant_scale_q_nope
            {{{1, 11776, 1536}, {1, 11776, 1536}}, ge::DT_INT8, ge::FORMAT_ND},         // query_norm
            {{{11776, 1}, {11776, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},                    // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.000875660550762735)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.000607527341571764)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend910_B3", MlaPrologV3_tiling_A2SocInfo, 4096);
    int64_t expectTilingKey = 3933344;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 非PA全量化 BSND qnorm_flag==false /mlaprolog_L0_1_128_1_1024_1024_128_BSND_2_000037
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test7)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{1, 1024, 7168}, {1, 1024, 7168}}, ge::DT_INT8, ge::FORMAT_ND},           // token_x 0
            {{{7168, 1536}, {7168, 1536}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 24576}, {1536, 24576}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},       // weight_uq_qr 2
            {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},           // weight_uk 3
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                             // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                               // rmsnorm_gamma_ckv 6
            {{{1, 1024, 64}, {1, 1024, 64}}, ge::DT_BF16, ge::FORMAT_ND},               // rope_sin 7
            {{{1, 1024, 64}, {1, 1024, 64}}, ge::DT_BF16, ge::FORMAT_ND},               // rope_cos 8
            {{{1, 1024, 1, 512}, {1, 1024, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},       // kv_cache 9
            {{{1, 1024, 1, 64}, {1, 1024, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},         // kr_cache 10
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                    // cache_index 11
            {{{1024, 1}, {1024, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},                      // dequant_scale_x 12
            {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},                      // dequant_scale_w_dq 13
            {{{1, 24576}, {1, 24576}}, ge::DT_FLOAT, ge::FORMAT_ND},                    // dequant_scale_w_uq_qr 14
            {{{1, 576}, {1, 576}}, ge::DT_FLOAT, ge::FORMAT_ND},                        // dequant_scale_w_dkv_kr 15
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                    // actual_seq_len 19
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // k_nope_clip_alpha 20
        },
        {
            {{{1, 1024, 128, 512}, {1, 1024, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // query
            {{{1, 1024, 128, 64}, {1, 1024, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},   // query_rope
            {{{1, 1024, 1, 512}, {1, 1024, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // kv_cache
            {{{1, 1024, 1, 64}, {1, 1024, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},       // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                                // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                                 // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                                // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.00263855234125857)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00306797597892537)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend910_B3", MlaPrologV3_tiling_A2SocInfo, 4096);
    int64_t expectTilingKey = 3933408;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 非PA全量化 TND qnorm_flag==false /mlaprolog_L0_4_128_1_2_2_128_TND_2_000016
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test8)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_INT8, ge::FORMAT_ND},                       // token_x 0
            {{{7168, 1536}, {7168, 1536}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 24576}, {1536, 24576}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},       // weight_uq_qr 2
            {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},           // weight_uk 3
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                             // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                               // rmsnorm_gamma_ckv 6
            {{{8, 64}, {8, 64}}, ge::DT_BF16, ge::FORMAT_ND},                           // rope_sin 7
            {{{8, 64}, {8, 64}}, ge::DT_BF16, ge::FORMAT_ND},                           // rope_cos 8
            {{{8, 1, 656}, {8, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},                   // kv_cache 9
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                                   // kr_cache 10
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                    // cache_index 11
            {{{8, 1}, {8, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // dequant_scale_x 12
            {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},                      // dequant_scale_w_dq 13
            {{{1, 24576}, {1, 24576}}, ge::DT_FLOAT, ge::FORMAT_ND},                    // dequant_scale_w_uq_qr 14
            {{{1, 576}, {1, 576}}, ge::DT_FLOAT, ge::FORMAT_ND},                        // dequant_scale_w_dkv_kr 15
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                    // actual_seq_len 19
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                                  // k_nope_clip_alpha 20
        },
        {
            {{{8, 128, 512}, {8, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // query
            {{{8, 128, 64}, {8, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},   // query_rope
            {{{8, 1, 656}, {8, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},     // kv_cache
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                     // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                    // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                     // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                    // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0022971812167027)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00235037235057241)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend910_B3", MlaPrologV3_tiling_A2SocInfo, 4096);
    int64_t expectTilingKey = 3933600;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 非PA全量化 BSND qnorm_flag==TRUE /mlaprolog_L0_1_128_1_6016_6016_128_BSND_2_000020
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test9)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{1, 6016, 7168}, {1, 6016, 7168}}, ge::DT_INT8, ge::FORMAT_ND},           // token_x 0
            {{{7168, 1536}, {7168, 1536}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 24576}, {1536, 24576}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},       // weight_uq_qr 2
            {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},           // weight_uk 3
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                             // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                               // rmsnorm_gamma_ckv 6
            {{{1, 6016, 64}, {1, 6016, 64}}, ge::DT_BF16, ge::FORMAT_ND},               // rope_sin 7
            {{{1, 6016, 64}, {1, 6016, 64}}, ge::DT_BF16, ge::FORMAT_ND},               // rope_cos 8
            {{{1, 6016, 1, 512}, {1, 6016, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND},       // kv_cache 9
            {{{1, 6016, 1, 64}, {1, 6016, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},         // kr_cache 10
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                    // cache_index 11
            {{{6016, 1}, {6016, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},                      // dequant_scale_x 12
            {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},                      // dequant_scale_w_dq 13
            {{{1, 24576}, {1, 24576}}, ge::DT_FLOAT, ge::FORMAT_ND},                    // dequant_scale_w_uq_qr 14
            {{{1, 576}, {1, 576}}, ge::DT_FLOAT, ge::FORMAT_ND},                        // dequant_scale_w_dkv_kr 15
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                                  // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                    // actual_seq_len 19
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // k_nope_clip_alpha 20
        },
        {
            {{{1, 6016, 128, 512}, {1, 6016, 128, 512}}, ge::DT_INT8, ge::FORMAT_ND}, // query
            {{{1, 6016, 128, 64}, {1, 6016, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},   // query_rope
            {{{1, 6016, 1, 512}, {1, 6016, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND},     // kv_cache
            {{{1, 6016, 1, 64}, {1, 6016, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},       // kr_cache
            {{{6016, 128, 1}, {6016, 128, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},          // dequant_scale_q_nope
            {{{1, 6016, 1536}, {1, 6016, 1536}}, ge::DT_INT8, ge::FORMAT_ND},         // query_norm
            {{{6016, 1}, {6016, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},                    // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.000971848152426739)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00141311948888896)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend910_B3", MlaPrologV3_tiling_A2SocInfo, 4096);
    int64_t expectTilingKey = 3933472;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 非PA全量化 TND qnorm_flag==false /mlaprolog_L0_1_128_1_512_512_128_TND_1_000015
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test10)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{512, 7168}, {512, 7168}}, ge::DT_BF16, ge::FORMAT_ND},                   // token_x 0
            {{{7168, 1536}, {7168, 1536}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 24576}, {1536, 24576}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},       // weight_uq_qr 2
            {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},           // weight_uk 3
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                             // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                               // rmsnorm_gamma_ckv 6
            {{{512, 64}, {512, 64}}, ge::DT_BF16, ge::FORMAT_ND},                       // rope_sin 7
            {{{512, 64}, {512, 64}}, ge::DT_BF16, ge::FORMAT_ND},                       // rope_cos 8
            {{{512, 1, 656}, {512, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},               // kv_cache 9
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                                   // kr_cache 10
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                    // cache_index 11
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // dequant_scale_x 12
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // dequant_scale_w_dq 13
            {{{1, 24576}, {1, 24576}}, ge::DT_FLOAT, ge::FORMAT_ND},                    // dequant_scale_w_uq_qr 14
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // dequant_scale_w_dkv_kr 15
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                    // actual_seq_len 19
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                                  // k_nope_clip_alpha 20
        },
        {
            {{{512, 128, 512}, {512, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // query
            {{{512, 128, 64}, {512, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},   // query_rope
            {{{512, 1, 656}, {512, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},     // kv_cache
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                         // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                        // dequant_scale_q_nope
            {{{512, 1536}, {512, 1536}}, ge::DT_INT8, ge::FORMAT_ND},         // query_norm
            {{{512, 1}, {512, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},              // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.000651222723526155)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00268072038395418)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend910_B3", MlaPrologV3_tiling_A2SocInfo, 4096);
    int64_t expectTilingKey = 3933536;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// mlaprolog_L0_2_128_1_4825_65536_128_TND_PA_BLK_NZ_1_2_000028
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test11)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{9650, 6144}, {9650, 6144}}, ge::DT_BF16, ge::FORMAT_ND},             // token_x 0
            {{{6144, 1536}, {6144, 1536}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},     // weight_dq 1
            {{{1536, 24576}, {1536, 24576}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},   // weight_uq_qr 2
            {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},       // weight_uk 3
            {{{6144, 576}, {6144, 576}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},       // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                         // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                           // rmsnorm_gamma_ckv 6
            {{{9650, 64}, {9650, 64}}, ge::DT_BF16, ge::FORMAT_ND},                 // rope_sin 7
            {{{9650, 64}, {9650, 64}}, ge::DT_BF16, ge::FORMAT_ND},                 // rope_cos 8
            {{{512, 256, 1, 512}, {512, 256, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND}, // kv_cache 9
            {{{512, 256, 1, 64}, {512, 256, 1, 64}}, ge::DT_INT8, ge::FORMAT_ND},   // kr_cache 10
            {{{38}, {38}}, ge::DT_INT64, ge::FORMAT_ND},                            // cache_index 11
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                // dequant_scale_x 12
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                // dequant_scale_w_dq 13
            {{{1, 24576}, {1, 24576}}, ge::DT_FLOAT, ge::FORMAT_ND},                // dequant_scale_w_uq_qr 14
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                // dequant_scale_w_dkv_kr 15
            {{{1, 512}, {1, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},                    // quant_scale_ckv 16
            {{{1, 64}, {1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},                      // quant_scale_ckr 17
            {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},                  // smooth_scales_cq 18
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},                              // actual_seq_len 19
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                // k_nope_clip_alpha 20
        },
        {
            {{{9650, 128, 512}, {9650, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // query
            {{{9650, 128, 64}, {9650, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},       // query_rope
            {{{512, 256, 1, 512}, {512, 256, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND}, // kv_cache
            {{{512, 256, 1, 64}, {512, 256, 1, 64}}, ge::DT_INT8, ge::FORMAT_ND},   // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                               // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.00324166008356131)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00218671504998371)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BLK_NZ")},
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
        &compileInfo, "Ascend910_B3", MlaPrologV3_tiling_A2SocInfo, 4096);
    int64_t expectTilingKey = 3949732;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// mlaprolog_L0_16_128_1_2_2_128_BSND_1_000000
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test12)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{16, 2, 7168}, {16, 2, 7168}}, ge::DT_BF16, ge::FORMAT_ND},         // token_x 0
            {{{7168, 1536}, {7168, 1536}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},   // weight_dq 1
            {{{1536, 24576}, {1536, 24576}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // weight_uq_qr 2
            {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // weight_uk 3
            {{{7168, 576}, {7168, 576}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},     // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                       // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                         // rmsnorm_gamma_ckv 6
            {{{16, 2, 64}, {16, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},             // rope_sin 7
            {{{16, 2, 64}, {16, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},             // rope_cos 8
            {{{16, 2, 1, 512}, {16, 2, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // kv_cache 9
            {{{16, 2, 1, 64}, {16, 2, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},       // kr_cache 10
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                              // cache_index 11
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // dequant_scale_x 12
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // dequant_scale_w_dq 13
            {{{1, 24576}, {1, 24576}}, ge::DT_FLOAT, ge::FORMAT_ND},              // dequant_scale_w_uq_qr 14
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // dequant_scale_w_dkv_kr 15
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // quant_scale_ckr 17
            {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},                // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                              // actual_seq_len 19
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // k_nope_clip_alpha 20
        },
        {
            {{{16, 2, 128, 512}, {16, 2, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // query
            {{{16, 2, 128, 64}, {16, 2, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},   // query_rope
            {{{16, 2, 1, 512}, {16, 2, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // kv_cache
            {{{16, 2, 1, 64}, {16, 2, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},       // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // dequant_scale_q_nope
            {{{16, 2, 1536}, {16, 2, 1536}}, ge::DT_INT8, ge::FORMAT_ND},         // query_norm
            {{{32, 1}, {32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},                    // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.00457361635559103)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00484664050635162)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend910_B3", MlaPrologV3_tiling_A2SocInfo, 4096);
    int64_t expectTilingKey = 3933280;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// mlaprolog_L0_4_128_1_2317_65536_128_BSND_PA_BLK_NZ_1_0_000011
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test13)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{4, 2317, 7168}, {4, 2317, 7168}}, ge::DT_BF16, ge::FORMAT_ND},         // token_x 0
            {{{7168, 1536}, {7168, 1536}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},       // weight_dq 1
            {{{1536, 24576}, {1536, 24576}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},     // weight_uq_qr 2
            {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},         // weight_uk 3
            {{{7168, 576}, {7168, 576}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},         // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                           // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                             // rmsnorm_gamma_ckv 6
            {{{4, 2317, 64}, {4, 2317, 64}}, ge::DT_BF16, ge::FORMAT_ND},             // rope_sin 7
            {{{4, 2317, 64}, {4, 2317, 64}}, ge::DT_BF16, ge::FORMAT_ND},             // rope_cos 8
            {{{2048, 128, 1, 512}, {2048, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // kv_cache 9
            {{{2048, 128, 1, 64}, {2048, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},   // kr_cache 10
            {{{4, 19}, {4, 19}}, ge::DT_INT64, ge::FORMAT_ND},                        // cache_index 11
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                  // dequant_scale_x 12
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                  // dequant_scale_w_dq 13
            {{{1, 24576}, {1, 24576}}, ge::DT_FLOAT, ge::FORMAT_ND},                  // dequant_scale_w_uq_qr 14
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                  // dequant_scale_w_dkv_kr 15
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                  // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                  // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                  // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                  // actual_seq_len 19
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                  // k_nope_clip_alpha 20
        },
        {
            {{{4, 2317, 128, 512}, {4, 2317, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // query
            {{{4, 2317, 128, 64}, {4, 2317, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},   // query_rope
            {{{2048, 128, 1, 512}, {2048, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // kv_cache
            {{{2048, 128, 1, 64}, {2048, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},   // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                                // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                                 // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                                // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0033201761781808)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.0034101718259947)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BLK_NZ")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend910_B3", MlaPrologV3_tiling_A2SocInfo, 4096);
    int64_t expectTilingKey = 3933284;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 非量化场景正常case
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test14)
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
        &compileInfo, "Ascend910_B3", MlaPrologV3_tiling_A2SocInfo, 4096);
    int64_t expectTilingKey = 3932177;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 非量化场景异常case：When weightQuantMode == 0, kvQuantMode must be within {0}, actually is 1.
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test15)
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
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend910_B3", MlaPrologV3_tiling_A2SocInfo, 4096);
    int64_t expectTilingKey = 3932177;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 半量化kv量化场景正常case
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test16)
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
        &compileInfo, "Ascend910_B3", MlaPrologV3_tiling_A2SocInfo, 4096);
    int64_t expectTilingKey = 3933345;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 半量化kv量化场景异常case：WeightQuantMode must be within {0, 1, 2}, actually is 3.
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test17)
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
        &compileInfo, "Ascend910_B3", MlaPrologV3_tiling_A2SocInfo, 4096);
    int64_t expectTilingKey = 3933345;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 半量化kv量化场景异常case：When weightQuantMode == 1, kvQuantMode must be within {0, 2, 3}, actually is 1.
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test18)
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
        &compileInfo, "Ascend910_B3", MlaPrologV3_tiling_A2SocInfo, 4096);
    int64_t expectTilingKey = 3933345;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 半量化kv量化场景异常case：Only support cacheMode {BSND, TND, PA_BSND, PA_NZ, PA_BLK_BSND, PA_BLK_NZ}, actually is
// ABCD.
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test19)
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
        &compileInfo, "Ascend910_B3", MlaPrologV3_tiling_A2SocInfo, 4096);
    int64_t expectTilingKey = 3933345;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 半量化kv量化场景异常case：The ckvkrRepoMode expected 0, but got 1.
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test20)
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
        &compileInfo, "Ascend910_B3", MlaPrologV3_tiling_A2SocInfo, 4096);
    int64_t expectTilingKey = 3933345;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 半量化kv量化场景异常case：The quantScaleRepoMode expected 0, but got 1.
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test21)
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
        &compileInfo, "Ascend910_B3", MlaPrologV3_tiling_A2SocInfo, 4096);
    int64_t expectTilingKey = 3933345;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 半量化kv量化场景异常case：The queryQuantMode expected 0, but got 1.
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test22)
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
        &compileInfo, "Ascend910_B3", MlaPrologV3_tiling_A2SocInfo, 4096);
    int64_t expectTilingKey = 3933345;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 半量化kv量化场景异常case：Get rmsnormEpsilonCq is nullptr
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test23)
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
        {}, &compileInfo, "Ascend910_B3", MlaPrologV3_tiling_A2SocInfo, 4096);
    int64_t expectTilingKey = 3933345;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 半量化kv量化场景异常case：Get rmsnormEpsilonCkv is nullptr
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test24)
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
        {{"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)}}, &compileInfo, "Ascend910_B3",
        MlaPrologV3_tiling_A2SocInfo, 4096);
    int64_t expectTilingKey = 3933345;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 半量化kv量化场景异常case：Get queryNormFlag is nullptr.
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test25)
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
        &compileInfo, "Ascend910_B3", MlaPrologV3_tiling_A2SocInfo, 4096);
    int64_t expectTilingKey = 3933345;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 半量化kv量化场景异常case：Get kvQuantMode is nullptr.
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test26)
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
        &compileInfo, "Ascend910_B3", MlaPrologV3_tiling_A2SocInfo, 4096);
    int64_t expectTilingKey = 3933345;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 半量化kv量化场景异常case：Get queryQuantMode is nullptr.
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test27)
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
        &compileInfo, "Ascend910_B3", MlaPrologV3_tiling_A2SocInfo, 4096);
    int64_t expectTilingKey = 3933345;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 半量化kv量化场景异常case：Get ckvkrRepoMode is nullptr.
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test28)
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
        &compileInfo, "Ascend910_B3", MlaPrologV3_tiling_A2SocInfo, 4096);
    int64_t expectTilingKey = 3933345;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 半量化kv量化场景异常case：Get quantScaleRepoMode is nullptr.
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test29)
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
        &compileInfo, "Ascend910_B3", MlaPrologV3_tiling_A2SocInfo, 4096);
    int64_t expectTilingKey = 3933345;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 半量化kv量化场景异常case：Get tileSize is nullptr.
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test30)
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
        &compileInfo, "Ascend910_B3", MlaPrologV3_tiling_A2SocInfo, 4096);
    int64_t expectTilingKey = 3933345;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 半量化kv量化场景异常case：Get qcQrScale is nullptr.
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test31)
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
        &compileInfo, "Ascend910_B3", MlaPrologV3_tiling_A2SocInfo, 4096);
    int64_t expectTilingKey = 3933345;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 半量化kv量化场景异常case：Get kcScale is nullptr.
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test32)
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
        &compileInfo, "Ascend910_B3", MlaPrologV3_tiling_A2SocInfo, 4096);
    int64_t expectTilingKey = 3933345;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 半量化kv量化场景异常case：Get cacheMode is nullptr
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test33)
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
        &compileInfo, "Ascend910_B3", MlaPrologV3_tiling_A2SocInfo, 4096);
    int64_t expectTilingKey = 3933345;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 全量化pertile场景正常case
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test34)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_INT8, ge::FORMAT_ND},                       // token_x 0
            {{{7168, 1536}, {7168, 1536}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 24576}, {1536, 24576}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},       // weight_uq_qr 2
            {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},           // weight_uk 3
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                             // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                               // rmsnorm_gamma_ckv 6
            {{{8, 64}, {8, 64}}, ge::DT_BF16, ge::FORMAT_ND},                           // rope_sin 7
            {{{8, 64}, {8, 64}}, ge::DT_BF16, ge::FORMAT_ND},                           // rope_cos 8
            {{{1, 16, 1, 656}, {1, 16, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},           // kv_cache 9
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                                   // kr_cache 10
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND},                                  // cache_index 11
            {{{8, 1}, {8, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // dequant_scale_x 12
            {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},                      // dequant_scale_w_dq 13
            {{{1, 24576}, {1, 24576}}, ge::DT_FLOAT, ge::FORMAT_ND},                    // dequant_scale_w_uq_qr 14
            {{{1, 576}, {1, 576}}, ge::DT_FLOAT, ge::FORMAT_ND},                        // dequant_scale_w_dkv_kr 15
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                    // actual_seq_len 19
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                                  // k_nope_clip_alpha 20
        },
        {
            {{{8, 128, 512}, {8, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // query
            {{{8, 128, 64}, {8, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},       // query_rope
            {{{1, 16, 1, 656}, {1, 16, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND}, // kv_cache
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                         // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                        // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                         // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                        // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0022971812167027)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00235037235057241)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend910_B3", MlaPrologV3_tiling_A2SocInfo, 4096);
    int64_t expectTilingKey = 3933601;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 全量化pertile场景正常case: krcache dtype = bf16
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test35)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_INT8, ge::FORMAT_ND},                       // token_x 0
            {{{7168, 1536}, {7168, 1536}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 24576}, {1536, 24576}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},       // weight_uq_qr 2
            {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},           // weight_uk 3
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                             // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                               // rmsnorm_gamma_ckv 6
            {{{8, 64}, {8, 64}}, ge::DT_BF16, ge::FORMAT_ND},                           // rope_sin 7
            {{{8, 64}, {8, 64}}, ge::DT_BF16, ge::FORMAT_ND},                           // rope_cos 8
            {{{1, 16, 1, 656}, {1, 16, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},           // kv_cache 9
            {{{0}, {0}}, ge::DT_BF16, ge::FORMAT_ND},                                   // kr_cache 10
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND},                                  // cache_index 11
            {{{8, 1}, {8, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // dequant_scale_x 12
            {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},                      // dequant_scale_w_dq 13
            {{{1, 24576}, {1, 24576}}, ge::DT_FLOAT, ge::FORMAT_ND},                    // dequant_scale_w_uq_qr 14
            {{{1, 576}, {1, 576}}, ge::DT_FLOAT, ge::FORMAT_ND},                        // dequant_scale_w_dkv_kr 15
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                    // actual_seq_len 19
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                                  // k_nope_clip_alpha 20
        },
        {
            {{{8, 128, 512}, {8, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // query
            {{{8, 128, 64}, {8, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},       // query_rope
            {{{1, 16, 1, 656}, {1, 16, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND}, // kv_cache
            {{{0}, {0}}, ge::DT_BF16, ge::FORMAT_ND},                         // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                        // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                         // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                        // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0022971812167027)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00235037235057241)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend910_B3", MlaPrologV3_tiling_A2SocInfo, 4096);
    int64_t expectTilingKey = 3933601;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 全量化pertile场景异常case：krCacheOut is not an empty tensor
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test36)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_INT8, ge::FORMAT_ND},                       // token_x 0
            {{{7168, 1536}, {7168, 1536}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 24576}, {1536, 24576}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},       // weight_uq_qr 2
            {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},           // weight_uk 3
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                             // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                               // rmsnorm_gamma_ckv 6
            {{{8, 64}, {8, 64}}, ge::DT_BF16, ge::FORMAT_ND},                           // rope_sin 7
            {{{8, 64}, {8, 64}}, ge::DT_BF16, ge::FORMAT_ND},                           // rope_cos 8
            {{{1, 16, 1, 656}, {1, 16, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},           // kv_cache 9
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                                   // kr_cache 10
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND},                                  // cache_index 11
            {{{8, 1}, {8, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // dequant_scale_x 12
            {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},                      // dequant_scale_w_dq 13
            {{{1, 24576}, {1, 24576}}, ge::DT_FLOAT, ge::FORMAT_ND},                    // dequant_scale_w_uq_qr 14
            {{{1, 576}, {1, 576}}, ge::DT_FLOAT, ge::FORMAT_ND},                        // dequant_scale_w_dkv_kr 15
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                    // actual_seq_len 19
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                                  // k_nope_clip_alpha 20
        },
        {
            {{{8, 128, 512}, {8, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // query
            {{{8, 128, 64}, {8, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},       // query_rope
            {{{1, 16, 1, 656}, {1, 16, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND}, // kv_cache
            {{{10}, {10}}, ge::DT_INT8, ge::FORMAT_ND},                       // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                        // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                         // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                        // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0022971812167027)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00235037235057241)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend910_B3", MlaPrologV3_tiling_A2SocInfo, 4096);
    int64_t expectTilingKey = 3933601;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 全量化pertile场景异常case：The ckvkrRepoMode expected 1, but got 0.
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test37)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_INT8, ge::FORMAT_ND},                       // token_x 0
            {{{7168, 1536}, {7168, 1536}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 24576}, {1536, 24576}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},       // weight_uq_qr 2
            {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},           // weight_uk 3
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                             // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                               // rmsnorm_gamma_ckv 6
            {{{8, 64}, {8, 64}}, ge::DT_BF16, ge::FORMAT_ND},                           // rope_sin 7
            {{{8, 64}, {8, 64}}, ge::DT_BF16, ge::FORMAT_ND},                           // rope_cos 8
            {{{1, 16, 1, 656}, {1, 16, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},           // kv_cache 9
            {{{3, 4, 3}, {3, 4, 3}}, ge::DT_INT8, ge::FORMAT_ND},                       // kr_cache 10
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND},                                  // cache_index 11
            {{{8, 1}, {8, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // dequant_scale_x 12
            {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},                      // dequant_scale_w_dq 13
            {{{1, 24576}, {1, 24576}}, ge::DT_FLOAT, ge::FORMAT_ND},                    // dequant_scale_w_uq_qr 14
            {{{1, 576}, {1, 576}}, ge::DT_FLOAT, ge::FORMAT_ND},                        // dequant_scale_w_dkv_kr 15
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                    // actual_seq_len 19
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                                  // k_nope_clip_alpha 20
        },
        {
            {{{8, 128, 512}, {8, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // query
            {{{8, 128, 64}, {8, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},       // query_rope
            {{{1, 16, 1, 656}, {1, 16, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND}, // kv_cache
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                         // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                        // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                         // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                        // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0022971812167027)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00235037235057241)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend910_B3", MlaPrologV3_tiling_A2SocInfo, 4096);
    int64_t expectTilingKey = 3933601;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 全量化pertile场景异常case：The quantScaleRepoMode expected 1, but got 0.
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test38)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_INT8, ge::FORMAT_ND},                       // token_x 0
            {{{7168, 1536}, {7168, 1536}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 24576}, {1536, 24576}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},       // weight_uq_qr 2
            {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},           // weight_uk 3
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                             // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                               // rmsnorm_gamma_ckv 6
            {{{8, 64}, {8, 64}}, ge::DT_BF16, ge::FORMAT_ND},                           // rope_sin 7
            {{{8, 64}, {8, 64}}, ge::DT_BF16, ge::FORMAT_ND},                           // rope_cos 8
            {{{1, 16, 1, 656}, {1, 16, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},           // kv_cache 9
            {{{3, 4, 3}, {3, 4, 3}}, ge::DT_INT8, ge::FORMAT_ND},                       // kr_cache 10
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND},                                  // cache_index 11
            {{{8, 1}, {8, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // dequant_scale_x 12
            {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},                      // dequant_scale_w_dq 13
            {{{1, 24576}, {1, 24576}}, ge::DT_FLOAT, ge::FORMAT_ND},                    // dequant_scale_w_uq_qr 14
            {{{1, 576}, {1, 576}}, ge::DT_FLOAT, ge::FORMAT_ND},                        // dequant_scale_w_dkv_kr 15
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                    // actual_seq_len 19
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                                  // k_nope_clip_alpha 20
        },
        {
            {{{8, 128, 512}, {8, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // query
            {{{8, 128, 64}, {8, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},       // query_rope
            {{{1, 16, 1, 656}, {1, 16, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND}, // kv_cache
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                         // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                        // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                         // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                        // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0022971812167027)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00235037235057241)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend910_B3", MlaPrologV3_tiling_A2SocInfo, 4096);
    int64_t expectTilingKey = 3933601;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 全量化pertile场景异常case：The queryQuantMode expected 0, but got 1.
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test39)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_INT8, ge::FORMAT_ND},                       // token_x 0
            {{{7168, 1536}, {7168, 1536}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 24576}, {1536, 24576}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},       // weight_uq_qr 2
            {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},           // weight_uk 3
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                             // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                               // rmsnorm_gamma_ckv 6
            {{{8, 64}, {8, 64}}, ge::DT_BF16, ge::FORMAT_ND},                           // rope_sin 7
            {{{8, 64}, {8, 64}}, ge::DT_BF16, ge::FORMAT_ND},                           // rope_cos 8
            {{{1, 16, 1, 656}, {1, 16, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},           // kv_cache 9
            {{{3, 4, 3}, {3, 4, 3}}, ge::DT_INT8, ge::FORMAT_ND},                       // kr_cache 10
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND},                                  // cache_index 11
            {{{8, 1}, {8, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // dequant_scale_x 12
            {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},                      // dequant_scale_w_dq 13
            {{{1, 24576}, {1, 24576}}, ge::DT_FLOAT, ge::FORMAT_ND},                    // dequant_scale_w_uq_qr 14
            {{{1, 576}, {1, 576}}, ge::DT_FLOAT, ge::FORMAT_ND},                        // dequant_scale_w_dkv_kr 15
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                    // actual_seq_len 19
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                                  // k_nope_clip_alpha 20
        },
        {
            {{{8, 128, 512}, {8, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // query
            {{{8, 128, 64}, {8, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},       // query_rope
            {{{1, 16, 1, 656}, {1, 16, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND}, // kv_cache
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                         // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                        // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                         // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                        // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0022971812167027)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00235037235057241)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend910_B3", MlaPrologV3_tiling_A2SocInfo, 4096);
    int64_t expectTilingKey = 3933601;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 全量化pertile场景异常case：Not support both cacheMode {PA_NZ, PA_BLK_BSND, PA_BLK_NZ} and pertile effective.
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test40)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_INT8, ge::FORMAT_ND},                       // token_x 0
            {{{7168, 1536}, {7168, 1536}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 24576}, {1536, 24576}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},       // weight_uq_qr 2
            {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},           // weight_uk 3
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                             // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                               // rmsnorm_gamma_ckv 6
            {{{8, 64}, {8, 64}}, ge::DT_BF16, ge::FORMAT_ND},                           // rope_sin 7
            {{{8, 64}, {8, 64}}, ge::DT_BF16, ge::FORMAT_ND},                           // rope_cos 8
            {{{1, 16, 1, 656}, {1, 16, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},           // kv_cache 9
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                                   // kr_cache 10
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND},                                  // cache_index 11
            {{{8, 1}, {8, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // dequant_scale_x 12
            {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},                      // dequant_scale_w_dq 13
            {{{1, 24576}, {1, 24576}}, ge::DT_FLOAT, ge::FORMAT_ND},                    // dequant_scale_w_uq_qr 14
            {{{1, 576}, {1, 576}}, ge::DT_FLOAT, ge::FORMAT_ND},                        // dequant_scale_w_dkv_kr 15
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                    // actual_seq_len 19
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                                  // k_nope_clip_alpha 20
        },
        {
            {{{8, 128, 512}, {8, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // query
            {{{8, 128, 64}, {8, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},       // query_rope
            {{{1, 16, 1, 656}, {1, 16, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND}, // kv_cache
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                         // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                        // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                         // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                        // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0022971812167027)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00235037235057241)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_NZ")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend910_B3", MlaPrologV3_tiling_A2SocInfo, 4096);
    int64_t expectTilingKey = 3933601;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 全量化pertile场景异常case：weightQuantMode must be within {0, 3}, actually is 10.
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test41)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_INT8, ge::FORMAT_ND},                       // token_x 0
            {{{7168, 1536}, {7168, 1536}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 24576}, {1536, 24576}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},       // weight_uq_qr 2
            {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},           // weight_uk 3
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                             // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                               // rmsnorm_gamma_ckv 6
            {{{8, 64}, {8, 64}}, ge::DT_BF16, ge::FORMAT_ND},                           // rope_sin 7
            {{{8, 64}, {8, 64}}, ge::DT_BF16, ge::FORMAT_ND},                           // rope_cos 8
            {{{1, 16, 1, 656}, {1, 16, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},           // kv_cache 9
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                                   // kr_cache 10
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND},                                  // cache_index 11
            {{{8, 1}, {8, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // dequant_scale_x 12
            {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},                      // dequant_scale_w_dq 13
            {{{1, 24576}, {1, 24576}}, ge::DT_FLOAT, ge::FORMAT_ND},                    // dequant_scale_w_uq_qr 14
            {{{1, 576}, {1, 576}}, ge::DT_FLOAT, ge::FORMAT_ND},                        // dequant_scale_w_dkv_kr 15
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                    // actual_seq_len 19
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                                  // k_nope_clip_alpha 20
        },
        {
            {{{8, 128, 512}, {8, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // query
            {{{8, 128, 64}, {8, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},       // query_rope
            {{{1, 16, 1, 656}, {1, 16, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND}, // kv_cache
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                         // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                        // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                         // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                        // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0022971812167027)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00235037235057241)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(10)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend910_B3", MlaPrologV3_tiling_A2SocInfo, 4096);
    int64_t expectTilingKey = 3933601;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 全量化pertile场景异常case：krCache[1] is not an empty tensor
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test42)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_INT8, ge::FORMAT_ND},                       // token_x 0
            {{{7168, 1536}, {7168, 1536}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 24576}, {1536, 24576}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},       // weight_uq_qr 2
            {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},           // weight_uk 3
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                             // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                               // rmsnorm_gamma_ckv 6
            {{{8, 64}, {8, 64}}, ge::DT_BF16, ge::FORMAT_ND},                           // rope_sin 7
            {{{8, 64}, {8, 64}}, ge::DT_BF16, ge::FORMAT_ND},                           // rope_cos 8
            {{{1, 16, 1, 656}, {1, 16, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},           // kv_cache 9
            {{{1}, {1}}, ge::DT_INT8, ge::FORMAT_ND},                                   // kr_cache 10
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND},                                  // cache_index 11
            {{{8, 1}, {8, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // dequant_scale_x 12
            {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},                      // dequant_scale_w_dq 13
            {{{1, 24576}, {1, 24576}}, ge::DT_FLOAT, ge::FORMAT_ND},                    // dequant_scale_w_uq_qr 14
            {{{1, 576}, {1, 576}}, ge::DT_FLOAT, ge::FORMAT_ND},                        // dequant_scale_w_dkv_kr 15
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                    // actual_seq_len 19
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                                  // k_nope_clip_alpha 20
        },
        {
            {{{8, 128, 512}, {8, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // query
            {{{8, 128, 64}, {8, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},       // query_rope
            {{{1, 16, 1, 656}, {1, 16, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND}, // kv_cache
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                         // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                        // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                         // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                        // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0022971812167027)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00235037235057241)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend910_B3", MlaPrologV3_tiling_A2SocInfo, 4096);
    int64_t expectTilingKey = 3933601;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}
// 950的MXFP8_FULL_QUANT_KV_NO_QUANT场景正常case
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test43)
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
            {{{1, 16, 1, 512}, {1, 16, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},                    // kv_cache 9
            {{{1, 16, 1, 64}, {1, 16, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // kr_cache 10
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
            {{{8, 1, 128, 512}, {8, 1, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // query
            {{{8, 1, 128, 64}, {8, 1, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},   // query_rope
            {{{1, 16, 1, 512}, {1, 16, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},   // kv_cache
            {{{1, 16, 1, 64}, {1, 16, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},     // kr_cache
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

// 950的MXFP8_FULL_QUANT_KV_NO_QUANT场景异常case：tokenx expected dtype DT_BFLOAT8_E4M3FN, but got DT_BF16
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test44)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 1, 7168}, {8, 1, 7168}}, ge::DT_BF16, ge::FORMAT_ND},                          // token_x 0
            {{{7168, 1536}, {7168, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 24576}, {1536, 24576}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},       // weight_uq_qr 2
            {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                    // weight_uk 3
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                                      // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                        // rmsnorm_gamma_ckv 6
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                              // rope_sin 7
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                              // rope_cos 8
            {{{1, 16, 1, 512}, {1, 16, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},                    // kv_cache 9
            {{{1, 16, 1, 64}, {1, 16, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // kr_cache 10
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
            {{{8, 1, 128, 512}, {8, 1, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // query
            {{{8, 1, 128, 64}, {8, 1, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},   // query_rope
            {{{1, 16, 1, 512}, {1, 16, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},   // kv_cache
            {{{1, 16, 1, 64}, {1, 16, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},     // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                          // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                           // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                          // dequant_scale_q_norm
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
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 950的MXFP8_FULL_QUANT_KV_NO_QUANT场景异常case：TokenX shape dim num allows only 3, got 2.
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test45)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                       // token_x 0
            {{{7168, 1536}, {7168, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 24576}, {1536, 24576}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},       // weight_uq_qr 2
            {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                    // weight_uk 3
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                                      // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                        // rmsnorm_gamma_ckv 6
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                              // rope_sin 7
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                              // rope_cos 8
            {{{1, 16, 1, 512}, {1, 16, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},                    // kv_cache 9
            {{{1, 16, 1, 64}, {1, 16, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // kr_cache 10
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
            {{{8, 1, 128, 512}, {8, 1, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // query
            {{{8, 1, 128, 64}, {8, 1, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},   // query_rope
            {{{1, 16, 1, 512}, {1, 16, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},   // kv_cache
            {{{1, 16, 1, 64}, {1, 16, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},     // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                          // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                           // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                          // dequant_scale_q_norm
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
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 950的MXFP8_FULL_QUANT_KV_NO_QUANT场景异常case：He allows only 7168, got 999.
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test46)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 1, 999}, {8, 1, 999}}, ge::DT_BF16, ge::FORMAT_ND},                                 // token_x
            {{{999, 1536}, {999, 1536}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},                         // weight_dq
            {{{1536, 32 * (128 + 64)}, {1536, 32 * (128 + 64)}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ}, // weight_uq_qr
            {{{32, 128, 512}, {32, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                           // weight_uk
            {{{999, 512 + 64}, {999, 512 + 64}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},                 // weight_dkv_kr
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                              // rmsnorm_gamma_cq
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                // rmsnorm_gamma_ckv
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // rope_sin
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // rope_cos
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},        // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},          // kr_cache
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
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},   // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                             // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                             // dequant_scale_q_norm
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
    int64_t expectTilingKey = 3933345;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 950的MXFP8_FULL_QUANT_KV_NO_QUANT场景异常case：tokenx datatype only supports [DT_BFLOAT16, DT_BFLOAT8_E4M3FN ], but
// got DT_INT8.
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test48)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 1, 7168}, {8, 1, 7168}}, ge::DT_INT8, ge::FORMAT_ND},                          // token_x 0
            {{{7168, 1536}, {7168, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 24576}, {1536, 24576}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},       // weight_uq_qr 2
            {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                    // weight_uk 3
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                                      // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                        // rmsnorm_gamma_ckv 6
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                              // rope_sin 7
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                              // rope_cos 8
            {{{1, 16, 1, 512}, {1, 16, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},                    // kv_cache 9
            {{{1, 16, 1, 64}, {1, 16, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // kr_cache 10
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
            {{{8, 1, 128, 512}, {8, 1, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // query
            {{{8, 1, 128, 64}, {8, 1, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},   // query_rope
            {{{1, 16, 1, 512}, {1, 16, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},   // kv_cache
            {{{1, 16, 1, 64}, {1, 16, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},     // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                          // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                           // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                          // dequant_scale_q_norm
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
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 950的MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR场景正常case，cache_mode：PA_BSND
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test49)
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

// 950的MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR场景正常case，cache_mode：PA_NZ
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test50)
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

// 950的MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR场景异常case：Only support cacheMode {PA_BSND, PA_NZ}, actually is BSND
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test51)
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
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                                    // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // dequant_scale_q_norm
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
    int64_t expectTilingKey = 3933729;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// 950的MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR场景异常case，The queryQuantMode expected 1, but got 0.
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test52)
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
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                                    // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0022971812167027)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00235037235057241)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
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

// 950的MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR场景异常case，When weightQuantMode == 0, kvQuantMode must be within {0},
// actually is 1.
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test53)
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
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                                    // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0022971812167027)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00235037235057241)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
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

// 950的MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR场景异常case，When weightQuantMode == 3, kvQuantMode must be within {0, 1},
// actually is 2.
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test54)
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
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                                    // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0022971812167027)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00235037235057241)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
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

// 950的MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR场景异常case，weightQuantMode must be within {0, 3}, actually is 2.
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test55)
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
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                                    // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0022971812167027)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00235037235057241)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
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

// 950的MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR场景异常case，tokenX dim num supports only [2, 3], but got 5.
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test56)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{1, 1, 1, 33, 7168}, {1, 1, 1, 33, 7168}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},   // token_x 0
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
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                                        // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                                       // dequant_scale_q_norm
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

// 950的MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR场景异常case，ropeSin expected shape [33, 64], but got [1, 33, 64].
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test57)
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
            {{{1, 33, 64}, {1, 33, 64}}, ge::DT_BF16, ge::FORMAT_ND},                            // rope_sin 7
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
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                                        // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                                       // dequant_scale_q_norm
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

// 950的MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR场景异常case，ropeCos dim num supports only [2, 3], but got 4.
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test58)
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
            {{{1, 1, 33, 64}, {1, 1, 33, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // rope_cos 8
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
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                                        // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                                       // dequant_scale_q_norm
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

// 950的MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR场景异常case，cacheIndex expected shape [33], but got [1, 33].
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test59)
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
            {{{1, 33}, {1, 33}}, ge::DT_INT64, ge::FORMAT_ND},                                   // cache_index 11
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
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                                        // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                                       // dequant_scale_q_norm
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

// 950的MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR场景异常case，dequantScaleX expected shape [33,224], but got [1, 33, 224].
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test60)
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
            {{{1, 33, 224}, {1, 33, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                   // dequant_scale_x 12
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
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                                        // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                                       // dequant_scale_q_norm
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

// 950的MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR场景异常case，query输入维度为5
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test61)
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
            {{{1, 1, 33, 8, 512}, {1, 1, 33, 8, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // query
            {{{33, 8, 64}, {33, 8, 64}}, ge::DT_BF16, ge::FORMAT_ND},                        // query_rope
            {{{254, 128, 1, 512}, {254, 128, 1, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // kv_cache
            {{{254, 128, 1, 64}, {254, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},            // kr_cache
            {{{33, 8, 1}, {33, 8, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                                        // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                                       // dequant_scale_q_norm
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

// 950的MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR场景异常case，query_rope输入维度为5
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test62)
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
            {{{1, 1, 33, 8, 64}, {1, 1, 33, 8, 64}}, ge::DT_BF16, ge::FORMAT_ND},            // query_rope
            {{{254, 128, 1, 512}, {254, 128, 1, 512}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // kv_cache
            {{{254, 128, 1, 64}, {254, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},            // kr_cache
            {{{33, 8, 1}, {33, 8, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                                        // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                                       // dequant_scale_q_norm
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

// 950的MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR场景异常case，dequant_scale_q_nope输入维度为4
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test63)
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
            {{{1, 33, 8, 1}, {1, 33, 8, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                                        // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                                       // dequant_scale_q_norm
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

// 950的NO_QUANT场景正常case，BS合轴; cacheMode = PA_BSND; T = 2
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test64)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{2, 7168}, {2, 7168}}, ge::DT_BF16, ge::FORMAT_ND},                       // token_x 0
            {{{7168, 1536}, {7168, 1536}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 1536}, {1536, 1536}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},         // weight_uq_qr 2
            {{{8, 128, 512}, {8, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},               // weight_uk 3
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                             // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                               // rmsnorm_gamma_ckv 6
            {{{2, 64}, {2, 64}}, ge::DT_BF16, ge::FORMAT_ND},                           // rope_sin 7
            {{{2, 64}, {2, 64}}, ge::DT_BF16, ge::FORMAT_ND},                           // rope_cos 8
            {{{2, 128, 1, 512}, {2, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},         // kv_cache 9
            {{{2, 128, 1, 64}, {2, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},           // kr_cache 10
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                                  // cache_index 11
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // dequant_scale_x 12
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // dequant_scale_w_dq 13
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // dequant_scale_w_uq_qr 14
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // dequant_scale_w_dkv_kr 15
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                    // actual_seq_len 19
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // k_nope_clip_alpha 20
        },
        {
            {{{2, 8, 512}, {2, 8, 512}}, ge::DT_BF16, ge::FORMAT_ND},           // query
            {{{2, 8, 64}, {2, 8, 64}}, ge::DT_BF16, ge::FORMAT_ND},             // query_rope
            {{{2, 128, 1, 512}, {2, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // kv_cache
            {{{2, 128, 1, 64}, {2, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},   // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                          // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_BF16, ge::FORMAT_ND},                           // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                          // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.36856476641669955)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.4911813887143225)},
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

// 950的NO_QUANT场景正常case，BS合轴; cacheMode = PA_NZ; T = 128
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test65)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{128, 7168}, {128, 7168}}, ge::DT_BF16, ge::FORMAT_ND},                   // token_x 0
            {{{7168, 1536}, {7168, 1536}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 192}, {1536, 192}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},           // weight_uq_qr 2
            {{{1, 128, 512}, {1, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},               // weight_uk 3
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                             // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                               // rmsnorm_gamma_ckv 6
            {{{128, 64}, {128, 64}}, ge::DT_BF16, ge::FORMAT_ND},                       // rope_sin 7
            {{{128, 64}, {128, 64}}, ge::DT_BF16, ge::FORMAT_ND},                       // rope_cos 8
            {{{62, 128, 1, 512}, {62, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},       // kv_cache 9
            {{{62, 128, 1, 64}, {62, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},         // kr_cache 10
            {{{128}, {128}}, ge::DT_INT64, ge::FORMAT_ND},                              // cache_index 11
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // dequant_scale_x 12
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // dequant_scale_w_dq 13
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // dequant_scale_w_uq_qr 14
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // dequant_scale_w_dkv_kr 15
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                    // actual_seq_len 19
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // k_nope_clip_alpha 20
        },
        {
            {{{128, 1, 512}, {128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},         // query
            {{{128, 1, 64}, {128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},           // query_rope
            {{{62, 128, 1, 512}, {62, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // kv_cache
            {{{62, 128, 1, 64}, {62, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},   // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_BF16, ge::FORMAT_ND},                             // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.9623901933410451)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.9581271563003967)},
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

// 950的MXFP8_FULL_QUANT_KV_NO_QUANT场景正常case，BS合轴; cacheMode = PA_NZ; T = 2
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test66)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{2, 7168}, {2, 7168}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                       // token_x 0
            {{{7168, 1536}, {7168, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 1536}, {1536, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},         // weight_uq_qr 2
            {{{8, 128, 512}, {8, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                        // weight_uk 3
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                                      // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                        // rmsnorm_gamma_ckv 6
            {{{2, 64}, {2, 64}}, ge::DT_BF16, ge::FORMAT_ND},                                    // rope_sin 7
            {{{2, 64}, {2, 64}}, ge::DT_BF16, ge::FORMAT_ND},                                    // rope_cos 8
            {{{62, 128, 1, 512}, {62, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},                // kv_cache 9
            {{{62, 128, 1, 64}, {62, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                  // kr_cache 10
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                                           // cache_index 11
            {{{2, 224}, {2, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                           // dequant_scale_x 12
            {{{1536, 224}, {1536, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // dequant_scale_w_dq 13
            {{{1536, 48}, {1536, 48}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},   // dequant_scale_w_uq_qr 14
            {{{576, 224}, {576, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},   // dequant_scale_w_dkv_kr 15
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // actual_seq_len 19
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // k_nope_clip_alpha 20
        },
        {
            {{{2, 8, 512}, {2, 8, 512}}, ge::DT_BF16, ge::FORMAT_ND},             // query
            {{{2, 8, 64}, {2, 8, 64}}, ge::DT_BF16, ge::FORMAT_ND},               // query_rope
            {{{62, 128, 1, 512}, {62, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // kv_cache
            {{{62, 128, 1, 64}, {62, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},   // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                    // query_norm
            {{{0}, {0}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                      // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.7397180646457813)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.6560849016287276)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_NZ")},
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
    int64_t expectTilingKey = 3933666;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 950的MXFP8_FULL_QUANT_KV_NO_QUANT场景正常case，BS合轴; cacheMode = PA_BSND; T = 31
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test67)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{31, 7168}, {31, 7168}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                     // token_x 0
            {{{7168, 1536}, {7168, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 1536}, {1536, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},         // weight_uq_qr 2
            {{{8, 128, 512}, {8, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                        // weight_uk 3
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                                      // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                        // rmsnorm_gamma_ckv 6
            {{{31, 64}, {31, 64}}, ge::DT_BF16, ge::FORMAT_ND},                                  // rope_sin 7
            {{{31, 64}, {31, 64}}, ge::DT_BF16, ge::FORMAT_ND},                                  // rope_cos 8
            {{{126, 128, 1, 512}, {126, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},              // kv_cache 9
            {{{126, 128, 1, 64}, {126, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                // kr_cache 10
            {{{31}, {31}}, ge::DT_INT64, ge::FORMAT_ND},                                         // cache_index 11
            {{{31, 224}, {31, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                         // dequant_scale_x 12
            {{{1536, 224}, {1536, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // dequant_scale_w_dq 13
            {{{1536, 48}, {1536, 48}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},   // dequant_scale_w_uq_qr 14
            {{{576, 224}, {576, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},   // dequant_scale_w_dkv_kr 15
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // actual_seq_len 19
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // k_nope_clip_alpha 20
        },
        {
            {{{31, 8, 512}, {31, 8, 512}}, ge::DT_BF16, ge::FORMAT_ND},             // query
            {{{31, 8, 64}, {31, 8, 64}}, ge::DT_BF16, ge::FORMAT_ND},               // query_rope
            {{{126, 128, 1, 512}, {126, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // kv_cache
            {{{126, 128, 1, 64}, {126, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},   // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                      // query_norm
            {{{0}, {0}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                        // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.860064509088248)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.4429179519204812)},
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

// ===================== RoPE 关闭用例 =====================
TEST_F(MlaPrologV3, MlaPrologV3_tiling_test_do_rope_false_a2)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 1, 7168}, {8, 1, 7168}}, ge::DT_INT8, ge::FORMAT_ND},                               // token_x
            {{{7168, 1536}, {7168, 1536}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},                       // weight_dq
            {{{1536, 32 * (128 + 64)}, {1536, 32 * (128 + 64)}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // weight_uq_qr
            {{{32, 128, 512}, {32, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                           // weight_uk
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},               // weight_dkv_kr
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                       // rmsnorm_gamma_cq
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                         // rmsnorm_gamma_ckv
            {{{0}, {0}}, ge::DT_BF16, ge::FORMAT_ND},                             // rope_sin empty (do_rope=false)
            {{{0}, {0}}, ge::DT_BF16, ge::FORMAT_ND},                             // rope_cos empty (do_rope=false)
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND}, // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},   // kr_cache
            {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},                      // cache_index
            {{{8 * 1, 1}, {8 * 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},              // dequant_scale_x
            {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},                // dequant_scale_w_dq
            {{{1, 32 * (128 + 64)}, {1, 32 * (128 + 64)}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dequant_scale_w_uq_qr
            {{{1, 512 + 64}, {1, 512 + 64}}, ge::DT_FLOAT, ge::FORMAT_ND},               // dequant_scale_w_dkv_kr
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // quant_scale_ckv
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // quant_scale_ckr
            {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // smooth_scales_cq
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                     // actual_seq_len
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // k_nope_clip_alpha
        },
        {
            {{{8, 1, 32, 512}, {8, 1, 32, 512}}, ge::DT_INT8, ge::FORMAT_ND},     // query
            {{{8, 1, 32, 64}, {8, 1, 32, 64}}, ge::DT_BF16, ge::FORMAT_ND},       // query_rope
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND}, // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},   // kr_cache
            {{{8 * 1, 32, 1}, {8 * 1, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},      // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                             // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                             // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"do_rope", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend910_B3", MlaPrologV3_tiling_A2SocInfo, 4096);
    int64_t expectTilingKey = 3671329;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MlaPrologV3, MlaPrologV3_tiling_test_do_rope_false)
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
            {{{0}, {0}}, ge::DT_BF16, ge::FORMAT_ND},                         // rope_sin 7 empty (do_rope=false)
            {{{0}, {0}}, ge::DT_BF16, ge::FORMAT_ND},                         // rope_cos 8 empty (do_rope=false)
            {{{1, 16, 1, 512}, {1, 16, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // kv_cache 9
            {{{1, 16, 1, 64}, {1, 16, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},   // kr_cache 10
            {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},                  // cache_index 11
            {{{8, 224}, {8, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},        // dequant_scale_x 12
            {{{1536, 224}, {1536, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},  // dequant_scale_w_dq 13
            {{{24576, 48}, {24576, 48}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},  // dequant_scale_w_uq_qr 14
            {{{576, 224}, {576, 224}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},    // dequant_scale_w_dkv_kr 15
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                          // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                          // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                          // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                          // actual_seq_len 19
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                          // k_nope_clip_alpha 20
        },
        {
            {{{8, 1, 128, 512}, {8, 1, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // query
            {{{8, 1, 128, 64}, {8, 1, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},   // query_rope
            {{{1, 16, 1, 512}, {1, 16, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},   // kv_cache
            {{{1, 16, 1, 64}, {1, 16, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},     // kr_cache
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
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"do_rope", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
    int64_t expectTilingKey = 3671521;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MlaPrologV3, MlaPrologV3_tiling_test_do_rope_false_combine_kr_empty)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV3",
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_INT8, ge::FORMAT_ND},                       // token_x 0
            {{{7168, 1536}, {7168, 1536}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},         // weight_dq 1
            {{{1536, 24576}, {1536, 24576}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},       // weight_uq_qr 2
            {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},           // weight_uk 3
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                             // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                               // rmsnorm_gamma_ckv 6
            {{{0}, {0}}, ge::DT_BF16, ge::FORMAT_ND},                         // rope_sin 7 empty (do_rope=false)
            {{{0}, {0}}, ge::DT_BF16, ge::FORMAT_ND},                         // rope_cos 8 empty (do_rope=false)
            {{{1, 16, 1, 656}, {1, 16, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND}, // kv_cache 9
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                         // kr_cache 10 empty
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND},                        // cache_index 11
            {{{8, 1}, {8, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},                  // dequant_scale_x 12
            {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},            // dequant_scale_w_dq 13
            {{{1, 24576}, {1, 24576}}, ge::DT_FLOAT, ge::FORMAT_ND},          // dequant_scale_w_uq_qr 14
            {{{1, 576}, {1, 576}}, ge::DT_FLOAT, ge::FORMAT_ND},              // dequant_scale_w_dkv_kr 15
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                          // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                          // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                          // smooth_scales_cq 18
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                          // actual_seq_len 19
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                        // k_nope_clip_alpha 20
        },
        {
            {{{8, 128, 512}, {8, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // query
            {{{8, 128, 64}, {8, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},       // query_rope
            {{{1, 16, 1, 656}, {1, 16, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND}, // kv_cache
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                         // kr_cache
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                        // dequant_scale_q_nope
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                         // query_norm
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                        // dequant_scale_q_norm
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0022971812167027)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00235037235057241)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"do_rope", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend910_B3", MlaPrologV3_tiling_A2SocInfo, 4096);
    int64_t expectTilingKey = 3671457;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}
