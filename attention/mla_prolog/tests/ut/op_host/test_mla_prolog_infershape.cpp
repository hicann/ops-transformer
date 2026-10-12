/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infer_shape_context_faker.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class MlaPrologProto : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MlaPrologProto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MlaPrologProto TearDown" << std::endl;
    }
};

TEST_F(MlaPrologProto, mla_prolog_infershape_1)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MlaProlog",
        {
            {{{48, 10, 7168}, {48, 10, 7168}}, ge::DT_BF16, ge::FORMAT_ND},           // token_x
            {{{7168, 1536}, {7168, 1536}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},       // weight_dq
            {{{1536, 768}, {1536, 768}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},         // weight_uq_qr
            {{{4, 128, 512}, {4, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},             // weight_uk
            {{{7168, 576}, {7168, 576}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},         // weight_dkv_kr
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                           // rmsnorm_gamma_cq
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                             // rmsnorm_gamma_ckv
            {{{48, 10, 64}, {48, 10, 64}}, ge::DT_BF16, ge::FORMAT_ND},               // rope_sin
            {{{48, 10, 64}, {48, 10, 64}}, ge::DT_BF16, ge::FORMAT_ND},               // rope_cos
            {{{49488, 16, 1, 512}, {49488, 16, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND}, // kv_cache
            {{{49488, 16, 1, 64}, {49488, 16, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},   // kr_cache
            {{{48, 10}, {48, 10}}, ge::DT_INT64, ge::FORMAT_ND},                      // cache_index
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                  // dequant_scale_x
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                  // dequant_scale_w_dq
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                  // dequant_scale_w_uq_qr
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                  // dequant_scale_w_dkv_kr
            {{{1, 512}, {1, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},                      // quant_scale_ckv
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                  // quant_scale_ckr
            {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},                    // smooth_scales_cq
        },
        {
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},  // query
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},  // query_rope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // kv_cache
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // kr_cache
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dequant_scale_q_nope
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {48, 10, 4, 512}, {48, 10, 4, 64}, {49488, 16, 1, 64}, {48, 10}, {}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// TEST(用例集名, 用例名)
TEST_F(MlaPrologProto, mla_prolog_infershape_2)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MlaProlog",
        {
            // input info
            {{{8, 1, 7168}, {8, 1, 7168}}, ge::DT_BF16, ge::FORMAT_ND},           // token_x 0
            {{{7168, 1536}, {7168, 1536}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},   //  weight_dq 1
            {{{1536, 192}, {1536, 192}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},     // weight_uq_qr 2
            {{{1, 128, 512}, {1, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},         // weight_uk 3
            {{{7168, 576}, {7168, 576}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},     // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                       // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                         // rmsnorm_gamma_ckv 6
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},               // rope_sin 7
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},               // rope_cos 8
            {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},                      // cache_index 9
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // kv_cache 10
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},   // kr_cache 11
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // dequant_scale_x 12
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // dequant_scale_w_dq 13
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // dequant_scale_w_uq_qr 14
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // dequant_scale_w_dkv_kr 15
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // smooth_scales_cq 18
        },
        {
            // output info
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}, // 0
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}, // 1
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}, // 2
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}, // 3
        },
        {// attr
         {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0005)},
         {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.0005)},
         {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")}});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {8, 1, 1, 512}, {8, 1, 1, 64}, {16, 128, 1, 512}, {16, 128, 1, 64}};      // 预期输出shape
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape); // 框架中已提供该接口
}

TEST_F(MlaPrologProto, mla_prolog_infershape_3)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MlaProlog",
        {
            // input info
            {{{32, 1, 7168}, {32, 1, 7168}}, ge::DT_BF16, ge::FORMAT_ND},           // token_x 0
            {{{7168, 1536}, {7168, 1536}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},     //  weight_dq 1
            {{{1536, 6144}, {1536, 6144}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},     // weight_uq_qr 2
            {{{32, 128, 512}, {32, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},         // weight_uk 3
            {{{7168, 576}, {7168, 576}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},       // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                         // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                           // rmsnorm_gamma_ckv 6
            {{{32, 1, 64}, {32, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},               // rope_sin 7
            {{{32, 1, 64}, {32, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},               // rope_cos 8
            {{{32, 1}, {32, 1}}, ge::DT_INT64, ge::FORMAT_ND},                      // cache_index 9
            {{{2048, 16, 1, 512}, {2048, 16, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // kv_cache 10
            {{{2048, 16, 1, 64}, {2048, 16, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},   // kr_cache 11
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                // dequant_scale_x 12
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                // dequant_scale_w_dq 13
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                // dequant_scale_w_uq_qr 14
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                // dequant_scale_w_dkv_kr 15
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                // smooth_scales_cq 18
        },
        {
            // output info
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}, // 0
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}, // 1
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}, // 2
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}, // 3
        },
        {// attr
         {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0005)},
         {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.0005)},
         {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")}});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {32, 1, 32, 512}, {32, 1, 32, 64}, {2048, 16, 1, 512}, {2048, 16, 1, 64}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MlaPrologProto, mla_prolog_infershape_4)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MlaProlog",
        {
            // input info
            {{{2, 1, 7680}, {2, 1, 7680}}, ge::DT_BF16, ge::FORMAT_ND},         // token_x 0
            {{{7680, 1536}, {7680, 1536}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ}, //  weight_dq 1
            {{{1536, 384}, {1536, 384}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},   // weight_uq_qr 2
            {{{2, 128, 512}, {2, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},       // weight_uk 3
            {{{7680, 576}, {7680, 576}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},   // weight_dkv_kr 4
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                     // rmsnorm_gamma_cq 5
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                       // rmsnorm_gamma_ckv 6
            {{{2, 1, 64}, {2, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},             // rope_sin 7
            {{{2, 1, 64}, {2, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},             // rope_cos 8
            {{{2, 1}, {2, 1}}, ge::DT_INT64, ge::FORMAT_ND},                    // cache_index 9
            {{{0, 16, 1, 512}, {0, 16, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},   // kv_cache 10
            {{{0, 16, 1, 64}, {0, 16, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},     // kr_cache 11
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // dequant_scale_x 12
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // dequant_scale_w_dq 13
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // dequant_scale_w_uq_qr 14
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // dequant_scale_w_dkv_kr 15
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // quant_scale_ckv 16
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // quant_scale_ckr 17
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // smooth_scales_cq 18
        },
        {
            // output info
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}, // 0
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}, // 1
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}, // 2
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}, // 3
        },
        {// attr
         {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0005)},
         {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.0005)},
         {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")}});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 1, 2, 512}, {2, 1, 2, 64}, {0, 16, 1, 512}, {0, 16, 1, 64}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// infer dataType
TEST_F(MlaPrologProto, dtype_infer_1)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto data_type_func = spaceRegistry->GetOpImpl("MlaProlog")->infer_datatype;
    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_BF16;
        ge::DataType input_ref1 = ge::DT_INT64;
        ge::DataType input_ref2 = ge::DT_FLOAT;
        ge::DataType output_ref = ge::DT_BF16;
        auto context_holder =
            gert::InferDataTypeContextFaker()
                .NodeIoNum(19, 4)
                .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                .NodeInputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(4, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                .NodeInputTd(5, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(6, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(7, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(8, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(9, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(10, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(11, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(13, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(14, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(15, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(16, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(17, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(18, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(3, ge::FORMAT_ND, ge::FORMAT_ND)
                .InputDataTypes({&input_ref, &input_ref, &input_ref, &input_ref, &input_ref, &input_ref, &input_ref,
                                 &input_ref, &input_ref, &input_ref1, &input_ref, &input_ref, &input_ref, &input_ref,
                                 &input_ref, &input_ref, &input_ref, &input_ref, &input_ref})
                .OutputDataTypes({&output_ref, &output_ref, &output_ref, &output_ref})
                .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
        EXPECT_EQ(context->GetOutputDataType(1), output_ref);
        EXPECT_EQ(context->GetOutputDataType(2), output_ref);
        EXPECT_EQ(context->GetOutputDataType(3), output_ref);
    }
}
