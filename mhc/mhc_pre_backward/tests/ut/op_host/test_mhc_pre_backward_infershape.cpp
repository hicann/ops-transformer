/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class MhcPreBackwardProto : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MhcPreBackwardInferShape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MhcPreBackwardInferShape TearDown" << std::endl;
    }
};

/*
 * 网络用例1：B=2, S=4096, n=4，D=1536, x数据类型bf16
 * alpha [0.5 0.5 0.5]，hc_eps 200
 * 预期结果：成功
 */
TEST_F(MhcPreBackwardProto, Ut_Check_Case01_B2_S4096_n4_D1536_BF16)
{
    float alpha[3] = {0.15f, 0.15f, 0.15f}; // alpha = [0.15, 0.15, 0.15]
    uint32_t B = 2;
    uint32_t S = 4096;
    uint32_t n = 4;
    uint32_t D = 1536;
    uint32_t nD = n * D;                 // 6144
    uint32_t fusionSize = n * n + 2 * n; // 24
    float hcEps = 0.000001f;

    gert::InfershapeContextPara infershapeContextPara(
        "MhcPreBackward",
        {
            {{{B, S, n, D}, {B, S, n, D}}, ge::DT_BF16, ge::FORMAT_ND},              // x
            {{{fusionSize, nD}, {fusionSize, nD}}, ge::DT_FLOAT, ge::FORMAT_ND},     // phi
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // alpha
            {{{B, S, D}, {B, S, D}}, ge::DT_BF16, ge::FORMAT_ND},                    // gradHIn
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // gradHPost
            {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},             // gradHRes
            {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // invRms
            {{{B, S, fusionSize}, {B, S, fusionSize}}, ge::DT_FLOAT, ge::FORMAT_ND}, // hMix
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // hPre
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // hPost
            {{{nD}, {nD}}, ge::DT_FLOAT, ge::FORMAT_ND}                              // gamma (optional)
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},  // gradX
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // gradPhi
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // gradAlpha
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // gradBias
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}  // gradGamma (optional)
        },
        {{"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {B, S, n, D},     // gradX: [B, S, n, D]
        {fusionSize, nD}, // gradPhi: [nD, fusionSize]
        {3},              // gradAlpha: [3]
        {fusionSize},     // gradBias: [fusionSize]
        {n, D}            // gradGamma: [n, D]
    };

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

/*
 * 网络用例2：B=2, S=4096, n=4，D=2048, x数据类型fp16
 * alpha [0.5 0.5 0.5]，hc_eps 200
 * 预期结果：成功
 */
TEST_F(MhcPreBackwardProto, Ut_Check_Case02_B2_S4096_n4_D2048_FP16)
{
    float alpha[3] = {0.35f, 0.35f, 0.35f}; // alpha = [0.35, 0.35, 0.35]
    uint32_t B = 2;
    uint32_t S = 4096;
    uint32_t n = 4;
    uint32_t D = 2048;
    uint32_t nD = n * D;                 // 8192
    uint32_t fusionSize = n * n + 2 * n; // 24
    float hcEps = 0.000001f;

    gert::InfershapeContextPara infershapeContextPara(
        "MhcPreBackward",
        {
            {{{B, S, n, D}, {B, S, n, D}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // x
            {{{fusionSize, nD}, {fusionSize, nD}}, ge::DT_FLOAT, ge::FORMAT_ND},     // phi
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // alpha
            {{{B, S, D}, {B, S, D}}, ge::DT_FLOAT16, ge::FORMAT_ND},                 // gradHIn
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // gradHPost
            {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},             // gradHRes
            {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // invRms
            {{{B, S, fusionSize}, {B, S, fusionSize}}, ge::DT_FLOAT, ge::FORMAT_ND}, // hMix
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // hPre
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // hPost
            {{{nD}, {nD}}, ge::DT_FLOAT, ge::FORMAT_ND}                              // gamma (optional)
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // gradX
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // gradPhi
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // gradAlpha
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // gradBias
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}    // gradGamma (optional)
        },
        {{"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {B, S, n, D},     // gradX: [B, S, n, D]
        {fusionSize, nD}, // gradPhi: [fusionSize, nD]
        {3},              // gradAlpha: [3]
        {fusionSize},     // gradBias: [fusionSize]
        {n, D}            // gradGamma: [n, D]
    };

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

/*
 * 网络用例3：B=2, S=4096, n=4，D=6144, x数据类型bf16
 * alpha [0.5 0.5 0.5]，hc_eps 200
 * 预期结果：成功
 */
TEST_F(MhcPreBackwardProto, Ut_Check_Case03_B2_S4096_n4_D6144_BF16)
{
    float alpha[3] = {-0.25f, -0.25f, -0.25f}; // alpha = [-0.25, -0.25, -0.25]
    uint32_t B = 2;
    uint32_t S = 4096;
    uint32_t n = 4;
    uint32_t D = 6144;
    uint32_t nD = n * D;                 // 24576
    uint32_t fusionSize = n * n + 2 * n; // 24
    float hcEps = 0.000001f;

    gert::InfershapeContextPara infershapeContextPara(
        "MhcPreBackward",
        {
            {{{B, S, n, D}, {B, S, n, D}}, ge::DT_BF16, ge::FORMAT_ND},              // x
            {{{fusionSize, nD}, {fusionSize, nD}}, ge::DT_FLOAT, ge::FORMAT_ND},     // phi
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // alpha
            {{{B, S, D}, {B, S, D}}, ge::DT_BF16, ge::FORMAT_ND},                    // gradHIn
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // gradHPost
            {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},             // gradHRes
            {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // invRms
            {{{B, S, fusionSize}, {B, S, fusionSize}}, ge::DT_FLOAT, ge::FORMAT_ND}, // hMix
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // hPre
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // hPost
            {{{nD}, {nD}}, ge::DT_FLOAT, ge::FORMAT_ND}                              // gamma (optional)
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},  // gradX
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // gradPhi
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // gradAlpha
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // gradBias
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}  // gradGamma (optional)
        },
        {{"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {B, S, n, D},     // gradX: [B, S, n, D]
        {fusionSize, nD}, // gradPhi: [fusionSize, nD]
        {3},              // gradAlpha: [3]
        {fusionSize},     // gradBias: [fusionSize]
        {n, D}            // gradGamma: [n, D]
    };

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

/*
 * 兼容性用例1：B=2, S=4096, n=4，D=1536, x数据类型bf16
 * alpha [0.6 0.6 0.6]，hc_eps 300
 * 预期结果：成功
 */
TEST_F(MhcPreBackwardProto, Ut_Check_Case04_B2_S4096_n4_D1536_BF16_Compat1)
{
    float alpha[3] = {0.45f, 0.45f, 0.45f}; // alpha = [0.45, 0.45, 0.45]
    uint32_t B = 2;
    uint32_t S = 4096;
    uint32_t n = 4;
    uint32_t D = 1536;
    uint32_t nD = n * D;                 // 6144
    uint32_t fusionSize = n * n + 2 * n; // 24
    float hcEps = 0.000001f;

    gert::InfershapeContextPara infershapeContextPara(
        "MhcPreBackward",
        {
            {{{B, S, n, D}, {B, S, n, D}}, ge::DT_BF16, ge::FORMAT_ND},              // x
            {{{fusionSize, nD}, {fusionSize, nD}}, ge::DT_FLOAT, ge::FORMAT_ND},     // phi
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // alpha
            {{{B, S, D}, {B, S, D}}, ge::DT_BF16, ge::FORMAT_ND},                    // gradHIn
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // gradHPost
            {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},             // gradHRes
            {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // invRms
            {{{B, S, fusionSize}, {B, S, fusionSize}}, ge::DT_FLOAT, ge::FORMAT_ND}, // hMix
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // hPre
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // hPost
            {{{nD}, {nD}}, ge::DT_FLOAT, ge::FORMAT_ND}                              // gamma (optional)
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},  // gradX
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // gradPhi
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // gradAlpha
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // gradBias
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}  // gradGamma (optional)
        },
        {{"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {B, S, n, D},     // gradX: [B, S, n, D]
        {fusionSize, nD}, // gradPhi: [fusionSize, nD]
        {3},              // gradAlpha: [3]
        {fusionSize},     // gradBias: [fusionSize]
        {n, D}            // gradGamma: [n, D]
    };

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

/*
 * 兼容性用例2：B=2, S=4096, n=4，D=2048, x数据类型fp16
 * alpha [0.4 0.4 0.4]，hc_eps 400
 * 预期结果：成功
 */
TEST_F(MhcPreBackwardProto, Ut_Check_Case05_B2_S4096_n4_D2048_FP16_Compat2)
{
    float alpha[3] = {0.55f, 0.55f, 0.55f}; // alpha = [0.55, 0.55, 0.55]
    uint32_t B = 2;
    uint32_t S = 4096;
    uint32_t n = 4;
    uint32_t D = 2048;
    uint32_t nD = n * D;                 // 8192
    uint32_t fusionSize = n * n + 2 * n; // 24
    float hcEps = 0.000001f;

    gert::InfershapeContextPara infershapeContextPara(
        "MhcPreBackward",
        {
            {{{B, S, n, D}, {B, S, n, D}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // x
            {{{fusionSize, nD}, {fusionSize, nD}}, ge::DT_FLOAT, ge::FORMAT_ND},     // phi
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // alpha
            {{{B, S, D}, {B, S, D}}, ge::DT_FLOAT16, ge::FORMAT_ND},                 // gradHIn
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // gradHPost
            {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},             // gradHRes
            {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // invRms
            {{{B, S, fusionSize}, {B, S, fusionSize}}, ge::DT_FLOAT, ge::FORMAT_ND}, // hMix
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // hPre
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // hPost
            {{{nD}, {nD}}, ge::DT_FLOAT, ge::FORMAT_ND}                              // gamma (optional)
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // gradX
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // gradPhi
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // gradAlpha
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // gradBias
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}    // gradGamma (optional)
        },
        {{"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {B, S, n, D},     // gradX: [B, S, n, D]
        {fusionSize, nD}, // gradPhi: [fusionSize, nD]
        {3},              // gradAlpha: [3]
        {fusionSize},     // gradBias: [fusionSize]
        {n, D}            // gradGamma: [n, D]
    };

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

/*
 * 兼容性用例3：B=2, S=4096, n=4，D=6144, x数据类型bf16
 * alpha [0.2 0.2 0.2]，hc_eps 250
 * 预期结果：成功
 */
TEST_F(MhcPreBackwardProto, Ut_Check_Case06_B2_S4096_n4_D6144_BF16_Compat3)
{
    float alpha[3] = {-0.35f, -0.35f, -0.35f}; // alpha = [-0.35, -0.35, -0.35]
    uint32_t B = 2;
    uint32_t S = 4096;
    uint32_t n = 4;
    uint32_t D = 6144;
    uint32_t nD = n * D;                 // 24576
    uint32_t fusionSize = n * n + 2 * n; // 24
    float hcEps = 0.000001f;

    gert::InfershapeContextPara infershapeContextPara(
        "MhcPreBackward",
        {
            {{{B, S, n, D}, {B, S, n, D}}, ge::DT_BF16, ge::FORMAT_ND},              // x
            {{{fusionSize, nD}, {fusionSize, nD}}, ge::DT_FLOAT, ge::FORMAT_ND},     // phi
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // alpha
            {{{B, S, D}, {B, S, D}}, ge::DT_BF16, ge::FORMAT_ND},                    // gradHIn
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // gradHPost
            {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},             // gradHRes
            {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // invRms
            {{{B, S, fusionSize}, {B, S, fusionSize}}, ge::DT_FLOAT, ge::FORMAT_ND}, // hMix
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // hPre
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // hPost
            {{{nD}, {nD}}, ge::DT_FLOAT, ge::FORMAT_ND}                              // gamma (optional)
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},  // gradX
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // gradPhi
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // gradAlpha
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // gradBias
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}  // gradGamma (optional)
        },
        {{"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {B, S, n, D},     // gradX: [B, S, n, D]
        {fusionSize, nD}, // gradPhi: [fusionSize, nD]
        {3},              // gradAlpha: [3]
        {fusionSize},     // gradBias: [fusionSize]
        {n, D}            // gradGamma: [n, D]
    };

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

/*
 * TND格式测试用例1：T=4096, n=4，D=1536, x数据类型bf16
 * alpha [0.5 0.5 0.5]，hc_eps 200
 * 预期结果：成功
 */
TEST_F(MhcPreBackwardProto, Ut_Check_Case07_TND_T4096_n4_D1536_BF16)
{
    float alpha[3] = {0.2f, 0.2f, 0.2f}; // alpha = [0.2, 0.2, 0.2]
    uint32_t T = 4096;
    uint32_t n = 4;
    uint32_t D = 1536;
    uint32_t nD = n * D;                 // 6144
    uint32_t fusionSize = n * n + 2 * n; // 24
    float hcEps = 0.000001f;

    gert::InfershapeContextPara infershapeContextPara(
        "MhcPreBackward",
        {
            {{{T, n, D}, {T, n, D}}, ge::DT_BF16, ge::FORMAT_ND},                // x (TND format)
            {{{fusionSize, nD}, {fusionSize, nD}}, ge::DT_FLOAT, ge::FORMAT_ND}, // phi
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},                           // alpha
            {{{T, D}, {T, D}}, ge::DT_BF16, ge::FORMAT_ND},                      // gradHIn (TD format)
            {{{T, n}, {T, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                     // gradHPost (TN format)
            {{{T, n, n}, {T, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},               // gradHRes (TNN format)
            {{{T}, {T}}, ge::DT_FLOAT, ge::FORMAT_ND},                           // invRms (T format)
            {{{T, fusionSize}, {T, fusionSize}}, ge::DT_FLOAT, ge::FORMAT_ND},   // hMix
            {{{T, n}, {T, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                     // hPre
            {{{T, n}, {T, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                     // hPost
            {{{nD}, {nD}}, ge::DT_FLOAT, ge::FORMAT_ND}                          // gamma (optional)
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},  // gradX
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // gradPhi
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // gradAlpha
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // gradBias
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}  // gradGamma (optional)
        },
        {{"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {T, n, D},        // gradX: [T, n, D]
        {fusionSize, nD}, // gradPhi: [fusionSize, nD]
        {3},              // gradAlpha: [3]
        {fusionSize},     // gradBias: [fusionSize]
        {n, D}            // gradGamma: [n, D]
    };

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

/*
 * TND格式测试用例2：T=8192, n=4，D=2048, x数据类型fp16
 * alpha [0.5 0.5 0.5]，hc_eps 200
 * 预期结果：成功
 */
TEST_F(MhcPreBackwardProto, Ut_Check_Case08_TND_T8192_n4_D2048_FP16)
{
    float alpha[3] = {0.6f, 0.6f, 0.6f}; // alpha = [0.6, 0.6, 0.6]
    uint32_t T = 8192;
    uint32_t n = 4;
    uint32_t D = 2048;
    uint32_t nD = n * D;                 // 8192
    uint32_t fusionSize = n * n + 2 * n; // 24
    float hcEps = 0.000001f;

    gert::InfershapeContextPara infershapeContextPara(
        "MhcPreBackward",
        {
            {{{T, n, D}, {T, n, D}}, ge::DT_FLOAT16, ge::FORMAT_ND},             // x (TND format)
            {{{fusionSize, nD}, {fusionSize, nD}}, ge::DT_FLOAT, ge::FORMAT_ND}, // phi
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},                           // alpha
            {{{T, D}, {T, D}}, ge::DT_FLOAT16, ge::FORMAT_ND},                   // gradHIn (TD format)
            {{{T, n}, {T, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                     // gradHPost (TN format)
            {{{T, n, n}, {T, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},               // gradHRes (TNN format)
            {{{T}, {T}}, ge::DT_FLOAT, ge::FORMAT_ND},                           // invRms (T format)
            {{{T, fusionSize}, {T, fusionSize}}, ge::DT_FLOAT, ge::FORMAT_ND},   // hMix
            {{{T, n}, {T, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                     // hPre
            {{{T, n}, {T, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                     // hPost
            {{{nD}, {nD}}, ge::DT_FLOAT, ge::FORMAT_ND}                          // gamma (optional)
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // gradX
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // gradPhi
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // gradAlpha
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // gradBias
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}    // gradGamma (optional)
        },
        {{"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {T, n, D},        // gradX: [T, n, D]
        {fusionSize, nD}, // gradPhi: [fusionSize, nD]
        {3},              // gradAlpha: [3]
        {fusionSize},     // gradBias: [fusionSize]
        {n, D}            // gradGamma: [n, D]
    };

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

/*
 * TND格式测试用例3：T=1024, n=4，D=512, x数据类型bf16
 * gradHRes为TN!格式 [T, N!] (2D)
 * 预期结果：成功
 */
TEST_F(MhcPreBackwardProto, Ut_Check_Case09_TND_T1024_n4_D512_BF16_TNFact)
{
    float alpha[3] = {0.3f, 0.3f, 0.3f};
    uint32_t T = 1024;
    uint32_t n = 4;
    uint32_t D = 512;
    uint32_t nD = n * D; // 2048
    uint32_t factN = 1;  // N! = 24 for n=4
    for (uint32_t i = 2; i <= n; i++) {
        factN *= i;
    } // Factorial(4) = 24
    uint32_t fusionSize = factN + 2 * n; // N!+2N = 32
    float hcEps = 0.000001f;

    gert::InfershapeContextPara infershapeContextPara(
        "MhcPreBackward",
        {
            {{{T, n, D}, {T, n, D}}, ge::DT_BF16, ge::FORMAT_ND},                // x (TND format)
            {{{fusionSize, nD}, {fusionSize, nD}}, ge::DT_FLOAT, ge::FORMAT_ND}, // phi
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},                           // alpha
            {{{T, D}, {T, D}}, ge::DT_BF16, ge::FORMAT_ND},                      // gradHIn (TD format)
            {{{T, n}, {T, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                     // gradHPost (TN format)
            {{{T, factN}, {T, factN}}, ge::DT_FLOAT, ge::FORMAT_ND},             // gradHRes (TN! format, 2D)
            {{{T}, {T}}, ge::DT_FLOAT, ge::FORMAT_ND},                           // invRms (T format)
            {{{T, fusionSize}, {T, fusionSize}}, ge::DT_FLOAT, ge::FORMAT_ND},   // hMix
            {{{T, n}, {T, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                     // hPre
            {{{T, n}, {T, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                     // hPost
            {{{nD}, {nD}}, ge::DT_FLOAT, ge::FORMAT_ND}                          // gamma (optional)
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},  // gradX
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // gradPhi
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // gradAlpha
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // gradBias
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}  // gradGamma (optional)
        },
        {{"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {T, n, D},        // gradX: [T, n, D]
        {fusionSize, nD}, // gradPhi: [fusionSize, nD]
        {3},              // gradAlpha: [3]
        {fusionSize},     // gradBias: [fusionSize]
        {n, D}            // gradGamma: [n, D]
    };

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

/*
 * BSND格式测试用例4：B=2, S=4096, n=4，D=1536, x数据类型bf16
 * gradHRes为BSN!格式 [B, S, N!] (3D)
 * 预期结果：成功
 */
TEST_F(MhcPreBackwardProto, Ut_Check_Case10_B2_S4096_n4_D1536_BF16_BSNFact)
{
    float alpha[3] = {0.25f, 0.25f, 0.25f};
    uint32_t B = 2;
    uint32_t S = 4096;
    uint32_t n = 4;
    uint32_t D = 1536;
    uint32_t nD = n * D; // 6144
    uint32_t factN = 1;  // N! = 24 for n=4
    for (uint32_t i = 2; i <= n; i++) {
        factN *= i;
    } // Factorial(4) = 24
    uint32_t fusionSize = factN + 2 * n; // N!+2N = 32
    float hcEps = 0.000001f;

    gert::InfershapeContextPara infershapeContextPara(
        "MhcPreBackward",
        {
            {{{B, S, n, D}, {B, S, n, D}}, ge::DT_BF16, ge::FORMAT_ND},              // x
            {{{fusionSize, nD}, {fusionSize, nD}}, ge::DT_FLOAT, ge::FORMAT_ND},     // phi
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // alpha
            {{{B, S, D}, {B, S, D}}, ge::DT_BF16, ge::FORMAT_ND},                    // gradHIn
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // gradHPost
            {{{B, S, factN}, {B, S, factN}}, ge::DT_FLOAT, ge::FORMAT_ND},           // gradHRes (BSN! format, 3D)
            {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // invRms
            {{{B, S, fusionSize}, {B, S, fusionSize}}, ge::DT_FLOAT, ge::FORMAT_ND}, // hMix
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // hPre
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // hPost
            {{{nD}, {nD}}, ge::DT_FLOAT, ge::FORMAT_ND}                              // gamma (optional)
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},  // gradX
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // gradPhi
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // gradAlpha
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // gradBias
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}  // gradGamma (optional)
        },
        {{"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {B, S, n, D},     // gradX: [B, S, n, D]
        {fusionSize, nD}, // gradPhi: [fusionSize, nD]
        {3},              // gradAlpha: [3]
        {fusionSize},     // gradBias: [fusionSize]
        {n, D}            // gradGamma: [n, D]
    };

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

/*
 * 边界测试用例1：B=1, S=1, n=4，D=64, x数据类型bf16
 * 最小尺寸测试
 * 预期结果：成功
 */
TEST_F(MhcPreBackwardProto, Ut_Check_Case11_B1_S1_n4_D1_BF16_Boundary)
{
    float alpha[3] = {0.1f, 0.1f, 0.1f}; // alpha = [0.1, 0.1, 0.1]
    uint32_t B = 1;
    uint32_t S = 1;
    uint32_t n = 4;
    uint32_t D = 64;
    uint32_t nD = n * D;                 // 256
    uint32_t fusionSize = n * n + 2 * n; // 24
    float hcEps = 0.000001f;

    gert::InfershapeContextPara infershapeContextPara(
        "MhcPreBackward",
        {
            {{{B, S, n, D}, {B, S, n, D}}, ge::DT_BF16, ge::FORMAT_ND},              // x
            {{{fusionSize, nD}, {fusionSize, nD}}, ge::DT_FLOAT, ge::FORMAT_ND},     // phi
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // alpha
            {{{B, S, D}, {B, S, D}}, ge::DT_BF16, ge::FORMAT_ND},                    // gradHIn
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // gradHPost
            {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},             // gradHRes
            {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // invRms
            {{{B, S, fusionSize}, {B, S, fusionSize}}, ge::DT_FLOAT, ge::FORMAT_ND}, // hMix
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // hPre
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // hPost
            {{{nD}, {nD}}, ge::DT_FLOAT, ge::FORMAT_ND}                              // gamma (optional)
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},  // gradX
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // gradPhi
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // gradAlpha
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // gradBias
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}  // gradGamma (optional)
        },
        {{"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {B, S, n, D},     // gradX: [B, S, n, D]
        {fusionSize, nD}, // gradPhi: [fusionSize, nD]
        {3},              // gradAlpha: [3]
        {fusionSize},     // gradBias: [fusionSize]
        {n, D}            // gradGamma: [n, D]
    };

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

/*
 * 边界测试用例2：B=1, S=65535, n=8，D=16384, x数据类型fp16
 * 大尺寸测试
 * 预期结果：成功
 */
TEST_F(MhcPreBackwardProto, Ut_Check_Case10_B1_S65535_n8_D8192_FP16_Boundary)
{
    float alpha[3] = {0.8f, 0.8f, 0.8f}; // alpha = [0.8, 0.8, 0.8]
    uint32_t B = 1;
    uint32_t S = 65535;
    uint32_t n = 8;
    uint32_t D = 16384;
    uint32_t nD = n * D;
    uint32_t fusionSize = n * n + 2 * n; // 80
    float hcEps = 0.000001f;

    gert::InfershapeContextPara infershapeContextPara(
        "MhcPreBackward",
        {
            {{{B, S, n, D}, {B, S, n, D}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // x
            {{{fusionSize, nD}, {fusionSize, nD}}, ge::DT_FLOAT, ge::FORMAT_ND},     // phi
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // alpha
            {{{B, S, D}, {B, S, D}}, ge::DT_FLOAT16, ge::FORMAT_ND},                 // gradHIn
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // gradHPost
            {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},             // gradHRes
            {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // invRms
            {{{B, S, fusionSize}, {B, S, fusionSize}}, ge::DT_FLOAT, ge::FORMAT_ND}, // hMix
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // hPre
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // hPost
            {{{nD}, {nD}}, ge::DT_FLOAT, ge::FORMAT_ND}                              // gamma (optional)
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // gradX
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // gradPhi
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // gradAlpha
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // gradBias
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}    // gradGamma (optional)
        },
        {{"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {B, S, n, D},     // gradX: [B, S, n, D]
        {fusionSize, nD}, // gradPhi: [fusionSize, nD]
        {3},              // gradAlpha: [3]
        {fusionSize},     // gradBias: [fusionSize]
        {n, D}            // gradGamma: [n, D]
    };

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
