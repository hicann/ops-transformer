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
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class RotaryPositionEmbedding3dInfershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "RotaryPositionEmbedding3dInfershape Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "RotaryPositionEmbedding3dInfershape Proto Test TearDown" << std::endl;
    }
};

TEST_F(RotaryPositionEmbedding3dInfershape, RotaryPositionEmbedding3d_infer_shape_fp16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "RotaryPositionEmbedding3d",
        {
            // input info
            {{{2, 64, 4, 8}, {2, 64, 4, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 4, 8}, {2, 64, 4, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr (no attributes for RotaryPositionEmbedding3d)
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 64, 4, 8}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(RotaryPositionEmbedding3dInfershape, RotaryPositionEmbedding3d_infer_shape_fp32)
{
    gert::InfershapeContextPara infershapeContextPara("RotaryPositionEmbedding3d",
                                                      {
                                                          // input info
                                                          {{{1, 32, 16}, {1, 32, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{1, 32, 16}, {1, 32, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          // output info
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          // attr
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 32, 16}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(RotaryPositionEmbedding3dInfershape, RotaryPositionEmbedding3d_infer_shape_bf16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "RotaryPositionEmbedding3d",
        {
            // input info
            {{{4, 8, 16, 32}, {4, 8, 16, 32}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 8, 16, 32}, {4, 8, 16, 32}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            // attr
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 8, 16, 32}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(RotaryPositionEmbedding3dInfershape, RotaryPositionEmbedding3d_infer_shape_3d_fp16)
{
    gert::InfershapeContextPara infershapeContextPara("RotaryPositionEmbedding3d",
                                                      {
                                                          // input info (B, S, D)
                                                          {{{1, 64, 128}, {1, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{1, 64, 128}, {1, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          // output info
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          // attr
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 64, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
