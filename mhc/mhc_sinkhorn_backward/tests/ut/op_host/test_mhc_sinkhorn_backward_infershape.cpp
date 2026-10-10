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
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class MhcSinkhornBackwardProto : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MhcSinkhornBackwardProto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MhcSinkhornBackwardProto TearDown" << std::endl;
    }
};

TEST_F(MhcSinkhornBackwardProto, MhcSinkhornBackward_infershape_bsnn)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MhcSinkhornBackward",
        {
            {{{4, 128, 8, 8}, {4, 128, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}, // 0: grad_y
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // 1: norm
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // 2: sum
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // 0: grad_input
        },
        {});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 128, 8, 8}, // grad_input
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MhcSinkhornBackwardProto, MhcSinkhornBackward_infershape_tnn)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MhcSinkhornBackward",
        {
            {{{512, 8, 8}, {512, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}, // 0: grad_y
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // 1: norm
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // 2: sum
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // 0: grad_input
        },
        {});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {512, 8, 8}, // grad_input
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MhcSinkhornBackwardProto, MhcSinkhornBackward_infershape_unknown_rank)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MhcSinkhornBackward",
        {
            {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND}, // 0: grad_y (unknown rank)
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},     // 1: norm
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},     // 2: sum
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // 0: grad_input
        },
        {});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-2}, // grad_input (unknown rank)
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MhcSinkhornBackwardProto, MhcSinkhornBackward_infershape_bsnn_n4)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MhcSinkhornBackward",
        {
            {{{2, 256, 4, 4}, {2, 256, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, // 0: grad_y
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // 1: norm
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // 2: sum
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // 0: grad_input
        },
        {});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 256, 4, 4}, // grad_input
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MhcSinkhornBackwardProto, MhcSinkhornBackward_infershape_tnn_n6)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MhcSinkhornBackward",
        {
            {{{-1, 6, 6}, {-1, 6, 6}}, ge::DT_FLOAT, ge::FORMAT_ND}, // 0: grad_y
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                 // 1: norm
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                 // 2: sum
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // 0: grad_input
        },
        {});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, 6, 6}, // grad_input
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MhcSinkhornBackwardProto, MhcSinkhornBackward_infershape_invalid_dim_count_2d)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MhcSinkhornBackward",
        {
            {{{128, 128}, {128, 128}}, ge::DT_FLOAT, ge::FORMAT_ND}, // 0: grad_y (2D, invalid)
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                 // 1: norm
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                 // 2: sum
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // 0: grad_input
        },
        {});

    std::vector<std::vector<int64_t>> expectOutputShape = {};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(MhcSinkhornBackwardProto, MhcSinkhornBackward_infershape_invalid_dim_count_5d)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MhcSinkhornBackward",
        {
            {{{2, 128, 8, 8, 1}, {2, 128, 8, 8, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}, // 0: grad_y (5D, invalid)
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // 1: norm
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // 2: sum
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // 0: grad_input
        },
        {});

    std::vector<std::vector<int64_t>> expectOutputShape = {};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(MhcSinkhornBackwardProto, MhcSinkhornBackward_infershape_bsnn_batch1)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MhcSinkhornBackward",
        {
            {{{1, 64, 8, 8}, {1, 64, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}, // 0: grad_y
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // 1: norm
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // 2: sum
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // 0: grad_input
        },
        {});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1, 64, 8, 8}, // grad_input
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MhcSinkhornBackwardProto, MhcSinkhornBackward_infershape_tnn_large)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MhcSinkhornBackward",
        {
            {{{4096, 8, 8}, {4096, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}, // 0: grad_y
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                     // 1: norm
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                     // 2: sum
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // 0: grad_input
        },
        {});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4096, 8, 8}, // grad_input
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
