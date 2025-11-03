/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file test_interleave_rope_tiling.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../op_host/interleave_rope_tiling.h"

using namespace std;

class InterleaveRopeTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "InterleaveRopeTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "InterleaveRopeTiling TearDown" << std::endl;
    }
};

static string TilingData2Str(const gert::TilingData* tiling_data)
{
    auto data = tiling_data->GetData();
    string result;
    for (size_t i = 0; i < tiling_data->GetDataSize(); i += sizeof(int64_t)) {
        result += std::to_string((reinterpret_cast<const int64_t*>(tiling_data->GetData())[i / sizeof(int64_t)]));
        result += " ";
    }

    return result;
}

TEST_F(InterleaveRopeTiling, interleave_rope_tiling_000)
{
    optiling::InterleaveRopeCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("InterleaveRope",
                                              {
                                                {{{32, 32, 1, 64}, {32, 32, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{32, 1, 1, 64}, {32, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{32, 1, 1, 64}, {32, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                },
                                                {
                                                {{{32, 32, 1, 64}, {32, 32, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                },
                                              &compileInfo);
    uint64_t expectTilingKey = 1000;
    string expectTilingData = "8 0 32 32 1 64 4 4 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(InterleaveRopeTiling, interleave_rope_tiling_001) 
{
    optiling::InterleaveRopeCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("InterleaveRope",
                                              {
                                                {{{32, 32, 4, 64}, {32, 32, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{32, 1, 1, 64}, {32, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{32, 1, 1, 64}, {32, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                },
                                                {
                                                {{{32, 32, 4, 64}, {32, 32, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                },
                                              &compileInfo);
    uint64_t expectTilingKey = 2000;
    string expectTilingData = "64 1 32 32 4 64 32 32 0 0 0 2 2 1 2 2 1 2 2 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(InterleaveRopeTiling, interleave_rope_tiling_002)
{
    optiling::InterleaveRopeCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("InterleaveRope",
                                              {
                                                {{{32, 32, 4, 64}, {32, 32, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{32, 1, 4, 64}, {32, 1, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{32, 1, 4, 64}, {32, 1, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                },
                                                {
                                                {{{32, 32, 4, 64}, {32, 32, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                },
                                              &compileInfo);
    uint64_t expectTilingKey = 3000;
    string expectTilingData = "32 0 32 32 4 64 1 1 0 0 0 4 4 1 4 4 1 4 4 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(InterleaveRopeTiling, interleave_rope_tiling_hidden_dim_not_64)
{
    optiling::InterleaveRopeCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("InterleaveRope",
                                              {
                                                {{{32, 32, 1, 63}, {32, 32, 1, 63}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{32, 1, 1, 64}, {32, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{32, 1, 1, 64}, {32, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                },
                                                {
                                                {{{32, 32, 1, 64}, {32, 32, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(InterleaveRopeTiling, interleave_rope_tiling_shape_len_not_4)
{
    optiling::InterleaveRopeCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("InterleaveRope",
                                              {
                                                {{{32, 32, 1}, {32, 32, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{32, 1, 1, 64}, {32, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{32, 1, 1, 64}, {32, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                },
                                                {
                                                {{{32, 32, 1, 64}, {32, 32, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(InterleaveRopeTiling, interleave_rope_tiling_sin_not_eq_cos)
{
    optiling::InterleaveRopeCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("InterleaveRope",
                                              {
                                                {{{32, 32, 1, 63}, {32, 32, 1, 63}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{32, 1, 1, 64}, {32, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{33, 1, 1, 64}, {33, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                },
                                                {
                                                {{{32, 32, 1, 64}, {32, 32, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(InterleaveRopeTiling, interleave_rope_tiling_cosB_not_eq_batch_size)
{
    optiling::InterleaveRopeCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("InterleaveRope",
                                              {
                                                {{{32, 32, 1, 64}, {32, 32, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{3, 1, 1, 64}, {3, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{3, 1, 1, 64}, {3, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                },
                                                {
                                                {{{32, 32, 1, 64}, {32, 32, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(InterleaveRopeTiling, interleave_rope_tiling_cosN_not_eq_1)
{
    optiling::InterleaveRopeCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("InterleaveRope",
                                              {
                                                {{{32, 32, 1, 64}, {32, 32, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{32, 3, 1, 64}, {32, 3, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{33, 3, 1, 64}, {33, 3, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                },
                                                {
                                                {{{32, 32, 1, 64}, {32, 32, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(InterleaveRopeTiling, interleave_rope_tiling_cosD_not_eq_HiddenDim)
{
    optiling::InterleaveRopeCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("InterleaveRope",
                                              {
                                                {{{32, 32, 1, 64}, {32, 32, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{32, 1, 1, 62}, {32, 1, 1, 62}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{33, 1, 1, 62}, {33, 1, 1, 62}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                },
                                                {
                                                {{{32, 32, 1, 64}, {32, 32, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

