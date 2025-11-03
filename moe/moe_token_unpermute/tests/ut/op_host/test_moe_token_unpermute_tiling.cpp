/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_moe_token_unpermute_tiling.cpp
 * \brief
 */
 #include <iostream>
 #include <gtest/gtest.h>
 #include "../../../op_host/moe_token_unpermute_tiling.h"
 #include "tiling_context_faker.h"
 #include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class MoeTokenUnpermuteTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "MoeTokenUnpermuteTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MoeTokenUnpermuteTiling TearDown" << std::endl;
  }
};

TEST_F(MoeTokenUnpermuteTiling, test_tiling_prob_none_bf16) {
  optiling::MoeTokenUnpermuteCompileInfo compileInfo = {}; 
  gert::TilingContextPara tilingContextPara("MoeTokenUnpermute",
                                          {
                                            {{{6144, 20480}, {6144, 20480}}, ge::DT_BF16, ge::FORMAT_ND},
                                            {{{6144}, {6144}}, ge::DT_INT32, ge::FORMAT_ND}
                                          },
                                          {{{{6144, 20480}, {6144, 20480}}, ge::DT_BF16, ge::FORMAT_ND},},
                                          {{"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
                                          &compileInfo);
  int64_t expectTilingKey = 0;
  string expectTilingData = "20480 1 6144 20480 1 0 96 0 96 1 0 4 ";
  std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
  ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteTiling, test_tiling_prob_not_none_bf16) {
    optiling::MoeTokenUnpermuteCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermute",
                                            {
                                              {{{49152,5120},{49152,5120}}, ge::DT_BF16, ge::FORMAT_ND},
                                              {{{49152},{49152}}, ge::DT_INT32, ge::FORMAT_ND}
                                            },
                                            {
                                              {{{6144,5120},{6144,5120}}, ge::DT_BF16, ge::FORMAT_ND}
                                            },
                                            {
                                              {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
                                            },
                                            &compileInfo);
    int64_t expectTilingKey = 0;
    string expectTilingData = "5120 1 49152 5120 1 0 768 0 768 1 0 4 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteTiling, test_tiling_prob_none_fp16) {
    optiling::MoeTokenUnpermuteCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermute",
                                            {
                                              {{{6144,5120},{6144,5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{6144},{6144}}, ge::DT_INT32, ge::FORMAT_ND}
                                            },
                                            {
                                              {{{6144,5120},{6144,5120}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                            },
                                            {
                                              {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
                                            },
                                            &compileInfo);
    int64_t expectTilingKey = 0;
    string expectTilingData = "5120 1 6144 5120 1 0 96 0 96 1 0 4 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteTiling, test_tiling_prob_not_none_fp16) {
    optiling::MoeTokenUnpermuteCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermute",
                                            {
                                              {{{49152,5120},{49152,5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{49152},{49152}}, ge::DT_INT32, ge::FORMAT_ND}
                                            },
                                            {
                                              {{{6144,5120},{6144,5120}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                            },
                                            {
                                              {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
                                            },
                                            &compileInfo);
    int64_t expectTilingKey = 0;
    string expectTilingData = "5120 1 49152 5120 1 0 768 0 768 1 0 4 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteTiling, test_tiling_prob_none_fp32) {
    optiling::MoeTokenUnpermuteCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermute",
                                            {
                                              {{{6144,5120},{6144,5120}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              {{{6144},{6144}}, ge::DT_INT32, ge::FORMAT_ND}
                                            },
                                            {
                                              {{{6144,5120},{6144,5120}}, ge::DT_FLOAT, ge::FORMAT_ND}
                                            },
                                            {
                                              {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
                                            },
                                            &compileInfo);
    int64_t expectTilingKey = 0;
    string expectTilingData = "5120 1 6144 5120 1 0 96 0 96 1 0 4 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteTiling, test_tiling_prob_not_none_fp32) {
    optiling::MoeTokenUnpermuteCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermute",
                                            {
                                              {{{49152,5120},{49152,5120}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              {{{49152},{49152}}, ge::DT_INT32, ge::FORMAT_ND}
                                            },
                                            {
                                              {{{6144,5120},{6144,5120}}, ge::DT_FLOAT, ge::FORMAT_ND}
                                            },
                                            {
                                              {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
                                            },
                                            &compileInfo);
    int64_t expectTilingKey = 0;
    string expectTilingData = "5120 1 49152 5120 1 0 768 0 768 1 0 4 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteTiling, test_tiling_prob_not_none_mix_bf16_fp32) {
    optiling::MoeTokenUnpermuteCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermute",
                                            {
                                              {{{49152,5120},{49152,5120}}, ge::DT_BF16, ge::FORMAT_ND},
                                              {{{49152},{49152}}, ge::DT_INT32, ge::FORMAT_ND}
                                            },
                                            {
                                              {{{6144,5120},{6144,5120}}, ge::DT_BF16, ge::FORMAT_ND}
                                            },
                                            {
                                              {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
                                            },
                                            &compileInfo);
    int64_t expectTilingKey = 0;
    string expectTilingData = "5120 1 49152 5120 1 0 768 0 768 1 0 4 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteTiling, test_tiling_prob_not_none_mix_bf16_fp16) {
    optiling::MoeTokenUnpermuteCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermute",
                                            {
                                              {{{49152,5120},{49152,5120}}, ge::DT_BF16, ge::FORMAT_ND},
                                              {{{49152},{49152}}, ge::DT_INT32, ge::FORMAT_ND}
                                            },
                                            {
                                              {{{6144,5120},{6144,5120}}, ge::DT_BF16, ge::FORMAT_ND}
                                            },
                                            {
                                              {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
                                            },
                                            &compileInfo);
    int64_t expectTilingKey = 0;
    string expectTilingData = "5120 1 49152 5120 1 0 768 0 768 1 0 4 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteTiling, test_tiling_prob_not_none_mix_fp16_fp32) {
    optiling::MoeTokenUnpermuteCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermute",
                                            {
                                              {{{49152,5120},{49152,5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{49152},{49152}}, ge::DT_INT32, ge::FORMAT_ND}
                                            },
                                            {
                                              {{{6144,5120},{6144,5120}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                            },
                                            {
                                              {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
                                            },
                                            &compileInfo);
    int64_t expectTilingKey = 0;
    string expectTilingData = "5120 1 49152 5120 1 0 768 0 768 1 0 4 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteTiling, test_tiling_prob_not_none_mix_fp16_bf16) {
    optiling::MoeTokenUnpermuteCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermute",
                                            {
                                              {{{49152,5120},{49152,5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{49152},{49152}}, ge::DT_INT32, ge::FORMAT_ND}
                                            },
                                            {
                                              {{{6144,5120},{6144,5120}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                            },
                                            {
                                              {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
                                            },
                                            &compileInfo);
    int64_t expectTilingKey = 0;
    string expectTilingData = "5120 1 49152 5120 1 0 768 0 768 1 0 4 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteTiling, test_tiling_prob_not_none_mix_fp32_fp16) {
    optiling::MoeTokenUnpermuteCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermute",
                                            {
                                              {{{49152,5120},{49152,5120}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              {{{49152},{49152}}, ge::DT_INT32, ge::FORMAT_ND}
                                            },
                                            {
                                              {{{6144,5120},{6144,5120}}, ge::DT_FLOAT, ge::FORMAT_ND}
                                            },
                                            {
                                              {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
                                            },
                                            &compileInfo);
    int64_t expectTilingKey = 0;
    string expectTilingData = "5120 1 49152 5120 1 0 768 0 768 1 0 4 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteTiling, test_tiling_prob_not_none_mix_fp32_bf16) {
    optiling::MoeTokenUnpermuteCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermute",
                                            {
                                              {{{49152,5120},{49152,5120}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              {{{49152},{49152}}, ge::DT_INT32, ge::FORMAT_ND}
                                            },
                                            {
                                              {{{6144,5120},{6144,5120}}, ge::DT_FLOAT, ge::FORMAT_ND}
                                            },
                                            {
                                              {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
                                            },
                                            &compileInfo);
    int64_t expectTilingKey = 0;
    string expectTilingData = "5120 1 49152 5120 1 0 768 0 768 1 0 4 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
