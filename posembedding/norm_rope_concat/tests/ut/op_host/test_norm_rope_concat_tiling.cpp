/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file test_norm_rope_concat_tiling.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../op_host/norm_rope_concat_tiling.h"

using namespace std;

class NormRopeConcatTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "NormRopeConcatTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "NormRopeConcatTiling TearDown" << std::endl;
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

TEST_F(NormRopeConcatTiling, norm_rope_concat_tiling_0)
{
    optiling::NormRopeConcatCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "NormRopeConcat",
        {
            {{{1, 8, 8, 64}, {1, 8, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 8, 8, 64}, {1, 8, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 8, 8, 64}, {1, 8, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 8, 8, 64}, {1, 8, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 8, 8, 64}, {1, 8, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 8, 8, 64}, {1, 8, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 64}, {16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 64}, {16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_added_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rope_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"concat_order", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
            {"is_training", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "1 8 8 8 8 8 8 16 16 16 16 1 8 8 64 64 8 64 8 4359484440192628669 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
