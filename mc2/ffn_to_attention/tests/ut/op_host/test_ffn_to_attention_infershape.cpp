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
#include "mc2_infer_shape_case_executor.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace FFNToAttentionUT {

class FFNToAttentionInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FFNToAttentionInferShapeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FFNToAttentionInferShapeTest TearDown" << std::endl;
    }
};

TEST_F(FFNToAttentionInferShapeTest, Basic)
{
    gert::StorageShape xShape = {{1584, 7168}, {}};
    gert::StorageShape sessionIdsShape = {{1584}, {}};
    gert::StorageShape microBatchIdsShape = {{1584}, {}};
    gert::StorageShape tokenIdsShape = {{1584}, {}};
    gert::StorageShape expertOffsetsShape = {{1584}, {}};
    gert::StorageShape actualTokenNumShape = {{1}, {}};
    gert::StorageShape attnRankTableShape = {{11}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "FFNToAttention",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {sessionIdsShape, ge::DT_INT32, ge::FORMAT_ND},
         {microBatchIdsShape, ge::DT_INT32, ge::FORMAT_ND},
         {tokenIdsShape, ge::DT_INT32, ge::FORMAT_ND},
         {expertOffsetsShape, ge::DT_INT32, ge::FORMAT_ND},
         {actualTokenNumShape, ge::DT_INT64, ge::FORMAT_ND},
         {attnRankTableShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9, 7168})}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    std::vector<std::vector<int64_t>> expectOutputShape = {};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

} // namespace FFNToAttentionUT
