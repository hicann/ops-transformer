/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

namespace AttentionToFfnV2UT {

class AttentionToFfnV2InferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AttentionToFfnV2InferShapeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AttentionToFfnV2InferShapeTest TearDown" << std::endl;
    }
};

TEST_F(AttentionToFfnV2InferShapeTest, Basic)
{
    gert::StorageShape contextShape = {{1}, {}};
    gert::StorageShape xShape = {{1, 16, 7168}, {}};
    gert::StorageShape sessionIdShape = {{1}, {}};
    gert::StorageShape microBatchIdShape = {{1}, {}};
    gert::StorageShape layerIdShape = {{1}, {}};
    gert::StorageShape expertIdsShape = {{1, 16, 8}, {}};
    gert::StorageShape expertRankTableShape = {{1, 9, 4}, {}};
    gert::StorageShape scalesShape = {{9, 7168}, {}};
    gert::StorageShape activeMaskShape = {{1, 16}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "AttentionToFfnV2",
        {{contextShape, ge::DT_INT32, ge::FORMAT_ND},
         {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {sessionIdShape, ge::DT_INT32, ge::FORMAT_ND},
         {microBatchIdShape, ge::DT_INT32, ge::FORMAT_ND},
         {layerIdShape, ge::DT_INT32, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND},
         {expertRankTableShape, ge::DT_INT32, ge::FORMAT_ND},
         {scalesShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {activeMaskShape, ge::DT_BOOL, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
         {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
         {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    std::vector<std::vector<int64_t>> expectOutputShape = {};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

} // namespace AttentionToFfnV2UT
