/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
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

class MatmulReduceScatterInfershape : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "MatmulReduceScatterInfershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MatmulReduceScatterInfershape TearDown" << std::endl;
    }
};

TEST_F(MatmulReduceScatterInfershape, basic) {
    gert::StorageShape x1StorageShape = {{8192, 1536}, {}};
    gert::StorageShape x2StorageShape = {{1536, 12288}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MatmulReduceScatter",
        {
            {x1StorageShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {x2StorageShape, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        }
    );

    std::vector<std::vector<int64_t>> expectOutputShape = {{1024, 12288}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MatmulReduceScatterInfershape, empty_tensor_test) {
    gert::StorageShape x1StorageShape = {{8192, 0}, {}};
    gert::StorageShape x2StorageShape = {{0, 12288}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MatmulReduceScatter",
        {
            {x1StorageShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {x2StorageShape, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)}
        }
    );

    ExecuteTestCase(infershapeContextPara);
}