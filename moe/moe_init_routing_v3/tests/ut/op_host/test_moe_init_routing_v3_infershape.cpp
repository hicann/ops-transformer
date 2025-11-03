// /**
//  * This program is free software, you can redistribute it and/or modify.
//  * Copyright (c) 2025 Huawei Technologies Co., Ltd.
//  * This file is a part of the CANN Open Software.
//  * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
//  * Please refer to the License for details. You may not use this file except in compliance with the License.
//  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
//  * See LICENSE in the root of the software repository for the full text of the License.
//  */

// /* !
//  * \file test_moe_init_routing_v3_infershape.cpp
//  * \brief
//  */
#include <gtest/gtest.h>
#include <iostream>
#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"
#include "infer_shaperange_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "infer_datatype_context_faker.h"
#include "log/log.h"

class MoeInitRoutingV3 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeInitRoutingV3 SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeInitRoutingV3 TearDown" << std::endl;
    }
};

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_1)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3",
                                                      {
                                                        {{{-2}, {-2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 30})}, 
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1}, {30}, {-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_2)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3",
                                                      {
                                                        {{{-1, -1}, {-1, -1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{-1, -1}, {-1, -1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{-1}, {-1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})}, 
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1}, {7}, {-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_3)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3",
                                                      {
                                                        {{{3, 128}, {3, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{3, 8}, {3, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})}, 
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{24, 128}, {24}, {7}, {24}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_4)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3",
                                                      {
                                                        {{{-2}, {-2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)}, 
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},  
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)}, 
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, 
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})}, 
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1}, {7}, {-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_5)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3",
                                                      {
                                                        {{{-2}, {-2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)}, 
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},  
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)}, 
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)}, 
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})}, 
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1}, {7}, {-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_6)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3",
                                                      {
                                                        {{{8 * 512, 1024}, {8 * 512, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{8 * 512, 512}, {8 * 512, 512}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{7}, {7}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)}, 
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},  
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)}, 
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, 
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})}, 
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{8 * 512 * 512, 1024}, {8 * 512 * 512}, {7}, {8 * 512 * 512}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_7)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3",
                                                      {
                                                        {{{8 * 512, 1024}, {8 * 512, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{8 * 512, 512}, {8 * 512, 512}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{7, 1}, {7, 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)}, 
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},  
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)}, 
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})}, 
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{8 * 512 * 512, 1024}, {8 * 512 * 512}, {7}, {}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_8)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3",
                                                      {
                                                        {{{8, 1024}, {8 , 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{8, 512}, {8, 512}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{-1}, {-1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)}, 
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},  
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)}, 
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)}, 
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})}, 
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{8 * 512, 1024}, {8 * 512}, {7}, {8 * 512}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_9)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3",
                                                      {
                                                        {{{2087, 192}, {2087, 192}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{2087, 7242}, {2087, 7242}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{-1}, {-1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)}, 
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},  
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)}, 
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({87, 222})}, 
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, 
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{15114054, 192}, {15114054}, {135}, {15114054}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_10)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3",
                                                      {
                                                        {{{2087, 192}, {2087, 192}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{2087, 7242}, {2087, 7242}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{2087}, {2087}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)}, 
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},  
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)}, 
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)}, 
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({87, 222})}, 
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, 
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{15114054, 192}, {15114054}, {135}, {15114054}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_11)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3",
                                                      {
                                                        {{{9223372036854775807, 1}, {9223372036854775807, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{-1, 1}, {-1, 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{-1}, {-1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)}, 
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},  
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)}, 
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({87, 222})}, 
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, 
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{9223372036854775807, 1}, {9223372036854775807}, {135}, {9223372036854775807}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_12)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3",
                                                      {
                                                        {{{2087, 192}, {2087, 192}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{2087, 7242}, {2087, 7242}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{-1}, {-1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)}, 
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, 
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)}, 
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)}, 
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({87, 222})}, 
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, 
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{15114054, 192}, {15114054}, {135}, {15114054}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_13)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3",
                                                      {
                                                        {{{2087, 192}, {2087, 192}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{-1, 7242}, {-1, 7242}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{-1}, {-1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)}, 
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, 
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)}, 
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)}, 
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({87, 222})}, 
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, 
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{15114054, 192}, {15114054}, {135}, {15114054}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_14)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3",
                                                      {
                                                        {{{2087, 192}, {2087, 192}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{2087, 7242}, {2087, 7242}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{-1}, {-1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)}, 
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, 
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)}, 
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)}, 
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({87, 222})}, 
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, 
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{15114054, 192}, {15114054}, {135}, {15114054}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_15)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3",
                                                      {
                                                        {{{2087, 192}, {2087, 192}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{2087, 7242}, {2087, 7242}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{2087}, {2087}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{-1}, {-1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)}, 
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, 
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)}, 
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)}, 
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({87, 222})}, 
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, 
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{15114054, 192}, {15114054}, {135}, {15114054}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_16)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3",
                                                      {
                                                        {{{2087, 192}, {2087, 192}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{2087, 7242}, {2087, 7242}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{2087}, {2087}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)}, 
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, 
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)}, 
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)}, 
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({87, 222})}, 
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, 
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{15114054, 192}, {15114054}, {135}, {15114054}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_17)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3",
                                                      {
                                                        {{{4, 14}, {4, 14}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{4, 5}, {4, 5}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)}, 
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)}, 
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)}, 
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)}, 
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 8})}, 
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{20, 14}, {20}, {256, 2}, {20}}; 
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_18)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3",
                                                      {
                                                        {{{4, 14}, {4, 14}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{4, 5}, {4, 5}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)}, 
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)}, 
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)}, 
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)}, 
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 8})}, 
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{20, 14}, {20}, {256, 2}, {20}}; 
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_datatype_01)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto data_type_func = spaceRegistry->GetOpImpl("MoeInitRoutingV3")->infer_datatype;
    
    if (data_type_func != nullptr) {
        ge::DataType fp16_ref = ge::DT_FLOAT16;
        ge::DataType fp32_ref = ge::DT_FLOAT;
        ge::DataType int32_ref = ge::DT_INT32;
        ge::DataType int64_ref = ge::DT_INT64;
        std::vector<int64_t> active_expert_range{1, 8};
        auto holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(4)
                                  .NodeIoNum(4, 5)
                                  .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs({{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                              {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                              {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                              {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              {"expert_tokens_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                              {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                              {"active_expert_range",
                                               Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(active_expert_range)},
                                              {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}})
                                  .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&fp16_ref, &int32_ref, &fp32_ref})
                                  .OutputDataTypes({&fp16_ref, &int32_ref, &int64_ref, &fp32_ref})
                                  .Build();
        ASSERT_EQ(data_type_func(holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
        auto expanded_x_output = holder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0);
        auto expanded_row_idx_output = holder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(1);
        auto expert_tokens_count_or_cumsum_output = holder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(2);
        auto expanded_scale_output = holder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(3);
        EXPECT_EQ(expanded_x_output, fp16_ref);
        EXPECT_EQ(expanded_row_idx_output, int32_ref);
        EXPECT_EQ(expert_tokens_count_or_cumsum_output, int64_ref);
        EXPECT_EQ(expanded_scale_output, fp32_ref);   
    }
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_datatype_02)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto data_type_func = spaceRegistry->GetOpImpl("MoeInitRoutingV3")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType int8_ref = ge::DT_INT8;
        ge::DataType fp16_ref = ge::DT_FLOAT16;
        ge::DataType fp32_ref = ge::DT_FLOAT;
        ge::DataType int32_ref = ge::DT_INT32;
        ge::DataType int64_ref = ge::DT_INT64;
        std::vector<int64_t> active_expert_range{1, 8};
        auto holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(4)
                                  .NodeIoNum(4, 5)
                                  .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs({{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                              {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                              {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                              {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              {"expert_tokens_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                              {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              {"active_expert_range",
                                               Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(active_expert_range)},
                                              {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}})
                                  .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&fp16_ref, &int32_ref, &fp32_ref})
                                  .OutputDataTypes({&int8_ref, &int32_ref, &int64_ref, &fp32_ref})
                                  .Build();
        ASSERT_EQ(data_type_func(holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
        auto expanded_x_output = holder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0);
        auto expanded_row_idx_output = holder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(1);
        auto expert_tokens_count_or_cumsum_output = holder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(2);
        auto expanded_scale_output = holder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(3);
        EXPECT_EQ(expanded_x_output, int8_ref);
        EXPECT_EQ(expanded_row_idx_output, int32_ref);
        EXPECT_EQ(expert_tokens_count_or_cumsum_output, int64_ref);
        EXPECT_EQ(expanded_scale_output, fp32_ref);   
    }
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_datatype_03)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto data_type_func = spaceRegistry->GetOpImpl("MoeInitRoutingV3")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType int8_ref = ge::DT_INT8;
        ge::DataType fp16_ref = ge::DT_FLOAT16;
        ge::DataType fp32_ref = ge::DT_FLOAT;
        ge::DataType int32_ref = ge::DT_INT32;
        ge::DataType int64_ref = ge::DT_INT64;
        std::vector<int64_t> active_expert_range{1, 8};
        auto holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(4)
                                  .NodeIoNum(4, 5)
                                  .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs({{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                              {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                              {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                              {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              {"expert_tokens_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                              {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                              {"active_expert_range",
                                               Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(active_expert_range)},
                                              {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}})
                                  .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&fp16_ref, &int32_ref, &fp32_ref})
                                  .OutputDataTypes({&int8_ref, &int32_ref, &int64_ref, &fp32_ref})
                                  .Build();
        ASSERT_EQ(data_type_func(holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
        auto expanded_x_output = holder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0);
        auto expanded_row_idx_output = holder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(1);
        auto expert_tokens_count_or_cumsum_output = holder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(2);
        auto expanded_scale_output = holder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(3);
        EXPECT_EQ(expanded_x_output, int8_ref);
        EXPECT_EQ(expanded_row_idx_output, int32_ref);
        EXPECT_EQ(expert_tokens_count_or_cumsum_output, int64_ref);
        EXPECT_EQ(expanded_scale_output, fp32_ref);  
    }
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_datatype_04)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto data_type_func = spaceRegistry->GetOpImpl("MoeInitRoutingV3")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType int8_ref = ge::DT_INT8;
        ge::DataType fp16_ref = ge::DT_FLOAT16;
        ge::DataType fp32_ref = ge::DT_FLOAT;
        ge::DataType int32_ref = ge::DT_INT32;
        ge::DataType int64_ref = ge::DT_INT64;
        std::vector<int64_t> active_expert_range{1, 7};
        auto holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(4)
                                  .NodeIoNum(4, 5)
                                  .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs({{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                              {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                              {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                              {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              {"expert_tokens_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                              {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                              {"active_expert_range",
                                               Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(active_expert_range)},
                                              {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}})
                                  .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&fp16_ref, &int32_ref, &fp32_ref})
                                  .OutputDataTypes({&int8_ref, &int32_ref, &int64_ref, &fp32_ref})
                                  .Build();
        ASSERT_EQ(data_type_func(holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
        auto expanded_x_output = holder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0);
        auto expanded_row_idx_output = holder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(1);
        auto expert_tokens_count_or_cumsum_output = holder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(2);
        auto expanded_scale_output = holder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(3);
        EXPECT_EQ(expanded_x_output, int8_ref);
        EXPECT_EQ(expanded_row_idx_output, int32_ref);
        EXPECT_EQ(expert_tokens_count_or_cumsum_output, int64_ref);
        EXPECT_EQ(expanded_scale_output, fp32_ref);  
    }
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_datatype_05)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto data_type_func = spaceRegistry->GetOpImpl("MoeInitRoutingV3")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType int8_ref = ge::DT_INT8;
        ge::DataType fp32_ref = ge::DT_FLOAT;
        ge::DataType int32_ref = ge::DT_INT32;
        ge::DataType int64_ref = ge::DT_INT64;
        std::vector<int64_t> active_expert_range{1, 7};
        auto holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(3)
                                  .NodeIoNum(3, 4)
                                  .NodeInputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs({{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                              {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                              {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                              {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              {"expert_tokens_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                              {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              {"active_expert_range",
                                               Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(active_expert_range)},
                                              {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}})
                                  .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&int8_ref, &int32_ref, &fp32_ref})
                                  .OutputDataTypes({&int8_ref, &int32_ref, &int64_ref, &fp32_ref})
                                  .Build();
        EXPECT_EQ(data_type_func(holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_FAILED);       
    }
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_datatype_06)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto data_type_func = spaceRegistry->GetOpImpl("MoeInitRoutingV3")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType int8_ref = ge::DT_INT8;
        ge::DataType fp32_ref = ge::DT_FLOAT;
        ge::DataType int32_ref = ge::DT_INT32;
        ge::DataType int64_ref = ge::DT_INT64;
        std::vector<int64_t> active_expert_range{1, 7};
        auto holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(3)
                                  .NodeIoNum(3, 4)
                                  .NodeInputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs({{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                              {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                              {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                              {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              {"expert_tokens_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                              {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                              {"active_expert_range",
                                               Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(active_expert_range)},
                                              {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}})
                                  .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&int8_ref, &int32_ref, &fp32_ref})
                                  .OutputDataTypes({&int8_ref, &int32_ref, &int64_ref, &fp32_ref})
                                  .Build();
        EXPECT_EQ(data_type_func(holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_FAILED);       
    }
}


TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infershape_range_00)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_range_func = spaceRegistry->GetOpImpl("MoeInitRoutingV3")->infer_shape_range;

    gert::Shape input_x_range_min{2, 2, 3, 8};
    gert::Shape input_x_range_max{2, -1, 3, 8};
    gert::Shape input_expert_idx_shape_range_min{2};
    gert::Shape input_expert_idx_shape_range_max{3};
    gert::Shape input_scale_shape_range_min{2};
    gert::Shape input_scale_shape_range_max{3};
    gert::Shape input_offset_shape_range_min{2};
    gert::Shape input_offset_shape_range_max{3};
    gert::Shape null1{};
    gert::Shape null2{};
    gert::Shape null3{};
    gert::Shape null4{};
    gert::Shape null5{};
    gert::Shape null6{};
    gert::Shape null7{};
    gert::Shape null8{};

    gert::Range<gert::Shape> input_x_range(&input_x_range_min, &input_x_range_max);
    gert::Range<gert::Shape> input_expert_idx_shape_range(&input_expert_idx_shape_range_min,
                                                          &input_expert_idx_shape_range_max);
    gert::Range<gert::Shape> input_scale_shape_range(&input_scale_shape_range_min, &input_scale_shape_range_max);
    gert::Range<gert::Shape> input_offset_shape_range(&input_offset_shape_range_min, &input_offset_shape_range_max);
    gert::Range<gert::Shape> output_0_shape_range(&null1, &null2);
    gert::Range<gert::Shape> output_1_shape_range(&null3, &null4);
    gert::Range<gert::Shape> output_2_shape_range(&null5, &null6);
    gert::Range<gert::Shape> output_3_shape_range(&null7, &null8);

    gert::Shape output_0_range_min{0, 0};
    gert::Shape output_0_range_max{-1, -1};
    gert::Range<gert::Shape> expect_output_0_shape_range(&output_0_range_min, &output_0_range_max);

    gert::Shape output_1_range_min{
        0,
    };
    gert::Shape output_1_range_max{
        -1,
    };
    gert::Range<gert::Shape> expect_output_1_shape_range(&output_1_range_min, &output_1_range_max);

    auto context_holder = gert::InferShapeRangeContextFaker()
                              .IrInputNum(4)
                              .NodeIoNum(4, 4)
                              .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputShapeRanges({&input_x_range, &input_expert_idx_shape_range,
                                                 &input_scale_shape_range, &input_offset_shape_range})
                              .OutputShapeRanges({&output_0_shape_range, &output_1_shape_range, &output_2_shape_range,
                                                  &output_3_shape_range})
                              .Build();

    auto context = context_holder.GetContext<gert::InferShapeRangeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(infer_shape_range_func(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(Ops::Base::ToString(*context->GetOutputShapeRange(0)->GetMin()), Ops::Base::ToString(output_0_range_min));
    EXPECT_EQ(Ops::Base::ToString(*context->GetOutputShapeRange(0)->GetMax()), Ops::Base::ToString(output_0_range_max));

    EXPECT_EQ(Ops::Base::ToString(*context->GetOutputShapeRange(1)->GetMin()), Ops::Base::ToString(output_1_range_min));
    EXPECT_EQ(Ops::Base::ToString(*context->GetOutputShapeRange(1)->GetMax()), Ops::Base::ToString(output_1_range_max));

    EXPECT_EQ(Ops::Base::ToString(*context->GetOutputShapeRange(2)->GetMin()), Ops::Base::ToString(output_1_range_min));
    EXPECT_EQ(Ops::Base::ToString(*context->GetOutputShapeRange(2)->GetMax()), Ops::Base::ToString(output_1_range_max));

    EXPECT_EQ(Ops::Base::ToString(*context->GetOutputShapeRange(3)->GetMin()), Ops::Base::ToString(output_1_range_min));
    EXPECT_EQ(Ops::Base::ToString(*context->GetOutputShapeRange(3)->GetMax()), Ops::Base::ToString(output_1_range_max));
}   

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_00)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3",
                                                      {
                                                        {{{8 * 512, 1024}, {8 * 512, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{8 * 512, 512}, {8 * 512, 512}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)}, 
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)}, 
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)}, 
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)}, 
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})}, 
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                      });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_01)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3",
                                                      {
                                                        {{{2087, 192}, {2087, 192}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{2087, 7242}, {2087, 7242}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{135, 1}, {135, 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)}, 
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)}, 
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)}, 
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({87, 222})}, 
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, 
                                                      });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_02)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3",
                                                      {
                                                        {{{2087, 192}, {2087, 192}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{2087, 7242}, {2087, 7242}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{135, 1}, {135, 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)}, 
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)}, 
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})}, 
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, 
                                                      });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_03)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3",
                                                      {
                                                        {{{2087, 192}, {2087, 192}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{2087, 7242}, {2087, 7242}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{2087}, {2087}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{-1}, {-1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)}, 
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)}, 
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)}, 
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})}, 
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, 
                                                      });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_04)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3",
                                                      {
                                                        {{{2087, 192}, {2087, 192}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{2087, 7242}, {2087, 7242}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{2087}, {2087}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{-1}, {-1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)}, 
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)}, 
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)}, 
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})}, 
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, 
                                                      });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_05)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3",
                                                      {
                                                        {{{2087, 192}, {2087, 192}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{2087, 7242}, {2087, 7242}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{2087}, {2087}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{-1}, {-1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)}, 
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)}, 
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)}, 
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)}, 
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})}, 
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, 
                                                      });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_06)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3",
                                                      {
                                                        {{{2087, 192}, {2087, 192}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{2087, 7242}, {2087, 7242}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{2087}, {2087}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{-1}, {-1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)}, 
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)}, 
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)}, 
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)}, 
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})}, 
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, 
                                                      });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_07)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3",
                                                      {
                                                        {{{2087, 192}, {2087, 192}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{2087, 7242}, {2087, 7242}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{2, 192}, {2, 192}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{-1}, {-1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)}, 
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)}, 
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, 
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})}, 
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, 
                                                      });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_08)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3",
                                                      {
                                                        {{{2087, 192}, {2087, 192}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{2087, 7242}, {2087, 7242}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{7, 191}, {7, 191}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{-1}, {-1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)}, 
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)}, 
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, 
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})}, 
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, 
                                                      });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}