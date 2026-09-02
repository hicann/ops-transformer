/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_mla_prolog_v3_tiling_common.h
 * \brief 公共头文件：MlaPrologV3 tiling UT 共享的 fixture 与 SocInfo 构造。
 */

#ifndef UTEST_MLA_PROLOG_V3_TILING_COMMON_H
#define UTEST_MLA_PROLOG_V3_TILING_COMMON_H

#include <iostream>
#include <gtest/gtest.h>
#include "../../../op_host/mla_prolog_v3_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

// 构造版本
static std::string MlaPrologV3_tiling_A2SocInfo = "{\n"
                                                  "  \"hardware_info\": {\n"
                                                  "    \"BT_SIZE\": 0,\n"
                                                  "    \"load3d_constraints\": \"1\",\n"
                                                  "    \"Intrinsic_fix_pipe_l0c2out\": false,\n"
                                                  "    \"Intrinsic_data_move_l12ub\": true,\n"
                                                  "    \"Intrinsic_data_move_l0c2ub\": true,\n"
                                                  "    \"Intrinsic_data_move_out2l1_nd2nz\": false,\n"
                                                  "    \"UB_SIZE\": 196608,\n"
                                                  "    \"L2_SIZE\": 201326592,\n"
                                                  "    \"L1_SIZE\": 524288,\n"
                                                  "    \"L0A_SIZE\": 65536,\n"
                                                  "    \"L0B_SIZE\": 65536,\n"
                                                  "    \"L0C_SIZE\": 131072,\n"
                                                  "    \"vector_core_cnt\": 40,\n"
                                                  "    \"cube_core_cnt\": 20,\n"
                                                  "    \"socVersion\": \"Ascend910_B3\"\n"
                                                  "  }\n"
                                                  "}";

// 构造版本
static std::string MlaPrologV3_tiling_950SocInfo = "{\n"
                                                   "  \"hardware_info\": {\n"
                                                   "    \"BT_SIZE\": 0,\n"
                                                   "    \"load3d_constraints\": \"1\",\n"
                                                   "    \"Intrinsic_fix_pipe_l0c2out\": false,\n"
                                                   "    \"Intrinsic_data_move_l12ub\": true,\n"
                                                   "    \"Intrinsic_data_move_l0c2ub\": true,\n"
                                                   "    \"Intrinsic_data_move_out2l1_nd2nz\": false,\n"
                                                   "    \"UB_SIZE\": 196608,\n"
                                                   "    \"L2_SIZE\": 201326592,\n"
                                                   "    \"L1_SIZE\": 524288,\n"
                                                   "    \"L0A_SIZE\": 65536,\n"
                                                   "    \"L0B_SIZE\": 65536,\n"
                                                   "    \"L0C_SIZE\": 131072,\n"
                                                   "    \"vector_core_cnt\": 40,\n"
                                                   "    \"cube_core_cnt\": 20,\n"
                                                   "    \"socVersion\": \"Ascend950\"\n"
                                                   "  }\n"
                                                   "}";
class MlaPrologV3 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MlaPrologV3 SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MlaPrologV3 TearDown" << std::endl;
    }
};

#endif // UTEST_MLA_PROLOG_V3_TILING_COMMON_H
