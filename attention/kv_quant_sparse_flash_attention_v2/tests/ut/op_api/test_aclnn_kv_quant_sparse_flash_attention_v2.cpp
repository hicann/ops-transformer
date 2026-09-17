/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

// kv_quant_sparse_flash_attention_v2 无 aclnn 两段式接口（op_host 下无 op_api/aclnn 源文件），
// 本文件为算子定义注册冒烟用例：opapi UT 存在时 libopapi_transformer_ut.so 会整包链接
// 各算子 op_host 的 *_def.cpp（见 cmake/custom_build.cmake 中 ops_aclnn 归档），其静态初始化
// 在库加载阶段完成 OpDef 注册。用例被执行即说明库加载与全部算子定义注册流程无崩溃、
// 无重复注册冲突。
TEST(kv_quant_sparse_flash_attention_v2_opapi_ut, op_def_registration_smoke)
{
    SUCCEED();
}
