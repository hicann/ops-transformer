/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file apply_rotary_pos_emb_grad_tiling.h
 * \brief kernel UT tiling shim.
 *
 * 文件名(opName_tiling.h)被 UT 框架(ut.cmake AddOpTestCase)识别并以 -include 方式强制包含,
 * 取代框架 gen_tiling_head_file 脚本从 tiling so 提取生成的 apply_rotary_pos_emb_grad_tiling_data.h
 * (该生成头会重复定义 kernel 侧 tiling 结构, 与 kernel 源码自带的
 * op_kernel/arch35/apply_rotary_pos_emb_grad_tiling_data.h 冲突).
 *
 * kernel 入口通过 REGISTER_TILING_DEFAULT + GET_TILING_DATA_WITH_STRUCT 读取 tiling:
 * REGISTER_TILING_DEFAULT 在 CPU 编译下由 CANN 头展开为无害的 section 常量, 无需处理;
 * GET_TILING_DATA_WITH_STRUCT 为 NPU 编译器内置宏, 此处按字节拷贝语义补齐.
 */

#ifndef APPLY_ROTARY_POS_EMB_GRAD_TILING_H
#define APPLY_ROTARY_POS_EMB_GRAD_TILING_H

#include "kernel_tiling/kernel_tiling.h"

#include <cstdint>
#include <cstring>

#define __CCE_UT_TEST__

// NPU 编译时该宏由编译框架按输入 dtype 注入, UT 下手动指定本用例 dtype (fp32)
#ifndef DTYPE_GRAD_QUERY_EMBED
#define DTYPE_GRAD_QUERY_EMBED float
#endif

// kernel 源码(apt.cpp/arch35 模板)使用裸 max/min, NPU 编译器内置;
// g++ 下 std::max 等模板候选重载解析歧义, 提供非模板精确匹配(int64_t 即 long)消歧
inline int64_t max(int64_t x, int64_t y)
{
    return x > y ? x : y;
}

inline int64_t min(int64_t x, int64_t y)
{
    return x < y ? x : y;
}

inline uint64_t max(uint64_t x, uint64_t y)
{
    return x > y ? x : y;
}

inline uint64_t min(uint64_t x, uint64_t y)
{
    return x < y ? x : y;
}

template <typename T>
inline void InitApplyRopeGradTilingData(const uint8_t *tiling, T *tilingData)
{
    (void)memcpy(tilingData, tiling, sizeof(T));
}

#undef GET_TILING_DATA_WITH_STRUCT
#define GET_TILING_DATA_WITH_STRUCT(tilingStruct, tilingData, tilingArg) \
    tilingStruct tilingData; \
    InitApplyRopeGradTilingData(tilingArg, &tilingData)

#endif // APPLY_ROTARY_POS_EMB_GRAD_TILING_H
