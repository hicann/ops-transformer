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
 * \file quant_reduce_scatter_gen_task.cpp
 * \brief 静态shape图下沉占位注册
 */

#include <vector>
#include <cstdint>
#include "register/op_impl_registry.h"

namespace ops {

static ge::Status QuantReduceScatterCalcOpParam(gert::ExeResGenerationContext *context)
{
    // 不设置 attached stream infos，避免框架生成 hcom wait/record 任务与 kHcom 隐藏输入
    (void)context;
    return ge::GRAPH_SUCCESS;
}

static ge::Status QuantReduceScatterGenTask(const gert::ExeResGenerationContext *context,
                                            std::vector<std::vector<uint8_t>> &tasks)
{
    // 保留框架默认生成的 aicore taks，不做任何注入/改写
    (void)context;
    (void)tasks;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(QuantReduceScatter).CalcOpParam(QuantReduceScatterCalcOpParam).GenerateTask(QuantReduceScatterGenTask);
} // namespace ops
