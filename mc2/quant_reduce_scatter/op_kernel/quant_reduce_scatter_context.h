/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 **/

/*!
 * \file quant_reduce_scatter_context.h
 * \brief
 */
#ifndef QUANT_REDUCE_SCATTER_CONTEXT_H
#define QUANT_REDUCE_SCATTER_CONTEXT_H

constexpr uint32_t HCCL_MAX_RANK_SIZE = 1024U;

// 低比特量化通信算子（quant_reduce_scatter / quant_all_reduce）公共 context 布局，
// 由 torch_extension 侧 CommChannelBuilderManager 创建并填充，kernel 侧只读
struct QuantReduceScatterContext {
    uint32_t rankId = 0;                           // 本 rank 在通信组内的 rank id
    uint32_t rankSizePerServer = 0;                // 单服务器内 rank 数
    uint64_t hcclBuffer_[HCCL_MAX_RANK_SIZE] = {}; // 各 rank 内置 HCCL buffer 的映射地址
};

#endif // QUANT_REDUCE_SCATTER_CONTEXT_H
