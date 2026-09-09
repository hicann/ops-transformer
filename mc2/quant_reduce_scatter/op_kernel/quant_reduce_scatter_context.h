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

struct QuantReduceScatterContext {
    uint32_t epRankId = 0;
    uint32_t rankSizePerServer = 0;
    uint64_t kfcContextAddr = 0; // 通信API所需的地址
    uint64_t epHcclBuffer_[HCCL_MAX_RANK_SIZE] = {};
    uint64_t hcommHandle_[HCCL_MAX_RANK_SIZE] = {}; // ROCE或者URMA通信所需句柄
};

#endif // QUANT_REDUCE_SCATTER_CONTEXT_H
