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
 * \file quant_reduce_scatter_tiling_data.h
 * \brief 定义TilingData
 */

#ifndef QUANT_REDUCE_SCATTER_TILING_DATA_H
#define QUANT_REDUCE_SCATTER_TILING_DATA_H

#include <cstdint>

struct QuantReduceScatterTilingInfo {
    uint64_t bs;
    uint64_t hiddenSize;
    uint64_t scaleHiddenSize;
    uint64_t aivNum;
    uint64_t totalWinSize; // Win区总大小，即HCCL_BUFFER_SIZE
    uint32_t xPerBlock;    // host 侧基于 TARGET_ITER 公式推荐的每块元素数
    uint32_t alignBlock;   // xPerBlock 对齐粒度（元素数，host/kernel共享）
    uint64_t hcclBufferSize;
};

struct QuantReduceScatterTilingData {
    // 注意：不能声明Mc2InitTiling/Mc2CcTiling字段，
    // GE/HCCL框架侧按tilingData中是否存在Mc2InitTiling结构识别task-sink MC2算子，并解析其内容创建通信资源，
    // 本算子走context张量MTE自同步，proto无group，该字段从未填充且kernel不读，残留会导致GE图加载时读到脏数据触发
    QuantReduceScatterTilingInfo quantReduceScatterTilingInfo;
};

#endif // QUANT_REDUCE_SCATTER_TILING_DATA_H
