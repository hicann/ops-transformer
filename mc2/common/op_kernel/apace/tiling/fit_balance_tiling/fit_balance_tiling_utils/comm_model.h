/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// fit_balance_tiling 通信模型: 线性拟合 y = k·x + 轮数·b + c
#pragma once

#include "fit_balance_types.h"

constexpr uint64_t FP4_PACK_RATIO = 2;

// 通信模型参数: 标定参数(k/b/c₀) + 通信形态 + 派生倍数——通信公式消费
struct CommModelParams {
    double slopeUsPerMb = 0;                     // 带宽斜率 k (us/MB)
    double roundOverheadUs = 0;                  // 每轮启动开销 b (us)
    double fixedOverheadUs = 0;                  // 一次性固定开销 c (us)
    uint64_t commMatrixLen = 0;                  // 通信矩阵另一维(K 或 N)
    uint64_t commDtypeSize = 0;                  // 通信每元素字节数
    uint64_t commDtypeSizeExpansionFraction = 1; // 子字节折算分母(fp4=2)
    uint64_t commVolumeMul = 1;                  // 派生: 单次通信物理字节倍数(单轮上限整形消费)
};

// 每元素有效字节数 = dtypeSize / fraction(fp4: 1/2 = 0.5B)
inline double GetRealDtypeSizes(const CommModelParams& comm)
{
    return static_cast<double>(comm.commDtypeSize) / comm.commDtypeSizeExpansionFraction;
}

// 单次通信调用的持续时间: y = k·x + b——不含一次性启动开销 c(c 是算子级的,
// 不管切几轮只付一次, 不归属任何单次调用; 流水线分析用本函数算各调用的持续时间)
inline double CommTime(const CommModelParams& comm, uint64_t mSize)
{
    const double commDataSize = static_cast<double>(mSize) * comm.commMatrixLen * GetRealDtypeSizes(comm);
    const double tmpSize = commDataSize / BYTES_PER_MB;
    return comm.slopeUsPerMb * tmpSize + comm.roundOverheadUs;
}

// 反解: x = (t − b) / k, 换算回 M 行数。目标时长不足每轮启动开销时返回 0。
inline uint64_t InverseCommTime(const CommModelParams& comm, double targetTime)
{
    if (targetTime <= comm.roundOverheadUs) {
        return 0;
    }
    const double tmpSize = (targetTime - comm.roundOverheadUs) / comm.slopeUsPerMb;
    return static_cast<uint64_t>(tmpSize * BYTES_PER_MB / (comm.commMatrixLen * GetRealDtypeSizes(comm)));
}

// 总通信耗时: y = k·X + 轮数·b + c
inline double TotalCommTime(const CommModelParams& comm, uint64_t totalLen, uint64_t numRounds)
{
    const double commDataSize = static_cast<double>(totalLen) * comm.commMatrixLen * GetRealDtypeSizes(comm);
    const double tmpSize = commDataSize / BYTES_PER_MB;
    return comm.slopeUsPerMb * tmpSize + comm.roundOverheadUs * static_cast<double>(numRounds) + comm.fixedOverheadUs;
}
