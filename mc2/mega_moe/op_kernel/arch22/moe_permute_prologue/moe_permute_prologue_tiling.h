/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_permute_prologue_tiling.h
 * \brief 发送侧 chunk Permute Prologue 的 host tiling。
 */

#ifndef MC2_MOE_PERMUTE_PROLOGUE_TILING_H
#define MC2_MOE_PERMUTE_PROLOGUE_TILING_H

#include <cstdint>
#include <cstddef>

namespace Mc2Tiling {

// host 侧 TilingBase：推导 workspace
// expandedX 行步长 rowStride 由 kernel 侧 MoePermutePrologue 按 ElementDst
// 模板类型推导（int8 → hidden+512，其余 → hidden），host 无需下发。
class MoePermutePrologueTilingBase {
public:
    uint64_t workspaceSize_ = 0;

    // 供上层 mega_moe tiling 调用的统一入口
    bool DoTiling(int64_t m, int64_t topK, int64_t expertNumIn)
    {
        numTokens_ = m;
        topK_ = topK;
        expertNum_ = expertNumIn;

        // workspace
        workspaceSize_ = CalcWorkspaceSize();

        return true;
    }

private:
    // workspace: expandedRowIdx 全量 + preSumBeforeRank/cumsumMM 元数据 + 16MB 余量
    uint64_t CalcWorkspaceSize() const
    {
        size_t metadataWs = static_cast<size_t>(numTokens_) * topK_ * sizeof(int32_t); // expandedRowIdx 全量
        metadataWs += static_cast<size_t>(expertNum_) * sizeof(int32_t) * 2;           // preSumBeforeRank + cumsumMM
        constexpr size_t RESERVED_16MB = 16U * 1024U * 1024U;
        return static_cast<uint64_t>(metadataWs + RESERVED_16MB);
    }

    int64_t numTokens_ = 0;
    int64_t topK_ = 0;
    int64_t expertNum_ = 0;
};

} // namespace Mc2Tiling

#endif // MC2_MOE_PERMUTE_PROLOGUE_TILING_H
