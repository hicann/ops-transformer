/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file weight_l2_prefetch.hpp
 * \brief A3(910_93) 权重 L2 预取：fire-and-forget 哑拷贝把权重字节拉进 L2（数据不消费），
 *        供 GEMM 真实 GM->L1 装载命中。W1 冷启动：AIC 在 flag0（轮就绪）wait 后、GMM1 组 0
 *        wait 前预取 W1 仅专家 0，与 pull g0 并发排空，空专家按 cumsum 过滤；scratch 为 L1
 *        功能区之上的 16KB 环写独占区，各核交错认领精确覆盖。BS 超过 MAX_BS 时不发射
 *        （大 BS 冷读已被深流水掩盖）。
 */

#ifndef MEGA_MOE_WEIGHT_L2_PREFETCH_HPP
#define MEGA_MOE_WEIGHT_L2_PREFETCH_HPP

#include "kernel_operator.h"

#include "get_tensor_addr.hpp"

// 覆盖前 EXPERTS 个专家（当前仅专家 0）；CHUNKS 为门槛：每专家权重 ≤ CHUNKS×coreNum×16KB
// （24 核即 12MB）才发射全量预取，超限整体跳过（部分覆盖形态实测负收益）
constexpr uint32_t WEIGHT_L2_PREFETCH_COLDSTART_EXPERTS = 1U;
constexpr uint32_t WEIGHT_L2_PREFETCH_COLDSTART_CHUNKS = 32U;

constexpr uint32_t WEIGHT_L2_PREFETCH_CHUNK_BYTES = 16U * 1024U;
// BS 门限：超过此值不发射（大 BS 冷读已被深流水掩盖）
constexpr int64_t WEIGHT_L2_PREFETCH_MAX_BS = 512;

// 预取每专家字节数（K*N element；int4 两 element 共 1 字节，其余每 element 2 字节）。
template <class ElementB>
__aicore__ inline uint64_t WeightL2PrefetchBytesPerExpert(int64_t k, int64_t n)
{
    if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
        return static_cast<uint64_t>(k) * static_cast<uint64_t>(n) / 2U;
    } else if constexpr (std::is_same_v<ElementB, int8_t>) {
        return static_cast<uint64_t>(k) * static_cast<uint64_t>(n);
    } else {
        static_assert(sizeof(ElementB) <= 2, "prefetch byte accounting assumes at most 2 bytes per element");
        return static_cast<uint64_t>(k) * static_cast<uint64_t>(n) * 2U;
    }
}

/*
 * AIC 侧区间哑预取：GM->L1 环写，[firstExpert, lastExpert) 逐专家全量，各核按
 * (GetBlockIdx(), GetBlockNum()) 交错认领精确覆盖；每专家权重超门槛（CHUNKS x coreNum
 * x CHUNK，见常量注释）时整体跳过——部分覆盖无收益。
 */
__aicore__ inline void WeightL2PrefetchAicRange(GM_ADDR weightList, bool isPerExpertWeightTensor,
                                                uint64_t bytesPerExpert, uint32_t firstExpert, uint32_t lastExpert,
                                                AscendC::LocalTensor<uint8_t> l1Scratch)
{
    if (bytesPerExpert == 0U || firstExpert >= lastExpert) {
        return;
    }
    uint32_t coreIdx = AscendC::GetBlockIdx();
    uint32_t coreNum = AscendC::GetBlockNum();
    uint64_t capBytes =
        static_cast<uint64_t>(WEIGHT_L2_PREFETCH_COLDSTART_CHUNKS) * coreNum * WEIGHT_L2_PREFETCH_CHUNK_BYTES;
    if (bytesPerExpert > capBytes) {
        return;
    }
    uint64_t limit = bytesPerExpert;
    l1Scratch.SetSize(WEIGHT_L2_PREFETCH_CHUNK_BYTES);
    AscendC::GlobalTensor<uint8_t> weightGm;
    for (uint32_t expertIdx = firstExpert; expertIdx < lastExpert; ++expertIdx) {
        // per-expert 模式取列表项；stacked 模式按专家字节步进（与 gmGroupOffsetB 同源）。
        __gm__ uint8_t* regionBase = GetTensorAddr<uint8_t>(isPerExpertWeightTensor ? expertIdx : 0U, weightList);
        if (!isPerExpertWeightTensor) {
            regionBase += static_cast<uint64_t>(expertIdx) * bytesPerExpert;
        }
        // 各核交错认领 16KB chunk，环写同一块保留区。
        for (uint64_t offset = static_cast<uint64_t>(coreIdx) * WEIGHT_L2_PREFETCH_CHUNK_BYTES; offset < limit;
             offset += static_cast<uint64_t>(coreNum) * WEIGHT_L2_PREFETCH_CHUNK_BYTES) {
            uint64_t remain = limit - offset;
            uint32_t chunkBytes = remain < WEIGHT_L2_PREFETCH_CHUNK_BYTES ? static_cast<uint32_t>(remain) :
                                                                            WEIGHT_L2_PREFETCH_CHUNK_BYTES;
            weightGm.SetGlobalBuffer(regionBase + offset);
            AscendC::DataCopy(l1Scratch, weightGm, chunkBytes);
        }
    }
}

// W1 冷启动：覆盖前 EXPERTS 个专家的 W1，由调用方在 flag0 后发射（g0-pull 窗排空）。
__aicore__ inline void WeightL2PrefetchAicColdStartW1(GM_ADDR weightList, bool isPerExpertWeightTensor,
                                                      uint64_t bytesPerExpert, uint32_t expertPerRank,
                                                      AscendC::LocalTensor<uint8_t> l1Scratch)
{
    uint32_t expertCount =
        expertPerRank < WEIGHT_L2_PREFETCH_COLDSTART_EXPERTS ? expertPerRank : WEIGHT_L2_PREFETCH_COLDSTART_EXPERTS;
    WeightL2PrefetchAicRange(weightList, isPerExpertWeightTensor, bytesPerExpert, 0U, expertCount, l1Scratch);
}

#endif // MEGA_MOE_WEIGHT_L2_PREFETCH_HPP
