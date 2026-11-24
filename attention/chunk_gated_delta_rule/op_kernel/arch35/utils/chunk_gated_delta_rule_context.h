/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "../../chunk_gated_delta_rule_tiling_data.h"
#include "chunk_gated_delta_rule_layout.h"

namespace ChunkGatedDeltaRule {

using namespace AscendC;

// 入口只打包 GM 地址；张量描述符直接绑定到 Context 内的 KernelParams。
struct CGDRInitParams {
    GM_ADDR query;
    GM_ADDR key;
    GM_ADDR value;
    GM_ADDR beta;
    GM_ADDR initState;
    GM_ADDR seqlens;
    GM_ADDR gOptional;
    GM_ADDR attnOut;
    GM_ADDR finalState;
};

template <typename stateType>
struct KernelParams {
    // GM 逻辑 shape：Q/K:[T,Nk,Dk]，V/O:[T,Nv,Dv]，beta/g:[T,Nv]。
    GlobalTensor<bfloat16_t> query;
    GlobalTensor<bfloat16_t> key;
    GlobalTensor<bfloat16_t> value;
    GlobalTensor<bfloat16_t> beta;
    GlobalTensor<float> g;
    GlobalTensor<stateType> initialState; // [B,Nv,Dv,Dk]，batch/head 按 stateStride0/1 寻址
    GlobalTensor<bfloat16_t> out;
    GlobalTensor<stateType> finalState;     // [B,Nv,Dv,Dk]，连续布局
    GlobalTensor<int32_t> actualSeqLengths; // [B]，各 batch 长度，总和为 T
    float scale;
    uint32_t nk;
    uint32_t nv;
    uint32_t dk;
    uint32_t dv;
    uint32_t aiCoreNum;
    uint64_t batchCount;
    uint64_t tokenCount;   // T 维元素数,序列前缀和的合法上界(host 已校验 T=sum(L_i),
                           // 图模式取不到 host 数据时由 Scheduler 钳制兜底)
    uint64_t stateStride0; // 初始 state 的 batch 步长，单位为元素；0 表示广播
    uint64_t stateStride1; // 初始 state 的 value-head 步长，单位为元素；0 表示广播

    // 优化方案：在最终参数对象中就地绑定描述符，省去逐层构造、赋值和复制参数对象。
    // SetGlobalBuffer 只设置地址/长度等元数据，不搬运 GM 数据。
    // hasGamma 来自 CGDR 的模板常量；内联后可消除无 g 分支，无需为此扩展 Context 的模板参数。
    __aicore__ inline KernelParams(const CGDRInitParams& addresses, const ChunkGatedDeltaRuleTilingData& tiling,
                                   bool hasGamma)
    {
        // Q, K: [T,Nk,Dk]；value 头 h 共享 QK 头 floor(h/(Nv/Nk))。
        const uint64_t queryKeyElements = tiling.t * tiling.nk * tiling.dk;
        query.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(addresses.query), queryKeyElements);
        key.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(addresses.key), queryKeyElements);

        // V, O: [T,Nv,Dv]。
        const uint64_t valueElements = tiling.t * tiling.nv * tiling.dv;
        value.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(addresses.value), valueElements);
        out.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(addresses.attnOut), valueElements);

        // beta, g: [T,Nv]；g 为 log decay，alpha=exp(g)，缺省 g 时 alpha=1。
        const uint64_t gateElements = tiling.t * tiling.nv;
        beta.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(addresses.beta), gateElements);
        if (hasGamma) {
            g.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(addresses.gOptional), gateElements);
        }

        // S_in, S_out: [B,Nv,Dv,Dk]。初态按 stateStride0/1 寻址（可广播），终态连续写回。
        const uint64_t stateElements = tiling.b * tiling.nv * tiling.dv * tiling.dk;
        initialState.SetGlobalBuffer(reinterpret_cast<__gm__ stateType*>(addresses.initState), stateElements);
        finalState.SetGlobalBuffer(reinterpret_cast<__gm__ stateType*>(addresses.finalState), stateElements);

        // seq_lens: [B]，存各 batch 长度，非累积终点；空序列的终态等于初态。
        actualSeqLengths.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(addresses.seqlens), tiling.b);

        scale = tiling.scale;
        nk = static_cast<uint32_t>(tiling.nk);
        nv = static_cast<uint32_t>(tiling.nv);
        dk = static_cast<uint32_t>(tiling.dk);
        dv = static_cast<uint32_t>(tiling.dv);
        aiCoreNum = static_cast<uint32_t>(tiling.aiCoreNum);
        batchCount = static_cast<uint64_t>(tiling.b);
        tokenCount = static_cast<uint64_t>(tiling.t);
        stateStride0 = static_cast<uint64_t>(tiling.stateStride0);
        stateStride1 = static_cast<uint64_t>(tiling.stateStride1);
    }
};

// 一个完整的 (batch, value-head) 任务；Scheduler 产出，BasicBlock 执行。
// tokenStart=sum(seq_lens[0:batchId]) 是原始TND位置，不取决于后续调度顺序。
// 整条序列的 chunk 递推期间，这四个字段保持不变。
// 任务元数据统一用64位，长度、分块和GM偏移计算不在中途截断。
// 当前接口的 seq_lens 仍为 int32，Nv<=64；类型加宽不改变接口约束。
struct WorkItem {
    uint64_t batchId;
    uint64_t tokenStart;
    uint64_t seqLength;
    uint64_t valueHead;
};

// 任务尺寸计算只依赖显式参数，不读取“当前任务”；行数均为有效 token/状态行数。
__aicore__ inline uint64_t ChunkNum(uint64_t seqLength)
{
    // ceil(L/C)，先除再补尾块，避免 L+C-1 的加法溢出。
    return seqLength / FusedLayout::CHUNK + (seqLength % FusedLayout::CHUNK != 0);
}

// 调用方保证 chunk < ChunkNum(seqLength)；空序列由 Streaming 单独复制初态。
__aicore__ inline uint32_t ChunkValidRows(uint64_t seqLength, uint64_t chunk)
{
    const uint64_t chunkBegin = chunk * FusedLayout::CHUNK;
    const uint64_t remaining = seqLength - chunkBegin;
    // 有效行数<=CHUNK，缩窄后可直接填写搬运/VF参数。
    return static_cast<uint32_t>(remaining < FusedLayout::CHUNK ? remaining : FusedLayout::CHUNK);
}

__aicore__ inline uint32_t SubBlockValidRows(uint32_t validRows, uint32_t subBlockIdx)
{
    const uint32_t rowBegin = subBlockIdx * FusedLayout::HALF_CHUNK;
    if (validRows <= rowBegin) {
        return 0;
    }
    const uint32_t remaining = validRows - rowBegin;
    return remaining < FusedLayout::HALF_CHUNK ? remaining : FusedLayout::HALF_CHUNK;
}

__aicore__ inline uint32_t StateValidRows(uint32_t dv, uint32_t subBlockIdx)
{
    const uint32_t rowBegin = subBlockIdx * FusedLayout::HALF_DIM;
    if (dv <= rowBegin) {
        return 0;
    }
    const uint32_t remaining = dv - rowBegin;
    return remaining < FusedLayout::HALF_DIM ? remaining : FusedLayout::HALF_DIM;
}

// Context 只持有本次 kernel 调用的只读参数，不保存随 Scheduler::Next 改变的任务。
// params 直接从入口地址和 tiling 构造；BasicBlock/Compute 借用只读 Context。
template <typename stateType>
struct KernelContext {
    const KernelParams<stateType> params;

    __aicore__ inline KernelContext(const CGDRInitParams& addresses, const ChunkGatedDeltaRuleTilingData& tiling,
                                    bool hasGamma)
        : params(addresses, tiling, hasGamma)
    {}

    // Q/K:[T,Nk,Dk]，V:[T,Nv,Dv]；每 Nv/Nk 个相邻 value 头共享一个 QK 头。
    __aicore__ inline uint32_t QueryKeyHead(uint64_t valueHead) const
    {
        // host已校验 Nk<=Nv<=64，QK头索引限定在[0,Nk)，可安全使用32位。
        return static_cast<uint32_t>(valueHead / (params.nv / params.nk));
    }

    // S:[B,Nv,Dv,Dk]，偏移单位为元素。初态 batch/head stride 可为 0，以支持广播。
    __aicore__ inline uint64_t InitialStateOffset(uint64_t batchId, uint64_t valueHead, uint32_t valueRowBegin) const
    {
        return batchId * params.stateStride0 + valueHead * params.stateStride1 +
               static_cast<uint64_t>(valueRowBegin) * params.dk;
    }

    // 终态始终连续写回；每个 AIV 独占 [valueRowBegin:valueRowBegin+validRows, :Dk]。
    __aicore__ inline uint64_t FinalStateOffset(uint64_t batchId, uint64_t valueHead, uint32_t valueRowBegin) const
    {
        return (batchId * params.nv + valueHead) * params.dv * params.dk +
               static_cast<uint64_t>(valueRowBegin) * params.dk;
    }
};

} // namespace ChunkGatedDeltaRule
