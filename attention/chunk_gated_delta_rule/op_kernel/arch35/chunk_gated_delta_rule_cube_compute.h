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

#include "utils/chunk_gated_delta_rule_context.h"
#include "utils/chunk_gated_delta_rule_cube_utils.h"
#include "utils/chunk_gated_delta_rule_sync.h"

namespace ChunkGatedDeltaRule {

using namespace AscendC;

// AIC 计算层：处理一个 head 的所有 chunk，组织前缀预取、矩阵乘和 Cube 搬运。
template <typename stateType, bool hasGamma>
class CubeCompute : private FusedLayout {
public:
    __aicore__ inline explicit CubeCompute(const KernelContext<stateType>& context)
        : context_(context)
    {}

    // BasicBlock 已选择融合路径；本函数完成当前 head 的全部 chunk 和跨核握手收尾。
    // 缓冲区绑定及临时矩阵仅在本类内部使用，不向调度层传递。
    __aicore__ inline void Run(const WorkItem& task)
    {
        if (task.seqLength == 0) {
            return;
        }
        const uint32_t qkHead = context_.QueryKeyHead(task.valueHead);
        const uint64_t chunkNum = ChunkNum(task.seqLength);
        Reset();
        NotifyVector<PIPE_MTE2, CubeToVectorEvent::ChunkSlotFree>(0);
        if (chunkNum > 1) {
            NotifyVector<PIPE_MTE2, CubeToVectorEvent::ChunkSlotFree>(1);
        }
        CopyPrefixQkGmToL1(qkHead, 0, task.tokenStart, ChunkValidRows(task.seqLength, 0));
        ComputeKeyKeyAndQueryKeyScores(0);
        for (uint64_t chunk = 0; chunk < chunkNum; ++chunk) {
            ComputeCubeChunk(task, qkHead, chunk);
        }
        // 收回最后一次发布的归还信号，避免计数 token 遗留到同一核组处理的下一个 head。
        WaitForVector<VectorToCubeEvent::StateDeltaBufferFree>();
        WaitForVector<VectorToCubeEvent::PrefixScoresBufferFree>(0);
        if (chunkNum > 1) {
            WaitForVector<VectorToCubeEvent::PrefixScoresBufferFree>(1);
        }
    }

private:
    // L1 完整槽图见 FusedLayout；本类私有区按实际容量衔接：
    // Q:[0,16)KiB -> K:[16,32)KiB -> 公共A:[32,40)KiB -> lowerTimesInverse:[40,42)KiB。
    // lowerTimesInverse 是紧凑 BF16[32,32]，只占2KiB；到 rawKcd 起点48KiB之间的6KiB当前未使用。
    static constexpr uint32_t L1_Q_OFFSET = 0;
    static constexpr uint32_t L1_K_OFFSET = L1_Q_OFFSET + L1_TILE_BYTES;
    static constexpr uint32_t L1_LOWER_TIMES_INVERSE_OFFSET = L1_INVERSE_OFFSET + CHUNK * CHUNK * sizeof(bfloat16_t);

    // L0C FP32 布局（半开区间，KiB）：
    // [  0, 64) 临时结果，M -> FIX；求逆[32,32](4KiB)、前缀/投影(32KiB)、stateDelta[D,D](64KiB)复用
    // [ 64, 96) U -> R:[C,D]，跨 MMAD 保留并原地累加，最终 FIX -> UB
    // [ 96,128) O_inter -> O:[C,D]，跨 MMAD 保留并原地累加，最终 FIX -> GM
    // [128,256) 当前未使用；三个活动区分开，避免临时结果覆盖仍存活的R/O。
    // 起点由前一区域最大shape推导；不是要求按64/96KiB对齐。
    static constexpr uint32_t L0C_RESIDUAL_OFFSET = DIM * DIM * sizeof(float);
    static constexpr uint32_t L0C_OUTPUT_OFFSET = L0C_RESIDUAL_OFFSET + CHUNK * DIM * sizeof(float);

    // L0A/L0B 各64KiB，均分 [0,32)/[32,64) 两个stage，MTE1装入 -> M消费。
    // 每stage最多 BF16[D,D]=32KiB；按MMAD步骤轮换，stage编号不等于chunk槽编号。
    // ComputeResidualAndStateOutput 的两次MMAD共用一份L0B状态，第二次完成才释放。
    static constexpr uint32_t L0AB_STAGES = 2;
    static constexpr uint32_t L0A_STAGE_BYTES = L0A_SIZE / L0AB_STAGES;
    static constexpr uint32_t L0B_STAGE_BYTES = L0B_SIZE / L0AB_STAGES;
    // 固定 ID 直接使用 MutexID；BASE + stage/slot 经整数提升，仅动态组合处显式转换。
    // stage/slot 均为 0/1：L0AB 使用 0/1，QK 使用 5/6。
    static constexpr MutexID AIC_L0AB_MUTEX_BASE = 0;
    static constexpr MutexID AIC_L0C_TMP_MUTEX = 2;
    static constexpr MutexID AIC_L0C_RESIDUAL_MUTEX = 3;
    static constexpr MutexID AIC_L0C_OUTPUT_MUTEX = 4;
    static constexpr MutexID AIC_L1_QK_MUTEX_BASE = 5;
    static constexpr MutexID AIC_L1_FIX_MUTEX = 7;

    static_assert(L1_Q_OFFSET + CHUNK * DIM * sizeof(bfloat16_t) <= L1_K_OFFSET, "AIC Q scratch overlaps K scratch");
    static_assert(L1_K_OFFSET + CHUNK * DIM * sizeof(bfloat16_t) <= L1_INVERSE_OFFSET,
                  "AIC Q/K scratch overlaps the shared inverse matrix");
    static_assert(L1_INVERSE_OFFSET + CHUNK * CHUNK * sizeof(bfloat16_t) <= L1_LOWER_TIMES_INVERSE_OFFSET,
                  "AIC inverse product overlaps the shared inverse matrix");
    static_assert(L1_LOWER_TIMES_INVERSE_OFFSET + HALF_CHUNK * HALF_CHUNK * sizeof(bfloat16_t) <= L1_RAW_KCD_OFFSET,
                  "AIC inverse product overlaps shared rawKcd");
    static_assert(CHUNK * 2 * DIM * sizeof(bfloat16_t) <= L0A_STAGE_BYTES,
                  "arch35 exact L0A tile exceeds one ping-pong stage");
    static_assert(DIM * DIM * sizeof(bfloat16_t) <= L0B_STAGE_BYTES,
                  "arch35 exact L0B tile exceeds one ping-pong stage");
    static_assert(DIM * DIM * sizeof(float) <= L0C_RESIDUAL_OFFSET, "AIC temporary accumulator overlaps residual");
    static_assert(L0C_RESIDUAL_OFFSET + CHUNK * DIM * sizeof(float) <= L0C_OUTPUT_OFFSET,
                  "AIC residual accumulator overlaps output");
    static_assert(L0C_OUTPUT_OFFSET + CHUNK * DIM * sizeof(float) <= L0C_SIZE, "AIC accumulators exceed L0C capacity");
    static_assert(AIC_L1_FIX_MUTEX < MUTEX_COUNT, "arch35 exact AIC mutex layout exceeds its allocation");

    // 一个 chunk 的内部工作集；shape 为补零后的计算范围，不分配缓冲区。
    struct ChunkMatrices {
        LocalTensor<bfloat16_t> inverse;           // A:[C,C]，由 A00/A11/-L10 逐步补齐
        LocalTensor<bfloat16_t> lowerTimesInverse; // (-L10)@A00:[C/2,C/2]
        LocalTensor<bfloat16_t> rawKcd;            // -beta*exp(G)*K:[C,DIM]
        LocalTensor<bfloat16_t> vBeta;             // beta*V:[C,DIM]
        LocalTensor<bfloat16_t> kcdQPrime;         // [Kcd;Q']:[2*C,DIM]，两份矩阵连续存储
        LocalTensor<bfloat16_t> state;             // S:[DIM,DIM]，逻辑方向 [Dv,Dk]
        LocalTensor<bfloat16_t> residualBf16;      // R:[C,DIM] 的 Cube 输入副本
        LocalTensor<bfloat16_t> kg;                // Kg:[C,DIM]
        LocalTensor<bfloat16_t> maskedQkt;         // M:[C,C]
        LocalTensor<float> residual;               // U -> R:[C,DIM]，L0C
        LocalTensor<float> output;                 // O_inter -> O:[C,DIM]，L0C
    };

    __aicore__ inline ChunkMatrices BindChunkMatrices(uint64_t chunk)
    {
        const uint32_t base = (chunk & 1) * L1_SLOT_STRIDE_BYTES;
        return {BufferViews::L1<bfloat16_t>(base + L1_INVERSE_OFFSET),
                BufferViews::L1<bfloat16_t>(base + L1_LOWER_TIMES_INVERSE_OFFSET),
                BufferViews::L1<bfloat16_t>(base + L1_RAW_KCD_OFFSET),
                BufferViews::L1<bfloat16_t>(base + L1_VBETA_OFFSET),
                BufferViews::L1<bfloat16_t>(base + L1_KCD_QPRIME_OFFSET),
                BufferViews::L1<bfloat16_t>(L1_STATE_OFFSET),
                BufferViews::L1<bfloat16_t>(base + L1_RESIDUAL_OFFSET),
                BufferViews::L1<bfloat16_t>(base + L1_KG_OFFSET),
                BufferViews::L1<bfloat16_t>(base + L1_MASKED_QKT_OFFSET),
                BufferViews::L0C<float>(L0C_RESIDUAL_OFFSET),
                BufferViews::L0C<float>(L0C_OUTPUT_OFFSET)};
    }

    __aicore__ inline void ComputeCubeChunk(const WorkItem& task, uint32_t qkHead, uint64_t chunk)
    {
        const auto x = BindChunkMatrices(chunk);
        const uint32_t slot = chunk & 1;
        if (chunk + 1 < ChunkNum(task.seqLength)) {
            // 优化方案：前缀只依赖 Q/K，预取下块并与 AIV 求逆、当前状态递推重叠。
            CopyPrefixQkGmToL1(qkHead, chunk + 1, task.tokenStart + (chunk + 1) * CHUNK,
                               ChunkValidRows(task.seqLength, chunk + 1));
            ComputeKeyKeyAndQueryKeyScores(chunk + 1);
        }

        WaitForVector<VectorToCubeEvent::InverseInputsReady>(slot);
        CompleteBlockLowerInverse(x);    // [4a/b] A10=A11@((-L10)@A00)，接 Vector [3b]。
        ComputeInverseWeightedKeys(x);   // [5a] Kcd=A@rawKcd:[C,Dk]。
        ComputeInverseWeightedValues(x); // [5b] U=A@(beta*V):[C,Dv]，保留在 residual L0C。
        WaitForVector<VectorToCubeEvent::StateReady>();
        ComputeResidualAndStateOutput(x, slot); // [6a/b] R=U+Kcd@S.T；O_inter=Q'@S.T。
        // [6c, Vector] 等待 R32 -> R16 及 UB -> L1；此后两次 MMAD 都消费同一份 R16。
        WaitForVector<VectorToCubeEvent::ResidualReady>();
        ComputeResidualKeyStateDelta(x, chunk); // [7] stateDelta=R16.T@Kg:[Dv,Dk]，交 Vector [9a]。
        WaitForVector<VectorToCubeEvent::CausalScoresReady>(slot);
        AccumulateCausalResidualOutput(x); // [8] O=O_inter+M@R16:[C,Dv]。

        if (chunk + 2 < ChunkNum(task.seqLength)) {
            // L1 操作数已进入 L0，chunk+2 可复用当前槽；独立 L0C/FIX 继续完成 O 的写回。
            NotifyVector<PIPE_MTE1, CubeToVectorEvent::ChunkSlotFree>(slot);
        }
        const uint64_t tokenStart = task.tokenStart + chunk * CHUNK;
        CopyOutputL0CToGm(task.valueHead, tokenStart, ChunkValidRows(task.seqLength, chunk), x.output);
    }

    __aicore__ inline void Reset()
    {
        l0abStage_ = 0;
    }

    struct CubeOperands {
        LocalTensor<bfloat16_t> lhs;
        LocalTensor<bfloat16_t> rhs;
        MutexID mutex;
    };
    __aicore__ inline uint32_t NextL0AbStage()
    {
        const uint32_t stage = l0abStage_;
        l0abStage_ ^= 1;
        return stage;
    }

    __aicore__ inline void CopyStateDeltaL0CToUb(LocalTensor<float> result, uint64_t chunk)
    {
        const MutexID resultMutex = AIC_L0C_TMP_MUTEX;
        // ComputeResidualKeyStateDelta 已结束 M 的使用期；此处成对管理 FIX 读取。
        Mutex::Lock<PIPE_FIX>(resultMutex);
        if (chunk != 0) {
            WaitForVector<VectorToCubeEvent::StateDeltaBufferFree>();
        }
        // FP32 接口直接传 FP32；BF16 接口先量化 stateDelta，再由 AIV 升为 FP32 与主状态累加。
        const uint32_t stateDeltaOffset = STATE_FP32 ? PrefixTransformsUbOffset(chunk & 1) : UB_STATE_DELTA_BF16_OFFSET;
        auto stateDelta = BufferViews::Ub<stateType>(stateDeltaOffset);
        CopyResultRowsL0CToUb(stateDelta, result, DIM, DIM);
        NotifyVector<PIPE_FIX, CubeToVectorEvent::StateDeltaReady>();
        Mutex::Unlock<PIPE_FIX>(resultMutex);
    }

    __aicore__ inline void CopyPrefixScoresL0CToUb(LocalTensor<float> result, uint32_t slot, bool waitReusable)
    {
        const MutexID resultMutex = AIC_L0C_TMP_MUTEX;
        // MatmulKeyKeyAndQueryKeyScores 已持有 PIPE_M；此处只释放一次，再取得 FIX 所有权。
        Mutex::Unlock<PIPE_M>(resultMutex);
        Mutex::Lock<PIPE_FIX>(resultMutex);
        if (waitReusable) {
            WaitForVector<VectorToCubeEvent::PrefixScoresBufferFree>(slot);
        }
        // [K前半,Q前半,K后半,Q后半]@K.T()，每 AIV 收到自己的 KKT/QKT:[C/2,C]。
        auto scores = BufferViews::Ub<bfloat16_t>(PrefixDirectOffset(slot));
        CopyResultRowsL0CToUb(scores, result, CHUNK * 2, CHUNK);
        NotifyVector<PIPE_FIX, CubeToVectorEvent::PrefixScoresReady>(slot);
        Mutex::Unlock<PIPE_FIX>(resultMutex);
    }

    __aicore__ inline void CopyResultL0CToL1(LocalTensor<bfloat16_t> dst, LocalTensor<float> l0c, uint32_t m,
                                             uint32_t n, uint32_t fullRows, MutexID l0cMutex, uint32_t rowStart = 0,
                                             uint32_t colStart = 0)
    {
        // 顺序不可交换：先占用 L1 目标，再结束 MMAD，最后允许 FIX 读取 L0C。
        Mutex::Lock<PIPE_FIX>(AIC_L1_FIX_MUTEX);
        Mutex::Unlock<PIPE_M>(l0cMutex);
        Mutex::Lock<PIPE_FIX>(l0cMutex);
        CopyTileL0CToL1(dst[GetZnOffset(rowStart, colStart, fullRows)], l0c, m, n, fullRows);
        Mutex::Unlock<PIPE_FIX>(l0cMutex);
        Mutex::Unlock<PIPE_FIX>(AIC_L1_FIX_MUTEX);
    }

    __aicore__ inline void CopyOutputL0CToGm(uint64_t valueHead, uint64_t tokenStart, uint32_t validRows,
                                             LocalTensor<float> l0c)
    {
        const MutexID outputMutex = AIC_L0C_OUTPUT_MUTEX;
        Mutex::Unlock<PIPE_M>(outputMutex);
        Mutex::Lock<PIPE_FIX>(outputMutex);
        const uint64_t outputOffset =
            tokenStart * context_.params.nv * context_.params.dv + valueHead * context_.params.dv;
        CopyRowsL0CToGm(context_.params.out[outputOffset], l0c, validRows, context_.params.dv, CHUNK,
                        context_.params.nv * context_.params.dv);
        Mutex::Unlock<PIPE_FIX>(outputMutex);
    }

    // 只绑定下一组 L0A/L0B 双缓冲；搬运与计算的 Lock/Unlock 由各阶段显式安排。
    __aicore__ inline CubeOperands NextOperands()
    {
        const uint32_t stage = NextL0AbStage();
        return {BufferViews::L0A<bfloat16_t>(stage * L0A_STAGE_BYTES),
                BufferViews::L0B<bfloat16_t>(stage * L0B_STAGE_BYTES),
                static_cast<MutexID>(AIC_L0AB_MUTEX_BASE + stage)};
    }

    // [5a, Cube] Kcd=A@rawKcd:[C,Dk]。Kcd 已带负号，是旧状态对 R 的贡献系数。
    // 结果以 BF16 写入 kcdQPrime 前 C 行，后 C 行的 Q' 由 Vector 准备。
    // rawKcd 来自 Vector [2b]，A 由 Vector [3b] + Cube [4] 完成；Kcd 经 BF16 FIX 交 [6a]。
    __aicore__ inline void ComputeInverseWeightedKeys(const ChunkMatrices& x)
    {
        const auto operands = NextOperands();
        auto result = BufferViews::L0C<float>(0);
        const MutexID resultMutex = AIC_L0C_TMP_MUTEX;
        Mutex::Lock<PIPE_MTE1>(operands.mutex);
        // CompleteBlockLowerInverse 的 FIX 刚写回 A；在读取该 L1 区域前等待它完成。
        Mutex::Lock<PIPE_MTE1>(AIC_L1_FIX_MUTEX);
        CopyL1ToL0<false>(operands.lhs, x.inverse, CHUNK, CHUNK, CHUNK);
        CopyL1ToL0<true>(operands.rhs, x.rawKcd, CHUNK, DIM, CHUNK);
        Mutex::Unlock<PIPE_MTE1>(AIC_L1_FIX_MUTEX);
        Mutex::Unlock<PIPE_MTE1>(operands.mutex);

        Mutex::Lock<PIPE_M>(operands.mutex);
        Mutex::Lock<PIPE_M>(resultMutex);
        Mmad(result, operands.lhs, operands.rhs, CHUNK, DIM, CHUNK);
        Mutex::Unlock<PIPE_M>(operands.mutex);
        // CopyResultL0CToL1 释放 result 的 M 锁，再交给 FIX；保留此次独立 BF16 舍入。
        CopyResultL0CToL1(x.kcdQPrime, result, CHUNK, DIM, DIM, resultMutex);
    }

    // [5b, Cube] U=A@(beta*V)：[C,C] @ [C,Dv] -> [C,Dv]，物理 [C,DIM]。
    // x.residual 此时存 U，尚不是 R；[6a] 原地叠加 Kcd@S.T 后才成为 R，避免额外 L0C 副本。
    __aicore__ inline void ComputeInverseWeightedValues(const ChunkMatrices& x)
    {
        const auto operands = NextOperands();
        // A 已在上一阶段读过；beta*V 的就绪由 ComputeCubeChunk 的 InverseInputsReady 保证。
        Mutex::Lock<PIPE_MTE1>(operands.mutex);
        CopyL1ToL0<false>(operands.lhs, x.inverse, CHUNK, CHUNK, CHUNK);
        CopyL1ToL0<true>(operands.rhs, x.vBeta, CHUNK, DIM, CHUNK);
        Mutex::Unlock<PIPE_MTE1>(operands.mutex);

        Mutex::Lock<PIPE_M>(operands.mutex);
        Mutex::Lock<PIPE_M>(AIC_L0C_RESIDUAL_MUTEX);
        Mmad(x.residual, operands.lhs, operands.rhs, CHUNK, DIM, CHUNK);
        Mutex::Unlock<PIPE_M>(operands.mutex);
        // 保留 residual 的 M 锁：ComputeResidualAndStateOutput 原地累加，CopyResidualL0CToUb 最终释放。
    }

    // [8, Cube] O=O_inter+M@R16：[C,C] @ [C,Dv] -> [C,Dv]。
    // x.output 由 [6b] 的 O_inter 原地变为 O；沿用其 FP32 L0C 和 M 锁，不重新初始化累加器。
    __aicore__ inline void AccumulateCausalResidualOutput(const ChunkMatrices& x)
    {
        const auto operands = NextOperands();
        // ComputeCubeChunk 已分别等待 Vector 的 CausalScoresReady 与 ResidualReady。
        Mutex::Lock<PIPE_MTE1>(operands.mutex);
        CopyL1ToL0<false>(operands.lhs, x.maskedQkt, CHUNK, CHUNK, CHUNK);
        CopyL1ToL0<true>(operands.rhs, x.residualBf16, CHUNK, DIM, CHUNK);
        Mutex::Unlock<PIPE_MTE1>(operands.mutex);

        Mutex::Lock<PIPE_M>(operands.mutex);
        Mmad(x.output, operands.lhs, operands.rhs, CHUNK, DIM, CHUNK, false);
        Mutex::Unlock<PIPE_M>(operands.mutex);
        // CopyOutputL0CToGm 负责释放 output 的 M 锁并取得 FIX 锁。
    }

    // [7, Cube -> Vector] stateDelta=R16.T@Kg：[Dv,C] @ [C,Dk] -> [Dv,Dk]，物理 [DIM,DIM]。
    // R16 来自 Vector [6c]，Kg 来自 [2b]；CopyStateDeltaL0CToUb + StateDeltaReady 交接到 [9a]。
    __aicore__ inline void ComputeResidualKeyStateDelta(const ChunkMatrices& x, uint64_t chunk)
    {
        const auto operands = NextOperands();
        auto result = BufferViews::L0C<float>(0);
        const MutexID resultMutex = AIC_L0C_TMP_MUTEX;
        Mutex::Lock<PIPE_MTE1>(operands.mutex);
        // 沿用 L1/FIX 缓冲区交接协议；R 的跨核就绪由 ComputeCubeChunk 单独等待。
        Mutex::Lock<PIPE_MTE1>(AIC_L1_FIX_MUTEX);
        // A 的 true 表示 R.T()；B 的 true 是 NZ -> L0B 的装载方式，Kg 数学上不转置。
        CopyL1ToL0<true>(operands.lhs, x.residualBf16, CHUNK, DIM, CHUNK);
        CopyL1ToL0<true>(operands.rhs, x.kg, CHUNK, DIM, CHUNK);
        Mutex::Unlock<PIPE_MTE1>(AIC_L1_FIX_MUTEX);
        Mutex::Unlock<PIPE_MTE1>(operands.mutex);

        Mutex::Lock<PIPE_M>(operands.mutex);
        Mutex::Lock<PIPE_M>(resultMutex);
        Mmad(result, operands.lhs, operands.rhs, DIM, DIM, CHUNK);
        Mutex::Unlock<PIPE_M>(operands.mutex);
        Mutex::Unlock<PIPE_M>(resultMutex);
        CopyStateDeltaL0CToUb(result, chunk); // FIX 分发给两个 AIV。
    }

    __aicore__ inline void CopyResidualL0CToUb(LocalTensor<float> l0cResidual, uint32_t slot)
    {
        auto residual = BufferViews::Ub<float>(PrefixTransformsUbOffset(slot));
        const MutexID residualMutex = AIC_L0C_RESIDUAL_MUTEX;
        // ComputeInverseWeightedValues 初始化 U 后持续持有 PIPE_M；ComputeResidualAndStateOutput 累加
        // Kcd@S.T()，在此交给 FIX。
        Mutex::Unlock<PIPE_M>(residualMutex);
        Mutex::Lock<PIPE_FIX>(residualMutex);
        // R32 保留 16 列分块布局，一条 FIX 按 M 平分给两个 AIV。
        // 每个 AIV 收到 [DIM/16, HALF_CHUNK, 16]，随后仅转一个 BF16 副本送回 L1。
        CopyResultNzL0CToUb(residual, l0cResidual, CHUNK, DIM);
        NotifyVector<PIPE_FIX, CubeToVectorEvent::ResidualReady>();
        Mutex::Unlock<PIPE_FIX>(residualMutex);
    }

    // [6a/6b, Cube] 两个明确的数学结果共享同一个 S:[Dv,Dk]；因此保留在同一函数中管理 L0B 锁。
    // 结果 R32 经 FIX -> Vector [6c]；O_inter 留 L0C，等待 [8] 累加 M@R16。
    __aicore__ inline void ComputeResidualAndStateOutput(const ChunkMatrices& matrices, uint32_t slot)
    {
        auto l0cResidual = matrices.residual;
        auto l0cOutput = matrices.output;
        auto l1KcdQPrime = matrices.kcdQPrime;
        auto l1State = matrices.state;
        // 优化方案：两次 MMAD 共用 L0B 中的 S:[Dv,Dk]，减少一次状态装载。
        // 输入 l0cResidual=U:[C,Dv]，原地累加为 R=U+Kcd*S^T；l0cOutput=Q'*S^T:[C,Dv]。
        // STATE_FREE 在 L1->L0B 完成后即可发出；L0B 的 Mutex 则保持到两次矩阵乘均结束。
        const uint32_t stateStage = NextL0AbStage();
        const uint32_t outputStage = NextL0AbStage();
        const MutexID stateMutex = static_cast<MutexID>(AIC_L0AB_MUTEX_BASE + stateStage);
        const MutexID outputMutex = static_cast<MutexID>(AIC_L0AB_MUTEX_BASE + outputStage);
        auto l0aKcd = BufferViews::L0A<bfloat16_t>(stateStage * L0A_STAGE_BYTES);
        auto l0aQPrime = BufferViews::L0A<bfloat16_t>(outputStage * L0A_STAGE_BYTES);
        auto l0bState = BufferViews::L0B<bfloat16_t>(stateStage * L0B_STAGE_BYTES);

        Mutex::Lock<PIPE_MTE1>(stateMutex);
        Mutex::Lock<PIPE_MTE1>(AIC_L1_FIX_MUTEX);
        CopyL1ToL0<false>(l0aKcd, l1KcdQPrime, CHUNK, DIM, DIM, 0, 0);
        CopyL1ToL0<false>(l0bState, l1State, DIM, DIM, DIM);
        Mutex::Unlock<PIPE_MTE1>(AIC_L1_FIX_MUTEX);
        NotifyVector<PIPE_MTE1, CubeToVectorEvent::StateBufferFree>();
        Mutex::Unlock<PIPE_MTE1>(stateMutex);
        Mutex::Lock<PIPE_M>(stateMutex);
        // [6a] R=U+Kcd@S.T；累加到 [5b] 的 U，不清零 L0C。
        Mmad(l0cResidual, l0aKcd, l0bState, CHUNK, DIM, DIM, false);
        CopyResidualL0CToUb(l0cResidual, slot);

        Mutex::Lock<PIPE_MTE1>(outputMutex);
        CopyL1ToL0<false>(l0aQPrime, l1KcdQPrime, CHUNK, DIM, DIM, CHUNK, 0);
        Mutex::Unlock<PIPE_MTE1>(outputMutex);
        Mutex::Lock<PIPE_M>(outputMutex);
        Mutex::Lock<PIPE_M>(AIC_L0C_OUTPUT_MUTEX);
        // [6b] O_inter=Q'@S.T；新的输出累加器，S 的 L0B 副本仍来自 [6a]。
        Mmad(l0cOutput, l0aQPrime, l0bState, CHUNK, DIM, DIM, true);
        // 两次 MMAD 均完成后，先归还共享 S 所在的槽，再归还 output 操作数所在的槽。
        Mutex::Unlock<PIPE_M>(stateMutex);
        Mutex::Unlock<PIPE_M>(outputMutex);
        // output 的 M 锁继续保留，供 AccumulateCausalResidualOutput 累加，最后由 CopyOutputL0CToGm 释放。
    }

    // [1, Cube -> Vector] 计算 KKT/QKT:[C,C]；FIX -> UB 后以 PrefixScoresReady 交给 Vector [3a]。
    __aicore__ inline void MatmulKeyKeyAndQueryKeyScores(LocalTensor<bfloat16_t> l1Q, LocalTensor<bfloat16_t> l1K,
                                                         uint32_t prefixSlot, bool waitDirectFree)
    {
        // 优化方案：L0A 按 [K前32行,Q前32行,K后32行,Q后32行] 拼接，一次 [2*C,DIM]*K^T
        // 同时得到 KKT/QKT:[2*C,C]，两个分数矩阵各为 [C,C]。
        // 结果沿 M 平分后，每个 AIV 恰好收到自己 32 个 token 的 KKT 和 QKT，复用同一份 L0B 的 K。
        const uint32_t stage = NextL0AbStage();
        const MutexID l0abMutex = static_cast<MutexID>(AIC_L0AB_MUTEX_BASE + stage);
        const MutexID qkMutex = static_cast<MutexID>(AIC_L1_QK_MUTEX_BASE + prefixSlot);
        auto l0a = BufferViews::L0A<bfloat16_t>(stage * L0A_STAGE_BYTES);
        auto l0b = BufferViews::L0B<bfloat16_t>(stage * L0B_STAGE_BYTES);
        auto l0c = BufferViews::L0C<float>(0);

        Mutex::Lock<PIPE_MTE1>(l0abMutex);
        Mutex::Lock<PIPE_MTE1>(qkMutex);
        CopyRowsL1ToL0A(l0a, l1K, HALF_CHUNK, DIM, CHUNK, 0, CHUNK * 2, 0);
        CopyRowsL1ToL0A(l0a, l1Q, HALF_CHUNK, DIM, CHUNK, 0, CHUNK * 2, HALF_CHUNK);
        CopyRowsL1ToL0A(l0a, l1K, HALF_CHUNK, DIM, CHUNK, HALF_CHUNK, CHUNK * 2, CHUNK);
        CopyRowsL1ToL0A(l0a, l1Q, HALF_CHUNK, DIM, CHUNK, HALF_CHUNK, CHUNK * 2, CHUNK + HALF_CHUNK);
        CopyL1ToL0<false>(l0b, l1K, CHUNK, DIM, CHUNK);
        Mutex::Unlock<PIPE_MTE1>(qkMutex);
        Mutex::Unlock<PIPE_MTE1>(l0abMutex);
        Mutex::Lock<PIPE_M>(l0abMutex);
        Mutex::Lock<PIPE_M>(AIC_L0C_TMP_MUTEX);
        Mmad(l0c, l0a, l0b, CHUNK * 2, CHUNK, DIM);
        Mutex::Unlock<PIPE_M>(l0abMutex);
        CopyPrefixScoresL0CToUb(l0c, prefixSlot, waitDirectFree);
    }

    __aicore__ inline void CopyPrefixQkGmToL1(uint32_t qkHead, uint64_t chunk, uint64_t tokenStart, uint32_t validRows)
    {
        const uint32_t slot = chunk & 1;
        const uint32_t slotBase = slot * L1_SLOT_STRIDE_BYTES;
        const uint64_t qkOffset = tokenStart * context_.params.nk * context_.params.dk + qkHead * context_.params.dk;
        auto l1Q = BufferViews::L1<bfloat16_t>(slotBase + L1_Q_OFFSET);
        auto l1K = BufferViews::L1<bfloat16_t>(slotBase + L1_K_OFFSET);

        const MutexID qkMutex = static_cast<MutexID>(AIC_L1_QK_MUTEX_BASE + slot);
        Mutex::Lock<PIPE_MTE2>(qkMutex);
        if (validRows < CHUNK || context_.params.dk < DIM) {
            // 先清完整物理 Q/K，再覆盖有效矩形；尾部零值保证 padding token 不影响分数和状态。
            constexpr uint16_t L1_QK_BLOCKS = CHUNK * DIM * sizeof(bfloat16_t) / BLOCK_SIZE;
            InitConstValueParams<bfloat16_t> zeroParams{1, L1_QK_BLOCKS, 0, static_cast<bfloat16_t>(0)};
            Fill(l1Q, zeroParams);
            Fill(l1K, zeroParams);
        }
        CopyGmToL1(context_.params.query[qkOffset], l1Q, validRows, context_.params.dk,
                   context_.params.nk * context_.params.dk, CHUNK);
        CopyGmToL1(context_.params.key[qkOffset], l1K, validRows, context_.params.dk,
                   context_.params.nk * context_.params.dk, CHUNK);
        Mutex::Unlock<PIPE_MTE2>(qkMutex);
    }

    // [1] 只依赖本块 Q/K，可在当前块状态递推期间提前计算下一块；数学上仍是下一块的分数。
    __aicore__ inline void ComputeKeyKeyAndQueryKeyScores(uint64_t chunk)
    {
        const uint32_t slot = chunk & 1;
        const uint32_t slotBase = slot * L1_SLOT_STRIDE_BYTES;
        auto l1Q = BufferViews::L1<bfloat16_t>(slotBase + L1_Q_OFFSET);
        auto l1K = BufferViews::L1<bfloat16_t>(slotBase + L1_K_OFFSET);

        MatmulKeyKeyAndQueryKeyScores(l1Q, l1K, slot, chunk >= 2);
    }

    // [4, Cube] 接 Vector [3b]：A=(I+L)^(-1)，两个 [32,32] 对角块 A00/A11 已在 L1。
    // x.inverse 的左下块暂存 -L10；本步骤结束后才是完整 A，右上块保持 0。
    // A10 = A11 @ ((-L10) @ A00)。两次矩乘分别写回 BF16，不能合并而改变舍入点。
    __aicore__ inline void CompleteBlockLowerInverse(const ChunkMatrices& x)
    {
        auto result = BufferViews::L0C<float>(0);
        const MutexID resultMutex = AIC_L0C_TMP_MUTEX;
        {
            // [4a] tmp=(-L10)@A00:[32,32]；FIX 为 BF16 写回 lowerTimesInverse。
            const auto operands = NextOperands();
            // 输入由 ComputeCubeChunk 的 InverseInputsReady 保证，此处没有前置 FIX 写回。
            Mutex::Lock<PIPE_MTE1>(operands.mutex);
            // 子块 shape 均为 [32,32]，但父矩阵 NZ 的物理行跨度仍为 C=64。
            CopyL1ToL0<false>(operands.lhs, x.inverse, HALF_CHUNK, HALF_CHUNK, CHUNK, HALF_CHUNK, 0);
            CopyL1ToL0<true>(operands.rhs, x.inverse, HALF_CHUNK, HALF_CHUNK, CHUNK, 0, 0);
            Mutex::Unlock<PIPE_MTE1>(operands.mutex);
            Mutex::Lock<PIPE_M>(operands.mutex);
            Mutex::Lock<PIPE_M>(resultMutex);
            Mmad(result, operands.lhs, operands.rhs, HALF_CHUNK, HALF_CHUNK, HALF_CHUNK);
            Mutex::Unlock<PIPE_M>(operands.mutex);
            CopyResultL0CToL1(x.lowerTimesInverse, result, HALF_CHUNK, HALF_CHUNK, HALF_CHUNK, resultMutex);
        }
        {
            // [4b] A10=A11@tmp:[32,32]；FIX 为 BF16 覆盖原先 -L10，完成完整逆矩阵。
            const auto operands = NextOperands();
            Mutex::Lock<PIPE_MTE1>(operands.mutex);
            // 等待上一步 FIX 写完 lowerTimesInverse，再加载 A11 和这个紧凑 [32,32] 中间积。
            Mutex::Lock<PIPE_MTE1>(AIC_L1_FIX_MUTEX);
            CopyL1ToL0<false>(operands.lhs, x.inverse, HALF_CHUNK, HALF_CHUNK, CHUNK, HALF_CHUNK, HALF_CHUNK);
            CopyL1ToL0<true>(operands.rhs, x.lowerTimesInverse, HALF_CHUNK, HALF_CHUNK, HALF_CHUNK);
            Mutex::Unlock<PIPE_MTE1>(AIC_L1_FIX_MUTEX);
            Mutex::Unlock<PIPE_MTE1>(operands.mutex);
            Mutex::Lock<PIPE_M>(operands.mutex);
            Mutex::Lock<PIPE_M>(resultMutex);
            Mmad(result, operands.lhs, operands.rhs, HALF_CHUNK, HALF_CHUNK, HALF_CHUNK);
            Mutex::Unlock<PIPE_M>(operands.mutex);
            // A10 覆盖原来的 -L10；目标仍保留父矩阵的 NZ 行跨度。
            CopyResultL0CToL1(x.inverse, result, HALF_CHUNK, HALF_CHUNK, CHUNK, resultMutex, HALF_CHUNK, 0);
        }
    }

    static constexpr bool STATE_FP32 = std::is_same_v<stateType, float>;
    const KernelContext<stateType>& context_;
    uint32_t l0abStage_;
};

} // namespace ChunkGatedDeltaRule
