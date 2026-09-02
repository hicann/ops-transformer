/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MINIMAX_SPARSE_ATTENTION_SPLIT_KV_KERNEL_A2_H
#define MINIMAX_SPARSE_ATTENTION_SPLIT_KV_KERNEL_A2_H

// Define dump switches before kernel_utils.hpp includes the block components.
// Keep this probe P-only so multiple large DumpTensor records cannot overrun
// the A2 device printf ring.
#ifndef KERNEL_DUMP
#define KERNEL_DUMP 0
#endif

#ifndef KERNEL_DUMP_SCORE
#define KERNEL_DUMP_SCORE 0
#endif

#ifndef KERNEL_DUMP_CORE
#define KERNEL_DUMP_CORE 9U
#endif

#ifndef KERNEL_DUMP_SUBBLOCK
#define KERNEL_DUMP_SUBBLOCK 1U
#endif

#ifndef KERNEL_DUMP_BI
#define KERNEL_DUMP_BI 0U
#endif

#ifndef KERNEL_DUMP_P
#define KERNEL_DUMP_P 0
#endif

#ifndef KERNEL_DUMP_PHASE1_ROWSUM
#define KERNEL_DUMP_PHASE1_ROWSUM 0
#endif

#ifndef KERNEL_DUMP_OPARTIAL
#define KERNEL_DUMP_OPARTIAL 0
#endif

#ifndef KERNEL_DUMP_OPARTIAL_Q_LIMIT
#define KERNEL_DUMP_OPARTIAL_Q_LIMIT 4U
#endif

// Temporary, scalar-only pipeline markers for A2 deadlock diagnosis.
// Keep this off in normal builds; the marker values are never consumed.
#ifndef KERNEL_DUMP_SYNC_TRACE
#define KERNEL_DUMP_SYNC_TRACE 0
#endif

#ifndef DEBUG_MODE
#define DEBUG_MODE 0
#endif

#include "msa_split_kv_kernel_utils_a2.hpp"
#include "../msa_split_kv_kernel_common.hpp"

// KERNEL_DUMP: 1=enable AscendC::DumpTensor for intermediate results
//              0=disable (normal production)

// Device printf is disabled by default on A2.  The runtime's printf parser
// cannot safely consume the mixed-width debug records used by this kernel and
// may corrupt the printf ring before DumpTensor records are emitted.  Host
// tiling logs and DumpTensor remain enabled for deterministic comparisons.
#ifndef A2_DEVICE_PRINTF
#define A2_DEVICE_PRINTF 0
#endif
#if A2_DEVICE_PRINTF
#define A2_PRINTF(...) printf(__VA_ARGS__)
#else
#define A2_PRINTF(...) \
    do { \
    } while (0)
#endif

using namespace NpuArch;
using namespace tla;
using MinimaxSaSplitKvKernelA2::MinimaxSaSplitKvKernelParamsA2;

namespace MinimaxSaSplitKvKernelA2 {

/*
 * =========================================================================================
 * MinimaxSaSplitKvRegularKernelA2 —— AtlasA2 平台 Sparse Attention Prefill 核心 Kernel
 * =========================================================================================
 *
 * 【一句话概括】
 *   这是稀疏注意力 (Sparse Attention) 的 Prefill 阶段核心 kernel。
 *   输入: Q(查询)、K(键)、V(值)、block_table(KV块映射)、CSR反向索引(哪些Q选择了哪些KV块)
 *   输出: O(注意力结果) = softmax(Q × K^T / √d) × V
 *   特点: 只计算 "被选中" 的 KV block (稀疏), 跳过不相关的 KV, 大幅减少计算量。
 *
 * 【什么是 Prefill?】
 *   LLM 推理分两阶段:
 *     - Prefill (本 kernel): 处理用户输入的 prompt, 一次性计算所有 token 的 KV cache
 *     - Decode: 逐 token 生成, 每次只用上一步的 KV cache
 *   Prefill 特点: Q 有很多 token (整个 prompt), 计算量大, 适合 Cube (矩阵乘) 并行
 *
 * 【什么是 KV-Gather-Q? (核心设计)】
 *   传统 Attention: 遍历每个 Q token, 每次加载对应的 KV block → KV 被重复加载多次
 *   本算子反向遍历: 遍历每个 KV block (只加载一次到 L1 缓存),
 *   然后通过 CSR 反向索引 "gather" (收集) 所有选择了该 KV block 的 Q token 做计算。
 *
 *   举例: 假设有 3 个 KV block, Q token 0/3/5 选择了 block 0, Q token 1/2 选择了 block 1...
 *   传统方式: Q0→加载block0, Q1→加载block1, Q2→加载block1(重复!), Q3→加载block0(重复!)...
 *   本算子:   block0→加载一次→计算 Q0,Q3,Q5; block1→加载一次→计算 Q1,Q2; ...
 *   优势: 每个 KV block 只从 GM 加载到 L1 一次, 大幅减少 GM 带宽消耗
 *
 * 【两阶段架构 (Phase1 + Phase2)】
 *   Phase1 (KV-centric partial compute):
 *     Cube 负责矩阵乘 (QK 和 PV), Vector 负责 softmax
 *     每个 KV block 产出一份 "部分结果" O_partial 写到 workspace
 *     同时记录 softmax 的 rowMax 和 rowSum (用于 Phase2 归一化)
 *
 *   Phase2 (FlashDecode-style combine):
 *     仅 Vector 执行, Cube 空转
 *     对每个 Q token, 合并 topK 个部分结果 → 最终输出 O
 *     合并公式: O = Σ_k scale[k] × (O_partial[k] / rowSum[k])
 *              其中 scale[k] = rowSum[k] × exp(rowMax[k] - globalMax) / Σ(rowSum × exp(...))
 *
 * 【模板参数】(由 kernel_interface.cpp 组装传入, 可理解为 "可插拔组件")
 *   BlockMmadQK           — Cube 侧 QK matmul 组件, 计算 S = Q × K^T (注意力分数)
 *   EpilogueOnlineSoftmax  — Vector 侧 online softmax 组件, 计算 P = exp(S - rowMax) (概率)
 *   BlockMmadPV           — Cube 侧 PV matmul 组件, 计算 O_partial = P × V (加权值)
 *   EpilogueRescaleO      — Vector 侧 Phase2 combine 组件, 合并 topK 结果 → O (最终输出)
 *
 * 【混合核模式 (KERNEL_TYPE_MIX_AIC_1_2)】
 *   A5 使用 1 AIC : 2 AIV 混合核模式:
 *     - AIC (Cube 核心): 执行矩阵乘 (MMAD), 擅长密集计算
 *     - AIV (Vector 核心): 执行向量运算 (softmax, cast 等), 每个 AIC 配 2 个 AIV
 *   CUBE 和 VEC 通过 cross-core flag 同步, 协同完成 Phase1 的三级流水线
 *
 * 【Phase1 任务划分】
 *   taskIdx = packedRow * kvHeads + kvHeadIdx
 *   - packedRow: MSA interleaved 打包的 KV block 行号 (跨 batch 交错排列)
 *   - kvHeadIdx: KV head 索引
 *   分核: for taskIdx = coreIdx; taskIdx < totalTaskNumP1; taskIdx += coreNum
 *   (每个核处理 stride=coreNum 的任务, 充分利用所有核)
 *
 * 【Phase2 任务划分】
 *   taskIdx = qToken * kvHeads + kvHeadIdx
 *   分核: for taskIdx = coreIdx; taskIdx < totalTaskNumP2; taskIdx += coreNum
 *   仅 VEC 执行, CUBE 空转 (Phase2 是纯向量运算, 不需要矩阵乘)
 *
 * 【Workspace 布局】 (三块独立 buffer, 顺序固定, kernel 侧按偏移读取)
 *   ┌─────────────────────────────────────────────────────────────────┐
 *   │ libapiWorkspace (libapiSize_)    ← 框架内部使用, 算子无需关心      │
 *   ├─────────────────────────────────────────────────────────────────┤
 *   │ accumOut    [totalQ × kvHeads × topK × groupSize × D]           │
 *   │   存储 Phase1 的部分结果 O_partial                                │
 *   │   fp32 (innerPrecise≠1) 或 bf16 (innerPrecise==1, 省一半空间)    │
 *   ├─────────────────────────────────────────────────────────────────┤
 *   │ softmaxMax  [totalQ × kvHeads × topK × groupSize] fp32          │
 *   │   每 slot 的 rowMax (online softmax 的最大值, 用于 Phase2 归一化)  │
 *   ├─────────────────────────────────────────────────────────────────┤
 *   │ softmaxSum  [totalQ × kvHeads × topK × groupSize] fp32          │
 *   │   每 slot 的 rowSum (online softmax 的累加和, 用于 Phase2 归一化)  │
 *   └─────────────────────────────────────────────────────────────────┘
 *   A5 无 GM S 中转 workspace (L0C→UB 直接 Fixpipe, 这是 A5 的硬件优势)
 *
 * 【三级流水线设计 (PRE_LAUNCH=2)】
 *   Phase1 的核心是 QK → softmax → PV 三级流水线:
 *
 *   时间 →    batch 0     batch 1     batch 2     batch 3     batch 4
 *   ┌──────┬──────────┬──────────┬──────────┬──────────┬──────────┐
 *   │ QK   │ QK(b0)   │ QK(b1)   │ QK(b2)   │ QK(b3)   │          │
 *   ├──────┼──────────┼──────────┼──────────┼──────────┼──────────┤
 *   │ SM   │          │ SM(b0)   │ SM(b1)   │ SM(b2)   │ SM(b3)   │
 *   ├──────┼──────────┼──────────┼──────────┼──────────┼──────────┤
 *   │ PV   │          │          │ PV(b0)   │ PV(b1)   │ PV(b2)   │
 *   └──────┴──────────┴──────────┴──────────┴──────────┴──────────┘
 *
 *   - QK (Cube): 计算 S = Q × K^T, 结果 fixpipe 到 UB
 *   - SM (Vector): 从 UB 读 S, 做 online softmax, 生成 P 写到 L1
 *   - PV (Cube): 从 L1 读 P, 计算 O_partial = P × V, 写到 GM workspace
 *
 *   PRE_LAUNCH=2 表示 QK 领先 PV 2 个 batch:
 *     - batch bi: QK 计算第 bi 批
 *     - batch bi: softmax 计算第 (bi-1) 批 (滞后 1 个 batch)
 *     - batch bi: PV 计算第 (bi-2) 批 (滞后 2 个 batch = PRE_LAUNCH)
 *   通过 ring buffer (多级缓冲) + cross-core flag (跨核同步信号) 实现流水线
 *
 * 【CSR 反向索引 (KV→Q) 说明】
 *   传统: Q→KV (每个 Q 去找 KV), 需要遍历所有 Q
 *   本算子: KV→Q (每个 KV block 去找选了它的 Q), 用 CSR 格式存储:
 *     k2qRowPtr [kvHeads, totalKvRows+1]: 行指针, 标记每行的起止位置
 *     k2qQIndices [kvHeads, maxNnz]: 列数据, 存储选择该 KV block 的 Q token id
 *     k2qSlotIndices [kvHeads, maxNnz]: 存储该 Q token 在 topK 中的 slot 编号
 *   读取方式 (类似 CSR 稀疏矩阵):
 *     csrStart = k2qRowPtr[kvHeadIdx, packedRow]
 *     csrEnd   = k2qRowPtr[kvHeadIdx, packedRow + 1]
 *     qTokens  = k2qQIndices[csrStart : csrEnd]  (选择该 block 的所有 Q token)
 * =========================================================================================
 */
template <class BlockMmadQK, class EpilogueOnlineSoftmax, class BlockMmadPV, class EpilogueRescaleO>
class MinimaxSaSplitKvRegularKernelA2 {
public:
    using ArchTag = typename BlockMmadPV::ArchTag;

    // === 数据类型定义 ===
    // 均由 GEMM Block 的模板参数推导, 保证 QK/PV/softmax 组件类型一致
    using ElementQ = typename BlockMmadQK::ElementQ; // Q 输入类型 (bf16, 从 GM 读取)
    using ElementK = typename BlockMmadQK::ElementK; // K 输入类型 (bf16, 从 GM 读取到 L1 常驻)
    using ElementS = typename BlockMmadQK::ElementS; // Score S = Q×K^T 的类型 (Cube fixpipe 到 GM)
    using ElementP = typename BlockMmadPV::ElementP; // P = softmax(S) 的类型 (bf16, VEC 写到 L1)
    using ElementV = typename BlockMmadPV::ElementV; // V 输入类型 (bf16, 从 GM 读取到 L1 常驻)
    using ElementO = typename BlockMmadQK::ElementQ; // 最终输出 O 类型 (bf16, 写到 GM)
    using ElementOTmp = float;                       // 中间累加类型 (fp32, Cube L0C 内部累加)
    // O_partial (workspace accumOut) dtype: 跟随 PV block 的 ElementO (= REDtype)
    //   float      → fp32 路径 (innerPrecise≠1): 精度高, workspace 大
    //   bfloat16_t → bf16 路径 (innerPrecise==1): 精度略低, 节省一半 workspace
    using ElementWorkspaceO = typename BlockMmadPV::ElementO;

    // 内存布局: K 按列存储 (转置), 其余按行存储
    using LayoutK = layout::ColumnMajor;
    using LayoutV = layout::RowMajor;
    using LayoutQ = layout::RowMajor;
    using LayoutS = layout::RowMajor;
    using LayoutP = layout::RowMajor;
    using LayoutO = layout::RowMajor;

    // === 流水线参数 ===
    // PRE_LAUNCH=2: QK 领先 PV 2 个 batch, 形成三级流水 (QK→softmax→PV)
    //   batch bi: QK 计算 bi, softmax 计算 bi-1, PV 计算 bi-2
    //   这样 QK/softmax/PV 可以同时执行不同 batch 的计算, 提高硬件利用率
    static constexpr uint32_t PRE_LAUNCH = 2U;
    // 跨核 ring buffer 级数 = PRE_LAUNCH+1 = 3: 需要足够多级保证流水线不冲突
    //   QK 写 stage 0, softmax 读 stage 0 写 stage 1, PV 读 stage 2 (3级轮转)
    static constexpr uint32_t MAX_CROSS_CORE_BUF_STAGES = PRE_LAUNCH + 1U; // 跨核 ring buffer 级数
    // UB S buffer 双缓冲: QK fixpipe 和 softmax 读写交替, 避免 QK 写和 SM 读同一块 UB 冲突
    static constexpr uint32_t UB_S_OTMP_BUF_STAGES = 2U; // UB S buffer 双缓冲
    // L1 P buffer 3 级: softmax 写 P 到 L1, PV 滞后 2 个 batch 读 P, 需要 3 级保活
    static constexpr uint32_t P_L1_BUF_NUM = PRE_LAUNCH + 1U; // L1 P buffer 级数 (3 级)

    static constexpr uint32_t FP32_ONE_BLOCK_SIZE = 8U; // fp32 每 32B block 的元素数 (32B/4B=8)
    // Must match BlockEpilogue's per-AIV Phase1 stats allocation.  A full
    // batch can give one AIV four groupSize=16 groups, requiring 64 values.
    static constexpr uint32_t SM_STATS_PER_SUBBLOCK = 64U;
    static constexpr uint32_t SM_STATS_PER_STAGE = SM_STATS_PER_SUBBLOCK * 2U;
    // === Workspace softmaxMax/softmaxSum 初始值 ===
    //   rowMax = -inf: 使未写入的 slot 被 Phase2 识别为无效 (NEG_INF_LSE 判断)
    //     为什么用 -inf? 如果一个 Q token 没有选择某个 KV block, 对应 slot 不会被 Phase1 写入
    //     Phase2 合并时看到 rowMax=-inf → scale=0 → 该 slot 不贡献到最终输出 (安全跳过)
    //   rowSum = 0: 使 scale=0, 无效 slot 不贡献到最终输出 (与 rowMax=-inf 配合)
    static constexpr float WS_ROWMAX_INIT = -3.4028235e38f; // -inf 的 fp32 近似值
    static constexpr float WS_ROWSUM_INIT = 0.0f;
    static constexpr uint32_t WS_INIT_EVT = 5U;      // V_MTE3/MTE3_V 空闲 event id (用于 init 的 V→MTE3 同步)
    static constexpr uint32_t WS_INIT_CHUNK = 4096U; // 初始化分块大小 (fp32 元素, 8 的倍数, 一次搬 16KB)

    // gmUb/glUb are 2-deep by ubSBufId inside the prefill softmax epilogue (see
    // reg_low_prec_bf16.hpp ctor). CopyPartialStatsToGm (MTE3) reads gmUb[ubSBufId]/
    // glUb[ubSBufId] right after each softmax; the epilogue waits
    // MTE3_V(ubSBufId + 4) before overwriting the same stage, and this kernel
    // returns that event after the copy. InitSyncFlags primes IDs 4/5 for the
    // first block of each stage. No single-buffer reuse -> no race.
    // Offsets alias the prefill softmax ctor's 2-deep gmUb/glUb regions (must stay
    // byte-identical to that ctor). gmUb = this AIV's rowMax (fp32), glUb = rowSum.
    static constexpr uint32_t SM_UB_BLOCK = 16384U; // A2 UB_UINT8_BLOCK_SIZE
    // Must match BlockEpilogue's A2 UB layout: LS/LP/FS occupy four stages,
    // then TV starts at stage 6 and row statistics start at stage 7.
    static constexpr uint32_t SM_UB_GM_OFFSET = 7U * SM_UB_BLOCK; // rowMax base
    static constexpr uint32_t SM_UB_STAGE_BYTES = SM_STATS_PER_STAGE * sizeof(float);
    static constexpr uint32_t SM_UB_GL_OFFSET = SM_UB_GM_OFFSET + 2U * SM_UB_STAGE_BYTES;

    static constexpr uint32_t COPY_GRANULARITY = 2;

    // CSR batch-read UB buffers (VEC only): replace per-element GlobalTensor::GetValue
    // (blacklisted API, ~100 cycles each) with one DataCopyPad per task + LocalTensor
    // GetValue (~few cycles each).  Free UB starts after GL stage 1.
    // The A2 UB layout leaves 14 KiB after two 8192-entry int32 CSR arrays.
    // Keeping an entire hot CSR row in UB avoids scalar GM reads in the
    // 128K causal workload, whose longest row is 2048 entries.
    static constexpr uint32_t CSR_UB_MAX_QTOKENS = 8192U;
    static constexpr uint32_t CSR_UB_QINDICES_OFFSET = SM_UB_GL_OFFSET + 2U * SM_UB_STAGE_BYTES;
    static constexpr uint32_t CSR_UB_SLOTK_OFFSET = CSR_UB_QINDICES_OFFSET + CSR_UB_MAX_QTOKENS * sizeof(int32_t);

    __aicore__ inline uint64_t TaskLinearIdx(uint32_t qToken, uint32_t kvHeadIdx) const
    {
        return static_cast<uint64_t>(qToken) * kvHeads_ + kvHeadIdx;
    }

    __aicore__ inline uint64_t SlotStatOffset(uint32_t qToken, uint32_t kvHeadIdx, uint32_t slotK) const
    {
        return TaskLinearIdx(qToken, kvHeadIdx) * topK_ * slotStatElems_ +
               static_cast<uint64_t>(slotK) * slotStatElems_;
    }

    // 初始化流水线同步事件标志 (预置为 "绿灯" 状态)
    //
    // 【为什么需要同步标志?】
    //   三级流水线中, CUBE 和 VEC 需要协作:
    //     CUBE 完成 QK → 写 UB S → 通知 VEC 读 S 做 softmax
    //     VEC 完成 softmax → 写 L1 P → 通知 CUBE 读 P 做 PV
    //   这些通知就是 "flag": SetFlag=发信号, WaitFlag=等信号
    //
    // 【为什么要预置?】
    //   流水线启动时, 第一批数据没有前置依赖:
    //     - QK 的第一批 K 数据加载 (MTE1→MTE2) 不需要等待
    //     - VEC 的第一批 softmax 读取 (V→MTE2) 不需要等待
    //   预置 flag 使第一批数据不被阻塞, 否则流水线会 "死锁" 在启动阶段
    //
    // 【CUBE 侧 flag 说明】
    //   MTE1_MTE2: K/Q 从 L1 加载到 L0A/L0B 的同步 (MTE1 写 L1, MTE2 读 L1)
    //   M_MTE1:    MMAD 计算完成后通知 MTE1 加载下一批数据
    //   FIX_M:     Fixpipe (L0C→UB) 完成后通知 MMAD 可以写下一批 L0C
    //   CrossCore (PIPE_MTE1): 通知 VEC 侧: K/Q 已加载到 L1, VEC 可以准备读取
    //
    // 【VEC 侧 flag 说明】
    //   V_MTE2:    V 计算完成 → MTE2 可以搬运数据到 UB
    //   MTE3_V:    MTE3 (UB→GM) 搬运完成 → V 可以覆盖 UB 数据
    //   S_MTE3/MTE3_S: softmax stats 的 UB→GM 搬运同步
    //   CrossCore (PIPE_V): 通知 CUBE 侧: softmax 完成, P 已写入 L1
    __aicore__ inline bool OwnsPhase1Task() const
    {
        // AIV block indices are linear across the two sub-blocks; map both
        // AIVs back to their co-located AIC index before testing ownership.
        const uint32_t logicalCoreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        return logicalCoreIdx < totalTaskNumP1_;
    }

    __aicore__ inline void InitSyncFlags()
    {
#ifdef __DAV_CUBE__
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID3);
        // A2: MODE_2 cross-core priming. CUBE primes smToMm2Flag IDs (2,3,4) on
        // PIPE_FIX so VEC's first Waits succeed. PIPE_FIX is proven working for
        // cross-core WAIT on CUBE (mm1ToSmFlag uses it). PIPE_MTE1 does NOT work
        // for cross-core WAIT on AIC — the Wait silently fails.
        if (OwnsPhase1Task()) {
            AscendC::CrossCoreSetFlag<0x2U, PIPE_FIX>(2);
            AscendC::CrossCoreSetFlag<0x2U, PIPE_FIX>(3);
            AscendC::CrossCoreSetFlag<0x2U, PIPE_FIX>(4);
        }
#endif
#ifdef __DAV_VEC__
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID5);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID1);
        // A2: MODE_2 cross-core priming. VEC primes mm1ToSmFlag IDs (0,1) on
        // PIPE_V so CUBE's first Waits succeed.
        if (OwnsPhase1Task()) {
            AscendC::CrossCoreSetFlag<0x2U, PIPE_V>(0);
            AscendC::CrossCoreSetFlag<0x2U, PIPE_V>(1);
        }
#endif
    }

    __aicore__ inline void ReleaseSyncFlags()
    {
#ifdef __DAV_CUBE__
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID3);
        // A2: MODE_2 cross-core drain. CUBE drains mm1ToSmFlag IDs (0,1) —
        // non-templated Wait (MODE=0/PIPE_S). No V1 dual-flags.
        if (OwnsPhase1Task()) {
            AscendC::CrossCoreWaitFlag<2, PIPE_FIX>(0);
            AscendC::CrossCoreWaitFlag<2, PIPE_FIX>(1);
        }
#endif
#ifdef __DAV_VEC__
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);
        // MTE3_V IDs 4/5 are drained at the end of Phase1 by the stats ring.
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID1);
        // A2: MODE_2 cross-core drain. VEC drains smToMm2Flag IDs (2,3,4) —
        // non-templated Wait (MODE=0/PIPE_S). Only subBlock 0 drains — MODE_2
        if (OwnsPhase1Task() && AscendC::GetSubBlockIdx() == 0U) {
            AscendC::CrossCoreWaitFlag<2, PIPE_MTE3>(2);
            AscendC::CrossCoreWaitFlag<2, PIPE_MTE3>(3);
            AscendC::CrossCoreWaitFlag<2, PIPE_MTE3>(4);
        }
#endif
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline static uint32_t VecRowSplit(uint32_t groupSize)
    {
        uint32_t half = RoundUp(groupSize, 8U) / 2U;
        return (half > groupSize) ? groupSize : half;
    }

    __aicore__ inline static uint32_t VecNumRows(uint32_t groupSize)
    {
        if (groupSize <= 1U) {
            return (AscendC::GetSubBlockIdx() == 0U) ? groupSize : 0U;
        }
        uint32_t split = VecRowSplit(groupSize);
        return (AscendC::GetSubBlockIdx() == 0U) ? split : (groupSize - split);
    }

    __aicore__ inline static uint32_t VecGlobalRowOffset(uint32_t groupSize)
    {
        return (AscendC::GetSubBlockIdx() == 0U) ? 0U : VecRowSplit(groupSize);
    }

    __aicore__ inline MinimaxSaSplitKvRegularKernelA2() {}

    __aicore__ inline void operator()(MinimaxSaSplitKvKernelParamsA2 const &params)
    {
        // ================================================================
        // Kernel 主入口: 按步骤执行 Phase1 + Phase2
        // ================================================================

        // 步骤 1: 从 GM tiling 数据中解析所有 shape/配置参数到成员变量
        //   tiling data 由 host 侧填充, 包含 batch/heads/D/blockSize/topK 等
        __gm__ MinimaxSaSplitKv::MinimaxSparseAttentionSplitKvTilingData *tilingData =
            reinterpret_cast<__gm__ MinimaxSaSplitKv::MinimaxSparseAttentionSplitKvTilingData *>(params.tiling);
        FetchBaseShapeInfo(tilingData);

        // 步骤 2: 绑定所有 GM GlobalTensor (输入/输出/workspace)
        //   GlobalTensor 是对 GM 地址的封装, 后续通过它访问 GM 数据
        AscendC::GlobalTensor<ElementQ> gQ;
        gQ.SetGlobalBuffer((__gm__ ElementQ *)params.q);
        AscendC::GlobalTensor<ElementK> gK;
        gK.SetGlobalBuffer((__gm__ ElementK *)params.k);
        AscendC::GlobalTensor<ElementV> gV;
        gV.SetGlobalBuffer((__gm__ ElementV *)params.v);
        AscendC::GlobalTensor<int32_t> gK2qRowPtr;
        gK2qRowPtr.SetGlobalBuffer((__gm__ int32_t *)params.k2qRowPtr);
        AscendC::GlobalTensor<int32_t> gK2qQIndices;
        gK2qQIndices.SetGlobalBuffer((__gm__ int32_t *)params.k2qQIndices);
        AscendC::GlobalTensor<int32_t> gK2qSlotIndices;
        gK2qSlotIndices.SetGlobalBuffer((__gm__ int32_t *)params.k2qSlotIndices);
        AscendC::GlobalTensor<int32_t> gBlockTable;
        gBlockTable.SetGlobalBuffer((__gm__ int32_t *)params.blockTable);
        gActualQseqlen_.SetGlobalBuffer((__gm__ int32_t *)params.actualQseqlen);
        gActualKvseqlen_.SetGlobalBuffer((__gm__ int32_t *)params.actualKvseqlen);
        AscendC::GlobalTensor<ElementO> gO;
        gO.SetGlobalBuffer((__gm__ ElementO *)params.o);
        AscendC::GlobalTensor<float> gSoftmaxLse;
        gSoftmaxLse.SetGlobalBuffer((__gm__ float *)params.softmaxLse);

        // 步骤 3: 绑定 workspace 三块 buffer: accumOut | softmaxMax | softmaxSum
        //   按顺序分配偏移, 与 host 侧 tiling CalculateWorkSpace 的布局一致
        uint64_t wsOffset = 0U;
        AscendC::GlobalTensor<ElementWorkspaceO> gAccumOut; // O_partial = P×V (未归一化的部分结果)
        gAccumOut.SetGlobalBuffer((__gm__ ElementWorkspaceO *)(params.workSpace + wsOffset));
        wsOffset += accumOutSize_ * sizeof(ElementWorkspaceO);
        AscendC::GlobalTensor<float> gSoftmaxMax; // 每 slot 的 rowMax (online softmax 最大值)
        gSoftmaxMax.SetGlobalBuffer((__gm__ float *)(params.workSpace + wsOffset));
        wsOffset += lseStatSize_ * sizeof(float);
        AscendC::GlobalTensor<float> gSoftmaxSum; // 每 slot 的 rowSum (online softmax 累加和)
        gSoftmaxSum.SetGlobalBuffer((__gm__ float *)(params.workSpace + wsOffset));
        wsOffset += lseStatSize_ * sizeof(float);

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();
        if (coreIdx == 0U) {
            A2_PRINTF("[MinimaxSaSplitKvA2] init coreNum=%u p1=%u p2=%u qHeads=%u kvHeads=%u group=%u topK=%u\n",
                      coreNum, totalTaskNumP1_, totalTaskNumP2_, qHeads_, kvHeads_, groupSize_, topK_);
        }
        // Save wsOffset (start of GM S/P staging region, after softmaxSum) for later
        // GM S/P workspace allocation after coreIdx remapping.
        // Align to 32 bytes: Nd2Nz DataCopy (GM→L1 for P) requires 32-byte aligned source.
        uint64_t gmStagingBaseOffset = (wsOffset + 31U) & ~static_cast<uint64_t>(31U);

        // 4. VEC 侧预置 workspace softmaxMax=-inf / softmaxSum=0
        //    Must run BEFORE InitSyncFlags: InitWorkspaceStats has PipeBarrier<PIPE_ALL>()
        //    which deadlocks if cross-core flags (CrossCoreSetFlag on PIPE_V/PIPE_MTE1)
        //    are pending — those flags can't be consumed until Phase1 main loop runs.
        InitWorkspaceStats(gSoftmaxMax, gSoftmaxSum);
        if (coreIdx == 0U) {
            A2_PRINTF("[MinimaxSaSplitKvA2] workspace stats initialized\n");
        }
        // 5. 初始化 Cube/Vector 跨核同步事件标志 (流水线事件预置)
        //    After InitWorkspaceStats' PipeBarrier<PIPE_ALL> — no pending cross-core flags.
        InitSyncFlags();
        if (coreIdx == 0U) {
            A2_PRINTF("[MinimaxSaSplitKvA2] sync flags initialized\n");
        }
        // 6. 全局同步: 确保 workspace 初始化完成
        AscendC::SyncAll<false>();
        if (coreIdx == 0U) {
            A2_PRINTF("[MinimaxSaSplitKvA2] init SyncAll passed\n");
        }
        // VEC: GetBlockIdx() returns linear AIV id (0..2*N-1). Divide by
        // subBlockNum (2) to get the MIX block index (0..N-1), matching CUBE's
        // GetBlockIdx(). This ensures both VEC subBlocks of the same MIX block
        // get the same coreIdx as their co-located CUBE, which is REQUIRED for:
        //   1. MODE_2 cross-core sync (both VEC subBlocks must participate)
        //   2. GM S/P workspace sharing (same slot as co-located CUBE)
        // CUBE: GetSubBlockNum()=1, so division is a no-op.
        coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t taskCoreIdx = coreIdx;
        uint32_t taskCoreNum = coreNum;
        uint32_t workspaceCoreIdx = coreIdx;
        // 3b. A2-only: allocate per-core GM S and GM P staging workspaces.
        //     Layout: [all_cores_gmS][all_cores_gmP] after softmaxSum.
        //     Both CUBE and VEC of the same AIC share the same workspace slot
        //     (coreIdx is remapped to AIC index for VEC).
        if (gmSWorkspaceSize_ > 0U) {
            constexpr uint32_t A2_L0_TILE_M = 64U;
            gmSStageElems_ = static_cast<uint64_t>(A2_L0_TILE_M) * blockSize_;
            gmPStageElems_ = static_cast<uint64_t>(A2_L0_TILE_M) * blockSize_;
            gmSWorkspace_.SetGlobalBuffer(
                (__gm__ ElementS *)(params.workSpace + gmStagingBaseOffset + workspaceCoreIdx * gmSWorkspaceSize_));
            // GetBlockNum() is the AIC block count on A2 for both Cube and
            // Vector kernels.  Vector has two subblocks per AIC, but that
            // does not change the workspace allocation count.
            uint32_t numAicCores = AscendC::GetBlockNum();
            uint64_t totalGmSBytes = static_cast<uint64_t>(numAicCores) * gmSWorkspaceSize_;
            gmPWorkspace_.SetGlobalBuffer((__gm__ ElementP *)(params.workSpace + gmStagingBaseOffset + totalGmSBytes +
                                                              workspaceCoreIdx * gmPWorkspaceSize_));
        }
        // 7. Phase 1: KV-centric partial compute (Cube QK/PV + Vector softmax)
        Phase1KvCentricCompute(taskCoreIdx, taskCoreNum, gQ, gK, gV, gK2qRowPtr, gK2qQIndices, gK2qSlotIndices,
                               gBlockTable, gAccumOut, gSoftmaxMax, gSoftmaxSum);
#ifdef __DAV_CUBE__
        // SyncAll orders cores, but does not drain the final asynchronous
        // L0C-to-GM O_partial write that Phase2 consumes.
        AscendC::PipeBarrier<PIPE_FIX>();
#endif
#if KERNEL_DUMP_SYNC_TRACE
#ifdef __DAV_CUBE__
        if (coreIdx < 4U) {
            auto traceL1 = resource.l1Buf.template GetBufferByByte<ElementP>(0);
            AscendC::DumpTensor(traceL1, 943, 1);
        }
#endif
        if (coreIdx < 4U) {
            auto traceUb = resource.ubBuf.template GetBufferByByte<float>(0);
            AscendC::DumpTensor(traceUb, 939, 1);
        }
#endif
        if (coreIdx == 0U) {
            A2_PRINTF("[MinimaxSaSplitKvA2] phase1 passed\n");
        }
        // 8. 全局同步: 确保 Phase1 全部完成, workspace 数据就绪
        AscendC::SyncAll<false>();
#ifdef __DAV_CUBE__
        // SyncAll only orders peers of the same core type.  After every AIC
        // has drained FIX and reached the AIC barrier, publish completion to
        // both co-located AIVs before they start reading O_partial in Phase2.
        AscendC::CrossCoreSetFlag<0x2U, PIPE_FIX>(5U);
#endif
#ifdef __DAV_VEC__
        AscendC::CrossCoreWaitFlag<0x2U, PIPE_V>(5U);
#endif
#if KERNEL_DUMP_SYNC_TRACE
        if (coreIdx < 4U) {
            auto traceUb = resource.ubBuf.template GetBufferByByte<float>(0);
            AscendC::DumpTensor(traceUb, 942, 1);
        }
#endif
        if (coreIdx == 0U) {
            A2_PRINTF("[MinimaxSaSplitKvA2] phase1 SyncAll passed\n");
        }
        // 9. Phase 2: 仅 VEC 执行, combine topK 个 partial 到最终输出
        // 9. Phase 2: 仅 VEC 执行, combine topK 个 partial 到最终输出
#ifdef __DAV_VEC__
        Phase2CombineScale(gO, gSoftmaxLse, gAccumOut, gSoftmaxMax, gSoftmaxSum);
#endif
        // 10. 释放同步事件标志
        ReleaseSyncFlags();
    }

private:
    __aicore__ inline void FetchBaseShapeInfo(
        __gm__ MinimaxSaSplitKv::MinimaxSparseAttentionSplitKvTilingData *tilingData)
    {
        batch_ = tilingData->batch;
        qHeads_ = tilingData->numHeads;
        kvHeads_ = tilingData->kvHeads;
        groupSize_ = tilingData->groupSize;
        embed_ = tilingData->embeddingSize;
        blockSize_ = tilingData->blockSize;
        queryTokenStride_ = tilingData->queryTokenStride;
        keyBlockStride_ = tilingData->keyBlockStride;
        valueBlockStride_ = tilingData->valueBlockStride;
        isKvContinuous_ = (tilingData->isKvContinuous != 0U);
        keyTokenStride_ = tilingData->keyTokenStride;
        valueTokenStride_ = tilingData->valueTokenStride;
        topK_ = tilingData->topK;
        totalKvRows_ = tilingData->numKvBlocks;
        maxBlocksPerBatch_ = tilingData->maxBlocksPerBatch;
        k2qNnzUpperBound_ = tilingData->k2qNnzUpperBound;
        totalTaskNumP1_ = tilingData->totalTaskNumP1;
        totalTaskNumP2_ = tilingData->totalTaskNumP2;
        scaleValue_ = tilingData->scaleValue;
        softmaxLseFlag_ = (tilingData->softmaxLseFlag != 0U);
        accumOutSize_ = tilingData->accumOutSize;
        lseStatSize_ = tilingData->lseStatSize;
        gmSWorkspaceSize_ = tilingData->gmSWorkspaceSize; // A2: per-core GM S bytes (A5=0)
        gmPWorkspaceSize_ = tilingData->gmPWorkspaceSize; // A2: per-core GM P bytes (A5=0)
        slotOElems_ = static_cast<uint64_t>(groupSize_) * embed_;
        // The A2 stats workspace is 32-byte aligned per sparse slot.  This
        // lets the vector DMA write a whole slot without touching its neighbour.
        slotStatElems_ = static_cast<uint64_t>(RoundUp(groupSize_, 8U));
    }

    // VEC-only: 初始化 workspace 的 softmaxMax 和 softmaxSum
    //
    // 【作用】
    //   将 workspace 中的 softmaxMax 全部填 -inf, softmaxSum 全部填 0
    //   这些是 "占位值", 确保未被 Phase1 写入的 slot 被 Phase2 正确跳过
    //
    // 【实现方式】
    //   1. VEC 在 UB 中用 Duplicate 填充 -inf 和 0 (一次性填满 WS_INIT_CHUNK=4096 个元素)
    //   2. 循环将 UB 数据 DataCopyPad 到 GM (分块搬运, 每块 4096 个 fp32 = 16KB)
    //   3. 多个 VEC 子核按 chunkIdx stride 分担工作 (并行初始化)
    //
    // 【时序约束】
    //   必须在 InitSyncFlags 之后执行: 本函数有 PipeBarrier<PIPE_ALL>(),
    //   如果有 pending 的 cross-core flag 会死锁 (flag 在 Phase1 主循环才能被消费)
    __aicore__ inline void InitWorkspaceStats(AscendC::GlobalTensor<float> &gSoftmaxMax,
                                              AscendC::GlobalTensor<float> &gSoftmaxSum)
    {
        if (lseStatSize_ == 0U) {
            return;
        }
#ifdef __DAV_VEC__
        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        uint32_t totalSubBlocks = AscendC::GetBlockNum() * subBlockNum;
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t numFullChunks = static_cast<uint32_t>(lseStatSize_ / WS_INIT_CHUNK);
        uint32_t tailElems = static_cast<uint32_t>(lseStatSize_ % WS_INIT_CHUNK);
        uint32_t totalChunks = numFullChunks + (tailElems > 0U ? 1U : 0U);

        AscendC::LocalTensor<float> ubMax = resource.ubBuf.template GetBufferByByte<float>(0);
        AscendC::LocalTensor<float> ubSum =
            resource.ubBuf.template GetBufferByByte<float>(WS_INIT_CHUNK * sizeof(float));
        AscendC::Duplicate(ubMax, WS_ROWMAX_INIT, WS_INIT_CHUNK);
        AscendC::Duplicate(ubSum, WS_ROWSUM_INIT, WS_INIT_CHUNK);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(WS_INIT_EVT);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(WS_INIT_EVT);

        for (uint32_t chunkIdx = blockIdx; chunkIdx < totalChunks; chunkIdx += totalSubBlocks) {
            uint64_t off = static_cast<uint64_t>(chunkIdx) * WS_INIT_CHUNK;
            uint32_t cnt = (chunkIdx < numFullChunks) ? WS_INIT_CHUNK : tailElems;
            uint16_t blockLen = static_cast<uint16_t>(cnt * sizeof(float));
            AscendC::DataCopyPad(gSoftmaxMax[off], ubMax, {1, blockLen, 0, 0});
            AscendC::DataCopyPad(gSoftmaxSum[off], ubSum, {1, blockLen, 0, 0});
        }
        // The following SyncAll only orders core control flow.  Drain the
        // asynchronous MTE3 writes so every core observes initialized stats.
        AscendC::PipeBarrier<PIPE_ALL>();
#endif
    }

    // packedRow 的 MSA INTERLEAVED (轮询) 打包解码
    //
    // 【什么是 INTERLEAVED 打包?】
    //   多 batch 的 KV block 行号不是 "batch0 全部行 + batch1 全部行" (连续),
    //   而是 "所有 batch 的第 0 行, 所有 batch 的第 1 行..." (交错):
    //     packedRow 0 = (batch0, block0)
    //     packedRow 1 = (batch1, block0)
    //     packedRow 2 = (batch2, block0)   ← 假设 batch0~2 都有 ≥1 个 block
    //     packedRow 3 = (batch0, block1)   ← 回到 batch0 的第 2 个 block
    //     ...
    //   这种排列有利于多核负载均衡: 不同核分到的 task 跨 batch, 避免某个 batch 特别长时其他核空闲
    //
    // 【为什么用增量状态机?】
    //   直接解码需要知道每个 batch 的行数 (rows_per_batch), batch 最多 4096 个
    //   → 需要 16KB UB 存储行数表 (不可接受)
    //   增量状态机只需 2 个标量 (batchIdx, kvBlockIdx), 跨 task 循环步进:
    //     batch 0 的行数 → 步进到 batch 1 → ... → 所有 batch 的当前 block 遍历完 → block++
    //   无需额外 UB, 且步进成本可忽略 (大多数 task 只步进 1 步)
    __aicore__ inline uint32_t KvRowsPerBatch(uint32_t batchIdx)
    {
        return CeilDiv(static_cast<uint32_t>(gActualKvseqlen_.GetValue(batchIdx)), blockSize_);
    }

    __aicore__ inline void InitPackedRowCoord(uint32_t &batchIdx, uint32_t &kvBlockIdx)
    {
        batchIdx = 0U;
        kvBlockIdx = 0U;
        while (batchIdx < batch_ && KvRowsPerBatch(batchIdx) == 0U) {
            ++batchIdx;
        }
    }

    __aicore__ inline void AdvancePackedRowCoord(uint32_t &batchIdx, uint32_t &kvBlockIdx)
    {
        batchIdx++;
        while (batchIdx < batch_ && kvBlockIdx >= KvRowsPerBatch(batchIdx)) {
            batchIdx++;
        }
        if (batchIdx >= batch_) {
            kvBlockIdx++;
            batchIdx = 0;
            while (batchIdx < batch_ && kvBlockIdx >= KvRowsPerBatch(batchIdx)) {
                batchIdx++;
            }
        }
    }

    // 预取当前 batch 的 Q/KV seqlen 等标量参数
    //
    // 【作用】
    //   状态机 (AdvancePackedRowCoord) 解出 (batchIdx, kvBlockIdx) 后,
    //   需要获取该 batch 的标量参数供后续计算使用:
    //     kvSeqlenBatch: KV 序列长度 → 计算 validSize (尾块有效 token 数)
    //     numBlocksB:    KV block 总数 → 判断是否尾块
    //     cumQStart:     Q token 累积起始偏移 → 计算 qToken 的全局位置
    //     qSeqlenBatch:  Q 序列长度 → 计算 causal mask 的 qPosition
    //
    // 【优化】
    //   Q 和 KV 在同一个 batch 内计算 (无跨 batch attention), qBatch == batchIdx
    //   所以只取一次 batchIdx 对应的参数, 后续 CalcCausalValidLen 可纯计算 (0 GM GetValue)
    __aicore__ inline void ResolveBatchQSide(uint32_t batchIdx, uint32_t &kvSeqlenBatch, uint32_t &numBlocksB,
                                             uint32_t &cumQStart, uint32_t &qSeqlenBatch, uint32_t &cumKvStart)
    {
        cumKvStart = 0U;
        if (isKvContinuous_) {
            for (uint32_t b = 0; b <= batchIdx; ++b) {
                uint32_t kvB = static_cast<uint32_t>(gActualKvseqlen_.GetValue(b));
                if (b < batchIdx) {
                    cumKvStart += kvB;
                } else {
                    kvSeqlenBatch = kvB;
                }
            }
        } else {
            kvSeqlenBatch = static_cast<uint32_t>(gActualKvseqlen_.GetValue(batchIdx));
        }
        numBlocksB = CeilDiv(kvSeqlenBatch, blockSize_);
        cumQStart = 0;
        for (uint32_t b = 0; b <= batchIdx; ++b) {
            uint32_t qB = static_cast<uint32_t>(gActualQseqlen_.GetValue(b));
            if (b < batchIdx) {
                cumQStart += qB;
            } else {
                qSeqlenBatch = qB;
            }
        }
    }

    // kvSeqlenBatch + numBlocksB are pre-fetched by ResolveBatchQSide for the batch
    // resolved by the state machine (avoids a redundant gActualKvseqlen_ GetValue here).
    __aicore__ inline uint32_t CalcKvBlockValidSize(uint32_t kvSeqlenBatch, uint32_t numBlocksB, uint32_t localBlockIdx)
    {
        if (localBlockIdx == numBlocksB - 1) {
            return kvSeqlenBatch - localBlockIdx * blockSize_;
        }
        return blockSize_;
    }

    // 计算 causal mask 的有效列数 (causalValidLen)
    //
    // 【什么是 causal mask?】
    //   在自回归 LLM 中, Q token 只能 "看到" 它之前的 KV token (不能看到未来)
    //   这叫 "因果注意力" (causal attention), 通过 causal mask 实现:
    //     - Q token 在位置 p, 只能 attend 到 KV 位置 0~p
    //     - 超出 p 的 KV 列被 mask 掉 (置 -inf 或跳过)
    //   本算子不写 -inf mask, 而是直接截断计算范围: 只算 [0, causalValidLen) 列
    //
    // 【causal mask 规则】
    //   每个 Q token 的 causal 边界由 Q 所在 batch 的 seqlen 决定:
    //     qPosition = kv_seqlen - q_seqlen + localQIdx  (Q 在序列中的绝对位置)
    //       解释: prefill 时 Q 是 prompt 的后半段, Q 起始位置 = kv_seqlen - q_seqlen
    //             localQIdx 是 Q token 在当前 batch 中的局部序号
    //     kvStartPos = localBlockIdx × blockSize         (KV block 的起始位置)
    //     causalValidLen = 0                          if qPosition < kvStartPos (完全在因果窗口外)
    //                    = min(validSize, qPosition - kvStartPos + 1)  otherwise
    //       解释: 如果 Q 位置 < KV block 起始位置 → 该 Q 完全看不到此 block → 0 列
    //             否则 → 有效列数 = min(block有效列数, Q位置-block起始位置+1)
    //
    // 【优化】所有输入参数已由 ResolveBatchQSide 预取, 此函数纯计算, 0 GM GetValue
    //   GM GetValue 有高延迟, 预取到寄存器后纯计算避免流水线停顿
    __aicore__ inline uint32_t CalcCausalValidLen(uint32_t qToken, uint32_t validSize, uint32_t kvStartPos,
                                                  uint32_t cumQStart, uint32_t qSeqlenBatch, uint32_t kvSeqlenBatch)
    {
        uint32_t localQIdx = qToken - cumQStart;
        uint32_t qPosition = kvSeqlenBatch - qSeqlenBatch + localQIdx;
        if (qPosition < kvStartPos) {
            return 0U;
        }
        uint32_t maxLen = qPosition - kvStartPos + 1;
        return (maxLen < validSize) ? maxLen : validSize;
    }

    // Copies rowMax + rowSum for 1 or 2 groups (PAIR) from UB to GM in one DataCopyPad each
    // (blockCount = ndNum). Mirrors the Q-gather/O-partial ndNum=2 pairing: the caller
    // computes the per-pair GM dst gap (dstStride, 32B-block units = end-of-group0 to
    // start-of-group1) and UB src gap (srcStride, 32B-block units) from the two groups'
    // actual (qToken, slotK) lseOff -- no uniform-stride assumption (ndNum=2 needs only
    // the one inter-pair gap, computed per pair). Falls back to blockCount=1 (single group)
    // at the caller when the pair's gaps aren't 32B-aligned or exceed the uint16 stride.
    __aicore__ inline void CopyPartialStatsToGm(AscendC::GlobalTensor<float> &gSoftmaxMax,
                                                AscendC::GlobalTensor<float> &gSoftmaxSum,
                                                const AscendC::LocalTensor<float> &rowMaxLocal,
                                                const AscendC::LocalTensor<float> &rowSumLocal, uint64_t lseOffset,
                                                uint32_t rowCount, uint16_t blockCount, uint32_t srcStride,
                                                uint32_t dstStride)
    {
        if (rowCount == 0U) {
            return;
        }
        // Each workspace slot is now padded to at least eight FP32 values, so
        // even a groupSize=4 transfer owns a full 32-byte GM transaction.
        // Use DMA rather than scalar stores: the Phase2 readers run on other
        // AIVs and need the store to be globally visible before SyncAll.
        if (rowCount < 8U && blockCount == 1U) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID3);
            AscendC::DataCopyPad(
                gSoftmaxMax[lseOffset], rowMaxLocal,
                AscendC::DataCopyExtParams{1U, static_cast<uint16_t>(rowCount * sizeof(float)), 0U, 0U, 0U});
            AscendC::DataCopyPad(
                gSoftmaxSum[lseOffset], rowSumLocal,
                AscendC::DataCopyExtParams{1U, static_cast<uint16_t>(rowCount * sizeof(float)), 0U, 0U, 0U});
            // The next group may immediately reuse this UB region.  Drain the
            // small, single-group MTE3 copy before returning to the caller.
            AscendC::PipeBarrier<PIPE_ALL>();
            return;
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID3);
        AscendC::DataCopyPad(gSoftmaxMax[lseOffset], rowMaxLocal,
                             AscendC::DataCopyExtParams{blockCount, static_cast<uint16_t>(rowCount * sizeof(float)),
                                                        srcStride, dstStride, 0});
        AscendC::DataCopyPad(gSoftmaxSum[lseOffset], rowSumLocal,
                             AscendC::DataCopyExtParams{blockCount, static_cast<uint16_t>(rowCount * sizeof(float)),
                                                        srcStride, dstStride, 0});
    }

    // Batched stats scatter: the softmax wrote this AIV's rowMax/rowSum into gmUb/glUb
    // (ubSBufId stage). Cores split by GROUP (not M/2): each AIV owns whole groups [gLo, gHi),
    // no group straddles the two AIVs; every owned group has rowCount = groupRows (never 0).
    // lseOff is computed on demand from the batch's qToken/slotK (== SlotStatOffset(qToken,
    // kvHeadIdx, slotK,0)), so no lseOff array is marshalled. gmUbAlias/glUbAlias are bound
    // ONCE before the loop to the stage base (loop-invariant ubSBufId); the per-group stride
    // (localOff) is applied via [localOff] in the loop. The rowMax/rowSum group base uses an
    // 8-float-aligned stride (matches the softmax epilogue's statsOff): CopyPartialStatsToGm's
    // DataCopyPad needs a 32B-aligned source; groupRows<8 (local groupSize=4) would otherwise
    // misalign it. No-op for groupRows>=8 (production groupSize=16).
    __aicore__ inline void ScatterBatchStats(AscendC::GlobalTensor<float> &gSoftmaxMax,
                                             AscendC::GlobalTensor<float> &gSoftmaxSum, uint32_t ubSBufId,
                                             uint32_t batchM, uint32_t groupRows, const uint32_t *qTokens,
                                             const uint32_t *slotKs, uint32_t kvHeadIdx, uint32_t groupCount,
                                             bool extraToSubBlock0)
    {
        // Match the softmax epilogue's group split across the two AIV sub-blocks.
        uint32_t gSplit = groupCount / 2U;
        if ((groupCount & 1U) != 0U && extraToSubBlock0) {
            ++gSplit;
        }
        uint32_t mCopyOffset = gSplit * groupRows;
        uint32_t mHalf = mCopyOffset < batchM ? mCopyOffset : batchM;
        uint32_t m = (AscendC::GetSubBlockIdx() == 0U) ? mHalf : (batchM - mHalf);
        if (m == 0U) {
            return;
        }
        uint32_t startRow = AscendC::GetSubBlockIdx() * mCopyOffset;
        uint32_t gLo = startRow / groupRows;
        uint32_t gHi = gLo + m / groupRows;
        // rowMax/rowSum group base stride = 8-float aligned (matches softmax epilogue's statsOff;
        // DataCopyPad source must be 32B-aligned). No-op for groupRows>=8 (production 16).
        uint32_t grpStride = RoundUp(groupRows, 8U);
        // Bind once to the stage base (loop-invariant); per-group offset via [localOff].
        // Add subBlock offset.  Each AIV owns up to four groupSize=16 groups,
        // so its rowMax/rowSum region must contain 64 floats.
        uint32_t subBlockStatsOff = AscendC::GetSubBlockIdx() * SM_STATS_PER_SUBBLOCK * sizeof(float);
        AscendC::LocalTensor<float> gmUbAlias = resource.ubBuf.template GetBufferByByte<float>(
            SM_UB_GM_OFFSET + ubSBufId * SM_UB_STAGE_BYTES + subBlockStatsOff);
        AscendC::LocalTensor<float> glUbAlias = resource.ubBuf.template GetBufferByByte<float>(
            SM_UB_GL_OFFSET + ubSBufId * SM_UB_STAGE_BYTES + subBlockStatsOff);

        // A2 DataCopyExtParams strides are measured in 32-byte blocks.  The
        // compact stats layout is indexed by (q, kvHead, slot), so adjacent
        // groups are not generally 32-byte aligned (and slot order is not
        // guaranteed).  Copying a pair with a byte stride silently writes the
        // second group to the wrong workspace location.  Copy each group
        // independently; this is the correctness-first path and preserves the
        // exact per-slot GM layout consumed by Phase2.
        for (uint32_t g = gLo; g < gHi; ++g) {
            uint32_t localOff = (g - gLo) * grpStride;
            uint64_t lseOffset = SlotStatOffset(qTokens[g], kvHeadIdx, slotKs[g]);
            if (qTokens[g] == 2U && kvHeadIdx == 0U && slotKs[g] == 1U) {
                A2_PRINTF("[A2 scatter probe] q=%u kv=%u slot=%u lse=%u local=%u "
                          "rowsum={%f,%f,%f,%f}\n",
                          qTokens[g], kvHeadIdx, slotKs[g], static_cast<uint32_t>(lseOffset), localOff,
                          glUbAlias.GetValue(localOff), glUbAlias.GetValue(localOff + 1U),
                          glUbAlias.GetValue(localOff + 2U), glUbAlias.GetValue(localOff + 3U));
            }
            CopyPartialStatsToGm(gSoftmaxMax, gSoftmaxSum, gmUbAlias[localOff], glUbAlias[localOff], lseOffset,
                                 groupRows, 1U, 0U, 0U);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ubSBufId + 4U);
    }

    // Phase 1: KV-centric partial compute (Cube QK/PV + Vector softmax)
    //
    // 【整体流程】 每个 task 处理 1 个 KV block × 1 KV head × 所有选中该 block 的 Q token:
    //
    //   ┌──────────────────────────────────────────────────────────────────────┐
    //   │ 步骤 1: MSA 坐标步进                                                  │
    //   │   packedRow (交错打包行号) → (batchIdx, localBlockIdx)                │
    //   │   通过增量状态机 AdvancePackedRowCoord 解码, 无需额外 UB               │
    //   ├──────────────────────────────────────────────────────────────────────┤
    //   │ 步骤 2: 查 block_table                                                │
    //   │   block_table[batchIdx, localBlockIdx] → physicalBlockId             │
    //   │   计算 validSize: 非尾块=blockSize, 尾块=kvSeqlen%blockSize           │
    //   ├──────────────────────────────────────────────────────────────────────┤
    //   │ 步骤 3: 从 CSR 读取 Q token 列表                                      │
    //   │   k2qRowPtr[kvHeadIdx, packedRow] → csrStart/csrEnd                   │
    //   │   k2qQIndices[csrStart:csrEnd] → 选择该 block 的 Q token id 列表      │
    //   │   k2qSlotIndices[csrStart:csrEnd] → 每个 Q token 的 slotK 编号        │
    //   ├──────────────────────────────────────────────────────────────────────┤
    //   │ 步骤 4: CUBE 加载 K/V 到 L1 (常驻)                                    │
    //   │   K/V 按 [physicalBlock, blockSize, kvHeads, D] 从 GM 加载到 L1      │
    //   │   每个 task 仅加载一次, 后续所有 Q token 共享这份 K/V                  │
    //   ├──────────────────────────────────────────────────────────────────────┤
    //   │ 步骤 5: 按批遍历 Q token (三级流水线主循环)                            │
    //   │   将 numQTokens 按 batchGroupsMax 分批, 每批最多 L0_TILE_M/groupSize  │
    //   │   个 Q token (受限于 L0 tile 大小)                                    │
    //   │                                                                       │
    //   │   每个 batch bi 的流水线时序:                                         │
    //   │     a. 从 CSR gather Q token id 和 slotK                             │
    //   │     b. CUBE: QK matmul — S = Q_tile × K^T, 结果 fixpipe 到 UB        │
    //   │     c. VEC:  online softmax — P = exp(S-rowMax), rowMax/rowSum       │
    //   │     d. VEC:  scatter rowMax/rowSum 到 workspace GM                   │
    //   │     e. CUBE: PV matmul — O_partial = P × V (滞后 PRE_LAUNCH=2 个 batch)│
    //   │                                                                       │
    //   │   三级流水:                                                          │
    //   │     时间→  bi=0     bi=1     bi=2     bi=3     bi=4                   │
    //   │     QK:    QK(b0)   QK(b1)   QK(b2)   QK(b3)                         │
    //   │     SM:             SM(b0)   SM(b1)   SM(b2)   SM(b3)                │
    //   │     PV:                      PV(b0)   PV(b1)   PV(b2)                │
    //   │     ← QK 领先 → ← softmax 滞后1 → ← PV 滞后2 (PRE_LAUNCH) →          │
    //   └──────────────────────────────────────────────────────────────────────┘
    __aicore__ inline void Phase1KvCentricCompute(
        uint32_t coreIdx, uint32_t coreNum, AscendC::GlobalTensor<ElementQ> const &gQ,
        AscendC::GlobalTensor<ElementK> const &gK, AscendC::GlobalTensor<ElementV> const &gV,
        AscendC::GlobalTensor<int32_t> const &gK2qRowPtr, AscendC::GlobalTensor<int32_t> const &gK2qQIndices,
        AscendC::GlobalTensor<int32_t> const &gK2qSlotIndices, AscendC::GlobalTensor<int32_t> const &gBlockTable,
        AscendC::GlobalTensor<ElementWorkspaceO> &gAccumOut, AscendC::GlobalTensor<float> &gSoftmaxMax,
        AscendC::GlobalTensor<float> &gSoftmaxSum)
    {
        // Batched QK/softmax/PV: BATCH_GROUPS qTokens per batch -> M = groupCount*groupSize_
        // (up to BlockMmadQK::L0_TILE_M=64). MAX_BATCH_GROUPS caps per-batch arrays; groupSize>=16
        // fills the tile, smaller groupSize underfills but stays correct.
        // 批量QK/softmax/PV：BATCH_GROUPS 每批qTokens数量 -> M = groupCount*groupSize_
        // （最大为BlockMmadQK::L0_TILE_M=64）。
        // MAX_BATCH_GROUPS限制每批数组的数量；
        // groupSize>=16时填满tile，groupSize较小时虽未填满但结果仍然正确。
        constexpr uint32_t MAX_BATCH_GROUPS = 8U;
        uint32_t batchGroupsMax = BlockMmadQK::L0_TILE_M / groupSize_;
        if (batchGroupsMax == 0U || batchGroupsMax > MAX_BATCH_GROUPS) {
            batchGroupsMax = MAX_BATCH_GROUPS;
        }
        // The A2 softmax keeps S/P in two 16-KiB stages and the float FS
        // workspace in one 32-KiB stage.  Both vector sub-blocks share the
        // latter stage, so one batched tile must fit in 8192 float elements
        // (RoundUp(blockSize, 16) columns).  Without this cap, groupSize=16
        // selected 8 groups (M=128); sub-block 1 started at row 64 and wrote
        // past the stage into the next pipeline buffer/statistics area.
        uint32_t maxGroupsBySoftmaxUb =
            EpilogueOnlineSoftmax::MAX_UB_S_ELEM_NUM / (RoundUp(blockSize_, 16U) * groupSize_);
        if (maxGroupsBySoftmaxUb == 0U) {
            maxGroupsBySoftmaxUb = 1U;
        }
        if (batchGroupsMax > maxGroupsBySoftmaxUb) {
            batchGroupsMax = maxGroupsBySoftmaxUb;
        }
#ifdef __DAV_CUBE__
        BlockMmadQK blockMmadQK;
        BlockMmadPV blockMmadPV;

        blockMmadQK.Init(resource, blockSize_, embed_, gmSWorkspace_);
        uint32_t qkL1Used = blockSize_ * embed_ * sizeof(ElementK) + BlockMmadQK::L0_TILE_M * embed_ * sizeof(ElementQ);
        // PV P L1 stage must hold the batched P [L0_TILE_M, blockSize_] (max batchM, K=validSize<=blockSize).
        // PV P L1阶段必须持有批量P [L0_TILE_M, blockSize_]（最大batchM，K=validSize<=blockSize）。
        blockMmadPV.Init(resource, qkL1Used, blockSize_, embed_, BlockMmadQK::L0_TILE_M);
        // A2: pass GM P workspace to CUBE PV for Nd2Nz GM→L1 P load.
        blockMmadPV.SetGmPWorkspace(gmPWorkspace_, static_cast<uint32_t>(gmPStageElems_));

        // CUBE Fixpipe dst aliases decode softmax lsUbTensor (offset 0,
        // MAX_UB_S_ELEM_NUM bf16 = 32768B/stage), shared cross-core with VEC softmax.
        // CUBE Fixpipe目标别名解码softmax的lsUbTensor（偏移量为0，MAX_UB_S_ELEM_NUM为32768B/stage），
        // 与VEC softmax共享跨核。
        AscendC::LocalTensor<ElementS> ubSTensor[UB_S_OTMP_BUF_STAGES];
        for (uint32_t i = 0; i < UB_S_OTMP_BUF_STAGES; i++) {
            ubSTensor[i] = resource.ubBuf.template GetBufferByByte<ElementS>(EpilogueOnlineSoftmax::MAX_UB_S_ELEM_NUM *
                                                                             sizeof(ElementS) * i);
        }

        int64_t keyRowStride =
            isKvContinuous_ ? static_cast<int64_t>(keyTokenStride_) : static_cast<int64_t>(kvHeads_) * embed_;
        int64_t valueRowStride =
            isKvContinuous_ ? static_cast<int64_t>(valueTokenStride_) : static_cast<int64_t>(kvHeads_) * embed_;
#endif
#ifdef __DAV_VEC__
        // decode bf16 regbase softmax self-allocates its UB (lsUb/lpUb/lmUb/gmUb/glUb/
        // dmUb/llUb at fixed offsets, see reg_low_prec_bf16.hpp ctor). VEC no longer
        // pre-allocates ubSTensor/ubPBuf/ubRowMax/ubRowSum/ubTmpBuf.
        EpilogueOnlineSoftmax epilogueSoftmax(resource, scaleValue_);
        // A2: pass GM S workspace to VEC epilogue for DataCopyPad GM→UB.
        epilogueSoftmax.SetGmSWorkspace(gmSWorkspace_, gmSStageElems_);
        // A2: pass GM P workspace to VEC epilogue for DataCopyPad UB→GM.
        epilogueSoftmax.SetGmPWorkspace(gmPWorkspace_, gmPStageElems_);

        uint32_t qkL1Used = blockSize_ * embed_ * sizeof(ElementK) + BlockMmadQK::L0_TILE_M * embed_ * sizeof(ElementQ);
        uint32_t vBufBytes = blockSize_ * embed_ * sizeof(ElementV);
        uint32_t l1PStageBytes = RoundUp(BlockMmadQK::L0_TILE_M, 16U) * RoundUp(blockSize_, 16U) * sizeof(ElementP);
        AscendC::LocalTensor<ElementP> l1PBuf[P_L1_BUF_NUM];
        for (uint32_t i = 0; i < P_L1_BUF_NUM; i++) {
            l1PBuf[i] = resource.l1Buf.template GetBufferByByte<ElementP>(qkL1Used + vBufBytes + l1PStageBytes * i);
        }
#endif

        // Seed packedRow=0 with the first batch that owns a KV block. Batches
        // with kv_len=0 do not own a packed CSR row and must be skipped.
        int32_t curPackedRow = 0;
        uint32_t batchIdx = 0;
        uint32_t kvBlockIdx = 0;
        InitPackedRowCoord(batchIdx, kvBlockIdx);

        // ===== Main loop: MODE_2 cross-core sync, full pipeline (QK || softmax || PV) =====
        // === Phase1 主循环: 按 taskIdx stride 分核遍历所有 Phase1 任务 ===
        for (uint32_t taskIdx = coreIdx; taskIdx < totalTaskNumP1_; taskIdx += coreNum) {
            // taskIdx → (packedRow, kvHeadIdx)
            uint32_t packedRow = taskIdx / kvHeads_;
            uint32_t kvHeadIdx = taskIdx % kvHeads_;
            if (packedRow >= totalKvRows_) {
                continue;
            }
            // --- MSA coord 增量步进: packedRow → (batchIdx, kvBlockIdx) ---
            // 状态机跨 task 持久化, AdvancePackedRowCoord 只向前步进
            while (curPackedRow < static_cast<int32_t>(packedRow)) {
                AdvancePackedRowCoord(batchIdx, kvBlockIdx);
                curPackedRow++;
            }

            // --- 获取当前 batch 的 Q/KV seqlen 等标量 ---
            // Q/KV 同 batch (无跨 batch attention), qBatch == batchIdx
            uint32_t kvSeqlenBatch = 0;
            uint32_t numBlocksB = 0;
            uint32_t cumQStart = 0;
            uint32_t qSeqlenBatch = 0;
            uint32_t cumKvStart = 0;
            ResolveBatchQSide(batchIdx, kvSeqlenBatch, numBlocksB, cumQStart, qSeqlenBatch, cumKvStart);
            uint32_t localBlockIdx = kvBlockIdx;

            // --- 从 CSR 读取选择该 block 的 Q token 列表 ---
            // csrStart = k2q_row_ptr[kvHeadIdx, packedRow]
            // csrEnd   = k2q_row_ptr[kvHeadIdx, packedRow + 1]
            // numQTokens = csrEnd - csrStart (选择该 block 的 Q token 数)
            uint64_t rowPtrBase = static_cast<uint64_t>(kvHeadIdx) * (totalKvRows_ + 1);
            uint32_t csrStart = static_cast<uint32_t>(gK2qRowPtr.GetValue(rowPtrBase + packedRow));
            uint32_t csrEnd = static_cast<uint32_t>(gK2qRowPtr.GetValue(rowPtrBase + packedRow + 1));
            uint32_t numQTokens = csrEnd - csrStart;
            if (numQTokens == 0) {
                continue; // 空 CSR 行: 无 Q token 选择该 block, 跳过
            }

            // --- 计算该 KV block 的有效 token 数 ---
            // 非尾块: validSize = blockSize; 尾块: validSize = kvSeqlen % blockSize
            uint32_t validSize = CalcKvBlockValidSize(kvSeqlenBatch, numBlocksB, localBlockIdx);
            if (validSize == 0) {
                continue;
            }

            uint64_t csrDataBase = static_cast<uint64_t>(kvHeadIdx) * k2qNnzUpperBound_;
            uint32_t kvStartPos = localBlockIdx * blockSize_; // KV block 在序列中的起始位置

#ifdef __DAV_VEC__
            // === VEC-only CSR batch-read: DataCopyPad q_indices/slot_indices GM→UB ===
            // Replaces per-element GlobalTensor::GetValue (blacklisted, ~100 cycles each)
            // with one batched DMA per task + LocalTensor::GetValue (~few cycles each).
            // CUBE keeps original GlobalTensor::GetValue (CUBE scalar is not the bottleneck).
            if (numQTokens <= CSR_UB_MAX_QTOKENS) {
                AscendC::LocalTensor<int32_t> ubQIdx =
                    resource.ubBuf.template GetBufferByByte<int32_t>(CSR_UB_QINDICES_OFFSET);
                AscendC::LocalTensor<int32_t> ubSlotK =
                    resource.ubBuf.template GetBufferByByte<int32_t>(CSR_UB_SLOTK_OFFSET);
                AscendC::DataCopyExtParams csrCp;
                csrCp.blockCount = 1U;
                csrCp.blockLen = static_cast<uint32_t>(numQTokens * sizeof(int32_t));
                csrCp.srcStride = 0;
                csrCp.dstStride = 0;
                AscendC::DataCopyPadExtParams<int32_t> csrPad(false, 0, 0, 0);
                AscendC::DataCopyPad(ubQIdx, gK2qQIndices[csrDataBase + csrStart], csrCp, csrPad);
                AscendC::DataCopyPad(ubSlotK, gK2qSlotIndices[csrDataBase + csrStart], csrCp, csrPad);
                // DataCopyPad uses MTE2 and the CSR entries are read with
                // LocalTensor::GetValue on Scalar below.
                // Static-tensor programming reserves event IDs 6/7.  This is
                // the only MTE2_S dependency in the A2 path, so ID0 is free.
                AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            }
#endif

#ifdef __DAV_CUBE__
            // --- CUBE: 加载 K/V 到 L1 (常驻, 每个 task 仅加载一次) ---
            uint64_t kvHeadOffset = static_cast<uint64_t>(kvHeadIdx) * embed_;
            uint64_t keyBlockBase;
            uint64_t valueBlockBase;
            if (isKvContinuous_) {
                uint64_t tokenStart =
                    static_cast<uint64_t>(cumKvStart) + static_cast<uint64_t>(localBlockIdx) * blockSize_;
                keyBlockBase = tokenStart * keyTokenStride_ + kvHeadOffset;
                valueBlockBase = tokenStart * valueTokenStride_ + kvHeadOffset;
            } else {
                int64_t btOffset = static_cast<int64_t>(batchIdx) * maxBlocksPerBatch_ + localBlockIdx;
                int32_t physicalBlockId = gBlockTable.GetValue(btOffset);
                keyBlockBase = static_cast<uint64_t>(physicalBlockId) * keyBlockStride_ + kvHeadOffset;
                valueBlockBase = static_cast<uint64_t>(physicalBlockId) * valueBlockStride_ + kvHeadOffset;
            }

            auto gmKLayout = tla::MakeLayout<ElementK, LayoutK>(keyRowStride, blockSize_);
            auto gmKTensor = tla::MakeTensor(gK[keyBlockBase], gmKLayout, Arch::PositionGM{});
            blockMmadQK.LoadKResident(gmKTensor, validSize, embed_);

            auto gmVLayout = tla::MakeLayout<ElementV, LayoutV>(blockSize_, valueRowStride);
            auto gmVTensor = tla::MakeTensor(gV[valueBlockBase], gmVLayout, Arch::PositionGM{});
            blockMmadPV.LoadVResident(gmVTensor, validSize, embed_);
#endif

            // === 批量 QK/softmax/PV: 将 numQTokens 按 batchGroupsMax 分批 ===
            // batchGroupsMax = min(L0_TILE_M/groupSize, MAX_BATCH_GROUPS)
            // 输入保证: 每个 qToken 的 causalValidLen >= 1 (有因果重叠)
            uint32_t numBatches = CeilDiv(numQTokens, batchGroupsMax);

            // qHeadStart is shared by the CUBE QK qBase computation below.
            uint32_t qHeadStart = kvHeadIdx * groupSize_;
            // PV wsOOff arithmetic precomputes (== SlotOOffset terms), constant within this
            // kvHead: perQTokenStride = kvHeads*topK*slotOElems, kvHeadBase = kvHeadIdx*topK*slotOElems.
            uint64_t perQTokenStride = static_cast<uint64_t>(kvHeads_) * topK_ * slotOElems_;
            uint64_t kvHeadBase = static_cast<uint64_t>(kvHeadIdx) * topK_ * slotOElems_;
            // QK qBase arithmetic precomputes (== qToken*qHeads*embed + qHeadStart*embed),
            // constant within this kvHead.
            uint64_t perQTokenStrideQ = queryTokenStride_;
            uint64_t qHeadStartBase = static_cast<uint64_t>(qHeadStart) * embed_;

            // qToken/slotK ring buffer: batch bi 的身份在 bi 填充, 在 bi+PRE_LAUNCH 被 PV 消费
            // (从 ring row 读取, 无需 GM 重读)。PRE_LAUNCH+1 级保持数据存活跨越 2-deep gap。
            // 派生偏移 (qBase/cvl/lseOff/wsOOff) 在使用点即时计算; 只有多次使用的 qToken/slotK 被缓存。
            constexpr uint32_t BATCH_RING_STAGES = PRE_LAUNCH + 1U;
            uint32_t batchQToken[BATCH_RING_STAGES][MAX_BATCH_GROUPS] = {};
            uint32_t batchSlotK[BATCH_RING_STAGES][MAX_BATCH_GROUPS] = {};

            // === 三级流水线主循环 ===
            // 循环范围: [0, numBatches + PRE_LAUNCH)
            //   bi < numBatches:          执行 QK + softmax (收集 + 计算)
            //   bi >= PRE_LAUNCH:         执行 PV (滞后 PRE_LAUNCH=2 个 batch)
            //   两段重叠: PRE_LAUNCH ≤ bi < numBatches 时 QK/softmax/PV 同时执行 (流水线满载)
            //   循环末尾 bi ≥ numBatches 时只剩 PV 排空 (QK/softmax 已完成)
            for (uint32_t bi = 0U; bi < numBatches + PRE_LAUNCH; bi++) {
                uint32_t stage = bi % BATCH_RING_STAGES;

                // --- QK + softmax for batch bi (collect + compute) ---
                if (bi < numBatches) {
                    uint32_t rem = numQTokens - bi * batchGroupsMax;
                    uint32_t batchGroupCount = (rem < batchGroupsMax) ? rem : batchGroupsMax;
                    uint32_t batchM = batchGroupCount * groupSize_;
                    // Alternate the owner of an odd query-group batch.  The
                    // two AIVs still process identical rows and reduction
                    // order; only the producer/consumer is rebalanced.
                    bool extraToSubBlock0 = ((taskIdx + bi) & 1U) == 0U;
#ifdef __DAV_VEC__
                    // VEC 侧: 计算每个 qToken 的 causalValidLen (causal mask 有效列长)
                    // 纯计算, 无 GM 访问 (cumQStart/qSeqlenBatch/kvSeqlenBatch 已预取)
                    uint32_t cvlArr[MAX_BATCH_GROUPS];
#endif
                    // 从 CSR gather Q token id 和 slotK
                    for (uint32_t g = 0U; g < batchGroupCount; g++) {
                        uint32_t csrRelIdx = bi * batchGroupsMax + g;
#ifdef __DAV_VEC__
                        // VEC: read from UB (batch-loaded via DataCopyPad above)
                        if (numQTokens <= CSR_UB_MAX_QTOKENS) {
                            AscendC::LocalTensor<int32_t> ubQIdx =
                                resource.ubBuf.template GetBufferByByte<int32_t>(CSR_UB_QINDICES_OFFSET);
                            AscendC::LocalTensor<int32_t> ubSlotK =
                                resource.ubBuf.template GetBufferByByte<int32_t>(CSR_UB_SLOTK_OFFSET);
                            batchQToken[stage][g] = static_cast<uint32_t>(ubQIdx.GetValue(csrRelIdx));
                            batchSlotK[stage][g] = static_cast<uint32_t>(ubSlotK.GetValue(csrRelIdx));
                        } else {
                            batchQToken[stage][g] =
                                static_cast<uint32_t>(gK2qQIndices.GetValue(csrDataBase + csrStart + csrRelIdx));
                            batchSlotK[stage][g] =
                                static_cast<uint32_t>(gK2qSlotIndices.GetValue(csrDataBase + csrStart + csrRelIdx));
                        }
#else
                        // CUBE: read from GM (original path)
                        batchQToken[stage][g] =
                            static_cast<uint32_t>(gK2qQIndices.GetValue(csrDataBase + csrStart + csrRelIdx));
                        batchSlotK[stage][g] =
                            static_cast<uint32_t>(gK2qSlotIndices.GetValue(csrDataBase + csrStart + csrRelIdx));
#endif
#ifdef __DAV_VEC__
                        // 计算 causal mask: 该 qToken 在此 KV block 中的有效列数
                        cvlArr[g] = CalcCausalValidLen(batchQToken[stage][g], validSize, kvStartPos, cumQStart,
                                                       qSeqlenBatch, kvSeqlenBatch);
#endif
                    }

                    // 流水线 buffer/stage id
                    uint32_t ubSBufId = bi % UB_S_OTMP_BUF_STAGES;
                    uint32_t l1PBufId = bi % P_L1_BUF_NUM;
                    // 跨核同步 flag: mm1ToSm (Cube QK → VEC softmax), smToMm2 (VEC softmax → Cube PV)
                    uint32_t mm1ToSmFlagId = ubSBufId;
                    uint32_t smToMm2FlagId = l1PBufId + UB_S_OTMP_BUF_STAGES;
                    Arch::CrossCoreFlag mm1ToSmFlag(mm1ToSmFlagId);
                    Arch::CrossCoreFlag smToMm2Flag(smToMm2FlagId);

#ifdef __DAV_CUBE__
                    // CUBE: QK matmul — S = Q_tile[batchM, D] × K[D, validSize]
                    // Q 从 GM 逐行 gather (qToken 非连续), 结果 S 写入 UB (fixpipe)
                    // One AIC/Cube produces the complete S tile consumed by
                    // both Vector subblocks.  Only the Vector softmax is
                    // split 1:2; sizing Cube's S tensor to half M truncates
                    // the second Vector half and corrupts the next stage.
                    uint32_t mAlignedB = RoundUp(batchM, 16u);
                    auto ubSLayout = tla::MakeLayout<ElementS, LayoutS>(mAlignedB, RoundUp(validSize, 16U));
                    auto ubSTensorTla = tla::MakeTensor(ubSTensor[ubSBufId], ubSLayout, Arch::PositionUB{});
                    if (AscendC::GetBlockIdx() == 0U) {
                        A2_PRINTF("[CUBE] before QK bi=%u numBatches=%u\n", bi, numBatches);
                    }
                    blockMmadQK(gQ, batchQToken[stage], batchGroupCount, groupSize_, ubSTensorTla, validSize, embed_,
                                perQTokenStrideQ, qHeadStartBase, numBatches, bi, mm1ToSmFlag);
                    if (AscendC::GetBlockIdx() == 0U) {
                        A2_PRINTF("[CUBE] after QK bi=%u\n", bi);
                    }
#endif
#ifdef __DAV_VEC__
                    // VEC: online softmax — 对 S 做 rowMax/rowSum, 生成 P=exp(S-rowMax) cast bf16 写 L1
                    // cvlArr 用于 mask: [cvl, validSize) 列置零 (causal mask)
                    auto l1PLayout = tla::MakeLayout<ElementP, layout::zN>(batchM, validSize);
                    auto l1PTensor = tla::MakeTensor(l1PBuf[l1PBufId], l1PLayout, Arch::PositionL1{});
                    epilogueSoftmax(l1PTensor, GemmCoord{batchM, validSize, embed_}, ubSBufId, l1PBufId, mm1ToSmFlag,
                                    smToMm2Flag, cvlArr, batchGroupCount, groupSize_, bi, taskIdx == 0U,
                                    extraToSubBlock0);
                    // VEC: scatter rowMax/rowSum to workspace GM
                    ScatterBatchStats(gSoftmaxMax, gSoftmaxSum, ubSBufId, batchM, groupSize_, batchQToken[stage],
                                      batchSlotK[stage], kvHeadIdx, batchGroupCount, extraToSubBlock0);
#endif
                }

                // --- PV for batch bi-PRE_LAUNCH (滞后 PRE_LAUNCH 个 batch) ---
                // 从 ring buffer 读取 batch bi-PRE_LAUNCH 的 qToken/slotK (在 bi=bDe 时填充)
                if (bi >= PRE_LAUNCH) {
                    uint32_t bDe = bi - PRE_LAUNCH;
                    if (bDe < numBatches) {
                        // P for logical batch bDe was produced in outer
                        // iteration bDe, then consumed PRE_LAUNCH iterations
                        // later. The GM stage and cross-core flag must retain
                        // that producer identity, rather than use bi.
                        uint32_t pStage = bDe % P_L1_BUF_NUM;
                        uint32_t smToMm2FlagId = pStage + UB_S_OTMP_BUF_STAGES;
                        Arch::CrossCoreFlag smToMm2Flag(smToMm2FlagId);
                        uint32_t stageDe = bDe % BATCH_RING_STAGES;
#ifdef __DAV_CUBE__
                        // CUBE: PV matmul — O_partial = P[batchM, validSize] × V[validSize, D]
                        // 结果 O_partial (未归一化) 写入 workspace accumOutGm
                        uint32_t pvRem = numQTokens - bDe * batchGroupsMax;
                        uint32_t pvGrpCnt = (pvRem < batchGroupsMax) ? pvRem : batchGroupsMax;
                        blockMmadPV.SetL1PBuf(pStage);
                        if (AscendC::GetBlockIdx() == 0U) {
                            A2_PRINTF("[CUBE] before PV bDe=%u numBatches=%u\n", bDe, numBatches);
                        }
                        blockMmadPV(gAccumOut, batchQToken[stageDe], batchSlotK[stageDe], pvGrpCnt, groupSize_, embed_,
                                    perQTokenStride, kvHeadBase, slotOElems_, numBatches, bDe, pStage, smToMm2Flag);
                        if (AscendC::GetBlockIdx() == 0U) {
                            A2_PRINTF("[CUBE] after PV bDe=%u\n", bDe);
                        }
                        // Dump O_partial (PV result) for each Q token in this batch
                        // {
                        //     uint32_t groupRows = groupSize_;
                        //     uint32_t nAct = embed_;
                        //     for (uint32_t g = 0U; g < pvGrpCnt; g++) {
                        //         uint64_t wsOOff = static_cast<uint64_t>(batchQToken[stageDe][g]) * perQTokenStride
                        //                         + kvHeadBase
                        //                         + static_cast<uint64_t>(batchSlotK[stageDe][g]) * slotOElems_;
                        //         printf("wsOOff = %d\n", wsOOff);
                        //         printf("groupRows = %d\n", groupRows);
                        //         printf("nAct = %d\n", nAct);
                        //         AscendC::DumpTensor(gAccumOut[wsOOff], 981, groupRows * nAct);
                        //     }
                        // }
#endif
                    }
                }
            }
#ifdef __DAV_CUBE__
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
#endif
        }
#ifdef __DAV_VEC__
        // Drain both stats stages before Phase2 reads their GM workspace.
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID5);
#endif
    }

    // Phase 2: FlashDecode-style combine (合并 topK 个部分结果)
    //
    // 【为什么需要 Phase2?】
    //   Phase1 中, 每个 Q token 的 topK 个 KV block 分别产出了 O_partial[k] 和 rowMax[k]/rowSum[k]
    //   但这些是 "局部" 的: 每个 O_partial[k] 只用了 1 个 KV block 的 V
    //   Phase2 需要把 topK 个局部结果 "合并" 成最终输出 O:
    //
    //     O = Σ_k scale[k] × (O_partial[k] / rowSum[k])
    //
    //   其中 scale[k] 是归一化权重 (类似 FlashAttention 的 combine 公式):
    //     scale[k] = rowSum[k] × exp(rowMax[k] - globalMax) / Σ_j(rowSum[j] × exp(rowMax[j] - globalMax))
    //
    // 【仅 VEC 执行, CUBE 空转】
    //   Phase2 是纯向量运算 (exp, div, reduce, cast), 不需要 Cube 的矩阵乘能力
    //
    // 【5 个子步骤】 (由 EpilogueRescaleO 组件实现):
    //   1. CopyLseIn:       从 GM 读 compact max/sum, Broadcast 到 UB [topK, dealRow, 8]
    //   2. ComputeScaleValue: 计算 scale 权重 scale[k] = rowSum[k]*exp(rowMax[k]-max_global) / Σ
    //   3. CopyAccumOutIn:  读 O_partial, 从 gmSoftmaxSum 读 rowSum, O_norm = O_partial / rowSum
    //   4. ReduceFinalRes:  out += scale[k] * O_norm[k], 遍历 topK 个 split
    //   5. CopyFinalResOut: cast fp32→bf16 (如需), 写 GM attentionOut
    //
    // 【无效 slot 跳过】
    //   如果 rowMax == -inf (InitWorkspaceStats 预置的初始值, 说明该 slot 没被 Phase1 写入):
    //     → scale=0, 跳过 CopyAccumOutIn/ReduceFinalRes, 不贡献到输出
    //
    // 【分核】
    //   coreIdx = GetBlockIdx() (线性 AIV id, 每个 AIV 子核独立编号)
    //   coreNum = GetBlockNum() × GetSubBlockNum() (覆盖所有 AIV 子核)
    //   for taskIdx = coreIdx; taskIdx < totalTaskNumP2; taskIdx += coreNum
    // DEBUG_MODE: 0=normal, 1=dump softmaxMax, 2=dump softmaxSum,
    //             3=dump accumOut (raw O_partial), 4=dump S from GM workspace
    __aicore__ inline void Phase2CombineScale(AscendC::GlobalTensor<ElementO> &gO,
                                              AscendC::GlobalTensor<float> &gSoftmaxLse,
                                              AscendC::GlobalTensor<ElementWorkspaceO> &gAccumOut,
                                              AscendC::GlobalTensor<float> &gSoftmaxMax,
                                              AscendC::GlobalTensor<float> &gSoftmaxSum)
    {
#ifdef __DAV_VEC__
#if DEBUG_MODE == 1 || DEBUG_MODE == 2 || DEBUG_MODE == 3
        // Debug: dump workspace data to output
        if (true) {
            AscendC::LocalTensor<float> ubBase = resource.ubBuf.template GetBufferByByte<float>(0);
            uint32_t subBlockNum = AscendC::GetSubBlockNum();
            uint32_t coreIdx = AscendC::GetBlockIdx();
            uint32_t coreNum = AscendC::GetBlockNum() * subBlockNum;
            // Reserve enough room for the largest supported GQA row tile.
            // DEBUG_MODE=3 reads fp32 O_partial and must cast it through the
            // normal float->bf16 path before returning it through the output
            // contract; a reinterpret cast would turn fp32 bit patterns into
            // invalid bf16 values (including NaNs).
            uint32_t debugDumpElems = groupSize_ * embed_;
            auto castBuf = ubBase[debugDumpElems].template ReinterpretCast<bfloat16_t>();
            if (coreNum == 0U) {
                coreNum = 1U;
            }
            for (uint32_t taskIdx = coreIdx; taskIdx < totalTaskNumP2_; taskIdx += coreNum) {
                uint32_t qToken = taskIdx / kvHeads_;
                uint32_t kvHeadIdx = taskIdx % kvHeads_;
                uint64_t attenOutOffset = static_cast<uint64_t>(qToken) * kvHeads_ * groupSize_ * embed_ +
                                          static_cast<uint64_t>(kvHeadIdx) * groupSize_ * embed_;
                uint32_t statStride = RoundUp(groupSize_, 8U);
                uint64_t taskStatBase = static_cast<uint64_t>(taskIdx) * topK_ * statStride;
#if DEBUG_MODE == 1
                // Dump compact [topK, groupSize] softmaxMax from the padded GM
                // workspace.  The host compare script maps these first values
                // directly to golden phase1_ws_max[q, kv, slot, group].
                AscendC::Duplicate<float>(ubBase, 0.0f, debugDumpElems);
                for (uint32_t slot = 0U; slot < topK_; ++slot) {
                    for (uint32_t group = 0U; group < groupSize_; ++group) {
                        uint32_t compactOff = slot * groupSize_ + group;
                        uint64_t paddedOff = taskStatBase + slot * statStride + group;
                        ubBase.SetValue(compactOff, gSoftmaxMax.GetValue(paddedOff));
                    }
                }
                if (taskIdx < 8U) {
                    A2_PRINTF("[A2 p2debug] block=%u sub=%u task=%u q=%u kv=%u group=%u off=%u max0=%f\n",
                              AscendC::GetBlockIdx(), AscendC::GetSubBlockIdx(), taskIdx, qToken, kvHeadIdx, groupSize_,
                              static_cast<uint32_t>(attenOutOffset), ubBase.GetValue(0));
                }
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast<bfloat16_t, float, false>(castBuf, ubBase, AscendC::RoundMode::CAST_RINT,
                                                        static_cast<uint64_t>(0), 1U,
                                                        AscendC::UnaryRepeatParams(1, 1, 4, 8));
                AscendC::PipeBarrier<PIPE_V>();
#elif DEBUG_MODE == 2
                // Dump compact [topK, groupSize] softmaxSum from the padded GM workspace.
                AscendC::Duplicate<float>(ubBase, 0.0f, debugDumpElems);
                for (uint32_t slot = 0U; slot < topK_; ++slot) {
                    for (uint32_t group = 0U; group < groupSize_; ++group) {
                        uint32_t compactOff = slot * groupSize_ + group;
                        uint64_t paddedOff = taskStatBase + slot * statStride + group;
                        ubBase.SetValue(compactOff, gSoftmaxSum.GetValue(paddedOff));
                    }
                }
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast<bfloat16_t, float, false>(castBuf, ubBase, AscendC::RoundMode::CAST_RINT,
                                                        static_cast<uint64_t>(0), 1U,
                                                        AscendC::UnaryRepeatParams(1, 1, 4, 8));
                AscendC::PipeBarrier<PIPE_V>();
#elif DEBUG_MODE == 3
                // Dump accumOut: copy directly from GM workspace
                AscendC::DataCopyExtParams copyParams;
                copyParams.blockCount = groupSize_;
                copyParams.blockLen = static_cast<uint16_t>(embed_ * sizeof(ElementWorkspaceO));
                copyParams.srcStride = 0;
                copyParams.dstStride = 0;
                AscendC::DataCopyPadExtParams<ElementWorkspaceO> padParams(false, 0, 0, 0);
                AscendC::DataCopyPad(ubBase.template ReinterpretCast<ElementWorkspaceO>(),
                                     gAccumOut[taskIdx * topK_ * groupSize_ * embed_], copyParams, padParams);
                // DataCopyPad is MTE2 asynchronous.  The following UB->GM
                // debug copy must wait for the input to be visible in UB;
                // a same-pipe barrier alone does not create that dependency.
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
#endif
                AscendC::SetVectorMask<int8_t>(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
                AscendC::Cast<ElementO, float, false>(castBuf, ubBase, AscendC::RoundMode::CAST_RINT,
                                                      static_cast<uint64_t>(0), (groupSize_ * embed_ + 63U) / 64U,
                                                      AscendC::UnaryRepeatParams(1, 1, 4, 8));
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::DataCopyExtParams outParams;
#if DEBUG_MODE == 3
                outParams.blockCount = groupSize_;
                outParams.blockLen = static_cast<uint16_t>(embed_ * sizeof(ElementO));
#else
                outParams.blockCount = 1U;
                outParams.blockLen = static_cast<uint16_t>(debugDumpElems * sizeof(ElementO));
#endif
                outParams.srcStride = 0;
                outParams.dstStride = 0;
                AscendC::DataCopyPadExtParams<ElementO> outPadParams(false, 0, 0, 0);
#if DEBUG_MODE == 3
                AscendC::DataCopyPad(gO[attenOutOffset], castBuf, outParams);
#else
                AscendC::DataCopyPad(gO[attenOutOffset], castBuf, outParams);
#endif
                // Debug modes intentionally reuse one UB area for every task.
                // Finish the transport before the next iteration overwrites it;
                // probe correctness matters more than overlap here.
                AscendC::PipeBarrier<PIPE_ALL>();
            }
        }
#else // DEBUG_MODE == 0
        AscendC::LocalTensor<float> ubBase = resource.ubBuf.template GetBufferByByte<float>(0);

        EpilogueRescaleO epilogueRescaleO;
        epilogueRescaleO.InitFDBuffers(ubBase, embed_, groupSize_, topK_, kvHeads_);
        // The two AIVs of one MIX block execute the same Phase2 task; the
        // epilogue splits that task's GQA rows by GetSubBlockIdx().
        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        uint32_t coreIdx = AscendC::GetBlockIdx() / subBlockNum;
        uint32_t coreNum = AscendC::GetBlockNum();
        if (coreNum == 0U) {
            coreNum = 1U;
        }
        for (uint32_t taskIdx = coreIdx; taskIdx < totalTaskNumP2_; taskIdx += coreNum) {
            epilogueRescaleO.FlashDecodeCompute(taskIdx, totalTaskNumP2_, gO, gSoftmaxLse, gAccumOut, gSoftmaxMax,
                                                gSoftmaxSum, qHeads_, kvHeads_, softmaxLseFlag_);
        }
#endif
#endif
    }

private:
    Arch::Resource<ArchTag> resource;
    AscendC::GlobalTensor<ElementS> gmSWorkspace_; // A2: GM S workspace for L0C→GM→UB
    uint64_t gmSWorkspaceSize_ = 0;                // A2: per-core GM S staging bytes (from tiling)
    uint64_t gmSStageElems_ = 0;                   // A2: elems per GM S stage (for VEC offset)

    AscendC::GlobalTensor<ElementP> gmPWorkspace_; // A2: GM P workspace for UB→GM→L1
    uint64_t gmPWorkspaceSize_ = 0;                // A2: per-core GM P staging bytes (from tiling)
    uint64_t gmPStageElems_ = 0;                   // A2: elems per GM P stage (for VEC/CUBE offset)

    uint32_t batch_;
    uint32_t qHeads_;
    uint32_t kvHeads_;
    uint32_t groupSize_;
    uint32_t embed_;
    uint32_t blockSize_;
    uint64_t queryTokenStride_;
    uint64_t keyBlockStride_;
    uint64_t valueBlockStride_;
    bool isKvContinuous_;
    uint64_t keyTokenStride_;
    uint64_t valueTokenStride_;
    uint32_t topK_;
    uint32_t totalKvRows_;
    uint32_t maxBlocksPerBatch_;
    uint32_t k2qNnzUpperBound_;
    uint32_t totalTaskNumP1_;
    uint32_t totalTaskNumP2_;
    float scaleValue_;
    bool softmaxLseFlag_ = false;
    uint64_t accumOutSize_;
    uint64_t lseStatSize_;
    uint64_t slotOElems_;
    uint64_t slotStatElems_;

    AscendC::GlobalTensor<int32_t> gActualQseqlen_;
    AscendC::GlobalTensor<int32_t> gActualKvseqlen_;
};

} // namespace MinimaxSaSplitKvKernelA2

#endif // MINIMAX_SPARSE_ATTENTION_SPLIT_KV_KERNEL_A2_H
