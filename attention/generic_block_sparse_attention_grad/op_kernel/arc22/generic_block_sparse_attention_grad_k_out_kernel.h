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
 * \file generic_block_sparse_attention_grad_kernel.h
 * \brief Block Sparse Attention Grad Kernel Implementation
 */

#ifndef GENERIC_BLOCK_SPARSE_ATTENTION_GRAD_K_OUT_KERNEL_H
#define GENERIC_BLOCK_SPARSE_ATTENTION_GRAD_K_OUT_KERNEL_H

#include "attn_infra/gbsag_base_defs.hpp"
#include "attn_infra/arch/gbsag_arch.hpp"
#include "attn_infra/arch/gbsag_resource.hpp"
#include "attn_infra/layout/gbsag_layout.hpp"

#include "attn_infra/gemm/block/gbsag_block_mmad.hpp"
#include "attn_infra/gemm/gbsag_gemm_dispatch_policy.hpp"
#include "attn_infra/gemm/gbsag_gemm_type.hpp"
#include "attn_infra/epilogue/block/gbsag_epilogue.hpp"
#include "attn_infra/epilogue/gbsag_epilogue_dispatch_policy.hpp"
#include "attn_infra/epilogue/block/gbsag_epilogue_fag_pre.hpp"
#include "attn_infra/epilogue/block/gbsag_epilogue_post.hpp"
#include "attn_infra/epilogue/block/gbsag_epilogue_softmaxgrad.hpp"
#include "attn_infra/epilogue/block/gbsag_gather_and_scatter.hpp"
#include "attn_infra/epilogue/block/gbsag_epilogue_simply_softmax.hpp"

using namespace NpuArch;

namespace GBSAG {
// 三槽软流水跨核 flag：第 3 槽使用 id 12/11，避开 SyncAll barrier 保留 id 8/9/10 的借用风险。
constexpr uint32_t CUBE_VEC_FLAG[3] = {7, 8, 12};
constexpr uint32_t VEC_CUBE_FLAG[3] = {9, 10, 11};
constexpr uint32_t CUBE_POST_FLAG = 0;

// K_OUT-local workspace constants formerly inherited from the monolithic kernel header.
constexpr uint32_t WORKSPACE_TILE_ROWS = 128;
constexpr uint32_t WORKSPACE_TILE_COLS = 128;
// 三槽软流水深度：Scatter 延后 2 拍 + 预取 1 拍，共 3 个 in-flight packet。
constexpr uint32_t WORKSPACE_SOFTPIPE_SLOT_NUM = 3;
// WORKSPACE_PINGPONG_COUNT 保持 2：既是 S/dP 槽内「FP32 tile + 低精度 P/dS tile」
// 的成对因子，也是 Cube mmad L1 pingpong 深度因子，与软流水槽数无关。
constexpr uint32_t WORKSPACE_PINGPONG_COUNT = 2;
constexpr uint32_t PACKET_DOUT_BASE_SLOT = WORKSPACE_SOFTPIPE_SLOT_NUM;
constexpr uint32_t PACKET_OUT_BASE_SLOT = 2 * WORKSPACE_SOFTPIPE_SLOT_NUM;
constexpr uint32_t PACKET_DQ_BASE_SLOT = 3 * WORKSPACE_SOFTPIPE_SLOT_NUM;
constexpr uint32_t FP32_TO_LOW_PRECISION_ELEMENT_RATIO = sizeof(float) / sizeof(uint16_t);
constexpr uint32_t NEXT_SEQUENCE_INDEX_OFFSET = 1;
constexpr uint64_t WORKSPACE_TILE_ELEMENTS = static_cast<uint64_t>(WORKSPACE_TILE_ROWS) * WORKSPACE_TILE_COLS;
constexpr uint64_t WORKSPACE_BLOCK_SIZE = WORKSPACE_TILE_ELEMENTS * WORKSPACE_PINGPONG_COUNT;
constexpr uint64_t WORKSPACE_P16_OFFSET_ELEMENT = WORKSPACE_BLOCK_SIZE;
constexpr uint64_t WORKSPACE_P16_OFFSET = WORKSPACE_P16_OFFSET_ELEMENT * sizeof(uint16_t);
// 每核 S/dP 软流水区 = 单槽步长 × 槽数（三槽）。
constexpr uint64_t WORKSPACE_BLOCK_SIZE_DB = WORKSPACE_BLOCK_SIZE * WORKSPACE_SOFTPIPE_SLOT_NUM;
constexpr uint64_t CUBE2_L1_OFFSET_BYTES = 131072;
constexpr uint64_t CUBE3_L1_OFFSET_BYTES = CUBE2_L1_OFFSET_BYTES * WORKSPACE_PINGPONG_COUNT;
constexpr uint32_t CUBE2_PINGPONG_OFFSET = WORKSPACE_PINGPONG_COUNT;
constexpr uint32_t CUBE3_PINGPONG_OFFSET = CUBE2_PINGPONG_OFFSET * WORKSPACE_PINGPONG_COUNT;

// Device-side Metadata ABI offsets. Keep these aligned with the host/AICPU layout header.
namespace GBSAGMeta {
constexpr uint32_t OFF_TOTAL_TASK_NUM = 0;
constexpr uint32_t OFF_TASK_NUM_PER_CORE = 1;
constexpr uint32_t OFF_TAIL_TASK_NUM = 2;
constexpr uint32_t OFF_TASK_TABLE_OFFSET = 3;
constexpr uint32_t OFF_CORE_NUM = 4;
constexpr uint32_t OFF_TASK_WORDS = 5;
constexpr uint32_t TASK_BEGIN_BATCH = 0;
constexpr uint32_t TASK_BEGIN_KV_HEAD = 1;
constexpr uint32_t TASK_BEGIN_KV_SEQ_OFFSET = 2;
} // namespace GBSAGMeta

template <class BlockMmadGBSAG1_, class BlockMmadGBSAG2_, class BlockMmadGBSAG3_, class EpilogueFAGPre_,
          class EpilogueFAGSfmg_, class EpilogueFAGOp_, class EpilogueFAGPost_, uint32_t INPUT_LAYOUT>
class GenericBlockSparseAttentionGradKOutKernel {
public:
    /// Parameters structure
    struct Params {
        // Data members
        GM_ADDR dout;
        GM_ADDR q;
        GM_ADDR k;
        GM_ADDR v;
        GM_ADDR out;
        GM_ADDR softmaxLse;
        GM_ADDR rsvdBlockIdx;
        GM_ADDR rsvdBlockCount;
        GM_ADDR metadata;
        GM_ADDR attentionMask;
        GM_ADDR cuSeqLengthsQ;
        GM_ADDR cuSeqLengthsKv;
        GM_ADDR actualQseqlen;
        GM_ADDR actualKvseqlen;
        GM_ADDR dq;
        GM_ADDR dk;
        GM_ADDR dv;
        GM_ADDR workspace;
        GM_ADDR tiling;

        // Methods
        __aicore__ inline Params() {}

        __aicore__ inline Params(GM_ADDR dout_, GM_ADDR q_, GM_ADDR k_, GM_ADDR v_, GM_ADDR out_, GM_ADDR softmaxLse_,
                                 GM_ADDR rsvdBlockIdx_, GM_ADDR rsvdBlockCount_, GM_ADDR metadata_,
                                 GM_ADDR attentionMask_, GM_ADDR cuSeqLengthsQ_, GM_ADDR cuSeqLengthsKv_,
                                 GM_ADDR actualQseqlen_, GM_ADDR actualKvseqlen_, GM_ADDR dq_, GM_ADDR dk_, GM_ADDR dv_,
                                 GM_ADDR workspace_, GM_ADDR tiling_data_)
            : dout(dout_),
              q(q_),
              k(k_),
              v(v_),
              out(out_),
              softmaxLse(softmaxLse_),
              rsvdBlockIdx(rsvdBlockIdx_),
              rsvdBlockCount(rsvdBlockCount_),
              metadata(metadata_),
              cuSeqLengthsQ(cuSeqLengthsQ_),
              cuSeqLengthsKv(cuSeqLengthsKv_),
              attentionMask(attentionMask_),
              actualQseqlen(actualQseqlen_),
              actualKvseqlen(actualKvseqlen_),
              dq(dq_),
              dk(dk_),
              dv(dv_),
              workspace(workspace_),
              tiling(tiling_data_)
        {}
    };

    // Public entry points: construct parameters, initialize, and run the kernel.
    __aicore__ inline GenericBlockSparseAttentionGradKOutKernel() {}

    /**
     * @brief GBSAG K_OUT Kernel 统一入口，仅编排初始化和计算流程。
     * @param params Kernel 输入、输出、workspace 和 tiling 参数。
     */
    __aicore__ inline void operator()(Params const &params)
    {
        if (!Init(params)) {
            return;
        }
        Process(params);
    }

private:
    using BlockMmadGBSAG1 = BlockMmadGBSAG1_;
    using BlockMmadGBSAG2 = BlockMmadGBSAG2_;
    using BlockMmadGBSAG3 = BlockMmadGBSAG3_;
    using EpilogueFAGPre = EpilogueFAGPre_;
    using EpilogueFAGSfmg = EpilogueFAGSfmg_;
    using EpilogueFAGOp = EpilogueFAGOp_;
    using EpilogueFAGPost = EpilogueFAGPost_;
    using PreParams = typename EpilogueFAGPre::Params;
    using PostParams = typename EpilogueFAGPost::Params;
    using SfmParams = typename EpilogueFAGOp_::Params;
    using ArchTag = typename BlockMmadGBSAG1_::ArchTag;
    using ElementInput = typename BlockMmadGBSAG1::ElementA;

    using L1TileShape = typename BlockMmadGBSAG1::L1TileShape;
    using ElementA1 = typename BlockMmadGBSAG1::ElementA;
    using LayoutA1 = typename BlockMmadGBSAG1::LayoutA;
    using ElementB1 = typename BlockMmadGBSAG1::ElementB;
    using LayoutB1 = typename BlockMmadGBSAG1::LayoutB;
    using ElementC1 = typename BlockMmadGBSAG1::ElementC;
    using LayoutC1 = typename BlockMmadGBSAG1::LayoutC;

    using ElementA2 = typename BlockMmadGBSAG2::ElementA;
    using LayoutA2 = typename BlockMmadGBSAG2::LayoutA;
    using ElementB2 = typename BlockMmadGBSAG2::ElementB;
    using LayoutB2 = typename BlockMmadGBSAG2::LayoutB;
    using ElementC2 = typename BlockMmadGBSAG2::ElementC;
    using LayoutC2 = typename BlockMmadGBSAG2::LayoutC;

    using ElementA3 = typename BlockMmadGBSAG3::ElementA;
    using LayoutA3 = typename BlockMmadGBSAG3::LayoutA;
    using ElementB3 = typename BlockMmadGBSAG3::ElementB;
    using LayoutB3 = typename BlockMmadGBSAG3::LayoutB;
    using ElementC3 = typename BlockMmadGBSAG3::ElementC;
    using LayoutC3 = typename BlockMmadGBSAG3::LayoutC;

    using QPacket = GBSAG::QPacket;

    /**
     * @brief 单个 K task 及其当前 Q Packet 的设备侧执行状态。
     */
    struct TaskInfo {
        uint32_t curBatchIdx;       // 当前 batch 索引。
        uint32_t curHeadIdx;        // 当前 Q head 索引。
        uint32_t curQSeqIdx;        // 当前 Packet 首个原始 Q 行号。
        uint32_t curCalQSize;       // 当前 Packet 的实际有效 Q 行数。
        uint32_t curCalKVSize;      // 当前 K task 的实际有效 KV 行数。
        uint32_t qSeqlen;           // 当前 batch 的有效 Q 序列长度。
        uint32_t kvSeqlen;          // 当前 batch 的有效 KV 序列长度。
        uint64_t qOffset;           // Q、dOut、dQ 的元素偏移。
        uint64_t kvOffset;          // K、V、dK、dV 的元素偏移。
        uint64_t sOffset;           // S、P、dP、dS workspace 的逻辑偏移。
        uint32_t curKvHeadIdx;      // 当前 KV head 索引。
        uint32_t curKBlockIdx;      // 当前稀疏 K block 索引。
        uint32_t curKSeqIdx;        // 当前 K task 在 batch 内的起始行。
        uint64_t qBatchBaseOffset;  // 当前 batch 的 Q 数据元素基址。
        uint64_t kvBatchBaseOffset; // 当前 batch 的 KV 数据逻辑基址。
        QPacket qPacket;            // 当前任务持有的离散 Q 行到连续 Packet 行映射。
    };

    /**
     * @brief Workspace 各分区的字节基址，统一由 InitKernelFields 计算。
     */
    struct WorkspaceLayout {
        uint64_t pBase;      // 低精度 P workspace 字节基址。
        uint64_t dpBase;     // FP32 dP workspace 字节基址。
        uint64_t dsBase;     // 低精度 dS workspace 字节基址。
        uint64_t dqBase;     // FP32 dQ workspace 字节基址。
        uint64_t dkBase;     // FP32 dK workspace 字节基址。
        uint64_t dvBase;     // FP32 dV workspace 字节基址。
        uint64_t gradBase;   // D 预计算 workspace 字节基址。
        uint64_t packetBase; // 全部 Packet workspace 的字节基址。
    };
    // Packet 行数上限。
    static constexpr uint32_t AGGREGATE_M = Q_PACKET_AGGREGATE_M;

    /**
     * @brief 初始化 Kernel 字段以及输入、workspace GlobalTensor。
     * @param params Kernel 输入、输出、workspace 和 tiling 参数。
     * @return 参数合法时返回 true，否则返回 false。
     */
    __aicore__ inline bool Init(Params const &params)
    {
        if (!InitKernelFields(params)) {
            return false;
        }
        InitInputGlobalTensors(params);
        InitWorkspaceGlobalTensors(params);
        return true;
    }

    /**
     * @brief 解析 tiling、核索引和派生尺寸，初始化后续流程使用的类字段。
     * @param params Kernel 参数，提供 tiling 地址。
     * @return 基础参数合法时返回 true，否则返回 false。
     */
    __aicore__ inline bool InitKernelFields(Params const &params)
    {
        tilingData = reinterpret_cast<__gm__ GenericBlockSparseAttentionGradTilingData *>(params.tiling);
        gMetadata.SetGlobalBuffer((__gm__ int32_t *)params.metadata);
        if (params.metadata == nullptr) {
            return false;
        }
        blockIdx = AscendC::GetBlockIdx(); // 当前物理 AIC/AIV 核索引。
        const uint32_t subBlockNum = AscendC::GetSubBlockNum();
        if (subBlockNum == 0) {
            return false;
        }
        // AIC 的 subBlockNum 为 1，两个配对 AIV 的 subBlockNum 为 2，统一映射到逻辑 AIC 编号。
        coreIdx = blockIdx / subBlockNum;

        numHeads = tilingData->numHeads;
        kvHeads = tilingData->kvHeads;
        groupSize = numHeads / kvHeads;
        headDim = tilingData->headDim;
        maxQSeqlen = tilingData->maxQSeqlen;
        maxKvSeqlen = tilingData->maxKvSeqlen;
        inputLayout = tilingData->inputLayout;
        useUniformQSeqlen = tilingData->useUniformQSeqlen != 0;
        useUniformKvSeqlen = tilingData->useUniformKvSeqlen != 0;
        blockShapeX = tilingData->blockShapeX;
        blockShapeY = tilingData->blockShapeY;
        basicKVBlockSize = tilingData->basicKVBlockSize;
        if (groupSize == 0 || blockShapeX == 0 || blockShapeY == 0 || basicKVBlockSize == 0) {
            return false;
        }

        sOutSize = tilingData->sOutSize;
        dPOutSize = tilingData->dPOutSize;
        dQOutSize = tilingData->dQOutSize;
        dKOutSize = tilingData->dKOutSize;
        dVOutSize = tilingData->dVOutSize;
        gradSize = tilingData->gradSize;

        // 以下偏移单位均为字节，保持 Host 下发 workspace 分区顺序不变。
        workspaceLayout.dqBase = sOutSize + dPOutSize;
        workspaceLayout.dkBase = workspaceLayout.dqBase + dQOutSize;
        workspaceLayout.dvBase = workspaceLayout.dkBase + dKOutSize;
        workspaceLayout.gradBase = workspaceLayout.dvBase + dVOutSize;
        workspaceLayout.packetBase = workspaceLayout.gradBase + gradSize;
        workspaceLayout.pBase = WORKSPACE_P16_OFFSET;
        workspaceLayout.dpBase = sOutSize;
        workspaceLayout.dsBase = sOutSize + WORKSPACE_P16_OFFSET;

        metadataCoreNum = static_cast<uint32_t>(gMetadata.GetValue(GBSAGMeta::OFF_CORE_NUM));
        metadataTaskNumPerCore = static_cast<uint32_t>(gMetadata.GetValue(GBSAGMeta::OFF_TASK_NUM_PER_CORE));
        metadataTailTaskNum = static_cast<uint32_t>(gMetadata.GetValue(GBSAGMeta::OFF_TAIL_TASK_NUM));
        const uint32_t metadataTaskWords = static_cast<uint32_t>(gMetadata.GetValue(GBSAGMeta::OFF_TASK_WORDS));
        const uint32_t metadataTaskTableOffset =
            static_cast<uint32_t>(gMetadata.GetValue(GBSAGMeta::OFF_TASK_TABLE_OFFSET));

        if (coreIdx >= metadataCoreNum) {
            inactiveCore = true;
            taskLength = 0;
        } else {
            inactiveCore = false;
            const uint32_t taskBase = metadataTaskTableOffset + coreIdx * metadataTaskWords;
            taskLength = metadataTaskNumPerCore + (coreIdx < metadataTailTaskNum ? 1 : 0);
        }

        inputSlotElements = static_cast<uint64_t>(AGGREGATE_M) * headDim;
        inputSlotBytes = inputSlotElements * sizeof(ElementInput);
        dqSlotElements = static_cast<uint64_t>(AGGREGATE_M) * headDim;
        const uint32_t packetCoreNum = tilingData->usedVecCoreNum / 2;
        const uint64_t packetBytesPerCore = packetCoreNum == 0 ? 0 : tilingData->packetWorkspaceSize / packetCoreNum;
        corePacketBase = workspaceLayout.packetBase + static_cast<uint64_t>(coreIdx) * packetBytesPerCore;

        maskQBlockNum = (maxQSeqlen + blockShapeX - 1) / blockShapeX;
        maskKvBlockNum = (maxKvSeqlen + blockShapeY - 1) / blockShapeY;
        gSOffset = static_cast<uint64_t>(coreIdx) * WORKSPACE_BLOCK_SIZE_DB;
        return true;
    }

    /**
     * @brief 绑定输入数据、实际序列长度和稀疏索引 GlobalTensor。
     * @param params Kernel 输入参数。
     */
    __aicore__ inline void InitInputGlobalTensors(Params const &params)
    {
        gMetadata.SetGlobalBuffer((__gm__ int32_t *)params.metadata);
        gCuSeqLengthsQ.SetGlobalBuffer((__gm__ int64_t *)params.cuSeqLengthsQ);
        gCuSeqLengthsKv.SetGlobalBuffer((__gm__ int64_t *)params.cuSeqLengthsKv);
        gDout.SetGlobalBuffer((__gm__ ElementInput *)params.dout);
        gOut.SetGlobalBuffer((__gm__ ElementInput *)params.out);
        gQ.SetGlobalBuffer((__gm__ ElementInput *)params.q);
        gRsvdBlockIdx.SetGlobalBuffer((__gm__ int32_t *)params.rsvdBlockIdx);
        gRsvdBlockCount.SetGlobalBuffer((__gm__ int32_t *)params.rsvdBlockCount);
        gActualQseqlen.SetGlobalBuffer((__gm__ int32_t *)params.actualQseqlen);
        gActualKvseqlen.SetGlobalBuffer((__gm__ int32_t *)params.actualKvseqlen);
#ifdef __DAV_C220_CUBE__
        gK.SetGlobalBuffer((__gm__ ElementInput *)params.k);
        gV.SetGlobalBuffer((__gm__ ElementInput *)params.v);
#endif
    }

    /**
     * @brief 按既有地址公式绑定公共、Packet 和 Cube workspace GlobalTensor。
     * @param params Kernel workspace 参数。
     */
    __aicore__ inline void InitWorkspaceGlobalTensors(Params const &params)
    {
        gDq.SetGlobalBuffer((__gm__ float *)(params.workspace + workspaceLayout.dqBase));
        gQPack.SetGlobalBuffer((__gm__ ElementInput *)(params.workspace + corePacketBase));
        gDoutPack.SetGlobalBuffer(
            (__gm__ ElementInput *)(params.workspace + corePacketBase + PACKET_DOUT_BASE_SLOT * inputSlotBytes));
        gOutPack.SetGlobalBuffer(
            (__gm__ ElementInput *)(params.workspace + corePacketBase + PACKET_OUT_BASE_SLOT * inputSlotBytes));
        gDqPack.SetGlobalBuffer(
            (__gm__ float *)(params.workspace + corePacketBase + PACKET_DQ_BASE_SLOT * inputSlotBytes));

#ifdef __DAV_C220_CUBE__
        gS.SetGlobalBuffer((__gm__ float *)params.workspace);
        gP.SetGlobalBuffer((__gm__ ElementInput *)(params.workspace + workspaceLayout.pBase));
        gDp.SetGlobalBuffer((__gm__ float *)(params.workspace + workspaceLayout.dpBase));
        gDs.SetGlobalBuffer((__gm__ ElementInput *)(params.workspace + workspaceLayout.dsBase));
        gDk.SetGlobalBuffer((__gm__ float *)(params.workspace + workspaceLayout.dkBase));
        gDv.SetGlobalBuffer((__gm__ float *)(params.workspace + workspaceLayout.dvBase));
#endif
    }

    /**
     * @brief 执行 K_OUT 前处理、K/Q/Packet 主循环和结果后处理。
     * @param params Kernel 输入、输出、workspace 和 tiling 参数。
     */
    __aicore__ inline void Process(Params const &params)
    {
        // 首个 K task 属于计算遍历状态，在计算流程开始时初始化。
        if (!inactiveCore) {
            InitKTaskInfo(gActualQseqlen, gActualKvseqlen, tilingData, numHeads, kvHeads, headDim, maxQSeqlen,
                          maxKvSeqlen, blockShapeY, basicKVBlockSize, inputLayout, coreIdx, taskInfo[0]);
        }

        // 前处理仅在 AIV 执行，随后同步 AIC/AIV 再进入 Packet 主循环。
#ifdef __DAV_C220_VEC__
        VecPre(params);
#endif
        AscendC::SyncAll<false>();

#ifdef __DAV_C220_CUBE__
        // Cube MatMul 服务和硬件事件在三层循环外初始化，每个 Kernel 实例仅构造一次。
        BlockMmadGBSAG1 blockMmad1(resource);
        BlockMmadGBSAG2 blockMmad2(resource, CUBE2_L1_OFFSET_BYTES, CUBE2_PINGPONG_OFFSET, false);
        BlockMmadGBSAG3 blockMmad3(resource, CUBE3_L1_OFFSET_BYTES, CUBE3_PINGPONG_OFFSET, true);
        uint32_t mmadFlag = 0;
        SetFlag();
#endif

        // Packet metadata 聚合器不持有 UB，在 AIC/AIV 公共控制流中生成一致的 Packet 序列。
        GBSAGAggregator aggregateOp;

#ifdef __DAV_C220_VEC__
        // Vector 核专属 Softmax、Q/dOut Gather 和 dQ Scatter 对象。
        EpilogueFAGSfmg sfmgOp(resource);
        EpilogueFAGOp sStmOp(resource);
        // Gather 使用输入类型，同一个 UB 临时区依次服务 Q 和 dOut。
        GBSAGGater<ElementInput> gatherOp(resource);
        // dQ Pack 为 FP32，Scatter 通过 AtomicAdd 累加到原始 dQ workspace。
        GBSAGScateer<float> scatterOp(resource);
#endif

        uint32_t pingpongFlag = 0;
        uint32_t count = 0;
        TaskInfo preTaskInfo;
        TaskInfo prePreTaskInfo;     // packet[count-2] 状态快照，供延后两拍的 dQ scatter 使用。
        uint32_t preSlot = 0;        // packet[count-1] 的三槽软流水槽序号。
        uint32_t prePreSlot = 0;     // packet[count-2] 的三槽软流水槽序号。
        uint32_t isComputed = false; // 该核分配的数据是否会参与计算的flag
        // AIC/AIV 共用同一套 K -> Q -> packet 主循环，仅核心计算通过宏隔离。
        // taskLength: 一个core 要处理的kv block 块数
        for (uint32_t i = 0; i < taskLength; i++) {
            // curInfo 引用当前 K task 槽，避免复制其中约 1.5 KB 的 Packet metadata。
            TaskInfo curInfo = taskInfo[i % 2];

            // 组聚合（优化2）：同一 GQA group 的 groupSize 个 head 共享稀疏模式，
            // 其 (token, head) 行合入同一组聚合 Packet；qHead 内层循环取消，
            // curHeadIdx 固定为 qHeadBegin，Gather/LSE/Scatter 在 tile 内自行携带 head 偏移。
            const uint32_t qHeadBegin = curInfo.curKvHeadIdx * groupSize;
            const uint32_t qBlockNum = (curInfo.qSeqlen + blockShapeX - 1) / blockShapeX;
            {
                // kv2qIdx/rsvdBlockCount 按 KV head 编排；同一 GQA group 的 Q head
                // 共享该行稀疏模式，因此这里不能使用 Q head 作为第一维索引。
                // `(batch,qHead,kBlock)` 稀疏kv块在 count Tensor 中的索引地址
                const uint64_t kvBlockCountIdx =
                    (static_cast<uint64_t>(curInfo.curBatchIdx) * kvHeads + curInfo.curKvHeadIdx) * maskKvBlockNum +
                    curInfo.curKBlockIdx;
                int32_t sparseBlockCount = gRsvdBlockCount.GetValue(kvBlockCountIdx);
                if (sparseBlockCount < 0) {
                    sparseBlockCount = 0;
                } else if (static_cast<uint32_t>(sparseBlockCount) > qBlockNum) {
                    sparseBlockCount = static_cast<int32_t>(qBlockNum);
                }

                // kv2qIdx 的最后一维固定为 maxQSeqlen，不能再按 Q block 数取 stride。
                // 当前稀疏kv块在 idx Tensor 中的起始地址
                const uint64_t kvBlockIdxAddr = kvBlockCountIdx * maxQSeqlen;

                // 当前参数描述一个 (batch, kvHead, kBlock) 对应的稀疏 Q block 索引行。
                GBSAGAggregator::Params aggregateParams(gRsvdBlockIdx, kvBlockIdxAddr, sparseBlockCount, qBlockNum,
                                                        blockShapeX, curInfo.qSeqlen, groupSize);
                aggregateOp.Init(aggregateParams);

                // 一次聚合128行的Q行, 即将kv对应的Q再进行128切分循环
                // packetCount：ceilDiv(packet_q, 128)
                const uint32_t packetCount = aggregateOp.GetPacketCount();
                if (packetCount > 0) {
                    isComputed = true;
                }
                for (uint32_t packetIdx = 0; packetIdx < packetCount; ++packetIdx) {
                    // 精简：AIV 填充 qStart（闭式标量 + 顺序走查），
                    // AIC 仅闭式标量（GEMM 只需要 rows），消除 AIC 侧冗余走查。
#ifdef __DAV_C220_VEC__
                    aggregateOp.Aggregate(curInfo.qPacket);
#else
                    aggregateOp.AggregateScalars(curInfo.qPacket);
#endif
                    // 后续阶段只读消费 TaskInfo 中的当前 Packet 映射。
                    const QPacket &packet = curInfo.qPacket;
                    curInfo.curHeadIdx = qHeadBegin;
                    // 当前 Packet 首个 tile 对应的原始 Q 行（仅 VEC 侧；qStart 只在 AIV 填充）。
#ifdef __DAV_C220_VEC__
                    curInfo.curQSeqIdx = packet.qStart[0];
#endif
                    // 尾 Packet 必须使用实际有效行数，不能替换为固定 128 行。
                    curInfo.curCalQSize = packet.rows;
                    // 当前 Packet 选择的 S/dP ping-pong workspace 槽。
                    curInfo.sOffset = gSOffset + WORKSPACE_BLOCK_SIZE * pingpongFlag;
                    // 所有离散 Q 行先聚合到 Packet Workspace，再进入统一计算流程。
#ifdef __DAV_C220_VEC__
                    GatherPacket(curInfo, curInfo.curHeadIdx, pingpongFlag, curInfo.qPacket, gatherOp);
                    AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(VEC_CUBE_FLAG[pingpongFlag]);
                    // 阶段 C：scatter packet[count-2]。其 dQ 已由两轮前的 cube 写完并 set flag，
                    // 延后两拍执行可与 cube 的后续 GEMM 重叠；槽数 3 恰等于最远周转拍数，天然无 WAR。
                    if (count > 1) {
                        AscendC::WaitEvent(CUBE_VEC_FLAG[prePreSlot]);
                        ScatterDqPacket(prePreTaskInfo, prePreTaskInfo.curHeadIdx, prePreSlot, prePreTaskInfo.qPacket,
                                        scatterOp);
                    }
#endif

#ifdef __DAV_C220_CUBE__
                    AscendC::WaitEvent(VEC_CUBE_FLAG[pingpongFlag]);
                    ComputeScoreAndDp(curInfo, pingpongFlag, blockMmad1, mmadFlag);
                    AscendC::CrossCoreSetFlag<2, PIPE_FIX>(CUBE_VEC_FLAG[pingpongFlag]);
#endif

                    if (count > 0) {
                        uint32_t prePingpongFlag = preSlot;
#ifdef __DAV_C220_VEC__
                        AscendC::WaitEvent(CUBE_VEC_FLAG[prePingpongFlag]);
                        GatherOutPacket(preTaskInfo, preTaskInfo.curHeadIdx, prePingpongFlag, preTaskInfo.qPacket,
                                        gatherOp);
                        VecPrepareD(params, preTaskInfo, prePingpongFlag, preTaskInfo.qPacket, sfmgOp);
                        ComputeSAndDs(params, preTaskInfo, preTaskInfo.curHeadIdx, prePingpongFlag, preTaskInfo.qPacket,
                                      sStmOp);
                        AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(VEC_CUBE_FLAG[prePingpongFlag]);
#endif

#ifdef __DAV_C220_CUBE__
                        AscendC::WaitEvent(VEC_CUBE_FLAG[prePingpongFlag]);
                        ComputeDqDkDv(preTaskInfo, prePingpongFlag, blockMmad2, blockMmad3, mmadFlag);
                        AscendC::CrossCoreSetFlag<2, PIPE_FIX>(CUBE_VEC_FLAG[prePingpongFlag]);
#endif
                    }
                    // 槽推进：prePre ← pre ← cur，dQ scatter 相应延后两拍执行。
                    prePreTaskInfo = preTaskInfo;
                    prePreSlot = preSlot;
                    preTaskInfo = curInfo;
                    preSlot = pingpongFlag;
                    count++;
                    pingpongFlag = (pingpongFlag + 1) % WORKSPACE_SOFTPIPE_SLOT_NUM;
                }
            }
            if (i != taskLength - 1) {
                UpdateNextKTaskInfo(gActualQseqlen, gActualKvseqlen, numHeads, kvHeads, headDim, maxQSeqlen,
                                    maxKvSeqlen, blockShapeY, basicKVBlockSize, inputLayout, curInfo,
                                    taskInfo[(i + 1) % 2]);
            }
        }

        // 三槽流水排空：先补 scatter packet[count-2]，再完整处理最后一个 packet。
        if (isComputed && count > 1) {
#ifdef __DAV_C220_VEC__
            AscendC::WaitEvent(CUBE_VEC_FLAG[prePreSlot]);
            ScatterDqPacket(prePreTaskInfo, prePreTaskInfo.curHeadIdx, prePreSlot, prePreTaskInfo.qPacket, scatterOp);
#endif
        }

        if (isComputed) {
            uint32_t prePingpongFlag = preSlot;
#ifdef __DAV_C220_VEC__
            AscendC::WaitEvent(CUBE_VEC_FLAG[prePingpongFlag]);
            GatherOutPacket(preTaskInfo, preTaskInfo.curHeadIdx, prePingpongFlag, preTaskInfo.qPacket, gatherOp);
            VecPrepareD(params, preTaskInfo, prePingpongFlag, preTaskInfo.qPacket, sfmgOp);
            ComputeSAndDs(params, preTaskInfo, preTaskInfo.curHeadIdx, prePingpongFlag, preTaskInfo.qPacket, sStmOp);
            AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(VEC_CUBE_FLAG[prePingpongFlag]);
#endif

#ifdef __DAV_C220_CUBE__
            AscendC::WaitEvent(VEC_CUBE_FLAG[prePingpongFlag]);
            ComputeDqDkDv(preTaskInfo, prePingpongFlag, blockMmad2, blockMmad3, mmadFlag);
            AscendC::CrossCoreSetFlag<2, PIPE_FIX>(CUBE_VEC_FLAG[prePingpongFlag]);
#endif

#ifdef __DAV_C220_VEC__
            AscendC::WaitEvent(CUBE_VEC_FLAG[prePingpongFlag]);
            ScatterDqPacket(preTaskInfo, preTaskInfo.curHeadIdx, prePingpongFlag, preTaskInfo.qPacket, scatterOp);
#endif
        }

#ifdef __DAV_C220_CUBE__
        AscendC::CrossCoreSetFlag<2, PIPE_FIX>(CUBE_POST_FLAG);
        WaitFlag();
#endif
#ifdef __DAV_C220_VEC__
        AscendC::WaitEvent(CUBE_POST_FLAG);
        AscendC::SyncAll<true>();
        VecPost(params);
#endif
    }

    /**
     * @brief 复制 K_OUT 推进需要的标量状态，不复制当前 Packet metadata。
     * @param src 当前 K task 状态。
     * @param dst 下一 K task 状态。
     *
     * KV offset、K block 和 KV 计算尺寸会在推进完成后重新计算，
     * 因此这里只复制确定下一任务位置所需的基础状态。
     */
    __aicore__ inline void CopyKTaskState(const TaskInfo &src, TaskInfo &dst)
    {
        dst.curBatchIdx = src.curBatchIdx;
        dst.qSeqlen = src.qSeqlen;
        dst.kvSeqlen = src.kvSeqlen;
        dst.curKvHeadIdx = src.curKvHeadIdx;
        dst.curKSeqIdx = src.curKSeqIdx;
        dst.qBatchBaseOffset = src.qBatchBaseOffset;
        dst.kvBatchBaseOffset = src.kvBatchBaseOffset;
    }

    __aicore__ inline void SetFlag()
    {
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
    }

    __aicore__ inline void WaitFlag()
    {
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
    }

    /**
     * @brief 根据当前 K task 的起始位置、稀疏 KV block 边界和序列尾部计算实际 KV 行数。
     * @param blockShapeY 稀疏 KV block 的行数。
     * @param basicKVBlockSize 单个 K task 的最大计算行数。
     * @param taskInfo K task 状态；函数更新 curKBlockIdx 和 curCalKVSize。
     */
    __aicore__ inline void UpdateKTaskCalSize(uint32_t blockShapeY, uint32_t basicKVBlockSize, TaskInfo &taskInfo)
    {
        taskInfo.curKBlockIdx = taskInfo.curKSeqIdx / blockShapeY;
        const uint32_t sparseEnd = (taskInfo.curKBlockIdx + 1) * blockShapeY;
        const uint32_t taskEnd = taskInfo.curKSeqIdx + basicKVBlockSize;
        const uint32_t validEnd = sparseEnd < taskInfo.kvSeqlen ? sparseEnd : taskInfo.kvSeqlen;
        taskInfo.curCalKVSize = taskEnd < validEnd ? basicKVBlockSize : validEnd - taskInfo.curKSeqIdx;
    }

    /**
     * @brief 初始化当前逻辑核负责的第一个 K task 及其 batch、head 和地址状态。
     * @param gActualQseqlen 各 batch 的实际 Q 序列长度。
     * @param gActualKvseqlen 各 batch 的实际 KV 序列长度。
     * @param tilingData Host 下发的 K_OUT tiling 数据。
     * @param numHeads Q head 数量。
     * @param kvHeads KV head 数量。
     * @param headDim 单个 head 的特征维度。
     * @param maxQSeqlen BNSD 场景使用的最大 Q 序列长度。
     * @param maxKvSeqlen BNSD 场景使用的最大 KV 序列长度。
     * @param blockShapeY 稀疏 KV block 的行数。
     * @param basicKVBlockSize 单个 K task 的最大计算行数。
     * @param inputLayout 输入布局，0 表示 TND，其他值表示 BNSD。
     * @param coreIdx 当前逻辑核索引。
     * @param taskInfo 输出的首个 K task 状态。
     */
    __aicore__ inline void InitKTaskInfo(AscendC::GlobalTensor<int32_t> gActualQseqlen,
                                         AscendC::GlobalTensor<int32_t> gActualKvseqlen,
                                         __gm__ GenericBlockSparseAttentionGradTilingData *tilingData,
                                         uint32_t numHeads, uint32_t kvHeads, uint32_t headDim, uint32_t maxQSeqlen,
                                         uint32_t maxKvSeqlen, uint32_t blockShapeY, uint32_t basicKVBlockSize,
                                         uint32_t inputLayout, uint32_t coreIdx, TaskInfo &taskInfo)
    {
        const uint32_t taskBase = static_cast<uint32_t>(gMetadata.GetValue(GBSAGMeta::OFF_TASK_TABLE_OFFSET)) +
                                  coreIdx * static_cast<uint32_t>(gMetadata.GetValue(GBSAGMeta::OFF_TASK_WORDS));
        taskInfo.curBatchIdx = static_cast<uint32_t>(gMetadata.GetValue(taskBase + GBSAGMeta::TASK_BEGIN_BATCH));
        taskInfo.curKvHeadIdx = static_cast<uint32_t>(gMetadata.GetValue(taskBase + GBSAGMeta::TASK_BEGIN_KV_HEAD));
        taskInfo.curKSeqIdx = static_cast<uint32_t>(gMetadata.GetValue(taskBase + GBSAGMeta::TASK_BEGIN_KV_SEQ_OFFSET));
        if (inputLayout == 0) {
            // TND 契约：cuSeqLengthsQ/Kv 必传（host 已校验），长度取前缀和差分；
            // 种子偏移 = cuSeq[b]（与差分同源，零额外 GM 读），替代原 KTask.preQ/preKv。
            const int64_t cuQCur = gCuSeqLengthsQ.GetValue(taskInfo.curBatchIdx);
            const int64_t cuKvCur = gCuSeqLengthsKv.GetValue(taskInfo.curBatchIdx);
            taskInfo.qSeqlen = static_cast<uint32_t>(
                gCuSeqLengthsQ.GetValue(taskInfo.curBatchIdx + NEXT_SEQUENCE_INDEX_OFFSET) - cuQCur);
            taskInfo.kvSeqlen = static_cast<uint32_t>(
                gCuSeqLengthsKv.GetValue(taskInfo.curBatchIdx + NEXT_SEQUENCE_INDEX_OFFSET) - cuKvCur);
            taskInfo.qBatchBaseOffset = static_cast<uint64_t>(cuQCur);
            taskInfo.kvBatchBaseOffset = static_cast<uint64_t>(cuKvCur);
            taskInfo.qBatchBaseOffset *= numHeads * headDim;
            taskInfo.kvOffset = (taskInfo.kvBatchBaseOffset * kvHeads + taskInfo.curKSeqIdx * kvHeads) * headDim +
                                taskInfo.curKvHeadIdx * headDim;
        } else {
            // packed（BNSD/BSND）：种子偏移 = b × N × maxSeqlen（物理 stride），替代原 KTask.preQ/preKv。
            taskInfo.qBatchBaseOffset = static_cast<uint64_t>(taskInfo.curBatchIdx) * static_cast<uint64_t>(numHeads) *
                                        static_cast<uint64_t>(maxQSeqlen);
            taskInfo.kvBatchBaseOffset = static_cast<uint64_t>(taskInfo.curBatchIdx) * static_cast<uint64_t>(kvHeads) *
                                         static_cast<uint64_t>(maxKvSeqlen);
            taskInfo.qSeqlen =
                useUniformQSeqlen ? maxQSeqlen : static_cast<uint32_t>(gActualQseqlen.GetValue(taskInfo.curBatchIdx));
            taskInfo.kvSeqlen = useUniformKvSeqlen ?
                                    maxKvSeqlen :
                                    static_cast<uint32_t>(gActualKvseqlen.GetValue(taskInfo.curBatchIdx));
            taskInfo.qBatchBaseOffset *= headDim;
            if (inputLayout == 2) {
                taskInfo.kvOffset = (taskInfo.kvBatchBaseOffset + static_cast<uint64_t>(taskInfo.curKSeqIdx) * kvHeads +
                                     taskInfo.curKvHeadIdx) *
                                    headDim;
            } else {
                taskInfo.kvOffset = (taskInfo.kvBatchBaseOffset +
                                     static_cast<uint64_t>(taskInfo.curKvHeadIdx) * maxKvSeqlen + taskInfo.curKSeqIdx) *
                                    headDim;
            }
        }
        UpdateKTaskCalSize(blockShapeY, basicKVBlockSize, taskInfo);
    }

    /**
     * @brief 推进到下一个 K task，处理 KV head、KV 序列和 batch 边界。
     * @param gActualQseqlen 各 batch 的实际 Q 序列长度。
     * @param gActualKvseqlen 各 batch 的实际 KV 序列长度。
     * @param numHeads Q head 数量。
     * @param kvHeads KV head 数量。
     * @param headDim 单个 head 的特征维度。
     * @param maxQSeqlen BNSD 场景使用的最大 Q 序列长度。
     * @param maxKvSeqlen BNSD 场景使用的最大 KV 序列长度。
     * @param blockShapeY 稀疏 KV block 的行数。
     * @param basicKVBlockSize 单个 K task 的最大计算行数。
     * @param inputLayout 输入布局，0 表示 TND，其他值表示 BNSD。
     * @param cur 当前已完成的 K task 状态。
     * @param next 输出的下一个 K task 状态。
     */
    __aicore__ inline void UpdateNextKTaskInfo(AscendC::GlobalTensor<int32_t> gActualQseqlen,
                                               AscendC::GlobalTensor<int32_t> gActualKvseqlen, uint32_t numHeads,
                                               uint32_t kvHeads, uint32_t headDim, uint32_t maxQSeqlen,
                                               uint32_t maxKvSeqlen, uint32_t blockShapeY, uint32_t basicKVBlockSize,
                                               uint32_t inputLayout, const TaskInfo &cur, TaskInfo &next)
    {
        CopyKTaskState(cur, next);
        bool nextBatch = false;
        if (inputLayout == 0) {
            if (cur.curKvHeadIdx + 1 < kvHeads) {
                next.curKvHeadIdx++;
            } else {
                next.curKvHeadIdx = 0;
                next.curKSeqIdx += cur.curCalKVSize;
                nextBatch = next.curKSeqIdx >= cur.kvSeqlen;
            }
        } else {
            next.curKSeqIdx += cur.curCalKVSize;
            if (next.curKSeqIdx >= cur.kvSeqlen) {
                next.curKSeqIdx = 0;
                if (cur.curKvHeadIdx + 1 < kvHeads) {
                    next.curKvHeadIdx++;
                } else {
                    next.curKvHeadIdx = 0;
                    nextBatch = true;
                }
            }
        }
        if (nextBatch) {
            next.curBatchIdx++;
            next.curKSeqIdx = 0;
            if (inputLayout == 0) {
                next.qBatchBaseOffset += static_cast<uint64_t>(cur.qSeqlen) * numHeads * headDim;
                next.kvBatchBaseOffset += cur.kvSeqlen;
                next.qSeqlen =
                    static_cast<uint32_t>(gCuSeqLengthsQ.GetValue(next.curBatchIdx + NEXT_SEQUENCE_INDEX_OFFSET) -
                                          gCuSeqLengthsQ.GetValue(next.curBatchIdx));
                next.kvSeqlen =
                    static_cast<uint32_t>(gCuSeqLengthsKv.GetValue(next.curBatchIdx + NEXT_SEQUENCE_INDEX_OFFSET) -
                                          gCuSeqLengthsKv.GetValue(next.curBatchIdx));
            } else {
                // BNSD/BSND Q base is an element offset; keep the physical batch stride in D elements.
                next.qBatchBaseOffset = static_cast<uint64_t>(next.curBatchIdx) * numHeads * maxQSeqlen * headDim;
                next.kvBatchBaseOffset = static_cast<uint64_t>(next.curBatchIdx) * kvHeads * maxKvSeqlen;
                next.qSeqlen =
                    useUniformQSeqlen ? maxQSeqlen : static_cast<uint32_t>(gActualQseqlen.GetValue(next.curBatchIdx));
                next.kvSeqlen = useUniformKvSeqlen ? maxKvSeqlen :
                                                     static_cast<uint32_t>(gActualKvseqlen.GetValue(next.curBatchIdx));
            }
        }
        if (inputLayout == 0) {
            next.kvOffset =
                (next.kvBatchBaseOffset * kvHeads + next.curKSeqIdx * kvHeads) * headDim + next.curKvHeadIdx * headDim;
        } else if (inputLayout == 2) {
            next.kvOffset =
                (next.kvBatchBaseOffset + static_cast<uint64_t>(next.curKSeqIdx) * kvHeads + next.curKvHeadIdx) *
                headDim;
        } else {
            next.kvOffset =
                (next.kvBatchBaseOffset + static_cast<uint64_t>(next.curKvHeadIdx) * maxKvSeqlen + next.curKSeqIdx) *
                headDim;
        }
        UpdateKTaskCalSize(blockShapeY, basicKVBlockSize, next);
    }

#ifdef __DAV_C220_VEC__
    /**
     * @brief 将当前 Packet 的离散 Q 和 dOut 行聚合到连续 Packet workspace。
     * @param curInfo 当前 K task 的 batch、序列和 workspace 状态。
     * @param qHead 当前 Q head 索引。
     * @param pingpongFlag 当前 Packet 双缓冲槽编号。
     * @param packet 离散 Q 行到连续 Packet 行的映射。
     * @param gatherOp Q/dOut 聚合服务对象。
     */
    __aicore__ inline void GatherPacket(const TaskInfo &curInfo, uint32_t qHead, uint32_t pingpongFlag,
                                        const QPacket &packet, GBSAGGater<ElementInput> &gatherOp)
    {
        const uint32_t tileRows = GBSAG::GetQPacketTileRows(groupSize);
        typename GBSAGGater<ElementInput>::Params qGatherParams(
            gQ, gQPack[pingpongFlag * inputSlotElements], curInfo.qBatchBaseOffset, qHead, numHeads,
            inputLayout == 0 ? curInfo.qSeqlen : maxQSeqlen, headDim, inputLayout, tileRows);
        gatherOp.Gather(qGatherParams, packet);

        typename GBSAGGater<ElementInput>::Params doutGatherParams(
            gDout, gDoutPack[pingpongFlag * inputSlotElements], curInfo.qBatchBaseOffset, qHead, numHeads,
            inputLayout == 0 ? curInfo.qSeqlen : maxQSeqlen, headDim, inputLayout, tileRows);
        gatherOp.Gather(doutGatherParams, packet);
    }

    /**
     * @brief 将当前 Packet 的离散 out 行聚合到独立连续 Packet workspace。
     */
    __aicore__ inline void GatherOutPacket(const TaskInfo &curInfo, uint32_t qHead, uint32_t pingpongFlag,
                                           const QPacket &packet, GBSAGGater<ElementInput> &gatherOp)
    {
        const uint32_t tileRows = GBSAG::GetQPacketTileRows(groupSize);
        typename GBSAGGater<ElementInput>::Params outGatherParams(
            gOut, gOutPack[pingpongFlag * inputSlotElements], curInfo.qBatchBaseOffset, qHead, numHeads,
            inputLayout == 0 ? curInfo.qSeqlen : maxQSeqlen, headDim, inputLayout, tileRows);
        gatherOp.Gather(outGatherParams, packet);
    }
#endif

#ifdef __DAV_C220_CUBE__
    /**
     * @brief 等待 Gather 完成，计算当前 Packet 的 attention score 和 dP。
     * @param curInfo 当前 K task 和 Packet 计算尺寸。
     * @param pingpongFlag 当前 Packet 双缓冲槽编号。
     * @param blockMmad1 Cube1 MatMul 服务对象。
     * @param mmadFlag MatMul 服务使用的流水标志。
     */
    __aicore__ inline void ComputeScoreAndDp(const TaskInfo &curInfo, uint32_t pingpongFlag,
                                             BlockMmadGBSAG1 &blockMmad1, uint32_t &mmadFlag)
    {
        LayoutA1 layoutA1;
        LayoutB1 layoutB1;
        LayoutC1 layoutC1(curInfo.curCalQSize, curInfo.curCalKVSize);
        if (inputLayout != 1) {
            layoutA1 = LayoutA1(curInfo.curCalQSize, headDim);
            layoutB1 = LayoutB1(headDim, curInfo.curCalKVSize, kvHeads * headDim);
        } else {
            layoutA1 = LayoutA1(curInfo.curCalQSize, headDim);
            layoutB1 = LayoutB1(headDim, curInfo.curCalKVSize);
        }
        GemmCoord actualShape1{curInfo.curCalQSize, curInfo.curCalKVSize, headDim};
        {
            blockMmad1(gQPack[pingpongFlag * inputSlotElements], gK[curInfo.kvOffset], gS[curInfo.sOffset], layoutA1,
                       layoutB1, layoutC1, actualShape1, mmadFlag);
            blockMmad1(gDoutPack[pingpongFlag * inputSlotElements], gV[curInfo.kvOffset], gDp[curInfo.sOffset],
                       layoutA1, layoutB1, layoutC1, actualShape1, mmadFlag);
        }
    }
#endif

#ifdef __DAV_C220_VEC__
    /**
     * @brief 等待 Cube1 完成，并按当前 AIV 的 Packet 行范围执行 SAndDs。
     * @param params Kernel 参数，提供原始 GM 和 workspace 地址。
     * @param curInfo 当前 K task 和 Packet 计算尺寸。
     * @param qHead 当前 Q head 索引。
     * @param packet 当前 Packet 行映射。
     * @param softmaxOp SAndDs 服务对象。
     */
    __aicore__ inline void ComputeSAndDs(Params const &params, const TaskInfo &curInfo, uint32_t qHead,
                                         uint32_t pingpongFlag, const QPacket &packet, EpilogueFAGOp &softmaxOp)
    {
        // 两个 AIV 按 Packet 实际行数二分，尾 Packet 不使用固定 128 行。
        const QPacketAivRange aivRange = GetQPacketAivRange(packet.rows);
        const uint32_t vecBegin = aivRange.begin;
        const uint32_t vecEnd = aivRange.end;

        const uint64_t executeRow = vecEnd - vecBegin;
        const uint64_t coreOffset = static_cast<uint64_t>(vecBegin) * curInfo.curCalKVSize;
        const uint64_t vector32Soffset = (curInfo.sOffset + coreOffset) * sizeof(float);
        const uint64_t vector16Soffset =
            (curInfo.sOffset * FP32_TO_LOW_PRECISION_ELEMENT_RATIO + coreOffset) * sizeof(ElementInput);
        GM_ADDR s = params.workspace + vector32Soffset;
        GM_ADDR dp = params.workspace + workspaceLayout.dpBase + vector32Soffset;
        GM_ADDR pWorkspace = params.workspace + workspaceLayout.pBase + vector16Soffset;
        GM_ADDR dsWorkspace = params.workspace + workspaceLayout.dsBase + vector16Soffset;
        // LSE remains packet-indexed; D stays in the shared AIV UB region.
        SfmParams sfmParams(s, params.softmaxLse, dp, pWorkspace, dsWorkspace, params.tiling, executeRow,
                            curInfo.curCalKVSize, executeRow * curInfo.curCalKVSize, qHead, vecBegin,
                            curInfo.qBatchBaseOffset, curInfo.qSeqlen, headDim, packet, tilingData->maskType,
                            curInfo.curKSeqIdx, curInfo.kvSeqlen);
        // 组聚合 Packet 语义：LSE 搬运需要 GQA group 大小来切分 token tile。
        sfmParams.groupSize = groupSize;
        softmaxOp(sfmParams);
    }
#endif

#ifdef __DAV_C220_CUBE__
    /**
     * @brief 等待 SAndDs 完成，计算当前 Packet 的 dQ、dK 和 dV。
     * @param curInfo 当前 K task 和 Packet 计算尺寸。
     * @param pingpongFlag 当前 Packet 双缓冲槽编号。
     * @param blockMmad2 dQ MatMul 服务对象。
     * @param blockMmad3 dK/dV MatMul 服务对象。
     * @param mmadFlag MatMul 服务使用的流水标志。
     */
    __aicore__ inline void ComputeDqDkDv(const TaskInfo &curInfo, uint32_t pingpongFlag, BlockMmadGBSAG2 &blockMmad2,
                                         BlockMmadGBSAG3 &blockMmad3, uint32_t &mmadFlag)
    {
        LayoutA2 layoutA2(curInfo.curCalQSize, curInfo.curCalKVSize);
        LayoutB2 layoutB2;
        LayoutC2 layoutC2;
        LayoutA3 layoutA3(curInfo.curCalKVSize, curInfo.curCalQSize);
        LayoutB3 layoutB3;
        LayoutC3 layoutC3;
        if (inputLayout != 1) {
            layoutB2 = LayoutB2(curInfo.curCalKVSize, headDim, kvHeads * headDim);
            layoutC2 = LayoutC2(curInfo.curCalQSize, headDim);
            layoutB3 = LayoutB3(curInfo.curCalQSize, headDim);
            layoutC3 = LayoutC3(curInfo.curCalKVSize, headDim, kvHeads * headDim);
        } else {
            layoutB2 = LayoutB2(curInfo.curCalKVSize, headDim);
            layoutC2 = LayoutC2(curInfo.curCalQSize, headDim);
            layoutB3 = LayoutB3(curInfo.curCalQSize, headDim);
            layoutC3 = LayoutC3(curInfo.curCalKVSize, headDim);
        }
        GemmCoord actualShape2{curInfo.curCalQSize, headDim, curInfo.curCalKVSize};
        GemmCoord actualShape3{curInfo.curCalKVSize, headDim, curInfo.curCalQSize};
        const uint64_t lowPrecisionSOffset = curInfo.sOffset * FP32_TO_LOW_PRECISION_ELEMENT_RATIO;
        {
            blockMmad2(gDs[lowPrecisionSOffset], gK[curInfo.kvOffset], gDqPack[pingpongFlag * dqSlotElements], layoutA2,
                       layoutB2, layoutC2, actualShape2, mmadFlag);
            blockMmad3(gP[lowPrecisionSOffset], gDoutPack[pingpongFlag * inputSlotElements], gDv[curInfo.kvOffset],
                       layoutA3, layoutB3, layoutC3, actualShape3, mmadFlag);
            blockMmad3(gDs[lowPrecisionSOffset], gQPack[pingpongFlag * inputSlotElements], gDk[curInfo.kvOffset],
                       layoutA3, layoutB3, layoutC3, actualShape3, mmadFlag);
        }
    }
#endif

#ifdef __DAV_C220_VEC__
    /**
     * @brief 将连续 dQ Packet 通过 FP32 AtomicAdd 恢复到原始 Q 行。
     * @param curInfo 当前 K task 的 batch 和序列状态。
     * @param qHead 当前 Q head 索引。
     * @param pingpongFlag 当前 Packet 双缓冲槽编号。
     * @param packet 当前 Packet 行映射。
     * @param scatterOp dQ 恢复服务对象。
     */
    __aicore__ inline void ScatterDqPacket(const TaskInfo &curInfo, uint32_t qHead, uint32_t pingpongFlag,
                                           const QPacket &packet, GBSAGScateer<float> &scatterOp)
    {
        const uint32_t tileRows = GBSAG::GetQPacketTileRows(groupSize);
        typename GBSAGScateer<float>::Params scatterParams(
            gDq, gDqPack[pingpongFlag * dqSlotElements], curInfo.qBatchBaseOffset, qHead, numHeads,
            inputLayout == 0 ? curInfo.qSeqlen : maxQSeqlen, headDim, inputLayout, tileRows);
        scatterOp.Scatter(scatterParams, packet, true);
    }
#endif

    /**
     * @brief 按当前 AIV 的 Packet 行范围计算 D = sum(dOut * out)。
     */
    __aicore__ inline void VecPrepareD(Params const &params, const TaskInfo &curInfo, uint32_t pingpongFlag,
                                       const QPacket &packet, EpilogueFAGSfmg &sfmgOp)
    {
        const QPacketAivRange range = GetQPacketAivRange(packet.rows);
        if (range.rows > 0) {
            const uint64_t inputSlotOffset = static_cast<uint64_t>(pingpongFlag) * inputSlotElements;
            const uint64_t inputRowOffset = static_cast<uint64_t>(range.begin) * headDim;
            sfmgOp.ProcessPacket(gDoutPack[inputSlotOffset + inputRowOffset],
                                 gOutPack[inputSlotOffset + inputRowOffset], range.rows, headDim, params.tiling);
        }
    }

    __aicore__ inline void VecPost(Params const &params)
    {
        GM_ADDR gDqGm = params.workspace + workspaceLayout.dqBase;
        GM_ADDR gDkGm = params.workspace + workspaceLayout.dkBase;
        GM_ADDR gDvGm = params.workspace + workspaceLayout.dvBase;

        PostParams postParams(params.dq, params.dk, params.dv, gDqGm, gDkGm, gDvGm, params.tiling, params.cuSeqLengthsQ,
                              params.cuSeqLengthsKv);
        EpilogueFAGPost vecPost(postParams);
        vecPost();
    }

    __aicore__ inline void VecPre(Params const &params)
    {
        GM_ADDR gDqWrkGm = params.workspace + workspaceLayout.dqBase;
        GM_ADDR gDkWrkGm = params.workspace + workspaceLayout.dkBase;
        GM_ADDR gDvWrkGm = params.workspace + workspaceLayout.dvBase;

        PreParams preParms(gDqWrkGm, gDkWrkGm, gDvWrkGm, params.tiling);
        EpilogueFAGPre vecPre(preParms);
        vecPre();
    }

    // 设备资源与 tiling。
    NpuArch::Arch::Resource<ArchTag> resource;                    // 当前架构的设备资源。
    __gm__ GenericBlockSparseAttentionGradTilingData *tilingData; // Host 下发的 tiling 数据。

    // 核索引、shape 和布局参数。
    uint32_t blockIdx;    // 当前物理 AIC/AIV 核索引。
    uint32_t coreIdx;     // AIC 与配对 AIV 共用的逻辑核索引。
    uint32_t numHeads;    // Q head 数量。
    uint32_t kvHeads;     // KV head 数量。
    uint32_t groupSize;   // 每个 KV head 对应的 Q head 数量。
    uint32_t headDim;     // 单个 head 的特征维度。
    uint32_t maxQSeqlen;  // 最大 Q 序列长度。
    uint32_t maxKvSeqlen; // 最大 KV 序列长度。
    uint32_t inputLayout; // 0 表示 TND，1 表示 BNSD，2 表示 BSND。
    bool useUniformQSeqlen = true;
    bool useUniformKvSeqlen = true;
    uint32_t blockShapeX;      // 稀疏 Q block 的行数。
    uint32_t blockShapeY;      // 稀疏 KV block 的行数。
    uint32_t basicKVBlockSize; // 单个 K task 的最大 KV 行数。

    // 任务和稀疏索引参数。
    uint32_t taskLength; // 当前逻辑核需要处理的 K task 数量。
    uint32_t metadataCoreNum = 0;
    uint32_t metadataTaskNumPerCore = 0;
    uint32_t metadataTailTaskNum = 0;
    bool inactiveCore = false;
    uint32_t maskQBlockNum;  // 最大 Q 序列对应的稀疏 Q block 数量。
    uint32_t maskKvBlockNum; // 最大 KV 序列对应的稀疏 KV block 数量。
    TaskInfo taskInfo[2];    // K task 推进使用的双缓冲状态。

    // workspace 分区尺寸和当前核 Packet 地址。
    uint64_t sOutSize;               // FP32 S workspace 字节数。
    uint64_t dPOutSize;              // dP workspace 字节数。
    uint64_t dQOutSize;              // FP32 dQ workspace 字节数。
    uint64_t dKOutSize;              // FP32 dK workspace 字节数。
    uint64_t dVOutSize;              // FP32 dV workspace 字节数。
    uint64_t gradSize;               // D 预计算 workspace 字节数。
    WorkspaceLayout workspaceLayout; // Workspace 各分区的统一字节基址。
    uint64_t inputSlotElements;      // 单个 Q/dOut Packet 槽的元素数。
    uint64_t inputSlotBytes;         // 单个 Q/dOut Packet 槽的字节数。
    uint64_t dqSlotElements;         // 单个 FP32 dQ Packet 槽的元素数。
    uint64_t corePacketBase;         // 当前逻辑核 Packet workspace 的字节基址。
    uint64_t gSOffset;               // 当前逻辑核 S/dP 双缓冲区的元素偏移。

    // 公共输入 GlobalTensor。
    AscendC::GlobalTensor<ElementInput> gDout;      // dOut 输入。
    AscendC::GlobalTensor<ElementInput> gOut;       // out 输入。
    AscendC::GlobalTensor<ElementInput> gQ;         // Q 输入。
    AscendC::GlobalTensor<int32_t> gRsvdBlockIdx;   // KV block 到 Q block 的稀疏索引。
    AscendC::GlobalTensor<int32_t> gRsvdBlockCount; // 每个 KV block 的有效稀疏索引数。
    AscendC::GlobalTensor<int32_t> gMetadata;       // Metadata Header/KTask words。
    AscendC::GlobalTensor<int32_t> gActualQseqlen;  // 各 batch 的实际 Q 序列长度。
    AscendC::GlobalTensor<int32_t> gActualKvseqlen; // 各 batch 的实际 KV 序列长度。
    AscendC::GlobalTensor<int64_t> gCuSeqLengthsQ;  // 各 batch 的 Q 前缀和。
    AscendC::GlobalTensor<int64_t> gCuSeqLengthsKv; // 各 batch 的 KV 前缀和。

    // 公共和 Packet workspace GlobalTensor。
    AscendC::GlobalTensor<float> gDq;              // 原始布局的 FP32 dQ workspace。
    AscendC::GlobalTensor<ElementInput> gQPack;    // 连续 Q Packet 双缓冲区。
    AscendC::GlobalTensor<ElementInput> gDoutPack; // 连续 dOut Packet 双缓冲区。
    AscendC::GlobalTensor<ElementInput> gOutPack;  // 连续 out Packet 双缓冲区。
    AscendC::GlobalTensor<float> gDqPack;          // 连续 FP32 dQ Packet 双缓冲区。

#ifdef __DAV_C220_CUBE__
    // Cube 输入和 workspace GlobalTensor，仅在 AIC 侧使用。
    AscendC::GlobalTensor<ElementInput> gK;  // K 输入。
    AscendC::GlobalTensor<ElementInput> gV;  // V 输入。
    AscendC::GlobalTensor<float> gS;         // FP32 attention score workspace。
    AscendC::GlobalTensor<ElementInput> gP;  // 低精度 softmax P workspace。
    AscendC::GlobalTensor<float> gDp;        // FP32 dP workspace。
    AscendC::GlobalTensor<ElementInput> gDs; // 低精度 dS workspace。
    AscendC::GlobalTensor<float> gDk;        // FP32 dK workspace。
    AscendC::GlobalTensor<float> gDv;        // FP32 dV workspace。
#endif
};

} // namespace GBSAG

#endif // GENERIC_BLOCK_SPARSE_ATTENTION_GRAD_K_OUT_KERNEL_H
