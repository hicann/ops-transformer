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
 * \file gbsag_epilogue_simply_softmax.hpp
 * \brief Block Epliogue Simply Softmax Kernel Implementation
 */

#ifndef CATLASS_EPILOGUE_BLOCK_GBSAG_EPILOGUE_SIMPLY_SOFTMAX_HPP
#define CATLASS_EPILOGUE_BLOCK_GBSAG_EPILOGUE_SIMPLY_SOFTMAX_HPP

#include "../../../attn_infra/arch/gbsag_resource.hpp"
#include "../../../attn_infra/epilogue/gbsag_epilogue_dispatch_policy.hpp"
#include "gbsag_epilogue_packet_ub_layout.hpp"
#include "gbsag_gather_and_scatter.hpp"
#include "kernel_operator.h"

using namespace AscendC;

template <typename InDtype>
struct SimplySoftMaxInfo {
    LocalTensor<float> sTensor;
    LocalTensor<float> lseBrocTensor;
    LocalTensor<float> pFp32Tensor;
    LocalTensor<InDtype> pFp16Tensor;

    GlobalTensor<float> sGm;
    GlobalTensor<float> lseGm;
    GlobalTensor<InDtype> pGm;
};

template <typename InDtype>
struct CalDsInfo {
    LocalTensor<float> dpFp32Tensor;
    LocalTensor<float> softmaxGradTensor;
    LocalTensor<float> pFp32Tensor;
    LocalTensor<InDtype> dsFp16Tensor;

    GlobalTensor<float> dpGm;
    GlobalTensor<InDtype> dsGm;
};

namespace NpuArch::Epilogue::Block {
template <typename InputDType, typename OutputDtype, uint32_t INPUT_LAYOUT>
class SimpltSoftmax {
public:
    using DispatchPolicy = EpilogueAtlasA2FAGPre;
    using ArchTag = typename DispatchPolicy::ArchTag;

    struct Params {
        // Data members
        GM_ADDR s;           // 连续
        GM_ADDR softmaxLse;  // 需要跳着搬运
        GM_ADDR dp;          // 连续
        GM_ADDR pWorkspace;  // 连续
        GM_ADDR dsWorkspace; // 连续
        GM_ADDR tilingData;
        uint64_t actualRow = 0;
        uint64_t actualCol = 0;
        uint64_t processNums = 0;
        uint64_t curCoreN1Idx = 0;
        uint64_t curCoreS1Idx = 0;
        // K_OUT 离散 Packet 的原始 Q 行映射与地址信息。
        uint64_t qBatchBaseOffset = 0;
        uint32_t qSeqlen = 0;
        uint32_t headDim = 0;
        GBSAG::QPacket packet;
        uint32_t maskType = 0;
        uint32_t curKSeqIdx = 0;
        uint32_t kvSeqlen = 0;
        uint32_t groupSize = 1; // GQA group 大小，组聚合 Packet 的 LSE 搬运使用。

        // Methods
        __aicore__ inline Params() {}

        __aicore__ inline Params(GM_ADDR s_, GM_ADDR softmaxLse_, GM_ADDR dp_, GM_ADDR pWorkspace_,
                                 GM_ADDR dsWorkspace_, GM_ADDR tilingData_, uint64_t acutualRow_, uint64_t actualCol_,
                                 uint64_t processNums_, uint64_t curN1_, uint64_t curS1_, uint64_t qBatchBaseOffset_,
                                 uint32_t qSeqlen_, uint32_t headDim_, GBSAG::QPacket const &packet_,
                                 uint32_t maskType_, uint32_t curKSeqIdx_, uint32_t kvSeqlen_)
            : s(s_),
              softmaxLse(softmaxLse_),
              dp(dp_),
              pWorkspace(pWorkspace_),
              dsWorkspace(dsWorkspace_),
              tilingData(tilingData_),
              actualRow(acutualRow_),
              actualCol(actualCol_),
              processNums(processNums_),
              curCoreN1Idx(curN1_),
              curCoreS1Idx(curS1_),
              qBatchBaseOffset(qBatchBaseOffset_),
              qSeqlen(qSeqlen_),
              headDim(headDim_),
              packet(packet_),
              maskType(maskType_),
              curKSeqIdx(curKSeqIdx_),
              kvSeqlen(kvSeqlen_)
        {}
    };

    NpuArch::Arch::Resource<ArchTag> &resource;

    // Keep one 64-row buffer: event synchronization, not buffer count, is the
    // dominant serialization point in this epilogue.
    constexpr static uint64_t DOUBLE_BUFFER = 2;
    constexpr static uint64_t STAGES = 1;
    constexpr static uint64_t BNSD = 1;
    constexpr static uint64_t BSND = 2;
    constexpr static uint64_t TND = 0;
    constexpr static uint64_t BRCB_BASE_NUM = 8;
    constexpr static uint64_t REPEAT_BYTE = 256;

    constexpr static uint64_t BLOCK_BYTE_SIZE = 32;
    constexpr static uint64_t BLOCK_FP32_NUM = 8;
    constexpr static uint64_t BLOCK_16_NUM = 16;
    constexpr static uint64_t SFMG_HIGH_PERF_N_FACTOR = 8;
    constexpr static uint64_t SFMG_HIGH_PERF_D_FACTOR = 64;
    constexpr static uint64_t baseM = 64;
    uint64_t cBlockIdx = 0;
    uint64_t cubeCoreIdx = 0;
    uint64_t vecCoreIdx = 0;
    uint64_t row = 0; // 当前core需要处理q方向的s数
    uint64_t col = 0; // 当前core需要处理kv方向的s数
    uint64_t align32Col = 0;
    uint64_t align16Col = 0;
    uint64_t alignCol = 0;
    uint64_t alignRow = 0;
    uint64_t curCoreN1Idx = 0; // q_n
    uint64_t curCoreS1Idx = 0; // q_s
    uint64_t maxQSeqlen = 0;
    uint64_t maxKvSeqlen = 0;
    uint64_t n1 = 0; // q_n

    uint64_t usedVecCoreNums = 0;
    uint64_t p16BaseBufLen = 0;
    uint64_t p32BaseBufLen = 0;
    float scaleValue = 0.0f;

    GlobalTensor<float> sGm;                // (N s1 s2)
    GlobalTensor<float> softmaxLseGm;       // (N s1 1)
    GlobalTensor<float> dpGm;               // (N s1 s2)
    GlobalTensor<InputDType> pWorkspaceGm;  // (N s1 s2)
    GlobalTensor<InputDType> dsWorkspaceGm; // (N s1 s2)

    LocalTensor<float> sTensor[STAGES];
    LocalTensor<float> lseBrocTensor[STAGES];
    LocalTensor<float> pFp32Tensor[STAGES];
    LocalTensor<InputDType> p16Tensor[STAGES];
    LocalTensor<float> dpFp32Tensor[STAGES];
    LocalTensor<float> softmaxGradTensor[STAGES];
    LocalTensor<float> dsTensor[STAGES];
    LocalTensor<InputDType> ds16Tensor[STAGES];
    uint64_t qBatchBaseOffset = 0;
    uint32_t packetQSeqlen = 0;
    uint32_t packetHeadDim = 0;
    GBSAG::QPacket packet;
    uint32_t maskType = 0;
    uint32_t packetCurKSeqIdx = 0;
    uint32_t packetKvSeqlen = 0;
    uint32_t packetGroupSize = 1; // GQA group 大小，组聚合 Packet 语义下使用。

    constexpr static uint32_t MASK_TYPE_CAUSAL = 1;

    __aicore__ inline explicit SimpltSoftmax(NpuArch::Arch::Resource<ArchTag> &resource_)
        : resource(resource_)
    {
        // baseM =  128;
        // 分核 一个core 最大 128 * 128 一个vec 64 * 128
        uint64_t sBufferLen = baseM * 128 * sizeof(float);               // max
        uint64_t lBufferLen = baseM * sizeof(float);                     // max
        uint64_t lBrobBufferLen = BRCB_BASE_NUM * baseM * sizeof(float); // max
        uint64_t p32BufferLen = baseM * 128 * sizeof(float);
        uint64_t dpBufLen = sBufferLen;
        uint64_t p16BufLen = baseM * 128 * sizeof(InputDType);
        uint64_t ds16BufLen = p16BufLen;
        uint64_t dBufLen = BRCB_BASE_NUM * baseM * sizeof(float);
        p16BaseBufLen = p16BufLen;
        p32BaseBufLen = p32BufferLen;

        uint64_t sftBufferAlign = sBufferLen + p16BufLen + lBrobBufferLen + lBufferLen * BRCB_BASE_NUM;

        for (uint64_t i = 0; i < STAGES; i++) {
            // 第一轮 softmax 计算空间划分
            // 由于分核关系，所以的输入都能放入ub中
            // 空间示意图： S32(P32) || p16 || lseBroc || [lse 预留] || dpFp32(ds) || softmaxgrad || ds16
            sTensor[i] = resource.ubBuf.template GetBufferByByte<float>((sBufferLen / 2) * i);
            pFp32Tensor[i] = sTensor[i]; // 复用s
            p16Tensor[i] = resource.ubBuf.template GetBufferByByte<InputDType>(sBufferLen + (p16BufLen / 2) * i);
            // lseBroc 保存当前计算块的离散 LSE [M, 8]（原 lse 槽位空间仍预留，保持 UB 布局不变）。
            lseBrocTensor[i] =
                resource.ubBuf.template GetBufferByByte<float>(sBufferLen + p16BufLen + (lBrobBufferLen / 2) * i);

            // 第二轮 ds 计算空间划分
            dpFp32Tensor[i] = resource.ubBuf.template GetBufferByByte<float>(sftBufferAlign + (dpBufLen / 2) * i);
            // Packet D remains in UB between SoftmaxGrad and CalDs.
            softmaxGradTensor[i] = resource.ubBuf.template GetBufferByByte<float>(
                PacketSoftmaxUbLayout::D_OFFSET + (PacketSoftmaxUbLayout::D_BYTES / STAGES) * i);
            dsTensor[i] = dpFp32Tensor[i];
            ds16Tensor[i] = resource.ubBuf.template GetBufferByByte<InputDType>(sftBufferAlign + dpBufLen + dBufLen +
                                                                                (p16BufLen / 2) * i);
        }
    }

    __aicore__ inline ~SimpltSoftmax() {}

    template <int32_t CORE_TYPE = g_coreType>
    __aicore__ inline void operator()(Params const &params);

    template <>
    __aicore__ inline void operator()<AscendC::AIC>(Params const &params)
    {}

    __aicore__ inline void Init(Params const &params)
    {
        vecCoreIdx = GetBlockIdx();
        cBlockIdx = vecCoreIdx;
        __gm__ GenericBlockSparseAttentionGradTilingData *tilingData =
            reinterpret_cast<__gm__ GenericBlockSparseAttentionGradTilingData *>(params.tilingData);
        usedVecCoreNums = tilingData->usedVecCoreNum;

        if (cBlockIdx >= usedVecCoreNums) {
            return;
        }

        maxQSeqlen = tilingData->maxQSeqlen;
        maxKvSeqlen = tilingData->maxKvSeqlen;
        n1 = tilingData->numHeads; // q_n
        col = params.actualCol;
        align32Col = (col + BLOCK_FP32_NUM - 1) / BLOCK_FP32_NUM * BLOCK_FP32_NUM; // fp32 对齐后的列数
        align16Col = (col + BLOCK_16_NUM - 1) / BLOCK_16_NUM * BLOCK_16_NUM;
        alignCol = (align32Col % BLOCK_16_NUM != 0) ? align16Col : align32Col;
        curCoreN1Idx = params.curCoreN1Idx;
        curCoreS1Idx = params.curCoreS1Idx;
        qBatchBaseOffset = params.qBatchBaseOffset;
        packetQSeqlen = params.qSeqlen;
        packetHeadDim = params.headDim;
        packet = params.packet;
        maskType = params.maskType;
        packetCurKSeqIdx = params.curKSeqIdx;
        packetKvSeqlen = params.kvSeqlen;
        packetGroupSize = params.groupSize > 0 ? params.groupSize : 1;
        scaleValue = tilingData->scaleValue;

        uint64_t s2 = packetKvSeqlen;
        row = params.actualRow;
        if (row <= 0) {
            return;
        }

        // 初始化 GM
        sGm.SetGlobalBuffer((__gm__ float *)params.s);
        softmaxLseGm.SetGlobalBuffer((__gm__ float *)params.softmaxLse);
        dpGm.SetGlobalBuffer((__gm__ float *)params.dp);
        pWorkspaceGm.SetGlobalBuffer((__gm__ InputDType *)params.pWorkspace);
        dsWorkspaceGm.SetGlobalBuffer((__gm__ InputDType *)params.dsWorkspace);
    }

    template <>
    __aicore__ inline void operator()<AscendC::AIV>(Params const &params)
    {
        Init(params);

        if (cBlockIdx >= usedVecCoreNums || row <= 0) {
            return;
        }

        // col <= 128
        // 计算单loop的计算量及loop次数
        uint64_t eleBaseBuffNum = p32BaseBufLen / STAGES / sizeof(float); // 基本buffer块的元素数量
        uint64_t bufferRows = baseM / STAGES;                             // 一次lopp可以执行的最多row行数
        uint64_t rowLoopTimes = row / bufferRows;
        uint64_t tailRowNum = row - rowLoopTimes * bufferRows;

        uint64_t ping = 0;

        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);

        // 删除原入口的 V→MTE2 排空对。它存在的唯一理由
        // 是旧布局里 prepD 的 V（cast+front）会读写 [0,144K)，与 S/LSE/dP 的
        // MTE2 载入区互踩，S 载入必须等 prepD 的 V 全部排空。现在 prepD 整体
        // 迁入 [104K,244K) 私有区（PacketSoftmaxUbLayout），与本阶段 MTE2 覆写
        // 的 [0,104K) 无任何 RAW/WAR——S 载入在 MTE2 FIFO 中直接排在 prepD 的
        // 输入载入之后，与 prepD 的 V 并行，prepD 的 V 时间不再裸露。
        // D 的交接（front 写 → CalDs 读）是纯 V→V，按序流水天然安全。

        // 不包含尾行处理
        for (uint64_t i = 0; i < rowLoopTimes; i++) {
            uint64_t curS1Idx = curCoreS1Idx + i * bufferRows;
            int32_t gmRowOffset = i * bufferRows * col;
            compute(gmRowOffset, bufferRows, col, curS1Idx, ping);
            if (STAGES == DOUBLE_BUFFER) {
                ping = 1 - ping;
            }
        }

        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);

        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
        if (tailRowNum > 0) {
            uint64_t curS1Idx = curCoreS1Idx + rowLoopTimes * bufferRows;
            int32_t gmOffset = rowLoopTimes * bufferRows * col;
            uint64_t tempRow = tailRowNum;
            compute(gmOffset, tempRow, col, curS1Idx, ping);
            if (STAGES == DOUBLE_BUFFER) {
                ping = 1 - ping;
            }
        }
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
    }

    __aicore__ inline void compute(int32_t gmOffset, uint64_t row, uint64_t col, uint64_t curS1Idx, uint64_t ping)
    {
        LocalTensor<float> dLocal = softmaxGradTensor[ping];
        struct SimplySoftMaxInfo<InputDType> runSftInfo = {
            sTensor[ping], lseBrocTensor[ping], pFp32Tensor[ping], p16Tensor[ping], sGm[gmOffset], softmaxLseGm,
                pWorkspaceGm[gmOffset]
        };
        struct CalDsInfo<InputDType> runDsInfo = {
            dpFp32Tensor[ping], dLocal, pFp32Tensor[ping], ds16Tensor[ping], dpGm[gmOffset], dsWorkspaceGm[gmOffset]
        };

        CalSimplySoft(runSftInfo, row, col, curS1Idx, ping);
        CalDs(runDsInfo, row, col, curS1Idx, ping);
    }

    __aicore__ inline void ApplyCausalMaskByRow(LocalTensor<float> &pLocal, uint32_t localRowStart, uint32_t rows,
                                                uint32_t col, uint32_t rowStride, uint32_t packetRowBegin)
    {
        if (maskType != MASK_TYPE_CAUSAL || rows == 0 || col == 0 || packet.tileCount == 0) {
            return;
        }

        const int64_t offset = static_cast<int64_t>(packetKvSeqlen) - static_cast<int64_t>(packetQSeqlen);
        const int64_t kSeqStart = static_cast<int64_t>(packetCurKSeqIdx);
        const int64_t kSeqEnd = kSeqStart + static_cast<int64_t>(col);
        const int64_t kLast = kSeqEnd - 1;
        const uint32_t packetRowEnd = packetRowBegin + rows;

        // 精简：整 token 切分下 tile 恒均匀（packStart = s * tileRows），
        // 相交 tile 是连续区间，闭式定位首尾即可；rsvdBlockIdx 输入契约为
        // 升序，token 随 tile 下标单调递增，首尾 tile 即本 AIV 的 Q token 范围。
        const uint32_t tileRows = GBSAG::GetQPacketTileRows(packetGroupSize);
        // 段号用移位求（tileRows 为 2 的幂时），避免 AIV 标量软件除法。
        const uint32_t tileShift = GBSAG::GetQPacketTileShift(tileRows);
        const uint32_t firstTile = GBSAG::SegIndexByTileRows(packetRowBegin, tileRows, tileShift);
        const uint32_t lastTile = GBSAG::SegIndexByTileRows(packetRowEnd - 1, tileRows, tileShift);
        const uint32_t qMin = packet.qStart[firstTile];
        // token 按升序排列，最后一个相交 tile 的 token 即为 qMax。
        const uint32_t qMax = packet.qStart[lastTile];

        const int64_t firstLegal = static_cast<int64_t>(qMin) + offset;
        const int64_t lastLegal = static_cast<int64_t>(qMax) + offset;
        if (firstLegal >= kLast) {
            return;
        }
        if (lastLegal < kSeqStart) {
            AscendC::Duplicate(pLocal[localRowStart * rowStride], 0.0f, rows * rowStride);
            AscendC::PipeBarrier<PIPE_V>();
            return;
        }

        bool hasFullRun = false;
        uint32_t fullRunStart = 0;
        // 阶梯从"行"粒度升到"tile"粒度——tile 内 G 行共享 token/legalEnd，
        // 循环次数 rows(<=64) → 本 chunk 的 tile 数（G=16 时 64→4）；
        // qStart[] 读、三分类判定、full-run 边界全部按 tile 一次。
        // 快路径（rowStride 为 64 倍数，标准 128 宽块）：部分 tile 用跨行周期
        // bit 掩码 Duplicate——dstRepStride=rowStride/8 使每 repeat 落在一行的
        // 同一 64 列组上，掩码 {word,word} 双词同值，对"每 repeat 复用低 64bit"
        // 与"128bit 顺序消费"两种硬件掩码语义均正确；每 tile <=2 次调用替代
        // 每行 2 次。尾块（rowStride%64!=0）回退原逐行路径。
        const bool fastPath = (rowStride % 64) == 0 && (rowStride / 8) <= 255;
        const uint32_t countEachRepeat = REPEAT_BYTE / sizeof(float);
        uint32_t localRow = localRowStart;
        for (uint32_t t = firstTile; t <= lastTile; ++t) {
            const uint32_t tileBegin = t * tileRows;
            const uint32_t chunkBegin = tileBegin > packetRowBegin ? tileBegin : packetRowBegin;
            const uint32_t chunkEnd = tileBegin + tileRows < packetRowEnd ? tileBegin + tileRows : packetRowEnd;
            const uint32_t rowsInTile = chunkEnd - chunkBegin;

            const uint32_t qAbs = packet.qStart[t];
            const int64_t legalEnd = static_cast<int64_t>(qAbs) + offset;

            if (legalEnd < kSeqStart) {
                // 整 tile 全遮蔽：只记账，零 V-op。
                if (!hasFullRun) {
                    fullRunStart = localRow;
                    hasFullRun = true;
                }
                localRow += rowsInTile;
                continue;
            }
            if (hasFullRun) {
                AscendC::Duplicate(pLocal[fullRunStart * rowStride], 0.0f, (localRow - fullRunStart) * rowStride);
                hasFullRun = false;
            }
            if (legalEnd >= kLast) {
                // 之后所有 tile 的 token 更大，必然全可见。
                break;
            }

            // 部分 tile：suffixStart 对 tile 内所有行相同，保证在 [1, col) 内。
            const uint32_t suffixStart = static_cast<uint32_t>(legalEnd + 1 - kSeqStart);
            if (fastPath) {
                for (uint32_t lo = 0; lo < col; lo += 64) {
                    uint64_t word;
                    if (suffixStart <= lo) {
                        word = ~0ULL; // 该 64 列组整组清零。
                    } else if (suffixStart < lo + 64) {
                        word = ~0ULL << (suffixStart - lo); // 清 [suffixStart, lo+64)。
                    } else {
                        continue; // 该组全部合法，无需处理。
                    }
                    uint64_t dupMask[2] = {word, word};
                    AscendC::Duplicate(pLocal[localRow * rowStride + lo], 0.0f, dupMask,
                                       static_cast<uint8_t>(rowsInTile), 1, static_cast<uint8_t>(rowStride / 8));
                }
            } else {
                // 尾块回退：行距非 64 倍数，沿用原对齐拆分逐行路径。
                const uint32_t alignedStart = ((suffixStart + countEachRepeat - 1) / countEachRepeat) * countEachRepeat;
                for (uint32_t r = 0; r < rowsInTile; ++r) {
                    const uint32_t rowBase = (localRow + r) * rowStride;
                    if (suffixStart < alignedStart) {
                        const uint32_t blockStart = (suffixStart / countEachRepeat) * countEachRepeat;
                        const uint32_t prefixEnd = alignedStart < col ? alignedStart : col;
                        const uint32_t prefixCount = prefixEnd - suffixStart;
                        const uint32_t maskBegin = suffixStart - blockStart;
                        const uint64_t prefixBits = (static_cast<uint64_t>(1) << prefixCount) - 1;
                        uint64_t mask[1] = {prefixBits << maskBegin};
                        AscendC::Duplicate(pLocal[rowBase + blockStart], 0.0f, mask, 1, 1, 8);
                    }
                    if (alignedStart < col) {
                        AscendC::Duplicate(pLocal[rowBase + alignedStart], 0.0f, col - alignedStart);
                    }
                }
            }
            localRow += rowsInTile;
        }
        if (hasFullRun) {
            AscendC::Duplicate(pLocal[fullRunStart * rowStride], 0.0f,
                               (localRowStart + rows - fullRunStart) * rowStride);
        }
        // Duplicate 的逐 bit mask 版本会修改向量 mask 状态，后续向量算子需要恢复为标准模式。
        AscendC::ResetMask();
        AscendC::SetMaskNorm();
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ApplyCausalMaskToPacketRuns(LocalTensor<float> &pLocal, uint32_t rows, uint32_t col,
                                                       uint32_t rowStride, uint32_t packetRowBegin)
    {
        ApplyCausalMaskByRow(pLocal, 0, rows, col, rowStride, packetRowBegin);
    }

    /*
     * brief: simply softmax
     * runSftInfo : 计算需要的ub上的tensor
     * row: 需要计算的行数
     * col: 需要计算的列数
     * curS1Idx: 当前计算的query的 s 维度的idx
     */
    __aicore__ inline void CalSimplySoft(struct SimplySoftMaxInfo<InputDType> runSftInfo, uint64_t row, uint64_t col,
                                         uint64_t curS1Idx, uint64_t ping)
    {
        // simply softtmax
        LocalTensor<float> &sLocal = runSftInfo.sTensor;
        LocalTensor<float> &lseFp32Brc = runSftInfo.lseBrocTensor;
        LocalTensor<float> &p32Local = runSftInfo.pFp32Tensor;
        LocalTensor<InputDType> &p16Local = runSftInfo.pFp16Tensor;

        GlobalTensor<float> s = runSftInfo.sGm;
        GlobalTensor<float> lseGm = runSftInfo.lseGm;
        GlobalTensor<InputDType> pGm = runSftInfo.pGm;

        uint64_t countAlign = row * alignCol;

        auto eventId = ping ? EVENT_ID4 : EVENT_ID5;

        wait_flag(PIPE_MTE3, PIPE_MTE2, eventId);

        if (align32Col * sizeof(InputDType) % BLOCK_BYTE_SIZE == 0) {
            DataCopyPad(sLocal, s, {static_cast<uint16_t>(row), static_cast<uint32_t>(col * sizeof(float)), 0, 0, 0},
                        {true, 0, static_cast<uint8_t>(align32Col - col), 0});
        } else {
            DataCopyPad(sLocal, s, {static_cast<uint16_t>(row), static_cast<uint32_t>(col * sizeof(float)), 0, 1, 0},
                        {true, 0, static_cast<uint8_t>(align32Col - col), 0});
        }

        LseCopy(lseGm, lseFp32Brc, row, curS1Idx);

        set_flag(PIPE_MTE2, PIPE_V, eventId);
        wait_flag(PIPE_MTE2, PIPE_V, eventId);

        Muls(sLocal, sLocal, (float)scaleValue, countAlign);
        AscendC::PipeBarrier<PIPE_V>();

        SubBrcb(p32Local, sLocal, lseFp32Brc, row, alignCol);
        AscendC::PipeBarrier<PIPE_V>();

        Exp(p32Local, p32Local, countAlign);
        AscendC::PipeBarrier<PIPE_V>();

        ApplyCausalMaskToPacketRuns(p32Local, static_cast<uint32_t>(row), static_cast<uint32_t>(col),
                                    static_cast<uint32_t>(alignCol), static_cast<uint32_t>(curS1Idx));

        Cast(p16Local, p32Local, AscendC::RoundMode::CAST_ROUND, countAlign);
        AscendC::PipeBarrier<PIPE_V>();

        set_flag(PIPE_V, PIPE_MTE3, eventId);
        wait_flag(PIPE_V, PIPE_MTE3, eventId);

        DataCopyPad(pGm, p16Local,
                    {static_cast<uint16_t>(row), static_cast<uint32_t>(col * sizeof(InputDType)), 0, 0, 0});
    }

    /*
     * brief: cal ds = p * (dp - D)
     * runDsInfo : 计算需要的ub上的tensor
     * row: 需要计算的行数
     * col: 需要计算的列数
     * curS1Idx: 当前计算的query的 s 维度的idx
     */
    __aicore__ inline void CalDs(struct CalDsInfo<InputDType> runDsInfo, uint64_t row, uint64_t col, uint64_t curS1Idx,
                                 uint64_t ping)
    {
        LocalTensor<float> dpLocal = runDsInfo.dpFp32Tensor;
        LocalTensor<float> dLocal = runDsInfo.softmaxGradTensor;
        LocalTensor<InputDType> ds16Tensor = runDsInfo.dsFp16Tensor;
        LocalTensor<float> &p32Local = runDsInfo.pFp32Tensor;

        GlobalTensor<float> dp = runDsInfo.dpGm;
        GlobalTensor<InputDType> ds = runDsInfo.dsGm;

        uint64_t countAlign = row * alignCol;

        auto eventId = ping ? EVENT_ID4 : EVENT_ID5;

        if (align32Col * sizeof(InputDType) % BLOCK_BYTE_SIZE == 0) {
            DataCopyPad(dpLocal, dp, {static_cast<uint16_t>(row), static_cast<uint32_t>(col * sizeof(float)), 0, 0, 0},
                        {true, 0, static_cast<uint8_t>(align32Col - col), 0});
        } else {
            DataCopyPad(dpLocal, dp, {static_cast<uint16_t>(row), static_cast<uint32_t>(col * sizeof(float)), 0, 1, 0},
                        {true, 0, static_cast<uint8_t>(align32Col - col), 0});
        }
        // packed 路径的 D 已由 SoftmaxGrad 阶段留在 UB（PacketSoftmaxUbLayout），
        // 此处无需再从 GM 搬入。

        set_flag(PIPE_MTE2, PIPE_V, eventId);
        wait_flag(PIPE_MTE2, PIPE_V, eventId);

        SubBrcb(dpLocal, dpLocal, dLocal, row, alignCol);
        AscendC::PipeBarrier<PIPE_V>();

        Mul(dpLocal, p32Local, dpLocal, countAlign);
        AscendC::PipeBarrier<PIPE_V>();

        Cast(ds16Tensor, dpLocal, AscendC::RoundMode::CAST_ROUND, countAlign);

        set_flag(PIPE_V, PIPE_MTE3, eventId);
        wait_flag(PIPE_V, PIPE_MTE3, eventId);

        DataCopyPad(ds, ds16Tensor,
                    {static_cast<uint16_t>(row), static_cast<uint32_t>(col * sizeof(InputDType)), 0, 0, 0});
        set_flag(PIPE_MTE3, PIPE_MTE2, eventId);
    }

    /**
     * 根据 QPacket tile 的 token 与 head 偏移计算 LSE 元素偏移。
     * curCoreN1Idx 为 GQA group 首 head（qHeadBegin）。
     */
    __aicore__ inline uint64_t GetPackedLseOffset(uint32_t token, uint32_t headOffset)
    {
        if constexpr (INPUT_LAYOUT == TND) {
            const uint64_t tokenBase = qBatchBaseOffset / (n1 * packetHeadDim);
            return (tokenBase + token) * n1 + curCoreN1Idx + headOffset;
        } else if constexpr (INPUT_LAYOUT == BSND) {
            const uint64_t qBatchBase = qBatchBaseOffset / packetHeadDim;
            return qBatchBase + static_cast<uint64_t>(token) * n1 + curCoreN1Idx + headOffset;
        } else {
            const uint64_t lseBase = qBatchBaseOffset / packetHeadDim;
            return lseBase + (curCoreN1Idx + headOffset) * maxQSeqlen + token;
        }
    }

    /*
     * lse copy (packed 语义)
     * lse input shape (b n s 1) or (t n 1)
     * out shape (b n s 8) or (n s 8)
     * dtype float
     */
    __aicore__ inline void LseCopy(GlobalTensor<float> &LseGm, LocalTensor<float> &lseFp32Brc, uint64_t count,
                                   uint64_t curS1Idx)
    {
        constexpr uint32_t FP32_BYTES = sizeof(float);
        const uint64_t packetRowEnd = curS1Idx + count;
        // 组聚合：逐 tile（token）搬运 LSE。Gather 侧 head 维已经聚合，
        // LSE 同样以 token 为单位发射一次 2D 搬运：
        //   TND/BSND 的 token 内 head 连续；BNSD 的 head 按 maxQSeqlen 等距。
        // 仅当相邻 token 的源间距 gap 均匀（G==1；或 TND/BSND 且 G==n1）时
        // 合并连续 token，保持与原逐行实现相同的搬运次数下限。
        const uint32_t groupSize = packetGroupSize;
        // 精简：tile 恒均匀，packStart/segmentRows/headStart 闭式推导。
        const uint32_t tileRows = GBSAG::GetQPacketTileRows(groupSize);
        uint64_t pendSrc = 0;
        uint32_t pendRow = 0;
        uint32_t pendBursts = 0;
        uint32_t pendGap = 0;
        uint32_t pendToken = 0;
        uint32_t pendHeads = 0;
        // 闭式定位相交段区间，只遍历本 chunk 的段（首段从 curS1Idx
        // 起，末段止于 packetRowEnd，段边界增量推进），免去全段扫描求交。
        // 段号用移位求（tileRows 为 2 的幂时），避免 AIV 标量软件除法。
        const uint32_t tileShift = GBSAG::GetQPacketTileShift(tileRows);
        const uint64_t firstSeg = GBSAG::SegIndexByTileRows(static_cast<uint32_t>(curS1Idx), tileRows, tileShift);
        const uint64_t lastSeg =
            GBSAG::SegIndexByTileRows(static_cast<uint32_t>(packetRowEnd - 1), tileRows, tileShift);
        uint64_t segBegin = firstSeg * tileRows;
        uint64_t segEnd = segBegin + tileRows;
        uint64_t copyBegin = curS1Idx;
        for (uint64_t segment = firstSeg; segment <= lastSeg; ++segment) {
            const uint64_t copyEnd = segEnd < packetRowEnd ? segEnd : packetRowEnd;
            const uint32_t headOffset = packet.headStart0 + static_cast<uint32_t>(copyBegin - segBegin);
            const uint32_t token = packet.qStart[static_cast<uint32_t>(segment)];
            const uint32_t localRowStart = static_cast<uint32_t>(copyBegin - curS1Idx);
            const uint32_t copyHeads = static_cast<uint32_t>(copyEnd - copyBegin);

            // 段游标先行推进：mergeable 分支的 continue 不再回到循环尾，
            // 增量必须放在所有 continue 之前；后续计算只依赖已导出的
            // headOffset/token/localRowStart/copyHeads。
            copyBegin = segEnd;
            segBegin = segEnd;
            segEnd += tileRows;
            const uint64_t lseOffset = GetPackedLseOffset(token, headOffset);
            uint32_t gap = 0;
            bool uniformMerge = false;
            if constexpr (INPUT_LAYOUT == TND || INPUT_LAYOUT == BSND) {
                if (groupSize == 1) {
                    gap = static_cast<uint32_t>((n1 - 1) * FP32_BYTES);
                    uniformMerge = true;
                } else {
                    gap = 0;
                    uniformMerge = (groupSize == n1);
                }
            } else {
                if (groupSize == 1) {
                    gap = 0;
                    uniformMerge = true;
                } else {
                    gap = static_cast<uint32_t>((maxQSeqlen - 1) * FP32_BYTES);
                    uniformMerge = false;
                }
            }
            const bool mergeable = uniformMerge && pendBursts > 0 && localRowStart == pendRow + pendBursts &&
                                   token == pendToken + 1 && headOffset == 0 && pendHeads == groupSize &&
                                   copyHeads == groupSize;
            if (mergeable) {
                pendBursts += copyHeads;
                pendToken = token;
                pendHeads = copyHeads;
                continue;
            }
            if (pendBursts > 0) {
                DataCopyPad(lseFp32Brc[pendRow * BRCB_BASE_NUM], LseGm[pendSrc],
                            {static_cast<uint16_t>(pendBursts), FP32_BYTES, pendGap, 0, 0}, {false, 0, 0, 0});
            }
            pendSrc = lseOffset;
            pendRow = localRowStart;
            pendBursts = copyHeads;
            pendGap = gap;
            pendToken = token;
            pendHeads = copyHeads;
        }
        if (pendBursts > 0) {
            DataCopyPad(lseFp32Brc[pendRow * BRCB_BASE_NUM], LseGm[pendSrc],
                        {static_cast<uint16_t>(pendBursts), FP32_BYTES, pendGap, 0, 0}, {false, 0, 0, 0});
        }
    }

    /*
     * brief: Compute the elementwise multiplication of a tensor of shape (m, n) and a tensor of shape
     * ubIn0:[m, n], ubIn1[m, 8]
     */
    __aicore__ inline void SubBrcb(LocalTensor<float> const &ubOut, LocalTensor<float> const &ubIn0,
                                   LocalTensor<float> const &ubIn1, uint64_t row, uint64_t col)
    {
        // 分核逻辑的关系，矩阵shape不会超过[128,128], 若启动俩个vector，shape不会超过[64, 128]
        // 所以，一次sub， 最大可计算[128, 64], 迭代次数为行数, 一次迭代计算元素大小最多为64
        // 所以，对列按照64切分，循环sub计算
        uint32_t countEachRepeat = REPEAT_BYTE / sizeof(float);           // 每次迭代计算的元素 64
        uint32_t colLoop = (col + countEachRepeat - 1) / countEachRepeat; // 上取整
        uint32_t remain = col % countEachRepeat;
        uint64_t mask = countEachRepeat; // 参与计算的元素个数
        uint8_t repeatTimes = row;
        AscendC::BinaryRepeatParams repeatParams;
        repeatParams.dstBlkStride = 1;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1BlkStride = 0;
        repeatParams.dstRepStride = col / BLOCK_FP32_NUM;
        repeatParams.src0RepStride = col / BLOCK_FP32_NUM;
        repeatParams.src1RepStride = 1;

        for (uint32_t i = 0; i < colLoop; i++) {
            if (i == colLoop - 1 && remain != 0) {
                mask = remain;
            }
            AscendC::Sub(ubOut[i * countEachRepeat], ubIn0[i * countEachRepeat], ubIn1[0], mask, repeatTimes,
                         repeatParams);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
};

} // namespace NpuArch::Epilogue::Block

#endif // CATLASS_EPILOGUE_BLOCK_GBSAG_EPILOGUE_SIMPLY_SOFTMAX_HPP
