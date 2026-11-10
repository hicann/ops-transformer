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
 * \file mega_moe_kernel_a3_dedup.hpp
 * \brief MegaMoeKernel 的 Dedup（通信零冗余）成员函数类外定义。唯一包含点：
 *        mega_moe_kernel_a3.hpp 末尾，不可独立包含。
 */

#ifndef MEGA_MOE_KERNEL_A3_DEDUP_HPP
#define MEGA_MOE_KERNEL_A3_DEDUP_HPP

namespace Catlass::Gemm::Kernel {

// 与 mega_moe_kernel_a3.hpp 保持同一 unqualified 名称查找环境
using namespace AscendC;
using namespace Mc2Tiling;

// FP/LP 表清零（每 chunk 一次）；全核 barrier 由调用方 SyncAll 合并收口。
template <class BlockMmad_, class BlockScheduler_, class ElementGroupList_, class BlockEpilogue1_,
          class BlockEpilogue2_, class BlockEpilogue3_, class BlockEpilogueDedup_, bool DedupOn_>
CATLASS_DEVICE void
MegaMoeKernel<BlockMmad_, BlockScheduler_, ElementGroupList_, BlockEpilogue1_, BlockEpilogue2_, BlockEpilogue3_,
              BlockEpilogueDedup_, DedupOn_>::ResetDistTokenFirstPosV2(GM_ADDR distTokenFirstPos,
                                                                       GM_ADDR distTokenLastPos, int32_t num,
                                                                       uint32_t coreIdx, uint32_t coreNum)
{
    int32_t totalAlignBlocks = (num + DEDUP_ALIGN32_INT32S - 1) / DEDUP_ALIGN32_INT32S;
    int32_t blocksPerCore = totalAlignBlocks / coreNum;
    int32_t remainingBlocks = totalAlignBlocks % coreNum;
    int32_t coreLen = 0;
    int32_t coreOffset = 0;
    if (coreIdx < remainingBlocks) {
        int32_t myBlocks = blocksPerCore + 1;
        coreLen = myBlocks * DEDUP_ALIGN32_INT32S;
        coreOffset = coreIdx * myBlocks * DEDUP_ALIGN32_INT32S;
    } else {
        int32_t myBlocks = blocksPerCore;
        coreLen = myBlocks * DEDUP_ALIGN32_INT32S;
        coreOffset = remainingBlocks * (blocksPerCore + 1) * DEDUP_ALIGN32_INT32S +
                     (coreIdx - remainingBlocks) * myBlocks * DEDUP_ALIGN32_INT32S;
    }

    AscendC::GlobalTensor<int32_t> fpGm;
    fpGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(distTokenFirstPos));
    AscendC::GlobalTensor<int32_t> lpGm;
    lpGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(distTokenLastPos));

    // 重置缓冲借 resource.ubBuf 头部（与全 kernel UB 记账同源；调用时序上 prologue 已
    // 完成、路由表未装载，offset 0 区无活跃数据）
    AscendC::LocalTensor<int32_t> tmp = resource.ubBuf.template GetBufferByByte<int32_t>(0);

    if (coreLen > 0) {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::Duplicate(tmp, 0, coreLen);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::DataCopy(fpGm[coreOffset], tmp, coreLen);
        AscendC::DataCopy(lpGm[coreOffset], tmp, coreLen);
        // 写侧完成回执（跨核 DMA 可见性前提）
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }

    AscendC::PipeBarrier<PIPE_ALL>();
}

// 物理行 row 所属组：最大 g 使 groupBase[g] ≤ row（二分）。slotWidth>k 时源行不可按
// srcRow·k 直算，须经组基址换算。
template <class BlockMmad_, class BlockScheduler_, class ElementGroupList_, class BlockEpilogue1_,
          class BlockEpilogue2_, class BlockEpilogue3_, class BlockEpilogueDedup_, bool DedupOn_>
CATLASS_DEVICE int32_t
MegaMoeKernel<BlockMmad_, BlockScheduler_, ElementGroupList_, BlockEpilogue1_, BlockEpilogue2_, BlockEpilogue3_,
              BlockEpilogueDedup_, DedupOn_>::DedupGroupOf(int32_t row, const AscendC::LocalTensor<int32_t>& groupBase,
                                                           int32_t er)
{
    int32_t lo = 0;
    int32_t hi = er;
    while (lo + 1 < hi) {
        int32_t mid = (lo + hi) / 2;
        if (groupBase.GetValue(mid) <= row) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// Dedup token 粒度 Dispatch：同一 token 对每目标卡只发一次，替代基线 ProcessRecvRound 的
// dstEpIdx 轮转 pull 段。外层专家串行、组内 token 连续分核（同组 cot 互异无写竞争，跨组
// 由组尾 SyncAll 隔断）。坐标：cot = dstEpIdx·layoutChunkTokens + srcIdx/topK；FP/LP 存
// tokenId+1（0 = 未出现）。首现行远端拉取（同一 cot 只拉一次），非首现行本卡复用（源行经
// DedupGroupOf 换算）。FP/LP/rowMeta 为跨核 DMA：写侧回执（函数尾统一 Wait）+ 组尾 SyncAll<true>。
// 标量读走 1-int pingpong（事件 ID4/5，禁组级事件链——组级链 × SyncAll 会挂死）。
template <class BlockMmad_, class BlockScheduler_, class ElementGroupList_, class BlockEpilogue1_,
          class BlockEpilogue2_, class BlockEpilogue3_, class BlockEpilogueDedup_, bool DedupOn_>
CATLASS_DEVICE void
MegaMoeKernel<BlockMmad_, BlockScheduler_, ElementGroupList_, BlockEpilogue1_, BlockEpilogue2_, BlockEpilogue3_,
              BlockEpilogueDedup_, DedupOn_>::DedupTokenDispatchV2(Params const& params, uint16_t& syncgmm1Idx,
                                                                   uint32_t& dequantSumTemp)
{
    const int32_t k = static_cast<int32_t>(params.problemShape.k());
    const int32_t topK = static_cast<int32_t>(params.topK);
    const int32_t er = static_cast<int32_t>(params.expertPerRank);
    const int32_t slotWidth = static_cast<int32_t>(params.gmm1InputExpertSlotWidth);
    const int32_t aRowStride = k + static_cast<int32_t>(ALIGN_512);
    const uint32_t maxOut = static_cast<uint32_t>(params.maxOutputSize);
    const int64_t bs = params.problemShape.m();
    const int32_t layoutChunkTokens = static_cast<int32_t>(bs < PERMUTE_CHUNK ? bs : PERMUTE_CHUNK);

    // Dedup 标量 UB 区：路由缓存高址区以下 1KB
    const uint32_t rankMaskReserved = AlignUp(params.EP * sizeof(int32_t), BYTE_PER_BLK);
    const uint32_t tpeCacheBytes = AlignUp(params.EP * expertPerRankAligned * sizeof(int32_t), BYTE_PER_BLK);
    const uint32_t routeCacheBytes = AlignUp(routeCacheNum * sizeof(int32_t), BYTE_PER_BLK);
    const uint32_t routeCacheUbOffset = ArchTag::UB_SIZE - rankMaskReserved - tpeCacheBytes - routeCacheBytes;
    const uint32_t dedupUbBase = routeCacheUbOffset - DEDUP_SCALAR_AREA_MARGIN;
    AscendC::LocalTensor<int32_t> groupBase = resource.ubBuf.template GetBufferByByte<int32_t>(dedupUbBase);
    const uint32_t bufBase = dedupUbBase + AscendC::AlignUp(static_cast<uint32_t>((er + 1) * sizeof(int32_t)), 32U);
    AscendC::LocalTensor<int32_t> srcIdxBuf = resource.ubBuf.template GetBufferByByte<int32_t>(bufBase);
    AscendC::LocalTensor<int32_t> fpBuf = resource.ubBuf.template GetBufferByByte<int32_t>(bufBase + 32);
    AscendC::DataCopyExtParams scalarParams{static_cast<uint16_t>(1), 4U, 0U, 0U, 0U};
    AscendC::DataCopyPadExtParams<float> padParamsF;
    AscendC::DataCopyExtParams scalarParamsRm{static_cast<uint16_t>(1), DEDUP_ROWMETA_PAYLOAD_BYTES, 0U, 0U, 0U};
    // arch22 GM→UB 的 Ext 重载需 4 参（isPad=false 默认无填充）
    AscendC::DataCopyPadExtParams<int32_t> padParams;

    AscendC::GlobalTensor<int32_t> fpGm;
    fpGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspaceInfo.ptrDistTokenFirstPos));
    AscendC::GlobalTensor<int32_t> lpGm;
    lpGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspaceInfo.ptrDistTokenLastPos));
    // rowMeta（32B/行，格式见 dedup_layout）：本段装配，槽 3 预写 -1，函数尾 fixup 补写 last 行
    AscendC::GlobalTensor<int32_t> rowMetaGm;
    rowMetaGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspaceInfo.ptrDedupRowMeta));
    // probs 与 slots 同下标（dstPos 序平行数组）
    AscendC::GlobalTensor<float> packProbsGm;
    packProbsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(shmem() + peermemInfo.offsetExpandedDstToSrcRow));
    AscendC::GlobalTensor<int32_t> allPackGm;
    allPackGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(shmem() + peermemInfo.offsetExpandedDstToSrcRow));
    AscendC::GlobalTensor<int8_t> gmRemoteA;

    // 组基址表（maxOutputSize 截断口径与 AIC GMM1 侧一致）
    uint32_t acc = 0;
    for (int32_t g = 0; g < er; ++g) {
        uint32_t cm = cumsumMM.GetValue((static_cast<int32_t>(params.EP) - 1) * er + g);
        if (acc >= maxOut) {
            cm = 0;
        } else if (acc + cm > maxOut) {
            cm = maxOut - acc;
        }
        groupBase.SetValue(g, static_cast<int32_t>(acc));
        acc += cm;
    }
    groupBase.SetValue(er, static_cast<int32_t>(acc));

    // pingpong 标量事件链：函数头预置、双缓冲轮换、函数尾统一收口
    AscendC::LocalTensor<int32_t> pingBufs[2] = {srcIdxBuf, fpBuf};
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID4);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID5);
    uint32_t pingId = 0;

    for (int32_t groupIdx = 0; groupIdx < er; ++groupIdx) {
        const int32_t preGroupSum = groupBase.GetValue(groupIdx);
        const uint32_t currentM =
            static_cast<uint32_t>(groupBase.GetValue(groupIdx + 1)) - static_cast<uint32_t>(preGroupSum);

        uint32_t coreLen = currentM / coreNum;
        uint32_t rem = currentM % coreNum;
        uint32_t coreOffset = 0;
        if (coreIdx < rem) {
            coreLen += 1;
            coreOffset = coreIdx * coreLen;
        } else {
            coreOffset = rem * (coreLen + 1) + (coreIdx - rem) * coreLen;
        }

        for (uint32_t i = coreOffset; i < coreOffset + coreLen; ++i) {
            const int32_t tokenId = preGroupSum + static_cast<int32_t>(i); // 物理行号（轮内）
            // 反查源卡（cumsumMM 顺序扫描）
            int32_t dstEpIdx = 0;
            while (dstEpIdx < static_cast<int32_t>(params.EP) - 1 &&
                   static_cast<int32_t>(i) >= cumsumMM.GetValue(dstEpIdx * er + groupIdx)) {
                ++dstEpIdx;
            }
            const int32_t prevCumsum = (dstEpIdx == 0) ? 0 : cumsumMM.GetValue((dstEpIdx - 1) * er + groupIdx);
            const int32_t rowInEp = static_cast<int32_t>(i) - prevCumsum;
            const int32_t rowSrc = ubPreSumBeforeRankCache.GetValue(dstEpIdx * er + groupIdx);

            // srcIdx 读（打包段 slots）
            AscendC::TEventID evId = (pingId == 0) ? EVENT_ID4 : EVENT_ID5;
            AscendC::LocalTensor<int32_t> pbuf = pingBufs[pingId];
            AscendC::LocalTensor<float> pbufF = pbuf.ReinterpretCast<float>();
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evId);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(evId);
            AscendC::DataCopyPad(
                pbuf, allPackGm[dstEpIdx * DEDUP_PACK_SECTION_COUNT * layoutChunkTokens * topK + rowSrc + rowInEp],
                scalarParams, padParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(evId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(evId);
            // slots 元素 = srcIdx（低 24bit）| rankOrdinal（高 8bit），一次读双字段
            const int32_t packedSlot = pbuf.GetValue(0);
            const int32_t srcIdx = packedSlot & DEDUP_SLOT_SRCIDX_MASK;
            const int32_t rankOrdinal = (packedSlot >> DEDUP_SLOT_RANK_ORDINAL_SHIFT) & DEDUP_SLOT_RANK_ORDINAL_MASK;
            const int32_t cot = dstEpIdx * layoutChunkTokens + srcIdx / topK;

            // probs 读（dstPos 序）；srcIdx 已消费，pbuf 复用
            AscendC::DataCopyPad(pbufF[0],
                                 packProbsGm[dstEpIdx * DEDUP_PACK_SECTION_COUNT * layoutChunkTokens * topK +
                                             layoutChunkTokens * topK + rowSrc + rowInEp],
                                 scalarParams, padParamsF);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(evId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(evId);
            const int32_t probBits = pbuf.GetValue(0);

            // FP 读（0 = 未出现）
            AscendC::DataCopyPad(pbuf, fpGm[cot], scalarParams, padParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(evId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(evId);
            const int32_t firstPos = pbuf.GetValue(0);

            // LP 写 + 条件 FP 写（同一 tokenId+1 值）
            pbuf.SetValue(0, tokenId + 1);
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(evId);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(evId);
            AscendC::DataCopyPad(lpGm[cot], pbuf, scalarParams);
            if (firstPos == 0) {
                AscendC::DataCopyPad(fpGm[cot], pbuf, scalarParams);
            }
            // 防 WAR：rowMeta 覆写前等 LP/FP 的 MTE3 读完
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(evId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(evId);
            // rowMeta 装配写（16B：[anchorRow|cot|probBits|lastRow=-1]，
            // 槽 1 = cot 低 24bit | rankOrdinal 高 8bit，格式见 mega_moe_dedup_layout）
            const int32_t anchorRow = (firstPos == 0) ? tokenId : firstPos - 1;
            pbuf.SetValue(0, anchorRow);
            pbuf.SetValue(1, cot | (rankOrdinal << DEDUP_ROWMETA_RANK_ORDINAL_SHIFT));
            pbuf.SetValue(2, probBits);
            pbuf.SetValue(3, -1);
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(evId);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(evId);
            AscendC::DataCopyPad(rowMetaGm[tokenId * DEDUP_ROWMETA_ROW_INT32S], pbuf, scalarParamsRm);
            // 轮尾回执（下一轮轮顶消费，函数尾收口）
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evId);
            AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(evId);
            pingId = (pingId + 1) % 2;

            const int64_t gmOffsetA = static_cast<int64_t>(preGroupSum) * slotWidth + static_cast<int64_t>(i) * k;
            if (firstPos == 0) {
                // 首次：远端拉取（同一 cot 只拉一次，行步长 k+512）
                gmRemoteA.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(shmem(0, dstEpIdx) + peermemInfo.offsetA));
                FetchAndPreprocessInt8ToInt4(static_cast<uint64_t>(gmOffsetA), gmPerTokenScale1[tokenId],
                                             gmRemoteA[static_cast<int64_t>(srcIdx / topK) * aRowStride], 1, k);
            } else {
                // 非首次：本卡复用（源行 firstPos−1 经 DedupGroupOf 换算）
                const int32_t srcRow = firstPos - 1;
                const int32_t gSrc = DedupGroupOf(srcRow, groupBase, er);
                const int64_t srcOffset = static_cast<int64_t>(groupBase.GetValue(gSrc)) * slotWidth +
                                          static_cast<int64_t>(srcRow - groupBase.GetValue(gSrc)) * k;
                CopyGMToGM(gmA1I4_I8[gmOffsetA], gmA1I4_I8[srcOffset], k, UB_MOVE_NUM);
                CopyGMToGM(gmPerTokenScale1[tokenId], gmPerTokenScale1[srcRow], 1, UB_MOVE_NUM);
            }
        }

        // 组尾 SyncAll + flag（组→GMM1 握手，与基线 dispatch 同节奏）
        AscendC::SyncAll<true>();
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(syncgmm1Idx / CROSS_CORE_FLAG_MAX_SET_COUNT);
        syncgmm1Idx++;

        // SwiGLU 分批记账（与基线 dispatch 同式）
        if (dequantSumTemp + currentM <= maxOut) {
            dequantSumTemp += currentM;
        } else if (dequantSumTemp < maxOut) {
            dequantSumTemp = maxOut;
        }
        if (IsSyncTask(groupIdx, er)) {
            nSyncSwiglu++;
            dequantSum[nSyncSwiglu] = dequantSumTemp;
        }
    }

    // pingpong 事件链收口：等 FP/LP 尾写完成
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID4);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID5);

    // rowMeta fixup：把该 cot 链末行写入槽 3（其余行保持 -1），combine 免 LP 依赖。
    // cot 连续均分全核 + 8 行攒批；staging 槽 32B 步进（UB→GM 源地址须 32B 对齐）
    {
        const int32_t numCots = static_cast<int32_t>(params.EP) * layoutChunkTokens;
        const int32_t coreNumS = static_cast<int32_t>(coreNum);
        const int32_t segLen = (numCots + coreNumS - 1) / coreNumS;
        const int32_t segStart = static_cast<int32_t>(coreIdx) * segLen;
        const int32_t segEnd = segStart + segLen < numCots ? segStart + segLen : numCots;
        AscendC::LocalTensor<int32_t> lpBlock = resource.ubBuf.template GetBufferByByte<int32_t>(0);
        AscendC::DataCopyExtParams lpBlockParams{static_cast<uint16_t>(1), 0U, 0U, 0U, 0U};
        AscendC::LocalTensor<int32_t> fixupStage = resource.ubBuf.template GetBufferByByte<int32_t>(bufBase + 64U);
        constexpr int32_t fixupBatch = 8;
        int32_t fixupRows[fixupBatch];
        int32_t fixupPend = 0;
        if (segEnd > segStart) {
            lpBlockParams.blockLen = static_cast<uint32_t>((segEnd - segStart) * 4);
            AscendC::DataCopyPad(lpBlock, lpGm[segStart], lpBlockParams, padParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID4);
            for (int32_t j = segStart; j < segEnd; ++j) {
                const int32_t lastVal = lpBlock.GetValue(j - segStart);
                if (lastVal > 0) {
                    fixupRows[fixupPend] = lastVal - 1;
                    fixupStage.SetValue(fixupPend * DEDUP_ROWMETA_ROW_INT32S, fixupRows[fixupPend]);
                    ++fixupPend;
                }
                if (fixupPend == fixupBatch || (j + 1 == segEnd && fixupPend > 0)) {
                    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID4);
                    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID4);
                    for (int32_t w = 0; w < fixupPend; ++w) {
                        AscendC::DataCopyPad(
                            rowMetaGm[fixupRows[w] * DEDUP_ROWMETA_ROW_INT32S + DEDUP_ROWMETA_LASTROW_SLOT],
                            fixupStage[w * DEDUP_ROWMETA_ROW_INT32S], scalarParams);
                    }
                    // 防 WAR：等 MTE3 读完再覆写（批粒度回执）
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID4);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID4);
                    fixupPend = 0;
                }
            }
        }
    }
}

// ================= Dedup Combine v2 =================
// 结构与 DedupTokenDispatchV2 对称（专家串行 + token 连续分核）。GMM2 就绪按组数据驱动：
// 头部只等组 0 flag，组循环内追赶式消费组 g+1 flag，末组豁免组尾 barrier。rowMeta 与
// scale2 按 64 行切片 pingpong 批读；SEND 行回执延迟消费（Finalize 补取）；组尾 SyncAll
// 双承重（GMM 发布 + 锚点链跨核可见）。

template <class BlockMmad_, class BlockScheduler_, class ElementGroupList_, class BlockEpilogue1_,
          class BlockEpilogue2_, class BlockEpilogue3_, class BlockEpilogueDedup_, bool DedupOn_>
CATLASS_DEVICE void
MegaMoeKernel<BlockMmad_, BlockScheduler_, ElementGroupList_, BlockEpilogue1_, BlockEpilogue2_, BlockEpilogue3_,
              BlockEpilogueDedup_, DedupOn_>::DedupIssueRowMetaSlice(AscendC::GlobalTensor<int32_t>& rowMetaGm,
                                                                     int32_t preGroupSumG, uint32_t coreOffsetG,
                                                                     uint32_t coreLenG, uint32_t sliceRows, uint32_t s,
                                                                     AscendC::LocalTensor<int32_t> (&ubRowMetaSl)[2],
                                                                     AscendC::LocalTensor<float> (&ubScale2Sl)[2],
                                                                     AscendC::DataCopyExtParams& sliceParamsRM,
                                                                     AscendC::DataCopyExtParams& sliceParamsS2)
{
    const uint32_t rowLo = s * sliceRows;
    uint32_t nrows = coreLenG - rowLo;
    if (nrows > sliceRows) {
        nrows = sliceRows;
    }
    sliceParamsRM.blockLen = static_cast<uint32_t>(nrows * 32);
    sliceParamsS2.blockLen = static_cast<uint32_t>(nrows * 4);
    AscendC::DataCopyPad(
        ubRowMetaSl[s & 1],
        rowMetaGm[(preGroupSumG + static_cast<int32_t>(coreOffsetG + rowLo)) * DEDUP_ROWMETA_ROW_INT32S], sliceParamsRM,
        AscendC::DataCopyPadExtParams<int32_t>{});
    AscendC::DataCopyPad(ubScale2Sl[s & 1], gmPerTokenScale2[preGroupSumG + static_cast<int32_t>(coreOffsetG + rowLo)],
                         sliceParamsS2, AscendC::DataCopyPadExtParams<float>{});
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>((s & 1) ? EVENT_ID7 : EVENT_ID6);
}

// 行派生：rowMeta 切片标量读 + cot 拆分 + 坐标换算 → RowCtx。i 为组内绝对行号；
// 片末行补发下下片（与行计算重叠）。
template <class BlockMmad_, class BlockScheduler_, class ElementGroupList_, class BlockEpilogue1_,
          class BlockEpilogue2_, class BlockEpilogue3_, class BlockEpilogueDedup_, bool DedupOn_>
CATLASS_DEVICE auto
MegaMoeKernel<BlockMmad_, BlockScheduler_, ElementGroupList_, BlockEpilogue1_, BlockEpilogue2_, BlockEpilogue3_,
              BlockEpilogueDedup_, DedupOn_>::DedupDeriveRow(AscendC::GlobalTensor<int32_t>& rowMetaGm,
                                                             AscendC::LocalTensor<int32_t> (&ubRowMetaSl)[2],
                                                             AscendC::LocalTensor<float> (&ubRowMetaSlF)[2],
                                                             AscendC::LocalTensor<float> (&ubScale2Sl)[2],
                                                             AscendC::DataCopyExtParams& sliceParamsRM,
                                                             AscendC::DataCopyExtParams& sliceParamsS2, uint32_t i,
                                                             uint32_t coreOffsetG, uint32_t coreLenG,
                                                             int32_t preGroupSumG, int32_t n2T,
                                                             int32_t layoutChunkTokensT, uint32_t& issued,
                                                             uint32_t totalSlices) ->
    typename BlockEpilogueDedup::RowCtx
{
    const uint32_t rowIdx = i - coreOffsetG;
    const uint32_t sliceIdx = rowIdx / DEDUP_COMBINE_SLICE_ROWS;
    const uint32_t inSlice = rowIdx % DEDUP_COMBINE_SLICE_ROWS; // 片内偏移（缓冲按片加载）
    const uint32_t bufId = sliceIdx & 1;
    if (inSlice == 0) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(bufId ? EVENT_ID7 : EVENT_ID6);
    }
    const int32_t tokenId = preGroupSumG + static_cast<int32_t>(i);
    const int32_t anchorRow = ubRowMetaSl[bufId].GetValue(inSlice * DEDUP_ROWMETA_ROW_INT32S + 0);
    // 槽 1 = cot（低 24bit）| rankOrdinal（高 8bit）
    const int32_t cotPacked = ubRowMetaSl[bufId].GetValue(inSlice * DEDUP_ROWMETA_ROW_INT32S + 1);
    const int32_t cot = cotPacked & DEDUP_ROWMETA_COT_MASK;
    const int32_t rankOrdinal = (cotPacked >> DEDUP_ROWMETA_RANK_ORDINAL_SHIFT) & DEDUP_SLOT_RANK_ORDINAL_MASK;
    const float prob = ubRowMetaSlF[bufId].GetValue(inSlice * DEDUP_ROWMETA_ROW_INT32S + 2);
    const int32_t lastRow = ubRowMetaSl[bufId].GetValue(inSlice * DEDUP_ROWMETA_ROW_INT32S + 3);
    const float scale = ubScale2Sl[bufId].GetValue(inSlice);

    // 片末读毕即补发下下片（与行计算重叠）
    const bool lastRowOfSlice = (inSlice == DEDUP_COMBINE_SLICE_ROWS - 1) || (rowIdx + 1 == coreLenG);
    if (lastRowOfSlice && issued < totalSlices) {
        DedupIssueRowMetaSlice(rowMetaGm, preGroupSumG, coreOffsetG, coreLenG, DEDUP_COMBINE_SLICE_ROWS, issued,
                               ubRowMetaSl, ubScale2Sl, sliceParamsRM, sliceParamsS2);
        issued++;
    }

    typename BlockEpilogueDedup::RowCtx ctx;
    ctx.curHiOff = (2LL * tokenId) * n2T;
    ctx.anchorF32Off = static_cast<int64_t>(anchorRow) * n2T;
    ctx.needAnchor = (anchorRow != tokenId);
    ctx.isLast = (lastRow == tokenId);
    ctx.dstRank = cot / layoutChunkTokensT;
    ctx.tokenInChunk = cot - ctx.dstRank * layoutChunkTokensT;
    ctx.dstRankOrdinal = rankOrdinal;
    ctx.scale = scale;
    ctx.prob = prob;
    return ctx;
}

template <class BlockMmad_, class BlockScheduler_, class ElementGroupList_, class BlockEpilogue1_,
          class BlockEpilogue2_, class BlockEpilogue3_, class BlockEpilogueDedup_, bool DedupOn_>
CATLASS_DEVICE void MegaMoeKernel<BlockMmad_, BlockScheduler_, ElementGroupList_, BlockEpilogue1_, BlockEpilogue2_,
                                  BlockEpilogue3_, BlockEpilogueDedup_, DedupOn_>::DedupCombineV2(Params const& params)
{
    const int32_t n2 = static_cast<int32_t>(params.problemShape.k());
    const int32_t er = static_cast<int32_t>(params.expertPerRank);
    const uint32_t maxOut = static_cast<uint32_t>(params.maxOutputSize);
    const int64_t bs = params.problemShape.m();
    const int32_t layoutChunkTokens = static_cast<int32_t>(bs < PERMUTE_CHUNK ? bs : PERMUTE_CHUNK);

    AscendC::GlobalTensor<int32_t> rowMetaGm;
    rowMetaGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspaceInfo.ptrDedupRowMeta));
    // gmC2 锚点槽 float 视图（槽 = 2·n₂ 半精度行位宽，FP32 单行占用；与 gmC2 同基址）
    AscendC::GlobalTensor<float> gmC2Anchor;
    gmC2Anchor.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspaceInfo.ptrC2));

    // 行级 epilogue 构造（UB 预算公式与 CacheRouteTablesToUb 同源）；cardsPerToken = K̂ =
    // min(topK, EP)——D 区 token 块行宽，与 host dSize 公式同口径
    const uint32_t rankMaskReserved = AlignUp(params.EP * sizeof(int32_t), BYTE_PER_BLK);
    const uint32_t tpeCacheBytes = AlignUp(params.EP * expertPerRankAligned * sizeof(int32_t), BYTE_PER_BLK);
    const uint32_t routeCacheBytes = AlignUp(routeCacheNum * sizeof(int32_t), BYTE_PER_BLK);
    const uint32_t routeCacheUbOffset = ArchTag::UB_SIZE - rankMaskReserved - tpeCacheBytes - routeCacheBytes;
    const int32_t cardsPerToken = (static_cast<int32_t>(params.topK) < static_cast<int32_t>(params.EP)) ?
                                      static_cast<int32_t>(params.topK) :
                                      static_cast<int32_t>(params.EP);
    typename BlockEpilogueDedup::Params dedupEpParams(n2, cardsPerToken, layoutChunkTokens,
                                                      reinterpret_cast<__gm__ float*>(params.ptrBias2), params.listLen,
                                                      shmem, peermemInfo.offsetD, routeCacheUbOffset);
    BlockEpilogueDedup dedupEp(resource, dedupEpParams);

    // 切片批读流选址：优先紧随 epilogue scratch，不足时借 route cache 区底部
    const uint32_t sliceBase = (routeCacheUbOffset >= dedupEp.scratchBytes + DEDUP_COMBINE_SLICE_BYTES) ?
                                   dedupEp.scratchBytes :
                                   routeCacheUbOffset;

    // 切片批读流（rowMeta 32B/行 + scale2 4B/行，双拷贝一次 MTE2_S Set）
    AscendC::LocalTensor<int32_t> ubRowMetaSl[2];
    AscendC::LocalTensor<float> ubRowMetaSlF[2];
    AscendC::LocalTensor<float> ubScale2Sl[2];
    for (int b = 0; b < 2; ++b) {
        ubRowMetaSl[b] =
            resource.ubBuf.template GetBufferByByte<int32_t>(sliceBase + b * DEDUP_COMBINE_SLICE_ROWS * 32);
        ubRowMetaSlF[b] = ubRowMetaSl[b].template ReinterpretCast<float>();
        ubScale2Sl[b] = resource.ubBuf.template GetBufferByByte<float>(sliceBase + 2 * DEDUP_COMBINE_SLICE_ROWS * 32 +
                                                                       b * DEDUP_COMBINE_SLICE_ROWS * 4);
    }
    AscendC::DataCopyExtParams sliceParamsRM{static_cast<uint16_t>(1), 0U, 0U, 0U, 0U};
    AscendC::DataCopyExtParams sliceParamsS2{static_cast<uint16_t>(1), 0U, 0U, 0U, 0U};

    icache_preload(8);

    // 头部：组 0 切片 + bias 装载藏 flag0 等待窗
    int32_t syncLoopIdx = 0;
    {
        uint32_t headM = cumsumMM.GetValue((static_cast<int32_t>(params.EP) - 1) * er);
        if (headM > maxOut) {
            headM = maxOut; // 单轮约束下恒不触发（防御，与基线钳制同构）
        }
        uint32_t coreLen0 = headM / coreNum;
        uint32_t rem0 = headM % coreNum;
        uint32_t coreOffset0 = 0;
        if (coreIdx < rem0) {
            coreLen0 += 1;
            coreOffset0 = coreIdx * coreLen0;
        } else {
            coreOffset0 = rem0 * (coreLen0 + 1) + (coreIdx - rem0) * coreLen0;
        }
        if (coreLen0 > 0) {
            DedupIssueRowMetaSlice(rowMetaGm, 0, coreOffset0, coreLen0, DEDUP_COMBINE_SLICE_ROWS, 0, ubRowMetaSl,
                                   ubScale2Sl, sliceParamsRM, sliceParamsS2);
            if (coreLen0 > DEDUP_COMBINE_SLICE_ROWS) {
                DedupIssueRowMetaSlice(rowMetaGm, 0, coreOffset0, coreLen0, DEDUP_COMBINE_SLICE_ROWS, 1, ubRowMetaSl,
                                       ubScale2Sl, sliceParamsRM, sliceParamsS2);
            }
        }
    }
    dedupEp.IssueGroupBiasLoad(0);
    AscendC::CrossCoreWaitFlag<0x2>(syncLoopIdx / CROSS_CORE_FLAG_MAX_SET_COUNT);
    syncLoopIdx++;
    AscendC::SyncAll<true>();

    int32_t preGroupSum = 0;
    for (int32_t groupIdx = 0; groupIdx < er; ++groupIdx) {
        uint32_t currentM = cumsumMM.GetValue((static_cast<int32_t>(params.EP) - 1) * er + groupIdx);
        if (static_cast<uint32_t>(preGroupSum) >= maxOut) {
            currentM = 0;
        } else if (static_cast<uint32_t>(preGroupSum) + currentM > maxOut) {
            currentM = maxOut - static_cast<uint32_t>(preGroupSum);
        }

        // 专家内 token 连续分核
        uint32_t coreLen = currentM / coreNum;
        uint32_t rem = currentM % coreNum;
        uint32_t coreOffset = 0;
        if (coreIdx < rem) {
            coreLen += 1;
            coreOffset = coreIdx * coreLen;
        } else {
            coreOffset = rem * (coreLen + 1) + (coreIdx - rem) * coreLen;
        }

        const uint32_t totalSlices = (coreLen + DEDUP_COMBINE_SLICE_ROWS - 1) / DEDUP_COMBINE_SLICE_ROWS;
        // 已预发射片数（组 0 由头部、组 g≥1 由上组尾）
        uint32_t issued = (coreLen > 0) ? (totalSlices > 1 ? 2 : 1) : 0;

        // 跨行流水：派生/IssueRead/ComputeWrite 逐行背靠背，缓冲集按行序轮转（集复用隔
        // 一整行，解除 MTE3 写 vs 下一行 MTE2 读串行链）。勿改为「读先行一拍」形态
        // （实测非确定竞态）
        for (uint32_t i = coreOffset; i < coreOffset + coreLen; ++i) {
            const typename BlockEpilogueDedup::RowCtx dedupCtx =
                DedupDeriveRow(rowMetaGm, ubRowMetaSl, ubRowMetaSlF, ubScale2Sl, sliceParamsRM, sliceParamsS2, i,
                               coreOffset, coreLen, preGroupSum, n2, layoutChunkTokens, issued, totalSlices);
            dedupEp.IssueRead(gmC2, gmC2Anchor, dedupCtx);
            dedupEp.ComputeWrite(gmC2Anchor, dedupCtx);
        }
        // 组尾收口：回执取走；锚点写退休由组尾 SyncAll<true> 保证
        dedupEp.Drain();

        // 下一组切片预取 + GMM2 flag 追赶消费：flag wait 须 hoist 到组尾 barrier 之前
        if (groupIdx + 1 < er) {
            uint32_t nextM = cumsumMM.GetValue((static_cast<int32_t>(params.EP) - 1) * er + groupIdx + 1);
            const uint32_t nextPre = static_cast<uint32_t>(preGroupSum) + currentM;
            if (nextPre >= maxOut) {
                nextM = 0;
            } else if (nextPre + nextM > maxOut) {
                nextM = maxOut - nextPre;
            }
            uint32_t nextLen = nextM / coreNum;
            uint32_t nextRem = nextM % coreNum;
            uint32_t nextOffset = 0;
            if (coreIdx < nextRem) {
                nextLen += 1;
                nextOffset = coreIdx * nextLen;
            } else {
                nextOffset = nextRem * (nextLen + 1) + (coreIdx - nextRem) * nextLen;
            }
            const uint32_t nextTotal = (nextLen + DEDUP_COMBINE_SLICE_ROWS - 1) / DEDUP_COMBINE_SLICE_ROWS;
            const int32_t nextPreSum = static_cast<int32_t>(nextPre);
            if (nextLen > 0) {
                DedupIssueRowMetaSlice(rowMetaGm, nextPreSum, nextOffset, nextLen, DEDUP_COMBINE_SLICE_ROWS, 0,
                                       ubRowMetaSl, ubScale2Sl, sliceParamsRM, sliceParamsS2);
                if (nextTotal > 1) {
                    DedupIssueRowMetaSlice(rowMetaGm, nextPreSum, nextOffset, nextLen, DEDUP_COMBINE_SLICE_ROWS, 1,
                                           ubRowMetaSl, ubScale2Sl, sliceParamsRM, sliceParamsS2);
                }
            }
            for (; syncLoopIdx <= groupIdx + 1; syncLoopIdx++) {
                AscendC::CrossCoreWaitFlag<0x2>(syncLoopIdx / CROSS_CORE_FLAG_MAX_SET_COUNT);
            }
            // 组尾 barrier（末组豁免）：锚点跨组跨核可见
            AscendC::SyncAll<true>();
            // 组 g+1 bias 预装载（本核有行才装），须在 barrier 之后：提前入队时 MTE2
            // 覆写 ubBiasList 与本组末行 V 的 Add(bias) 读构成 WAR
            if (nextLen > 0) {
                dedupEp.IssueGroupBiasLoad(groupIdx + 1);
            }
        }
        preGroupSum += static_cast<int32_t>(currentM);
    }
    // 末组 Set（index er）刻意不消费：其在 AIC 末组 SyncAll 之后才发出，此处补取会与
    // 该 SyncAll 互锁（实测 aicore timeout）；残留一次为协议固有形态（与基线一致）
    // SEND 悬挂回执补取
    dedupEp.Finalize();
}

} // namespace Catlass::Gemm::Kernel

#endif // MEGA_MOE_KERNEL_A3_DEDUP_HPP
