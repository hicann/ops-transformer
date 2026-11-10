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
 * \file moe_permute_prologue_dedup.hpp
 * \brief 零冗余（Dedup）Prologue：量化按原始 token 序写 A 区，排序流按专家桶直写打包段，
 *        不散射数据；其余与基线 moe_permute_prologue.hpp 同构。
 */

#ifndef MOE_PERMUTE_PROLOGUE_DEDUP_HPP
#define MOE_PERMUTE_PROLOGUE_DEDUP_HPP

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "../utils/const_args.hpp"

namespace MegaMoeDedup {

template <class ElementSrc_>
class DedupPrologue {
public:
    using ArchTag = Catlass::Arch::AtlasA2; // 与基线 prologue policy 一致
    using ElementSrc = ElementSrc_;
    using ElementDst = int8_t; // A 区恒为 int8 + 行尾 scale

    // 量化行流水双缓冲
    static constexpr uint32_t UB_STAGES = 2;
    // 打包 run 缓冲下限，兼作 core0 哨兵补尾 staging
    static constexpr int64_t RUN_BUF_MIN_ELEMS = 2048;

    struct Params {
        int64_t numTokens; // 本 chunk 行数（chunkTokens）
        int64_t hidden;
        int64_t numTopk;
        int64_t numExperts;     // 全局专家数 = EP·EPR
        int64_t expertsPerRank; // EPR：rankOrdinal 计算的卡号因子（卡 = 专家号 / EPR）
        int64_t alignedNumExperts;
        uint32_t alignedNumExpertsBytes;
        uint32_t rowStride; // A 区行步长 = hidden + 512（与基线 prologue 同式）
        uint32_t outLen;    // 单行拷出长度 = hidden + 32（行尾 scale 所在 32B block 一并拷出）
        int64_t streamLen;  // N = numTokens·numTopk（打包段 slots/probs 两平面各 N 个元素）

        CATLASS_DEVICE
        Params(int64_t numTokens_, int64_t hidden_, int64_t numTopk_, int64_t numExperts_, int64_t expertsPerRank_)
            : numTokens(numTokens_),
              hidden(hidden_),
              numTopk(numTopk_),
              numExperts(numExperts_),
              expertsPerRank(expertsPerRank_),
              alignedNumExperts(RoundUp<8>(numExperts_)),
              alignedNumExpertsBytes(static_cast<uint32_t>(alignedNumExperts * sizeof(int32_t))),
              rowStride(static_cast<uint32_t>(hidden_) + ALIGN_512),
              outLen(static_cast<uint32_t>(hidden_) + Catlass::BYTE_PER_BLK),
              streamLen(numTokens_ * numTopk_)
        {}
    };

    CATLASS_DEVICE
    DedupPrologue(Catlass::Arch::Resource<ArchTag>& resource, Params const& params)
        : params_(params)
    {
        // A3 上逻辑核索引需按 subblock 展开
        if ASCEND_IS_AIV {
            coreIdx = get_block_idx() + get_subblockid() * get_block_num();
            coreNum = get_block_num() * get_subblockdim();
        } else {
            coreIdx = AscendC::GetBlockIdx();
            coreNum = AscendC::GetBlockNum();
        }
        const int64_t hidden = params_.hidden;
        int64_t tokenStartIdx = 0;
        int64_t tokenEndIdx = 0;
        int64_t numTokensCurCore = 0;
        SplitCore(params_.numTokens, tokenStartIdx, tokenEndIdx, numTokensCurCore);

        uint32_t ubOffset = 0;
        // 常驻区（计数/走查两阶段共用）
        coreExpertCount = resource.ubBuf.template GetBufferByByte<int32_t>(ubOffset);
        ubOffset += params_.alignedNumExpertsBytes;
        // 他核计数 scratch（coreExpertCount 保持本核原值，走查复用）
        scratchCount = resource.ubBuf.template GetBufferByByte<int32_t>(ubOffset);
        ubOffset += params_.alignedNumExpertsBytes;
        totalExpertCount = resource.ubBuf.template GetBufferByByte<int32_t>(ubOffset);
        ubOffset += params_.alignedNumExpertsBytes;
        prefixExpertCount = resource.ubBuf.template GetBufferByByte<int32_t>(ubOffset);
        ubOffset += params_.alignedNumExpertsBytes;
        tokenPerExpertOffset = resource.ubBuf.template GetBufferByByte<int32_t>(ubOffset);
        ubOffset += params_.alignedNumExpertsBytes;
        // 桶在 UB run 区的基址 / 落位游标
        localOff = resource.ubBuf.template GetBufferByByte<int32_t>(ubOffset);
        ubOffset += params_.alignedNumExpertsBytes;
        localCursor = resource.ubBuf.template GetBufferByByte<int32_t>(ubOffset);
        ubOffset += params_.alignedNumExpertsBytes;
        // 本核 topkIds 切片缓冲（各核尺寸不同）
        const uint32_t perCoreBytes =
            RoundUp<Catlass::BYTE_PER_BLK>(static_cast<uint32_t>(numTokensCurCore * params_.numTopk * sizeof(int32_t)));
        topkIds = resource.ubBuf.template GetBufferByByte<int32_t>(ubOffset);
        ubOffset += perCoreBytes;
        // union 区：量化行流水 / 打包直写同址复用（量化 MTE3 落盘并两次 SyncAll 后才打包）
        const uint32_t unionBase = ubOffset;
        uint32_t quantOffset = unionBase;
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            tokenInList[i] = resource.ubBuf.template GetBufferByByte<ElementSrc>(quantOffset);
            quantOffset += RoundUp<Catlass::BYTE_PER_BLK>(static_cast<uint32_t>(hidden * sizeof(ElementSrc)));
            tokenOutList[i] = resource.ubBuf.template GetBufferByByte<ElementDst>(quantOffset);
            quantOffset += RoundUp<Catlass::BYTE_PER_BLK>(static_cast<uint32_t>(hidden * sizeof(ElementDst))) +
                           Catlass::BYTE_PER_BLK;
        }
        // 量化临时缓冲：work [0,h) + abs/max 复用 [h,2h)；int32/half 复用 work（与基线同式）
        constexpr int32_t QUANT_TMP_FLOATS_PER_HIDDEN = 2;
        tmpBuffer = resource.ubBuf.template GetBufferByByte<float>(quantOffset);
        quantOffset +=
            RoundUp<Catlass::BYTE_PER_BLK>(static_cast<uint32_t>(QUANT_TMP_FLOATS_PER_HIDDEN * hidden * sizeof(float)));

        // 打包缓冲：slots/probs 两平面 run 区（rankOrdinal 内联 slots 高 8bit）+ probs 只读切片
        runBufElems = numTokensCurCore * params_.numTopk + 8 * params_.numExperts;
        if (runBufElems < RUN_BUF_MIN_ELEMS) {
            runBufElems = RUN_BUF_MIN_ELEMS;
        }
        uint32_t packOffset = unionBase;
        const uint32_t runBytes = RoundUp<Catlass::BYTE_PER_BLK>(static_cast<uint32_t>(runBufElems * sizeof(int32_t)));
        slotsRun = resource.ubBuf.template GetBufferByByte<int32_t>(packOffset);
        packOffset += runBytes;
        probsRun = resource.ubBuf.template GetBufferByByte<float>(packOffset);
        packOffset += runBytes;
        probsSlice = resource.ubBuf.template GetBufferByByte<float>(packOffset);
        packOffset += perCoreBytes;
        // 行索引表 staging（token·topK 项，逐核连续；量化区已死，与打包区同址复用）
        rowIdxRun = resource.ubBuf.template GetBufferByByte<int32_t>(packOffset);
        packOffset += (perCoreBytes > 0U) ? perCoreBytes : Catlass::BYTE_PER_BLK;
    }

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementSrc> const& gmX, AscendC::GlobalTensor<int32_t> const& gmTopkIds,
                    AscendC::GlobalTensor<float> const& gmProbs, AscendC::GlobalTensor<ElementDst>& gmA,
                    AscendC::GlobalTensor<int32_t>& gmPackSlots, AscendC::GlobalTensor<float>& gmPackProbs,
                    AscendC::GlobalTensor<int32_t>& gmTokenPerExpert, GM_ADDR workspaceCountsAddr,
                    GM_ADDR workspaceExpandRowIdxAddr, GM_ADDR xActiveMaskAddr = nullptr)
    {
        // 行索引表落 workspace expandedRowIdx 段（Dedup 路径无其他读者）：表项 = D 行号或 -1
        gmRowIdx.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspaceExpandRowIdxAddr));
        // 被 mask 的 token 不参与计数与打包；只读输入，标量读无 cache 陈旧风险
        hasXActiveMask = (xActiveMaskAddr != nullptr);
        if (hasXActiveMask) {
            gmXActiveMask.SetGlobalBuffer(reinterpret_cast<__gm__ bool*>(xActiveMaskAddr));
        }

        int64_t tokenStartIdx = 0;
        int64_t tokenEndIdx = 0;
        int64_t numTokensCurCore = 0;
        SplitCore(params_.numTokens, tokenStartIdx, tokenEndIdx, numTokensCurCore);

        // 原始序量化 → 本卡 peermem A 区
        QuantizeOriginalOrder(gmX, gmA, tokenStartIdx, tokenEndIdx);

        AscendC::Duplicate<int32_t>(coreExpertCount, 0, params_.alignedNumExperts);
        AscendC::DataCopyExtParams topkCopyParams{
            1, static_cast<uint32_t>(numTokensCurCore * params_.numTopk * sizeof(int32_t)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<int32_t> topkPadParams{false, 0, 0, 0};
        AscendC::DataCopyPad(topkIds, gmTopkIds[tokenStartIdx * params_.numTopk], topkCopyParams, topkPadParams);
        AscendC::PipeBarrier<PIPE_ALL>();
        for (int64_t t = 0; t < numTokensCurCore; ++t) {
            if (hasXActiveMask && !gmXActiveMask.GetValue(tokenStartIdx + t)) {
                continue;
            }
            for (int64_t k = 0; k < params_.numTopk; ++k) {
                int32_t e = topkIds.GetValue(t * params_.numTopk + k);
                if (e >= 0 && e < params_.numExperts) {
                    coreExpertCount.SetValue(e, coreExpertCount.GetValue(e) + 1);
                }
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::GlobalTensor<int32_t> expertCountPerCore;
        // 基准 = workspace 起点；计数写本核槽位，前缀归约读全部核槽位
        expertCountPerCore.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspaceCountsAddr));
        AscendC::DataCopy(expertCountPerCore[coreIdx * params_.alignedNumExperts], coreExpertCount,
                          params_.alignedNumExperts);
        AscendC::SyncAll<true>();

        // 两级前缀和 + TPE 写出
        AscendC::Duplicate<int32_t>(prefixExpertCount, 0, params_.alignedNumExperts);
        AscendC::PipeBarrier<PIPE_V>();
        for (int64_t c = 0; c < coreIdx; ++c) {
            AscendC::DataCopy(scratchCount, expertCountPerCore[c * params_.alignedNumExperts],
                              params_.alignedNumExperts);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::Add(prefixExpertCount, prefixExpertCount, scratchCount, params_.alignedNumExperts);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::DataCopy(totalExpertCount, prefixExpertCount, params_.alignedNumExperts);
        AscendC::PipeBarrier<PIPE_V>();
        for (int64_t c = coreIdx; c < coreNum; ++c) {
            AscendC::DataCopy(scratchCount, expertCountPerCore[c * params_.alignedNumExperts],
                              params_.alignedNumExperts);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::Add(totalExpertCount, totalExpertCount, scratchCount, params_.alignedNumExperts);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        countCopyParams = AscendC::DataCopyExtParams{1, params_.alignedNumExpertsBytes, 0, 0, 0};
        AscendC::DataCopyPad(gmTokenPerExpert, totalExpertCount, countCopyParams);
        // SyncAll 自带全流水同步，此处无需额外的 PipeBarrier
        AscendC::SyncAll<true>();

        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        int32_t globalPrefix = 0;
        for (int64_t e = 0; e < params_.numExperts; ++e) {
            tokenPerExpertOffset.SetValue(e, globalPrefix);
            globalPrefix += totalExpertCount.GetValue(e);
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);

        AscendC::Add(tokenPerExpertOffset, tokenPerExpertOffset, prefixExpertCount, params_.alignedNumExperts);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);

        ScatterAndPackDirect(gmProbs, gmPackSlots, gmPackProbs, tokenStartIdx, tokenEndIdx);
        // MTE3 写回执（跨核可见性由调用方 SyncAll 收口）
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }

private:
    Params params_;
    int64_t coreIdx = 0;
    int64_t coreNum = 0;
    // 常驻区
    AscendC::LocalTensor<int32_t> coreExpertCount; // 本核计数（走查复用）
    AscendC::LocalTensor<int32_t> scratchCount;    // 他核计数 scratch
    AscendC::LocalTensor<int32_t> totalExpertCount;
    AscendC::LocalTensor<int32_t> prefixExpertCount;
    AscendC::LocalTensor<int32_t> tokenPerExpertOffset;
    AscendC::LocalTensor<int32_t> localOff;    // 桶基址（UB run 区布局）
    AscendC::LocalTensor<int32_t> localCursor; // 走查落位游标（初值 = localOff）
    AscendC::LocalTensor<int32_t> topkIds;
    int64_t runBufElems = 0;
    // union 区（量化行流水）
    AscendC::LocalTensor<ElementSrc> tokenInList[UB_STAGES];
    AscendC::LocalTensor<ElementDst> tokenOutList[UB_STAGES];
    AscendC::LocalTensor<float> tmpBuffer;
    // union 区（打包直写）
    AscendC::LocalTensor<int32_t> slotsRun;
    AscendC::LocalTensor<float> probsRun;
    AscendC::LocalTensor<float> probsSlice;
    AscendC::LocalTensor<int32_t> rowIdxRun; // 去重行索引表 staging（token 序）
    AscendC::GlobalTensor<int32_t> gmRowIdx; // workspace expandedRowIdx 段
    AscendC::DataCopyExtParams countCopyParams;
    // hasXActiveMask=false 时 gmXActiveMask 不访问
    bool hasXActiveMask = false;
    AscendC::GlobalTensor<bool> gmXActiveMask;

    CATLASS_DEVICE
    void SplitCore(int64_t numTokens, int64_t& tokenStartIdx, int64_t& tokenEndIdx, int64_t& numTokensCurCore)
    {
        // 连续切分：保证全局 (token, k) 稳定序
        const int64_t baseNum = numTokens / coreNum;
        const int64_t remainder = numTokens % coreNum;
        const bool hasExtra = coreIdx < static_cast<uint32_t>(remainder);
        numTokensCurCore = baseNum + static_cast<int64_t>(hasExtra);
        tokenStartIdx = baseNum * coreIdx + (hasExtra ? static_cast<int64_t>(coreIdx) : remainder);
        tokenEndIdx = tokenStartIdx + numTokensCurCore;
    }

    // 原始序量化：行流水事件链同基线，差异仅在按原始行号写出
    CATLASS_DEVICE
    void QuantizeOriginalOrder(AscendC::GlobalTensor<ElementSrc> const& gmX, AscendC::GlobalTensor<ElementDst>& gmA,
                               int64_t tokenStartIdx, int64_t tokenEndIdx)
    {
        for (uint32_t id = 0; id < UB_STAGES; ++id) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(id);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(id);
        }
        uint32_t tokenListId = 0;
        for (int64_t tokenIdx = tokenStartIdx; tokenIdx < tokenEndIdx; ++tokenIdx) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(tokenListId);
            AscendC::DataCopy(tokenInList[tokenListId], gmX[tokenIdx * params_.hidden], params_.hidden);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(tokenListId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(tokenListId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(tokenListId);
            QuantizeToken(tokenOutList[tokenListId], tmpBuffer, tokenInList[tokenListId],
                          static_cast<uint32_t>(params_.hidden));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(tokenListId);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(tokenListId);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(tokenListId);
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(tokenListId);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(tokenListId);
            // 原始 token 序写 A 区（Dedup 不重排）
            AscendC::DataCopy(gmA[tokenIdx * params_.rowStride], tokenOutList[tokenListId], params_.outLen);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(tokenListId);
            tokenListId = (tokenListId + 1 < UB_STAGES) ? (tokenListId + 1) : 0;
        }
        for (uint32_t id = 0; id < UB_STAGES; ++id) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(id);
        }
    }

    // bf16 -> int8 per-token 量化：scale = 127/max|x|，写行尾；与基线 QuantizeToken
    // 逐行对齐，修改须两边同步
    CATLASS_DEVICE
    void QuantizeToken(AscendC::LocalTensor<ElementDst>& dstTensor, AscendC::LocalTensor<float>& sharedTmpBuffer,
                       AscendC::LocalTensor<ElementSrc>& srcTensor, uint32_t count)
    {
        AscendC::Cast(sharedTmpBuffer, srcTensor, AscendC::RoundMode::CAST_NONE, count);
        AscendC::PipeBarrier<PIPE_V>();
        auto maxSrc = sharedTmpBuffer[count];
        AscendC::Abs(maxSrc, sharedTmpBuffer, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceMax(sharedTmpBuffer[count], sharedTmpBuffer[count], sharedTmpBuffer[count], count);
        AscendC::PipeBarrier<PIPE_V>();
        float maxAbs = sharedTmpBuffer.GetValue(count);
        float dynamicScale = (maxAbs > 0.0f) ? (127.0f / maxAbs) : 127.0f;
        AscendC::Muls(sharedTmpBuffer, sharedTmpBuffer, dynamicScale, count);
        AscendC::PipeBarrier<PIPE_V>();
        auto intTmpBuffer = sharedTmpBuffer.ReinterpretCast<int32_t>();
        auto halfTmpTensor = sharedTmpBuffer.ReinterpretCast<half>();
        AscendC::Cast(intTmpBuffer, sharedTmpBuffer, AscendC::RoundMode::CAST_RINT, count);
        AscendC::SetDeqScale(static_cast<half>(1.0f));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(halfTmpTensor, intTmpBuffer, AscendC::RoundMode::CAST_ROUND, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(dstTensor, halfTmpTensor, AscendC::RoundMode::CAST_TRUNC, count);
        AscendC::PipeBarrier<PIPE_V>();
        dstTensor.template ReinterpretCast<float>().SetValue(count / 4, 1.0f / dynamicScale);
    }

    // 源序走查 + 按专家 run 直写打包段 + 产出去重行索引表：GM 落位与散射式实现逐位相同；
    // 无效/mask 条目不产出，尾部 [Σvalid, n) 哨兵由 core0 补填。rankOrdinal = token 去重
    // 卡序列（topK 展开序首现）中目标卡的序数——D 区 token 块槽位因子；本函数同时把
    // 「首现 → token·K̂ + rankOrdinal / 其余 -1」写入行索引表供 unpermute 使用
    CATLASS_DEVICE
    void ScatterAndPackDirect(AscendC::GlobalTensor<float> const& gmProbs, AscendC::GlobalTensor<int32_t>& gmPackSlots,
                              AscendC::GlobalTensor<float>& gmPackProbs, int64_t tokenStartIdx, int64_t tokenEndIdx)
    {
        const int64_t n = params_.streamLen;
        const int64_t topk = params_.numTopk;
        const int32_t epr = static_cast<int32_t>(params_.expertsPerRank);
        // D 区 token 块行宽 K̂ = min(topK, EP)，行号 = token·K̂ + 卡首现序
        const int64_t ep = params_.numExperts / params_.expertsPerRank;
        const int64_t kHat = (topk < ep) ? topk : ep;
        const int64_t tokensCurCore = tokenEndIdx - tokenStartIdx;
        AscendC::DataCopyExtParams probsCopyParams{1, static_cast<uint32_t>(tokensCurCore * topk * sizeof(float)), 0, 0,
                                                   0};
        AscendC::DataCopyPadExtParams<float> probsPadParams{false, 0, 0, 0};
        AscendC::DataCopyPad(probsSlice, gmProbs[tokenStartIdx * topk], probsCopyParams, probsPadParams);
        AscendC::PipeBarrier<PIPE_ALL>();
        // 桶基址：run DMA 的 UB 源地址须 32B 对齐，故每桶垫 ≤7 元素；GM 布局不受影响
        int32_t acc = 0;
        for (int64_t e = 0; e < params_.numExperts; ++e) {
            localOff.SetValue(e, acc);
            localCursor.SetValue(e, acc);
            acc += coreExpertCount.GetValue(e);
            acc = (acc + 7) & ~7; // 桶基址 8 元素（32B）对齐
        }
        // 源序走查（slots 值 = chunk 内 srcIdx）
        uint32_t rowIdxOffset = 0;
        int64_t topkBase = 0;
        const int32_t srcBase = static_cast<int32_t>(tokenStartIdx * topk);
        // 首现序表：本 token 的去重卡序列（首现序即序数），核内局部数组
        // （去重卡数 ≤ K̂ ≤ 0xFF，host 位宽门保证）
        int32_t cardSeq[256];
        // 各卡首现 k：第一遍建序表时记录，第二遍据此判「首现」
        int32_t firstK[256];
        for (int64_t tokenIdx = tokenStartIdx; tokenIdx < tokenEndIdx;
             ++tokenIdx, rowIdxOffset += topk, topkBase += topk) {
            if (hasXActiveMask && !gmXActiveMask.GetValue(tokenIdx)) {
                // mask token 无 D 行可读：表项全 -1（unpermute 输出 0 行）
                for (int64_t k = 0; k < topk; ++k) {
                    rowIdxRun.SetValue(static_cast<int32_t>(rowIdxOffset + k), -1);
                }
                continue; // mask token 计数阶段已跳过，落位同样跳过
            }
            int32_t distinctCards = 0;
            for (int64_t k2 = 0; k2 < topk; ++k2) {
                const int32_t e2 = topkIds.GetValue(topkBase + k2);
                if (e2 < 0 || e2 >= params_.numExperts) {
                    continue; // 无效条目与卡集合同口径剔除（首现序的卡集合只含有效卡）
                }
                const int32_t c2 = e2 / epr;
                bool dup = false;
                for (int32_t d = 0; d < distinctCards; ++d) {
                    if (cardSeq[d] == c2) {
                        dup = true; // c2 已首现，序列不重复入表
                        break;
                    }
                }
                if (!dup) {
                    firstK[distinctCards] = static_cast<int32_t>(k2);
                    cardSeq[distinctCards++] = c2;
                }
            }
            for (int64_t k = 0; k < topk; ++k) {
                const int32_t expert = topkIds.GetValue(topkBase + k);
                const int32_t tblIdx = static_cast<int32_t>(rowIdxOffset + k);
                if (expert < 0 || expert >= params_.numExperts) {
                    rowIdxRun.SetValue(tblIdx, -1); // 非法 id：与 mask 同语义，永不入段
                    continue;
                }
                // rankOrdinal = card 在本 token 去重卡序列（首现序）中的序数：序表每 token
                // 建一次，行内 O(K̂) 标量查表
                const int32_t card = expert / epr;
                int32_t rankOrdinal = 0;
                while (cardSeq[rankOrdinal] != card) {
                    ++rankOrdinal; // card 必在表内（同一 token 的有效展开），K̂ ≤ 0xFF
                }
                // 行索引表：该卡首现 → D 行号 token·K̂ + rankOrdinal；同卡后续条目 → -1
                // （与 slots 同一 rankOrdinal）
                rowIdxRun.SetValue(tblIdx, (firstK[rankOrdinal] == static_cast<int32_t>(k)) ?
                                               static_cast<int32_t>(tokenIdx * kHat + rankOrdinal) :
                                               -1);
                const int32_t lp = localCursor.GetValue(expert);
                // j 内联 slots 高 8bit（srcIdx ≤ 24bit 由 host 位宽门保证），不占独立平面
                slotsRun.SetValue(lp, (srcBase + static_cast<int32_t>(rowIdxOffset + k)) |
                                          (rankOrdinal << DEDUP_SLOT_RANK_ORDINAL_SHIFT));
                probsRun.SetValue(lp, probsSlice.GetValue(rowIdxOffset + k));
                localCursor.SetValue(expert, lp + 1);
            }
        }
        // S 落盘后每非空桶 2 次 DMA 直写 GM 终位
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        for (int64_t e = 0; e < params_.numExperts; ++e) {
            const int32_t cnt = coreExpertCount.GetValue(e);
            if (cnt == 0) {
                continue;
            }
            const int64_t gmPos = tokenPerExpertOffset.GetValue(e); // = run GM 起点
            const int64_t ubPos = localOff.GetValue(e);
            AscendC::DataCopyPad(gmPackSlots[gmPos], slotsRun[ubPos],
                                 {1, static_cast<uint32_t>(cnt * sizeof(int32_t)), 0, 0, 0});
            AscendC::DataCopyPad(gmPackProbs[gmPos], probsRun[ubPos],
                                 {1, static_cast<uint32_t>(cnt * sizeof(float)), 0, 0, 0});
        }
        // 行索引表落盘（本核 token 区间连续；标量写已由上方 S_MTE3 回执覆盖）
        if (tokensCurCore > 0) {
            AscendC::DataCopyPad(gmRowIdx[tokenStartIdx * topk], rowIdxRun,
                                 {1, static_cast<uint32_t>(tokensCurCore * topk * sizeof(int32_t)), 0, 0, 0});
        }
        // 尾部哨兵 [Σvalid, n)：slots 的 rankOrdinal 段 0xFF（srcIdx 段 0）/ probs=0.0f，
        // core0 用 run 缓冲 staging 补填
        if (coreIdx == 0) {
            int64_t sumValid = 0;
            for (int64_t e = 0; e < params_.numExperts; ++e) {
                sumValid += totalExpertCount.GetValue(e);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            for (int64_t p0 = sumValid; p0 < n; p0 += runBufElems) {
                const int64_t cur = (n - p0 < runBufElems) ? (n - p0) : runBufElems;
                AscendC::Duplicate<int32_t>(slotsRun, DEDUP_SLOT_SENTINEL, static_cast<uint32_t>(cur));
                AscendC::Duplicate<float>(probsRun, 0.0f, static_cast<uint32_t>(cur));
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::DataCopyPad(gmPackSlots[p0], slotsRun,
                                     {1, static_cast<uint32_t>(cur * sizeof(int32_t)), 0, 0, 0});
                AscendC::DataCopyPad(gmPackProbs[p0], probsRun,
                                     {1, static_cast<uint32_t>(cur * sizeof(float)), 0, 0, 0});
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            }
        }
    }
};

} // namespace MegaMoeDedup

#endif // MOE_PERMUTE_PROLOGUE_DEDUP_HPP
