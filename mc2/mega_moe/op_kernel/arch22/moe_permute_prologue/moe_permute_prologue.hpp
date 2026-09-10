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
 * \file moe_permute_prologue.hpp
 * \brief 发送侧 chunk Permute Prologue
 *
 * 按专家密排重排 (permute) token，同时输出 expandRowIdx / tokenPerExpert
 *
 */

#ifndef MC2_MOE_PERMUTE_PROLOGUE_HPP
#define MC2_MOE_PERMUTE_PROLOGUE_HPP

#include "kernel_operator.h"
#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"

namespace MoePermute {

enum class MoeQuantMode : uint32_t {
    kNone = 0,
    kPertoken = 1,
};

struct MoePermuteProloguePolicy {
    static constexpr MoeQuantMode quantmode = MoeQuantMode::kPertoken;
    using ArchTag = Catlass::Arch::AtlasA2;
};

struct MoePermutePrologueNonQuantPolicy {
    static constexpr MoeQuantMode quantmode = MoeQuantMode::kNone;
    using ArchTag = Catlass::Arch::AtlasA2;
};

template <uint32_t UB_STAGES_, class DispatchPolicy_, class ElementSrc_, class ElementDst_>
class MoePermutePrologue {
public:
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
    using DispatchPolicy = DispatchPolicy_;
    using ArchTag = typename DispatchPolicy_::ArchTag;
    using ElementSrc = ElementSrc_;
    using ElementDst = ElementDst_;
    static constexpr MoeQuantMode QUANT_MODE = DispatchPolicy_::quantmode;

    static constexpr uint32_t MAX_TOKENS = 1024;
    static constexpr uint32_t MAX_TOPK = 32;
    static constexpr uint32_t MAX_HIDDEN = 10240;
    static constexpr uint32_t MAX_EXPERTS = 1024;

    struct Params {
        int64_t numTokens;
        int64_t hidden;
        int64_t numTopk;
        int64_t numExperts;
        int64_t alignedNumExperts;
        uint32_t alignedNumExpertsBytes;
        // expandedX 行步长（元素）：量化（int8）行尾含 per-token scale，取 hidden+512；非量化取 hidden
        uint32_t rowStride;

        CATLASS_DEVICE
        Params(int64_t numTokens_, int64_t hidden_, int64_t numTopk_, int64_t numExperts_)
            : numTokens(numTokens_),
              hidden(hidden_),
              numTopk(numTopk_),
              numExperts(numExperts_),
              alignedNumExperts(RoundUp<8>(numExperts_)),
              alignedNumExpertsBytes(static_cast<uint32_t>(alignedNumExperts * sizeof(int32_t))),
              rowStride(CalcRowStride(hidden_))
        {}

    private:
        // 行步长内部固定：量化（int8）场景 expandedX 行尾含 per-token scale，取 hidden+512；非量化取 hidden
        CATLASS_DEVICE
        static constexpr uint32_t CalcRowStride(int64_t hidden_)
        {
            if constexpr (std::is_same_v<ElementDst, int8_t>) {
                return static_cast<uint32_t>(hidden_) + 512;
            } else {
                return static_cast<uint32_t>(hidden_);
            }
        }
    };

    CATLASS_DEVICE
    MoePermutePrologue(Catlass::Arch::Resource<ArchTag> &resource, Params const &params)
        : params_(params)
    {
        // 与 A3 核函数(mega_moe_kernel_a3.hpp)一致的逻辑核索引：
        // GetBlockIdx()/GetBlockNum() 在 A3 上返回的是物理核信息，需结合 subblock 展开为逻辑核
        if ASCEND_IS_AIV {
            coreIdx = get_block_idx() + get_subblockid() * get_block_num();
            coreNum = get_block_num() * get_subblockdim();
        } else {
            coreIdx = AscendC::GetBlockIdx();
            coreNum = AscendC::GetBlockNum();
        }
        const int64_t numExperts = params_.numExperts;
        const int64_t hidden = params_.hidden;
        const int64_t numTopk = params_.numTopk;

        int64_t tokenStartIdx = 0;
        int64_t tokenEndIdx = 0;
        int64_t numTokensCurCore = 0;
        SplitCore(MAX_TOKENS, tokenStartIdx, tokenEndIdx, numTokensCurCore);

        uint32_t ubOffset = 0;
        coreExpertCount = resource.ubBuf.template GetBufferByByte<int32_t>(ubOffset);
        ubOffset += params_.alignedNumExpertsBytes;
        totalExpertCount = resource.ubBuf.template GetBufferByByte<int32_t>(ubOffset);
        ubOffset += params_.alignedNumExpertsBytes;
        prefixExpertCount = resource.ubBuf.template GetBufferByByte<int32_t>(ubOffset);
        ubOffset += params_.alignedNumExpertsBytes;
        tokenPerExpertOffset = resource.ubBuf.template GetBufferByByte<int32_t>(ubOffset);
        ubOffset += params_.alignedNumExpertsBytes;

        topkIds = resource.ubBuf.template GetBufferByByte<int32_t>(ubOffset);
        expandRowIdx = resource.ubBuf.template GetBufferByByte<int32_t>(ubOffset);
        ubOffset += RoundUp<Catlass::BYTE_PER_BLK>(static_cast<uint32_t>(numTokensCurCore * numTopk * sizeof(int32_t)));
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            if constexpr (std::is_same_v<ElementDst, int8_t>) {
                tokenInList[i] = resource.ubBuf.template GetBufferByByte<ElementSrc>(ubOffset);
                ubOffset += RoundUp<Catlass::BYTE_PER_BLK>(static_cast<uint32_t>(hidden * sizeof(ElementSrc)));
                tokenOutList[i] = resource.ubBuf.template GetBufferByByte<ElementDst>(ubOffset);
                ubOffset += RoundUp<Catlass::BYTE_PER_BLK>(static_cast<uint32_t>(hidden * sizeof(ElementDst))) +
                            Catlass::BYTE_PER_BLK;
            } else {
                tokenInList[i] = resource.ubBuf.template GetBufferByByte<ElementSrc>(ubOffset);
                tokenOutList[i] = resource.ubBuf.template GetBufferByByte<ElementDst>(ubOffset);
                ubOffset += RoundUp<Catlass::BYTE_PER_BLK>(static_cast<uint32_t>(hidden * sizeof(ElementDst)));
            }
        }
        if constexpr (std::is_same_v<ElementDst, int8_t>) {
            // 量化临时缓冲：work [0,h) + abs/max 复用 [h,2h)；int32/half 复用 work，取 2*hidden 个 float
            tmpBuffer = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
            ubOffset += RoundUp<Catlass::BYTE_PER_BLK>(static_cast<uint32_t>(2 * hidden * sizeof(float)));
        }
    }

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementSrc> const &gmX, AscendC::GlobalTensor<int32_t> const &gmTopkIds,
                    AscendC::GlobalTensor<ElementDst> &gmExpandX, AscendC::GlobalTensor<int32_t> &gmExpandRowIdx,
                    AscendC::GlobalTensor<int32_t> &gmTokenPerExpert, GM_ADDR workspaceAddr,
                    GM_ADDR xActiveMaskAddr = nullptr)
    {
        const int64_t numTokens = params_.numTokens;
        const int64_t hidden = params_.hidden;
        const int64_t numTopk = params_.numTopk;
        const int64_t numExperts = params_.numExperts;
        // x_active_mask（shape [bs]，调用方已偏移到 chunk 基址）：被 mask 的 token 不参与
        // 计数与散射，其 expandRowIdx 全条目写 -1，由 unpermute 按无效索引跳过（输出 0）。
        // mask 为只读输入张量，执行期间不变化，标量读无 cache 陈旧风险。
        hasXActiveMask = (xActiveMaskAddr != nullptr);
        if (hasXActiveMask) {
            gmXActiveMask.SetGlobalBuffer(reinterpret_cast<__gm__ bool *>(xActiveMaskAddr));
        }
        // 与构造器一致：A3 上需用 subblock 展开计算逻辑核索引/总数
        if ASCEND_IS_AIV {
            coreIdx = get_block_idx() + get_subblockid() * get_block_num();
            coreNum = get_block_num() * get_subblockdim();
        } else {
            coreIdx = AscendC::GetBlockIdx();
            coreNum = AscendC::GetBlockNum();
        }

        int64_t tokenStartIdx = 0;
        int64_t tokenEndIdx = 0;
        int64_t numTokensCurCore = 0;
        SplitCore(numTokens, tokenStartIdx, tokenEndIdx, numTokensCurCore);

        AscendC::Duplicate<int32_t>(coreExpertCount, 0, params_.alignedNumExperts);
        AscendC::DataCopyExtParams topkCopyParams{
            1, static_cast<uint32_t>(numTokensCurCore * numTopk * sizeof(int32_t)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<int32_t> topkPadParams{false, 0, 0, 0};
        AscendC::DataCopyPad(topkIds, gmTopkIds[tokenStartIdx * numTopk], topkCopyParams, topkPadParams);
        AscendC::PipeBarrier<PIPE_ALL>();
        for (int64_t t = 0; t < numTokensCurCore; ++t) {
            if (hasXActiveMask && !gmXActiveMask.GetValue(tokenStartIdx + t)) {
                continue;
            }
            for (int64_t k = 0; k < numTopk; ++k) {
                int32_t e = topkIds.GetValue(t * numTopk + k);
                if (e >= 0 && e < numExperts) {
                    coreExpertCount.SetValue(e, coreExpertCount.GetValue(e) + 1);
                }
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::GlobalTensor<int32_t> expertCountPerCore;
        // 基准为 workspace 起点；Phase1 写本核槽位 coreIdx*alignedE，Phase2 读全部核槽位 c*alignedE
        expertCountPerCore.SetGlobalBuffer((__gm__ int32_t *)(workspaceAddr));
        AscendC::DataCopy(expertCountPerCore[coreIdx * params_.alignedNumExperts], coreExpertCount,
                          params_.alignedNumExperts);
        AscendC::SyncAll<true>();

        AscendC::Duplicate<int32_t>(prefixExpertCount, 0, params_.alignedNumExperts);
        AscendC::PipeBarrier<PIPE_V>();
        for (int64_t c = 0; c < coreIdx; ++c) {
            AscendC::DataCopy(coreExpertCount, expertCountPerCore[c * params_.alignedNumExperts],
                              params_.alignedNumExperts);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::Add(prefixExpertCount, prefixExpertCount, coreExpertCount, params_.alignedNumExperts);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::DataCopy(totalExpertCount, prefixExpertCount, params_.alignedNumExperts);
        AscendC::PipeBarrier<PIPE_V>();
        for (int64_t c = coreIdx; c < coreNum; ++c) {
            AscendC::DataCopy(coreExpertCount, expertCountPerCore[c * params_.alignedNumExperts],
                              params_.alignedNumExperts);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::Add(totalExpertCount, totalExpertCount, coreExpertCount, params_.alignedNumExperts);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        // 全数据缓存 clean+invalidate 已统一放到调用侧（mega_moe_kernel_a3.hpp，prologue
        // 调用后）：确保本核此前 UB→GM 的写（tokenPerExpert 等）对下游
        // （allgather/cumsum）及对端 dispatch 立即可见，规避跨 chunk 复用 workspace 时的
        // cache 陈旧读竞态。
        countCopyParams = AscendC::DataCopyExtParams{1, params_.alignedNumExpertsBytes, 0, 0, 0};
        AscendC::DataCopyPad(gmTokenPerExpert, totalExpertCount, countCopyParams);
        // SyncAll 自带全流水同步，此处无需额外的 PipeBarrier
        AscendC::SyncAll<true>();

        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        int32_t globalPrefix = 0;
        for (int64_t e = 0; e < numExperts; ++e) {
            tokenPerExpertOffset.SetValue(e, globalPrefix);
            globalPrefix += totalExpertCount.GetValue(e);
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);

        AscendC::Add(tokenPerExpertOffset, tokenPerExpertOffset, prefixExpertCount, params_.alignedNumExperts);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        if constexpr (std::is_same_v<ElementDst, bfloat16_t>) {
            PermuteTokens<false>(tokenStartIdx, tokenEndIdx, gmX, gmExpandX);
        } else {
            PermuteTokens<true>(tokenStartIdx, tokenEndIdx, gmX, gmExpandX);
        }
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::DataCopyPad(gmExpandRowIdx[tokenStartIdx * numTopk], expandRowIdx, topkCopyParams);
        AscendC::SyncAll<true>();
    }

private:
    Params params_;
    int64_t coreIdx = 0;
    int64_t coreNum = 0;
    AscendC::LocalTensor<int32_t> totalExpertCount;
    AscendC::LocalTensor<int32_t> prefixExpertCount;
    AscendC::LocalTensor<int32_t> coreExpertCount;
    AscendC::LocalTensor<int32_t> tokenPerExpertOffset;
    AscendC::LocalTensor<int32_t> topkIds;
    AscendC::LocalTensor<int32_t> expandRowIdx;
    AscendC::LocalTensor<ElementSrc> tokenInList[UB_STAGES];
    AscendC::LocalTensor<ElementDst> tokenOutList[UB_STAGES];
    AscendC::LocalTensor<float> tmpBuffer;
    AscendC::DataCopyExtParams countCopyParams;
    // x_active_mask（只读输入），hasXActiveMask=false 时 gmXActiveMask 不访问
    bool hasXActiveMask = false;
    AscendC::GlobalTensor<bool> gmXActiveMask;

    CATLASS_DEVICE
    void SplitCore(int64_t numTokens, int64_t &tokenStartIdx, int64_t &tokenEndIdx, int64_t &numTokensCurCore)
    {
        const int64_t baseNum = numTokens / coreNum;
        const int64_t remainder = numTokens % coreNum;
        const bool hasExtra = coreIdx < static_cast<uint32_t>(remainder);
        numTokensCurCore = baseNum + static_cast<int64_t>(hasExtra);
        tokenStartIdx = baseNum * coreIdx + (hasExtra ? static_cast<int64_t>(coreIdx) : remainder);
        tokenEndIdx = tokenStartIdx + numTokensCurCore;
    }

    template <bool QUANT>
    CATLASS_DEVICE void PermuteTokens(int64_t tokenStartIdx, int64_t tokenEndIdx,
                                      AscendC::GlobalTensor<ElementSrc> const &gmX,
                                      AscendC::GlobalTensor<ElementDst> &gmExpandX)
    {
        const int64_t hidden = params_.hidden;
        const int64_t numTopk = params_.numTopk;
        // expandedX 行步长（元素）：quant=hidden+512（行尾含 scale），非quant=hidden
        const int64_t rowStride = params_.rowStride;
        // 单行实际拷贝元素数：quant 需要把行尾 scale 一并拷出
        const int64_t outLen = hidden + (QUANT ? static_cast<int64_t>(Catlass::BYTE_PER_BLK) : 0);

        for (uint32_t id = 0; id < UB_STAGES; ++id) {
            if constexpr (QUANT) {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(id);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(id);
            } else {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(id);
            }
        }
        uint32_t tokenListId = 0;
        uint32_t rowIdxOffset = 0;
        int64_t topkBase = 0; // 本 token 的 topk 在本核切片内的偏移
        for (int64_t tokenIdx = tokenStartIdx; tokenIdx < tokenEndIdx;
             tokenIdx++, rowIdxOffset += numTopk, topkBase += numTopk) {
            if constexpr (QUANT) {
                // 等上一轮 V 读完 tokenIn 后才覆盖 tokenIn
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(tokenListId);
                AscendC::DataCopy(tokenInList[tokenListId], gmX[tokenIdx * hidden], hidden); // MTE2 加载
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(tokenListId);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(tokenListId); // V 等加载完成
                // 等上一轮 MTE3 读完 tokenOut 后才覆盖 tokenOut
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(tokenListId);
                QuantizeToken(tokenOutList[tokenListId], tmpBuffer, tokenInList[tokenListId], hidden); // V/S 量化
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(tokenListId); // V 读完 tokenIn，tokenIn 可复用
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(tokenListId);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(tokenListId); // MTE3 等量化 V 落盘
                AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(tokenListId);
                AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(tokenListId); // MTE3 等 scale S 落盘
                ScatterToken(tokenOutList[tokenListId], gmExpandX, rowIdxOffset, rowStride, outLen, topkBase,
                             tokenIdx);                                    // MTE3 散写
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(tokenListId); // MTE3 读完 tokenOut，tokenOut 可复用
            } else {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(tokenListId);
                AscendC::DataCopy(tokenInList[tokenListId], gmX[tokenIdx * hidden], hidden); // MTE2 加载
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(tokenListId);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(tokenListId); // MTE3 等加载完成
                ScatterToken(tokenInList[tokenListId], gmExpandX, rowIdxOffset, rowStride, outLen, topkBase,
                             tokenIdx);                                       // MTE3 散写
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(tokenListId); // 散写完成，可复用
            }
            tokenListId = (tokenListId + 1 < UB_STAGES) ? (tokenListId + 1) : 0;
        }
        for (uint32_t id = 0; id < UB_STAGES; ++id) {
            if constexpr (QUANT) {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(id);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(id);
            } else {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(id);
            }
        }
    }

    CATLASS_DEVICE
    void ScatterToken(AscendC::LocalTensor<ElementDst> const &tokenOut, AscendC::GlobalTensor<ElementDst> &gmExpandX,
                      uint32_t rowIdxOffset, int64_t rowStride, int64_t copyLen, int64_t topkBase, int64_t tokenIdx)
    {
        // x_active_mask 融合（不改写输入 topk_ids）：被 mask 的 token 计数阶段已跳过
        // （tokenPerExpertOffset 不含其配额），此处必须同样跳过散射——否则 DataCopy 会
        // 写超出行配额踩踏后续 expert 数据区。expandRowIdx 全条目写 -1，
        // 由 unpermute 按 invalid（<0）索引跳过并输出 0 行。
        if (hasXActiveMask && !gmXActiveMask.GetValue(tokenIdx)) {
            for (int64_t k = 0; k < params_.numTopk; k++) {
                expandRowIdx.SetValue(rowIdxOffset + k, -1);
            }
            return;
        }
        for (int64_t k = 0; k < params_.numTopk; k++) {
            int32_t expert = topkIds.GetValue(topkBase + k);
            if (expert < 0 || expert >= params_.numExperts) {
                expandRowIdx.SetValue(rowIdxOffset + k, -1);
                continue;
            }
            int32_t dstPos = tokenPerExpertOffset.GetValue(expert);
            AscendC::DataCopy(gmExpandX[dstPos * rowStride], tokenOut, copyLen);
            expandRowIdx.SetValue(rowIdxOffset + k, dstPos);
            tokenPerExpertOffset.SetValue(expert, dstPos + 1);
        }
    }

    // bf16 -> int8 per-token 量化：scale = max|src| / 127，量化值 = round(src / scale)。
    // 结果写 dstTensor[0, count)；per-token scale（float）写 dstTensor 行尾 [count, count+4) 字节。
    // sharedTmpBuffer(float) 内存复用（参考 MoeDistributeDispatchA2::QuantProcess）：
    //   work [0,count)；abs 与 ReduceMax 的 max 复用 [count,2*count)；int32/half 复用 work（前类型字节数 >=
    //   后类型即可复用）
    CATLASS_DEVICE
    void QuantizeToken(AscendC::LocalTensor<ElementDst> &dstTensor, AscendC::LocalTensor<float> &sharedTmpBuffer,
                       AscendC::LocalTensor<ElementSrc> &srcTensor, uint32_t count)
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
};

} // namespace MoePermute

#endif // MC2_MOE_PERMUTE_PROLOGUE_HPP
