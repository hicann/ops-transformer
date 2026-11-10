/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_EPILOGUE_W4A8_POST_PER_TOKEN_DEDUP_HPP
#define CATLASS_EPILOGUE_BLOCK_EPILOGUE_W4A8_POST_PER_TOKEN_DEDUP_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/layout/layout.hpp"

#include "dispatch_policy_custom.hpp"
#include "get_tensor_addr.hpp"
#include "hccl_shmem.hpp"
#include "const_args.hpp"

namespace Catlass::Epilogue::Block {

// 通信零冗余（Dedup）W4A8 combine 行级 epilogue：锚点槽（2·n₂ 半精度行位宽）原位存 FP32
// 单行部分和（跨行累加语义 = FP32，无 FP16 量程/微值损失），last 行 ×prob 后 BF16 直写
// 源卡 D 区，其余行写回锚点。
template <uint32_t UB_STAGES_, class CType_, class LayoutPerTokenScale_, class DType_, class TileCopy_>
class BlockEpilogue<EpilogueAtlasA2W4A8PostPerTokenDedup<UB_STAGES_>, CType_,
                    Gemm::GemmType<float, LayoutPerTokenScale_>, DType_, TileCopy_> {
public:
    using DispatchPolicy = EpilogueAtlasA2W4A8PostPerTokenDedup<UB_STAGES_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;

    // Data infos
    using ElementC = typename CType_::Element; // gmC2 元素（锚点槽双半行/出值）
    using ElementD = typename DType_::Element; // last 远传元素
    using TTypeC = Gemm::GemmType<ElementC, layout::RowMajor>;
    using TTypeD = Gemm::GemmType<ElementD, layout::RowMajor>;
    using CopyGmToUbC = typename TileCopy_::CopyGmToUbC;
    using CopyUbToGmD = typename TileCopy_::CopyUbToGmD;
    using TTypeAnchor = Gemm::GemmType<float, layout::RowMajor>;
    using CopyGmToUbAnchor = Tile::CopyGm2Ub<ArchTag, TTypeAnchor>;
    using CopyUbToGmAnchor = Tile::CopyUb2Gm<ArchTag, TTypeAnchor>;

    // N 分块上界（512 倍数，保证尾块 32B 对齐）；UB 不足时构造降半
    static constexpr int32_t CHUNK_ELEMS_MAX = 4096;
    // 每元素 UB：CH/CL/出值 2B×3 + anchorF32/bias/fp32/fp32L 4B×4
    static constexpr uint32_t BYTES_PER_ELEM_PER_STAGE = 22;

    struct Params {
        int32_t n2;
        int32_t cardsPerToken;     // D 区 token 块行宽 K̂ = min(topK, EP)
        int32_t layoutChunkTokens; // D 区块步长因子（min(m, 1024)，尾 chunk 不改步长）
        __gm__ float* ptrBias2;
        int32_t listLen;
        HcclShmem<false> shmem;
        int64_t offsetD;
        uint32_t ubBudgetLimit{0}; // routeCacheUbOffset（UB 预算入参）
        CATLASS_DEVICE
        Params() {}
        CATLASS_DEVICE
        Params(int32_t n2_, int32_t cardsPerToken_, int32_t layoutChunkTokens_, __gm__ float* ptrBias2_,
               int32_t listLen_, HcclShmem<false>& shmem_, int64_t offsetD_, uint32_t ubBudgetLimit_)
            : n2(n2_),
              cardsPerToken(cardsPerToken_),
              layoutChunkTokens(layoutChunkTokens_),
              ptrBias2(ptrBias2_),
              listLen(listLen_),
              shmem(shmem_),
              offsetD(offsetD_),
              ubBudgetLimit(ubBudgetLimit_)
        {}
    };

    // kernel 侧读取：块长与 scratch 占用
    int32_t chunkElems;
    uint32_t scratchBytes;

    // 行上下文，供 IssueRead/ComputeWrite 共用
    struct RowCtx {
        int64_t curHiOff;     // 当前行双行高半行元素偏移（低半行 +n₂ 派生）
        int64_t anchorF32Off; // 锚点 FP32 行元素偏移（float 计；槽 = 2·n₂ 半精度行位宽）
        bool needAnchor;      // anchorRow != tokenId（隐式 INIT 判别）
        bool isLast;          // last 行合并 ACCUM+SEND
        int32_t dstRank;      // SEND 目标 rank（cot / layoutChunkTokens）
        int32_t tokenInChunk;
        int32_t dstRankOrdinal; // 目标卡在 token 去重卡序列（topK 展开序首现）中的序数
        float scale;
        float prob;
    };

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> const& resource, Params const& params_ = Params{})
        : params_(params_)
    {
        // UB 预算不足时降档：预算须同时容纳 scratch 与 combine 切片流缓冲（正常域不触发）
        chunkElems = CHUNK_ELEMS_MAX;
        scratchBytes = UB_STAGES * static_cast<uint32_t>(chunkElems) * BYTES_PER_ELEM_PER_STAGE;
        if (params_.ubBudgetLimit < scratchBytes + DEDUP_COMBINE_SLICE_BYTES) {
            chunkElems = CHUNK_ELEMS_MAX / 2;
            scratchBytes = UB_STAGES * static_cast<uint32_t>(chunkElems) * BYTES_PER_ELEM_PER_STAGE;
        }
        // 每集 3 个事件，集间轮转
        for (uint32_t s = 0; s < UB_STAGES; ++s) {
            eventMTE2VList[s] = static_cast<int32_t>(3 * s);
            eventVMTE3List[s] = static_cast<int32_t>(3 * s + 1);
            eventMTE3M2List[s] = static_cast<int32_t>(3 * s + 2);
        }
        // CH/CL 各 2·chunkElems（[0,chunkElems) 为 hi 半区，其后 lo 半区）；锚点 FP32 单行
        size_t ubOff = 0;
        for (uint32_t s = 0; s < UB_STAGES; ++s) {
            ubCurList[s] = resource.ubBuf.template GetBufferByByte<ElementC>(ubOff);
            ubOff += static_cast<size_t>(chunkElems) * 2 * sizeof(ElementC);
            ubAnchorList[s] = resource.ubBuf.template GetBufferByByte<float>(ubOff);
            ubOff += static_cast<size_t>(chunkElems) * sizeof(float);
            ubOutDList[s] = resource.ubBuf.template GetBufferByByte<ElementD>(ubOff);
            ubOff += static_cast<size_t>(chunkElems) * sizeof(ElementD);
            ubBiasList[s] = resource.ubBuf.template GetBufferByByte<float>(ubOff);
            ubOff += static_cast<size_t>(chunkElems) * sizeof(float);
            ubFp32List[s] = resource.ubBuf.template GetBufferByByte<float>(ubOff);
            ubOff += static_cast<size_t>(chunkElems) * sizeof(float);
            ubFp32LList[s] = resource.ubBuf.template GetBufferByByte<float>(ubOff);
            ubOff += static_cast<size_t>(chunkElems) * sizeof(float);
            pendingList[s] = false;
        }
        issueSeq_ = 0;
        computeSeq_ = 0;
    }

    CATLASS_DEVICE
    ~BlockEpilogue() {}

    // 组级 bias 装载：须在组尾 SyncAll<true> 之后入队（提前入队时 MTE2 覆写与本组末行
    // V 读构成 WAR）；listLen==EPR 时不加组偏移（g≥1 越界）。
    CATLASS_DEVICE
    void IssueGroupBiasLoad(int32_t groupIdx)
    {
        const int32_t arrayGroupIdx = params_.listLen == 1 ? 0 : groupIdx;
        const int64_t biasBaseOff = params_.listLen == 1 ? static_cast<int64_t>(groupIdx) * params_.n2 : 0;
        AscendC::GlobalTensor<float> weightAux;
        weightAux.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(
            GetTensorAddr<float>(arrayGroupIdx, reinterpret_cast<GM_ADDR>(params_.ptrBias2))));
        int32_t chunkIdx = 0;
        for (int64_t col = 0; col < params_.n2; col += chunkElems, ++chunkIdx) {
            const int32_t len = static_cast<int32_t>(params_.n2 - col) < chunkElems ?
                                    static_cast<int32_t>(params_.n2 - col) :
                                    chunkElems;
            DataCopyExtParams biasParams{1, static_cast<uint32_t>(len * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> biasPad{false, 0, 0, 0};
            AscendC::DataCopyPad(ubBiasList[chunkIdx & 1], weightAux[biasBaseOff + col], biasParams, biasPad);
        }
    }

    // 行读发射：缓冲集按发射序轮转，集复用回执在块头延迟取走。勿改为「读先行
    // 一拍」形态（实测非确定竞态）。
    CATLASS_DEVICE
    void IssueRead(AscendC::GlobalTensor<ElementC>& gmC2, AscendC::GlobalTensor<float>& gmAnchor, RowCtx const& r)
    {
        const int32_t n2 = params_.n2;
        const bool merged = (n2 <= chunkElems); // 双半行物理连续 → 合并搬运
        for (int64_t col = 0; col < n2; col += chunkElems) {
            const int32_t buf = static_cast<int32_t>(issueSeq_ & 1U);
            ++issueSeq_;
            const int32_t evM2V = eventMTE2VList[buf];
            // 回执延迟取走（条件化防死等）
            if (pendingList[buf]) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventMTE3M2List[buf]);
                pendingList[buf] = false;
            }
            const int32_t len =
                static_cast<int32_t>(n2 - col) < chunkElems ? static_cast<int32_t>(n2 - col) : chunkElems;
            // lo 半区基址：merged 时整块布局 lo 落在 n2，多 chunk 时半区隔 chunkElems
            const int64_t loOff = merged ? n2 : chunkElems;
            layout::RowMajor chunkLayout{1, static_cast<uint32_t>(len)};
            layout::RowMajor mergedLayout{1, static_cast<uint32_t>(2 * len)};
            if (merged) {
                copyGmToUbC_(ubCurList[buf], gmC2[r.curHiOff], mergedLayout, mergedLayout);
            } else {
                // 多 chunk 逐块基址 staging（hi→[0,len)，lo→[loOff,loOff+len)），不越界
                copyGmToUbC_(ubCurList[buf], gmC2[r.curHiOff + col], chunkLayout, chunkLayout);
                copyGmToUbC_(ubCurList[buf][static_cast<uint32_t>(loOff)], gmC2[r.curHiOff + n2 + col], chunkLayout,
                             chunkLayout);
            }
            if (r.needAnchor) {
                copyGmToUbAnchor_(ubAnchorList[buf], gmAnchor[r.anchorF32Off + col], chunkLayout, chunkLayout);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evM2V);
        }
    }

    // 行计算+写发射：行读完成门 → 逐列块 V 链 → V_MTE3 握手 → MTE3 写 → 回执置位
    CATLASS_DEVICE
    void ComputeWrite(AscendC::GlobalTensor<float>& gmAnchor, RowCtx const& r)
    {
        const int32_t n2 = params_.n2;
        // D 区 token 块：行 = tokenInChunk·cardsPerToken + dstRankOrdinal（块内槽位 = 目标卡
        // 首现序；行字节 = n₂·2B 为 128 倍数时任意整行槽位天然对齐）
        const int64_t dstOff =
            (static_cast<int64_t>(r.tokenInChunk) * params_.cardsPerToken + static_cast<int64_t>(r.dstRankOrdinal)) *
            n2;
        const float scaleProb = r.scale * r.prob;
        const bool merged = (n2 <= chunkElems); // lo 半区基址选择（同 IssueRead）
        // 行读完成门：FP32 锚点行与本源行 FP16 双半行同槽，列块 c 的锚点写会覆盖后续
        // 列块的 hi 半行读区；MTE2 读与随后 MTE3 写分属不同流水、无顺序保证。故先消费
        // 本行全部列块的 MTE2_V 完成回执（DMA 完成）再发首个锚点写；列块数受双缓冲
        // 约束 ≤2（事件 ID 按块奇偶不重复置位），单列块形状直通。
        const int32_t chunkCount = static_cast<int32_t>((n2 + chunkElems - 1) / chunkElems);
        const uint32_t bufParity = computeSeq_ & 1U;
        for (int32_t c = 0; c < chunkCount; ++c) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventMTE2VList[(bufParity + static_cast<uint32_t>(c)) & 1U]);
        }
        for (int64_t col = 0; col < n2; col += chunkElems) {
            const int32_t buf = static_cast<int32_t>(computeSeq_ & 1U);
            ++computeSeq_;
            const int32_t evVM3 = eventVMTE3List[buf];
            const int32_t len =
                static_cast<int32_t>(n2 - col) < chunkElems ? static_cast<int32_t>(n2 - col) : chunkElems;
            const int64_t loOff = merged ? n2 : chunkElems; // lo 半区基址（同 IssueRead）
            layout::RowMajor chunkLayout{1, static_cast<uint32_t>(len)};
            // bias 按列块奇偶索引（与 IssueGroupBiasLoad 同键）
            const int32_t biasBuf = static_cast<int32_t>((col / chunkElems) & 1);

            AscendC::PipeBarrier<PIPE_V>();
            // V：反量化（lo×16 + hi）+ bias + ×(scale·prob)
            AscendC::Cast(ubFp32List[buf], ubCurList[buf][static_cast<uint32_t>(loOff)], AscendC::RoundMode::CAST_NONE,
                          len);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(ubFp32LList[buf], ubCurList[buf], AscendC::RoundMode::CAST_NONE, len);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Axpy(ubFp32List[buf], ubFp32LList[buf], 16.0f, len);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(ubFp32List[buf], ubFp32List[buf], ubBiasList[biasBuf], len);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(ubFp32List[buf], ubFp32List[buf], scaleProb, len);
            AscendC::PipeBarrier<PIPE_V>();

            // 锚点合并（隐式 INIT 判别）：acc += 锚点 FP32 行
            if (r.needAnchor) {
                AscendC::Add(ubFp32List[buf], ubFp32List[buf], ubAnchorList[buf], len);
                AscendC::PipeBarrier<PIPE_V>();
            }

            // 出值：last 行 bf16（CAST_RINT）直写源卡 D 区，其余行 FP32 直写回锚点
            if (r.isLast) {
                AscendC::Cast(ubOutDList[buf], ubFp32List[buf], AscendC::RoundMode::CAST_RINT, len);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evVM3);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evVM3);
                __gm__ void* dstPeermemPtr = params_.shmem(params_.offsetD, r.dstRank);
                AscendC::GlobalTensor<ElementD> gmRemote;
                gmRemote.SetGlobalBuffer(reinterpret_cast<__gm__ ElementD*>(dstPeermemPtr));
                copyUbToGmD_(gmRemote[dstOff + col], ubOutDList[buf], chunkLayout, chunkLayout);
            } else {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evVM3);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evVM3);
                copyUbToGmAnchor_(gmAnchor[r.anchorF32Off + col], ubFp32List[buf], chunkLayout, chunkLayout);
            }
            // 写侧回执：下一同集 IssueRead 块头或组尾 Drain 延迟消费，Set/Wait 一对一配对
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventMTE3M2List[buf]);
            pendingList[buf] = true;
        }
    }

    // 组尾收口：取走 pending 回执；锚点写退休由调用方紧随的 SyncAll<true> 保证
    CATLASS_DEVICE
    void Drain()
    {
        for (uint32_t s = 0; s < UB_STAGES; ++s) {
            if (pendingList[s]) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventMTE3M2List[s]);
                pendingList[s] = false;
            }
        }
    }

    CATLASS_DEVICE
    void Finalize()
    {
        Drain();
    }

private:
    Params params_;
    // ubCur 2·chunkElems（[0,chunkElems) 为 hi 半区，其后 lo 半区）；ubAnchor 为 FP32 单行
    AscendC::LocalTensor<ElementC> ubCurList[UB_STAGES];
    AscendC::LocalTensor<float> ubAnchorList[UB_STAGES];
    AscendC::LocalTensor<ElementD> ubOutDList[UB_STAGES];
    AscendC::LocalTensor<float> ubBiasList[UB_STAGES];
    AscendC::LocalTensor<float> ubFp32List[UB_STAGES];
    AscendC::LocalTensor<float> ubFp32LList[UB_STAGES];
    bool pendingList[UB_STAGES];
    // IssueRead/ComputeWrite 各自的列块发射序计数（锁步推进）
    uint32_t issueSeq_;
    uint32_t computeSeq_;
    int32_t eventMTE2VList[UB_STAGES];
    int32_t eventVMTE3List[UB_STAGES];
    int32_t eventMTE3M2List[UB_STAGES];
    CopyGmToUbC copyGmToUbC_;
    CopyUbToGmD copyUbToGmD_;
    CopyGmToUbAnchor copyGmToUbAnchor_;
    CopyUbToGmAnchor copyUbToGmAnchor_;
};

} // namespace Catlass::Epilogue::Block

#endif // CATLASS_EPILOGUE_BLOCK_EPILOGUE_W4A8_POST_PER_TOKEN_DEDUP_HPP
