/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_KERNEL_A2_BF16_W4A8_HPP
#define MEGA_MOE_KERNEL_A2_BF16_W4A8_HPP
#include <limits>
#include "kernel_operator.h"
// A2 跨 server：flag 等待按经过时间限界，需要 GetSystemCycle()
#include "basic_api/kernel_operator_sys_var_intf.h"

#include "utils/copy_l0c_to_gm_custom.hpp"

#include "catlass/catlass.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/detail/callback.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"

#include "utils/block_mmad_w4a4.hpp"
#include "utils/copy_gm_to_l1_custom.hpp"
#include "utils/block_epilogue_w4a8post_pertoken_v2.hpp"
#include "utils/block_epilogue_w4a8post_pertoken_swiglu.hpp"
#include "utils/hccl_shmem.hpp"
#include "utils/routing_count_utils.hpp"
#include "a2_a3_copy_helpers.hpp"
#include "a2_routing_helpers.hpp"
#include "utils/const_args.hpp"
#include "utils/block_epilogue_pertoken_swiglu.hpp"
#include "utils/block_epilogue_w4a8post_pertoken_swiglu.hpp"
#include "utils/block_epilogue_w4a8post_pertoken_v2.hpp"
#include "mega_moe_tiling_a2a3.h"
#include "utils/layout3d.hpp"

#include "moe_permute_prologue/moe_permute_prologue.hpp"
#include "unpermute/mc2_mega_moe_moe_token_unpermute.h"
#include "utils/get_tensor_addr.hpp"
#include "mega_moe_exception_dump_policy.h"

namespace Catlass::Gemm::Kernel {

using namespace AscendC;
using namespace Mc2Tiling;

template <class BlockMmad_, class BlockScheduler_, class ElementGroupList_, class BlockEpilogue1_,
          class BlockEpilogue2_, bool BatchedDispatch = false>
class MegaMoeKernelA2Bf16W4A8 {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;
    using ElementScale = uint64_t;
    using LayoutScale = typename layout::VectorLayout;
    using ElementPerTokenScale = float;
    using LayoutPerTokenScale = typename layout::VectorLayout;
    using BlockScheduler = BlockScheduler_;
    using BlockEpilogue1 = BlockEpilogue1_;
    using BlockEpilogue2 = BlockEpilogue2_;
    using ElementD1 = typename BlockEpilogue1::ElementD;
    using LayoutD1 = typename BlockEpilogue1::LayoutD;
    using ElementD2 = typename BlockEpilogue2::ElementD;
    using LayoutD2 = typename BlockEpilogue2::LayoutD;
    using ElementABefore = std::conditional_t<std::is_same_v<ElementA, AscendC::int4b_t>, int8_t, ElementA>;

    static constexpr uint32_t DISPATCH_BATCH_ROWS = 8;
    static constexpr uint32_t DISPATCH_PEER_WORKERS = MegaMoeConfig::A2_DISPATCH_PEER_WORKERS;
    static constexpr uint32_t RECV_PEER_WORKERS = 4;
    static constexpr uint32_t DISPATCH_EXPERT_BATCH = 8;
    static constexpr int32_t UB_MOVE_NUM = 16 * 1024;

    struct Params {
        uint32_t dispatchExpertBatch{DISPATCH_EXPERT_BATCH};
        bool decodeFastPath{false};
        // Data members
        GemmCoord problemShape;
        __gm__ ElementABefore* ptrA;
        LayoutA layoutA;
        LayoutA layoutA2;
        __gm__ ElementB* ptrB1;
        LayoutB layoutB1;
        __gm__ float* ptrBias1;
        __gm__ ElementB* ptrB2;
        LayoutB layoutB2;
        __gm__ float* ptrBias2;
        // The unified op def exposes optional bias1/bias2 inputs; when absent
        // ptrBias1/ptrBias2 are nullptr. Per-channel scale info is fully carried
        // by `weight_scales1` / `weight_scales2` (forwarded as `ptrScale1` / `ptrScale2`).
        __gm__ ElementScale* ptrScale1;
        LayoutScale layoutScale1;
        __gm__ ElementScale* ptrScale2;
        LayoutScale layoutScale2;
        __gm__ ElementD2* ptrOutput;
        LayoutD1 layoutD1;
        LayoutD2 layoutD2;
        GM_ADDR ptrWorkspace;
        GM_ADDR ptrExpertTokenNums;
        int32_t EP;
        int32_t listLen;
        int32_t expertPerRank;
        uint64_t maxOutputSize;
        // for reuse workspace, gmm1 out preRow Stride use max(n,k)
        uint32_t gmmOutPreRowStride;
        // Padded token-slot stride for GMM1 input. It prevents packed SwiGLU
        // output from overwriting unread GMM1 rows when k < n/2.
        uint32_t gmmA1PreRowStride = 0;
        //--------------
        GM_ADDR expertIdx;
        GM_ADDR moeInitRoutingQuantV2Scale;
        GM_ADDR moeInitRoutingQuantV2Offset;
        GM_ADDR expandedX;
        GM_ADDR expandedRowIdx;
        GM_ADDR expertTokensCountOrCumsum;
        GM_ADDR expertTokensBeforeCapacity;
        GM_ADDR dynamicQuantScale;
        GM_ADDR probs;
        GM_ADDR ptrXActiveMask;
        GM_ADDR ptrScales;
        int64_t topK;
        uint32_t epilogueCoreNum;
        // Epilogue scheduling granularity. Default 0 keeps the legacy W4A8
        // behavior (no special throttling). INT8 path historically uses
        // `expertPerRank - 1`; BF16 path historically uses `expertPerRank - 2`.
        uint32_t epilogueGranularity{0};
        float swigluLimit;
        uint32_t activationCode{0};
        float activationParams1{Epilogue::SwigluOaiActivation::DEFAULT_ALPHA};
        float activationParams2{Epilogue::SituActivation::DEFAULT_BETA};
        GM_ADDR contextGM{nullptr};
        GM_ADDR tilingGM{nullptr};

        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GemmCoord problemShape_, uint32_t EP_, uint32_t listLen_, uint32_t expertPerRank_,
               uint64_t maxOutputSize_, int64_t topK_, uint32_t epilogueCoreNum_, GM_ADDR contextGM_, GM_ADDR ptrA_,
               LayoutA layoutA_, LayoutA layoutA2_, GM_ADDR ptrB1_, LayoutB layoutB1_, GM_ADDR ptrBias1_,
               GM_ADDR ptrB2_, LayoutB layoutB2_, GM_ADDR ptrBias2_, GM_ADDR ptrScale1_, LayoutScale layoutScale1_,
               GM_ADDR ptrScale2_, LayoutScale layoutScale2_, GM_ADDR ptrOutput_, LayoutD1 layoutD1_,
               LayoutD2 layoutD2_, GM_ADDR expertIdx_, GM_ADDR moeInitRoutingQuantV2Scale_,
               GM_ADDR moeInitRoutingQuantV2Offset_, GM_ADDR expertTokensBeforeCapacity_, GM_ADDR probs_,
               GM_ADDR ptrWorkspace_, GM_ADDR gmExpertTokenNums_, GM_ADDR ptrXActiveMask_, GM_ADDR ptrScales_,
               uint32_t epilogueGranularity_ = 0, float swigluLimit_ = std::numeric_limits<float>::infinity(),
               uint32_t activationCode_ = 0, float activationParams1_ = Epilogue::SwigluOaiActivation::DEFAULT_ALPHA,
               float activationParams2_ = Epilogue::SituActivation::DEFAULT_BETA, GM_ADDR tilingGM_ = nullptr)
            : problemShape(problemShape_),
              EP(EP_),
              listLen(listLen_),
              expertPerRank(expertPerRank_),
              maxOutputSize(maxOutputSize_),
              topK(topK_),
              epilogueCoreNum(epilogueCoreNum_),
              epilogueGranularity(epilogueGranularity_),
              swigluLimit(swigluLimit_),
              activationCode(activationCode_),
              activationParams1(activationParams1_),
              activationParams2(activationParams2_),
              contextGM(contextGM_),
              tilingGM(tilingGM_),
              ptrA(reinterpret_cast<__gm__ ElementABefore*>(ptrA_)),
              layoutA(layoutA_),
              layoutA2(layoutA2_),
              ptrB1(reinterpret_cast<__gm__ ElementB*>(ptrB1_)),
              layoutB1(layoutB1_),
              ptrBias1(reinterpret_cast<__gm__ float*>(ptrBias1_)),
              ptrB2(reinterpret_cast<__gm__ ElementB*>(ptrB2_)),
              layoutB2(layoutB2_),
              ptrBias2(reinterpret_cast<__gm__ float*>(ptrBias2_)),
              ptrScale1(reinterpret_cast<__gm__ ElementScale*>(ptrScale1_)),
              layoutScale1(layoutScale1_),
              ptrScale2(reinterpret_cast<__gm__ ElementScale*>(ptrScale2_)),
              layoutScale2(layoutScale2_),
              ptrOutput(reinterpret_cast<__gm__ ElementD2*>(ptrOutput_)),
              layoutD1(layoutD1_),
              layoutD2(layoutD2_),
              expertIdx(expertIdx_),
              moeInitRoutingQuantV2Scale(moeInitRoutingQuantV2Scale_),
              moeInitRoutingQuantV2Offset(moeInitRoutingQuantV2Offset_),
              expertTokensBeforeCapacity(expertTokensBeforeCapacity_),
              probs(probs_),
              ptrXActiveMask(ptrXActiveMask_),
              ptrScales(ptrScales_),
              ptrWorkspace(ptrWorkspace_),
              ptrExpertTokenNums(gmExpertTokenNums_)
        {
            gmmOutPreRowStride = problemShape.n() > problemShape.k() ? problemShape.n() : problemShape.k();
            gmmA1PreRowStride = static_cast<uint32_t>(layoutA.stride(0));
        }
    };

    // Batched dispatch preserves the arithmetic and row layout for each weight dtype.
    CATLASS_DEVICE static int64_t PermuteChunk(Params const& params)
    {
        return MegaMoeConfig::GetPermuteChunk(BatchedDispatch);
    }
    CATLASS_DEVICE static uint32_t RoutingPadding(Params const& params)
    {
        return BatchedDispatch ? BYTE_PER_BLK : ALIGN_512;
    }

    CATLASS_DEVICE static bool UseSenderMsd(Params const& params)
    {
        const uint32_t h = params.problemShape.k();
        return BatchedDispatch && params.gmmA1PreRowStride >= h &&
               params.gmmA1PreRowStride <= h + RoutingPadding(params);
    }
    CATLASS_DEVICE bool UseExpertParallelDispatch(Params const& params)
    {
        return BatchedDispatch;
    }
    CATLASS_DEVICE
    MegaMoeKernelA2Bf16W4A8(Params const& params)
    {
        if ASCEND_IS_AIC {
            coreIdx = AscendC::GetBlockIdx();
            coreNum = AscendC::GetBlockNum();
        }

        if ASCEND_IS_AIV {
            coreIdx = get_block_idx() + get_subblockid() * get_block_num();
            coreNum = get_block_num() * get_subblockdim();
        }

        routingPadding_ = RoutingPadding(params);
        initBuffer(params);
    }

    CATLASS_DEVICE
    ~MegaMoeKernelA2Bf16W4A8() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const& params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const& params)
    {
        const int64_t bs = params.problemShape.m();
        const int64_t numChunks = (bs + PermuteChunk(params) - 1) / PermuteChunk(params);
        for (int64_t chunkIdx = 0; chunkIdx < numChunks; ++chunkIdx) {
            // cumsumMM 每 chunk 指向独立缓冲区（不复用，规避跨 chunk cache 陈旧读）；
            // syncgmm / C2V / V2C 握手按 chunk 归零复用（与 AIV 每 chunk 的 set 序列严格配对）
            cumsumMM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(
                workspaceInfo.ptrcumsumMM +
                chunkIdx * static_cast<int64_t>(paddedExpertNumAligned) * params.EP * sizeof(int32_t)));
            GMM1(params);
            GMM2(params);
        }
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const& params)
    {
        DispatchAndCombine(params);
    }

private:
    CATLASS_DEVICE
    int32_t RuntimeRank(Params const& params) const
    {
        (void)params;
        return shmem.Rank();
    }

    CATLASS_DEVICE void initBuffer(Params const& params)
    {
        auto tmpContext = reinterpret_cast<__gm__ Mc2Aclnn::Mc2MoeContext*>(params.contextGM);
        shmem.initShmem(tmpContext);
        if ASCEND_IS_AIV {
            GM_ADDR dumpBase = shmem();
            exceptionDump_.Init(dumpBase, params.tilingGM);
        }
        qp_info_ = reinterpret_cast<__gm__ HcclAiRMAInfo*>(
            reinterpret_cast<__gm__ HcclA2CombineOpParam*>(shmem.WinContext_)->aiRMAInfo);
        serverId_ = static_cast<int32_t>(RuntimeRank(params) / SERVER_RANK_SIZE_A2);

        workspaceInfo = WorkspaceInfo(params);
        peermemInfo = PeermemInfo(params, shmem);

        cumsumMM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspaceInfo.ptrcumsumMM));
        gmS.SetGlobalBuffer(params.ptrScale1);
        gmA1I4.SetGlobalBuffer(reinterpret_cast<__gm__ int4b_t*>(workspaceInfo.ptrA1Int4));
        gmA1I4_I8.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(workspaceInfo.ptrA1Int4));
        if (UseSenderMsd(params)) {
            gmA1I4.SetGlobalBuffer(reinterpret_cast<__gm__ int4b_t*>(shmem() + peermemInfo.offsetA));
            gmA1I4_I8.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(shmem() + peermemInfo.offsetA));
        }
        gmA2I4.SetGlobalBuffer(reinterpret_cast<__gm__ int4b_t*>(workspaceInfo.ptrA2Int4));
        gmA2I4_I8.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(workspaceInfo.ptrA2Int4));
        gmS2.SetGlobalBuffer(params.ptrScale2);
        gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementABefore*>(shmem() + peermemInfo.offsetA));

        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC*>(workspaceInfo.ptrC));
        gmC2.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC*>(workspaceInfo.ptrC2));

        gmPerTokenScale1.SetGlobalBuffer(
            reinterpret_cast<__gm__ ElementPerTokenScale*>(workspaceInfo.ptrPerTokenScale));
        if (UseSenderMsd(params)) {
            gmPerTokenScale1.SetGlobalBuffer(
                reinterpret_cast<__gm__ float*>(shmem() + peermemInfo.offsetPeerPerTokenScale));
        }
        gmPerTokenScale2.SetGlobalBuffer(
            reinterpret_cast<__gm__ ElementPerTokenScale*>(workspaceInfo.ptrPerTokenScale2));

        tokenPerExpert.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(shmem() + peermemInfo.offsetPeerTokenPerExpert));

        paddedExpertNumAligned = AlignUp(params.EP * params.expertPerRank + 1, ALIGN_128);
        tokenPerExpertLayout = Layout3D(paddedExpertNumAligned, params.expertPerRank);
        preSumBeforeRankForDispatch.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(workspaceInfo.ptrSumBeforeRankForDispatch));
        preSumBeforeRankForCombine.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(workspaceInfo.ptrSumBeforeRankForCombine));

        if ASCEND_IS_AIV {
            InitRouteCache(params);
        }
    }

    CATLASS_DEVICE void InitRouteCache(Params const& params)
    {
        // UB 顶部依次存放 TPE 发送/接收视图和两个 preSum 视图；每 chunk 刷新。
        routeCacheNum = params.EP * params.expertPerRank;
        expertPerRankAligned = AlignUp(params.expertPerRank, 8);
        uint32_t routeCacheBytes = AlignUp(routeCacheNum * sizeof(int32_t), BYTE_PER_BLK);
        uint32_t tpeCacheBytes = AlignUp(params.EP * expertPerRankAligned * sizeof(int32_t), BYTE_PER_BLK);
        uint32_t routeCacheUbOffset =
            ArchTag::UB_SIZE - 3 * routeCacheBytes - tpeCacheBytes - (BatchedDispatch ? 2 * routeCacheBytes : 0);
        ubTpeSendCache = resource.ubBuf.template GetBufferByByte<int32_t>(routeCacheUbOffset);
        ubTpeSendCache.SetSize(routeCacheNum);
        ubTpeRecvCache = resource.ubBuf.template GetBufferByByte<int32_t>(routeCacheUbOffset + routeCacheBytes);
        if constexpr (BatchedDispatch) {
            ubBatchSendBase = resource.ubBuf.template GetBufferByByte<int32_t>(ArchTag::UB_SIZE - 2 * routeCacheBytes);
            ubBatchRecvBase = resource.ubBuf.template GetBufferByByte<int32_t>(ArchTag::UB_SIZE - routeCacheBytes);
        }
        ubPreSumDispatchCache =
            resource.ubBuf.template GetBufferByByte<int32_t>(routeCacheUbOffset + routeCacheBytes + tpeCacheBytes);
        ubPreSumDispatchCache.SetSize(routeCacheNum);
        ubPreSumCombineCache =
            resource.ubBuf.template GetBufferByByte<int32_t>(routeCacheUbOffset + 2 * routeCacheBytes + tpeCacheBytes);
        ubPreSumCombineCache.SetSize(routeCacheNum);
    }

    template <typename T>
    CATLASS_DEVICE void CopyGMToGM(AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, int32_t elemNum,
                                   int32_t ubMoveNum)
    {
        Detail::CopyGMToGMShared<ArchTag>(resource, dst, src, elemNum, ubMoveNum);
    }

    CATLASS_DEVICE void FetchAndPreprocessInt8ToInt4(uint64_t gmOffsetA, AscendC::GlobalTensor<float> dstScale,
                                                     AscendC::GlobalTensor<int8_t> src, int32_t rows,
                                                     int32_t hiddenSize, uint32_t aStride)
    {
        if (BatchedDispatch && hiddenSize == 2048 && routingPadding_ == BYTE_PER_BLK &&
            aStride >= static_cast<uint32_t>(hiddenSize) && aStride % BYTE_PER_BLK == 0) {
            Detail::FetchAndPreprocessInt8ToInt4BatchedShared<ArchTag>(resource, gmA1I4_I8, gmOffsetA, dstScale, src,
                                                                       rows, hiddenSize, aStride, routingPadding_);
            return;
        }
        Detail::FetchAndPreprocessInt8ToInt4Shared<ArchTag>(resource, gmA1I4_I8, gmOffsetA, dstScale, src, rows,
                                                            hiddenSize, aStride, coreNum, routingPadding_);
    }

    // cumsumMM 由 core 0 用 MTE3 写、其他核用标量读，标量读会命中本核 D-Cache。
    // 该 workspace 地址在不同 layer 复用，cache line 可能仍保留上一层的前缀和，
    // 累计后会把 rowStart 推过 maxOutputSize，发送侧丢弃接收侧仍在等待的 rows。
    // 因此每次标量读前先 dcci，确保读到本轮值。
    CATLASS_DEVICE int32_t ReadCumsum(int64_t idx)
    {
        __gm__ int32_t* p = const_cast<__gm__ int32_t*>(cumsumMM.GetPhyAddr()) + idx;
        gm_dcci(reinterpret_cast<__gm__ uint8_t*>(p));
        return gm_load(p);
    }

    CATLASS_DEVICE
    void GetCumsumForMMAIV(AscendC::GlobalTensor<int32_t>& tokenPerExpert, AscendC::GlobalTensor<int32_t>& result,
                           uint32_t expertPerRank, uint32_t EP)
    {
        Detail::GetCumsumForMMAIVShared<ArchTag>(resource, paddedExpertNumAligned, tokenPerExpert, result,
                                                 expertPerRank, EP);
    }

    CATLASS_DEVICE void CacheRouteTablesToUb(Params const& params)
    {
        if (routeCacheNum == 0) {
            return;
        }
        const int32_t rank = RuntimeRank(params);
        if constexpr (BatchedDispatch) {
            // 发送视图：本 rank 的路由表（winIn TPE 的 srcRank=rank 槽，flat [dstEpIdx][g]，连续 EP*epr）
            AscendC::DataCopyPad(ubTpeSendCache, tokenPerExpert[tokenPerExpertLayout(rank, 0, 0)],
                                 {1, static_cast<uint16_t>(routeCacheNum * sizeof(int32_t)), 0, 0}, {});
        } else {
            // 发送视图：本 rank 的路由表。直接取 windowsOut（allgather 广播的数据源），
            // 不再依赖 winIn 的 self-copy：全模型下观测到 self-copy 与广播值不一致，
            // 发送侧会对 rows==0 的 cell 不发 flag，而接收侧 allgather 仍显示 rows>0。
            AscendC::GlobalTensor<int32_t> tpeSendSrc;
            tpeSendSrc.SetGlobalBuffer(
                reinterpret_cast<__gm__ int32_t*>(shmem.windowsOutAddr() + peermemInfo.offsetPeerTokenPerExpert));
            AscendC::DataCopyPad(ubTpeSendCache, tpeSendSrc[tokenPerExpertLayout(rank, 0, 0)],
                                 {1, static_cast<uint16_t>(routeCacheNum * sizeof(int32_t)), 0, 0}, {});
        }
        // 接收视图：[srcEpIdx][g] = 各源 rank 发给本 rank 的计数（跨 srcEpIdx 取 rank 平面，
        // 与 A3 CacheRouteTablesToUb 相同的 2D DataCopyPad）
        AscendC::DataCopyPad(
            ubTpeRecvCache, tokenPerExpert[rank * params.expertPerRank],
            {static_cast<uint16_t>(params.EP), static_cast<uint16_t>(params.expertPerRank * sizeof(int32_t)),
             static_cast<uint16_t>((paddedExpertNumAligned - params.expertPerRank) * sizeof(int32_t)), 0},
            {});
        // preSum 两表连续存放，整段搬入
        AscendC::DataCopyPad(ubPreSumDispatchCache, preSumBeforeRankForDispatch,
                             {1, static_cast<uint16_t>(routeCacheNum * sizeof(int32_t)), 0, 0}, {});
        AscendC::DataCopyPad(ubPreSumCombineCache, preSumBeforeRankForCombine,
                             {1, static_cast<uint16_t>(routeCacheNum * sizeof(int32_t)), 0, 0}, {});
        if (UseExpertParallelDispatch(params)) {
            // Cache the total count and this source's prefix for every destination expert.
            AscendC::DataCopyPad(ubBatchSendBase, cumsumMM[tokenPerExpertLayout(params.EP - 1, 0, 0)],
                                 {1, static_cast<uint16_t>(routeCacheNum * sizeof(int32_t)), 0, 0}, {});
            if (rank != 0) {
                AscendC::DataCopyPad(ubBatchRecvBase, cumsumMM[tokenPerExpertLayout(rank - 1, 0, 0)],
                                     {1, static_cast<uint16_t>(routeCacheNum * sizeof(int32_t)), 0, 0}, {});
            }
        }
        // 标量读取前确保 MTE2 搬入完成（与 A3 CacheRouteTablesToUb 一致）
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        if (UseExpertParallelDispatch(params)) {
            for (uint32_t peer = 0; peer < params.EP; ++peer) {
                uint32_t expertBase = 0;
                for (uint32_t group = 0; group < params.expertPerRank; ++group) {
                    const uint32_t index = peer * params.expertPerRank + group;
                    const uint32_t count = ubBatchSendBase.GetValue(index);
                    const uint32_t sourcePrefix = rank == 0 ? 0 : ubBatchRecvBase.GetValue(index);
                    ubBatchSendBase.SetValue(index, expertBase + sourcePrefix);
                    expertBase += count;
                }
            }
            // Reuse the source-prefix scratch for local receive offsets, in expert-major order.
            uint32_t expertBase = 0;
            for (uint32_t group = 0; group < params.expertPerRank; ++group) {
                for (uint32_t peer = 0; peer < params.EP; ++peer) {
                    const uint32_t index = peer * expertPerRankAligned + group;
                    ubBatchRecvBase.SetValue(index, expertBase);
                    expertBase += ubTpeRecvCache.GetValue(index);
                }
            }
        }
    }

    CATLASS_DEVICE
    void GMM1(Params const& params)
    {
        icache_preload(ICACHE_PRELOAD_LINES);
        BlockScheduler blockScheduler;
        BlockMmad blockMmad(resource);

        int64_t gmGroupOffsetA = 0;
        int64_t gmGroupOffsetB = 0;
        int64_t gmGroupOffsetC = 0;
        uint32_t startCoreIdx = 0;
        uint32_t syncGroupIdx = 0;
        const uint32_t readyExperts = UseDecodeFastPath<BatchedDispatch>(params) ? params.dispatchExpertBatch : 1U;
        int64_t preCurrentmSum = 0;
        int32_t syncLoopIdx = -1;

        uint16_t syncgmmIdx = 0;
        AscendC::CrossCoreWaitFlag<0x2>(syncgmmIdx /
                                        CROSS_CORE_FLAG_MAX_SET_COUNT); // Wait for AIV to finish cumsum for matmul
        syncgmmIdx++;
        AscendC::GlobalTensor<ElementB> gmB1;

        for (uint32_t groupIdx = 0; groupIdx < params.expertPerRank; ++groupIdx) {
            uint32_t currentM = cumsumMM(tokenPerExpertLayout(params.EP - 1, RuntimeRank(params), groupIdx));
            if (preCurrentmSum >= params.maxOutputSize) {
                currentM = 0;
            } else if (preCurrentmSum + currentM >= params.maxOutputSize) {
                currentM = params.maxOutputSize - preCurrentmSum;
            }
            currentM = currentM * GATE_UP_ROWS_PER_TOKEN;

            uint32_t coreLoops = 0;
            if (!UseDecodeFastPath<BatchedDispatch>(params) || currentM != 0) {
                int32_t arrayGroupIdx = params.listLen == 1 ? 0 : groupIdx;
                gmB1.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB*>(
                    GetTensorAddr<ElementB>(arrayGroupIdx, reinterpret_cast<GM_ADDR>(params.ptrB1))));
                gmS.SetGlobalBuffer(reinterpret_cast<__gm__ ElementScale*>(
                    GetTensorAddr<int64_t>(arrayGroupIdx, reinterpret_cast<GM_ADDR>(params.ptrScale1))));

                if (currentM <= L1TileShape::M) {
                    gmB1.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
                }
                GemmCoord inGroupProblemShape{currentM, params.problemShape.n(), params.problemShape.k()};
                LayoutA layoutA = params.layoutA.GetTileLayout(inGroupProblemShape.GetCoordMK());
                LayoutB layoutB1 = params.layoutB1;
                LayoutScale layoutScale = params.layoutScale1;
                LayoutC layoutC(inGroupProblemShape.m(), inGroupProblemShape.n(), params.gmmOutPreRowStride);
                blockScheduler.Update(inGroupProblemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
                coreLoops = blockScheduler.GetCoreLoops();
                // Determine the starting loopIdx of the current core under the current groupIdx
                uint32_t startLoopIdx = ((coreIdx < startCoreIdx) ? (coreIdx + coreNum) : coreIdx) - startCoreIdx;
                // Loop through the matmul of each groupIdx

                for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                    for (; syncGroupIdx <= groupIdx; syncGroupIdx += readyExperts) {
                        AscendC::CrossCoreWaitFlag<0x2>(syncgmmIdx / CROSS_CORE_FLAG_MAX_SET_COUNT);
                        syncgmmIdx++;
                    }

                    // Compute block location
                    GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
                    GemmCoord actualBlockShape = blockScheduler.GetActualBlockShape(blockCoord);
                    // Compute initial location in logical coordinates
                    int64_t gmOffsetA, gmOffsetB, gmOffsetC;
                    Detail::GetGmmTileOffsets<L1TileShape>(blockCoord, layoutA, layoutB1, layoutC, gmOffsetA, gmOffsetB,
                                                           gmOffsetC);
                    int64_t gmOffsetS = blockCoord.n() * L1TileShape::N +
                                        (params.listLen == 1 ? groupIdx * params.problemShape.n() : 0);
                    if (currentM > 0) {
                        if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
                            blockMmad(gmS[gmOffsetS], layoutScale, gmA1I4[gmGroupOffsetA + gmOffsetA], layoutA,
                                      gmB1[gmGroupOffsetB + gmOffsetB], layoutB1, gmC[gmGroupOffsetC + gmOffsetC],
                                      layoutC, actualBlockShape, Callback{}, Callback{});
                        } else {
                            blockMmad(gmS[gmOffsetS], layoutScale, gmA1I4[gmGroupOffsetA + gmOffsetA], layoutA,
                                      gmB1[gmGroupOffsetB + gmOffsetB], layoutB1, gmC[gmGroupOffsetC + gmOffsetC],
                                      layoutC, actualBlockShape);
                        }
                        if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
                            blockMmad.SynchronizeBlock();
                        }
                    }
                }
            }

            if (IsActivationBoundary<BatchedDispatch>(params, groupIdx)) {
                syncLoopIdx++;
                if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
                    blockMmad.SynchronizeBlock();
                }
                blockMmad.Finalize(syncLoopIdx, SYNCFLAGC2V);
            }
            preCurrentmSum += currentM / GATE_UP_ROWS_PER_TOKEN;

            gmGroupOffsetA += static_cast<int64_t>(currentM) * params.gmmA1PreRowStride;
            if (params.listLen == 1) {
                gmGroupOffsetB += static_cast<int64_t>(params.problemShape.k()) * params.problemShape.n();
            }
            gmGroupOffsetC += currentM * params.gmmOutPreRowStride;
            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
        }

        for (; syncGroupIdx < params.expertPerRank; syncGroupIdx += readyExperts) {
            AscendC::CrossCoreWaitFlag<0x2>(syncgmmIdx / CROSS_CORE_FLAG_MAX_SET_COUNT);
            syncgmmIdx++;
        }
    }

    CATLASS_DEVICE
    void GMM2(Params const& params)
    {
        icache_preload(ICACHE_PRELOAD_LINES);
        BlockScheduler blockScheduler;
        BlockMmad blockMmad(resource);

        uint32_t n2 = params.problemShape.k();
        uint32_t k2 = params.problemShape.n() / 2;

        int64_t gmGroupOffsetA = 0;
        int64_t gmGroupOffsetB = 0;
        int64_t gmGroupOffsetC = 0;

        uint32_t startCoreIdx = 0;

        int64_t preCurrentmSum = 0;
        int32_t syncLoopIdx = -1;

        for (uint32_t groupIdx = 0; groupIdx < params.expertPerRank; ++groupIdx) {
            uint32_t currentM = cumsumMM(tokenPerExpertLayout(params.EP - 1, RuntimeRank(params), groupIdx));
            if (preCurrentmSum >= params.maxOutputSize) {
                currentM = 0;
            } else if (preCurrentmSum + currentM > params.maxOutputSize) {
                currentM = params.maxOutputSize - preCurrentmSum;
            }
            currentM = currentM * GATE_UP_ROWS_PER_TOKEN;

            if (groupIdx == 0 || IsActivationBoundary<BatchedDispatch>(params, groupIdx - 1)) {
                AscendC::CrossCoreWaitFlag<0x2>(SYNCFLAGV2C);
            }
            if (UseDecodeFastPath<BatchedDispatch>(params) && currentM == 0) {
                if (params.listLen == 1) {
                    gmGroupOffsetB += static_cast<int64_t>(k2) * n2;
                }
                continue;
            }
            AscendC::GlobalTensor<ElementB> gmB2;
            int32_t arrayGroupIdx = params.listLen == 1 ? 0 : groupIdx;
            gmB2.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB*>(
                GetTensorAddr<ElementB>(arrayGroupIdx, reinterpret_cast<GM_ADDR>(params.ptrB2))));
            gmS2.SetGlobalBuffer(reinterpret_cast<__gm__ ElementScale*>(
                GetTensorAddr<int64_t>(arrayGroupIdx, reinterpret_cast<GM_ADDR>(params.ptrScale2))));
            if (currentM <= L1TileShape::M) {
                gmB2.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            }
            GemmCoord inGroupProblemShape{currentM, n2, k2}; // M N K

            LayoutA layoutA = params.layoutA2.GetTileLayout(inGroupProblemShape.GetCoordMK());
            LayoutB layoutB2 = params.layoutB2;
            LayoutScale layoutScale = params.layoutScale2;
            LayoutC layoutC = LayoutC(inGroupProblemShape.m(), inGroupProblemShape.n());

            blockScheduler.Update(inGroupProblemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
            uint32_t coreLoops = blockScheduler.GetCoreLoops();

            // Determine the starting loopIdx of the current core under the current groupIdx
            uint32_t startLoopIdx = ((coreIdx < startCoreIdx) ? (coreIdx + coreNum) : coreIdx) - startCoreIdx;
            // Loop through the matmul of each groupIdx

            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                if (loopIdx + coreNum >= coreLoops) {
                    syncLoopIdx = groupIdx;
                }
                // Compute block location
                GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = blockScheduler.GetActualBlockShape(blockCoord);

                // Compute initial location in logical coordinates
                int64_t gmOffsetA, gmOffsetB, gmOffsetC;
                Detail::GetGmmTileOffsets<L1TileShape>(blockCoord, layoutA, layoutB2, layoutC, gmOffsetA, gmOffsetB,
                                                       gmOffsetC);
                int64_t gmOffsetS = blockCoord.n() * L1TileShape::N +
                                    (params.listLen == 1 ? groupIdx * n2 : 0); // One scale group per expert
                if (currentM > 0) {
                    int64_t gmOffsetS = blockCoord.n() * L1TileShape::N + (params.listLen == 1 ? groupIdx * n2 : 0);
                    if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
                        blockMmad(gmS2[gmOffsetS], layoutScale, gmA2I4[gmGroupOffsetA + gmOffsetA], layoutA,
                                  gmB2[gmGroupOffsetB + gmOffsetB], layoutB2, gmC2[gmGroupOffsetC + gmOffsetC], layoutC,
                                  actualBlockShape, Callback{}, Callback{}, syncLoopIdx, 0);
                    } else {
                        blockMmad(gmS2[gmOffsetS], layoutScale, gmA2I4[gmGroupOffsetA + gmOffsetA], layoutA,
                                  gmB2[gmGroupOffsetB + gmOffsetB], layoutB2, gmC2[gmGroupOffsetC + gmOffsetC], layoutC,
                                  actualBlockShape);
                    }
                }
            }
            preCurrentmSum += currentM / GATE_UP_ROWS_PER_TOKEN;
            gmGroupOffsetA += inGroupProblemShape.m() * inGroupProblemShape.k();
            if (params.listLen == 1) {
                gmGroupOffsetB += inGroupProblemShape.k() * inGroupProblemShape.n();
            }
            gmGroupOffsetC += inGroupProblemShape.m() * inGroupProblemShape.n();

            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
        }
        if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
            blockMmad.SynchronizeBlock();
        }
        blockMmad.Finalize(params.expertPerRank - 1, 0);
    }

    CATLASS_DEVICE
    void CrossRankSyncAndlocalTokenPerExpertAllGatherAndGetSumPreRankV2(Params const& params,
                                                                        int64_t localTokenPerExpertOffset)
    {
        const int32_t rank = RuntimeRank(params);
        Detail::CrossRankAllGatherShared<ArchTag>(resource, shmem, peermemInfo, tokenPerExpertLayout, tokenPerExpert,
                                                  preSumBeforeRankForDispatch, preSumBeforeRankForCombine, qp_info_,
                                                  localTokenPerExpertOffset, rank, coreIdx, coreNum, serverId_,
                                                  paddedExpertNumAligned, params.EP, params.expertPerRank);
    }

    CATLASS_DEVICE
    void ResetTokenPerExpert(const Params& params, int32_t num)
    {
        Detail::ResetTokenPerExpertShared(resource, shmem, peermemInfo, tokenPerExpert, coreNum, coreIdx, params.EP,
                                          params.expertPerRank, num);
    }

    // 跨 server 时 flag 的 32B 源数据来自 winOut 的暂存区。每个 (dst, group) 使用
    // 独立槽位：正常路径写 MAGIC、跳过路径写 MAGIC+1，若共用槽位，同组其它 dst 的
    // 并发 post 会在 NIC 读取前覆盖内容，导致对端读到错误标记。暂存区放在 flag 区第
    // EP*expertPerRank 个槽之后；chunk 末尾的 ResetTokenPerExpert 只清零前
    // EP*expertPerRank 个槽，不会覆盖这里。
    CATLASS_DEVICE int64_t FlagStagingSlot(int32_t dstEpIdx, int32_t groupIdx, Params const& params) const
    {
        return static_cast<int64_t>(params.EP * params.expertPerRank) +
               static_cast<int64_t>(dstEpIdx * params.expertPerRank + groupIdx);
    }

    // rows == 0（或 rowStart 超过 maxOutputSize）时给对端写一个跳过标记 MAGIC+1，
    // 避免接收侧按 allgather 计数继续等待一个永远不会发送的 flag。
    CATLASS_DEVICE void PostSkipFlag(int32_t dstEpIdx, int32_t groupIdx, Params const& params)
    {
        const int32_t rank = RuntimeRank(params);
        if (dstEpIdx == rank) {
            return;
        }
        __gm__ int32_t* remoteFlag = reinterpret_cast<__gm__ int32_t*>(shmem(0, dstEpIdx) + peermemInfo.offsetFlag) +
                                     (rank * params.expertPerRank + groupIdx) * PeermemInfo::kFlagSlotI32;
        if ((dstEpIdx / SERVER_RANK_SIZE_A2) == serverId_) {
            AscendC::PipeBarrier<PIPE_ALL>();
            gm_store(remoteFlag, FLAG_VALUE_MAGIC + 1);
            gm_dcci(reinterpret_cast<__gm__ uint8_t*>(remoteFlag));
            return;
        }
        uint32_t copyInNum = params.problemShape.k() + ALIGN_512;
        uint64_t ubRdmaOffset = 2ULL * copyInNum * sizeof(uint32_t);
        AscendC::LocalTensor<uint64_t> rdmaUbLocal = resource.ubBuf.template GetBufferByByte<uint64_t>(ubRdmaOffset);
        AscendC::LocalTensor<uint32_t> rdmaUbLocalHead =
            resource.ubBuf.template GetBufferByByte<uint32_t>(ubRdmaOffset + UB_ALIGN);
        AscendC::LocalTensor<int32_t> flagUb =
            resource.ubBuf.template GetBufferByByte<int32_t>(ubRdmaOffset + 2 * UB_ALIGN);
        AscendC::Duplicate<int32_t>(flagUb, 0, PeermemInfo::kFlagPayloadI32);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        flagUb.SetValue(0, FLAG_VALUE_MAGIC + 1);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::GlobalTensor<int32_t> localFlagGm;
        localFlagGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(shmem.windowsOutAddr() + peermemInfo.offsetFlag) +
                                    FlagStagingSlot(dstEpIdx, groupIdx, params) * PeermemInfo::kFlagSlotI32);
        AscendC::DataCopy(localFlagGm, flagUb, PeermemInfo::kFlagPayloadI32);
        // 保证 32B 暂存写已落到 GM，NIC 读取源地址时才不会拿到旧内容
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::PipeBarrier<PIPE_ALL>();
        AIVRDMAPostSend((GM_ADDR)localFlagGm.GetPhyAddr(), (GM_ADDR)remoteFlag, static_cast<uint64_t>(dstEpIdx),
                        PeermemInfo::kFlagPayloadI32 * sizeof(int32_t), qp_info_, rdmaUbLocal, rdmaUbLocalHead);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    CATLASS_DEVICE void SendExpertBatch(Params const& params, uint32_t firstGroup, uint32_t groups)
    {
        const uint32_t tasks = groups * params.EP;
        const uint32_t workers =
            Detail::DispatchWorkerCount(params.decodeFastPath, coreNum, tasks, DISPATCH_PEER_WORKERS);
        const uint32_t worker = coreIdx % workers;
        const uint32_t stride = params.problemShape.k() + RoutingPadding(params);
        AscendC::GlobalTensor<ElementABefore> src, dst;
        src.SetGlobalBuffer(
            reinterpret_cast<__gm__ ElementABefore*>(shmem.windowsOutAddr() + peermemInfo.offsetWinOutA));
        for (uint32_t task = coreIdx / workers; task < tasks; task += coreNum / workers) {
            const uint32_t peer = task % params.EP;
            const uint32_t group = firstGroup + task / params.EP;
            const uint32_t index = peer * params.expertPerRank + group;
            const uint32_t base = ubBatchSendBase.GetValue(index);
            if (base >= params.maxOutputSize) {
                continue;
            }
            const uint32_t rows = min(static_cast<uint32_t>(ubTpeSendCache.GetValue(index)),
                                      static_cast<uint32_t>(params.maxOutputSize - base));
            uint32_t first;
            const uint32_t workerRows = Detail::PartitionDispatchRows(rows, workers, worker, first);
            if (first >= rows) {
                continue;
            }
            const uint32_t source = ubPreSumDispatchCache.GetValue(index) + first;
            dst.SetGlobalBuffer(reinterpret_cast<__gm__ ElementABefore*>(shmem(0, peer) + peermemInfo.offsetA));
            SendTokensV3<ElementABefore>(dst[static_cast<int64_t>(base + first) * stride],
                                         src[static_cast<int64_t>(source) * stride], workerRows,
                                         params.problemShape.k(), peer, group, base + first, params,
                                         static_cast<uint64_t>(base + first) * params.gmmA1PreRowStride, false);
        }
    }
    CATLASS_DEVICE void PublishExpertBatch(Params const& params, int32_t rank, uint32_t firstGroup, uint32_t groups)
    {
        // All producers have completed MTE3 and passed the full-AIV send barrier.
        // Assign each peer to one publisher, including when EP exceeds the AIV count.
        for (uint32_t peer = coreIdx; peer < params.EP; peer += coreNum) {
            for (uint32_t group = firstGroup; group < firstGroup + groups; ++group) {
                const uint32_t index = peer * params.expertPerRank + group;
                const bool hasRows =
                    ubBatchSendBase.GetValue(index) < params.maxOutputSize && ubTpeSendCache.GetValue(index) != 0;
                PublishCooperativeDispatch(params, rank, peer, group, hasRows);
            }
        }
    }
    CATLASS_DEVICE void RecvExpertBatch(Params const& params, uint32_t firstGroup, uint32_t groups)
    {
        const uint32_t tasks = groups * params.EP;
        const uint32_t workers = Detail::DispatchWorkerCount(params.decodeFastPath, coreNum, tasks, RECV_PEER_WORKERS);
        const uint32_t worker = coreIdx % workers;
        for (uint32_t task = coreIdx / workers; task < tasks; task += coreNum / workers) {
            const uint32_t peer = task % params.EP;
            const uint32_t group = firstGroup + task / params.EP;
            const uint32_t index = peer * expertPerRankAligned + group;
            const uint32_t base = ubBatchRecvBase.GetValue(index);
            if (base >= params.maxOutputSize) {
                continue;
            }
            const uint32_t rows = min(static_cast<uint32_t>(ubTpeRecvCache.GetValue(index)),
                                      static_cast<uint32_t>(params.maxOutputSize - base));
            uint32_t first;
            const uint32_t workerRows = Detail::PartitionDispatchRows(rows, workers, worker, first);
            // Sender-MSD receive performs readiness only. Worker0 represents this pair;
            // the full-AIV receive barrier broadcasts completion to all producers.
            if (UseSenderMsd(params) && peer / SERVER_RANK_SIZE_A2 == serverId_ && worker != 0)
                continue;
            if (first < rows) {
                RecvTokensV3<ElementABefore>(workerRows, base + first, params.problemShape.k(), peer, group, params,
                                             static_cast<uint64_t>(base + first) * params.gmmA1PreRowStride);
            }
        }
    }
    CATLASS_DEVICE
    void PublishCooperativeDispatch(Params const& params, int32_t rank, uint32_t peer, int32_t groupIdx, bool sentRows)
    {
        // Called after send SyncAll, after every worker's MTE3 completion.
        if (static_cast<int32_t>(peer) == rank || !sentRows) {
            return;
        }
        auto* flag = reinterpret_cast<__gm__ int32_t*>(shmem(0, peer) + peermemInfo.offsetFlag) +
                     (rank * params.expertPerRank + groupIdx) * PeermemInfo::kFlagSlotI32;
        AscendC::PipeBarrier<PIPE_ALL>();
        gm_store(flag, FLAG_VALUE_MAGIC);
        gm_dcci(reinterpret_cast<__gm__ uint8_t*>(flag));
    }

    // ============================================================
    // SendTokensV3：发送侧，三路径（self-rank / 同server / 跨server）
    //   - rows == 0 时提前 return（不写 flag，避免对端死等）
    //   - 路径 A（self-rank）：windowsOut→UB→本地 peer mem，inline 做 scale 分离，无需 flag
    //   - 路径 B（同 server）：IPC 直写远端 peer mem，完成后 DataCopyPad 写 epoch flag
    //   - 路径 C（跨 server）：批量 RDMA 数据，再 RDMA epoch flag
    // ============================================================
    template <typename T>
    CATLASS_DEVICE void SendTokensV3(AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, int32_t rows,
                                     uint32_t hiddenSize, int32_t dstEpIdx, int32_t groupIdx, int32_t rowStart,
                                     Params const& params, uint64_t gmOffsetA, bool publishFlag = true)
    {
        // rows == 0：无 token 可发。给对端写跳过标记，避免接收侧按 allgather 计数继续等待。
        if (rows == 0) {
            PostSkipFlag(dstEpIdx, groupIdx, params);
            return;
        }
        uint32_t copyInNum = hiddenSize + RoutingPadding(params);

        // UB layout:
        //   [0, copyInNum*sizeof(T))             : bufT     (token slab)
        //   [ubRdmaOffset, +UB_ALIGN)            : rdmaUbLocal     (doorbell)
        //   [ubRdmaOffset+UB_ALIGN, +UB_ALIGN)   : rdmaUbLocalHead (head ptr)
        //   [ubFlagOffset, +UB_ALIGN)            : flagUb          (epoch flag)
        uint64_t ubRdmaOffset = BatchedDispatch ? AlignUp(DISPATCH_BATCH_ROWS * copyInNum * sizeof(T), UB_ALIGN) :
                                                  2 * copyInNum * sizeof(uint32_t);
        AscendC::LocalTensor<uint64_t> rdmaUbLocal = resource.ubBuf.template GetBufferByByte<uint64_t>(ubRdmaOffset);
        AscendC::LocalTensor<uint32_t> rdmaUbLocalHead =
            resource.ubBuf.template GetBufferByByte<uint32_t>(ubRdmaOffset + UB_ALIGN);
        uint64_t ubFlagOffset = ubRdmaOffset + 2 * UB_ALIGN;
        AscendC::LocalTensor<int32_t> flagUb = resource.ubBuf.template GetBufferByByte<int32_t>(ubFlagOffset);
        AscendC::LocalTensor<T> bufT = resource.ubBuf.template GetBufferByByte<T>(0);
        AscendC::LocalTensor<float> bufScale = bufT[hiddenSize].template ReinterpretCast<float>();

        int32_t dstServerId = dstEpIdx / SERVER_RANK_SIZE_A2;

        using TType = Gemm::GemmType<T, layout::RowMajor>;
        using CopyGmToUb = Epilogue::Tile::CopyGm2Ub<ArchTag, TType>;
        using CopyUbToGm = Epilogue::Tile::CopyUb2Gm<ArchTag, TType>;
        CopyGmToUb copyGmToUb;
        CopyUbToGm copyUbToGm;

        // -- Path A: self-rank --
        if (static_cast<int32_t>(RuntimeRank(params)) == dstEpIdx) {
            FetchAndPreprocessInt8ToInt4(gmOffsetA, gmPerTokenScale1[rowStart], src, rows, hiddenSize,
                                         params.gmmA1PreRowStride);
            return;
        }

        // -- Path B: same server (IPC direct write) --
        if (dstServerId == serverId_) {
            if (UseSenderMsd(params)) {
                AscendC::GlobalTensor<int8_t> finalInput;
                AscendC::GlobalTensor<float> finalScale;
                finalInput.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(shmem(0, dstEpIdx) + peermemInfo.offsetA));
                finalScale.SetGlobalBuffer(
                    reinterpret_cast<__gm__ float*>(shmem(0, dstEpIdx) + peermemInfo.offsetPeerPerTokenScale));
                Detail::FetchAndPreprocessInt8ToInt4BatchedShared<ArchTag>(
                    resource, finalInput, gmOffsetA, finalScale[rowStart], src, rows, hiddenSize,
                    params.gmmA1PreRowStride, RoutingPadding(params));
            } else {
                Detail::CopyDispatchRows<BatchedDispatch ? DISPATCH_BATCH_ROWS : 1>(dst, src, bufT, rows, copyInNum,
                                                                                    copyGmToUb, copyUbToGm);
            }
            if (!publishFlag)
                return;
            // 写远端 dispatch flag 槽（IPC 直写，与 CrossRankSync 同 server 路径一致）：
            //   1) PipeBarrier<PIPE_ALL>：保证上方 MTE3 数据已 commit，远端先看到数据再看到 flag；
            //   2) gm_store：scalar 写 4B magic；
            //   3) gm_dcci：clean 本核 D-Cache，使远端通过 IPC / scalar 都能立即看到。
            // 每槽 64B 独占 cache line（kFlagSlotI32 = 16 个 int32），避免 false sharing。
            __gm__ int32_t* remoteFlag =
                reinterpret_cast<__gm__ int32_t*>(shmem(0, dstEpIdx) + peermemInfo.offsetFlag) +
                (RuntimeRank(params) * params.expertPerRank + groupIdx) * PeermemInfo::kFlagSlotI32;

            AscendC::PipeBarrier<PIPE_ALL>();
            gm_store(remoteFlag, FLAG_VALUE_MAGIC);
            gm_dcci((__gm__ uint8_t*)remoteFlag);
            return;
        }

        // -- Path C: cross-server (RDMA data then RDMA flag) --
        {
            // 1. 批量 RDMA 数据（rows 个 token 在 src 中连续，一次完成）
            //    src 已指向 windowsOut 中 rowSrc 起始位置，RDMA-registered
            uint64_t totalBytes = static_cast<uint64_t>(rows) * copyInNum * sizeof(T);
            AIVRDMAPostSend((GM_ADDR)src.GetPhyAddr(), (GM_ADDR)dst.GetPhyAddr(), static_cast<uint64_t>(dstEpIdx),
                            totalBytes, qp_info_, rdmaUbLocal, rdmaUbLocalHead);
            AscendC::PipeBarrier<PIPE_ALL>();

            // 2. 在 UB 准备 32B flag payload：[0] = magic，其余 7 个 i32 = 0
            //    32B padding 是为了与 CrossRankSync 跨 server 路径一致（RDMA 最小写单元）
            //    并消除 UB 残留写到对端槽的风险
            AscendC::Duplicate<int32_t>(flagUb, 0, PeermemInfo::kFlagPayloadI32);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            flagUb.SetValue(0, FLAG_VALUE_MAGIC);
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);

            // 3. UB → 本地 winOut dispatch flag 槽（32B 一次写），供 RDMA 源使用
            AscendC::GlobalTensor<int32_t> localFlagGm;
            localFlagGm.SetGlobalBuffer(
                reinterpret_cast<__gm__ int32_t*>(shmem.windowsOutAddr() + peermemInfo.offsetFlag) +
                FlagStagingSlot(dstEpIdx, groupIdx, params) * PeermemInfo::kFlagSlotI32);
            AscendC::DataCopy(localFlagGm, flagUb, PeermemInfo::kFlagPayloadI32);
            // 保证 32B 暂存写已落到 GM，NIC 读取源地址时才不会拿到旧内容
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            AscendC::PipeBarrier<PIPE_ALL>();

            // 4. RDMA 32B 到对端同槽（同 QP 顺序保证 flag 在数据之后到达接收侧）
            GM_ADDR remoteFlagAddr = reinterpret_cast<GM_ADDR>(
                reinterpret_cast<uintptr_t>(shmem(0, dstEpIdx)) + static_cast<uintptr_t>(peermemInfo.offsetFlag) +
                static_cast<uintptr_t>(RuntimeRank(params) * params.expertPerRank + groupIdx) *
                    static_cast<uintptr_t>(PeermemInfo::kFlagSlotI32 * sizeof(int32_t)));
            AIVRDMAPostSend((GM_ADDR)localFlagGm.GetPhyAddr(), remoteFlagAddr, static_cast<uint64_t>(dstEpIdx),
                            PeermemInfo::kFlagPayloadI32 * sizeof(int32_t), // 32B
                            qp_info_, rdmaUbLocal, rdmaUbLocalHead);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    // ============================================================
    // RecvTokensV3：接收侧，等待 epoch flag 后做 scale 分离
    //   - srcEpIdx == rank：路径 A 已处理，直接 return
    //   - rows2 == 0：无 token 可接收，直接 return（避免等待永远不到的 flag）
    //   - 等 flag：gm_signal_wait_until_ne(addr, groupIdx)（等到值 != groupIdx，即 >= groupIdx+1）
    //   - 数据已就绪：逐 token CopyGm2Ub + CopyUb2Gm（只写 hiddenSize）+ DataCopyPad scale
    //   - 使用 catlass Tile copy，与 SendTokensV3 对齐一致性，正确处理非对齐边界
    // ============================================================
    template <typename T>
    CATLASS_DEVICE void RecvTokensV3(int32_t rows2, int32_t rowStart2, uint32_t hiddenSize, int32_t srcEpIdx,
                                     int32_t groupIdx, Params const& params, uint64_t gmOffsetA)
    {
        const int32_t rank = RuntimeRank(params);
        // 自身 rank 已在 SendTokensV3 路径 A 中完成搬运（含 scale 分离），跳过
        if (srcEpIdx == rank)
            return;
        // 无 token 时跳过，避免等待永远不会到达的 flag
        if (rows2 == 0)
            return;

        // 等待 dispatch flag：精确匹配 FLAG_VALUE_MAGIC，避免上一轮残留误判
        //   slotIdx = srcEpIdx * expertPerRank + groupIdx
        //   每槽 16 个 int32（= 64B = 一个 cache line）
        __gm__ int32_t* flagAddr = reinterpret_cast<__gm__ int32_t*>(shmem() + peermemInfo.offsetFlag) +
                                   (srcEpIdx * params.expertPerRank + groupIdx) * PeermemInfo::kFlagSlotI32;
        if constexpr (BatchedDispatch) {
            while (true) {
                gm_dcci(flagAddr);
                if (gm_load(flagAddr) == FLAG_VALUE_MAGIC) {
                    break;
                }
            }
        } else {
            // flag 由对端核（同 server gm_store / 跨 server RDMA）写入本 rank GM。
            // 这里用 MTE2 DMA 读轮询，而不是 scalar dcci+load：后者会在 NIC 正在写的
            // 同一条 cache line 上持续做 clean/invalidate，A2 跨 server 场景下会把发送端
            // AI-RDMA QP 的完成拖住（HW tail 长时间不推进，flag WQE 一直不执行）。
            // 产品 A2 dispatch 算子（moe_distribute_dispatch_a2_layered.h）的 flag 等待
            // 同样使用 MTE2 读。
            // 等待按经过时间设上限，不用轮询次数：A2 的 GetSystemCycle() 为 50 cycles/us，
            // 5s 远大于正常 flag 到达延迟，且远小于 234s 设备看门狗；每 1024 轮检查一次。
            constexpr uint32_t kFlagWaitCheckMask = 1024U - 1U;
            constexpr uint64_t kFlagWaitTimeoutCycles = 5ULL * 1000ULL * 1000ULL * 50ULL;
            const int64_t flagWaitStart = AscendC::GetSystemCycle();
            uint32_t flagWaitPoll = 0;
            AscendC::LocalTensor<int32_t> flagPollUb = resource.ubBuf.template GetBufferByByte<int32_t>(4 * UB_ALIGN);
            AscendC::GlobalTensor<int32_t> flagSlotGm;
            flagSlotGm.SetGlobalBuffer(flagAddr);
            while (true) {
                AscendC::DataCopy(flagPollUb, flagSlotGm, PeermemInfo::kFlagPayloadI32);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
                const int32_t flagVal = flagPollUb.GetValue(0);
                if (flagVal == FLAG_VALUE_MAGIC) {
                    break;
                }
                if (flagVal == FLAG_VALUE_MAGIC + 1) {
                    return;
                }
                if ((++flagWaitPoll & kFlagWaitCheckMask) == 0U &&
                    static_cast<uint64_t>(AscendC::GetSystemCycle() - flagWaitStart) >= kFlagWaitTimeoutCycles) {
                    AscendC::Trap();
                    return;
                }
            }
        }

        if (UseSenderMsd(params) && srcEpIdx / SERVER_RANK_SIZE_A2 == serverId_) {
            AscendC::PipeBarrier<PIPE_ALL>();
            return;
        }

        uint32_t copyInNum = hiddenSize + RoutingPadding(params);
        AscendC::GlobalTensor<T> gmDstA;
        gmDstA.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(shmem() + peermemInfo.offsetA));
        AscendC::PipeBarrier<PIPE_ALL>();
        FetchAndPreprocessInt8ToInt4(gmOffsetA, gmPerTokenScale1[rowStart2], gmDstA[rowStart2 * copyInNum], rows2,
                                     hiddenSize, params.gmmA1PreRowStride);
        return;
    }

    CATLASS_DEVICE
    void CombineSetFlag() {}

    CATLASS_DEVICE
    void DispatchAndCombine(Params const& params)
    {
        icache_preload(ICACHE_PRELOAD_LINES);
        exceptionDump_.Dump(shmem() + peermemInfo.offsetPeerTokenPerExpert,
                            static_cast<size_t>(paddedExpertNumAligned) * params.expertPerRank *
                                static_cast<uint32_t>(shmem.RankSize()) * sizeof(int32_t));
        shmem.DumpSyncRegions(exceptionDump_);
        int64_t localTokenPerExpertOffset =
            peermemInfo.offsetPeerTokenPerExpert + tokenPerExpertLayout(RuntimeRank(params), 0, 0) * sizeof(int32_t);
        int64_t localTokenPerExpertWinOutOffset =
            peermemInfo.offsetPeerTokenPerExpert + tokenPerExpertLayout(RuntimeRank(params), 0, 0) * sizeof(int32_t);
        GM_ADDR localTokenPerExpert = shmem.windowsOutAddr() + localTokenPerExpertWinOutOffset;

        const int64_t bs = params.problemShape.m();
        const int64_t hidden = params.problemShape.k();
        const int64_t topK = params.topK;
        const int64_t numChunks = (bs + PermuteChunk(params) - 1) / PermuteChunk(params);
        const uint32_t n2 = params.problemShape.k(); // unpermute 的隐藏维（GMM2 输出列数）

        // 输入 X 始终为 bf16；量化路径输出 int8（per-token），非量化输出 bf16 透传。
        using PrologueSrc = bfloat16_t;
        using PrologueDst = std::conditional_t<kRoutingIsQuant, int8_t, bfloat16_t>;
        using ProloguePolicy = std::conditional_t<kRoutingIsQuant, MoePermute::MoePermuteProloguePolicy,
                                                  MoePermute::MoePermutePrologueNonQuantPolicy>;

        // ExpertTokenNums 输出清零（进入 chunk 循环前一次，后续按 chunk 累加）
        {
            AscendC::GlobalTensor<int32_t> gmExpertTokenNums;
            gmExpertTokenNums.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.ptrExpertTokenNums));
            if (coreIdx == 0) {
                AscendC::LocalTensor<int32_t> zeroBuf = resource.ubBuf.template GetBufferByByte<int32_t>(0);
                AscendC::Duplicate(zeroBuf, 0, params.expertPerRank);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::DataCopyPad(gmExpertTokenNums[0], zeroBuf,
                                     {1, static_cast<uint16_t>(params.expertPerRank * sizeof(int32_t)), 0, 0});
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            }
        }

        for (int64_t chunkIdx = 0; chunkIdx < numChunks; ++chunkIdx) {
            const int64_t tokenBase = chunkIdx * PermuteChunk(params);
            const int64_t chunkTokens =
                (bs - tokenBase) < PermuteChunk(params) ? (bs - tokenBase) : PermuteChunk(params);
            // cumsumMM 每 chunk 指向独立缓冲区（不复用，规避跨 chunk cache 陈旧读）
            cumsumMM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(
                workspaceInfo.ptrcumsumMM +
                chunkIdx * static_cast<int64_t>(paddedExpertNumAligned) * params.EP * sizeof(int32_t)));

            // 发送侧 Permute Prologue（chunk 范围，x_active_mask 融合于 prologue，
            // 不改写输入 topk_ids）
            // 完成 sort + histogram + gather(quant) + 跨 rank 归约。prologue 的 UB 按单
            // chunk（MAX_TOKENS=1024）分配，numTokens 必须按 chunk 传入，不得使用全量 bs。
            exceptionDump_.UpdateStage(MC2MegaMoeAdump::Stage::MOE_INIT_ROUTING);
            {
                // prologue 模板首参为 UB 流水级数，后续为 Policy/Src/Dst
                if constexpr (BatchedDispatch) {
                    typename MoePermute::MoePermutePrologue<PROLOGUE_UB_STAGES, ProloguePolicy, PrologueSrc,
                                                            PrologueDst, false, 8192, BYTE_PER_BLK>::Params
                        prologueParams(chunkTokens, hidden, topK, params.expertPerRank * params.EP);
                    MoePermute::MoePermutePrologue<PROLOGUE_UB_STAGES, ProloguePolicy, PrologueSrc, PrologueDst, false,
                                                   8192, BYTE_PER_BLK>
                        prologue(resource, prologueParams);
                    AscendC::GlobalTensor<PrologueSrc> gmPermX;
                    gmPermX.SetGlobalBuffer(reinterpret_cast<__gm__ PrologueSrc*>(
                        reinterpret_cast<GM_ADDR>(params.ptrA) + tokenBase * hidden * sizeof(PrologueSrc)));
                    AscendC::GlobalTensor<int32_t> gmPermTopkIds;
                    gmPermTopkIds.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(
                        reinterpret_cast<GM_ADDR>(params.expertIdx) + tokenBase * topK * sizeof(int32_t)));
                    // winOut A 按单 chunk 分配，逐 chunk 复用，源基址不变
                    AscendC::GlobalTensor<PrologueDst> gmPermExpandX;
                    gmPermExpandX.SetGlobalBuffer(
                        reinterpret_cast<__gm__ PrologueDst*>(shmem.windowsOutAddr() + peermemInfo.offsetWinOutA));
                    AscendC::GlobalTensor<int32_t> gmTokenPerExpert;
                    gmTokenPerExpert.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(localTokenPerExpert));
                    // expandedRowIdx 按单 chunk 尺寸分配，逐 chunk 复用（值为 chunk 局部 permute 行号）
                    AscendC::GlobalTensor<int32_t> gmPermExpandRowIdx;
                    gmPermExpandRowIdx.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspaceInfo.expandedRowIdx));
                    prologue(gmPermX, gmPermTopkIds, gmPermExpandX, gmPermExpandRowIdx, gmTokenPerExpert,
                             params.ptrWorkspace,
                             params.ptrXActiveMask != nullptr ? params.ptrXActiveMask + tokenBase : nullptr);
                } else {
                    typename MoePermute::MoePermutePrologue<PROLOGUE_UB_STAGES, ProloguePolicy, PrologueSrc,
                                                            PrologueDst>::Params prologueParams(chunkTokens, hidden,
                                                                                                topK,
                                                                                                params.expertPerRank *
                                                                                                    params.EP);
                    MoePermute::MoePermutePrologue<PROLOGUE_UB_STAGES, ProloguePolicy, PrologueSrc, PrologueDst>
                        prologue(resource, prologueParams);
                    AscendC::GlobalTensor<PrologueSrc> gmPermX;
                    gmPermX.SetGlobalBuffer(reinterpret_cast<__gm__ PrologueSrc*>(
                        reinterpret_cast<GM_ADDR>(params.ptrA) + tokenBase * hidden * sizeof(PrologueSrc)));
                    AscendC::GlobalTensor<int32_t> gmPermTopkIds;
                    gmPermTopkIds.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(
                        reinterpret_cast<GM_ADDR>(params.expertIdx) + tokenBase * topK * sizeof(int32_t)));
                    // winOut A 按单 chunk 分配，逐 chunk 复用，源基址不变
                    AscendC::GlobalTensor<PrologueDst> gmPermExpandX;
                    gmPermExpandX.SetGlobalBuffer(
                        reinterpret_cast<__gm__ PrologueDst*>(shmem.windowsOutAddr() + peermemInfo.offsetWinOutA));
                    AscendC::GlobalTensor<int32_t> gmTokenPerExpert;
                    gmTokenPerExpert.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(localTokenPerExpert));
                    // expandedRowIdx 按单 chunk 尺寸分配，逐 chunk 复用（值为 chunk 局部 permute 行号）
                    AscendC::GlobalTensor<int32_t> gmPermExpandRowIdx;
                    gmPermExpandRowIdx.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspaceInfo.expandedRowIdx));
                    prologue(gmPermX, gmPermTopkIds, gmPermExpandX, gmPermExpandRowIdx, gmTokenPerExpert,
                             params.ptrWorkspace,
                             params.ptrXActiveMask != nullptr ? params.ptrXActiveMask + tokenBase : nullptr);
                }
            }
            AscendC::SyncAll<true>();
            // counts all-gather + cumsum（chunk 局部，覆盖重算）
            exceptionDump_.UpdateStage(MC2MegaMoeAdump::Stage::ALLGATHER_TOKEN_PER_EXPERT);
            CrossRankSyncAndlocalTokenPerExpertAllGatherAndGetSumPreRankV2(params, localTokenPerExpertOffset);

            if (coreIdx == 0) {
                exceptionDump_.UpdateStage(MC2MegaMoeAdump::Stage::CUMSUM_TOKEN_PER_EXPERT);
                GetCumsumForMMAIV(tokenPerExpert, cumsumMM, params.expertPerRank, params.EP);
            }

            AscendC::SyncAll<true>();
            // 路由表刷新到本核 UB：此后 dispatch/combine/epilogue 的标量读取不再直接读 GM
            CacheRouteTablesToUb(params);

            // ExpertTokenNums 按 chunk 累加（core0，UB 读改写）
            {
                AscendC::GlobalTensor<int32_t> gmExpertTokenNums;
                gmExpertTokenNums.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.ptrExpertTokenNums));
                if (coreIdx == 0) {
                    const int32_t rank = RuntimeRank(params);
                    AscendC::LocalTensor<int32_t> ubSum = resource.ubBuf.template GetBufferByByte<int32_t>(0);
                    AscendC::LocalTensor<int32_t> ubOut = resource.ubBuf.template GetBufferByByte<int32_t>(
                        AlignUp(params.expertPerRank * sizeof(int32_t), UB_ALIGN));
                    AscendC::DataCopyPad(ubSum, cumsumMM[tokenPerExpertLayout(params.EP - 1, rank, 0)],
                                         {1, static_cast<uint16_t>(params.expertPerRank * sizeof(int32_t)), 0, 0}, {});
                    AscendC::DataCopyPad(ubOut, gmExpertTokenNums[0],
                                         {1, static_cast<uint16_t>(params.expertPerRank * sizeof(int32_t)), 0, 0}, {});
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                    AscendC::Add(ubOut, ubOut, ubSum, params.expertPerRank);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                    AscendC::DataCopyPad(gmExpertTokenNums[0], ubOut,
                                         {1, static_cast<uint16_t>(params.expertPerRank * sizeof(int32_t)), 0, 0});
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
                }
            }
            AscendC::SyncAll<true>();

            uint16_t syncgmm1Idx = 0;
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(syncgmm1Idx / CROSS_CORE_FLAG_MAX_SET_COUNT);
            syncgmm1Idx++;

            uint32_t prevGroupSum1Arr[MAX_RANK_PER_CORE] = {0};
            uint32_t dequantSum1 = 0;
            uint32_t dequantSum2 = 0;
            uint32_t prevGroupSum2 = 0;
            nSyncSwiglu = 0;
            dequantSum[0] = 0;
            uint32_t dequantSumTemp = 0;
            icache_preload(ICACHE_PRELOAD_LINES);
            exceptionDump_.UpdateStage(MC2MegaMoeAdump::Stage::DISPATCH);
            if constexpr (BatchedDispatch) {
                for (uint32_t first = 0; first < params.expertPerRank; first += params.dispatchExpertBatch) {
                    const uint32_t groups =
                        min(params.dispatchExpertBatch, static_cast<uint32_t>(params.expertPerRank) - first);
                    if (!UseDecodeFastPath<BatchedDispatch>(params) ||
                        Detail::HasDispatchBatchRows(params, first, groups, ubTpeSendCache, ubTpeRecvCache,
                                                     expertPerRankAligned)) {
                        SendExpertBatch(params, first, groups);
                        AscendC::SyncAll<true>();
                        PublishExpertBatch(params, RuntimeRank(params), first, groups);
                        RecvExpertBatch(params, first, groups);
                        AscendC::SyncAll<true>();
                    }
                    for (uint32_t groupIdx = first; groupIdx < first + groups; ++groupIdx) {
                        const uint32_t currentMRecv =
                            cumsumMM(tokenPerExpertLayout(params.EP - 1, RuntimeRank(params), groupIdx));
                        if (!UseDecodeFastPath<BatchedDispatch>(params) || groupIdx == first) {
                            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(syncgmm1Idx / CROSS_CORE_FLAG_MAX_SET_COUNT);
                            syncgmm1Idx++;
                        }
                        AccumulateSingleDequantRows(currentMRecv, params.maxOutputSize, dequantSumTemp);
                        if (IsActivationBoundary<BatchedDispatch>(params, groupIdx)) {
                            nSyncSwiglu++;
                            dequantSum[nSyncSwiglu] = dequantSumTemp;
                        }
                    }
                }
            } else {
                for (int32_t groupIdx = 0; groupIdx < params.expertPerRank; ++groupIdx) {
                    uint32_t currentMRecv = static_cast<uint32_t>(
                        ReadCumsum(tokenPerExpertLayout(params.EP - 1, RuntimeRank(params), groupIdx)));
                    uint32_t currentMSend = 0;
                    uint32_t copyInNum = params.problemShape.k() + RoutingPadding(params);
                    for (int32_t dstEpIdx = static_cast<int32_t>(coreIdx); dstEpIdx < static_cast<int32_t>(params.EP);
                         dstEpIdx += static_cast<int32_t>(coreNum)) {
                        uint32_t arrIdx = static_cast<uint32_t>(dstEpIdx) / coreNum;
                        uint32_t rowStart =
                            prevGroupSum1Arr[arrIdx] + ((RuntimeRank(params) == 0) ?
                                                            0u :
                                                            static_cast<uint32_t>(ReadCumsum(tokenPerExpertLayout(
                                                                RuntimeRank(params) - 1, dstEpIdx, groupIdx))));
                        currentMSend =
                            static_cast<uint32_t>(ReadCumsum(tokenPerExpertLayout(params.EP - 1, dstEpIdx, groupIdx)));
                        if (rowStart >= params.maxOutputSize) {
                            PostSkipFlag(dstEpIdx, groupIdx, params);
                        }
                        if (rowStart < params.maxOutputSize) {
                            // chunk 模式：路由表读走本核 UB 缓存（CacheRouteTablesToUb 每 chunk 刷新）
                            uint32_t rows = static_cast<uint32_t>(
                                ubTpeSendCache.GetValue(dstEpIdx * params.expertPerRank + groupIdx));
                            if (rowStart + rows > params.maxOutputSize) {
                                rows = params.maxOutputSize - rowStart;
                            }
                            uint32_t rowSrc = static_cast<uint32_t>(
                                ubPreSumDispatchCache.GetValue(dstEpIdx * params.expertPerRank + groupIdx));
                            AscendC::GlobalTensor<ElementABefore> gmSrcA;
                            gmSrcA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementABefore*>(shmem.windowsOutAddr() +
                                                                                            peermemInfo.offsetWinOutA));
                            int64_t gmSrcOffset = static_cast<int64_t>(rowSrc) * copyInNum;
                            AscendC::GlobalTensor<ElementABefore> gmRemoteDstA;
                            gmRemoteDstA.SetGlobalBuffer(
                                reinterpret_cast<__gm__ ElementABefore*>(shmem(0, dstEpIdx) + peermemInfo.offsetA));
                            int64_t gmDstOffset = static_cast<int64_t>(rowStart) * copyInNum;
                            SendTokensV3<ElementABefore>(gmRemoteDstA[gmDstOffset], gmSrcA[gmSrcOffset],
                                                         static_cast<int32_t>(rows), params.problemShape.k(), dstEpIdx,
                                                         groupIdx, static_cast<int32_t>(rowStart), params,
                                                         static_cast<uint64_t>(rowStart) * params.gmmA1PreRowStride);
                        }
                        prevGroupSum1Arr[arrIdx] += currentMSend;
                    }

                    AscendC::SyncAll<true>();

                    // RECV 阶段：各 core 并行等 epoch flag 后做 scale 分离
                    for (int32_t srcEpIdx = static_cast<int32_t>(coreIdx); srcEpIdx < static_cast<int32_t>(params.EP);
                         srcEpIdx += static_cast<int32_t>(coreNum)) {
                        uint32_t rowStart2 =
                            prevGroupSum2 + ((srcEpIdx == 0) ? 0u :
                                                               static_cast<uint32_t>(ReadCumsum(tokenPerExpertLayout(
                                                                   srcEpIdx - 1, RuntimeRank(params), groupIdx))));
                        if (rowStart2 < params.maxOutputSize) {
                            uint32_t rows2 = static_cast<uint32_t>(
                                ubTpeRecvCache.GetValue(srcEpIdx * expertPerRankAligned + groupIdx));
                            if (rows2 + rowStart2 > params.maxOutputSize) {
                                rows2 = params.maxOutputSize - rowStart2;
                            }
                            RecvTokensV3<ElementABefore>(static_cast<int32_t>(rows2), static_cast<int32_t>(rowStart2),
                                                         params.problemShape.k(), srcEpIdx, groupIdx, params,
                                                         static_cast<uint64_t>(rowStart2) * params.gmmA1PreRowStride);
                        }
                    }
                    AscendC::SyncAll<true>();

                    prevGroupSum2 += currentMRecv;

                    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(syncgmm1Idx / CROSS_CORE_FLAG_MAX_SET_COUNT);
                    syncgmm1Idx++;

                    AccumulateSingleDequantRows(currentMRecv, params.maxOutputSize, dequantSumTemp);

                    if (IsActivationBoundary<BatchedDispatch>(params, groupIdx)) {
                        nSyncSwiglu++;
                        dequantSum[nSyncSwiglu] = dequantSumTemp;
                    }
                }
            }
            if constexpr (BatchedDispatch) {
                RunPostprocess<true>(params);
            } else {
                RunPostprocess<false>(params);
            }
            AscendC::SyncAll<true>();
            if constexpr (BatchedDispatch) {
                exceptionDump_.UpdateStage(MC2MegaMoeAdump::Stage::RESET_TOKEN_PER_EXPERT);
                ResetTokenPerExpert(params, params.EP * paddedExpertNumAligned);
                AscendC::SyncAll<true>();
            }
            // 先屏障再清零：屏障写与 dispatch flag 走同一个 QP、且排在其后，
            // 屏障通过即表示本 chunk 对端的所有 flag（含跳过标记）都已执行完，
            // 之后的清零不会和迟到的旧标记竞争。
            exceptionDump_.UpdateStage(MC2MegaMoeAdump::Stage::CROSS_RANK_SYNC);
            {
                // 3 * UB_ALIGN scratch: payload + rdma doorbell + rdma head.
                // UB at offset 0 is unused at this point in the kernel.
                AscendC::LocalTensor<int32_t> ctrBuffer = resource.ubBuf.template GetBufferByByte<int32_t>(0);
                shmem.CrossRankSync(ctrBuffer);
            }

            // KernelMoeTokenUnpermute uses get_block_num() (= AIC tile count), not full AIV count.
            // Use coreNum/2 for tiling and run only on one subblock to match blockIdx/blockNum semantics.
            // 按 chunk 执行：D 区按 chunk 局部复用（CrossRankSync 已保证对端 combine 写到达），
            // expandedRowIdx 为 chunk 局部 permute 行号，probs / 输出按 tokenBase 偏移。
            if (BatchedDispatch || get_subblockid() == 1) {
                exceptionDump_.UpdateStage(MC2MegaMoeAdump::Stage::UNPERMUTE);
                MoeTokenUnpermuteTilingData tilingData;
                MoeTokenUnpermuteTiling(chunkTokens * topK, n2, topK, tilingData,
                                        BatchedDispatch ? coreNum : coreNum / AIV_PER_AIC);
                KernelMoeTokenUnpermute<ElementD2, int32_t, float, true> kernelMoeTokenUnpermuteOp;
                kernelMoeTokenUnpermuteOp.Init(
                    shmem() + peermemInfo.offsetD, workspaceInfo.expandedRowIdx,
                    reinterpret_cast<GM_ADDR>(reinterpret_cast<__gm__ float*>(params.probs) + tokenBase * topK),
                    reinterpret_cast<GM_ADDR>(params.ptrOutput + tokenBase * hidden), &tilingData, BatchedDispatch);
                kernelMoeTokenUnpermuteOp.Process();
            }

            // chunk 边界：unpermute 读完 D/expandedRowIdx 后才允许下一 chunk 覆盖
            AscendC::SyncAll<true>();
            if constexpr (!BatchedDispatch) {
                exceptionDump_.UpdateStage(MC2MegaMoeAdump::Stage::RESET_TOKEN_PER_EXPERT);
                ResetTokenPerExpert(params, params.EP * paddedExpertNumAligned);
                AscendC::SyncAll<true>();
                // 第二次屏障：所有 rank 都清零完成后才允许进入下一 chunk 的发送，
                // 避免下一 chunk 的 flag 在别人清零前写入、又被那次清零抹掉。
                {
                    AscendC::LocalTensor<int32_t> ctrBuffer = resource.ubBuf.template GetBufferByByte<int32_t>(0);
                    shmem.CrossRankSync(ctrBuffer);
                }
            }
        } // for chunkIdx
    }

    template <bool EnableBatched>
    CATLASS_DEVICE void RunPostprocess(Params const& params)
    {
        using ActivationEpilogue = typename Epilogue::Block::SelectPostprocess<BlockEpilogue1, EnableBatched>::Type;
        using CombineEpilogue = typename Epilogue::Block::SelectPostprocess<BlockEpilogue2, EnableBatched>::Type;
        const uint32_t n2 = params.problemShape.k();
        typename CombineEpilogue::Params epilogueParams{
            static_cast<int32_t>(params.EP),
            static_cast<int32_t>(params.expertPerRank),
            static_cast<int32_t>(RuntimeRank(params)),
            reinterpret_cast<__gm__ int32_t*>(shmem() + peermemInfo.offsetPeerTokenPerExpert),
            params.layoutD2,
            static_cast<int32_t>(n2),
            static_cast<int32_t>(L1TileShape::N),
            shmem,
            peermemInfo.offsetD,
            tokenPerExpertLayout};
        epilogueParams.offsetWinOutD = peermemInfo.offsetWinOutD;
        // epilogue 内 tokenPerExpert / preSumBeforeRank 标量读取走本核 UB 缓存
        // （每 chunk 已由 CacheRouteTablesToUb 刷新，规避跨 chunk GM 标量读 D-Cache 陈旧）
        epilogueParams.useUbRouteCache = true;
        epilogueParams.ubRouteStride = expertPerRankAligned;
        epilogueParams.ubTokenPerExpert = ubTpeRecvCache;
        epilogueParams.ubPreSumBeforeRank = ubPreSumCombineCache;

        uint32_t n = params.problemShape.n();
        CombineEpilogue blockEpilogue2(resource, epilogueParams);
        ActivationEpilogue blockEpilogue1(resource, n);

        exceptionDump_.UpdateStage(MC2MegaMoeAdump::Stage::SWIGLU);
        for (int32_t syncIdx = 0; syncIdx < nSyncSwiglu; syncIdx++) {
            AscendC::CrossCoreWaitFlag<0x2>(SYNCFLAGC2V);
            AscendC::SyncAll<true>();
            uint32_t curRowNum = dequantSum[syncIdx + 1] - dequantSum[syncIdx];
            if (curRowNum > 0) {
                uint32_t rowStartThisCore = dequantSum[syncIdx];
                MatrixCoord offsetC{rowStartThisCore, 0};
                MatrixCoord shapeC{curRowNum, params.problemShape.n()};
                LayoutC layoutC{curRowNum, params.gmmOutPreRowStride};
                int64_t gmOffsetC = layoutC.GetOffset(offsetC);
                int64_t gmOffsetD = params.layoutD1.GetOffset(offsetC);
                blockEpilogue1(gmC[gmOffsetC * 2], shapeC, gmPerTokenScale1[rowStartThisCore],
                               reinterpret_cast<__gm__ float*>(params.ptrBias1), gmA2I4_I8[gmOffsetD],
                               cumsumMM[tokenPerExpertLayout(params.EP - 1, RuntimeRank(params), 0) -
                                        tokenPerExpertLayout(0, params.EP - 1, 0)],
                               rowStartThisCore, gmPerTokenScale2[rowStartThisCore], params.expertPerRank, params.EP,
                               RuntimeRank(params), params.listLen, resource, params.epilogueCoreNum,
                               params.swigluLimit, params.activationCode, params.activationParams1,
                               params.activationParams2, params.gmmOutPreRowStride);
            }
            AscendC::SyncAll<true>();
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNCFLAGV2C);
        }
        blockEpilogue1.Finalize();
        exceptionDump_.UpdateStage(MC2MegaMoeAdump::Stage::COMBINE);

        CombineV2(params, blockEpilogue2);
    }

    template <class CombineEpilogue>
    CATLASS_DEVICE void CombineV2(Params const& params, CombineEpilogue& blockEpilogue)
    {
        blockEpilogue.InitFlag();
        BlockScheduler blockScheduler;
        int32_t syncLoopIdx = 0;
        uint32_t startCoreIdx = 0;
        uint32_t aicCoreNum = coreNum / AIV_PER_AIC;
        uint32_t aicCoreIdx = get_block_idx();
        uint32_t aivSubCoreIdx = get_subblockid();
        uint32_t preSrcExpertSum = 0;
        uint32_t n2 = params.problemShape.k();
        uint32_t k2 = params.problemShape.n() / 2;
        AscendC::LocalTensor<uint64_t> rdmaUbLocal =
            resource.ubBuf.template GetBufferByByte<uint64_t>(RDMA_UB_OFFSET_BYTES);
        AscendC::LocalTensor<uint32_t> rdmaUbLocalHead =
            resource.ubBuf.template GetBufferByByte<uint32_t>(RDMA_UB_OFFSET_BYTES + UB_ALIGN);
        AscendC::GlobalTensor<ElementD2> gmLocalWindowsOut;
        gmLocalWindowsOut.SetGlobalBuffer(
            reinterpret_cast<__gm__ ElementD2*>(shmem.windowsOutAddr() + peermemInfo.offsetWinOutD));
        int64_t gmGroupOffsetC = 0;
        uint32_t aivCoreNum = coreNum;
        uint32_t aivCoreIdx = coreIdx;

        icache_preload(ICACHE_PRELOAD_LINES);
        for (uint32_t groupIdx = 0; groupIdx < params.expertPerRank; ++groupIdx) {
            uint32_t currentExpertM = cumsumMM(tokenPerExpertLayout(params.EP - 1, RuntimeRank(params), groupIdx));
            if (preSrcExpertSum >= params.maxOutputSize) {
                currentExpertM = 0;
            } else if (preSrcExpertSum + currentExpertM > params.maxOutputSize) {
                currentExpertM = params.maxOutputSize - preSrcExpertSum;
            }
            currentExpertM = currentExpertM * GATE_UP_ROWS_PER_TOKEN;

            GemmCoord inGroupProblemShape{currentExpertM, n2, k2}; // M N K
            blockScheduler.Update(inGroupProblemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
            uint32_t coreLoops = blockScheduler.GetCoreLoops();
            uint32_t startLoopIdx =
                ((aicCoreIdx < startCoreIdx) ? (aicCoreIdx + aicCoreNum) : aicCoreIdx) - startCoreIdx;

            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += aicCoreNum) {
                GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = blockScheduler.GetActualBlockShape(blockCoord);
                int32_t m0 = 32;
                int32_t m_rows = (actualBlockShape.m() + m0 - 1) / m0;
                int32_t aiv_m_rows = m_rows / SUBBLOCK_NUM_PER_AIC;
                if (aivSubCoreIdx == 1 && aiv_m_rows * SUBBLOCK_NUM_PER_AIC < m_rows) {
                    aiv_m_rows += 1;
                }
                uint32_t m_offset = blockCoord.m() * L1TileShape::M; // blockOffset
                if (aivSubCoreIdx == 1) {
                    m_offset += (m_rows / SUBBLOCK_NUM_PER_AIC) * m0;
                }

                for (; syncLoopIdx <= groupIdx; syncLoopIdx++) {
                    int32_t flag_id = syncLoopIdx / CROSS_CORE_FLAG_MAX_SET_COUNT;
                    AscendC::CrossCoreWaitFlag<0x2>(flag_id);
                }

                for (int32_t cur_row = 0; cur_row < aiv_m_rows; cur_row++) {
                    GemmCoord realTileCoord{m_offset, blockCoord.n() * L1TileShape::N, 1};
                    uint32_t actualm = m0;
                    if (aivSubCoreIdx == 1 && cur_row == aiv_m_rows - 1) {
                        actualm = actualBlockShape.m() - (m_rows / SUBBLOCK_NUM_PER_AIC) * m0 - cur_row * m0;
                    }
                    GemmCoord realTileShape{actualm, actualBlockShape.n(), 1};
                    blockEpilogue(gmC2, gmPerTokenScale2, reinterpret_cast<__gm__ float*>(params.ptrBias2),
                                  realTileCoord, realTileShape, groupIdx, preSrcExpertSum * 2,
                                  preSumBeforeRankForCombine, params.listLen);
                    m_offset += m0;
                }
            }

            if (Detail::CanSkipCombineExpertSync<BatchedDispatch>(params.EP, coreNum, DISPATCH_PEER_WORKERS)) {
                // Single-server tiles write disjoint destinations; no RDMA posting follows.
                preSrcExpertSum += currentExpertM / GATE_UP_ROWS_PER_TOKEN;
                startCoreIdx = (startCoreIdx + coreLoops) % aicCoreNum;
                continue;
            }
            AscendC::SyncAll<true>();
            int32_t preSumRankInExpert = 0;
            for (int32_t dstEpIdx = 0; dstEpIdx < params.EP; ++dstEpIdx) {
                // chunk 模式：路由表读走本核 UB 缓存（CacheRouteTablesToUb 每 chunk 刷新）
                int32_t lenRankInExpert = ubTpeRecvCache.GetValue(dstEpIdx * expertPerRankAligned + groupIdx);
                int32_t stRankInExpert = preSumRankInExpert;
                int32_t edRankInExpert = stRankInExpert + lenRankInExpert;
                preSumRankInExpert += lenRankInExpert;
                if (stRankInExpert >= static_cast<int32_t>(currentExpertM)) {
                    break;
                }
                if ((dstEpIdx % coreNum) != static_cast<int32_t>(coreIdx)) {
                    continue;
                }
                if ((dstEpIdx / SERVER_RANK_SIZE_A2) == static_cast<int32_t>(serverId_)) {
                    continue;
                }

                int32_t edData = min(edRankInExpert, static_cast<int32_t>(currentExpertM));
                uint32_t lenData = edData - stRankInExpert;
                if (lenData == 0) {
                    continue;
                }

                AscendC::GlobalTensor<ElementD2> gmRemotePeer;
                __gm__ void* dstPeermemPtr = shmem(peermemInfo.offsetD, dstEpIdx);
                gmRemotePeer.SetGlobalBuffer(reinterpret_cast<__gm__ ElementD2*>(dstPeermemPtr));

                int32_t dstExpertOffset = ubPreSumCombineCache.GetValue(dstEpIdx * params.expertPerRank + groupIdx);
                MatrixCoord srcOffset{preSrcExpertSum + static_cast<uint32_t>(stRankInExpert), 0};
                MatrixCoord dstOffset{static_cast<uint32_t>(dstExpertOffset), 0};
                int64_t gmSrcOffset = params.layoutD2.GetOffset(srcOffset);
                int64_t gmDstOffset = params.layoutD2.GetOffset(dstOffset);
                uint64_t messageLen = static_cast<uint64_t>(lenData) * static_cast<uint64_t>(n2) * sizeof(ElementD2);
                AIVRDMAPostSend((GM_ADDR)gmLocalWindowsOut[gmSrcOffset].GetPhyAddr(),
                                (GM_ADDR)gmRemotePeer[gmDstOffset].GetPhyAddr(), static_cast<uint64_t>(dstEpIdx),
                                messageLen, qp_info_, rdmaUbLocal, rdmaUbLocalHead);
            }
            preSrcExpertSum += currentExpertM / GATE_UP_ROWS_PER_TOKEN;
            startCoreIdx = (startCoreIdx + coreLoops) % aicCoreNum;
        }
        for (; syncLoopIdx < params.expertPerRank; syncLoopIdx++) {
            int32_t flag_id = syncLoopIdx / CROSS_CORE_FLAG_MAX_SET_COUNT;
            AscendC::CrossCoreWaitFlag<0x2>(flag_id);
        }
        blockEpilogue.Finalize();
    }

private:
    struct WorkspaceInfo {
        GM_ADDR ptrPermuteCounts;
        GM_ADDR ptrA;
        GM_ADDR ptrPerTokenScale;
        GM_ADDR ptrcumsumMM;
        GM_ADDR ptrC;
        GM_ADDR ptrC2;
        GM_ADDR ptrPermutedToken;
        GM_ADDR ptrPerTokenScale2;
        GM_ADDR expandedRowIdx;
        GM_ADDR ptrA1Int4;
        GM_ADDR ptrA2Int4;
        GM_ADDR ptrSumBeforeRankForDispatch;
        GM_ADDR ptrSumBeforeRankForCombine;
        __gm__ float* ptrSoftFlagBase;

        CATLASS_DEVICE
        WorkspaceInfo() {}

        CATLASS_DEVICE
        WorkspaceInfo(const Params& params)
        {
            uint32_t k2 = params.problemShape.n() / 2;
            uint32_t n2 = params.problemShape.k();
            uint64_t workspaceOffset = 0;
            // prologue 每核计数区（每核槽位 coreIdx*alignedE，全部核 allgather 读取；
            // 尺寸与 host 侧 128 * alignedE * 4 一致，PERMUTE_MAX_CORES = 128）
            ptrPermuteCounts = params.ptrWorkspace;
            const int64_t alignedE = AlignUp(static_cast<int64_t>(params.expertPerRank) * params.EP, 8);
            workspaceOffset +=
                static_cast<uint64_t>(PERMUTE_MAX_CORES) * static_cast<uint64_t>(alignedE) * sizeof(int32_t);

            // expandedRowIdx 按单 chunk 尺寸分配（逐 chunk 复用）
            expandedRowIdx = params.ptrWorkspace + workspaceOffset;
            const int64_t chunkTokens =
                params.problemShape.m() < PermuteChunk(params) ? params.problemShape.m() : PermuteChunk(params);
            workspaceOffset += AlignUp(chunkTokens, 256) * params.topK * sizeof(int32_t);

            // cumsum 按 chunk 数分配多份（不复用），规避 AIC/AIV 跨 chunk 读取旧 cumsumMM 的
            // cache 陈旧问题。
            // 每份尺寸 = paddedExpertNumAligned * EP int32；host 侧 workspace 分配必须同步放大。
            ptrcumsumMM = params.ptrWorkspace + workspaceOffset;
            const int64_t numChunks = (params.problemShape.m() + PermuteChunk(params) - 1) / PermuteChunk(params);
            uint64_t paddedExpertNumAligned = AlignUp(params.EP * params.expertPerRank + 1, ALIGN_128);
            workspaceOffset += static_cast<uint64_t>(numChunks) * paddedExpertNumAligned * params.EP * sizeof(int32_t);

            ptrPerTokenScale = params.ptrWorkspace + workspaceOffset;
            workspaceOffset += params.maxOutputSize * sizeof(ElementPerTokenScale);

            ptrPerTokenScale2 = params.ptrWorkspace + workspaceOffset;
            workspaceOffset += params.maxOutputSize * sizeof(ElementPerTokenScale);

            // gmm out resues workspace
            ptrC = params.ptrWorkspace + workspaceOffset;  // 7
            ptrC2 = params.ptrWorkspace + workspaceOffset; // 8
            workspaceOffset += params.maxOutputSize * params.gmmOutPreRowStride * sizeof(ElementC) * 2;

            ptrA1Int4 = params.ptrWorkspace + workspaceOffset;
            ptrA2Int4 = params.ptrWorkspace + workspaceOffset;
            workspaceOffset += params.maxOutputSize * (params.problemShape.k() > k2 ? params.problemShape.k() : k2);

            ptrSumBeforeRankForDispatch = params.ptrWorkspace + workspaceOffset;
            workspaceOffset += paddedExpertNumAligned * sizeof(int32_t);

            ptrSumBeforeRankForCombine = params.ptrWorkspace + workspaceOffset;
            workspaceOffset += params.EP * sizeof(int32_t) * params.expertPerRank;

            ptrSoftFlagBase = reinterpret_cast<__gm__ float*>(params.ptrWorkspace + workspaceOffset);
            workspaceOffset += params.EP * sizeof(int32_t) * FLAGSTRIDE;
        }
    };

    struct PeermemInfo {
        int64_t offsetA;
        int64_t offsetPeerPerTokenScale;
        int64_t offsetPeerTokenPerExpert;
        int64_t offsetD;
        // Only used by the A2 BatchWrite path: location inside peer-memory
        // where one int32 epoch-flag per source rank is written by SendTokensV3
        // and polled by RecvTokensV3. On A3 we don't push flags through
        // peer-mem at all, so this stays 0.
        int64_t offsetFlag{0};
        int64_t offsetWinOutA; // A tensor in winOut (dispatch, outgoing tokens)
        int64_t offsetWinOutD; // D tensor in winOut (combine, outgoing FFN results)
        int64_t offsetAllgatherFlag;
        int64_t offsetWinOutPeerPerTokenScale; // per-token scale in winOut

        // 每个 flag 槽占 16 个 int32（= 64B = 1 个 cache line），与 CrossRankSync 一致
        static constexpr int64_t kFlagSlotI32 = 16;
        // 跨 server RDMA 时一次写入的 payload 大小：8 个 int32 = 32B，首 4B 为 magic，其余为 padding
        static constexpr int64_t kFlagPayloadI32 = 8;

        CATLASS_DEVICE
        PeermemInfo() {}

        CATLASS_DEVICE
        PeermemInfo(const Params& params, const HcclShmem<true>& shmem)
        {
            const int64_t EP = params.EP;
            const int64_t E = params.expertPerRank;
            const int64_t M = params.maxOutputSize;
            const int64_t h = params.problemShape.k();
            const int64_t bs = params.problemShape.m();
            const int64_t topK = params.topK;

            // Dispatch Flag: EP×MAX_EXPERTS_PER_RANK(128)×64B，每 slot 占 64B cache line
            // E 取最大值 128，使 flag 区域位置固定，与实际 expertPerRank 无关
            int64_t flagSize = EP * MAX_EXPERTS_PER_RANK * 64;
            // Allgather Flag: EP×64B（与 BF16 统一布局，预留）
            int64_t allgatherFlagSize = EP * 64;
            // PerTokenScaleSize: M × sizeof(float)
            int64_t offsetPerTokenScaleSize = M * sizeof(float);
            // AAfterDispatchSize: dispatch 接收数据区W4A8，M × (h + RoutingPadding(params))，原始token
            int64_t AAfterDispatchSize = M * (h + RoutingPadding(params));

            // 布局：... → winIn TPE → winOut TPE → allgatherFlag → flag（winIn/winOut 共享）→ [CrossRankSync tail]
            int64_t usableEnd =
                static_cast<int64_t>(shmem.SegmentSize()) - static_cast<int64_t>(shmem.TailReservedSize());
            // TPE size: EP × paddedExpertNumAligned × sizeof(int32_t)
            // （与 ResetTokenPerExpert 中 num 一致）
            int64_t tpeSize =
                EP * AlignUp(EP * MAX_EXPERTS_PER_RANK + 1, ALIGN_128) * static_cast<int64_t>(sizeof(int32_t));

            // WinIn: A/scale/D 从前向后；flag 紧贴 usableEnd，allgatherFlag/TPE 从后往前
            offsetA = RESERVED_SPACE_SIZE;
            offsetPeerPerTokenScale = offsetA + AAfterDispatchSize;
            offsetD = offsetPeerPerTokenScale + offsetPerTokenScaleSize;
            offsetFlag = usableEnd - flagSize;
            offsetAllgatherFlag = offsetFlag - allgatherFlagSize;
            offsetPeerTokenPerExpert = offsetAllgatherFlag - tpeSize; // winOut TPE

            // WinOut: A/scale/D 从前向后布局
            // ABeforeDispatchSize: dispatch 发送数据区W4A8，单 chunk（不超过 bs）× topK × (h +
            // RoutingPadding(params))， 逐 chunk 复用，与 host winOut A 区下限公式一致
            int64_t ABeforeDispatchSize =
                (bs < PermuteChunk(params) ? bs : PermuteChunk(params)) * topK * (h + RoutingPadding(params));
            offsetWinOutA = RESERVED_SPACE_SIZE;
            offsetWinOutPeerPerTokenScale = offsetWinOutA + ABeforeDispatchSize;
            offsetWinOutD = offsetWinOutPeerPerTokenScale + offsetPerTokenScaleSize;
        }
    };

    Arch::Resource<ArchTag> resource;

    uint32_t coreIdx;
    uint32_t coreNum;
    uint32_t nSyncSwiglu;

    Params params;
    WorkspaceInfo workspaceInfo;
    PeermemInfo peermemInfo;

    uint32_t dequantSum[16] = {0};

    AscendC::GlobalTensor<ElementABefore> gmA;
    AscendC::GlobalTensor<ElementC> gmC;
    AscendC::GlobalTensor<ElementScale> gmS;

    AscendC::GlobalTensor<int4b_t> gmA1I4;
    AscendC::GlobalTensor<int8_t> gmA1I4_I8;
    AscendC::GlobalTensor<int8_t> gmB1_I8;
    AscendC::GlobalTensor<int4b_t> gmA2I4;
    AscendC::GlobalTensor<int8_t> gmA2I4_I8;

    AscendC::GlobalTensor<ElementD1> gmPermutedToken;
    AscendC::GlobalTensor<ElementScale> gmS2;
    AscendC::GlobalTensor<ElementC> gmC2;

    AscendC::GlobalTensor<ElementPerTokenScale> gmPerTokenScale1;
    AscendC::GlobalTensor<ElementPerTokenScale> gmPerTokenScale2;

    AscendC::GlobalTensor<int32_t> tokenPerExpert;
    AscendC::GlobalTensor<int32_t> cumsumMM;
    AscendC::GlobalTensor<int32_t> preSumBeforeRankForDispatch;
    AscendC::GlobalTensor<int32_t> preSumBeforeRankForCombine;
    Layout3D tokenPerExpertLayout;
    int32_t paddedExpertNumAligned;

    // 路由表 UB 缓存（仅 AIV 使用）：[TPE 发送视图][TPE 接收视图][preSumDispatch][preSumCombine]
    // 紧凑索引均为 [dstEpIdx * expertPerRank + groupIdx]，每 chunk 由 CacheRouteTablesToUb 刷新
    static constexpr int32_t kRouteCacheBufCnt = 4; // 路由表 4 份独立 UB 缓存：send/recv 的 tokenPerExpert 与 preSum
    int32_t routeCacheNum = 0;
    int32_t expertPerRankAligned = 0;
    AscendC::LocalTensor<int32_t> ubTpeSendCache;
    AscendC::LocalTensor<int32_t> ubTpeRecvCache;
    AscendC::LocalTensor<int32_t> ubPreSumDispatchCache;
    AscendC::LocalTensor<int32_t> ubPreSumCombineCache;

    HcclShmem<true> shmem;

    // A2 state.
    __gm__ HcclAiRMAInfo* qp_info_{nullptr};
    int32_t serverId_{0};

    uint32_t routingPadding_{ALIGN_512};
    AscendC::LocalTensor<int32_t> ubBatchSendBase, ubBatchRecvBase;
    static constexpr bool kRoutingIsQuant = true;
    // ExceptionDump引擎：记录执行阶段时间戳，并提供Dump接口由host侧dump指定GM地址内容。
    MC2MegaMoeAdump::ExceptionDumpEngine<kRoutingIsQuant, ArchTag> exceptionDump_;
};
} // namespace Catlass::Gemm::Kernel

#endif // MEGA_MOE_KERNEL_A2_BF16_W4A8_HPP
