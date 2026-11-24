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
 * \file mega_moe_kernel_a3.hpp
 * \brief
 */

#ifndef MEGA_MOE_KERNEL_A3_HPP
#define MEGA_MOE_KERNEL_A3_HPP

// Guard against 3rdparty copy_l0c_to_gm.hpp so the custom PER_CHANNEL
// specialization in copy_l0c_to_gm_custom.hpp takes precedence.
#define CATLASS_GEMM_TILE_COPY_L0C_TO_GM_HPP
#include <limits>
#include <type_traits>
#include "kernel_operator.h"

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
#include "utils/block_mmad_preload_async_fixpipe_quant.hpp"
#include "utils/copy_gm_to_l1_custom.hpp"
#include "utils/block_epilogue_w4a8post_pertoken_v2.hpp"
#include "utils/block_epilogue_w4a8post_pertoken_swiglu.hpp"
#include "utils/block_epilogue_pertoken_v2.hpp"
#include "utils/block_epilogue_pertoken_swiglu.hpp"
#include "utils/block_epilogue_pertoken_row.hpp"
#include "utils/hccl_shmem.hpp"
#include "utils/routing_count_utils.hpp"
#include "utils/const_args.hpp"
#include "utils/block_epilogue_pertoken_swiglu.hpp"
#include "utils/block_epilogue_w4a8post_pertoken_swiglu.hpp"
#include "utils/block_epilogue_w4a8post_pertoken_v2.hpp"
#include "mega_moe_tiling_a2a3.h"
#include "utils/layout3d.hpp"

#include "moe_permute_prologue/moe_permute_prologue.hpp"
#include "moe_permute_prologue/moe_mask_routing.hpp"
#include "moe_permute_prologue/moe_permute_prologue_dedup.hpp"
#include "utils/block_epilogue_w4a8post_pertoken_dedup.hpp"
#include "unpermute/mc2_mega_moe_moe_token_unpermute.h"
#include "mega_moe_shared_schedule.h"
#include "utils/get_tensor_addr.hpp"
#include "utils/weight_l2_prefetch.hpp"
#include "mega_moe_exception_dump_policy.h"
#include "a2_a3_copy_helpers.hpp"
#include "a2_routing_helpers.hpp"

namespace Catlass::Gemm::Kernel {

using namespace AscendC;
using namespace Mc2Tiling;

constexpr uint16_t SYNCFLAGC2V = 9;
constexpr uint16_t SYNCFLAGV2C = 10;
// Reserved for the shared pipeline (GMM1 once, GMM2 once or per chunk). Do not introduce
// SyncAll<false> while these signals are live (it also uses flags 11-13).
// Input-ready and activation-ready reuse V2C=12 in strictly ordered phases;
// flag 14 remains exclusively owned by SyncAll<true> on AIV.
constexpr uint16_t SYNCFLAG_SHARED_GMM1_C2V = 11;
constexpr uint16_t SYNCFLAG_SHARED_GMM1_V2C = 12;
constexpr uint16_t SYNCFLAG_SHARED_GMM2_C2V = 13;
constexpr uint16_t SYNCFLAG_SHARED_INPUT_V2C = SYNCFLAG_SHARED_GMM1_V2C;
constexpr int32_t UB_MOVE_NUM = 16 * 1024;

constexpr int64_t PERMUTE_CHUNK = Mc2Tiling::MEGA_MOE_PERMUTE_CHUNK;
static_assert(PERMUTE_CHUNK == ::PERMUTE_CHUNK);
static_assert(PERMUTE_CHUNK == MegaMoeSharedGmm2::CHUNK, "Shared GMM2 must follow routed token chunks");
// 相邻 chunk 通过 AIV 屏障及轮级握手串行化，因此 C2V/V2C 固定占用 flag 9/10，
// 每 chunk 复用同一组 flag（syncgmm 流 0-8、C2V=9、V2C=10、GMM2-combine 0-8）。
// 注意 mode0/1/2 下 flagId 仅 0-15，超出截取低 4bit；若改为按 chunk 偏移必须满足
// C2V(chunk n) != V2C(chunk m) 且不落入 syncgmm 流 0-8 与共享专家的 11-13。
constexpr uint16_t PERMUTE_CHUNK_FLAG_STRIDE = 0;
// prologue 每核计数区按 AIV 核数上限分配（host 侧 workspace 须按实际 aivNum 分配且 <= 此值）
constexpr int64_t PERMUTE_MAX_CORES = 128;

// Atlas A3 (910_93) kernel.
// Mask 接收状态由 DispatchAndCombine 写入、CopyMaskRows 读取，非 Mask 实例使用空基类。
template <bool ENABLED>
struct MegaMoeMaskState {};

template <>
struct MegaMoeMaskState<true> {
    bool maskMultipleRounds_{false};
    int64_t maskRoundIdx_{0};
    uint32_t slotStride_{0};
};

struct MegaMoeDeferredEpilogue {
    template <class Resource, class Params>
    CATLASS_DEVICE MegaMoeDeferredEpilogue(Resource const&, Params const&)
    {}
};

// 默认关闭掩码路由，编译期开关由 Mask 后处理类型提供。
template <class Epilogue, class = void>
struct MegaMoeMaskPolicy {
    static constexpr bool enabled = false;
    static constexpr bool blockedOutput = false;
};

template <class Epilogue>
struct MegaMoeMaskPolicy<Epilogue, std::void_t<decltype(Epilogue::MASK_ROUTING)>> {
    static constexpr bool enabled = Epilogue::MASK_ROUTING;
    static constexpr bool blockedOutput = Epilogue::MASK_BLOCKED_OUTPUT;
};

// DedupOn_：通信零冗余实例；false = 基线实例（if constexpr 裁剪后与基线逐位一致）。
// 仅 W4A8 链双实例化（入口按 tilingData.common.dedupOn 运行时选择）。
template <class BlockMmad_, class BlockScheduler_, class ElementGroupList_, class BlockEpilogue1_,
          class BlockEpilogue2_, class BlockEpilogue3_, class SharedBf16Mmad_, class SharedBf16Epilogue_,
          bool SharedGmm2Bf16_, class SharedBlockMmad_, class SharedBlockEpilogue1_, class SharedBlockEpilogue2_,
          class BlockEpilogueDedup_, bool DedupOn_ = false, bool BatchedDispatch = false>
class MegaMoeKernel : private MegaMoeMaskState<MegaMoeMaskPolicy<BlockEpilogue3_>::enabled> {
public:
    static constexpr bool MASK_ROUTING = MegaMoeMaskPolicy<BlockEpilogue3_>::enabled;
    static constexpr bool BLOCKED_OUTPUT_ = MegaMoeMaskPolicy<BlockEpilogue3_>::blockedOutput;
    using BlockMmad = BlockMmad_;
    using BlockEpilogueDedup = BlockEpilogueDedup_;
    using ArchTag = typename BlockMmad::ArchTag;
    using MaskRouting = MoePermute::MoeMaskRouting<ArchTag>;
    using L1TileShape = typename BlockMmad::L1TileShape;
    static_assert(!MASK_ROUTING || !DedupOn_, "A3 Mask and Dedup use different routed layouts");
    static_assert(!MASK_ROUTING || (L1TileShape::M == Mc2Tiling::MegaMoeMask::L1_TILE_M &&
                                    L1TileShape::N == Mc2Tiling::MegaMoeMask::L1_TILE_N),
                  "A3 Mask tiling must match the kernel L1 tile");
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
    using BlockEpilogue3 = BlockEpilogue3_;
    using ElementD1 = typename BlockEpilogue1::ElementD;
    // Shared scheduling uses the same flags as the routed pipeline, but owns
    // its compute types. In particular, routed MSD must not split shared A8.
    using SharedBlockMmad = SharedBlockMmad_;
    using SharedL1TileShape = typename SharedBlockMmad::L1TileShape;
    using SharedElementA = typename SharedBlockMmad::ElementA;
    using SharedElementB = typename SharedBlockMmad::ElementB;
    using SharedElementC = typename SharedBlockMmad::ElementC;
    using SharedEpilogue1 = std::conditional_t<SharedGmm2Bf16_, SharedBf16Epilogue_, SharedBlockEpilogue1_>;
    using SharedEpilogue2 = SharedBlockEpilogue2_;
    using SharedElementD1 = typename SharedEpilogue1::ElementD;
    static constexpr bool kSharedIsW4A8 = std::is_same_v<SharedElementB, AscendC::int4b_t>;
    static constexpr bool kSharedIsQuant = kSharedIsW4A8 || std::is_same_v<SharedElementB, int8_t>;
    using SharedElementABefore = std::conditional_t<kSharedIsW4A8, int8_t, SharedElementA>;
    static_assert(!SharedGmm2Bf16_ || (std::is_same_v<SharedElementB, int8_t> &&
                                       std::is_same_v<typename SharedEpilogue2::ElementD, bfloat16_t>),
                  "Mixed shared experts require INT8 shared GMM1 and BF16 output");
    using ElementTopkWeights = DTYPE_TOPK_WEIGHTS;
    using LayoutD1 = typename BlockEpilogue1::LayoutD;
    using ElementD2 = typename BlockEpilogue2::ElementD;
    using LayoutD2 = typename BlockEpilogue2::LayoutD;
    using ElementABefore = std::conditional_t<std::is_same_v<ElementA, AscendC::int4b_t>, int8_t, ElementA>;

private:
public:
    /// Parameters structure
    struct Params {
        uint32_t dispatchExpertBatch{1};
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
        // 接收端轮次切分：轮表份数上限（host 下发 recvRoundsMax）。
        // maxOutputSize 语义为单轮接收预算 B；单个发送 chunk 内接收行数 R>B 时，
        // 由 AIV/AIC 各自从 chunk cumsumMM + B 推导轮数（ceil(R/B)）并串行多轮处理。
        uint64_t recvRoundsMax{1};
        // for reuse workspace, gmm1 out preRow Stride use max(n,k)
        // single axis must lower than uint32, layoutC need uint32
        uint32_t gmmOutPreRowStride;
        // GMM1 input reuses the SwiGLU output workspace. Inputs are K-dense
        // inside each expert, while every expert reserves currentM * max(K, K2)
        // elements. The tail gap prevents an earlier expert's K2-wide SwiGLU
        // output from overwriting a later expert's unconsumed GMM1 input.
        uint32_t gmm1InputExpertSlotWidth;
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
        __gm__ SharedElementB* ptrSharedB1{nullptr};
        __gm__ SharedElementB* ptrSharedB2{nullptr};
        __gm__ ElementScale* ptrSharedScale1{nullptr};
        __gm__ ElementScale* ptrSharedScale2{nullptr};
        __gm__ float* ptrSharedBias1{nullptr};
        __gm__ float* ptrSharedBias2{nullptr};
        uint32_t sharedExpertNum{0};
        uint64_t sharedWorkspaceOffset{0};
        uint32_t sharedN{0};
        LayoutB layoutSharedB1{};
        LayoutB layoutSharedWeight2{};
        typename SharedBf16Mmad_::LayoutB layoutSharedB2{}; // BF16 GMM2 in the existing INT8/BF16 path.
        // 算子tiling数据地址（GM），ADump启动时由0核dump到Tiling段
        GM_ADDR tilingGM{nullptr};
        GM_ADDR ptrMaskBuffer{nullptr};
        //--------------
        // 通信零冗余：构造后由 Process 赋值。dedupOn=0 时下列字段全 0、布局与
        // 基线逐位一致；=1 时偏移由 host mega_moe_dedup_layout 计算经 tilingData 透传。
        uint32_t dedupOn{0};
        int64_t offsetExpandedDstToSrcRow{0}; // peermem 打包区基址（rank 段 [slots|probs]）
        int64_t dedupOffsetD{0};              // Dedup D 区基址（发送方 rank 分段）
        int64_t distTokenFirstPosSize{0};     // workspace FP 表段字节数（EP·chunkTokens·4B）
        int64_t distTokenLastPosSize{0};      // workspace LP 表段字节数
        int64_t dedupRowMetaSize{0};          // workspace rowMeta 段字节数（recvRoundBudget·32B）

        // Methods
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
               float activationParams2_ = Epilogue::SituActivation::DEFAULT_BETA, GM_ADDR tilingGM_ = nullptr,
               GM_ADDR ptrMaskBuffer_ = nullptr)
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
              ptrMaskBuffer(ptrMaskBuffer_),
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
            sharedN = problemShape.n();
            uint32_t k2 = problemShape.n() / 2;
            gmm1InputExpertSlotWidth = problemShape.k() > k2 ? problemShape.k() : k2;
        }
    };

    // Methods
    CATLASS_DEVICE
    MegaMoeKernel(Params const& params)
    {
        if ASCEND_IS_AIC {
            coreIdx = AscendC::GetBlockIdx();
            coreNum = AscendC::GetBlockNum();
        }

        if ASCEND_IS_AIV {
            coreIdx = get_block_idx() + get_subblockid() * get_block_num();
            coreNum = get_block_num() * get_subblockdim();
        }

        initBuffer(params);
    }

    CATLASS_DEVICE bool IsCurrentRankMasked()
    {
        if ASCEND_IS_AIC {
            // The rank-mask UB copy belongs to AIV. Cube must read the input
            // from GM so both core types take the same early-exit branch.
            if (maskBufferAddr == nullptr) {
                return false;
            }
            AscendC::GlobalTensor<int32_t> rankMaskGm;
            rankMaskGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(maskBufferAddr));
            return rankMaskGm.GetValue(shmem.Rank()) != 0;
        }
        return IsRankMasked(shmem.Rank());
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const& params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const& params)
    {
        if (IsCurrentRankMasked()) {
            return;
        }
        if (params.sharedExpertNum > 0U) {
            if constexpr (kSharedIsQuant) {
                AscendC::CrossCoreWaitFlag<0x2>(SYNCFLAG_SHARED_INPUT_V2C);
            }
            SharedGMM1(params);
        }
        syncgmmIdx = 0;
        const int64_t bs = params.problemShape.m();
        const int64_t numChunks = NumChunks(params);
        for (int64_t chunkIdx = 0; chunkIdx < numChunks; ++chunkIdx) {
            // cumsumMM 每 chunk 指向独立缓冲区（不复用，规避跨 chunk cache 陈旧读）；
            // chunkCumsumMM 保持 chunk 级指向，供轮数推导与轮表生成读取。
            GM_ADDR chunkCumsumPtr = ChunkCumsumPtr(params, chunkIdx);
            chunkCumsumMM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(chunkCumsumPtr));
            cumsumMM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(chunkCumsumPtr));
            // 接收端轮次切分：R ≤ B 时单轮（退化零回归）；R > B 时每轮等待 AIV 轮表就绪后
            // 切换 cumsumMM 到轮表。syncgmmIdx/syncgmm1Idx 按轮归零，flag 槽位按轮复用。
            int64_t numRecvRounds = 1;
            for (int64_t roundIdx = 0;; ++roundIdx) {
                // 原 GMM1 入口的初始等待外移到轮级：rounds==1 时即 chunk cumsum/路由就绪信号，
                // rounds>1 时为 BuildRoundTables 完成后的轮表就绪信号
                {
                    AscendC::CrossCoreWaitFlag<0x2>(syncgmmIdx / CROSS_CORE_FLAG_MAX_SET_COUNT);
                }
                syncgmmIdx++;
                W1PrefetchRange w1PrefetchRangeThisRound{};
                // W1 冷启动：flag0 后、GMM1 组 0 wait 前入队哑拷贝，与 pull g0 并发排空；
                // 仅首轮发射一次（权重静态）
                if (chunkIdx == 0 && roundIdx == 0 && params.problemShape.m() <= WEIGHT_L2_PREFETCH_MAX_BS) {
                    // 空专家过滤
                    uint32_t currentM0 = cumsumMM.GetValue((params.EP - 1) * params.expertPerRank);
                    if (currentM0 > 0) {
                        BlockMmad l1EndProbe(resource);
                        uint32_t w1ScratchOffset = AlignUp(l1EndProbe.GetL1EndOffset(), WEIGHT_L2_PREFETCH_CHUNK_BYTES);
                        if (w1ScratchOffset + WEIGHT_L2_PREFETCH_CHUNK_BYTES <= ArchTag::L1_SIZE) {
                            w1PrefetchRangeThisRound = WeightL2PrefetchAicColdStartW1(
                                reinterpret_cast<GM_ADDR>(params.ptrB1), params.listLen != 1,
                                WeightL2PrefetchBytesPerExpert<ElementB>(params.problemShape.k(),
                                                                         params.problemShape.n()),
                                static_cast<uint32_t>(params.expertPerRank),
                                resource.l1Buf.template GetBufferByByte<uint8_t>(w1ScratchOffset));
                        }
                    }
                }
                if (roundIdx == 0) {
                    numRecvRounds = ComputeNumRecvRounds(params);
                }
                if (numRecvRounds > 1) {
                    SetRoundTableBuffers(params, chunkIdx, roundIdx);
                }
                GMM1(params, chunkIdx, w1PrefetchRangeThisRound);
                const auto sharedSchedule =
                    MegaMoeSharedGmm2::ForRound(params.sharedExpertNum, SharedGmm2Chunked(params), chunkIdx, roundIdx);
                const bool sameChunkLate = SharedGmm2Chunked(params);
                if (sharedSchedule.gmm2 && !sameChunkLate) {
                    if (sharedSchedule.activation) {
                        AscendC::CrossCoreWaitFlag<0x2>(SYNCFLAG_SHARED_GMM1_V2C);
                    }
                    const auto range = SharedGmm2Range(params, chunkIdx);
                    SharedGMM2(params, range.base, range.rows);
                }
                if constexpr (!std::is_same_v<ElementB, AscendC::int4b_t>) {
                    {
                        AscendC::CrossCoreWaitFlag<0x2>(SYNCFLAGV2C + chunkIdx * PERMUTE_CHUNK_FLAG_STRIDE);
                    }
                    GMM2(params, chunkIdx);
                } else {
                    GMM2(params, chunkIdx);
                }
                if (sharedSchedule.gmm2 && sameChunkLate) {
                    // Routed GMM2 releases its L1/L0 before shared GMM2 reuses them.
                    if (sharedSchedule.activation) {
                        AscendC::CrossCoreWaitFlag<0x2>(SYNCFLAG_SHARED_GMM1_V2C);
                    }
                    const auto range = SharedGmm2Range(params, chunkIdx);
                    SharedGMM2(params, range.base, range.rows);
                }
                syncgmmIdx = 0; // 轮间从 flag 0 复用 syncgmm 握手（与 AIV syncgmm1Idx 对称）
                if (roundIdx + 1 >= numRecvRounds) {
                    break;
                }
            }
        }
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const& params)
    {
        if (IsCurrentRankMasked()) {
            return;
        }
        DispatchAndCombine(params);
    }

private:
    CATLASS_DEVICE bool SharedGmm2Chunked(Params const& params) const
    {
        return MegaMoeSharedGmm2::CanOverlap(params.problemShape.m(), params.sharedExpertNum,
                                             (SharedGmm2Bf16_ && std::is_same_v<ElementB, int8_t>) ||
                                                 (!SharedGmm2Bf16_ && std::is_same_v<ElementB, AscendC::int4b_t> &&
                                                  std::is_same_v<SharedElementB, int8_t>),
                                             params.ptrXActiveMask != nullptr || params.ptrMaskBuffer != nullptr,
                                             params.recvRoundsMax, params.maxOutputSize, params.EP, params.topK,
                                             params.expertPerRank, static_cast<uint32_t>(GetPermuteChunk(params)));
    }
    CATLASS_DEVICE bool SharedGmm2WaitAfterCrossRank(Params const& params) const
    {
        // W4A8 joins all AIVs before its split-subblock cross-rank protocol.
        return SharedGmm2Bf16_ && SharedGmm2Chunked(params);
    }
    CATLASS_DEVICE MegaMoeSharedGmm2::Range SharedGmm2Range(Params const& params, int64_t chunk) const
    {
        return SharedGmm2Chunked(params) ?
                   MegaMoeSharedGmm2::TokenRange(params.problemShape.m(), chunk,
                                                 static_cast<uint32_t>(GetPermuteChunk(params))) :
                   MegaMoeSharedGmm2::Range{0U, static_cast<uint32_t>(params.problemShape.m())};
    }

    CATLASS_DEVICE
    int32_t RuntimeRank(Params const& params) const
    {
        (void)params;
        return shmem.Rank();
    }

    CATLASS_DEVICE static int64_t GetPermuteChunk(Params const& params)
    {
        return MegaMoeConfig::GetPermuteChunk(BatchedDispatch);
    }

    CATLASS_DEVICE void initBuffer(Params const& params)
    {
        auto tmpContext = reinterpret_cast<__gm__ Mc2Aclnn::Mc2MoeContext*>(params.contextGM);
        shmem.initShmem(tmpContext);
        if ASCEND_IS_AIV {
            GM_ADDR dumpBase = shmem();
            exceptionDump_.Init(dumpBase, params.tilingGM);
        }
        workspaceInfo = RouteWorkspaceInfo(params);
        peermemInfo = RoutePeermemInfo(params, shmem);

        cumsumMM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspaceInfo.ptrcumsumMM));

        if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
            gmS.SetGlobalBuffer(params.ptrScale1);
            gmA1I4.SetGlobalBuffer(reinterpret_cast<__gm__ int4b_t*>(workspaceInfo.ptrA1Int4));
            gmA1I4_I8.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(workspaceInfo.ptrA1Int4));
            gmA2I4.SetGlobalBuffer(reinterpret_cast<__gm__ int4b_t*>(workspaceInfo.ptrA2Int4));
            gmA2I4_I8.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(workspaceInfo.ptrA2Int4));
            gmS2.SetGlobalBuffer(params.ptrScale2);
        } else {
            gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementABefore*>(workspaceInfo.ptrA));
            gmPermutedToken.SetGlobalBuffer(reinterpret_cast<__gm__ ElementD1*>(workspaceInfo.ptrPermutedToken));
        }

        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC*>(workspaceInfo.ptrC));
        gmC2.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC*>(workspaceInfo.ptrC2));

        if (params.sharedExpertNum > 0U) {
            gmSharedGmmBuffer.SetGlobalBuffer(
                reinterpret_cast<__gm__ SharedElementC*>(workspaceInfo.ptrSharedGmmBuffer));
            gmSharedActivation.SetGlobalBuffer(
                reinterpret_cast<__gm__ SharedElementD1*>(workspaceInfo.ptrSharedActivation));
            gmSharedResult.SetGlobalBuffer(reinterpret_cast<__gm__ ElementD2*>(workspaceInfo.ptrSharedResult));
            if constexpr (kSharedIsQuant) {
                gmSharedInput.SetGlobalBuffer(reinterpret_cast<__gm__ SharedElementA*>(workspaceInfo.ptrSharedInput));
                gmSharedInputScale.SetGlobalBuffer(
                    reinterpret_cast<__gm__ ElementPerTokenScale*>(workspaceInfo.ptrSharedInputScale));
                gmSharedActivationScale.SetGlobalBuffer(
                    reinterpret_cast<__gm__ ElementPerTokenScale*>(workspaceInfo.ptrSharedActivationScale));
            }
        }

        gmPerTokenScale1.SetGlobalBuffer(
            reinterpret_cast<__gm__ ElementPerTokenScale*>(workspaceInfo.ptrPerTokenScale));
        gmPerTokenScale2.SetGlobalBuffer(
            reinterpret_cast<__gm__ ElementPerTokenScale*>(workspaceInfo.ptrPerTokenScale2));

        tokenPerExpert.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(shmem() + peermemInfo.offsetPeerTokenPerExpert));

        paddedExpertNumAligned = AlignUp(params.EP * params.expertPerRank + 1, ALIGN_128);
        tokenPerExpertLayout = Layout3D(paddedExpertNumAligned, params.expertPerRank);
        preSumBeforeRank.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspaceInfo.ptrSumBeforeRank));
        maskBufferAddr = params.ptrMaskBuffer;
        InitRankMask(params);

        if ASCEND_IS_AIV {
            InitRouteCache(params);
        }

        InitCombineMode(params);
    }

    CATLASS_DEVICE void InitRankMask(Params const& params)
    {
        if ASCEND_IS_AIV {
            if (maskBufferAddr != nullptr) {
                uint32_t rankMaskBytes = AlignUp(params.EP * sizeof(int32_t), BYTE_PER_BLK);
                uint32_t rankMaskUbOffset = ArchTag::UB_SIZE - rankMaskBytes;
                rankMaskLocal = resource.ubBuf.template GetBufferByByte<int32_t>(rankMaskUbOffset);
                rankMaskLocal.SetSize(rankMaskBytes / sizeof(int32_t));
                CopyRankMaskToLocal(params);
            }
        }
    }

    CATLASS_DEVICE void InitRouteCache(Params const& params)
    {
        // UB 顶部保留 rankMask、preSumBeforeRank 和 tokenPerExpert 缓存。
        routeCacheNum = params.EP * params.expertPerRank;
        expertPerRankAligned = AlignUp(params.expertPerRank, 8);
        uint32_t tpeCacheBytes = AlignUp(params.EP * expertPerRankAligned * sizeof(int32_t), BYTE_PER_BLK);
        uint32_t routeCacheBytes = AlignUp(routeCacheNum * sizeof(int32_t), BYTE_PER_BLK);
        uint32_t rankMaskReserved = AlignUp(params.EP * sizeof(int32_t), BYTE_PER_BLK);
        uint32_t routeCacheUbOffset = ArchTag::UB_SIZE - rankMaskReserved - tpeCacheBytes - routeCacheBytes;
        ubTokenPerExpertCache = resource.ubBuf.template GetBufferByByte<int32_t>(routeCacheUbOffset);
        ubPreSumBeforeRankCache = resource.ubBuf.template GetBufferByByte<int32_t>(routeCacheUbOffset + tpeCacheBytes);
    }

    CATLASS_DEVICE void InitCombineMode(Params const& params)
    {
        isCombineV1 = false;
        // Mask 实例使用 CombineV2 写回专家结果。
        if constexpr (std::is_same_v<ElementB, int8_t> && !MASK_ROUTING) {
            isCombineV1 = true;
            if (BatchedDispatch || params.problemShape.m() * params.topK <= 4096) {
                isCombineV1 = false;
            }
        }
    }

    CATLASS_DEVICE
    void CopyRankMaskToLocal(Params const& params)
    {
        if (maskBufferAddr == nullptr) {
            return;
        }
        AscendC::GlobalTensor<int32_t> rankMaskGm;
        rankMaskGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(maskBufferAddr));
        AscendC::DataCopyPad(rankMaskLocal, rankMaskGm, {1, static_cast<uint16_t>(params.EP * sizeof(int32_t)), 0, 0},
                             {});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    }

    CATLASS_DEVICE
    bool IsRankMasked(int32_t rank)
    {
        if (maskBufferAddr == nullptr) {
            return false;
        }
        return rankMaskLocal.GetValue(rank) != 0;
    }

    template <typename T>
    CATLASS_DEVICE void CopyGMToGM(AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, int32_t elemNum,
                                   int32_t ubMoveNum)
    {
        Detail::CopyGMToGMShared<ArchTag>(resource, dst, src, elemNum, ubMoveNum);
    }

    CATLASS_DEVICE void PublishMaskDataReady(Params const& p);

    CATLASS_DEVICE void WaitMaskDataReady(Params const& p);

    CATLASS_DEVICE void ResetMaskDataReady(Params const& p);
    CATLASS_DEVICE void CopyMaskRows(Params const& p, uint32_t srcRank, uint32_t expert, uint32_t rows,
                                     uint32_t savedRow, uint64_t gmOffsetA, int32_t& pingpong);

    template <typename T>
    CATLASS_DEVICE void CopyGMToGMPerToken(AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<float> dstScale,
                                           AscendC::GlobalTensor<T> src, int32_t rows, int32_t hiddenSize,
                                           int32_t ubMoveNum, int32_t& pingpongId)
    {
        constexpr int32_t BufferNum = 2;
        AscendC::LocalTensor<T> tmpBuffer1 = resource.ubBuf.template GetBufferByByte<T>(0);
        constexpr int tmpBufferOffset = 96 * 1024;
        AscendC::LocalTensor<T> tmpBuffer2 = resource.ubBuf.template GetBufferByByte<T>(tmpBufferOffset);
        // quant 路径远端 expandedX 行步长 = hiddenSize + routingPadding_（行尾含 per-token float scale + padding）
        uint32_t copyInNum = hiddenSize + routingPadding_;
        auto processCount = CeilDiv(rows, ubMoveNum);
        for (uint32_t processIndex = 0; processIndex < processCount; ++processIndex) {
            pingpongId = (pingpongId + 1) % BufferNum;
            AscendC::TEventID EVENT_ID = pingpongId == 0 ? EVENT_ID0 : EVENT_ID1;
            AscendC::LocalTensor<T> buf = pingpongId == 0 ? tmpBuffer1 : tmpBuffer2;
            AscendC::LocalTensor<float> bufScale = buf[hiddenSize].template ReinterpretCast<float>();
            auto inputOffset = processIndex * ubMoveNum * copyInNum;

            int32_t rowNum = ubMoveNum;
            if (processIndex == processCount - 1) {
                rowNum = rows - processIndex * ubMoveNum;
            }

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
            int64_t dataLen = rowNum * copyInNum;
            AscendC::DataCopy(buf, src[inputOffset], dataLen);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
            auto outputOffset = processIndex * ubMoveNum * hiddenSize;
            // DataCopyExtParams UB 侧 block 间距 = srcStride*32 + blockLen（向上 32B 对齐），
            // 必须等于 UB 内实际行距 copyInNum（hiddenSize + routingPadding_），否则 blockCount>=2 时
            // 第二行会读到上一行行尾 padding。
            // payload: blockLen=hiddenSize，srcStride = (copyInNum-hiddenSize)/32 = routingPadding_/32
            // scale:   blockLen=4，srcStride = copyInNum/32 - 1，使 srcStride*32+4 对齐后 = copyInNum
#define U16(x) static_cast<uint16_t>(x)
            AscendC::DataCopyPad(dst[outputOffset], buf,
                                 {U16(rowNum), U16(hiddenSize), U16(routingPadding_ / 32), 0, 0});
            AscendC::DataCopyPad(dstScale[processIndex * ubMoveNum], bufScale,
                                 {U16(rowNum), U16(sizeof(float)), U16(copyInNum / 32 - 1), 0, 0});
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
        }
    }

    CATLASS_DEVICE void FetchAndPreprocessInt8ToInt4(uint64_t gmOffsetA, AscendC::GlobalTensor<float> dstScale,
                                                     AscendC::GlobalTensor<int8_t> src, int32_t rows,
                                                     int32_t hiddenSize, bool useBatched = false)
    {
        if (useBatched && hiddenSize == 2048) {
            Detail::FetchAndPreprocessInt8ToInt4BatchedShared<ArchTag>(
                resource, gmA1I4_I8, gmOffsetA, dstScale, src, rows, hiddenSize, static_cast<uint32_t>(hiddenSize),
                routingPadding_);
            return;
        }
        Detail::FetchAndPreprocessInt8ToInt4Shared<ArchTag>(resource, gmA1I4_I8, gmOffsetA, dstScale, src, rows,
                                                            hiddenSize, static_cast<uint32_t>(hiddenSize), coreNum,
                                                            routingPadding_);
    }

    // 取出各来源卡上属于本卡的专家计数，只有搬运描述由两条路径共用。
    CATLASS_DEVICE void LoadExpertCountsForCumsum(AscendC::LocalTensor<int32_t>& countsUb,
                                                  AscendC::GlobalTensor<int32_t>& tokenPerExpert,
                                                  uint32_t expertPerRank, uint32_t rankId, uint32_t EP)
    {
        AscendC::DataCopyPad(countsUb, tokenPerExpert[rankId * expertPerRank],
                             {U16(EP), U16(expertPerRank * sizeof(int32_t)),
                              U16((paddedExpertNumAligned - expertPerRank) * sizeof(int32_t)), 0},
                             {});
    }

    CATLASS_DEVICE
    void GetCumsumForMMAIV(AscendC::GlobalTensor<int32_t>& tokenPerExpert, AscendC::GlobalTensor<int32_t>& result,
                           uint32_t expertPerRank, uint32_t rankId, uint32_t EP)
    {
        int32_t expertPerRankAligned = AlignUp(expertPerRank, 8);
        AscendC::LocalTensor<int32_t> tmpBuffer1 = resource.ubBuf.template GetBufferByByte<int32_t>(0);
        AscendC::LocalTensor<int32_t> tmpResult =
            resource.ubBuf.template GetBufferByByte<int32_t>(EP * expertPerRank * sizeof(int32_t));
#define U16(x) static_cast<uint16_t>(x)

        LoadExpertCountsForCumsum(tmpBuffer1, tokenPerExpert, expertPerRank, rankId, EP);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        for (uint32_t i = 1; i < EP; ++i) {
            AscendC::Add(tmpBuffer1[i * expertPerRankAligned], tmpBuffer1[i * expertPerRankAligned],
                         tmpBuffer1[(i - 1) * expertPerRankAligned], expertPerRank);
            AscendC::PipeBarrier<PIPE_V>();
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

        AscendC::DataCopyPad(result, tmpBuffer1, {U16(EP), U16((expertPerRank) * sizeof(int32_t)), 0, 0});
        // cumsumMM 写出落盘后再返回，消费方（dispatch/GMM）经 SyncAll 读取时数据已就绪
        AscendC::PipeBarrier<PIPE_MTE3>();
    }

    CATLASS_DEVICE
    void GetMaskCumsumForMMAIV(AscendC::GlobalTensor<int32_t>& tokenPerExpert, AscendC::GlobalTensor<int32_t>& result,
                               uint32_t expertPerRank, uint32_t rankId, uint32_t EP, GM_ADDR expertTokenNums);

    // 将 tokenPerExpert / preSumBeforeRank 从 GM 搬入本核 UB 缓存（每 chunk 在
    // AllGather + cumsum + SyncAll 之后由全部 AIV 核各刷一份），后续 dispatch /
    // combine / epilogue 的标量读取一律走 UB，不再直接读 GM。
    CATLASS_DEVICE void CacheRouteTablesToUb(Params const& params)
    {
        if (routeCacheNum == 0) {
            return;
        }
        // DataCopyPad pads each UB row to 32 bytes. Preserve that layout:
        // dispatch/combine index counts with expertPerRankAligned, including
        // non-aligned cases such as EP=32 with 12 experts per rank.
        uint32_t runtimeRank = static_cast<uint32_t>(RuntimeRank(params));
        AscendC::DataCopyPad(
            ubTokenPerExpertCache, tokenPerExpert[runtimeRank * params.expertPerRank],
            {static_cast<uint16_t>(params.EP), static_cast<uint16_t>(params.expertPerRank * sizeof(int32_t)),
             static_cast<uint16_t>((paddedExpertNumAligned - params.expertPerRank) * sizeof(int32_t)), 0},
            {});
        // preSumBeforeRank 连续存放，整段搬入
        if constexpr (!MASK_ROUTING) {
            AscendC::DataCopyPad(ubPreSumBeforeRankCache, preSumBeforeRank,
                                 {1, static_cast<uint16_t>(routeCacheNum * sizeof(int32_t)), 0, 0}, {});
        }
        // 标量读取前确保 MTE2 搬入完成（与 CopyRankMaskToLocal 一致）
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    }

    // 发送侧 chunk 总数
    CATLASS_DEVICE static int64_t NumChunks(Params const& params)
    {
        const int64_t bs = params.problemShape.m();
        return (bs + GetPermuteChunk(params) - 1) / GetPermuteChunk(params);
    }

    // AIC scalar loads cache whole lines. Distinct rounds/chunks must not share
    // a line that AIV will update later, even when EP * expertPerRank is tiny.
    CATLASS_DEVICE static uint64_t ChunkCumsumStride(Params const& params)
    {
        const uint64_t bytes = static_cast<uint64_t>(params.EP) * params.EP * params.expertPerRank * sizeof(int32_t);
        if constexpr (!BatchedDispatch) {
            return AlignUp(bytes, 64UL);
        }
        return (bytes + CACHE_LINE - 1) / CACHE_LINE * CACHE_LINE;
    }

    CATLASS_DEVICE static uint64_t RoundTableSegmentBytes(Params const& params)
    {
        const uint64_t bytes = static_cast<uint64_t>(params.EP) * params.expertPerRank * sizeof(int32_t);
        if constexpr (!BatchedDispatch) {
            return AlignUp(bytes, 64UL);
        }
        return (bytes + CACHE_LINE - 1) / CACHE_LINE * CACHE_LINE;
    }

    // GetValue caches a whole 64-byte DCache line. Distinct generations must not
    // share a line: EP2/Epr2 otherwise preloads the next chunk before it is written.
    CATLASS_DEVICE GM_ADDR ChunkCumsumPtr(Params const& params, int64_t chunkIdx) const
    {
        return workspaceInfo.ptrcumsumMM + chunkIdx * ChunkCumsumStride(params);
    }

    // 接收端轮次切分
    // 单个发送 chunk 内本卡待接收总行数 R = Σ_g cumsumMM[(EP-1)*epr+g]，
    // 轮数 = ceil(R / B)，B = params.maxOutputSize（单轮接收预算）。
    // 调用前提：cumsumMM 已指向 chunk 级缓冲。AIV/AIC 两侧独立调用，结果一致。
    CATLASS_DEVICE int64_t ComputeNumRecvRounds(Params const& params)
    {
        const int64_t roundBudget = static_cast<int64_t>(params.maxOutputSize);
        int64_t totalRows = 0;
        for (uint32_t g = 0; g < params.expertPerRank; ++g) {
            totalRows += cumsumMM.GetValue((params.EP - 1) * params.expertPerRank + g);
        }
        int64_t rounds = (totalRows + roundBudget - 1) / roundBudget;
        return rounds < 1 ? 1 : rounds;
    }

    // 轮表段索引：每份三段 [cumsumMM_r | tokenPerExpert_r | preSumBeforeRank_r]
    enum class RoundTableSeg : int32_t {
        kCumsum = 0,
        kTokenPerExpert = 1,
        kPreSum = 2
    };
    // 轮表每份段数（host 侧 workspace 公式按同值分配）
    static constexpr int64_t ROUND_TABLE_SEGMENTS = 3;

    // (chunk, round) 轮表基址：每份三段 [cumsumMM_r | tokenPerExpert_r | preSumBeforeRank_r]，
    // 各 EP*expertPerRank int32，段间按 CACHE_LINE 隔离。
    CATLASS_DEVICE GM_ADDR RoundTablePtr(Params const& params, int64_t chunkIdx, int64_t roundIdx,
                                         RoundTableSeg which) const
    {
        const int64_t seg = static_cast<int64_t>(RoundTableSegmentBytes(params));
        return workspaceInfo.ptrRoundTables +
               ((chunkIdx * static_cast<int64_t>(params.recvRoundsMax) + roundIdx) * ROUND_TABLE_SEGMENTS +
                static_cast<int64_t>(which)) *
                   seg;
    }

    // 轮表生成（仅 numRecvRounds > 1 时由 core0 调用）：由 chunk 级 cumsumMM 推导 (chunk, roundIdx)
    // 的三张轮表并写 GM。count[s][g] = cumsumMM[s*epr+g] - cumsumMM[(s-1)*epr+g]，不依赖 UB 缓存
    // （其内容已被上一轮轮表覆盖）。第 r 轮覆盖行区间 [r*B, min((r+1)*B, R))：
    //   tokenPerExpert_r[s][g] = max(0, min((r+1)*B, blkLo+cnt) - max(r*B, blkLo))
    //   preSumBeforeRank_r[s][g] = senderPrefix[s][g] + max(0, r*B - blkLo)
    // blkLo is a receiver-order offset, NOT an offset in the sender's A/D buffers.
    //   cumsumMM_r[(s-1)*epr+g] = Σ_{s'<s} tokenPerExpert_r[g][s']，[(EP-1)*epr+g] = 轮内 group 行数
    CATLASS_DEVICE void FillRoundTables(AscendC::LocalTensor<int32_t>& ubCumR, AscendC::LocalTensor<int32_t>& ubTpeR,
                                        AscendC::LocalTensor<int32_t>& ubPreSumR,
                                        AscendC::LocalTensor<int32_t>& ubSenderPrefix, int64_t epr, int64_t ep,
                                        int64_t roundLo, int64_t roundHi)
    {
        for (int64_t g = 0; g < epr; ++g) {
            int64_t prevGroupSum = 0;
            for (int64_t gg = 0; gg < g; ++gg) {
                prevGroupSum += chunkCumsumMM.GetValue((ep - 1) * epr + gg);
            }
            int64_t pre = 0;
            for (int64_t s = 0; s < ep; ++s) {
                const int64_t cumCur = chunkCumsumMM.GetValue(s * epr + g);
                const int64_t cumPrev = s > 0 ? chunkCumsumMM.GetValue((s - 1) * epr + g) : 0;
                const int64_t blkLo = prevGroupSum + cumPrev;     // 块起点（chunk 行空间）
                const int64_t blkHi = blkLo + (cumCur - cumPrev); // 块终点（开区间）
                const int64_t lo = roundLo > blkLo ? roundLo : blkLo;
                const int64_t hi = blkHi < roundHi ? blkHi : roundHi;
                const int64_t m = hi > lo ? (hi - lo) : 0;
                if (s > 0) {
                    ubCumR.SetValue((s - 1) * epr + g, static_cast<int32_t>(pre));
                }
                ubTpeR.SetValue(s * epr + g, static_cast<int32_t>(m));
                const int64_t senderOffset = ubSenderPrefix.GetValue(s * epr + g);
                // Empty intersections must not manufacture out-of-range peer pointers.
                ubPreSumR.SetValue(s * epr + g, static_cast<int32_t>(senderOffset + (m > 0 ? lo - blkLo : 0)));
                pre += m;
            }
            ubCumR.SetValue((ep - 1) * epr + g, static_cast<int32_t>(pre));
        }
    }

    CATLASS_DEVICE void BuildRoundTables(Params const& params, int64_t chunkIdx, int64_t roundIdx)
    {
        if (coreIdx != 0) {
            return;
        }
        const int64_t B = static_cast<int64_t>(params.maxOutputSize);
        const int64_t epr = static_cast<int64_t>(params.expertPerRank);
        const int64_t ep = static_cast<int64_t>(params.EP);
        const int64_t roundLo = roundIdx * B;
        const int64_t roundHi = roundLo + B; // 轮右界（开区间）
        const int64_t n = ep * epr;

        const int64_t tableBytes = AlignUp(n * sizeof(int32_t), BYTE_PER_BLK);

        AscendC::LocalTensor<int32_t> ubCumR = resource.ubBuf.template GetBufferByByte<int32_t>(0);
        AscendC::LocalTensor<int32_t> ubTpeR = resource.ubBuf.template GetBufferByByte<int32_t>(tableBytes);
        AscendC::LocalTensor<int32_t> ubPreSumR = resource.ubBuf.template GetBufferByByte<int32_t>(2 * tableBytes);
        AscendC::LocalTensor<int32_t> ubSenderPrefix = resource.ubBuf.template GetBufferByByte<int32_t>(3 * tableBytes);
        // The member preSumBeforeRank and UB cache may refer to the previous
        // round. Reload the original chunk's sender prefixes from their fixed
        // workspace allocation; MTE2 also avoids stale scalar GM cache reads.
        AscendC::GlobalTensor<int32_t> gmSenderPrefix;
        gmSenderPrefix.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspaceInfo.ptrSumBeforeRank));
        AscendC::DataCopyPad(ubSenderPrefix, gmSenderPrefix, {1, static_cast<uint16_t>(n * sizeof(int32_t)), 0, 0}, {});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);

        FillRoundTables(ubCumR, ubTpeR, ubPreSumR, ubSenderPrefix, epr, ep, roundLo, roundHi);

        // UB 标量写完成后再发起 MTE3 搬出（与 AllGatherTokenPerExpert 的 S_MTE3 同步一致）
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::GlobalTensor<int32_t> gmDst;
        AscendC::DataCopyParams cpParams{1, static_cast<uint16_t>(n * sizeof(int32_t)), 0, 0};
        gmDst.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(RoundTablePtr(params, chunkIdx, roundIdx, RoundTableSeg::kCumsum)));
        AscendC::DataCopyPad(gmDst, ubCumR, cpParams);
        gmDst.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(
            RoundTablePtr(params, chunkIdx, roundIdx, RoundTableSeg::kTokenPerExpert)));
        AscendC::DataCopyPad(gmDst, ubTpeR, cpParams);
        gmDst.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(RoundTablePtr(params, chunkIdx, roundIdx, RoundTableSeg::kPreSum)));
        AscendC::DataCopyPad(gmDst, ubPreSumR, cpParams);
        // 轮表 GM 写出落盘后再返回，消费方（其他 AIV 的 MTE2 搬入 / AIC 标量读）经 SyncAll 读取
        // （与 GetCumsumForMMAIV 的 cumsumMM 写出约束一致）
        AscendC::PipeBarrier<PIPE_MTE3>();
    }

    // 将成员 cumsumMM 切换到 (chunk, round) 轮表，并设置轮表 tpe_r / preSum_r 的 GT
    // （chunkCumsumMM 保持指向 chunk 级缓冲，供下一轮 BuildRoundTables 使用）
    CATLASS_DEVICE void SetRoundTableBuffers(Params const& params, int64_t chunkIdx, int64_t roundIdx)
    {
        cumsumMM.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(RoundTablePtr(params, chunkIdx, roundIdx, RoundTableSeg::kCumsum)));
        roundTokenPerExpertGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(
            RoundTablePtr(params, chunkIdx, roundIdx, RoundTableSeg::kTokenPerExpert)));
        roundPreSumBeforeRankGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(RoundTablePtr(params, chunkIdx, roundIdx, RoundTableSeg::kPreSum)));
        // CombineV2 的 epilogue 以成员 GT 读 D 区写回偏移表，轮内切到 preSum_r
        // （= 对应发送 rank 的 chunk 行偏移，D 区布局不受轮次影响）；
        // 调用方须在下一 chunk 的 AllGather 前恢复其 chunk 级指向
        preSumBeforeRank.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(RoundTablePtr(params, chunkIdx, roundIdx, RoundTableSeg::kPreSum)));
    }

    // 将本轮轮表 tpe_r / preSum_r 刷新到本核 UB 路由缓存（计数按 expertPerRankAligned 补齐，前缀紧凑，
    // dispatch / CombineV1 / CombineV2 / BlockEpilogue3 读取路径零改动）。仅 rounds>1 时调用。
    CATLASS_DEVICE void CacheRoundTablesToUb(Params const& params)
    {
        if (routeCacheNum == 0) {
            return;
        }
        AscendC::DataCopyPad(
            ubTokenPerExpertCache, roundTokenPerExpertGm,
            {static_cast<uint16_t>(params.EP), static_cast<uint16_t>(params.expertPerRank * sizeof(int32_t)), 0, 0},
            {});
        AscendC::DataCopyPad(ubPreSumBeforeRankCache, roundPreSumBeforeRankGm,
                             {1, static_cast<uint16_t>(routeCacheNum * sizeof(int32_t)), 0, 0}, {});
        // 标量读取前确保 MTE2 搬入完成（与 CacheRouteTablesToUb 一致）
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    }

    CATLASS_DEVICE
    void PrepareSharedInput(Params const& params)
    {
        if constexpr (!kSharedIsQuant) {
            return;
        } else {
            const uint32_t m = params.problemShape.m();
            const uint32_t k = params.problemShape.k();
            const uint32_t rowsPerCore = m / coreNum;
            const uint32_t remainder = m % coreNum;
            const uint32_t rowCount = coreIdx < remainder ? rowsPerCore + 1U : rowsPerCore;
            const uint32_t rowStart = coreIdx * rowsPerCore + (coreIdx < remainder ? coreIdx : remainder);

            uint32_t ubOffset = 0U;
            auto inputLocal = resource.ubBuf.template GetBufferByByte<ElementD2>(ubOffset);
            ubOffset += AlignUp(k * sizeof(ElementD2), 32U);
            auto inputFp32 = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
            ubOffset += AlignUp(k * sizeof(float), 32U);
            auto tempFp32 = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
            ubOffset += AlignUp(k * sizeof(float), 32U);
            auto quantLocal = resource.ubBuf.template GetBufferByByte<int8_t>(ubOffset);
            ubOffset += AlignUp(k * sizeof(int8_t), 32U);
            auto reduceMaxLocal = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
            ubOffset += 32U;

            AscendC::LocalTensor<int4b_t> highI4;
            AscendC::LocalTensor<int4b_t> lowI4;
            AscendC::LocalTensor<half> highHalf;
            AscendC::LocalTensor<half> lowHalf;
            AscendC::LocalTensor<half> lowHalf2;
            AscendC::LocalTensor<int16_t> lowMask;
            if constexpr (kSharedIsW4A8) {
                highI4 = resource.ubBuf.template GetBufferByByte<int4b_t>(ubOffset);
                ubOffset += AlignUp(k / 2U, 32U);
                lowI4 = resource.ubBuf.template GetBufferByByte<int4b_t>(ubOffset);
                ubOffset += AlignUp(k / 2U, 32U);
                highHalf = resource.ubBuf.template GetBufferByByte<half>(ubOffset);
                ubOffset += AlignUp(k * sizeof(half), 32U);
                lowHalf = resource.ubBuf.template GetBufferByByte<half>(ubOffset);
                ubOffset += AlignUp(k * sizeof(half), 32U);
                lowHalf2 = resource.ubBuf.template GetBufferByByte<half>(ubOffset);
                ubOffset += AlignUp(k * sizeof(half), 32U);
                lowMask = resource.ubBuf.template GetBufferByByte<int16_t>(ubOffset);
                AscendC::Duplicate(lowMask, static_cast<int16_t>(0x0F0F), 128U);
                AscendC::PipeBarrier<PIPE_V>();
            }

            AscendC::GlobalTensor<ElementD2> input;
            AscendC::GlobalTensor<int8_t> output;
            AscendC::GlobalTensor<float> outputScale;
            input.SetGlobalBuffer(reinterpret_cast<__gm__ ElementD2*>(params.ptrA));
            output.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(workspaceInfo.ptrSharedInput));
            outputScale.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspaceInfo.ptrSharedInputScale));

            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
            for (uint32_t localRow = 0U; localRow < rowCount; ++localRow) {
                const uint32_t row = rowStart + localRow;
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
                AscendC::DataCopyPad(inputLocal, input[static_cast<uint64_t>(row) * k],
                                     {1U, static_cast<uint32_t>(k * sizeof(ElementD2)), 0U, 0U, 0U}, {});
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID6);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID6);

                AscendC::Cast(inputFp32, inputLocal, AscendC::RoundMode::CAST_NONE, k);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Abs(tempFp32, inputFp32, k);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::ReduceMax(reduceMaxLocal, tempFp32, tempFp32, k);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID6);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID6);

                const float absMax = reduceMaxLocal.GetValue(0U);
                const float quantScale = absMax > 0.0F ? 127.0F / absMax : 127.0F;
                const float dequantScale = 1.0F / quantScale;
                AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID6);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID6);
                AscendC::Muls(tempFp32, inputFp32, quantScale, k);
                AscendC::PipeBarrier<PIPE_V>();
                // Match routed quantization: round in FP32 before narrowing.
                auto quantInt32 = tempFp32.template ReinterpretCast<int32_t>();
                auto quantHalf = tempFp32.template ReinterpretCast<half>();
                AscendC::Cast(quantInt32, tempFp32, AscendC::RoundMode::CAST_RINT, k);
                AscendC::SetDeqScale(static_cast<half>(1.0F));
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(quantHalf, quantInt32, AscendC::RoundMode::CAST_ROUND, k);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(quantLocal, quantHalf, AscendC::RoundMode::CAST_TRUNC, k);
                AscendC::PipeBarrier<PIPE_V>();

                if constexpr (kSharedIsW4A8) {
                    AscendC::Cast(highHalf, quantLocal, AscendC::RoundMode::CAST_NONE, k);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Muls(highHalf, highHalf, static_cast<half>(0.0625F), k);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Cast(highI4, highHalf, AscendC::RoundMode::CAST_FLOOR, k);

                    const uint32_t repeat = (k / 2U) / 128U;
                    const uint32_t tail = (k % 256U) / 2U;
                    if (repeat > 0U) {
                        AscendC::And(lowHalf.template ReinterpretCast<int16_t>(),
                                     quantLocal.template ReinterpretCast<int16_t>(), lowMask, 128U, repeat,
                                     {1U, 1U, 1U, 8U, 8U, 0U});
                    }
                    if (tail > 0U) {
                        AscendC::And(lowHalf[repeat * 128U].template ReinterpretCast<int16_t>(),
                                     quantLocal[repeat * 256U].template ReinterpretCast<int16_t>(), lowMask, tail, 1U,
                                     {1U, 1U, 1U, 8U, 8U, 0U});
                    }
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Cast(lowHalf2, lowHalf.template ReinterpretCast<int8_t>(), AscendC::RoundMode::CAST_NONE,
                                  k);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Adds(highHalf, lowHalf2, static_cast<half>(-8.0F), k);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Cast(lowI4, highHalf, AscendC::RoundMode::CAST_NONE, k);
                }

                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID6);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID6);
                if constexpr (kSharedIsW4A8) {
                    AscendC::DataCopy(output[static_cast<uint64_t>(row) * k], highI4.template ReinterpretCast<int8_t>(),
                                      k / 2U);
                    AscendC::DataCopy(output[static_cast<uint64_t>(row) * k + k / 2U],
                                      lowI4.template ReinterpretCast<int8_t>(), k / 2U);
                } else {
                    AscendC::DataCopyPad(output[static_cast<uint64_t>(row) * k], quantLocal,
                                         {1U, static_cast<uint32_t>(k * sizeof(int8_t)), 0U, 0U, 0U});
                }
                reduceMaxLocal.SetValue(0U, dequantScale);
                AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID7);
                AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID7);
                AscendC::DataCopyPad(outputScale[row], reduceMaxLocal,
                                     {1U, static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U});
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
            }
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
            {
                AscendC::SyncAll<true>();
            }
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNCFLAG_SHARED_INPUT_V2C);
        }
    }

    // BF16 consumes original X directly. Quantized paths consume the dense
    // input published above; in both cases GMM1 overlaps routed preparation.
    CATLASS_DEVICE
    void SharedGMM1(Params const& params)
    {
        BlockScheduler blockScheduler;
        SharedBlockMmad blockMmad(resource);
        AscendC::GlobalTensor<SharedElementA> sharedInput;
        AscendC::GlobalTensor<SharedElementB> sharedWeight;
        AscendC::GlobalTensor<ElementScale> sharedWeightScale;
        if constexpr (kSharedIsQuant) {
            sharedInput.SetGlobalBuffer(reinterpret_cast<__gm__ SharedElementA*>(workspaceInfo.ptrSharedInput));
        } else {
            sharedInput.SetGlobalBuffer(reinterpret_cast<__gm__ SharedElementA*>(params.ptrA));
        }

        uint32_t startCoreIdx = 0U;
        const uint32_t m = params.problemShape.m();
        constexpr uint32_t msdFactor = std::is_same_v<SharedElementB, AscendC::int4b_t> ? 2U : 1U;
        const uint32_t computeM = m * msdFactor;
        const uint32_t n = params.sharedN;
        const uint32_t k = params.problemShape.k();
        for (uint32_t sharedIdx = 0U; sharedIdx < params.sharedExpertNum; ++sharedIdx) {
            sharedWeight.SetGlobalBuffer(reinterpret_cast<__gm__ SharedElementB*>(
                GetTensorAddr<SharedElementB>(sharedIdx, reinterpret_cast<GM_ADDR>(params.ptrSharedB1))));
            if constexpr (kSharedIsQuant) {
                sharedWeightScale.SetGlobalBuffer(reinterpret_cast<__gm__ ElementScale*>(
                    GetTensorAddr<int64_t>(sharedIdx, reinterpret_cast<GM_ADDR>(params.ptrSharedScale1))));
            }
            if (m <= SharedL1TileShape::M) {
                sharedWeight.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            }

            GemmCoord sharedShape{computeM, n, k};
            LayoutA sharedLayoutA = LayoutA{computeM, k};
            LayoutC sharedLayoutC{computeM, n};
            blockScheduler.Update(sharedShape, MakeCoord(SharedL1TileShape::M, SharedL1TileShape::N));
            const uint32_t coreLoops = blockScheduler.GetCoreLoops();
            const uint32_t startLoopIdx = ((coreIdx < startCoreIdx) ? (coreIdx + coreNum) : coreIdx) - startCoreIdx;

            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = blockScheduler.GetActualBlockShape(blockCoord);
                MatrixCoord offsetA{blockCoord.m() * SharedL1TileShape::M, blockCoord.k() * SharedL1TileShape::K};
                MatrixCoord offsetB{blockCoord.k() * SharedL1TileShape::K, blockCoord.n() * SharedL1TileShape::N};
                MatrixCoord offsetC{blockCoord.m() * SharedL1TileShape::M, blockCoord.n() * SharedL1TileShape::N};
                const int64_t gmOffsetA = sharedLayoutA.GetOffset(offsetA);
                const int64_t gmOffsetB = params.layoutSharedB1.GetOffset(offsetB);
                const int64_t gmOffsetC =
                    static_cast<int64_t>(sharedIdx) * computeM * n + sharedLayoutC.GetOffset(offsetC);
                if constexpr (std::is_same_v<SharedElementB, AscendC::int4b_t>) {
                    const int64_t gmOffsetS = blockCoord.n() * SharedL1TileShape::N;
                    blockMmad(sharedWeightScale[gmOffsetS], LayoutScale{params.sharedN}, sharedInput[gmOffsetA],
                              sharedLayoutA, sharedWeight[gmOffsetB], params.layoutSharedB1,
                              gmSharedGmmBuffer[gmOffsetC], sharedLayoutC, actualBlockShape, Callback{}, Callback{});
                } else if constexpr (std::is_same_v<SharedElementB, int8_t>) {
                    const int64_t gmOffsetS = blockCoord.n() * SharedL1TileShape::N;
                    blockMmad(sharedInput[gmOffsetA], sharedLayoutA, sharedWeight[gmOffsetB], params.layoutSharedB1,
                              gmSharedGmmBuffer[gmOffsetC], sharedLayoutC, sharedWeightScale[gmOffsetS],
                              LayoutScale{params.sharedN}, actualBlockShape);
                } else {
                    blockMmad(sharedInput[gmOffsetA], sharedLayoutA, sharedWeight[gmOffsetB], params.layoutSharedB1,
                              gmSharedGmmBuffer[gmOffsetC], sharedLayoutC, gmS, LayoutScale{params.sharedN},
                              actualBlockShape);
                }
            }
            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
        }

        if constexpr (SharedBlockMmad::DispatchPolicy::ASYNC) {
            blockMmad.SynchronizeBlock();
        }
        blockMmad.Finalize(0, SYNCFLAG_SHARED_GMM1_C2V);
    }

    // AIV publishes the activated shared input while routed GMM1 is running.
    // Mixed INT8/BF16 GMM2 writes directly to the BF16 shared-result buffer.
    CATLASS_DEVICE
    void SharedGMM2Bf16(Params const& params, uint32_t tokenBase, uint32_t rows)
    {
        using SharedMmad = SharedBf16Mmad_;
        using SharedTile = typename SharedMmad::L1TileShape;
        static_assert(MegaMoeSharedGmm2::CHUNK % SharedTile::M == 0, "Shared chunks must not split a baseline M tile");
        BlockScheduler scheduler;
        SharedMmad mm(resource);
        const uint32_t m = params.problemShape.m();
        const uint32_t n = params.problemShape.k();
        const uint32_t k = params.sharedN / 2U;
        // Keep the full-batch tile -> physical AIC mapping. ENABLE_SHUFFLE_K
        // starts K at GetBlockIdx()%kTileCount: a local-M reswizzle would
        // change per-element accumulation order even without splitting K.
        const GemmCoord shape{m, n, k};
        const LayoutA layoutA{rows, k};
        const LayoutC layoutC{rows, n};
        AscendC::GlobalTensor<bfloat16_t> activation, weight, output;
        activation.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(workspaceInfo.ptrSharedActivation));
        output.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(workspaceInfo.ptrSharedResult));
        uint32_t startCore = 0U;
        for (uint32_t expert = 0U; expert < params.sharedExpertNum; ++expert) {
            weight.SetGlobalBuffer(GetTensorAddr<bfloat16_t>(expert, reinterpret_cast<GM_ADDR>(params.ptrSharedB2)));
            if (m <= SharedTile::M) {
                weight.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            }
            scheduler.Update(shape, MakeCoord(SharedTile::M, SharedTile::N));
            const uint32_t loops = scheduler.GetCoreLoops();
            const uint32_t first = ((coreIdx < startCore) ? coreIdx + coreNum : coreIdx) - startCore;
            for (uint32_t loop = first; loop < loops; loop += coreNum) {
                const GemmCoord block = scheduler.GetBlockCoord(loop);
                const uint32_t fullRow = block.m() * SharedTile::M;
                if (fullRow < tokenBase || fullRow >= tokenBase + rows) {
                    continue;
                }
                const GemmCoord actual = scheduler.GetActualBlockShape(block);
                const MatrixCoord a{fullRow - tokenBase, block.k() * SharedTile::K};
                const MatrixCoord b{block.k() * SharedTile::K, block.n() * SharedTile::N};
                const MatrixCoord c{fullRow - tokenBase, block.n() * SharedTile::N};
                // Full-batch expert strides remain unchanged, including tails.
                const uint64_t aOffset =
                    MegaMoeSharedGmm2::ExpertOffset(expert, m, tokenBase, k) + layoutA.GetOffset(a);
                const uint64_t cOffset =
                    MegaMoeSharedGmm2::ExpertOffset(expert, m, tokenBase, n) + layoutC.GetOffset(c);
                mm(activation[aOffset], layoutA, weight[params.layoutSharedB2.GetOffset(b)], params.layoutSharedB2,
                   output[cOffset], layoutC, gmS2, params.layoutScale2, actual);
            }
            startCore = (startCore + loops) % coreNum;
        }
        if constexpr (SharedMmad::DispatchPolicy::ASYNC) {
            mm.SynchronizeBlock();
        }
        mm.Finalize(0, SYNCFLAG_SHARED_GMM2_C2V);
    }

    CATLASS_DEVICE
    void SharedGMM2(Params const& params, uint32_t tokenBase, uint32_t rows)
    {
        if constexpr (SharedGmm2Bf16_) {
            SharedGMM2Bf16(params, tokenBase, rows);
            return;
        }
        BlockScheduler blockScheduler;
        SharedBlockMmad blockMmad(resource);
        AscendC::GlobalTensor<SharedElementA> sharedActivation;
        AscendC::GlobalTensor<SharedElementB> sharedWeight;
        AscendC::GlobalTensor<ElementScale> sharedWeightScale;
        AscendC::GlobalTensor<SharedElementC> sharedOutput;
        sharedActivation.SetGlobalBuffer(reinterpret_cast<__gm__ SharedElementA*>(workspaceInfo.ptrSharedActivation));
        if constexpr (kSharedIsQuant) {
            sharedOutput.SetGlobalBuffer(reinterpret_cast<__gm__ SharedElementC*>(workspaceInfo.ptrSharedGmmBuffer));
        } else {
            sharedOutput.SetGlobalBuffer(reinterpret_cast<__gm__ SharedElementC*>(workspaceInfo.ptrSharedResult));
        }

        uint32_t startCoreIdx = 0U;
        const uint32_t m = params.problemShape.m();
        constexpr uint32_t msdFactor = std::is_same_v<SharedElementB, AscendC::int4b_t> ? 2U : 1U;
        const uint32_t computeM = m * msdFactor;
        const uint32_t n = params.problemShape.k();
        const uint32_t k = params.sharedN / 2U;
        for (uint32_t sharedIdx = 0U; sharedIdx < params.sharedExpertNum; ++sharedIdx) {
            sharedWeight.SetGlobalBuffer(reinterpret_cast<__gm__ SharedElementB*>(
                GetTensorAddr<SharedElementB>(sharedIdx, reinterpret_cast<GM_ADDR>(params.ptrSharedB2))));
            if constexpr (kSharedIsQuant) {
                sharedWeightScale.SetGlobalBuffer(reinterpret_cast<__gm__ ElementScale*>(
                    GetTensorAddr<int64_t>(sharedIdx, reinterpret_cast<GM_ADDR>(params.ptrSharedScale2))));
            }
            if (m <= SharedL1TileShape::M) {
                sharedWeight.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            }

            GemmCoord sharedShape{rows * msdFactor, n, k};
            LayoutA sharedLayoutA = LayoutA{computeM, k};
            LayoutC sharedLayoutC{computeM, n};
            blockScheduler.Update(sharedShape, MakeCoord(SharedL1TileShape::M, SharedL1TileShape::N));
            const uint32_t coreLoops = blockScheduler.GetCoreLoops();
            const uint32_t startLoopIdx = ((coreIdx < startCoreIdx) ? (coreIdx + coreNum) : coreIdx) - startCoreIdx;

            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = blockScheduler.GetActualBlockShape(blockCoord);
                MatrixCoord offsetA{blockCoord.m() * SharedL1TileShape::M, blockCoord.k() * SharedL1TileShape::K};
                MatrixCoord offsetB{blockCoord.k() * SharedL1TileShape::K, blockCoord.n() * SharedL1TileShape::N};
                MatrixCoord offsetC{blockCoord.m() * SharedL1TileShape::M, blockCoord.n() * SharedL1TileShape::N};
                const int64_t gmOffsetA =
                    MegaMoeSharedGmm2::ExpertOffset(sharedIdx, computeM, tokenBase * msdFactor, k) +
                    sharedLayoutA.GetOffset(offsetA);
                const int64_t gmOffsetB = params.layoutSharedWeight2.GetOffset(offsetB);
                const int64_t gmOffsetC =
                    MegaMoeSharedGmm2::ExpertOffset(sharedIdx, computeM, tokenBase * msdFactor, n) +
                    sharedLayoutC.GetOffset(offsetC);
                if constexpr (std::is_same_v<SharedElementB, AscendC::int4b_t>) {
                    const int64_t gmOffsetS = blockCoord.n() * SharedL1TileShape::N;
                    blockMmad(sharedWeightScale[gmOffsetS], params.layoutScale2, sharedActivation[gmOffsetA],
                              sharedLayoutA, sharedWeight[gmOffsetB], params.layoutSharedWeight2,
                              sharedOutput[gmOffsetC], sharedLayoutC, actualBlockShape, Callback{}, Callback{});
                } else if constexpr (std::is_same_v<SharedElementB, int8_t>) {
                    const int64_t gmOffsetS = blockCoord.n() * SharedL1TileShape::N;
                    blockMmad(sharedActivation[gmOffsetA], sharedLayoutA, sharedWeight[gmOffsetB],
                              params.layoutSharedWeight2, sharedOutput[gmOffsetC], sharedLayoutC,
                              sharedWeightScale[gmOffsetS], params.layoutScale2, actualBlockShape);
                } else {
                    blockMmad(sharedActivation[gmOffsetA], sharedLayoutA, sharedWeight[gmOffsetB],
                              params.layoutSharedWeight2, sharedOutput[gmOffsetC], sharedLayoutC, gmS2,
                              params.layoutScale2, actualBlockShape);
                }
            }
            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
        }

        if constexpr (SharedBlockMmad::DispatchPolicy::ASYNC) {
            blockMmad.SynchronizeBlock();
        }
        blockMmad.Finalize(0, SYNCFLAG_SHARED_GMM2_C2V);
    }

    CATLASS_DEVICE
    void GMM1(Params const& params, int64_t chunkIdx, W1PrefetchRange w1PrefetchRangeThisRound)
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

        // 说明：原 GMM1 入口的 "等待 AIV 完成 cumsum" 初始等待已外移到 AIC operator() 的
        // 轮循环内（轮表就绪信号），syncgmmIdx 由调用方维护并已指向第一个 per-group 信号。

        AscendC::GlobalTensor<ElementB> gmB1;

        for (uint32_t groupIdx = 0; groupIdx < params.expertPerRank; ++groupIdx) {
            uint32_t currentM = cumsumMM.GetValue((params.EP - 1) * params.expertPerRank + groupIdx);
            if (preCurrentmSum >= params.maxOutputSize) {
                currentM = 0;
            } else if (preCurrentmSum + currentM >= params.maxOutputSize) {
                currentM = params.maxOutputSize - preCurrentmSum;
            }

            if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
                currentM = currentM * GATE_UP_ROWS_PER_TOKEN;
            }

            if (UseDecodeFastPath<BatchedDispatch>(params) && currentM == 0 &&
                groupIdx + 1 != params.epilogueGranularity && groupIdx + 1 != params.expertPerRank) {
                if (params.listLen == 1) {
                    gmGroupOffsetB += static_cast<int64_t>(params.problemShape.k()) * params.problemShape.n();
                }
                continue;
            }
            int32_t arrayGroupIdx = params.listLen == 1 ? 0 : groupIdx;
            gmB1.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB*>(
                GetTensorAddr<ElementB>(arrayGroupIdx, reinterpret_cast<GM_ADDR>(params.ptrB1))));
            gmS.SetGlobalBuffer(reinterpret_cast<__gm__ ElementScale*>(
                GetTensorAddr<int64_t>(arrayGroupIdx, reinterpret_cast<GM_ADDR>(params.ptrScale1))));

            if constexpr (!std::is_same_v<ElementB, AscendC::int4b_t>) {
                AscendC::PipeBarrier<PIPE_ALL>();
            }

            // Only experts in this round's issued range bypass the small-M DISABLE rule.
            // Reset explicitly on every group: gmB1 is reused across experts.
            if (currentM > 0 && currentM <= L1TileShape::M && !w1PrefetchRangeThisRound.Contains(groupIdx)) {
                gmB1.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            } else {
                gmB1.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_NORMAL);
            }

            GemmCoord inGroupProblemShape{currentM, params.problemShape.n(), params.problemShape.k()};
            LayoutA layoutA = params.layoutA.GetTileLayout(inGroupProblemShape.GetCoordMK());
            LayoutB layoutB1 = params.layoutB1;
            LayoutScale layoutScale = params.layoutScale1;
            LayoutC layoutC(inGroupProblemShape.m(), inGroupProblemShape.n(), params.gmmOutPreRowStride);
            blockScheduler.Update(inGroupProblemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
            uint32_t coreLoops = blockScheduler.GetCoreLoops();
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
                // Scalar submission interval, not an added Cube completion wait.
                // Compute initial location in logical coordinates
                int64_t gmOffsetA, gmOffsetB, gmOffsetC;
                Detail::GetGmmTileOffsets<L1TileShape>(blockCoord, layoutA, layoutB1, layoutC, gmOffsetA, gmOffsetB,
                                                       gmOffsetC);
                if (currentM > 0) {
                    if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
                        int64_t gmOffsetS = blockCoord.n() * L1TileShape::N +
                                            (params.listLen == 1 ? groupIdx * params.problemShape.n() : 0);
                        if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
                            blockMmad(gmS[gmOffsetS], layoutScale, gmA1I4[gmGroupOffsetA + gmOffsetA], layoutA,
                                      gmB1[gmGroupOffsetB + gmOffsetB], layoutB1, gmC[gmGroupOffsetC + gmOffsetC],
                                      layoutC, actualBlockShape, Callback{}, Callback{});
                        } else {
                            blockMmad(gmS[gmOffsetS], layoutScale, gmA1I4[gmGroupOffsetA + gmOffsetA], layoutA,
                                      gmB1[gmGroupOffsetB + gmOffsetB], layoutB1, gmC[gmGroupOffsetC + gmOffsetC],
                                      layoutC, actualBlockShape);
                        }
                    } else if constexpr (std::is_same_v<ElementA, int8_t>) {
                        int64_t gmOffsetS = blockCoord.n() * L1TileShape::N +
                                            (params.listLen == 1 ? groupIdx * params.problemShape.n() : 0);
                        blockMmad(gmA[gmGroupOffsetA + gmOffsetA], layoutA, gmB1[gmGroupOffsetB + gmOffsetB], layoutB1,
                                  gmC[gmGroupOffsetC + gmOffsetC], layoutC, gmS[gmOffsetS], layoutScale,
                                  actualBlockShape);
                    } else {
                        blockMmad(gmA[gmGroupOffsetA + gmOffsetA], layoutA, gmB1[gmGroupOffsetB + gmOffsetB], layoutB1,
                                  gmC[gmGroupOffsetC + gmOffsetC], layoutC, gmS, layoutScale, actualBlockShape);
                    }
                }
            }

            if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
                if (IsActivationBoundary<BatchedDispatch>(params, groupIdx)) {
                    syncLoopIdx++;
                    if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
                        blockMmad.SynchronizeBlock();
                    }
                    blockMmad.Finalize(syncLoopIdx, SYNCFLAGC2V + chunkIdx * PERMUTE_CHUNK_FLAG_STRIDE);
                }
            } else {
                if ((groupIdx + 1) == params.epilogueGranularity && (groupIdx < params.expertPerRank - 1)) {
                    syncLoopIdx++;
                    if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
                        blockMmad.SynchronizeBlock();
                    }
                    blockMmad.Finalize(syncLoopIdx, SYNCFLAGC2V + chunkIdx * PERMUTE_CHUNK_FLAG_STRIDE);
                }
            }

            if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
                preCurrentmSum += currentM / GATE_UP_ROWS_PER_TOKEN;
            } else {
                preCurrentmSum += currentM;
            }
            // 每个专家之间在k小于k2的时候不是密排，所以这里不取k
            gmGroupOffsetA += inGroupProblemShape.m() * params.gmm1InputExpertSlotWidth;
            if (params.listLen == 1) {
                gmGroupOffsetB += inGroupProblemShape.k() * inGroupProblemShape.n();
            }
            gmGroupOffsetC += inGroupProblemShape.m() * params.gmmOutPreRowStride;
            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
        }

        for (; syncGroupIdx < params.expertPerRank; syncGroupIdx += readyExperts) {
            AscendC::CrossCoreWaitFlag<0x2>(syncgmmIdx / CROSS_CORE_FLAG_MAX_SET_COUNT);
            syncgmmIdx++;
        }

        if constexpr (!std::is_same_v<ElementB, AscendC::int4b_t>) {
            if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
                blockMmad.SynchronizeBlock();
            }
            blockMmad.Finalize(syncLoopIdx + 1, SYNCFLAGC2V + chunkIdx * PERMUTE_CHUNK_FLAG_STRIDE);
        }
    }

    CATLASS_DEVICE
    void GMM2(Params const& params, int64_t chunkIdx)
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

        if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
            AscendC::PipeBarrier<PIPE_ALL>();
        }

        int64_t preCurrentmSum = 0;
        int32_t syncLoopIdx = -1;
        uint32_t lastDequantExpertNum = params.expertPerRank;
        if constexpr (!std::is_same_v<ElementB, AscendC::int4b_t>) {
            if (params.epilogueGranularity < params.expertPerRank) {
                lastDequantExpertNum = params.expertPerRank - params.epilogueGranularity;
            }
        }

        for (uint32_t groupIdx = 0; groupIdx < params.expertPerRank; ++groupIdx) {
            uint32_t currentM = cumsumMM.GetValue((params.EP - 1) * params.expertPerRank + groupIdx);
            if (preCurrentmSum >= params.maxOutputSize) {
                currentM = 0;
            } else if (preCurrentmSum + currentM > params.maxOutputSize) {
                currentM = params.maxOutputSize - preCurrentmSum;
            }

            if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
                currentM = currentM * GATE_UP_ROWS_PER_TOKEN;
            }

            if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
                if (groupIdx == 0 || IsActivationBoundary<BatchedDispatch>(params, groupIdx - 1)) {
                    AscendC::CrossCoreWaitFlag<0x2>(SYNCFLAGV2C + chunkIdx * PERMUTE_CHUNK_FLAG_STRIDE);
                }
            } else {
                if (params.expertPerRank > lastDequantExpertNum &&
                    groupIdx + (UseDecodeFastPath<BatchedDispatch>(params) ? 0U : 1U) ==
                        params.expertPerRank - lastDequantExpertNum) {
                    AscendC::CrossCoreWaitFlag<0x2>(SYNCFLAGV2C + chunkIdx * PERMUTE_CHUNK_FLAG_STRIDE);
                }
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
                if (currentM > 0) {
                    if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
                        int64_t gmOffsetS = blockCoord.n() * L1TileShape::N + (params.listLen == 1 ? groupIdx * n2 : 0);
                        if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
                            blockMmad(gmS2[gmOffsetS], layoutScale, gmA2I4[gmGroupOffsetA + gmOffsetA], layoutA,
                                      gmB2[gmGroupOffsetB + gmOffsetB], layoutB2, gmC2[gmGroupOffsetC + gmOffsetC],
                                      layoutC, actualBlockShape, Callback{}, Callback{}, syncLoopIdx, 0);
                        }
                    } else if constexpr (std::is_same_v<ElementA, int8_t>) {
                        int64_t gmOffsetS = blockCoord.n() * L1TileShape::N + (params.listLen == 1 ? groupIdx * n2 : 0);
                        blockMmad(gmPermutedToken[gmGroupOffsetA + gmOffsetA], layoutA,
                                  gmB2[gmGroupOffsetB + gmOffsetB], layoutB2, gmC2[gmGroupOffsetC + gmOffsetC], layoutC,
                                  gmS2[gmOffsetS], layoutScale, actualBlockShape, syncLoopIdx, 0);
                    } else {
                        blockMmad(gmPermutedToken[gmGroupOffsetA + gmOffsetA], layoutA,
                                  gmB2[gmGroupOffsetB + gmOffsetB], layoutB2, gmC2[gmGroupOffsetC + gmOffsetC], layoutC,
                                  gmS2, layoutScale, actualBlockShape, syncLoopIdx, 0);
                    }
                }
            }
            if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
                preCurrentmSum += currentM / GATE_UP_ROWS_PER_TOKEN;
            } else {
                preCurrentmSum += currentM;
            }
            gmGroupOffsetA += inGroupProblemShape.m() * inGroupProblemShape.k();
            if (params.listLen == 1) {
                gmGroupOffsetB += inGroupProblemShape.k() * inGroupProblemShape.n();
            }
            gmGroupOffsetC += inGroupProblemShape.m() * inGroupProblemShape.n();

            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
        }
        if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
            blockMmad.SynchronizeBlock();
            if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
                blockMmad.Finalize(params.expertPerRank - 1, 0);
            }
        }
        if constexpr (std::is_same_v<ElementB, int8_t>) {
            if (isCombineV1) {
                blockMmad.Finalize(params.expertPerRank - 1, 0);
            }
        }
    }

    CATLASS_DEVICE
    void AllGatherTokenPerExpert(Params const& params, int64_t localTokenPerExpertOffset, uint32_t commCoreNum)
    {
        uint32_t numPerCore = paddedExpertNumAligned;
        AscendC::LocalTensor<int32_t> tmpBuffer = resource.ubBuf.template GetBufferByByte<int32_t>(0);
        AscendC::LocalTensor<int32_t> prevSumBuf = tmpBuffer[numPerCore];

        int32_t runtimeRank = RuntimeRank(params);
        // Dedup 打包段元素数（slots+probs 两段，N = layoutChunkTokens·topK）
        const int64_t dedupLayoutChunkTokens =
            (params.problemShape.m() < GetPermuteChunk(params)) ? params.problemShape.m() : GetPermuteChunk(params);
        const int32_t dedupPackElems =
            static_cast<int32_t>(DEDUP_PACK_SECTION_COUNT * dedupLayoutChunkTokens * params.topK);
        for (int32_t dstEpIdx = coreIdx; dstEpIdx < params.EP; dstEpIdx += (MASK_ROUTING ? commCoreNum : coreNum)) {
            if (dstEpIdx == runtimeRank || IsRankMasked(dstEpIdx)) {
                continue;
            }
            AscendC::GlobalTensor<int32_t> srcAddress;
            srcAddress.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(shmem() + localTokenPerExpertOffset));
            AscendC::GlobalTensor<int32_t> dstAddress;
            __gm__ void* dstPeermemPtr = shmem(localTokenPerExpertOffset, dstEpIdx);
            dstAddress.SetGlobalBuffer((__gm__ int32_t*)dstPeermemPtr);

            if constexpr (DedupOn_) {
                // 打包段先于 TPE 推送：同核同源同目的 MTE3 按序，远端见 TPE flag 即知打包段
                // 已落地，不新增跨卡同步点
                const int64_t packOffset = peermemInfo.offsetExpandedDstToSrcRow +
                                           runtimeRank * static_cast<int64_t>(dedupPackElems) * sizeof(int32_t);
                AscendC::GlobalTensor<int32_t> packSrc;
                packSrc.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(shmem() + packOffset));
                AscendC::GlobalTensor<int32_t> packDst;
                packDst.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(shmem(packOffset, dstEpIdx)));
                CopyGMToGM<int32_t>(packDst, packSrc, dedupPackElems, DEDUP_PACK_UB_MOVE_ELEMS);
            }

            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            using TType = Gemm::GemmType<int32_t, layout::RowMajor>;
            using CopyGmToUb = Epilogue::Tile::CopyGm2Ub<ArchTag, TType>;
            using CopyUbToGm = Epilogue::Tile::CopyUb2Gm<ArchTag, TType>;
            CopyGmToUb copyGmToUb;
            CopyUbToGm copyUbToGm;
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            copyGmToUb(tmpBuffer, srcAddress, layout::RowMajor{1, numPerCore}, layout::RowMajor{1, numPerCore});
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::Adds(tmpBuffer, tmpBuffer, 0x800000, numPerCore);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            copyUbToGm(dstAddress, tmpBuffer, layout::RowMajor{1, numPerCore}, layout::RowMajor{1, numPerCore});
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
        for (int32_t dstEpIdx = coreIdx; dstEpIdx < params.EP; dstEpIdx += (MASK_ROUTING ? commCoreNum : coreNum)) {
            bool rankAvailable = !IsRankMasked(dstEpIdx);
            if (dstEpIdx != runtimeRank) {
                int32_t intPer512 = CACHE_LINE / sizeof(int);
                for (int32_t checkIdx = 0; checkIdx < paddedExpertNumAligned; checkIdx += intPer512) {
                    __gm__ int32_t* sync_check =
                        reinterpret_cast<__gm__ int32_t*>(shmem() + peermemInfo.offsetPeerTokenPerExpert) +
                        tokenPerExpertLayout(dstEpIdx, 0, checkIdx);
                    if (!rankAvailable) {
                        break;
                    }
                    gm_signal_wait_until_ne(sync_check, 0);
                }
                if (rankAvailable) {
                    AscendC::DataCopy(tmpBuffer, tokenPerExpert[tokenPerExpertLayout(dstEpIdx, 0, 0)], numPerCore);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                    AscendC::Adds(tmpBuffer, tmpBuffer, -0x800000, numPerCore);
                } else {
                    AscendC::Duplicate(tmpBuffer, 0, numPerCore);
                }

                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::DataCopy(tokenPerExpert[tokenPerExpertLayout(dstEpIdx, 0, 0)], tmpBuffer, numPerCore);
            } else {
                AscendC::DataCopy(tmpBuffer, tokenPerExpert[tokenPerExpertLayout(dstEpIdx, 0, 0)], numPerCore);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            if constexpr (MASK_ROUTING) {
                continue;
            }
            int32_t prevSum = 0;
            int32_t start = runtimeRank * params.expertPerRank;
            for (int32_t i = 0; i < start; i++) {
                prevSum += tmpBuffer(i);
            }
            for (int32_t k = 0; k < params.expertPerRank; k++) {
                prevSumBuf(k) = prevSum;
                prevSum += tmpBuffer(start + k);
            }
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::DataCopyPad(
                preSumBeforeRank[dstEpIdx * params.expertPerRank], prevSumBuf,
                AscendC::DataCopyParams{1, static_cast<uint16_t>(params.expertPerRank * sizeof(int32_t)), 0, 0});
            // preSum 为 DMA 写，消费侧经 UB 缓存读取，由函数尾 SyncAll 收口可见，无需
            // DCCI；若改为跨核标量直读 GM，须加 DCCI 清洗。
        }

        if constexpr (!MASK_ROUTING) {
            AscendC::SyncAll<true>();
        }
    }

    CATLASS_DEVICE
    void ResetTokenPerExpert(int32_t num)
    {
        if (coreIdx != coreNum - 1 || num <= 0) {
            return;
        }
        // The full EP-by-padded-expert region can exceed UB, so reuse one UB-sized zero tile for all GM chunks.
        int32_t resetNumPerLoop = num < UB_MOVE_NUM ? num : UB_MOVE_NUM;
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::LocalTensor<int32_t> tmp = resource.ubBuf.template GetBufferByByte<int32_t>(0);
        AscendC::Duplicate(tmp, 0, resetNumPerLoop);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        for (int32_t offset = 0; offset < num; offset += resetNumPerLoop) {
            int32_t remainingNum = num - offset;
            int32_t currentNum = remainingNum < resetNumPerLoop ? remainingNum : resetNumPerLoop;
            AscendC::DataCopy(tokenPerExpert[offset], tmp, currentNum);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    }

    CATLASS_DEVICE
    void CombineSetFlag()
    {
        if constexpr (std::is_same_v<ElementB, int8_t>) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
            AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID3);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        } else if constexpr (!std::is_same_v<ElementB, AscendC::int4b_t>) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        }
    }

    CATLASS_DEVICE
    void CombineWaitFlag()
    {
        if constexpr (std::is_same_v<ElementB, int8_t>) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        } else if constexpr (!std::is_same_v<ElementB, AscendC::int4b_t>) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        }
    }

    CATLASS_DEVICE
    void UpdateDequantSums(Params const& params, int32_t groupIdx, uint32_t currentM, uint32_t& dequantSumTemp,
                           uint32_t& dequantSum1, uint32_t& dequantSum2)
    {
        if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
            AccumulateSingleDequantRows(currentM, params.maxOutputSize, dequantSumTemp);

            if (IsActivationBoundary<BatchedDispatch>(params, groupIdx)) {
                nSyncSwiglu++;
                dequantSum[nSyncSwiglu] = dequantSumTemp;
            }
        } else {
            AccumulateSplitDequantRows(currentM, groupIdx, params.epilogueGranularity, params.maxOutputSize,
                                       dequantSum1, dequantSum2);
        }
    }

    // Dedup 构件：此处仅声明，定义在 mega_moe_kernel_a3_dedup.hpp（本文件末尾包含）
    CATLASS_DEVICE
    void ResetDistTokenFirstPosV2(GM_ADDR distTokenFirstPos, GM_ADDR distTokenLastPos, int32_t num, uint32_t coreIdx,
                                  uint32_t coreNum);

    CATLASS_DEVICE
    int32_t DedupGroupOf(int32_t row, const AscendC::LocalTensor<int32_t>& groupBase, int32_t er);

    CATLASS_DEVICE
    void DedupTokenDispatchV2(Params const& params, uint16_t& syncgmm1Idx, uint32_t& dequantSumTemp);

    // ================= Dedup Combine v2 =================
    // 切片发射：组第 s 片 = 组内行 [s·SR, min((s+1)·SR, coreLenG))，rowMeta 与 scale2
    // 一次 MTE2_S Set 覆盖，缓冲/事件按 s 奇偶轮转（EVENT_ID6/7）。片宽/缓冲常量见
    // const_args.hpp（DEDUP_COMBINE_SLICE_*）。
    CATLASS_DEVICE
    void DedupIssueRowMetaSlice(AscendC::GlobalTensor<int32_t>& rowMetaGm, int32_t preGroupSumG, uint32_t coreOffsetG,
                                uint32_t coreLenG, uint32_t sliceRows, uint32_t s,
                                AscendC::LocalTensor<int32_t> (&ubRowMetaSl)[2],
                                AscendC::LocalTensor<float> (&ubScale2Sl)[2], AscendC::DataCopyExtParams& sliceParamsRM,
                                AscendC::DataCopyExtParams& sliceParamsS2);

    CATLASS_DEVICE
    typename BlockEpilogueDedup::RowCtx DedupDeriveRow(
        AscendC::GlobalTensor<int32_t>& rowMetaGm, AscendC::LocalTensor<int32_t> (&ubRowMetaSl)[2],
        AscendC::LocalTensor<float> (&ubRowMetaSlF)[2], AscendC::LocalTensor<float> (&ubScale2Sl)[2],
        AscendC::DataCopyExtParams& sliceParamsRM, AscendC::DataCopyExtParams& sliceParamsS2, uint32_t i,
        uint32_t coreOffsetG, uint32_t coreLenG, int32_t preGroupSumG, int32_t n2T, int32_t layoutChunkTokensT,
        uint32_t& issued, uint32_t totalSlices);

    CATLASS_DEVICE
    void DedupCombineV2(Params const& params);

    template <typename T>
    CATLASS_DEVICE void CopyBf16BatchRows(AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, uint32_t elements,
                                          bool& started, uint32_t& slot)
    {
        // Same storage regions as the original BF16 CopyGMToGM helper.
        auto slot0 = resource.ubBuf.template GetBufferByByte<T>(0);
        auto slot1 = resource.ubBuf.template GetBufferByByte<T>(96 * 1024);
        slot0.SetSize(UB_MOVE_NUM);
        slot1.SetSize(UB_MOVE_NUM);
        if (!started) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
            started = true;
        }
        using Type = Gemm::GemmType<T, layout::RowMajor>;
        Epilogue::Tile::CopyGm2Ub<ArchTag, Type> load;
        Epilogue::Tile::CopyUb2Gm<ArchTag, Type> store;
        for (uint32_t offset = 0; offset < elements; offset += UB_MOVE_NUM) {
            const uint32_t count = min(static_cast<uint32_t>(UB_MOVE_NUM), elements - offset);
            const layout::RowMajor shape{1, count};
            const AscendC::TEventID event = slot == 0 ? EVENT_ID0 : EVENT_ID1;
            auto buffer = slot == 0 ? slot0 : slot1;
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
            load(buffer, src[offset], shape, shape);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(event);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event);
            store(dst[offset], buffer, shape, shape);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
            slot ^= 1U;
        }
    }

    CATLASS_DEVICE void DispatchExpertBatch(Params const& params, uint32_t firstGroup, uint32_t groups,
                                            uint32_t firstGroupBase, int32_t& pingpongIdx)
    {
        const uint32_t tasks = groups * params.EP;
        const uint32_t workers = Detail::DispatchWorkerCount(params.decodeFastPath, coreNum, tasks, 4U);
        const uint32_t worker = coreIdx % workers;
        const bool useBf16BatchCopy = std::is_same_v<ElementB, bfloat16_t> && !params.decodeFastPath &&
                                      params.dispatchExpertBatch == MegaMoeConfig::DEFAULT_DISPATCH_EXPERTS;
        bool bf16CopyStarted = false;
        uint32_t bf16Slot = 0;
        for (uint32_t task = coreIdx / workers; task < tasks; task += coreNum / workers) {
            const uint32_t peer = task % params.EP;
            const uint32_t group = firstGroup + task / params.EP;
            if (IsRankMasked(peer))
                continue;
            uint32_t groupBase = firstGroupBase;
            for (uint32_t previous = firstGroup; previous < group; ++previous) {
                groupBase += cumsumMM.GetValue((params.EP - 1) * params.expertPerRank + previous);
            }
            uint32_t rowInGroup = peer == 0 ? 0 : cumsumMM.GetValue((peer - 1) * params.expertPerRank + group);
            uint32_t row = groupBase + rowInGroup;
            if (row >= params.maxOutputSize)
                continue;
            uint32_t rows = ubTokenPerExpertCache.GetValue(peer * expertPerRankAligned + group);
            if (row + rows > params.maxOutputSize)
                rows = params.maxOutputSize - row;
            uint32_t first;
            const uint32_t workerRows = Detail::PartitionDispatchRows(rows, workers, worker, first);
            if (workerRows == 0)
                continue;
            row += first;
            rowInGroup += first;
            const uint64_t sourceRow = ubPreSumBeforeRankCache.GetValue(peer * params.expertPerRank + group) + first;
            AscendC::GlobalTensor<ElementABefore> input;
            input.SetGlobalBuffer(reinterpret_cast<__gm__ ElementABefore*>(shmem(0, peer) + peermemInfo.offsetA));
            const uint64_t outputOffset = static_cast<uint64_t>(groupBase) * params.gmm1InputExpertSlotWidth +
                                          static_cast<uint64_t>(rowInGroup) * params.problemShape.k();
            if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
                FetchAndPreprocessInt8ToInt4(outputOffset, gmPerTokenScale1[row],
                                             input[sourceRow * (params.problemShape.k() + routingPadding_)], workerRows,
                                             params.problemShape.k(), true);
            } else if constexpr (std::is_same_v<ElementB, int8_t>) {
                CopyGMToGMPerToken(gmA[outputOffset], gmPerTokenScale1[row],
                                   input[sourceRow * (params.problemShape.k() + routingPadding_)], workerRows,
                                   params.problemShape.k(), 8, pingpongIdx);
            } else {
                auto dst = gmA[outputOffset];
                auto src = input[sourceRow * params.problemShape.k()];
                const uint32_t elements = workerRows * params.problemShape.k();
                if (useBf16BatchCopy) {
                    CopyBf16BatchRows(dst, src, elements, bf16CopyStarted, bf16Slot);
                } else {
                    CopyGMToGM(dst, src, elements, UB_MOVE_NUM);
                }
            }
        }
        if (bf16CopyStarted) {
            // Drain both stores before the existing all-AIV barrier and Cube readiness.
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        }
    }

    // 接收轮处理：轮内复用原有 dispatch → swiglu → combine 流水。
    // 前置条件：cumsumMM / preSumBeforeRank / UB 路由缓存已指向本轮数据
    // （rounds==1 时为 chunk 级表，rounds>1 时调用方已切换到轮表并刷新 UB 缓存）。
    // 入口发出轮就绪信号（与 AIC 轮级 CrossCoreWaitFlag 配对），轮末 SyncAll
    // 保证轮间串行，C2V/V2C 与 syncgmm flag 槽位按轮复用（AIC 侧每轮同样归零）。
    CATLASS_DEVICE
    void ProcessRecvRound(Params const& params, int64_t chunkIdx, bool runSharedActivation, bool finishSharedGmm2,
                          bool resetMaskEpoch)
    {
        int32_t runtimeRank = RuntimeRank(params);
        // 轮就绪信号：rounds==1 时即 chunk cumsum/路由就绪信号，rounds>1 时为轮表就绪信号
        uint16_t syncgmm1Idx = 0;
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(syncgmm1Idx / CROSS_CORE_FLAG_MAX_SET_COUNT);
        syncgmm1Idx++;

        // dispatch（接收轮局部）；Dedup 实例改走 DedupTokenDispatchV2，下列基线局部量
        // 在 if constexpr 裁剪后不引用
        [[maybe_unused]] uint32_t prevGroupSum1 = 0;
        nSyncSwiglu = 0;
        dequantSum[0] = 0;
        uint32_t dequantSumTemp = 0;
        [[maybe_unused]] uint32_t dequantSum1 = 0;
        [[maybe_unused]] uint32_t dequantSum2 = 0;
        [[maybe_unused]] int32_t pingpongIdx = 0;
        icache_preload(ICACHE_PRELOAD_LINES);
        if constexpr (std::is_same_v<ElementB, int8_t>) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        }
        exceptionDump_.UpdateStage(MC2MegaMoeAdump::Stage::DISPATCH);
        if constexpr (DedupOn_) {
            DedupTokenDispatchV2(params, syncgmm1Idx, dequantSumTemp);
        } else if constexpr (!BatchedDispatch) {
            for (int32_t groupIdx = 0; groupIdx < params.expertPerRank; ++groupIdx) {
                // ----------------------------------------------------------
                // Each core pulls its `dstEpIdx`'s tokens out of the remote
                // rank's local peer-mem and runs the pre-process step.
                // ----------------------------------------------------------
                uint32_t currentM = cumsumMM.GetValue((params.EP - 1) * params.expertPerRank + groupIdx);
                for (int32_t dstEpIdx = coreIdx; dstEpIdx < params.EP; dstEpIdx += coreNum) {
                    if (IsRankMasked(dstEpIdx)) {
                        continue;
                    }
                    uint32_t rowStartInGroup =
                        dstEpIdx == 0 ? 0 : cumsumMM.GetValue((dstEpIdx - 1) * params.expertPerRank + groupIdx);
                    uint32_t rowStart = rowStartInGroup + prevGroupSum1;
                    if (rowStart < params.maxOutputSize) {
                        uint32_t rows = ubTokenPerExpertCache.GetValue(dstEpIdx * expertPerRankAligned + groupIdx);
                        if (rowStart + rows > params.maxOutputSize) {
                            rows = params.maxOutputSize - rowStart;
                        }
                        uint32_t rowSrc =
                            MASK_ROUTING ? 0 :
                                           ubPreSumBeforeRankCache.GetValue(dstEpIdx * params.expertPerRank + groupIdx);
                        GM_ADDR otherRankPtr = shmem(0, dstEpIdx);
                        AscendC::GlobalTensor<ElementABefore> gmRemoteA;
                        gmRemoteA.SetGlobalBuffer(
                            reinterpret_cast<__gm__ ElementABefore*>(otherRankPtr + peermemInfo.offsetA));

                        // 在做dispatch的时候，因为复用了gmm1的A矩阵与swiglu的输出
                        // 所以dispatch的时候每个专家之间间隔从密排改为max（k， k2）
                        // 但是专家之内仍然是密排的
                        int64_t gmOffsetA = static_cast<int64_t>(prevGroupSum1) * params.gmm1InputExpertSlotWidth +
                                            static_cast<int64_t>(rowStartInGroup) * params.problemShape.k();
                        if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
                            int64_t gmOffsetPeer = rowSrc * (params.problemShape.k() + ALIGN_512);
                            FetchAndPreprocessInt8ToInt4(static_cast<uint64_t>(gmOffsetA), gmPerTokenScale1[rowStart],
                                                         gmRemoteA[gmOffsetPeer], rows, params.problemShape.k());
                        } else if constexpr (std::is_same_v<ElementB, int8_t>) {
                            // quant 路径：远端行步长 k+512（行尾含 per-token scale）
                            int64_t gmOffsetPeer = rowSrc * (params.problemShape.k() + ALIGN_512);
                            int32_t ubMoveNum = 2;
                            if constexpr (MASK_ROUTING) {
                                CopyMaskRows(params, dstEpIdx, groupIdx, rows, rowStart, gmOffsetA, pingpongIdx);
                            } else {
                                CopyGMToGMPerToken(gmA[gmOffsetA], gmPerTokenScale1[rowStart], gmRemoteA[gmOffsetPeer],
                                                   rows, params.problemShape.k(), ubMoveNum, pingpongIdx);
                            }
                        } else {
                            // 非 quant 路径：远端 expandedX 行步长 = k（密排），可直接连续拷贝 rows * k
                            int64_t gmOffsetPeer = rowSrc * params.problemShape.k();
                            CopyGMToGM(gmA[gmOffsetA], gmRemoteA[gmOffsetPeer], rows * params.problemShape.k(),
                                       UB_MOVE_NUM);
                        }
                    }
                }
                AscendC::SyncAll<true>();
                AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(syncgmm1Idx / CROSS_CORE_FLAG_MAX_SET_COUNT);
                syncgmm1Idx++;

                prevGroupSum1 += currentM;

                UpdateDequantSums(params, groupIdx, currentM, dequantSumTemp, dequantSum1, dequantSum2);
            }
        } else {
            const uint32_t batchGroups = params.dispatchExpertBatch;
            for (uint32_t firstGroup = 0; firstGroup < params.expertPerRank; firstGroup += batchGroups) {
                const uint32_t groups = min(batchGroups, params.expertPerRank - firstGroup);
                bool hasRows = !UseDecodeFastPath<BatchedDispatch>(params);
                for (uint32_t i = 0; !hasRows && i < groups; ++i) {
                    hasRows = cumsumMM.GetValue((params.EP - 1) * params.expertPerRank + firstGroup + i) != 0;
                }
                if (hasRows) {
                    DispatchExpertBatch(params, firstGroup, groups, prevGroupSum1, pingpongIdx);
                    AscendC::SyncAll<true>();
                }
                for (uint32_t groupIdx = firstGroup; groupIdx < firstGroup + groups; ++groupIdx) {
                    const uint32_t currentM = cumsumMM.GetValue((params.EP - 1) * params.expertPerRank + groupIdx);
                    if (!UseDecodeFastPath<BatchedDispatch>(params) || groupIdx == firstGroup) {
                        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(syncgmm1Idx / CROSS_CORE_FLAG_MAX_SET_COUNT);
                        ++syncgmm1Idx;
                    }
                    prevGroupSum1 += currentM;
                    UpdateDequantSums(params, groupIdx, currentM, dequantSumTemp, dequantSum1, dequantSum2);
                }
            }
        }
        if constexpr (std::is_same_v<ElementB, int8_t>) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        }

        if (runSharedActivation) {
            SharedGMM1Epilogue(params);
        }
        RunPostprocess<BatchedDispatch>(params, chunkIdx, dequantSum1, dequantSum2, finishSharedGmm2);
        if constexpr (MASK_ROUTING) {
            // 同一 chunk 的各轮共享计数和就绪槽位，最后一轮结束后再清零。
            if (resetMaskEpoch) {
                exceptionDump_.UpdateStage(MC2MegaMoeAdump::Stage::RESET_TOKEN_PER_EXPERT);
                AscendC::PipeBarrier<PIPE_ALL>();
                ResetTokenPerExpert(params.EP * paddedExpertNumAligned);
                ResetMaskDataReady(params);
            }
        }
        AscendC::SyncAll<true>();
        if (finishSharedGmm2 && SharedGmm2Chunked(params) && !SharedGmm2WaitAfterCrossRank(params)) {
            const auto range = SharedGmm2Range(params, chunkIdx);
            SharedGMM2Epilogue(params, range.base, range.rows);
        }
    }

    template <bool EnableBatched>
    CATLASS_DEVICE void RunPostprocess(Params const& params, int64_t chunkIdx, uint32_t dequantSum1,
                                       uint32_t dequantSum2, bool finishSharedGmm2)
    {
        using ActivationEpilogue = typename Epilogue::Block::SelectPostprocess<BlockEpilogue1, EnableBatched>::Type;
        using CombineEpilogue = typename Epilogue::Block::SelectPostprocess<
            BlockEpilogue3, EnableBatched && std::is_same_v<ElementB, AscendC::int4b_t>>::Type;
        const int32_t runtimeRank = RuntimeRank(params);
        uint32_t n2 = params.problemShape.k();

        typename BlockEpilogue2::Params epilogueParams2{
            static_cast<int32_t>(params.EP), static_cast<int32_t>(params.expertPerRank),
            reinterpret_cast<__gm__ int32_t*>(shmem() + peermemInfo.offsetPeerTokenPerExpert),
            static_cast<int32_t>(n2)};

        typename CombineEpilogue::Params epilogueParams3{
            static_cast<int32_t>(params.EP),
            static_cast<int32_t>(params.expertPerRank),
            runtimeRank,
            reinterpret_cast<__gm__ int32_t*>(shmem() + peermemInfo.offsetPeerTokenPerExpert),
            params.layoutD2,
            static_cast<int32_t>(n2),
            static_cast<int32_t>(L1TileShape::N),
            shmem,
            peermemInfo.offsetD,
            tokenPerExpertLayout};
        // epilogue 内 tokenPerExpert / preSumBeforeRank 标量读取走本核 UB 缓存
        if constexpr (MASK_ROUTING) {
            epilogueParams3.ptrRouteIndex = workspaceInfo.ptrRouteIndex;
            if constexpr (BLOCKED_OUTPUT_) {
                const int64_t remaining = params.problemShape.m() - chunkIdx * PERMUTE_CHUNK;
                const int64_t tokens = remaining < PERMUTE_CHUNK ? remaining : PERMUTE_CHUNK;
                epilogueParams3.blockedRows = static_cast<uint32_t>(tokens * params.topK);
            }
        }
        epilogueParams3.useUbRouteCache = true;
        epilogueParams3.ubTokenPerExpert = ubTokenPerExpertCache;
        epilogueParams3.ubPreSumBeforeRank = ubPreSumBeforeRankCache;
        epilogueParams3.ubRouteStride = static_cast<int32_t>(expertPerRankAligned);

        uint32_t n = params.problemShape.n();
        using EarlyEpilogue2 = std::conditional_t<MASK_ROUTING, MegaMoeDeferredEpilogue, BlockEpilogue2>;
        EarlyEpilogue2 blockEpilogue2(resource, epilogueParams2);
        CombineEpilogue blockEpilogue3(resource, epilogueParams3);
        exceptionDump_.UpdateStage(MC2MegaMoeAdump::Stage::SWIGLU);
        ActivationEpilogue blockEpilogue1(resource, n);
        if constexpr (EnableBatched) {
            if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
                blockEpilogue1.EnableEightRows(true);
            } else {
                blockEpilogue1.EnableRowBatch(true, resource);
            }
        }
        if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
            for (int32_t syncIdx = 0; syncIdx < nSyncSwiglu; syncIdx++) {
                AscendC::CrossCoreWaitFlag<0x2>(SYNCFLAGC2V + chunkIdx * PERMUTE_CHUNK_FLAG_STRIDE);
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
                                   reinterpret_cast<__gm__ float*>(params.ptrBias1), gmA2I4_I8[gmOffsetD], cumsumMM,
                                   rowStartThisCore, gmPerTokenScale2[rowStartThisCore], params.expertPerRank,
                                   params.EP, runtimeRank, params.listLen, resource, params.epilogueCoreNum,
                                   params.swigluLimit, params.activationCode, params.activationParams1,
                                   params.activationParams2, params.gmmOutPreRowStride);
                }
                AscendC::SyncAll<true>();
                AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNCFLAGV2C + chunkIdx * PERMUTE_CHUNK_FLAG_STRIDE);
            }
        } else {
            AscendC::CrossCoreWaitFlag<0x2>(SYNCFLAGC2V + chunkIdx * PERMUTE_CHUNK_FLAG_STRIDE);
            AscendC::SyncAll<true>();
            if (dequantSum1 > 0) {
                uint32_t rowStartThisCore = 0;
                MatrixCoord offsetC{0U, 0};
                MatrixCoord shapeC{dequantSum1, params.problemShape.n()};
                LayoutC layoutC;
                layoutC = LayoutC{dequantSum1, params.gmmOutPreRowStride};
                int64_t gmOffsetC = layoutC.GetOffset(offsetC);
                int64_t gmOffsetD = params.layoutD1.GetOffset(offsetC);
                if constexpr (std::is_same_v<ElementB, int8_t>) {
                    blockEpilogue1(gmC[gmOffsetC], shapeC, gmPerTokenScale1[rowStartThisCore],
                                   gmPermutedToken[gmOffsetD], gmPerTokenScale2[rowStartThisCore], resource,
                                   params.epilogueCoreNum, params.swigluLimit, params.activationCode,
                                   params.activationParams1, params.activationParams2, params.gmmOutPreRowStride);
                } else {
                    blockEpilogue1(gmC[gmOffsetC], shapeC, gmPermutedToken[gmOffsetD], resource, params.epilogueCoreNum,
                                   params.swigluLimit, params.activationCode, params.activationParams1,
                                   params.activationParams2, params.gmmOutPreRowStride);
                }
            }
            AscendC::SyncAll<true>();
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNCFLAGV2C + chunkIdx * PERMUTE_CHUNK_FLAG_STRIDE);

            if ((params.epilogueGranularity < params.expertPerRank && params.epilogueGranularity > 0)) {
                AscendC::CrossCoreWaitFlag<0x2>(SYNCFLAGC2V + chunkIdx * PERMUTE_CHUNK_FLAG_STRIDE);
                AscendC::SyncAll<true>();
                if (dequantSum2 > 0) {
                    uint32_t rowStartThisCore = dequantSum1;
                    MatrixCoord offsetC{rowStartThisCore, 0};
                    uint32_t dequantLen = dequantSum2;
                    MatrixCoord shapeC{dequantLen, params.problemShape.n()};
                    LayoutC layoutC;
                    layoutC = LayoutC{dequantLen, params.gmmOutPreRowStride};
                    int64_t gmOffsetC = layoutC.GetOffset(offsetC);
                    int64_t gmOffsetD = params.layoutD1.GetOffset(offsetC);
                    if constexpr (std::is_same_v<ElementB, int8_t>) {
                        blockEpilogue1(gmC[gmOffsetC], shapeC, gmPerTokenScale1[rowStartThisCore],
                                       gmPermutedToken[gmOffsetD], gmPerTokenScale2[rowStartThisCore], resource,
                                       coreNum, params.swigluLimit, params.activationCode, params.activationParams1,
                                       params.activationParams2, params.gmmOutPreRowStride);
                    } else {
                        blockEpilogue1(gmC[gmOffsetC], shapeC, gmPermutedToken[gmOffsetD], resource, coreNum,
                                       params.swigluLimit, params.activationCode, params.activationParams1,
                                       params.activationParams2, params.gmmOutPreRowStride);
                    }
                }
                AscendC::SyncAll<true>();
                // mode2 语义：40 逻辑核全部 SetFlag（每 block 2 次 Set → AIC 1 次 Wait）
                AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNCFLAGV2C + chunkIdx * PERMUTE_CHUNK_FLAG_STRIDE);
            }
        }
        blockEpilogue1.Finalize();
        const bool sameChunkLate = SharedGmm2Chunked(params);
        if (finishSharedGmm2 && !sameChunkLate) {
            // All AIVs consume the completion signal exactly once, before
            // split-subblock Unpermute. Quantized results retain the BF16
            // rounding point before FP32 routed/shared reduction.
            const auto range = SharedGmm2Range(params, chunkIdx);
            SharedGMM2Epilogue(params, range.base, range.rows);
        }
        exceptionDump_.UpdateStage(MC2MegaMoeAdump::Stage::COMBINE);
        if constexpr (MASK_ROUTING) {
            CombineSetFlag();
            CombineV2(params, blockEpilogue3);
        } else if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
            if constexpr (DedupOn_) {
                // Dedup Combine v2：token 粒度锚点累加（FP32 (hi,lo) 双半行拆分）+ last 行
                // 远端直传 D 区 + 组 flag 数据驱动消费。blockEpilogue3 构造仅记 UB 偏移、
                // Dedup 实例不调用（零物理冲突）。
                DedupCombineV2(params);
            } else {
                blockEpilogue3.InitFlag();
                CombineV2(params, blockEpilogue3);
            }
        } else if (isCombineV1) {
            blockEpilogue2.SetFlag();
            CombineV1(params, blockEpilogue2);
        } else {
            CombineSetFlag();
            CombineV2(params, blockEpilogue3);
        }
    }

    CATLASS_DEVICE
    void WaitSharedGmm1Ready()
    {
        {
            AscendC::CrossCoreWaitFlag<0x2>(SYNCFLAG_SHARED_GMM1_C2V);
        }
        {
            AscendC::SyncAll<true>();
        }
    }

    CATLASS_DEVICE
    void SharedGMM1Epilogue(Params const& params)
    {
        WaitSharedGmm1Ready();

        const uint32_t m = params.problemShape.m();
        const uint32_t n = params.sharedN;
        const uint32_t k2 = n / 2U;
        constexpr uint32_t msdFactor = kSharedIsW4A8 ? 2U : 1U;
        SharedEpilogue1 sharedEpilogue(resource, n);
        for (uint32_t sharedIdx = 0U; sharedIdx < params.sharedExpertNum; ++sharedIdx) {
            if constexpr (std::is_same_v<SharedElementC, int32_t>) {
                sharedEpilogue.SetWeightScales(
                    GetTensorAddr<uint64_t>(sharedIdx, reinterpret_cast<GM_ADDR>(params.ptrSharedScale1)));
            }
            MatrixCoord shapeC{m, n};
            const int64_t gmOffsetC = static_cast<int64_t>(sharedIdx) * m * n * msdFactor;
            const int64_t gmOffsetD = static_cast<int64_t>(sharedIdx) * m * k2;
            if constexpr (SharedGmm2Bf16_) {
                sharedEpilogue(gmSharedGmmBuffer[gmOffsetC], shapeC, gmSharedActivation[gmOffsetD], resource,
                               params.epilogueCoreNum, params.swigluLimit, params.activationCode,
                               params.activationParams1, params.activationParams2, n, Callback{}, &gmSharedInputScale);
            } else if constexpr (kSharedIsW4A8) {
                AscendC::GlobalTensor<float> sharedBias;
                sharedBias.SetGlobalBuffer(
                    GetTensorAddr<float>(sharedIdx, reinterpret_cast<GM_ADDR>(params.ptrSharedBias1)));
                sharedEpilogue.SharedExpert(
                    gmSharedGmmBuffer[gmOffsetC], shapeC, gmSharedInputScale, sharedBias, gmSharedActivation[gmOffsetD],
                    gmSharedActivationScale[static_cast<uint64_t>(sharedIdx) * m], resource, params.epilogueCoreNum,
                    params.swigluLimit, params.activationCode, params.activationParams1, params.activationParams2, n);
            } else if constexpr (std::is_same_v<SharedElementB, int8_t>) {
                sharedEpilogue(gmSharedGmmBuffer[gmOffsetC], shapeC, gmSharedInputScale, gmSharedActivation[gmOffsetD],
                               gmSharedActivationScale[static_cast<uint64_t>(sharedIdx) * m], resource,
                               params.epilogueCoreNum, params.swigluLimit, params.activationCode,
                               params.activationParams1, params.activationParams2, n);
            } else {
                sharedEpilogue(gmSharedGmmBuffer[gmOffsetC], shapeC, gmSharedActivation[gmOffsetD], resource,
                               params.epilogueCoreNum, params.swigluLimit, params.activationCode,
                               params.activationParams1, params.activationParams2, n);
            }
        }
        sharedEpilogue.Finalize();
        // Each shared expert owns a disjoint GM range. Drain the local
        // epilogue pipeline first, then synchronize once before AIC consumes
        // any of the published activation rows.
        {
            AscendC::SyncAll<true>();
        }
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNCFLAG_SHARED_GMM1_V2C);
    }

    CATLASS_DEVICE
    void SharedGMM2Epilogue(Params const& params, uint32_t tokenBase, uint32_t rows)
    {
        {
            AscendC::CrossCoreWaitFlag<0x2>(SYNCFLAG_SHARED_GMM2_C2V);
        }
        {
            AscendC::SyncAll<true>();
        }
        if constexpr (kSharedIsQuant && !SharedGmm2Bf16_) {
            const uint32_t m = params.problemShape.m();
            const uint32_t n = params.problemShape.k();
            constexpr uint32_t msdFactor = kSharedIsW4A8 ? 2U : 1U;
            if constexpr (kSharedIsW4A8) {
                const int32_t runtimeRank = RuntimeRank(params);
                typename BlockEpilogue3::Params epilogueParams{
                    static_cast<int32_t>(params.EP),
                    static_cast<int32_t>(params.expertPerRank),
                    runtimeRank,
                    reinterpret_cast<__gm__ int32_t*>(shmem() + peermemInfo.offsetPeerTokenPerExpert),
                    params.layoutD2,
                    static_cast<int32_t>(n),
                    static_cast<int32_t>(L1TileShape::N),
                    shmem,
                    peermemInfo.offsetD,
                    tokenPerExpertLayout};
                BlockEpilogue3 sharedEpilogue(resource, epilogueParams);
                for (uint32_t sharedIdx = 0U; sharedIdx < params.sharedExpertNum; ++sharedIdx) {
                    AscendC::GlobalTensor<float> sharedBias;
                    sharedBias.SetGlobalBuffer(
                        GetTensorAddr<float>(sharedIdx, reinterpret_cast<GM_ADDR>(params.ptrSharedBias2)));
                    const uint64_t cOffset = static_cast<uint64_t>(sharedIdx) * m * n * msdFactor;
                    const uint64_t resultOffset = static_cast<uint64_t>(sharedIdx) * m * n;
                    const uint64_t scaleOffset = static_cast<uint64_t>(sharedIdx) * m;
                    sharedEpilogue.SharedExpert(gmSharedGmmBuffer[cOffset], gmSharedActivationScale[scaleOffset],
                                                sharedBias, gmSharedResult[resultOffset], m, n, params.epilogueCoreNum);
                }
                AscendC::SyncAll<true>();
            } else {
                typename SharedEpilogue2::Params epilogueParams{
                    static_cast<int32_t>(params.EP), static_cast<int32_t>(params.expertPerRank),
                    reinterpret_cast<__gm__ int32_t*>(shmem() + peermemInfo.offsetPeerTokenPerExpert),
                    static_cast<int32_t>(n)};
                SharedEpilogue2 sharedEpilogue(resource, epilogueParams);
                for (uint32_t sharedIdx = 0U; sharedIdx < params.sharedExpertNum; ++sharedIdx) {
                    if constexpr (std::is_same_v<SharedElementC, int32_t>) {
                        sharedEpilogue.SetWeightScales(
                            GetTensorAddr<uint64_t>(sharedIdx, reinterpret_cast<GM_ADDR>(params.ptrSharedScale2)));
                    }
                    const uint64_t cOffset = MegaMoeSharedGmm2::ExpertOffset(sharedIdx, m, tokenBase, n);
                    const uint64_t resultOffset = MegaMoeSharedGmm2::ExpertOffset(sharedIdx, m, tokenBase, n);
                    const uint64_t scaleOffset = MegaMoeSharedGmm2::ExpertOffset(sharedIdx, m, tokenBase, 1);
                    sharedEpilogue.SharedExpert(gmSharedGmmBuffer[cOffset], MatrixCoord{rows, n},
                                                gmSharedActivationScale[scaleOffset], gmSharedResult[resultOffset],
                                                params.epilogueCoreNum);
                }
                AscendC::SyncAll<true>();
            }
        }
    }

    CATLASS_DEVICE void MaskPrologue(Params const& p, int64_t tokenBase, uint32_t tokens, int64_t countOffset);

    template <uint32_t InputPadding>
    CATLASS_DEVICE void PermuteChunk(Params const& params, int64_t chunkTokens, int64_t tokenBase,
                                     GM_ADDR localTokenPerExpert)
    {
        if (GetPermuteChunk(params) == MegaMoeConfig::BATCHED_PERMUTE_CHUNK_TOKENS) {
            RunChunkPrologueCapacity<MegaMoeConfig::BATCHED_PERMUTE_CHUNK_TOKENS, InputPadding>(
                params, chunkTokens, tokenBase, localTokenPerExpert);
        } else {
            RunChunkPrologueCapacity<MegaMoeConfig::DEFAULT_PERMUTE_CHUNK_TOKENS, InputPadding>(
                params, chunkTokens, tokenBase, localTokenPerExpert);
        }
    }

    template <uint32_t MaxTokens, uint32_t InputPadding>
    CATLASS_DEVICE void RunChunkPrologueCapacity(Params const& params, int64_t chunkTokens, int64_t tokenBase,
                                                 GM_ADDR localTokenPerExpert)
    {
        constexpr bool quant = std::is_same_v<ElementB, AscendC::int4b_t> || std::is_same_v<ElementB, int8_t>;
        using PrologueSrc = bfloat16_t;
        using PrologueDst = std::conditional_t<quant, int8_t, bfloat16_t>;
        using ProloguePolicy = std::conditional_t<quant, MoePermute::MoePermuteProloguePolicy,
                                                  MoePermute::MoePermutePrologueNonQuantPolicy>;
        using Prologue = MoePermute::MoePermutePrologue<PROLOGUE_UB_STAGES, ProloguePolicy, PrologueSrc, PrologueDst,
                                                        false, MaxTokens, InputPadding>;
        const int64_t hidden = params.problemShape.k();
        const int64_t topK = params.topK;
        typename Prologue::Params prologueParams(chunkTokens, hidden, topK, params.expertPerRank * params.EP);
        Prologue prologue(resource, prologueParams);
        AscendC::GlobalTensor<PrologueSrc> gmPermX;
        gmPermX.SetGlobalBuffer(reinterpret_cast<__gm__ PrologueSrc*>(reinterpret_cast<GM_ADDR>(params.ptrA) +
                                                                      tokenBase * hidden * sizeof(PrologueSrc)));
        AscendC::GlobalTensor<int32_t> gmPermTopkIds;
        gmPermTopkIds.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(reinterpret_cast<GM_ADDR>(params.expertIdx) +
                                                                        tokenBase * topK * sizeof(int32_t)));
        AscendC::GlobalTensor<PrologueDst> gmPermExpandX;
        gmPermExpandX.SetGlobalBuffer(reinterpret_cast<__gm__ PrologueDst*>(shmem() + peermemInfo.offsetA));
        AscendC::GlobalTensor<int32_t> gmTokenPerExpert;
        gmTokenPerExpert.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(localTokenPerExpert));
        AscendC::GlobalTensor<int32_t> gmPermExpandRowIdx;
        gmPermExpandRowIdx.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspaceInfo.expandedRowIdx));
        prologue(gmPermX, gmPermTopkIds, gmPermExpandX, gmPermExpandRowIdx, gmTokenPerExpert,
                 workspaceInfo.ptrPermuteCounts,
                 params.ptrXActiveMask != nullptr ? params.ptrXActiveMask + tokenBase : nullptr);
    }

    CATLASS_DEVICE
    void DispatchAndCombine(Params const& params)
    {
        using PrologueSrc = bfloat16_t;
        icache_preload(ICACHE_PRELOAD_LINES);
        exceptionDump_.Dump(shmem() + peermemInfo.offsetPeerTokenPerExpert,
                            static_cast<size_t>(paddedExpertNumAligned) * params.expertPerRank *
                                static_cast<uint32_t>(shmem.RankSize()) * sizeof(int32_t));
        shmem.DumpSyncRegions(exceptionDump_);
        int32_t runtimeRank = RuntimeRank(params);
        int64_t localTokenPerExpertOffset =
            peermemInfo.offsetPeerTokenPerExpert + tokenPerExpertLayout(runtimeRank, 0, 0) * sizeof(int32_t);
        GM_ADDR localTokenPerExpert = shmem() + localTokenPerExpertOffset;

        constexpr bool kRoutingIsQuant = std::is_same_v<ElementB, AscendC::int4b_t> || std::is_same_v<ElementB, int8_t>;
        const int64_t hidden = params.problemShape.k();
        const int64_t topK = params.topK;
        const int64_t bs = params.problemShape.m();
        const int64_t numChunks = (bs + GetPermuteChunk(params) - 1) / GetPermuteChunk(params);
        const uint32_t n2 = params.problemShape.k(); // unpermute 的隐藏维（GMM2 输出列数）

        // ExpertTokenNums 输出清零（进入 chunk 循环前一次）
        {
            AscendC::GlobalTensor<int32_t> gmExpertTokenNums;
            gmExpertTokenNums.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.ptrExpertTokenNums));
            if (coreIdx == 0) {
                AscendC::LocalTensor<int32_t> zeroBuf = resource.ubBuf.template GetBufferByByte<int32_t>(0);
                AscendC::Duplicate(zeroBuf, 0, params.expertPerRank);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::DataCopyPad(gmExpertTokenNums, zeroBuf,
                                     {1, static_cast<uint16_t>(params.expertPerRank * sizeof(int32_t)), 0, 0});
                AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
                if constexpr (MASK_ROUTING) {
                    SyncEvent<AscendC::HardEvent::MTE3_V>();
                }
            }
        }
        if (params.sharedExpertNum > 0U) {
            PrepareSharedInput(params);
        }
        // Dense input preparation shares low UB with routing temporaries.
        CopyRankMaskToLocal(params);

        for (int64_t chunkIdx = 0; chunkIdx < numChunks; ++chunkIdx) {
            const int64_t tokenBase = chunkIdx * GetPermuteChunk(params);
            const int64_t chunkTokens =
                (bs - tokenBase) < GetPermuteChunk(params) ? (bs - tokenBase) : GetPermuteChunk(params);
            // cumsumMM 每 chunk 指向独立缓冲区（不复用，规避跨 chunk cache 陈旧读）
            cumsumMM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ChunkCumsumPtr(params, chunkIdx)));
            // chunkCumsumMM 保持 chunk 级指向，供轮数推导与轮表生成读取
            chunkCumsumMM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ChunkCumsumPtr(params, chunkIdx)));
            // 上一 chunk 轮次可能已将 preSumBeforeRank 切至轮表，恢复 chunk 级指向
            // （本 chunk 的 AllGatherTokenPerExpert / CacheRouteTablesToUb 依赖 chunk 级缓冲）
            preSumBeforeRank.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspaceInfo.ptrSumBeforeRank));

            // 发送侧 Permute Prologue（chunk 范围，x_active_mask 融合于 prologue，
            // 不改写输入 topk_ids）
            exceptionDump_.UpdateStage(MC2MegaMoeAdump::Stage::MOE_INIT_ROUTING);
            {
                if constexpr (DedupOn_) {
                    // Dedup Prologue（与基线 prologue 同调用形态）：原始序量化写 A 区 + 计数
                    // 散射元数据排序流 + 打包段 + TPE
                    typename MegaMoeDedup::DedupPrologue<PrologueSrc, ElementTopkWeights>::Params dedupPrologueParams(
                        chunkTokens, hidden, topK, params.expertPerRank * params.EP, params.expertPerRank);
                    MegaMoeDedup::DedupPrologue<PrologueSrc, ElementTopkWeights> dedupPrologue(resource,
                                                                                               dedupPrologueParams);
                    AscendC::GlobalTensor<PrologueSrc> gmDedupX;
                    gmDedupX.SetGlobalBuffer(reinterpret_cast<__gm__ PrologueSrc*>(
                        reinterpret_cast<GM_ADDR>(params.ptrA) + tokenBase * hidden * sizeof(PrologueSrc)));
                    AscendC::GlobalTensor<int32_t> gmDedupTopkIds;
                    gmDedupTopkIds.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(
                        reinterpret_cast<GM_ADDR>(params.expertIdx) + tokenBase * topK * sizeof(int32_t)));
                    AscendC::GlobalTensor<ElementTopkWeights> gmDedupProbs;
                    gmDedupProbs.SetGlobalBuffer(reinterpret_cast<__gm__ ElementTopkWeights*>(params.probs) +
                                                 tokenBase * topK);
                    AscendC::GlobalTensor<int8_t> gmDedupA;
                    gmDedupA.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(shmem() + peermemInfo.offsetA));
                    // 打包段本 rank 槽：[slots N·i32 | probs N·f32]，段步长 2N·4B（rankOrdinal
                    // 内联 slots 高 8bit，不占独立平面）；N = layoutChunkTokens·topK
                    // （尾 chunk 不改步长，布局口径见 dedup_layout）
                    const int64_t layoutChunkTokens = (bs < GetPermuteChunk(params)) ? bs : GetPermuteChunk(params);
                    GM_ADDR packBase =
                        shmem() + peermemInfo.offsetExpandedDstToSrcRow +
                        runtimeRank * DEDUP_PACK_SECTION_COUNT * layoutChunkTokens * topK * sizeof(int32_t);
                    AscendC::GlobalTensor<int32_t> gmPackSlots;
                    gmPackSlots.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(packBase));
                    AscendC::GlobalTensor<float> gmPackProbs;
                    gmPackProbs.SetGlobalBuffer(
                        reinterpret_cast<__gm__ float*>(packBase + layoutChunkTokens * topK * sizeof(int32_t)));
                    AscendC::GlobalTensor<int32_t> gmDedupTokenPerExpert;
                    gmDedupTokenPerExpert.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(localTokenPerExpert));
                    dedupPrologue(gmDedupX, gmDedupTopkIds, gmDedupProbs, gmDedupA, gmPackSlots, gmPackProbs,
                                  gmDedupTokenPerExpert, workspaceInfo.ptrPermuteCounts, workspaceInfo.expandedRowIdx,
                                  params.ptrXActiveMask != nullptr ? params.ptrXActiveMask + tokenBase : nullptr);
                } else if constexpr (MASK_ROUTING) {
                    this->slotStride_ = AlignUp(static_cast<uint32_t>(chunkTokens) * static_cast<uint32_t>(params.topK),
                                                MaskRouting::SLOT_ALIGN);
                    MaskPrologue(params, tokenBase, chunkTokens, localTokenPerExpertOffset);
                } else {
                    if constexpr (kRoutingIsQuant) {
                        if (routingPadding_ == BYTE_PER_BLK)
                            PermuteChunk<BYTE_PER_BLK>(params, chunkTokens, tokenBase, localTokenPerExpert);
                        else
                            PermuteChunk<ALIGN_512>(params, chunkTokens, tokenBase, localTokenPerExpert);
                    } else {
                        PermuteChunk<ALIGN_512>(params, chunkTokens, tokenBase, localTokenPerExpert);
                    }
                }
            }
            if constexpr (MASK_ROUTING) {
                AscendC::PipeBarrier<PIPE_ALL>();
            } else {
                AscendC::SyncAll<true>();
            }

            // counts all-gather + cumsum（chunk 局部，覆盖重算）
            exceptionDump_.UpdateStage(MC2MegaMoeAdump::Stage::ALLGATHER_TOKEN_PER_EXPERT);
            if constexpr (!MASK_ROUTING) {
                AllGatherTokenPerExpert(params, localTokenPerExpertOffset, coreNum);
            }
            if (coreIdx == 0) {
                exceptionDump_.UpdateStage(MC2MegaMoeAdump::Stage::CUMSUM_TOKEN_PER_EXPERT);
                if constexpr (MASK_ROUTING) {
                    GetMaskCumsumForMMAIV(tokenPerExpert, cumsumMM, params.expertPerRank, runtimeRank, params.EP,
                                          params.ptrExpertTokenNums);
                } else {
                    GetCumsumForMMAIV(tokenPerExpert, cumsumMM, params.expertPerRank, runtimeRank, params.EP);
                }
            }
            if constexpr (!MASK_ROUTING) {
                AscendC::SyncAll<true>();
            }
            if constexpr (DedupOn_) {
                // FP/LP 每 chunk 清零（cot 键空间 EP·layoutChunkTokens，尾 chunk 不改步长，
                // 布局口径见 dedup_layout）。函数自带写侧回执 + PipeBarrier；全核 barrier 由
                // 下方既有 SyncAll 合并收口，零新增 barrier。
                const int64_t dedupBs = params.problemShape.m();
                const int32_t dedupLayoutChunkTokens =
                    static_cast<int32_t>(dedupBs < GetPermuteChunk(params) ? dedupBs : GetPermuteChunk(params));
                ResetDistTokenFirstPosV2(workspaceInfo.ptrDistTokenFirstPos, workspaceInfo.ptrDistTokenLastPos,
                                         static_cast<int32_t>(params.EP) * dedupLayoutChunkTokens, coreIdx, coreNum);
            }
            // 路由表刷新到本核 UB：此后 dispatch/combine/epilogue 的标量读取不再直接读 GM
            CacheRouteTablesToUb(params);
            if constexpr (MASK_ROUTING) {
                MaskRouting routing(resource, chunkTokens, topK, params.EP * params.expertPerRank, coreIdx, coreNum);
                AscendC::GlobalTensor<uint8_t> masks;
                masks.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(shmem() + peermemInfo.offsetMask));
                AscendC::GlobalTensor<int32_t> maskIndicesGm;
                maskIndicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspaceInfo.ptrMaskIndices));
                routing.Expand(masks, maskIndicesGm, ubTokenPerExpertCache, params.EP, params.expertPerRank,
                               expertPerRankAligned);
                WaitMaskDataReady(params);
            }

            if constexpr (!MASK_ROUTING) {
                AscendC::GlobalTensor<int32_t> gmExpertTokenNums;
                gmExpertTokenNums.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.ptrExpertTokenNums));
                if (coreIdx == 0) {
                    AscendC::LocalTensor<int32_t> ubSum = resource.ubBuf.template GetBufferByByte<int32_t>(0);
                    AscendC::LocalTensor<int32_t> ubOut = resource.ubBuf.template GetBufferByByte<int32_t>(
                        AlignUp(params.expertPerRank * sizeof(int32_t), 32));
                    const AscendC::DataCopyParams countCopy{
                        1, static_cast<uint16_t>(params.expertPerRank * sizeof(int32_t)), 0, 0};
                    AscendC::DataCopyPad(ubSum, cumsumMM[(params.EP - 1) * params.expertPerRank], countCopy, {});
                    AscendC::DataCopyPad(ubOut, gmExpertTokenNums, countCopy, {});
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                    AscendC::Add(ubOut, ubOut, ubSum, params.expertPerRank);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                    AscendC::DataCopyPad(gmExpertTokenNums, ubOut, countCopy);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
                }
                AscendC::SyncAll<true>();
            }
            // 接收端轮次切分：R = Σ_g chunk cumsum 行数，轮数 = ceil(R/B)。
            // 此处 cumsumMM 仍指向 chunk 级缓冲，轮数推导必须在切换轮表之前完成；
            // AIC/AIV 两侧独立推导，结果一致，无需额外跨核通信。
            const int64_t numRecvRounds = ComputeNumRecvRounds(params);
            if constexpr (MASK_ROUTING) {
                this->maskMultipleRounds_ = numRecvRounds > 1;
            }
            // 接收轮次循环：rounds==1 时退化为原 chunk 局部流水（零回归）；rounds>1 时
            // 每轮先由 core0 生成轮表并切换全核视图，再执行轮内 dispatch/swiglu/combine。
            // 轮间串行由 ProcessRecvRound 末尾 SyncAll 保证，flag 槽位按轮复用。
            for (int64_t roundIdx = 0;; ++roundIdx) {
                if (numRecvRounds > 1) {
                    // 轮表生成（core0 读 chunkCumsumMM）→ SyncAll 全核可见 →
                    // 切换 cumsumMM / preSumBeforeRank 至轮表并刷新本核 UB 路由缓存
                    BuildRoundTables(params, chunkIdx, roundIdx);
                    AscendC::SyncAll<true>();
                    SetRoundTableBuffers(params, chunkIdx, roundIdx);
                    CacheRoundTablesToUb(params);
                }
                const auto sharedSchedule =
                    MegaMoeSharedGmm2::ForRound(params.sharedExpertNum, SharedGmm2Chunked(params), chunkIdx, roundIdx);
                if constexpr (MASK_ROUTING) {
                    this->maskRoundIdx_ = roundIdx;
                }
                ProcessRecvRound(params, chunkIdx, sharedSchedule.activation, sharedSchedule.gmm2,
                                 roundIdx + 1 >= numRecvRounds);
                if (roundIdx + 1 >= numRecvRounds) {
                    break;
                }
            }

            if constexpr (!MASK_ROUTING) {
                exceptionDump_.UpdateStage(MC2MegaMoeAdump::Stage::RESET_TOKEN_PER_EXPERT);
                ResetTokenPerExpert(params.EP * paddedExpertNumAligned);
            }

            exceptionDump_.UpdateStage(MC2MegaMoeAdump::Stage::CROSS_RANK_SYNC);
            if constexpr (!std::is_same_v<ElementB, int8_t>) {
                shmem.InitStatusTargetSum();
                if (get_subblockid() == 0) {
                    AscendC::LocalTensor<int32_t> ctrBuffer = resource.ubBuf.template GetBufferByByte<int32_t>(0);
                    if (maskBufferAddr == nullptr) {
                        shmem.CrossRankSyncV2Set(ctrBuffer);
                    } else {
                        shmem.CrossRankSyncV2SetWithMask(ctrBuffer, rankMaskLocal);
                    }
                    if constexpr (DedupOn_) {
                        // 全核反排：s0 亦参与 token 分片（就绪确认 = V2Set 尾部 wait RECV_SYNC，
                        // 48 核上报齐 ⇒ 全部 EP 状态字已被见证，协议零改动）。全核编号 =
                        // block_idx + subblockid(0)·blockNum，核数 = coreNum；索引源 = prologue
                        // 行索引表，反排复用基线 kernel
                        const int64_t kHat = (topK < params.EP) ? topK : params.EP;
                        MoeTokenUnpermuteTilingData tilingData;
                        MoeTokenUnpermuteTiling(chunkTokens * topK, n2, topK, tilingData, coreNum);
                        tilingData.num_out_tokens = chunkTokens * kHat; // 行号上界（有效性判据）
                        KernelMoeTokenUnpermute<ElementD2, int32_t, float, false> kernelMoeTokenUnpermuteOp;
                        kernelMoeTokenUnpermuteOp.Init(
                            shmem() + peermemInfo.offsetD, workspaceInfo.expandedRowIdx, nullptr,
                            reinterpret_cast<GM_ADDR>(params.ptrOutput + tokenBase * hidden), &tilingData, true);
                        kernelMoeTokenUnpermuteOp.Process();
                    }
                } else {
                    uint32_t uboffset = 0;
                    uint32_t aicCoreNum = coreNum / AIV_PER_AIC;
                    uint32_t aicCoreIdx = get_block_idx();
                    uint32_t sendRankNum_ = params.EP / aicCoreNum;
                    uint32_t remainderRankNum = params.EP % aicCoreNum;
                    if (aicCoreIdx < remainderRankNum) {
                        sendRankNum_++;
                    }
                    AscendC::LocalTensor<float> statusTensor = resource.ubBuf.template GetBufferByByte<float>(uboffset);
                    uboffset += sendRankNum_ * UB_ALIGN;
                    AscendC::LocalTensor<float> gatherMaskOutTensor =
                        resource.ubBuf.template GetBufferByByte<float>(uboffset);
                    uboffset += AlignUp(params.EP * sizeof(float), 32);
                    AscendC::LocalTensor<uint32_t> gatherTmpTensor =
                        resource.ubBuf.template GetBufferByByte<uint32_t>(uboffset);
                    uboffset += AlignUp(sizeof(uint32_t), 32);
                    AscendC::LocalTensor<float> statusSumOutTensor =
                        resource.ubBuf.template GetBufferByByte<float>(uboffset);
                    uboffset += AlignUp(sizeof(float), 32);
                    if (maskBufferAddr == nullptr) {
                        shmem.CrossRankSyncV2Wait(statusTensor, gatherMaskOutTensor, gatherTmpTensor,
                                                  statusSumOutTensor);
                    } else {
                        shmem.CrossRankSyncV2WaitWithMask(rankMaskLocal);
                    }

                    exceptionDump_.UpdateStage(MC2MegaMoeAdump::Stage::UNPERMUTE);
                    if constexpr (DedupOn_) {
                        // 与 s0 共用全核分片（编号 = block_idx + subblockid(1)·blockNum，核数 =
                        // coreNum）；索引源 = prologue 行索引表，反排复用基线 kernel
                        const int64_t kHat = (topK < params.EP) ? topK : params.EP;
                        MoeTokenUnpermuteTilingData tilingData;
                        MoeTokenUnpermuteTiling(chunkTokens * topK, n2, topK, tilingData, coreNum);
                        tilingData.num_out_tokens = chunkTokens * kHat; // 行号上界（有效性判据）
                        KernelMoeTokenUnpermute<ElementD2, int32_t, float, false> kernelMoeTokenUnpermuteOp;
                        kernelMoeTokenUnpermuteOp.Init(
                            shmem() + peermemInfo.offsetD, workspaceInfo.expandedRowIdx, nullptr,
                            reinterpret_cast<GM_ADDR>(params.ptrOutput + tokenBase * hidden), &tilingData, true);
                        kernelMoeTokenUnpermuteOp.Process();
                    } else {
                        MoeTokenUnpermuteTilingData tilingData;
                        MoeTokenUnpermuteTiling(chunkTokens * topK, n2, topK, tilingData, coreNum / AIV_PER_AIC);
                        KernelMoeTokenUnpermute<ElementD2, int32_t, ElementTopkWeights, true,
                                                BLOCKED_OUTPUT_ ? L1TileShape::N : 0, true>
                            kernelMoeTokenUnpermuteOp;
                        kernelMoeTokenUnpermuteOp.Init(
                            shmem() + peermemInfo.offsetD, workspaceInfo.expandedRowIdx,
                            reinterpret_cast<GM_ADDR>(reinterpret_cast<__gm__ ElementTopkWeights*>(params.probs) +
                                                      tokenBase * topK),
                            reinterpret_cast<GM_ADDR>(params.ptrOutput + tokenBase * hidden), &tilingData, false,
                            workspaceInfo.ptrSharedResult, params.sharedExpertNum, params.ptrXActiveMask, bs,
                            tokenBase);
                        kernelMoeTokenUnpermuteOp.Process();
                    }
                }
            } else {
                {
                    if (maskBufferAddr == nullptr) {
                        shmem.CrossRankSync();
                    } else {
                        shmem.CrossRankSyncWithMask(rankMaskLocal);
                    }
                }

                if (SharedGmm2WaitAfterCrossRank(params)) {
                    // CrossRankSync uses routed results only. Join before Unpermute reads the shared result.
                    const auto range = SharedGmm2Range(params, chunkIdx);
                    SharedGMM2Epilogue(params, range.base, range.rows);
                }
                exceptionDump_.UpdateStage(MC2MegaMoeAdump::Stage::UNPERMUTE);
                MoeTokenUnpermuteTilingData tilingData;
                MoeTokenUnpermuteTiling(chunkTokens * topK, n2, topK, tilingData, coreNum);
                KernelMoeTokenUnpermute<ElementD2, int32_t, ElementTopkWeights, true,
                                        BLOCKED_OUTPUT_ ? L1TileShape::N : 0, true>
                    kernelMoeTokenUnpermuteOp;
                kernelMoeTokenUnpermuteOp.Init(
                    shmem() + peermemInfo.offsetD, workspaceInfo.expandedRowIdx,
                    reinterpret_cast<GM_ADDR>(reinterpret_cast<__gm__ ElementTopkWeights*>(params.probs) +
                                              tokenBase * topK),
                    reinterpret_cast<GM_ADDR>(params.ptrOutput + tokenBase * hidden), &tilingData, true,
                    workspaceInfo.ptrSharedResult, params.sharedExpertNum, params.ptrXActiveMask, bs, tokenBase);
                kernelMoeTokenUnpermuteOp.Process();
            }

            // chunk 边界：unpermute 读完 D/expandRowIdx 后才允许覆盖
            {
                AscendC::SyncAll<true>();
            }
            // chunk 边界无需 Mix 全核屏障（与 AIC 配对移除）：workspace 复用由 C2V/V2C + syncgmm 握手串行化
            // （syncgmm 握手已按接收轮归零复用，见接收轮次循环）
        } // for chunkIdx
    }

    CATLASS_DEVICE
    // CombineV1: selected when ElementB == int8_t && m * topK > 4096.
    void CombineV1(Params const& params, BlockEpilogue2& blockEpilogue)
    {
        uint32_t n2 = params.problemShape.k();
        int32_t prevGroupSum2 = 0;
        int32_t runtimeRank = RuntimeRank(params);

        icache_preload(ICACHE_PRELOAD_LINES);
        for (uint32_t t_groupIdx = 0; t_groupIdx < params.expertPerRank; ++t_groupIdx) {
            int32_t flagId = t_groupIdx / CROSS_CORE_FLAG_MAX_SET_COUNT;
            AscendC::CrossCoreWaitFlag<0x2>(flagId);
            AscendC::SyncAll<true>();

            uint32_t groupIdx = t_groupIdx;

            for (int32_t dstEpIdx = coreIdx; dstEpIdx < params.EP; dstEpIdx += coreNum) {
                if (IsRankMasked(dstEpIdx)) {
                    continue;
                }
                __gm__ void* dstPeermemPtr = shmem(peermemInfo.offsetD, dstEpIdx);
                AscendC::GlobalTensor<ElementD2> gmRemotePeer;
                gmRemotePeer.SetGlobalBuffer(reinterpret_cast<__gm__ ElementD2*>(dstPeermemPtr));
                uint32_t srcRowOffset =
                    (dstEpIdx == 0 ? 0 : cumsumMM.GetValue((dstEpIdx - 1) * params.expertPerRank + groupIdx)) +
                    prevGroupSum2;
                if (srcRowOffset < params.maxOutputSize) {
                    uint32_t dataRows = ubTokenPerExpertCache.GetValue(dstEpIdx * expertPerRankAligned + groupIdx);
                    if (srcRowOffset + dataRows > params.maxOutputSize) {
                        dataRows = params.maxOutputSize - srcRowOffset;
                    }
                    uint32_t dstRowOffset =
                        ubPreSumBeforeRankCache.GetValue(dstEpIdx * params.expertPerRank + groupIdx);
                    MatrixCoord offsetC{srcRowOffset, 0};
                    MatrixCoord offsetPeer{dstRowOffset, 0};
                    MatrixCoord shapeC{dataRows, n2};
                    int64_t gmOffsetC = params.layoutD2.GetOffset(offsetC);
                    int64_t gmOffsetPeer = params.layoutD2.GetOffset(offsetPeer);
                    if constexpr (std::is_same_v<ElementA, int8_t>) {
                        blockEpilogue(gmC2[gmOffsetC], shapeC, gmPerTokenScale2[srcRowOffset],
                                      gmRemotePeer[gmOffsetPeer]);
                    }
                }
            }
            prevGroupSum2 += cumsumMM.GetValue((params.EP - 1) * params.expertPerRank + groupIdx);
        }
        blockEpilogue.Finalize();
    }

    template <class CombineEpilogue>
    CATLASS_DEVICE
        // CombineV2: selected when W4A8 (int4b_t), or ElementB == int8_t && m * topK <= 4096,
        //   or BF16/FP16 (all non-INT8 paths).
        void
        CombineV2(Params const& params, CombineEpilogue& blockEpilogue)
    {
        BlockScheduler blockScheduler;
        int32_t syncLoopIdx = 0;
        uint32_t startCoreIdx = 0;
        uint32_t aicCoreNum = coreNum / AIV_PER_AIC;
        uint32_t aicCoreIdx = get_block_idx();
        uint32_t aivSubCoreIdx = get_subblockid();
        uint32_t preSrcExpertSum = 0;
        uint32_t n2 = params.problemShape.k();
        uint32_t k2 = params.problemShape.n() / 2;

        int64_t gmGroupOffsetC = 0;
        uint32_t aivCoreNum = coreNum;
        uint32_t aivCoreIdx = coreIdx;

        int32_t m0 = 32;
        if constexpr (std::is_same_v<ElementB, int8_t>) {
            m0 = 16;
        }

        icache_preload(ICACHE_PRELOAD_LINES);
        for (uint32_t groupIdx = 0; groupIdx < params.expertPerRank; ++groupIdx) {
            uint32_t currentExpertM = cumsumMM.GetValue((params.EP - 1) * params.expertPerRank + groupIdx);
            if (preSrcExpertSum >= params.maxOutputSize) {
                currentExpertM = 0;
            } else if (preSrcExpertSum + currentExpertM > params.maxOutputSize) {
                currentExpertM = params.maxOutputSize - preSrcExpertSum;
            }

            if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
                currentExpertM = currentExpertM * GATE_UP_ROWS_PER_TOKEN;
            }
            GemmCoord inGroupProblemShape{currentExpertM, n2, k2}; // M N K
            blockScheduler.Update(inGroupProblemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
            uint32_t coreLoops = blockScheduler.GetCoreLoops();

            uint32_t startLoopIdx =
                ((aicCoreIdx < startCoreIdx) ? (aicCoreIdx + aicCoreNum) : aicCoreIdx) - startCoreIdx;

            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += aicCoreNum) {
                GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = blockScheduler.GetActualBlockShape(blockCoord);

                int32_t m_rows = (actualBlockShape.m() + m0 - 1) / m0;
                int32_t aiv_m_rows = m_rows / SUBBLOCK_NUM_PER_AIC;
                if (aivSubCoreIdx == 1 && aiv_m_rows * SUBBLOCK_NUM_PER_AIC < m_rows) {
                    aiv_m_rows += 1;
                }
                uint32_t m_offset = blockCoord.m() * L1TileShape::M;
                if (aivSubCoreIdx == 1) {
                    m_offset += (m_rows / SUBBLOCK_NUM_PER_AIC) * m0;
                }

                for (; syncLoopIdx <= groupIdx; syncLoopIdx++) {
                    int32_t flag_id = syncLoopIdx / CROSS_CORE_FLAG_MAX_SET_COUNT;
                    AscendC::CrossCoreWaitFlag<0x2>(flag_id);
                }

                if constexpr (MASK_ROUTING) {
                    for (int32_t cur_row = 0; cur_row < aiv_m_rows;) {
                        const int32_t rowUnits = cur_row + 1 < aiv_m_rows ? 2 : 1;
                        GemmCoord realTileCoord{m_offset, blockCoord.n() * L1TileShape::N, 1};
                        uint32_t actualm = m0 * rowUnits;
                        if (aivSubCoreIdx == 1 && cur_row + rowUnits == aiv_m_rows) {
                            actualm = actualBlockShape.m() - (m_rows / SUBBLOCK_NUM_PER_AIC) * m0 - cur_row * m0;
                        }
                        GemmCoord realTileShape{actualm, actualBlockShape.n(), 1};
                        blockEpilogue(gmC2, gmPerTokenScale2, realTileCoord, realTileShape, groupIdx, preSrcExpertSum,
                                      preSumBeforeRank);
                        m_offset += m0 * rowUnits;
                        cur_row += rowUnits;
                    }
                    continue;
                }
                for (int32_t cur_row = 0; cur_row < aiv_m_rows; cur_row++) {
                    GemmCoord realTileCoord{m_offset, blockCoord.n() * L1TileShape::N, 1};
                    uint32_t actualm = m0;
                    if (aivSubCoreIdx == 1 && cur_row == aiv_m_rows - 1) {
                        actualm = actualBlockShape.m() - (m_rows / SUBBLOCK_NUM_PER_AIC) * m0 - cur_row * m0;
                    }
                    GemmCoord realTileShape{actualm, actualBlockShape.n(), 1};
                    if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
                        blockEpilogue(gmC2, gmPerTokenScale2, reinterpret_cast<__gm__ float*>(params.ptrBias2),
                                      realTileCoord, realTileShape, groupIdx, preSrcExpertSum * 2, preSumBeforeRank,
                                      params.listLen);
                    } else if constexpr (std::is_same_v<ElementB, int8_t>) {
                        blockEpilogue(gmC2, gmPerTokenScale2, realTileCoord, realTileShape, groupIdx, preSrcExpertSum,
                                      preSumBeforeRank);
                    } else {
                        blockEpilogue(gmC2, realTileCoord, realTileShape, groupIdx, preSrcExpertSum, preSumBeforeRank);
                    }
                    m_offset += m0;
                }
            }

            if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
                preSrcExpertSum += currentExpertM / GATE_UP_ROWS_PER_TOKEN;
            } else {
                preSrcExpertSum += currentExpertM;
            }
            startCoreIdx = (startCoreIdx + coreLoops) % aicCoreNum;
        }

        if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
            for (; syncLoopIdx < params.expertPerRank; syncLoopIdx++) {
                int32_t flag_id = syncLoopIdx / CROSS_CORE_FLAG_MAX_SET_COUNT;
                AscendC::CrossCoreWaitFlag<0x2>(flag_id);
            }
        }
        blockEpilogue.Finalize();
        CombineWaitFlag();
    }

private:
    struct WorkspaceInfo {
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
        GM_ADDR ptrPermuteCounts;
        GM_ADDR ptrSumBeforeRank;
        GM_ADDR ptrRoundTables;
        __gm__ float* ptrSoftFlagBase;
        // Dedup workspace 增量段（仅 DedupOn_=true 实例赋值，追加于基线段之后；
        // 段序与 host tiling 同源）
        GM_ADDR ptrDistTokenFirstPos;
        GM_ADDR ptrDistTokenLastPos;
        GM_ADDR ptrDedupRowMeta;
        GM_ADDR ptrSharedGmmBuffer;
        GM_ADDR ptrSharedActivation;
        GM_ADDR ptrSharedInput;
        GM_ADDR ptrSharedInputScale;
        GM_ADDR ptrSharedActivationScale;
        GM_ADDR ptrSharedResult;

        CATLASS_DEVICE
        WorkspaceInfo() {}

        CATLASS_DEVICE
        WorkspaceInfo(const Params& params)
        {
            uint64_t workspaceOffset = 0;
            ptrPermuteCounts = params.ptrWorkspace;
            const int64_t alignedE = (static_cast<int64_t>(params.expertPerRank) * params.EP + 7) / 8 * 8;
            workspaceOffset +=
                static_cast<uint64_t>(PERMUTE_MAX_CORES) * static_cast<uint64_t>(alignedE) * sizeof(int32_t);

            // expandedRowIdx 按单 chunk 尺寸分配（逐 chunk 复用）
            expandedRowIdx = params.ptrWorkspace + workspaceOffset;
            workspaceOffset += AlignUp(GetPermuteChunk(params), 256) * params.topK * sizeof(int32_t);

            // cumsum 按 chunk 数分配多份（不复用），规避 AIC 跨 chunk 读取旧 cumsumMM 的 cache 陈旧问题。
            // 每份 EP*EP*expertPerRank int32 并按 CACHE_LINE 隔离；与 host 分配一致。
            ptrcumsumMM = params.ptrWorkspace + workspaceOffset;
            const int64_t bs = params.problemShape.m();
            const int64_t numChunks = (bs + GetPermuteChunk(params) - 1) / GetPermuteChunk(params);
            workspaceOffset += static_cast<uint64_t>(numChunks) * ChunkCumsumStride(params);

            // 接收侧轮表区：每 (chunk, round) 独立一份 [cumsumMM_r | tokenPerExpert_r | preSumBeforeRank_r]，
            // Logical values remain compact within each segment; segments and generations
            // are padded to 64 bytes, matching host allocation and RoundTablePtr.
            ptrRoundTables = params.ptrWorkspace + workspaceOffset;
            workspaceOffset +=
                static_cast<uint64_t>(numChunks) * params.recvRoundsMax * 3UL * RoundTableSegmentBytes(params);

            ptrPerTokenScale = params.ptrWorkspace + workspaceOffset;
            if constexpr (std::is_same_v<ElementB, AscendC::int4b_t> || std::is_same_v<ElementB, int8_t>) {
                workspaceOffset += params.maxOutputSize * sizeof(ElementPerTokenScale);
            }

            ptrPerTokenScale2 = params.ptrWorkspace + workspaceOffset;
            if constexpr (std::is_same_v<ElementB, AscendC::int4b_t> || std::is_same_v<ElementB, int8_t>) {
                workspaceOffset += params.maxOutputSize * sizeof(ElementPerTokenScale);
            }

            if constexpr (MASK_ROUTING) {
                workspaceOffset =
                    RoundUp<ALIGN_512>(reinterpret_cast<uint64_t>(params.ptrWorkspace + workspaceOffset)) -
                    reinterpret_cast<uint64_t>(params.ptrWorkspace);
            }

            // gmm out resues workspace
            ptrC = params.ptrWorkspace + workspaceOffset;
            ptrC2 = params.ptrWorkspace + workspaceOffset;
            workspaceOffset += params.maxOutputSize * params.gmmOutPreRowStride * sizeof(ElementC);

            if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
                // when W4A8 M
                workspaceOffset += params.maxOutputSize * params.gmmOutPreRowStride * sizeof(ElementC);
            }

            if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
                // swiglu out and gmm2 prechannel dequant res reuse workspace
                ptrA1Int4 = params.ptrWorkspace + workspaceOffset;
                ptrA2Int4 = params.ptrWorkspace + workspaceOffset;
                // sizeof(ElementA) is 1
                workspaceOffset += params.maxOutputSize * params.gmm1InputExpertSlotWidth;
            } else {
                ptrA = params.ptrWorkspace + workspaceOffset;
                ptrPermutedToken = params.ptrWorkspace + workspaceOffset;
                workspaceOffset += params.maxOutputSize * params.gmm1InputExpertSlotWidth * sizeof(ElementA);
            }

            ptrSumBeforeRank = params.ptrWorkspace + workspaceOffset;
            workspaceOffset += params.EP * AlignUp(params.expertPerRank, 16) * sizeof(int32_t);

            // W4A8 not use
            if constexpr (!std::is_same_v<ElementB, AscendC::int4b_t>) {
                ptrSoftFlagBase = reinterpret_cast<__gm__ float*>(params.ptrWorkspace + workspaceOffset);
                workspaceOffset += params.EP * sizeof(int32_t) * FLAGSTRIDE;
            }

            // Dedup 增量段：FP → LP → rowMeta；段尺寸由 host dedup_layout 经 tilingData
            // 透传，kernel 不重算
            if constexpr (DedupOn_) {
                ptrDistTokenFirstPos = params.ptrWorkspace + workspaceOffset;
                workspaceOffset += static_cast<uint64_t>(params.distTokenFirstPosSize);
                ptrDistTokenLastPos = params.ptrWorkspace + workspaceOffset;
                workspaceOffset += static_cast<uint64_t>(params.distTokenLastPosSize);
                ptrDedupRowMeta = params.ptrWorkspace + workspaceOffset;
                workspaceOffset += static_cast<uint64_t>(params.dedupRowMetaSize);
            }
            ptrSharedGmmBuffer = params.ptrWorkspace + params.sharedWorkspaceOffset;
            constexpr uint64_t sharedMsdFactor = kSharedIsW4A8 ? 2UL : 1UL;
            const uint32_t sharedRowStride =
                params.sharedN > params.problemShape.k() ? params.sharedN : params.problemShape.k();
            const uint64_t sharedGmmElementCount = static_cast<uint64_t>(params.sharedExpertNum) *
                                                   params.problemShape.m() * sharedRowStride * sharedMsdFactor;
            const auto alignSharedBuffer = [](uint64_t size) {
                constexpr uint64_t sharedBufferAlignment = 512UL;
                return (size + sharedBufferAlignment - 1UL) / sharedBufferAlignment * sharedBufferAlignment;
            };
            ptrSharedActivation =
                ptrSharedGmmBuffer + alignSharedBuffer(sharedGmmElementCount * sizeof(SharedElementC));
            const uint64_t sharedActivationElementCount =
                static_cast<uint64_t>(params.sharedExpertNum) * params.problemShape.m() * params.sharedN / 2U;
            GM_ADDR sharedWorkspaceTail =
                ptrSharedActivation + alignSharedBuffer(sharedActivationElementCount * sizeof(SharedElementD1));

            if constexpr (kSharedIsQuant) {
                ptrSharedInput = sharedWorkspaceTail;
                sharedWorkspaceTail += alignSharedBuffer(static_cast<uint64_t>(params.problemShape.m()) *
                                                         params.problemShape.k() * sizeof(SharedElementABefore));
                ptrSharedInputScale = sharedWorkspaceTail;
                sharedWorkspaceTail +=
                    alignSharedBuffer(static_cast<uint64_t>(params.problemShape.m()) * sizeof(float));
                ptrSharedActivationScale = sharedWorkspaceTail;
                sharedWorkspaceTail += alignSharedBuffer(static_cast<uint64_t>(params.sharedExpertNum) *
                                                         params.problemShape.m() * sizeof(float));
                ptrSharedResult = sharedWorkspaceTail;
            } else {
                ptrSharedInput = nullptr;
                ptrSharedInputScale = nullptr;
                ptrSharedActivationScale = nullptr;
                ptrSharedResult = ptrSharedGmmBuffer;
            }
        }
    };

    struct PeermemInfo {
        int64_t offsetA;
        int64_t offsetPeerPerTokenScale;
        int64_t offsetPeerTokenPerExpert;
        int64_t offsetD;
        // Dedup 新增区（仅 DedupOn_=true 实例赋值）
        int64_t offsetExpandedDstToSrcRow; // 打包区基址（rank 段 [slots|probs]）

        CATLASS_DEVICE
        PeermemInfo() {}

        CATLASS_DEVICE
        PeermemInfo(const Params& params, const HcclShmem<false>& shmem)
        {
            // 布局：10MB reserved → A → scale → D → cumsum → flag
            const int64_t segSize = static_cast<int64_t>(shmem.SegmentSize());
            const int64_t EP = params.EP;
            const int64_t E = params.expertPerRank;
            const int64_t M = params.maxOutputSize;
            const int64_t h = params.problemShape.k();
            const int64_t topK = params.topK;

            constexpr bool RoutingIsQuant =
                std::is_same_v<ElementB, AscendC::int4b_t> || std::is_same_v<ElementB, int8_t>;

            // 尾部：CrossRankSync + tokenPerExpert
            int64_t tailSyncSize = static_cast<int64_t>(shmem.TailReservedSize());
            int64_t tokenPerExpertSize =
                EP * AlignUp(EP * MAX_EXPERTS_PER_RANK + 1, ALIGN_128) * static_cast<int64_t>(sizeof(int32_t));

            // A: dispatch 数据区（单 chunk 尺寸，逐 chunk 复用）
            // 通信布局按 min(bs, configured chunk size) 分配，与 host 容量公式一致
            const int64_t bs = params.problemShape.m();
            const int64_t chunkTokens =
                bs < GetPermuteChunk(params) ? bs : static_cast<int64_t>(GetPermuteChunk(params));
            // quant 路径：行步长 = h + routingPadding（行尾含 per-token scale），ElementABefore=int8_t(1B)
            // 非 quant 路径：行步长 = h（密排），ElementABefore=bf16/fp16(2B)
            int64_t offsetASize = RoutingIsQuant ?
                                      (chunkTokens * topK * (h + routingPadding_) * sizeof(ElementABefore)) :
                                      (chunkTokens * topK * h * sizeof(ElementABefore));

            // scale: per-token 量化 scale 区（仅量化路径，本地侧独立 buffer，与远端 A 区行尾 scale 区分）
            int64_t perTokenScaleSize = RoutingIsQuant ? (chunkTokens * topK * sizeof(float)) : 0;

            // D: 输出区（单 chunk 尺寸，逐 chunk 复用）
            int64_t DSize = chunkTokens * topK * h * sizeof(int16_t);

            offsetA = RESERVED_SPACE_SIZE;
            if constexpr (DedupOn_) {
                // Dedup：A 区原始序 chunkTokens 行（行步长 h+512），后接打包区与 D 区；偏移由
                // host dedup_layout 经 tilingData 透传（kernel 不重算）；无 perTokenScale 保留区
                offsetPeerPerTokenScale = 0;
                offsetExpandedDstToSrcRow = params.offsetExpandedDstToSrcRow;
                offsetD = params.dedupOffsetD;
            } else if constexpr (RoutingIsQuant) {
                offsetPeerPerTokenScale = offsetA + offsetASize;
                offsetD = offsetPeerPerTokenScale + perTokenScaleSize;
            } else {
                offsetPeerPerTokenScale = 0;
                offsetD = offsetA + offsetASize;
            }
            // 两个offset倒排并固定大小防止清零的位置与下一个case需要的不相同
            // 这个flag在shmem中计算并使用
            const int64_t offsetFlag = shmem.SegmentSize() - shmem.TailReservedSize();
            offsetPeerTokenPerExpert = offsetFlag - tokenPerExpertSize;
        }
    };

    struct MaskWorkspaceInfo : WorkspaceInfo {
        GM_ADDR ptrMaskIndices{nullptr};
        GM_ADDR ptrRouteIndex{nullptr};

        CATLASS_DEVICE MaskWorkspaceInfo() {}
        CATLASS_DEVICE MaskWorkspaceInfo(const Params& params);
    };

    struct MaskPeermemInfo : PeermemInfo {
        int64_t offsetMask{0};
        int64_t offsetDataReady{0};

        CATLASS_DEVICE MaskPeermemInfo() {}
        CATLASS_DEVICE MaskPeermemInfo(const Params& params, const HcclShmem<false>& shmem);
    };

    using RouteWorkspaceInfo = std::conditional_t<MASK_ROUTING, MaskWorkspaceInfo, WorkspaceInfo>;
    using RoutePeermemInfo = std::conditional_t<MASK_ROUTING, MaskPeermemInfo, PeermemInfo>;

    Arch::Resource<ArchTag> resource;

    uint32_t coreIdx;
    uint32_t coreNum;
    uint32_t nSyncSwiglu;
    // AIC 侧跨核 syncgmm flag 计数（与 AIV 侧 syncgmm1Idx 对齐；接收轮次切分后每轮归零，
    // flag 槽位按轮复用，见 AIC operator() 轮循环）
    uint16_t syncgmmIdx = 0;

    Params params;
    RouteWorkspaceInfo workspaceInfo;
    RoutePeermemInfo peermemInfo;

    uint32_t dequantSum[kMaxDequantSyncGroups] = {0};

    // ========== Common tensors (all types) ==========
    AscendC::GlobalTensor<ElementC> gmC;
    AscendC::GlobalTensor<ElementC> gmC2;
    AscendC::GlobalTensor<ElementPerTokenScale> gmPerTokenScale1;
    AscendC::GlobalTensor<ElementPerTokenScale> gmPerTokenScale2;
    AscendC::GlobalTensor<int32_t> tokenPerExpert;
    AscendC::GlobalTensor<int32_t> cumsumMM;
    // chunk 级 cumsumMM（不随后续接收轮次切换而重指，轮表生成 / 轮数推导统一从这里读）
    AscendC::GlobalTensor<int32_t> chunkCumsumMM;
    // 接收侧轮表（仅 rounds>1 时使用，紧凑 EP*expertPerRank 布局，与 UB 缓存同构）
    AscendC::GlobalTensor<int32_t> roundTokenPerExpertGm;
    AscendC::GlobalTensor<int32_t> roundPreSumBeforeRankGm;
    AscendC::GlobalTensor<int32_t> preSumBeforeRank;
    Layout3D tokenPerExpertLayout;
    HcclShmem<false> shmem;
    int32_t paddedExpertNumAligned;
    static constexpr uint32_t routingPadding_ =
        BatchedDispatch && std::is_same_v<ElementB, int8_t> ? BYTE_PER_BLK : ALIGN_512;

    bool isCombineV1;

    template <AscendC::HardEvent EVENT, AscendC::TEventID ID = EVENT_ID0>
    CATLASS_DEVICE static void SyncEvent()
    {
        AscendC::SetFlag<EVENT>(ID);
        AscendC::WaitFlag<EVENT>(ID);
    }
    GM_ADDR maskBufferAddr{nullptr};
    AscendC::LocalTensor<int32_t> rankMaskLocal;
    // 路由表 UB 缓存（仅 AIV 使用）：tokenPerExpert 取 [dstEpIdx][runtimeRank][groupIdx] 紧凑平面，
    // preSumBeforeRank 为连续 EP*expertPerRank；每 chunk 由 CacheRouteTablesToUb 刷新
    AscendC::LocalTensor<int32_t> ubTokenPerExpertCache;
    AscendC::LocalTensor<int32_t> ubPreSumBeforeRankCache;
    uint32_t routeCacheNum{0};
    uint32_t expertPerRankAligned{0};
    // ExceptionDump引擎：记录执行阶段时间戳，并提供Dump接口由host侧dump指定GM地址内容。
    // 基址取通信域首地址（shmem()()），根据kRoutingIsQuant选择对应tiling结构体的Policy，
    // ArchTag传入Policy供架构差异扩展。
    static constexpr bool kRoutingIsQuant =
        std::is_same_v<ElementB, AscendC::int4b_t> || std::is_same_v<ElementB, int8_t>;
    MC2MegaMoeAdump::ExceptionDumpEngine<kRoutingIsQuant, ArchTag> exceptionDump_;

    AscendC::GlobalTensor<ElementScale> gmS;
    AscendC::GlobalTensor<ElementScale> gmS2;
    AscendC::GlobalTensor<int4b_t> gmA1I4;
    AscendC::GlobalTensor<int8_t> gmA1I4_I8;
    AscendC::GlobalTensor<int4b_t> gmA2I4;
    AscendC::GlobalTensor<int8_t> gmA2I4_I8;

    AscendC::GlobalTensor<ElementABefore> gmA;
    AscendC::GlobalTensor<ElementD1> gmPermutedToken;
    AscendC::GlobalTensor<SharedElementC> gmSharedGmmBuffer;
    AscendC::GlobalTensor<SharedElementD1> gmSharedActivation;
    AscendC::GlobalTensor<SharedElementA> gmSharedInput;
    AscendC::GlobalTensor<ElementPerTokenScale> gmSharedInputScale;
    AscendC::GlobalTensor<ElementPerTokenScale> gmSharedActivationScale;
    AscendC::GlobalTensor<ElementD2> gmSharedResult;
};
} // namespace Catlass::Gemm::Kernel

// Dedup 构件类外成员定义（须在 MegaMoeKernel 完整定义之后包含；本文件为唯一包含点）
#include "mega_moe_kernel_a3_dedup.hpp"

// Mask 构件类外定义；本文件为唯一包含点。
#include "mega_moe_kernel_a3_mask.hpp"

#endif // MEGA_MOE_KERNEL_HPP
