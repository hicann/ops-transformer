/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file all_gather_matmul_aiv_mode.h
 * \brief
 */

#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "adv_api/hccl/hccl.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../../common/op_kernel/moe_distribute_base_a2.h"
#include "all_gather_matmul_aiv_mode_tiling.h"
#include "all_gather_matmul_aiv_mode_util.h"
#include "all_gather_matmul_aiv_mode_padding.h"
#include "all_gather_matmul_aiv_mode_dequant.h"

#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/tla_catlass.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/arch/tla_arch_arch.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/layout/tla_layout_layout.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/block/tla_gemm_block_mmad.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/block/tla_gemm_block_swizzle.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/tla_gemm_dispatch_policy.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/tla_gemm_gemm_type.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/tla_gemm_coord.hpp"

#include "gather_matmul.hpp"

using namespace AscendC;

constexpr static uint32_t BUFFER_NUM = 2U;     // 多buf
constexpr static uint32_t STATE_OFFSET = 512U; // 状态空间偏移地址
constexpr static uint32_t BLOCK_SIZE = 32U;
constexpr static uint32_t B32_PER_BLOCK = 8U;
constexpr static uint32_t B64_PER_BLOCK = 4U;
constexpr static int32_t CUBE_MATRIX_SIZE_B16 = 256;        // 16 * 16
constexpr static int32_t L0AB_PINGPONG_BUFFER_SIZE = 32768; // 32 KB
constexpr static int32_t FLAG_ZERO_IDX = 0;
constexpr static int32_t FLAG_ONE_IDX = 1;
constexpr static int32_t FLAG_VALUE = 1;
constexpr static int32_t FLAG_TWO_IDX = 2;
constexpr static int32_t FLAG_THREE_IDX = 3;
constexpr static int32_t USED_UB_SIZE = 160 * 1024;
constexpr static int32_t TILE_SHAPE_128 = 128;
constexpr static int32_t TILE_SHAPE_256 = 256;
constexpr static int32_t TILE_SHAPE_512 = 512;
constexpr static int32_t TILE_SHAPE_64 = 64;
constexpr static int32_t FLAG_OFFSET = 180 * 1024 * 1024 / sizeof(int32_t);

using namespace Catlass;

namespace AllGatherMatmulAIVModeImpl {

template <typename T>
using supportTypeForDataCopy = typename std::conditional<std::is_same<T, AscendC::int4b_t>::value, int8_t, T>::type;

template <typename T, typename ReturnType = size_t>
struct TILE_SHAPE_K_512B {
    static constexpr ReturnType value = Catlass::BytesToBits(512) / Catlass::SizeOfBits<T>::value;
};

template <typename T, typename ReturnType = size_t>
struct TILE_SHAPE_K_256B {
    static constexpr ReturnType value = Catlass::BytesToBits(256) / Catlass::SizeOfBits<T>::value;
};

template <typename T, typename ReturnType = size_t>
struct TILE_SHAPE_K_128B {
    static constexpr ReturnType value = Catlass::BytesToBits(128) / Catlass::SizeOfBits<T>::value;
};

// AGMM : AllGatherMatmulAIVMode
#define TemplateAGMMClass \
    typename X1Type, typename X2Type, typename BiasType, typename x2ScaleType, typename YType, bool weightNZ, bool TA, \
        bool TB
#define TemplateAGMMFunc X1Type, X2Type, BiasType, x2ScaleType, YType, weightNZ, TA, TB

using namespace AscendC;
template <TemplateAGMMClass>
class AllGatherMatmulAIVMode {
    constexpr static bool quantFlag =
        (std::is_same<X1Type, int8_t>::value && std::is_same<X2Type, int8_t>::value) ||
        (std::is_same<X1Type, AscendC::int4b_t>::value && std::is_same<X2Type, AscendC::int4b_t>::value);
    using supportX1Type = supportTypeForDataCopy<X1Type>;
    using supportX2Type = supportTypeForDataCopy<X2Type>;
    constexpr static uint32_t UB_OFFSET =
        Catlass::BytesToBits(97440) / Catlass::SizeOfBits<supportX1Type>::value; // 2 是 size of T

public:
    __aicore__ inline AllGatherMatmulAIVMode(){};
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR x1ScaleGM, GM_ADDR x2ScaleGM,
                                GM_ADDR cGM, GM_ADDR allgatherGM, GM_ADDR workspaceGM, GM_ADDR tilingGM);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ResetFlags(int32_t num_flags);
    __aicore__ inline void CopyGatherResult(int32_t cal_idx);
    __aicore__ inline void AIVInit();
    __aicore__ inline void AICInit();
    __aicore__ inline void Padding();
    __aicore__ inline void Dequant(int32_t cal_idx);
    __aicore__ inline void AllGatherPerTokenScale(int64_t buff_st);
    __aicore__ inline void CatlassMatmul();
    template <typename LayoutB>
    __aicore__ inline void DispatchMatmul(const LayoutB &layoutB);
    template <typename LayoutB, int32_t TileM, int32_t TileN>
    __aicore__ inline void LaunchMatmul(const LayoutB &layoutB);
    __aicore__ inline void MoveWithSplit(__gm__ supportX1Type *gm_src, int64_t rank_offset, int64_t len);
    __aicore__ inline void MoveToOtherRankWithSkip(__gm__ supportX1Type *gm_src, int64_t rank_offset, int32_t len,
                                                   int32_t rank_st, int32_t skip_num, int32_t group_num,
                                                   int32_t rank_scope);
    __aicore__ inline void MoveResultFromPeerMemToOut(__gm__ supportX1Type *gm_src, __gm__ supportX1Type *gm_dst,
                                                      int32_t actual_m);
    __aicore__ inline void MoveResultToDst(__gm__ supportX1Type *gm_src, __gm__ supportX1Type *gm_dst, int32_t len);
    __aicore__ inline void MoveResultFromSrcToDst(__gm__ supportX1Type *gm_src, __gm__ supportX1Type *gm_dst,
                                                  int32_t len);
    __aicore__ inline void CrossRankSyncV1(int32_t flag_idx, int32_t flag_data);
    __aicore__ inline void CrossRankSyncV2(int32_t flag_idx, int32_t flag_data);

private:
    GlobalTensor<X1Type> aGMTensor_;
    GlobalTensor<X2Type> bGMTensor_;
    GlobalTensor<YType> cGMTensor_;
    GlobalTensor<YType> dataGMTensor_;
    GlobalTensor<int64_t> flagGMTensor_;

    GM_ADDR aGM_;
    GM_ADDR bGM_;
    GM_ADDR cGM_;
    GM_ADDR x1ScaleGM_;
    GM_ADDR x2ScaleGM_;
    GM_ADDR allgatherGM_;
    GM_ADDR windowInGM_;
    GM_ADDR windowOutGM_;
    GM_ADDR stateAddrPerRank[8];

    TBuf<AscendC::TPosition::VECCALC> uBuf_;

    int32_t m;
    int32_t k;
    int32_t n;
    int32_t m0;
    int32_t k0;
    int32_t n0;
    int32_t m_loop;
    int32_t k_loop;
    int32_t n_loop;
    int32_t pValue;
    int32_t aligned_a;
    int32_t aligned_b;
    int32_t cal_count;
    int64_t gm_a_pingpong_size;
    int32_t max_ub_ping_pong_size;
    int32_t m_align;
    int64_t k_align;
    int32_t n_align;
    int32_t comm_npu_split;
    int32_t comm_data_split;
    int32_t comm_direct;
    int32_t len_per_loop;
    int32_t core_count;
    int64_t data_len;
    int64_t num_per_rank_move;
    int64_t src_offset;
    int64_t rank_offset;
    int32_t swizzlCount;
    int32_t swizzlDirect;

    int32_t max_move_m;
    int32_t max_move_k = 20480;

    int32_t worldSize{0};
    int32_t rankId{0};
    int32_t coreIdx{0};
    int32_t aivIdx{0};
    int32_t coreNum{0};
    int32_t blockIdx{0};

    uint64_t aAlignSize{0};
    uint64_t bAlignSize{0};
    bool hasAAlign{false};
    bool hasBAlign{false};
    bool quanFlag{false};
    bool needAivDequant{false};
    bool isX2ScaleTypeInt64{false};
    DequantType dequantType;
    bool needPerChannel;
    bool needPerToken;
    bool accumWorkSpacePingPong{false};

    __gm__ supportX1Type *gm_peer_mem;

    GM_ADDR gm_a_align;
    GM_ADDR gm_b_align;
    __gm__ supportX1Type *gm_a_src;
    __gm__ supportX2Type *gm_b_src;
    __gm__ int32_t *gm_accum;
    GM_ADDR gm_scale_workspace;

    DequantRunner<YType> dequantRunner;
    Arch::Resource<Arch::AtlasA2> resource;
    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
};

template <TemplateAGMMClass>
__aicore__ inline void AllGatherMatmulAIVMode<TemplateAGMMFunc>::Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM,
                                                                      GM_ADDR x1ScaleGM, GM_ADDR x2ScaleGM, GM_ADDR cGM,
                                                                      GM_ADDR allgatherGM, GM_ADDR workspaceGM,
                                                                      GM_ADDR tilingGM)
{
    auto tiling = (__gm__ AllGatherMatmulAIVModeTilingData *)tilingGM;
    GET_TILING_DATA(tilingData, tilingGM);

    auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    aGM_ = aGM;
    bGM_ = bGM;
    cGM_ = cGM;
    allgatherGM_ = allgatherGM;
    x1ScaleGM_ = x1ScaleGM;
    x2ScaleGM_ = x2ScaleGM;

    m = tilingData.allGatherMatmulInfo.M;
    k = tilingData.allGatherMatmulInfo.K;
    n = tilingData.allGatherMatmulInfo.N;

    m0 = tilingData.cocTiling.m0; // tiling 寻优
    k0 = tilingData.cocTiling.k0; // tiling 寻优
    n0 = tilingData.cocTiling.n0; // tiling 寻优

    m_loop = tilingData.cocTiling.mLoop;
    n_loop = tilingData.cocTiling.nLoop;
    k_loop = tilingData.cocTiling.kLoop;
    pValue = tilingData.cocTiling.pValue;

    max_ub_ping_pong_size = tilingData.cocTiling.ubMoveNum / MAX_BLOCK_COUNT;
    comm_npu_split = tilingData.cocTiling.commNpuSplit;   // tiling 寻优
    comm_data_split = tilingData.cocTiling.commDataSplit; // tiling 寻优
    comm_direct = tilingData.cocTiling.commDirect;        // tiling 寻优
    len_per_loop = tilingData.cocTiling.lenPerLoop;       // tiling 寻优
    swizzlCount = tilingData.cocTiling.swizzlCount;       // 家里tiling写死
    swizzlDirect = tilingData.cocTiling.swizzlDirect;     // 家里tiling写死
    core_count = comm_npu_split * comm_data_split;

    coreIdx = GetBlockIdx();              // 0-48核
    coreNum = GetBlockNum();              // 24
    aivIdx = GetSubBlockIdx();            // 0-1
    blockIdx = coreIdx / GetTaskRation(); // 0-24核

    aAlignSize = tilingData.allGatherMatmulInfo.aAlignSize;
    bAlignSize = tilingData.allGatherMatmulInfo.bAlignSize;
    hasAAlign = tilingData.allGatherMatmulInfo.hasAAlign;
    hasBAlign = (tilingData.allGatherMatmulInfo.hasBAlign && !(weightNZ));
    gm_a_align = reinterpret_cast<GM_ADDR>(hasAAlign ? workspaceGM : 0);
    gm_b_align = reinterpret_cast<GM_ADDR>(hasBAlign ? workspaceGM + aAlignSize : 0);
    gm_a_src = reinterpret_cast<__gm__ supportX1Type *>(hasAAlign ? gm_a_align : aGM_);
    gm_b_src = reinterpret_cast<__gm__ supportX2Type *>(hasBAlign ? gm_b_align : bGM_);
    gm_accum = reinterpret_cast<__gm__ int32_t *>(quantFlag ? workspaceGM + aAlignSize + bAlignSize : 0);

    m_align = Block512B<X1Type>::AlignUp(m);
    k_align = Block512B<X1Type>::AlignUp(k);
    n_align = Block512B<X1Type>::AlignUp(n);
    aligned_a = hasAAlign;
    aligned_b = hasBAlign;

    isX2ScaleTypeInt64 = tilingData.allGatherMatmulInfo.isX2ScaleTypeInt64;
    dequantType = tilingData.allGatherMatmulInfo.dequantType;
    needAivDequant = quantFlag && (dequantType == PER_TOKEN || std::is_same<YType, bfloat16_t>::value);
    bool needPerChannelA8W8 = quantFlag && !(isX2ScaleTypeInt64 && std::is_same<YType, float16_t>::value);
    needPerChannel = std::is_same_v<X1Type, AscendC::int4b_t> || needPerChannelA8W8;
    needPerToken = quantFlag && dequantType == PER_TOKEN;

    bool is910C = tilingData.allGatherMatmulInfo.is910C;
    if (is910C) {
        __gm__ HcclOpResParam *winContext_{nullptr};
        winContext_ = (__gm__ HcclOpResParam *)contextGM0;
        rankId = winContext_->localUsrRankId;
        worldSize = winContext_->rankSize;
        for (int i = 0; i < worldSize; i++) {
            stateAddrPerRank[i] =
                (GM_ADDR)((i == rankId) ?
                              winContext_->localWindowsIn :
                              ((HcclRankRelationResV2 *)(winContext_->remoteRes[i].nextDevicePtr))->windowsIn);
        }
    } else {
        __gm__ HcclA2CombineOpParam *winContext_{nullptr};
        winContext_ = (__gm__ HcclA2CombineOpParam *)contextGM0;
        rankId = winContext_->rankId;
        worldSize = winContext_->rankNum;
        for (int i = 0; i < worldSize; i++) {
            stateAddrPerRank[i] = (GM_ADDR)winContext_->windowsIn[i];
        }
    }

    accumWorkSpacePingPong = tilingData.allGatherMatmulInfo.accumWorkSpacePingPong;
    int32_t workspaceM = accumWorkSpacePingPong ? MAX_BLOCK_COUNT * m0 * pValue : m;
    uint64_t gm_scale_workspace_st = static_cast<uint64_t>(aAlignSize) + bAlignSize +
                                     static_cast<uint64_t>(workspaceM) * n * worldSize * sizeof(int32_t);
    gm_scale_workspace = needPerToken ? workspaceGM + gm_scale_workspace_st : 0;

    AllGatherMatmulAIVMode<TemplateAGMMFunc>::AICInit();

    AllGatherMatmulAIVMode<TemplateAGMMFunc>::AIVInit();
}

template <TemplateAGMMClass>
__aicore__ inline void AllGatherMatmulAIVMode<TemplateAGMMFunc>::AICInit()
{
    if ASCEND_IS_AIC {
        SetLoadDataPaddingValue(0);
        SetAtomicNone();
        SetFixpipeNz2ndFlag(1, 0, 0);
        gm_peer_mem = reinterpret_cast<__gm__ supportX1Type *>(stateAddrPerRank[rankId]);
    }
}

template <TemplateAGMMClass>
__aicore__ inline void AllGatherMatmulAIVMode<TemplateAGMMFunc>::AIVInit()
{
    if ASCEND_IS_AIV {
        SetAtomicNone();
        SetMaskNormImpl();
        SetVectorMask<int32_t>((uint64_t)-1, (uint64_t)-1);
        __gm__ supportX1Type *buff[8];
        for (int i = 0; i < worldSize; ++i) {
            buff[i] = reinterpret_cast<__gm__ supportX1Type *>(stateAddrPerRank[i]);
        }

        cal_count = DivCeil(m_loop, pValue);
        gm_a_pingpong_size = m0 * k_align * pValue * worldSize;

        data_len = m * k_align;                    // 数据量
        num_per_rank_move = m0 * k_align * pValue; // 每轮搬运到其他卡的数据量
        src_offset = 0;                            // 当前份数据的起始位置
        rank_offset = rankId * num_per_rank_move;
    }
}

template <TemplateAGMMClass>
__aicore__ inline void AllGatherMatmulAIVMode<TemplateAGMMFunc>::Process()
{
    if ASCEND_IS_AIV {
        TPipe pipe;
        pipe.InitBuffer(uBuf_, USED_UB_SIZE);
        Padding();
        int32_t num_flags = 4;
        // flag[0] - flag[3] 清0
        ResetFlags(num_flags);
        PipeBarrier<PIPE_ALL>();

        for (int32_t cal_idx = 0; cal_idx < cal_count + MAX_BLOCK_COUNT; ++cal_idx) {
            uint64_t flag_idx = cal_idx % MAX_BLOCK_COUNT;

            if (cal_idx == cal_count - 1) {
                num_per_rank_move = data_len - src_offset;
            }

            if (cal_idx == 1) {
                // 聚合perToken scale
                AllGatherPerTokenScale(gm_a_pingpong_size);
            }

            // wait aic
            if (cal_idx >= MAX_BLOCK_COUNT) {
                WaitEvent(flag_idx);
            }

            Mc2AivSync::SetAndWaitAivSync(flag_idx);
            if (cal_idx < cal_count) {
                // Step 2: Rank sync
                CrossRankSyncV1(FLAG_ZERO_IDX, cal_idx + 1);
                Mc2AivSync::SetAndWaitAivSync(flag_idx);
            }

            if (cal_idx < cal_count && aivIdx == 0 && blockIdx < core_count) {
                int64_t gm_rank_offset = static_cast<int64_t>(flag_idx) * gm_a_pingpong_size + rank_offset;
                if constexpr (std::is_same_v<X1Type, AscendC::int4b_t>) {
                    MoveWithSplit(reinterpret_cast<__gm__ int8_t *>(gm_a_src) + src_offset / 2, gm_rank_offset,
                                  num_per_rank_move);
                } else {
                    MoveWithSplit(reinterpret_cast<__gm__ X1Type *>(gm_a_src) + src_offset, gm_rank_offset,
                                  num_per_rank_move);
                }
                src_offset += num_per_rank_move;
            } else if (cal_idx > 0 && cal_idx < cal_count + 1 && aivIdx == 1 && blockIdx >= core_count &&
                       blockIdx < worldSize + core_count) { // peermem to out
                CopyGatherResult(cal_idx);
            }

            // dequant
            if (cal_idx >= MAX_BLOCK_COUNT) {
                Dequant(cal_idx - MAX_BLOCK_COUNT);
            }

            if (cal_idx < cal_count) {
                Mc2AivSync::SetAndWaitAivSync(flag_idx);
                CrossRankSyncV2(FLAG_ONE_IDX, cal_idx + 1);
                Mc2AivSync::SetAndWaitAivSync(flag_idx);
                // 发送aic同步
                Mc2AivSync::SetAicSync(flag_idx);
            }
        }

        ResetFlags(num_flags);

        Mc2AivSync::SetAndWaitAivSync(FLAG_ONE_IDX);

        if (blockIdx < worldSize && aivIdx == 1) {
            CheckBuffFlag((__gm__ int32_t *)stateAddrPerRank[blockIdx] + FLAG_OFFSET + FLAG_ZERO_IDX, uBuf_, 0);
        }

        PipeBarrier<PIPE_ALL>();
    }
    CatlassMatmul();

    SyncAll<false>();
}

template <TemplateAGMMClass>
__aicore__ inline void AllGatherMatmulAIVMode<TemplateAGMMFunc>::ResetFlags(int32_t num_flags)
{
    for (int32_t idx = 0; idx < num_flags; ++idx) {
        if (blockIdx == 0 && aivIdx == 0) {
            SetBuffFlag((__gm__ int32_t *)stateAddrPerRank[rankId] + FLAG_OFFSET + idx, uBuf_, 0);
        }
    }
}

template <TemplateAGMMClass>
__aicore__ inline void AllGatherMatmulAIVMode<TemplateAGMMFunc>::CopyGatherResult(int32_t cal_idx)
{
    // 如果剩余的core数不够，则循环搬运
    int32_t other_core_num = coreNum - core_count;                         // 剩余的core数
    int32_t cycle_num = (other_core_num + worldSize - 1) / other_core_num; // 循环次数
    uint64_t s2_flag_idx = (cal_idx - 1) % MAX_BLOCK_COUNT;
    int64_t src_offset = static_cast<int64_t>(cal_idx - 1) * pValue * m0 * k_align;
    int32_t s2_actual_m = cal_idx == cal_count ? m - (cal_idx - 1) * pValue * m0 : pValue * m0;

    for (int32_t cycle_idx = 0; cycle_idx < cycle_num; ++cycle_idx) {
        int32_t s2_other_rank = blockIdx - core_count + cycle_idx * other_core_num;
        int64_t other_rank_offset = static_cast<int64_t>(s2_flag_idx) * gm_a_pingpong_size +
                                    static_cast<int64_t>(s2_other_rank) * pValue * m0 * k_align;
        int64_t dst_offset =
            static_cast<int64_t>(s2_other_rank) * m * k + static_cast<int64_t>(cal_idx - 1) * pValue * m0 * k;
        if (s2_other_rank >= worldSize) {
            break;
        }

        if (s2_other_rank != rankId) {
            if constexpr (std::is_same_v<X1Type, AscendC::int4b_t>) {
                MoveResultFromPeerMemToOut((__gm__ int8_t *)stateAddrPerRank[rankId] + other_rank_offset / 2,
                                           reinterpret_cast<__gm__ int8_t *>(allgatherGM_) + dst_offset / 2,
                                           s2_actual_m);
            } else {
                MoveResultFromPeerMemToOut((__gm__ X1Type *)stateAddrPerRank[rankId] + other_rank_offset,
                                           reinterpret_cast<__gm__ X1Type *>(allgatherGM_) + dst_offset, s2_actual_m);
            }
        } else {
            if constexpr (std::is_same_v<X1Type, AscendC::int4b_t>) {
                MoveResultFromPeerMemToOut(reinterpret_cast<__gm__ int8_t *>(gm_a_src) + src_offset / 2,
                                           reinterpret_cast<__gm__ int8_t *>(allgatherGM_) + dst_offset / 2,
                                           s2_actual_m);
            } else {
                MoveResultFromPeerMemToOut(reinterpret_cast<__gm__ X1Type *>(gm_a_src) + src_offset,
                                           reinterpret_cast<__gm__ X1Type *>(allgatherGM_) + dst_offset, s2_actual_m);
            }
        }
    }
}

} // namespace AllGatherMatmulAIVModeImpl

#include "all_gather_matmul_aiv_mode_matmul.h"
#include "all_gather_matmul_aiv_mode_data_move.h"
