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
 * \file block_epilogue_pertoken_v2.hpp
 * \brief
 */

#ifndef CATLASS_EPILOGUE_BLOCK_EPILOGUE_PER_TOKEN_V2_ONLY_HPP
#define CATLASS_EPILOGUE_BLOCK_EPILOGUE_PER_TOKEN_V2_ONLY_HPP

#include <limits>
#include <type_traits>

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/detail/callback.hpp"

#include "const_args.hpp"
#include "hccl_shmem.hpp"
#include "layout3d.hpp"
#include "per_token_v2_route_utils.hpp"
#include "dispatch_policy_custom.hpp"

namespace Catlass::Epilogue::Block {

template <bool IS_A2_>
struct IS_A2_T {
    static constexpr bool Value = IS_A2_;
};

// 选择 A3 按返回索引写回的后处理特化。
template <bool BLOCKED_OUTPUT>
struct MaskIndexedRouteTagT {
    static constexpr uint32_t RANK_SHIFT = 16;
    static constexpr uint32_t SLOT_MASK = (1U << RANK_SHIFT) - 1;
};

template <bool BLOCKED_OUTPUT>
struct MaskOutputLayoutState {};

template <>
struct MaskOutputLayoutState<true> {
    uint32_t blockedRows{0};
};

constexpr uint32_t MASK_PAIR_ROWS = 2;

template <class T>
CATLASS_DEVICE bool CopyMaskPair(AscendC::GlobalTensor<T> const& dst, AscendC::LocalTensor<T> const& src,
                                 uint32_t first, uint32_t second, uint32_t columns, uint32_t dstStride, uint32_t ubGap,
                                 uint32_t column = 0)
{
    const uint32_t bytes = columns * sizeof(T);
    const uint64_t distance = static_cast<uint64_t>(second - first) * dstStride * sizeof(T);
    if (second <= first || distance < bytes || distance - bytes > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    AscendC::DataCopyPad(dst[static_cast<uint64_t>(first) * dstStride + column], src,
                         {MASK_PAIR_ROWS, bytes, ubGap, static_cast<uint32_t>(distance - bytes), 0});
    return true;
}

namespace detail {

// Only A2 routes cross-server output through a separate window.
// An empty base keeps this state out of A3 Params.
template <bool IS_A2>
struct PerTokenV2RouteState {
    CATLASS_DEVICE PerTokenV2RouteState() {}
};

template <>
struct PerTokenV2RouteState<true> {
    int64_t offsetWinOutD{0};
    int32_t serverId{0};

    CATLASS_DEVICE PerTokenV2RouteState() {}

    CATLASS_DEVICE PerTokenV2RouteState(int64_t offsetWinOutD_, int32_t serverId_)
        : offsetWinOutD(offsetWinOutD_),
          serverId(serverId_)
    {}
};

template <class BaseParams, bool BLOCKED_OUTPUT>
struct PerTokenV2MaskParams : BaseParams, MaskOutputLayoutState<BLOCKED_OUTPUT> {
    using BaseParams::BaseParams;
    GM_ADDR ptrRouteIndex{nullptr};
};

template <bool MASK_ROUTING, uint32_t UB_STAGES>
struct PerTokenV2MaskBuffers {};

template <uint32_t UB_STAGES>
struct PerTokenV2MaskBuffers<true, UB_STAGES> {
    AscendC::LocalTensor<int32_t> routeIndexUbList[UB_STAGES];
};

// A2 and A3 share the tile pipeline. Platform and dequantization choices are
// fixed at compile time, retaining each policy's event and copy sequence.
template <class DispatchPolicy_, class CType_, class LayoutPerTokenScale_, class DType_, class TileCopy_, bool IS_A2_,
          bool NEED_DEQUANT_, bool MASK_ROUTING_ = false, bool BLOCKED_OUTPUT_ = false>
class PerTokenV2Impl {
public:
    using DispatchPolicy = DispatchPolicy_;
    using ArchTag = typename DispatchPolicy::ArchTag;
    static constexpr uint32_t UB_STAGES = DispatchPolicy::UB_STAGES;
    static constexpr bool IS_A2 = IS_A2_;
    static constexpr bool NEED_DEQUANT = NEED_DEQUANT_;

    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;
    using ElementPerTokenScale = float;
    using LayoutPerTokenScale = LayoutPerTokenScale_;
    using ElementD = typename DType_::Element;
    using LayoutD = typename DType_::Layout;

    using CopyScaleGmToUb = Epilogue::Tile::CopyGm2Ub<ArchTag, Gemm::GemmType<float, layout::VectorLayout>>;
    using CopyGmToUbC = typename TileCopy_::CopyGmToUbC;
    using CopyUbToGmD = typename TileCopy_::CopyUbToGmD;

    struct Params : PerTokenV2RouteState<IS_A2> {
        __gm__ int32_t* ptrTokenPerExpert{nullptr};
        int32_t EP;
        int32_t expertPerRank;
        int32_t n2;
        LayoutC layoutC;
        int32_t n0;
        int32_t rank;
        HcclShmem<IS_A2> shmem;
        int64_t offsetD;
        Layout3D tokenPerExpertLayout;
        bool useUbRouteCache{false};
        int32_t ubRouteStride{0};
        AscendC::LocalTensor<int32_t> ubTokenPerExpert;
        AscendC::LocalTensor<int32_t> ubPreSumBeforeRank;

        CATLASS_DEVICE
        Params() {};
        CATLASS_DEVICE
        Params(int32_t EP_, int32_t expertPerRank_, int32_t rank_, __gm__ int32_t* ptrTokenPerExpert_, LayoutC layoutC_,
               int32_t n2_, int32_t n0_, HcclShmem<IS_A2>& shmem_, int64_t offsetD_, Layout3D tokenPerExpertLayout_)
            : ptrTokenPerExpert(ptrTokenPerExpert_),
              EP(EP_),
              expertPerRank(expertPerRank_),
              rank(rank_),
              layoutC(layoutC_),
              n2(n2_),
              n0(n0_),
              shmem(shmem_),
              offsetD(offsetD_),
              tokenPerExpertLayout(tokenPerExpertLayout_)
        {}
        template <bool ENABLE_A2 = IS_A2, std::enable_if_t<ENABLE_A2 && IS_A2, int> = 0>
        CATLASS_DEVICE Params(int32_t EP_, int32_t expertPerRank_, int32_t rank_, __gm__ int32_t* ptrTokenPerExpert_,
                              LayoutC layoutC_, int32_t n2_, int32_t n0_, HcclShmem<IS_A2>& shmem_, int64_t offsetD_,
                              int64_t offsetWinOutD_, int32_t serverId_, Layout3D tokenPerExpertLayout_)
            : PerTokenV2RouteState<IS_A2>(offsetWinOutD_, serverId_),
              ptrTokenPerExpert(ptrTokenPerExpert_),
              EP(EP_),
              expertPerRank(expertPerRank_),
              rank(rank_),
              layoutC(layoutC_),
              n2(n2_),
              n0(n0_),
              shmem(shmem_),
              offsetD(offsetD_),
              tokenPerExpertLayout(tokenPerExpertLayout_)
        {}
    };

    using RouteParams = std::conditional_t<MASK_ROUTING_, PerTokenV2MaskParams<Params, BLOCKED_OUTPUT_>, Params>;

    CATLASS_DEVICE
    PerTokenV2Impl(Arch::Resource<ArchTag> const& resource, RouteParams const& params = RouteParams{})
        : params(params)
    {
        n0 = params.n0;
        size_t ubOffset = 0;
        for (int32_t i = 0; i < (MASK_ROUTING_ ? UB_STAGES : 2); i++) {
            ubCList[i] = resource.ubBuf.template GetBufferByByte<ElementC>(ubOffset);
            ubOffset += max_len * sizeof(ElementC);
            ubDList[i] = resource.ubBuf.template GetBufferByByte<ElementD>(ubOffset);
            ubOffset += max_len * sizeof(ElementD);
            ubFp32List[i] = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
            ubOffset += max_len * sizeof(float);
            scaleUbList[i] = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
            ubOffset += (max_len / n0) * sizeof(float);
            source_scale_offset[i] = -1;
        }
        if constexpr (MASK_ROUTING_) {
            for (uint32_t i = 0; i < UB_STAGES; ++i) {
                maskBuffers.routeIndexUbList[i] = resource.ubBuf.template GetBufferByByte<int32_t>(ubOffset);
                ubOffset += 128 * sizeof(int32_t);
            }
        } else {
            tokenPerExpert.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.ptrTokenPerExpert));
            tokenPerExpertLayout = params.tokenPerExpertLayout;
        }
        is_ping = true;
    }

    CATLASS_DEVICE
    void SetFlag()
    {
        if constexpr (!IS_A2) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
            AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID3);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        }
    }

    CATLASS_DEVICE
    void InitFlag() {}

    CATLASS_DEVICE
    void Finalize()
    {
        if constexpr (IS_A2 && NEED_DEQUANT) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        } else if constexpr (IS_A2) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        }
    }

    CATLASS_DEVICE
    ~PerTokenV2Impl() {}

    CATLASS_DEVICE int32_t GetLenRankInExpert(int32_t dstEpIdx, int32_t groupIdx) const
    {
        return ::Catlass::Epilogue::Block::detail::GetLenRankInExpert(params, tokenPerExpert, tokenPerExpertLayout,
                                                                      dstEpIdx, groupIdx);
    }

    CATLASS_DEVICE int32_t GetDstExpertOffset(int32_t dstEpIdx, int32_t groupIdx,
                                              AscendC::GlobalTensor<int32_t> const& preSumBeforeRank) const
    {
        return ::Catlass::Epilogue::Block::detail::GetDstExpertOffset(params, preSumBeforeRank, dstEpIdx, groupIdx);
    }

    template <typename TileElement>
    CATLASS_DEVICE void CopyRoutedSlice(AscendC::LocalTensor<TileElement>& ubTile, GemmCoord& blockCoord,
                                        GemmCoord& actualBlockShape, int32_t preSrcExpertSum, int32_t dstEpIdx,
                                        int32_t stData, uint32_t lenData, uint32_t dstOffsetInExpert,
                                        int32_t dstExpertOffset, int32_t tileOffset)
    {
        AscendC::GlobalTensor<ElementD> gmRemotePeer;
        __gm__ void* dstPeermemPtr = params.shmem(params.offsetD, dstEpIdx);
        gmRemotePeer.SetGlobalBuffer(reinterpret_cast<__gm__ ElementD*>(dstPeermemPtr));
        MatrixCoord dstOffset{dstOffsetInExpert + dstExpertOffset, blockCoord.n()};
        int64_t gmDstOffset = params.layoutC.GetOffset(dstOffset);
        auto gmTileD = gmRemotePeer[gmDstOffset];
        LayoutC layoutGM2{lenData, actualBlockShape.n(), params.n2};
        LayoutC layoutUB2{lenData, actualBlockShape.n(), n0};
        if constexpr (IS_A2) {
            bool isCrossServer = (dstEpIdx / SERVER_RANK_SIZE_A2) != params.serverId;
            if (isCrossServer) {
                AscendC::GlobalTensor<ElementD> gmLocalWindowsOut;
                gmLocalWindowsOut.SetGlobalBuffer(
                    reinterpret_cast<__gm__ ElementD*>(params.shmem.windowsOutAddr() + params.offsetWinOutD));
                MatrixCoord srcOffset{(uint32_t)(stData + preSrcExpertSum), blockCoord.n()};
                int64_t gmSrcOffset = params.layoutC.GetOffset(srcOffset);
                auto gmTileLocal = gmLocalWindowsOut[gmSrcOffset];
                copyUbToGmD(gmTileLocal, ubTile[tileOffset * n0], layoutGM2, layoutUB2);
            } else {
                copyUbToGmD(gmTileD, ubTile[tileOffset * n0], layoutGM2, layoutUB2);
            }
        } else {
            copyUbToGmD(gmTileD, ubTile[tileOffset * n0], layoutGM2, layoutUB2);
        }
    }

    template <typename TileElement>
    CATLASS_DEVICE void CopyRoutedTile(AscendC::LocalTensor<TileElement>& ubTile, GemmCoord& blockCoord,
                                       GemmCoord& actualBlockShape, int32_t groupIdx, int32_t preSrcExpertSum,
                                       AscendC::GlobalTensor<int32_t> const& preSumBeforeRank, int32_t stTile,
                                       int32_t edTile, int32_t& preSumRankInExpert, int32_t& tileOffset)
    {
        if constexpr (MASK_ROUTING_) {
            using MaskRoute = MaskIndexedRouteTagT<BLOCKED_OUTPUT_>;
            auto& routeIndex = maskBuffers.routeIndexUbList[is_ping];
            int32_t lenTile = actualBlockShape.m();
            AscendC::GlobalTensor<ElementD> gmRemotePeer;
            uint32_t previousRank = std::numeric_limits<uint32_t>::max();
            for (uint32_t row = 0; row < lenTile;) {
                const uint32_t route = routeIndex.GetValue(row);
                const uint32_t dstEpIdx = route >> MaskRoute::RANK_SHIFT;
                uint32_t dstRow = route & MaskRoute::SLOT_MASK;
                if (dstEpIdx != previousRank) {
                    previousRank = dstEpIdx;
                    __gm__ void* dstPeermemPtr = params.shmem(params.offsetD, dstEpIdx);
                    if constexpr (BLOCKED_OUTPUT_) {
                        // 通信 D 区按 [列块][路由槽位][块内列] 排列，此处定位当前列块的起点。
                        gmRemotePeer.SetGlobalBuffer(reinterpret_cast<__gm__ ElementD*>(dstPeermemPtr) +
                                                     static_cast<uint64_t>(blockCoord.n()) * params.blockedRows);
                    } else {
                        gmRemotePeer.SetGlobalBuffer(reinterpret_cast<__gm__ ElementD*>(dstPeermemPtr));
                    }
                }
                auto dst = gmRemotePeer[static_cast<uint64_t>(dstRow) * params.n2 + blockCoord.n()];
                if constexpr (BLOCKED_OUTPUT_) {
                    dst = gmRemotePeer[static_cast<uint64_t>(dstRow) * n0];
                }
                if (row + 1 < lenTile) {
                    const uint32_t nextRoute = routeIndex.GetValue(row + 1);
                    bool paired = (nextRoute >> MaskRoute::RANK_SHIFT) == dstEpIdx &&
                                  CopyMaskPair(gmRemotePeer, ubTile[row * n0], dstRow, nextRoute & MaskRoute::SLOT_MASK,
                                               actualBlockShape.n(), BLOCKED_OUTPUT_ ? n0 : params.n2,
                                               (n0 - actualBlockShape.n()) * sizeof(ElementD) / BYTE_PER_BLK,
                                               BLOCKED_OUTPUT_ ? 0 : blockCoord.n());
                    if (paired) {
                        row += MASK_PAIR_ROWS;
                        continue;
                    }
                }
                copyUbToGmD(dst, ubTile[row * n0], LayoutC{1, actualBlockShape.n(), params.n2},
                            LayoutC{1, actualBlockShape.n(), n0});
                ++row;
            }
            return;
        }
        for (int32_t dstEpIdx = 0; dstEpIdx < params.EP; dstEpIdx++) {
            int32_t lenRankInExpert = GetLenRankInExpert(dstEpIdx, groupIdx);
            int32_t dstExpertOffset = GetDstExpertOffset(dstEpIdx, groupIdx, preSumBeforeRank);
            auto routeRange =
                ::Catlass::Epilogue::Block::detail::GetRouteRange(lenRankInExpert, stTile, edTile, preSumRankInExpert);
            if (routeRange.action == ::Catlass::Epilogue::Block::detail::RouteAction::STOP) {
                break;
            }
            if (routeRange.action == ::Catlass::Epilogue::Block::detail::RouteAction::SKIP) {
                continue;
            }
            int32_t stData = routeRange.stData;
            uint32_t lenData = routeRange.lenData;
            uint32_t dstOffsetInExpert = routeRange.dstOffsetInExpert;
            CopyRoutedSlice(ubTile, blockCoord, actualBlockShape, preSrcExpertSum, dstEpIdx, stData, lenData,
                            dstOffsetInExpert, dstExpertOffset, tileOffset);
            tileOffset += lenData;
        }
    }

    template <typename EventId>
    CATLASS_DEVICE void ApplyTokenScale(AscendC::GlobalTensor<ElementPerTokenScale> const& gmPerTokenScale,
                                        GemmCoord& blockCoord, GemmCoord& actualBlockShape, int32_t preSrcExpertSum,
                                        EventId event_id, EventId event_id_2, AscendC::LocalTensor<float>& ubCFp32,
                                        AscendC::LocalTensor<ElementPerTokenScale>& scaleUb)
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event_id_2);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(event_id_2);

        int32_t gmScaleOffset = preSrcExpertSum + blockCoord.m();
        layout::VectorLayout scaleLayout{actualBlockShape.m()};
        if (source_scale_offset[event_id] != gmScaleOffset) {
            source_scale_offset[event_id] = gmScaleOffset;
            copyScaleGmToUb(scaleUb, gmPerTokenScale[gmScaleOffset], scaleLayout, scaleLayout);
            if constexpr (MASK_ROUTING_) {
                AscendC::GlobalTensor<int32_t> routeIndex;
                routeIndex.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.ptrRouteIndex));
                AscendC::DataCopyPad(maskBuffers.routeIndexUbList[is_ping], routeIndex[gmScaleOffset],
                                     {1, static_cast<uint32_t>(actualBlockShape.m() * sizeof(int32_t)), 0, 0, 0}, {});
            }
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(event_id_2);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event_id_2);

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event_id_2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(event_id_2);
        AscendC::PipeBarrier<PIPE_V>();
        for (int32_t row = 0; row < actualBlockShape.m(); ++row) {
            float scale = scaleUb(row);
            Muls<float, false>(ubCFp32[n0 * row], ubCFp32[n0 * row], scale, -1, (actualBlockShape.n() + 127) / 128 * 2,
                               {1, 1, 8, 8});
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementC> const& gmC,
                    AscendC::GlobalTensor<ElementPerTokenScale> const& gmPerTokenScale, GemmCoord& blockCoord,
                    GemmCoord& actualBlockShape, int32_t groupIdx, int32_t preSrcExpertSum,
                    AscendC::GlobalTensor<int32_t> preSumBeforeRank)
    {
        if constexpr (!NEED_DEQUANT) {
            (void)gmPerTokenScale;
            operator()(gmC, blockCoord, actualBlockShape, groupIdx, preSrcExpertSum, preSumBeforeRank);
            return;
        }
        is_ping = !is_ping;
        auto event_id = is_ping ? EVENT_ID0 : EVENT_ID1;
        auto event_id_2 = is_ping ? EVENT_ID2 : EVENT_ID3;

        auto& ubC = ubCList[is_ping];
        auto& ubD = ubDList[is_ping];
        using GmCOffset = std::conditional_t<IS_A2, int32_t, int64_t>;
        GmCOffset gmCOffset =
            static_cast<GmCOffset>(preSrcExpertSum) * params.n2 + blockCoord.m() * params.n2 + blockCoord.n();
        auto gmTileC = gmC[gmCOffset];
        auto& ubCFp32 = ubFp32List[is_ping];
        auto& scaleUb = scaleUbList[is_ping];

        LayoutC layoutGM{actualBlockShape.m(), actualBlockShape.n(), params.n2};
        LayoutC layoutUB{actualBlockShape.m(), actualBlockShape.n(), n0};

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event_id);
        copyGmToUbC(ubC, gmTileC, layoutUB, layoutGM);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event_id);

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event_id);
        AscendC::Cast<float, ElementC, false>(ubCFp32, ubC, AscendC::RoundMode::CAST_NONE, -1, repeat, {1, 1, 8, 4});
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event_id);

        ApplyTokenScale(gmPerTokenScale, blockCoord, actualBlockShape, preSrcExpertSum, event_id, event_id_2, ubCFp32,
                        scaleUb);

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(event_id);
        AscendC::Cast<ElementD, float, false>(ubD, ubCFp32, AscendC::RoundMode::CAST_RINT, -1, repeat, {1, 1, 4, 8});
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(event_id_2);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event_id_2);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event_id);

        int32_t lenTile = actualBlockShape.m();
        int32_t stTile = blockCoord.m();
        int32_t edTile = stTile + lenTile;
        int32_t preSumRankInExpert = 0;
        int32_t tileOffset = 0;

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event_id);
        CopyRoutedTile(ubD, blockCoord, actualBlockShape, groupIdx, preSrcExpertSum, preSumBeforeRank, stTile, edTile,
                       preSumRankInExpert, tileOffset);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(event_id);
    }

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementC> const& gmC, GemmCoord& blockCoord, GemmCoord& actualBlockShape,
                    int32_t groupIdx, int32_t preSrcExpertSum, AscendC::GlobalTensor<int32_t> preSumBeforeRank)
    {
        is_ping = !is_ping;
        auto event_id = is_ping ? EVENT_ID0 : EVENT_ID1;

        auto& ubC = ubCList[is_ping];
        int32_t gmCOffset = preSrcExpertSum * params.n2 + blockCoord.m() * params.n2 + blockCoord.n();
        auto gmTileC = gmC[gmCOffset];

        LayoutC layoutGM{actualBlockShape.m(), actualBlockShape.n(), params.n2};
        LayoutC layoutUB{actualBlockShape.m(), actualBlockShape.n(), n0};

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
        copyGmToUbC(ubC, gmTileC, layoutUB, layoutGM);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);

        int32_t lenTile = actualBlockShape.m();
        int32_t stTile = blockCoord.m();
        int32_t edTile = stTile + lenTile;
        int32_t preSumRankInExpert = 0;
        int32_t tileOffset = 0;

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);
        CopyRoutedTile(ubC, blockCoord, actualBlockShape, groupIdx, preSrcExpertSum, preSumBeforeRank, stTile, edTile,
                       preSumRankInExpert, tileOffset);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
    }

private:
    RouteParams params;
    AscendC::LocalTensor<ElementC> ubCList[UB_STAGES];
    AscendC::LocalTensor<ElementD> ubDList[UB_STAGES];
    AscendC::LocalTensor<float> ubFp32List[UB_STAGES];
    AscendC::LocalTensor<float> scaleUbList[UB_STAGES];
    PerTokenV2MaskBuffers<MASK_ROUTING_, UB_STAGES> maskBuffers;
    int32_t source_scale_offset[UB_STAGES];

    int32_t max_len = 8 * 32 / 4 * 128;
    int32_t n0;
    bool is_ping = false;

    int32_t repeat = 128;

    CopyGmToUbC copyGmToUbC;
    CopyUbToGmD copyUbToGmD;

    CopyScaleGmToUb copyScaleGmToUb;
    AscendC::GlobalTensor<int32_t> tokenPerExpert;
    Layout3D tokenPerExpertLayout;
};

} // namespace detail

// A3 retains its IS_A2_T dispatch tag and its SetFlag/Finalize protocol.
template <uint32_t UB_STAGES_, class CType_, class LayoutPerTokenScale_, class DType_, class TileCopy_, bool IS_A2_>
class BlockEpilogue<EpilogueAtlasA2PerTokenDequantV2<UB_STAGES_>, CType_, Gemm::GemmType<float, LayoutPerTokenScale_>,
                    DType_, TileCopy_, IS_A2_T<IS_A2_>>
    : public detail::PerTokenV2Impl<EpilogueAtlasA2PerTokenDequantV2<UB_STAGES_>, CType_, LayoutPerTokenScale_, DType_,
                                    TileCopy_, IS_A2_, true> {
    using Base = detail::PerTokenV2Impl<EpilogueAtlasA2PerTokenDequantV2<UB_STAGES_>, CType_, LayoutPerTokenScale_,
                                        DType_, TileCopy_, IS_A2_, true>;

public:
    using Base::Base;
};

template <uint32_t UB_STAGES_, class CType_, class LayoutPerTokenScale_, class DType_, class TileCopy_,
          bool BLOCKED_OUTPUT>
class BlockEpilogue<EpilogueAtlasA2PerTokenDequantV2<UB_STAGES_>, CType_, Gemm::GemmType<float, LayoutPerTokenScale_>,
                    DType_, TileCopy_, MaskIndexedRouteTagT<BLOCKED_OUTPUT>>
    : public detail::PerTokenV2Impl<EpilogueAtlasA2PerTokenDequantV2<UB_STAGES_>, CType_, LayoutPerTokenScale_, DType_,
                                    TileCopy_, false, true, true, BLOCKED_OUTPUT> {
    using Base = detail::PerTokenV2Impl<EpilogueAtlasA2PerTokenDequantV2<UB_STAGES_>, CType_, LayoutPerTokenScale_,
                                        DType_, TileCopy_, false, true, true, BLOCKED_OUTPUT>;

public:
    static constexpr bool MASK_ROUTING = true;
    static constexpr bool MASK_BLOCKED_OUTPUT = BLOCKED_OUTPUT;
    using Params = typename Base::RouteParams;
    using Base::Base;
};

template <uint32_t UB_STAGES_, class CType_, class LayoutPerTokenScale_, class DType_, class TileCopy_>
class BlockEpilogue<EpilogueAtlasA2PerTokenDequantV2Int8<UB_STAGES_>, CType_,
                    Gemm::GemmType<float, LayoutPerTokenScale_>, DType_, TileCopy_>
    : public detail::PerTokenV2Impl<EpilogueAtlasA2PerTokenDequantV2Int8<UB_STAGES_>, CType_, LayoutPerTokenScale_,
                                    DType_, TileCopy_, true, true> {
    using Base = detail::PerTokenV2Impl<EpilogueAtlasA2PerTokenDequantV2Int8<UB_STAGES_>, CType_, LayoutPerTokenScale_,
                                        DType_, TileCopy_, true, true>;

public:
    using Base::Base;
};

template <uint32_t UB_STAGES_, class CType_, class LayoutPerTokenScale_, class DType_, class TileCopy_>
class BlockEpilogue<EpilogueAtlasA2PerTokenDequantV2BF16<UB_STAGES_>, CType_,
                    Gemm::GemmType<float, LayoutPerTokenScale_>, DType_, TileCopy_>
    : public detail::PerTokenV2Impl<EpilogueAtlasA2PerTokenDequantV2BF16<UB_STAGES_>, CType_, LayoutPerTokenScale_,
                                    DType_, TileCopy_, true, false> {
    using Base = detail::PerTokenV2Impl<EpilogueAtlasA2PerTokenDequantV2BF16<UB_STAGES_>, CType_, LayoutPerTokenScale_,
                                        DType_, TileCopy_, true, false>;

public:
    using Base::Base;
};

} // namespace Catlass::Epilogue::Block
#endif
