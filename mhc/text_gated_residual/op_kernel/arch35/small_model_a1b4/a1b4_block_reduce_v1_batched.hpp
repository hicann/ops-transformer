/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or
 * modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS
 * SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT
 * NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of
 * the software repository for the full text of the License.
 */
#pragma once
#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "../tiles/hc_forward_tiles.hpp"
namespace Catlass::Epilogue::SmallModelA1B4 {
struct BlockReduceV1Batched {
    template <class E, class Resource>
    CATLASS_DEVICE static void Run(Resource& resource, GM_ADDR partial, GM_ADDR gatePartial, GM_ADDR act,
                                   GM_ADDR inject, uint32_t M, uint32_t R, uint32_t split, uint32_t gateSplit,
                                   bool gate)
    {
        using namespace AscendC;
        constexpr uint32_t RawFloats = 192u * 1024u / sizeof(float), MaxTile = 2944u;
        constexpr uint32_t NativeOffset = 196608u, SrcOffset = 202496u, WorkOffset = 214272u, OutOffset = 226048u,
                           UbBytes = 231936u;
        static_assert(sizeof(E) == 2, "native half precision required");
        static_assert(UbBytes <= Catlass::Arch::Ascend950::UB_SIZE, "batched fused UB overflow");
        auto acc = resource.ubBuf.template GetBufferByByte<float>(0);
        auto native = resource.ubBuf.template GetBufferByByte<E>(NativeOffset);
        auto src = resource.ubBuf.template GetBufferByByte<float>(SrcOffset);
        auto work = resource.ubBuf.template GetBufferByByte<float>(WorkOffset);
        auto out = resource.ubBuf.template GetBufferByByte<E>(OutOffset);
        GlobalTensor<float> dp, gp;
        dp.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(partial));
        gp.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gatePartial));
        GlobalTensor<E> da, di;
        da.SetGlobalBuffer(reinterpret_cast<__gm__ E*>(act));
        di.SetGlobalBuffer(reinterpret_cast<__gm__ E*>(inject));
        uint32_t sub = GetSubBlockNum() == 0u ? 1u : GetSubBlockNum();
        uint32_t aivNum = GetBlockNum() * sub, id = GetBlockIdx();
        uint64_t elements = static_cast<uint64_t>(M) * R;
        uint32_t tile = (elements / aivNum + 63u) / 64u * 64u;
        if (tile == 0u)
            tile = 64u;
        uint32_t cap = RawFloats / split / 64u * 64u;
        tile = tile < cap ? tile : cap;
        uint32_t loops = (elements + tile - 1u) / tile;
        Tile::TileDivSilu<Catlass::Arch::Ascend950, Gemm::GemmType<float, layout::RowMajor>, 8192, false> silu;
        PipeBarrier<PIPE_ALL>();
        SetFlag<HardEvent::MTE3_MTE2>(0);
        for (uint32_t task = id; task < loops; task += aivNum) {
            uint64_t off = static_cast<uint64_t>(task) * tile;
            uint32_t n = elements - off < tile ? elements - off : tile;
            WaitFlag<HardEvent::MTE3_MTE2>(0);
            DataCopyExtParams cp(split, n * sizeof(float), (elements - n) * sizeof(float), 0, 0);
            DataCopyPadExtParams<float> pad(false, 0, 0, 0);
            DataCopyPad(acc, dp[off], cp, pad);
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);
            for (uint32_t k = 1; k < split; ++k) {
                Add(acc, acc, acc[k * n], n);
                PipeBarrier<PIPE_V>();
            }
            if constexpr (std::is_same_v<E, half>)
                Cast(native, acc, RoundMode::CAST_NONE, n);
            else
                Cast(native, acc, RoundMode::CAST_RINT, n);
            PipeBarrier<PIPE_V>();
            Cast(src, native, RoundMode::CAST_NONE, n);
            PipeBarrier<PIPE_V>();
            Muls(src, src, 0.25f, n);
            PipeBarrier<PIPE_V>();
            Cast(native, src, RoundMode::CAST_RINT, n);
            PipeBarrier<PIPE_V>();
            Cast(src, native, RoundMode::CAST_NONE, n);
            PipeBarrier<PIPE_V>();
            silu(src, work, n, 0.25f, false);
            Cast(out, src, RoundMode::CAST_RINT, n);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            DataCopy(da[off], out, n);
            SetFlag<HardEvent::MTE3_MTE2>(0);
        }
        WaitFlag<HardEvent::MTE3_MTE2>(0);
        // Gate scheduling is independent of down flat tiles. Each AIV owns complete four-row groups.
        if (gate && inject != nullptr) {
            uint32_t groups = (M + 3u) / 4u, begin = groups * id / aivNum * 4u, end = groups * (id + 1u) / aivNum * 4u;
            end = end < M ? end : M;
            for (uint32_t row = begin; row < end; row += 4u) {
                uint32_t nr = end - row < 4u ? end - row : 4u, len = nr * 8u;
                DataCopyExtParams cp(nr, 4u * sizeof(float), (32u - 4u) * sizeof(float), 0, 0);
                DataCopyPadExtParams<float> pad(true, 0, 4, 0.0f);
                for (uint32_t k = 0; k < gateSplit; ++k)
                    DataCopyPad(acc[k * len], gp[static_cast<uint64_t>(k) * M * 32u + static_cast<uint64_t>(row) * 32u],
                                cp, pad);
                SetFlag<HardEvent::MTE2_V>(0);
                WaitFlag<HardEvent::MTE2_V>(0);
                for (uint32_t k = 1; k < gateSplit; ++k) {
                    Add(acc, acc, acc[k * len], len);
                    PipeBarrier<PIPE_V>();
                }
                for (uint32_t j = 0; j < nr; ++j) {
                    if constexpr (std::is_same_v<E, half>)
                        Cast(native[j * 16u], acc[j * 8u], RoundMode::CAST_NONE, 8u);
                    else
                        Cast(native[j * 16u], acc[j * 8u], RoundMode::CAST_RINT, 8u);
                    PipeBarrier<PIPE_V>();
                    Cast(src[j * 8u], native[j * 16u], RoundMode::CAST_NONE, 8u);
                    PipeBarrier<PIPE_V>();
                }
                Muls(src, src, 0.25f, len);
                PipeBarrier<PIPE_V>();
                for (uint32_t j = 0; j < nr; ++j) {
                    Cast(native[j * 16u], src[j * 8u], RoundMode::CAST_RINT, 8u);
                    PipeBarrier<PIPE_V>();
                    Cast(src[j * 8u], native[j * 16u], RoundMode::CAST_NONE, 8u);
                    PipeBarrier<PIPE_V>();
                }
                Muls(acc, src, -1.0f, len);
                PipeBarrier<PIPE_V>();
                Exp(acc, acc, len);
                PipeBarrier<PIPE_V>();
                Adds(acc, acc, 1.0f, len);
                PipeBarrier<PIPE_V>();
                Duplicate(work, 1.0f, len);
                PipeBarrier<PIPE_V>();
                Div(work, work, acc, len);
                PipeBarrier<PIPE_V>();
                Muls(work, work, 2.0f, len);
                PipeBarrier<PIPE_V>();
                for (uint32_t j = 0; j < nr; ++j)
                    Cast(out[j * 16u], work[j * 8u], RoundMode::CAST_RINT, 4u);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(0);
                WaitFlag<HardEvent::V_MTE3>(0);
                DataCopyExtParams outcp(1, 4u * sizeof(E), 0, 0, 0);
                for (uint32_t j = 0; j < nr; ++j)
                    DataCopyPad(di[static_cast<uint64_t>(row + j) * 4u], out[j * 16u], outcp);
                PipeBarrier<PIPE_ALL>();
            }
        }
        PipeBarrier<PIPE_ALL>();
    }
};
} // namespace Catlass::Epilogue::SmallModelA1B4
