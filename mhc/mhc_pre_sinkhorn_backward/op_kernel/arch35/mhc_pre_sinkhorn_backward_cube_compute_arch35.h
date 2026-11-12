/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN
 * Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not
 * use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT
 * WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY,
 * OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the
 * License.
 */
#ifndef MHC_PRE_SINKHORN_BACKWARD_CUBE_COMPUTE_H
#define MHC_PRE_SINKHORN_BACKWARD_CUBE_COMPUTE_H
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "mhc_pre_sinkhorn_backward_utils_arch35.h"

namespace MhcPreSinkhornBackwardNs {
using namespace AscendC;
using namespace MhcPreSinkhornBackwardUtils;

constexpr uint32_t CUBE_L0_EVENT_BASE = 3U;
constexpr uint32_t CUBE_BLOCK_BYTES = 32U;
constexpr FixpipeConfig C0_UB_CONFIG = {CO2Layout::ROW_MAJOR, true};

__aicore__ inline uint64_t BasicCubeCeilDiv(uint64_t x, uint64_t n)
{
    return (x + n - 1U) / n;
}

__aicore__ inline uint64_t BasicCubeAlign(uint64_t x, uint64_t n)
{
    return BasicCubeCeilDiv(x, n) * n;
}

class MhcPreSinkhornBackwardCubeCompute {
public:
    __aicore__ inline void Init(TPipe* pipe, const MhcPreSinkhornBackwardArch35TilingData* tiling, bool directC0);

    // C0/C1 计算流程
    __aicore__ inline void ProcessC0(const MMConfig& config, const GlobalTensor<float>& a, const GlobalTensor<float>& b,
                                     const GlobalTensor<float>& c, uint32_t m, uint32_t n, uint32_t k,
                                     uint32_t bGmStride, uint64_t rowStart, uint32_t localColumn);

    __aicore__ inline void ProcessC1(const MMConfig& config, const GlobalTensor<float>& a, const GlobalTensor<float>& b,
                                     const GlobalTensor<float>& c, uint32_t m, uint32_t n, uint32_t k,
                                     uint32_t aGmStride, uint32_t bGmStride, uint32_t cGmStride, bool atomicAdd,
                                     uint64_t rowStart);

    // L1 缓存与流水同步
    __aicore__ inline void PrepareFirstC0(const MMConfig& config, const GlobalTensor<float>& b, uint32_t columns,
                                          uint32_t k, uint32_t stride);

    __aicore__ inline void PreloadHMixGrad(const GlobalTensor<float>& hMixGrad);

    __aicore__ inline void PrepareHMixGradTile(const GlobalTensor<float>& hMixGrad, uint32_t rows, uint64_t rowStart,
                                               bool loadNewTile);
    __aicore__ inline void ReleaseHMixGradTile(uint64_t rowStart);

    __aicore__ inline void WaitC0UbBuffersReleased();
    __aicore__ inline void End();

private:
    __aicore__ inline void InitOperandCache(TPipe* pipe, const MhcPreSinkhornBackwardArch35TilingData* tiling);
    __aicore__ inline LocalTensor<float> GetHMixGradTile(uint64_t rowStart) const;
    __aicore__ inline void ClearL1Buffer(const LocalTensor<float>& dst, uint32_t elements);

    __aicore__ inline void PreparePhiCache(const GlobalTensor<float>& phi, uint32_t columns, uint32_t gmStride);

    __aicore__ inline void ComputeTileC0(const MMConfig& config, const GlobalTensor<float>& a,
                                         const GlobalTensor<float>& c, uint32_t m, uint32_t n, uint32_t k,
                                         uint32_t aGmStride, uint32_t cGmStride, const LocalTensor<float>& cachedA,
                                         const LocalTensor<float>& cachedB);

    __aicore__ inline void ComputeTileC1(const MMConfig& config, const GlobalTensor<float>& a,
                                         const GlobalTensor<float>& b, const GlobalTensor<float>& c, uint32_t m,
                                         uint32_t n, uint32_t k, uint32_t aGmStride, uint32_t bGmStride,
                                         uint32_t cGmStride, bool atomicAdd, const LocalTensor<float>& cachedA);

    __aicore__ inline void ProcessKRangeC0(const MMConfig& config, const LocalTensor<float>& currentC,
                                           const GlobalTensor<float>& a, uint32_t m, uint32_t n, uint32_t k,
                                           uint32_t aGmStride, const LocalTensor<float>& cachedA,
                                           const LocalTensor<float>& cachedB);

    __aicore__ inline void ProcessKRangeC1(const MMConfig& config, const LocalTensor<float>& currentC,
                                           const GlobalTensor<float>& a, const GlobalTensor<float>& b, uint32_t m,
                                           uint32_t n, uint32_t k, uint32_t aGmStride, uint32_t bGmStride,
                                           const LocalTensor<float>& cachedA);

    __aicore__ inline void ProcessL0KRange(const MMConfig& config, const LocalTensor<float>& currentC,
                                           const LocalTensor<float>& currentA1, const LocalTensor<float>& currentB1,
                                           uint32_t m, uint32_t n, uint32_t currentKL1, uint32_t aL1Stride,
                                           uint32_t bL1Stride, bool transA, bool initC);

    __aicore__ inline void FinishC0Tile(const GlobalTensor<float>& c, const LocalTensor<float>& currentC, uint32_t m,
                                        uint32_t n, uint32_t cGmStride, uint32_t cBufferIndex);

    __aicore__ inline void FinishTile(const GlobalTensor<float>& c, const LocalTensor<float>& currentC, uint32_t m,
                                      uint32_t n, uint32_t cGmStride, bool atomicAdd, uint32_t cBufferIndex);

    __aicore__ inline void PreloadB1(const MMConfig& config, const GlobalTensor<float>& b, uint32_t n, uint32_t k,
                                     uint32_t bGmStride);

    __aicore__ inline void CopyInA1(const LocalTensor<float>& dst, const GlobalTensor<float>& src, uint32_t m,
                                    uint32_t k, uint32_t gmStride);

    __aicore__ inline void CopyInA1Trans(const LocalTensor<float>& dst, const GlobalTensor<float>& src, uint32_t m,
                                         uint32_t k, uint32_t gmStride);

    __aicore__ inline void CopyInB1(const LocalTensor<float>& dst, const GlobalTensor<float>& src, uint32_t n,
                                    uint32_t k, uint32_t gmStride);

    __aicore__ inline void CopyInA2(const LocalTensor<float>& dst, const LocalTensor<float>& src, uint32_t m,
                                    uint32_t k, uint32_t rowStride);

    __aicore__ inline void CopyInA2Trans(const LocalTensor<float>& dst, const LocalTensor<float>& src, uint32_t m,
                                         uint32_t k, uint32_t kL1);

    __aicore__ inline void CopyInB2(const LocalTensor<float>& dst, const LocalTensor<float>& src, uint32_t n,
                                    uint32_t k, uint32_t kL1);

    __aicore__ inline void Compute(const LocalTensor<float>& c, const LocalTensor<float>& a,
                                   const LocalTensor<float>& b, uint32_t m, uint32_t n, uint32_t k, bool initC);

    __aicore__ inline void CopyOut(const GlobalTensor<float>& dst, const LocalTensor<float>& src, uint32_t m,
                                   uint32_t n, uint32_t dstStride);

    TBuf<TPosition::A1> l1Buf_;
    // 缓存布局：[可选 hMixGrad 槽 0][可选槽 1][可选 Phi]。
    TBuf<TPosition::A1> operandCacheBuf_;
    LocalTensor<float> hMixGradCache_;
    LocalTensor<float> phiCache_;
    HMixGradCacheMode hMixGradMode_ = HMixGradCacheMode::NONE;
    bool cachePhi_ = false;
    uint32_t hMixGradRows_ = 0;
    uint32_t hMixGradValidRows_ = 0;
    uint32_t hMixGradColumns_ = 0;
    uint32_t hMixGradCacheElements_ = 0;
    bool firstC0BReady_ = false;
    bool directC0_ = false; // C0 结果直传 AIV UB
    uint32_t c0UbBufferIndex_ = 0;
    LocalTensor<float> aL1Local_;
    LocalTensor<float> bL1Local_;
    LocalTensor<float> aL0Local_;
    LocalTensor<float> bL0Local_;
    LocalTensor<float> cL0Local_;
    uint32_t l1PingPongID_{0};
    uint32_t l0PingPongID_{0};
    uint32_t cl0PingPongID_{0};
    uint32_t maxBufferDepth_{0};
    uint32_t l1SingleBufferElems_{0};
    uint32_t l0ABSingleBufferElems_{0};
    uint32_t l0CSingleBufferElems_{0};
};

__aicore__ inline void MhcPreSinkhornBackwardCubeCompute::ClearL1Buffer(const LocalTensor<float>& dst,
                                                                        uint32_t elements)
{
    const uint16_t blockNum = static_cast<uint16_t>(elements * sizeof(float) / CUBE_BLOCK_BYTES);
    InitConstValueParams<float> zeroParams(1, blockNum, 0, 0.0f);
    InitConstValue(dst, zeroParams);
    PipeBarrier<PIPE_MTE2>();
}

__aicore__ inline void MhcPreSinkhornBackwardCubeCompute::PrepareFirstC0(const MMConfig& config,
                                                                         const GlobalTensor<float>& b, uint32_t columns,
                                                                         uint32_t k, uint32_t stride)
{
    // V0 期间预加载 Phi；hMixGrad 有效数据在全核同步后加载。
    if (cachePhi_) {
        PreparePhiCache(b, columns, stride);
        return;
    }
    // 普通路径仅预加载首个 ND 分块，后续 C0 仍按需加载 B1。
    PreloadB1(config, b, Min(ND_BLOCK_SIZE, columns), k, stride);
    firstC0BReady_ = true;
}

__aicore__ inline void MhcPreSinkhornBackwardCubeCompute::PreparePhiCache(const GlobalTensor<float>& phi,
                                                                          uint32_t columns, uint32_t gmStride)
{
    CopyInB1(phiCache_, phi, columns, hMixGradColumns_, gmStride);
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID5);
}

__aicore__ inline LocalTensor<float> MhcPreSinkhornBackwardCubeCompute::GetHMixGradTile(uint64_t rowStart) const
{
    const uint32_t slot = hMixGradMode_ == HMixGradCacheMode::TILED ? (rowStart / BS_BLOCK_SIZE) % DOUBLE_BUFFER : 0U;
    return hMixGradCache_[slot * hMixGradCacheElements_];
}

__aicore__ inline void MhcPreSinkhornBackwardCubeCompute::ReleaseHMixGradTile(uint64_t rowStart)
{
    const uint32_t slot = (rowStart / BS_BLOCK_SIZE) % DOUBLE_BUFFER;
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID6 + slot);
}

__aicore__ inline void MhcPreSinkhornBackwardCubeCompute::WaitC0UbBuffersReleased()
{
    if (directC0_) {
        CrossCoreWaitFlag(C0_UB_BUFFER_FREE_FLAG_BASE);
        CrossCoreWaitFlag(C0_UB_BUFFER_FREE_FLAG_BASE + 1);
    }
}

__aicore__ inline void MhcPreSinkhornBackwardCubeCompute::Init(TPipe* pipe,
                                                               const MhcPreSinkhornBackwardArch35TilingData* tiling,
                                                               bool directC0)
{
    directC0_ = directC0;
    maxBufferDepth_ = tiling->maxBufferDepth;
    l1SingleBufferElems_ = tiling->l1SingleBufferElems;
    l0ABSingleBufferElems_ = tiling->l0ABSingleBufferElems;
    l0CSingleBufferElems_ = tiling->l0CSingleBufferElems;
    pipe->InitBuffer(l1Buf_, tiling->l1UsedBytes);
    InitOperandCache(pipe, tiling);
    aL1Local_ = l1Buf_.Get<float>();
    bL1Local_ = aL1Local_[tiling->l1BOffsetElems];
    aL0Local_ = LocalTensor<float>(TPosition::A2, 0, tiling->l0ABUsedBytes / sizeof(float));
    bL0Local_ = LocalTensor<float>(TPosition::B2, 0, tiling->l0ABUsedBytes / sizeof(float));
    cL0Local_ = LocalTensor<float>(TPosition::CO1, 0, tiling->l0CUsedBytes / sizeof(float));

    for (uint32_t bufferIndex = 0; bufferIndex < maxBufferDepth_; ++bufferIndex) {
        // 初始化 L1、L0A/B 和 L0C 缓冲区的空闲事件。
        SetFlag<HardEvent::MTE1_MTE2>(bufferIndex);
        SetFlag<HardEvent::M_MTE1>(CUBE_L0_EVENT_BASE + bufferIndex);
        SetFlag<HardEvent::FIX_M>(bufferIndex);
    }
}

__aicore__ inline void MhcPreSinkhornBackwardCubeCompute::InitOperandCache(
    TPipe* pipe, const MhcPreSinkhornBackwardArch35TilingData* tiling)
{
    const auto& plan = tiling->cubeCachePlan;
    hMixGradMode_ = plan.hMixGradMode;
    cachePhi_ = plan.cachePhi != 0U;
    hMixGradColumns_ = tiling->fusionSize;
    if (plan.bufferElements == 0U) {
        return;
    }
    hMixGradValidRows_ = plan.validRows;
    hMixGradRows_ = plan.rows;
    hMixGradCacheElements_ = plan.slotElements;
    const uint32_t hMixGradElements = plan.slotElements * plan.slotCount;
    pipe->InitBuffer(operandCacheBuf_, plan.bufferElements * sizeof(float));
    const auto cacheBase = operandCacheBuf_.Get<float>();
    // 按 Host 计划绑定缓存视图。
    if (hMixGradElements > 0U) {
        hMixGradCache_ = cacheBase;
        ClearL1Buffer(hMixGradCache_, hMixGradElements);
    }
    if (cachePhi_) {
        phiCache_ = cacheBase[hMixGradElements];
    }
    if (hMixGradMode_ == HMixGradCacheMode::TILED) {
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID6);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID7);
    }
}

__aicore__ inline void MhcPreSinkhornBackwardCubeCompute::PrepareHMixGradTile(const GlobalTensor<float>& hMixGrad,
                                                                              uint32_t rows, uint64_t rowStart,
                                                                              bool loadNewTile)
{
    const uint32_t slot = (rowStart / BS_BLOCK_SIZE) % DOUBLE_BUFFER;
    const auto cache = GetHMixGradTile(rowStart);
    if (!loadNewTile) {
        return;
    }
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID6 + slot);
    if (rows < hMixGradRows_) {
        // 清除复用槽中的尾部残留数据。
        ClearL1Buffer(cache, hMixGradCacheElements_);
    }
    Nd2NzParams p;
    p.ndNum = 1;
    p.nValue = rows;
    p.dValue = hMixGradColumns_;
    p.srcNdMatrixStride = 1;
    p.srcDValue = hMixGradColumns_;
    p.dstNzC0Stride = hMixGradRows_;
    p.dstNzNStride = 1;
    p.dstNzMatrixStride = 1;
    DataCopy(cache, hMixGrad, p);
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID6 + slot);
    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID6 + slot);
}

__aicore__ inline void MhcPreSinkhornBackwardCubeCompute::PreloadHMixGrad(const GlobalTensor<float>& hMixGrad)
{
    if (hMixGradMode_ != HMixGradCacheMode::FULL) {
        return;
    }
    Nd2NzParams p;
    p.ndNum = 1;
    p.nValue = hMixGradValidRows_;
    p.dValue = hMixGradColumns_;
    p.srcNdMatrixStride = 1;
    p.srcDValue = hMixGradColumns_;
    p.dstNzC0Stride = hMixGradRows_;
    p.dstNzNStride = 1;
    p.dstNzMatrixStride = 1;
    DataCopy(hMixGradCache_, hMixGrad, p);
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID6);
    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID6);
}

__aicore__ inline void MhcPreSinkhornBackwardCubeCompute::ProcessC0(
    const MMConfig& config, const GlobalTensor<float>& a, const GlobalTensor<float>& b, const GlobalTensor<float>& c,
    uint32_t m, uint32_t n, uint32_t k, uint32_t bGmStride, uint64_t rowStart, uint32_t localColumn)
{
    // A/B 分别选择缓存或普通加载。
    LocalTensor<float> cachedB;
    if (cachePhi_) {
        if (rowStart == 0U && localColumn == 0U) {
            WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID5);
        }
        cachedB = phiCache_[localColumn * BasicCubeAlign(k, BLOCK_CUBE)];
    } else if (firstC0BReady_) {
        firstC0BReady_ = false;
    } else {
        PreloadB1(config, b, n, k, bGmStride);
    }
    for (uint32_t offsetM = 0; offsetM < m; offsetM += config.baseM) {
        uint32_t remainM = m - offsetM;
        uint32_t currentM = remainM < config.baseM ? remainM : config.baseM;
        for (uint32_t offsetN = 0; offsetN < n; offsetN += config.baseN) {
            uint32_t remainN = n - offsetN;
            uint32_t currentN = remainN < config.baseN ? remainN : config.baseN;
            LocalTensor<float> cachedA;
            if (hMixGradMode_ != HMixGradCacheMode::NONE) {
                cachedA = GetHMixGradTile(rowStart)[offsetM * AuxGetC0Size<float>()];
            }
            ComputeTileC0(config, a[offsetM * k], c[offsetM * ND_BLOCK_SIZE + offsetN], currentM, currentN, k, k,
                          ND_BLOCK_SIZE, cachedA, cachedB);
        }
        CrossCoreSetFlag<CROSS_CORE_FLAG_INDEX, PIPE_FIX>(CROSS_CORE_WAIT_FLAG_C0);
    }
}

__aicore__ inline void MhcPreSinkhornBackwardCubeCompute::ProcessC1(
    const MMConfig& config, const GlobalTensor<float>& a, const GlobalTensor<float>& b, const GlobalTensor<float>& c,
    uint32_t m, uint32_t n, uint32_t k, uint32_t aGmStride, uint32_t bGmStride, uint32_t cGmStride, bool atomicAdd,
    uint64_t rowStart)
{
    for (uint32_t offsetM = 0; offsetM < m; offsetM += config.baseM) {
        uint32_t remainM = m - offsetM;
        uint32_t currentM = remainM < config.baseM ? remainM : config.baseM;
        for (uint32_t offsetN = 0; offsetN < n; offsetN += config.baseN) {
            uint32_t remainN = n - offsetN;
            uint32_t currentN = remainN < config.baseN ? remainN : config.baseN;
            // C1 的 M 维是 H 通道；缓存保持 [BS,H] 布局，转置在 L1 -> L0 时完成。
            LocalTensor<float> cachedA;
            if (hMixGradMode_ != HMixGradCacheMode::NONE) {
                cachedA = GetHMixGradTile(rowStart)[offsetM * hMixGradRows_];
            }
            ComputeTileC1(config, a[offsetM], b[offsetN], c[offsetM * cGmStride + offsetN], currentM, currentN, k,
                          aGmStride, bGmStride, cGmStride, atomicAdd, cachedA);
        }
    }
}

__aicore__ inline void MhcPreSinkhornBackwardCubeCompute::End()
{
    if (hMixGradMode_ == HMixGradCacheMode::TILED) {
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID6);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID7);
    }
    for (uint32_t bufferIndex = 0; bufferIndex < maxBufferDepth_; ++bufferIndex) {
        WaitFlag<HardEvent::MTE1_MTE2>(bufferIndex);
        WaitFlag<HardEvent::M_MTE1>(CUBE_L0_EVENT_BASE + bufferIndex);
        WaitFlag<HardEvent::FIX_M>(bufferIndex);
    }
}

__aicore__ inline void MhcPreSinkhornBackwardCubeCompute::ComputeTileC0(
    const MMConfig& config, const GlobalTensor<float>& a, const GlobalTensor<float>& c, uint32_t m, uint32_t n,
    uint32_t k, uint32_t aGmStride, uint32_t cGmStride, const LocalTensor<float>& cachedA,
    const LocalTensor<float>& cachedB)
{
    uint32_t cDepth = config.dbL0C;
    uint32_t cBufferIndex = cl0PingPongID_ % cDepth;
    LocalTensor<float> currentC = cL0Local_[cBufferIndex * l0CSingleBufferElems_];
    WaitFlag<HardEvent::FIX_M>(cBufferIndex);

    ProcessKRangeC0(config, currentC, a, m, n, k, aGmStride, cachedA, cachedB);
    FinishC0Tile(c, currentC, m, n, cGmStride, cBufferIndex);
}

__aicore__ inline void MhcPreSinkhornBackwardCubeCompute::FinishC0Tile(const GlobalTensor<float>& c,
                                                                       const LocalTensor<float>& currentC, uint32_t m,
                                                                       uint32_t n, uint32_t cGmStride,
                                                                       uint32_t cBufferIndex)
{
    if (directC0_) {
        SetFlag<HardEvent::M_FIX>(cBufferIndex);
        WaitFlag<HardEvent::M_FIX>(cBufferIndex);
        uint32_t slot = c0UbBufferIndex_++ % DOUBLE_BUFFER;
        CrossCoreWaitFlag(C0_UB_BUFFER_FREE_FLAG_BASE + slot);
        // V0 结束后复用 queue0 的两个 32 KiB 缓冲区，向配对 AIV 直传 128x64 的 C0 结果。
        LocalTensor<float> c0UbBuffer(TPosition::VECIN, C0_UB_BUFFER_OFFSET + slot * INOUT_QUEUE_SIZE,
                                      INOUT_QUEUE_SIZE / sizeof(float));
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> params;
        params.nSize = ND_BLOCK_SIZE;
        params.mSize = m;
        params.dstStride = ND_BLOCK_SIZE / 2;
        params.srcStride = BasicCubeAlign(m, BLOCK_CUBE);
        params.quantPre = QuantMode_t::NoQuant;
        params.dualDstCtl = 2;
        params.unitFlag = 0;
        params.params.ndNum = 1;
        params.params.srcNdStride = 1;
        params.params.dstNdStride = 1;
        Fixpipe<float, float, C0_UB_CONFIG>(c0UbBuffer, currentC, params);
        SetFlag<HardEvent::FIX_M>(cBufferIndex);
        ++cl0PingPongID_;
    } else {
        FinishTile(c, currentC, m, n, cGmStride, false, cBufferIndex);
    }
}

__aicore__ inline void MhcPreSinkhornBackwardCubeCompute::ComputeTileC1(
    const MMConfig& config, const GlobalTensor<float>& a, const GlobalTensor<float>& b, const GlobalTensor<float>& c,
    uint32_t m, uint32_t n, uint32_t k, uint32_t aGmStride, uint32_t bGmStride, uint32_t cGmStride, bool atomicAdd,
    const LocalTensor<float>& cachedA)
{
    uint32_t cDepth = config.dbL0C;
    uint32_t cBufferIndex = cl0PingPongID_ % cDepth;
    LocalTensor<float> currentC = cL0Local_[cBufferIndex * l0CSingleBufferElems_];
    WaitFlag<HardEvent::FIX_M>(cBufferIndex);

    ProcessKRangeC1(config, currentC, a, b, m, n, k, aGmStride, bGmStride, cachedA);
    FinishTile(c, currentC, m, n, cGmStride, atomicAdd, cBufferIndex);
}

__aicore__ inline void MhcPreSinkhornBackwardCubeCompute::FinishTile(const GlobalTensor<float>& c,
                                                                     const LocalTensor<float>& currentC, uint32_t m,
                                                                     uint32_t n, uint32_t cGmStride, bool atomicAdd,
                                                                     uint32_t cBufferIndex)
{
    SetFlag<HardEvent::M_FIX>(cBufferIndex);
    WaitFlag<HardEvent::M_FIX>(cBufferIndex);
    if (atomicAdd) {
        // 下一次原子写入 gradPhi 前，等待上一次 Fixpipe 完成。
        PipeBarrier<PIPE_FIX>();
        SetAtomicAdd<float>();
    }
    CopyOut(c, currentC, m, n, cGmStride);
    if (atomicAdd) {
        SetAtomicNone();
    }
    SetFlag<HardEvent::FIX_M>(cBufferIndex);
    ++cl0PingPongID_;
}

__aicore__ inline void MhcPreSinkhornBackwardCubeCompute::ProcessKRangeC0(
    const MMConfig& config, const LocalTensor<float>& currentC, const GlobalTensor<float>& a, uint32_t m, uint32_t n,
    uint32_t k, uint32_t aGmStride, const LocalTensor<float>& cachedA, const LocalTensor<float>& cachedB)
{
    uint32_t l1K = config.l1K;
    uint32_t l1Depth = config.depthA1;
    const bool useOrdinaryL1 = hMixGradMode_ == HMixGradCacheMode::NONE || !cachePhi_;
    for (uint32_t offsetKL1 = 0; offsetKL1 < k; offsetKL1 += l1K) {
        uint32_t remainKL1 = k - offsetKL1;
        uint32_t currentKL1 = remainKL1 < l1K ? remainKL1 : l1K;
        uint32_t l1BufferIndex = (offsetKL1 / l1K) % l1Depth;
        LocalTensor<float> currentA1 = aL1Local_[l1BufferIndex * l1SingleBufferElems_];
        LocalTensor<float> currentB1 = bL1Local_[l1BufferIndex * l1SingleBufferElems_];

        if (useOrdinaryL1) {
            WaitFlag<HardEvent::MTE1_MTE2>(l1BufferIndex);
        }
        if (hMixGradMode_ != HMixGradCacheMode::NONE) {
            currentA1 = cachedA[offsetKL1 * hMixGradRows_];
        } else {
            CopyInA1(currentA1, a[offsetKL1], m, currentKL1, aGmStride);
        }
        if (cachePhi_) {
            currentB1 = cachedB[offsetKL1 * AuxGetC0Size<float>()];
        }
        if (useOrdinaryL1) {
            SetFlag<HardEvent::MTE2_MTE1>(l1BufferIndex);
            WaitFlag<HardEvent::MTE2_MTE1>(l1BufferIndex);
        }

        const uint32_t aStride =
            hMixGradMode_ != HMixGradCacheMode::NONE ? hMixGradRows_ : BasicCubeAlign(m, BLOCK_CUBE);
        // Phi 常驻区保存完整 H；普通 B1 每槽只保存 currentKL1，物理行跨度不同。
        const uint32_t bStride = cachePhi_ ? k : currentKL1;
        ProcessL0KRange(config, currentC, currentA1, currentB1, m, n, currentKL1, aStride, bStride, false,
                        offsetKL1 == 0U);

        // MTE1 完成最后一次读取后释放对应 L1 缓冲区。
        if (useOrdinaryL1) {
            SetFlag<HardEvent::MTE1_MTE2>(l1BufferIndex);
            ++l1PingPongID_;
        }
    }
}

__aicore__ inline void MhcPreSinkhornBackwardCubeCompute::ProcessKRangeC1(
    const MMConfig& config, const LocalTensor<float>& currentC, const GlobalTensor<float>& a,
    const GlobalTensor<float>& b, uint32_t m, uint32_t n, uint32_t k, uint32_t aGmStride, uint32_t bGmStride,
    const LocalTensor<float>& cachedA)
{
    uint32_t l1K = config.l1K;
    uint32_t l1Depth = config.depthA1;
    for (uint32_t offsetKL1 = 0; offsetKL1 < k; offsetKL1 += l1K) {
        uint32_t remainKL1 = k - offsetKL1;
        uint32_t currentKL1 = remainKL1 < l1K ? remainKL1 : l1K;
        uint32_t l1BufferIndex = l1PingPongID_ % l1Depth;
        LocalTensor<float> currentA1 = aL1Local_[l1BufferIndex * l1SingleBufferElems_];
        LocalTensor<float> currentB1 = bL1Local_[l1BufferIndex * l1SingleBufferElems_];

        WaitFlag<HardEvent::MTE1_MTE2>(l1BufferIndex);
        if (hMixGradMode_ != HMixGradCacheMode::NONE) {
            currentA1 = cachedA[offsetKL1 * AuxGetC0Size<float>()];
        } else {
            CopyInA1Trans(currentA1, a[offsetKL1 * aGmStride], m, currentKL1, aGmStride);
        }
        CopyInB1(currentB1, b[offsetKL1 * bGmStride], n, currentKL1, bGmStride);
        SetFlag<HardEvent::MTE2_MTE1>(l1BufferIndex);
        WaitFlag<HardEvent::MTE2_MTE1>(l1BufferIndex);

        const uint32_t aStride = hMixGradMode_ != HMixGradCacheMode::NONE ? hMixGradRows_ : currentKL1;
        ProcessL0KRange(config, currentC, currentA1, currentB1, m, n, currentKL1, aStride, currentKL1, true,
                        offsetKL1 == 0U);

        SetFlag<HardEvent::MTE1_MTE2>(l1BufferIndex);
        ++l1PingPongID_;
    }
}

__aicore__ inline void MhcPreSinkhornBackwardCubeCompute::ProcessL0KRange(
    const MMConfig& config, const LocalTensor<float>& currentC, const LocalTensor<float>& currentA1,
    const LocalTensor<float>& currentB1, uint32_t m, uint32_t n, uint32_t currentKL1, uint32_t aL1Stride,
    uint32_t bL1Stride, bool transA, bool initC)
{
    uint32_t baseK = config.baseK;
    uint32_t l0Depth = config.dbL0A;
    for (uint32_t offsetKL0 = 0; offsetKL0 < currentKL1; offsetKL0 += baseK) {
        uint32_t remainKL0 = currentKL1 - offsetKL0;
        uint32_t currentKL0 = remainKL0 < baseK ? remainKL0 : baseK;
        uint32_t l0BufferIndex = l0PingPongID_ % l0Depth;
        uint32_t l0Offset = l0BufferIndex * l0ABSingleBufferElems_;
        LocalTensor<float> currentA2 = aL0Local_[l0Offset];
        LocalTensor<float> currentB2 = bL0Local_[l0Offset];

        WaitFlag<HardEvent::M_MTE1>(CUBE_L0_EVENT_BASE + l0BufferIndex);
        if (transA) {
            CopyInA2Trans(currentA2, currentA1[offsetKL0 * AuxGetC0Size<float>()], m, currentKL0, aL1Stride);
        } else {
            CopyInA2(currentA2, currentA1[offsetKL0 * aL1Stride], m, currentKL0, aL1Stride);
        }
        CopyInB2(currentB2, currentB1[offsetKL0 * AuxGetC0Size<float>()], n, currentKL0, bL1Stride);
        SetFlag<HardEvent::MTE1_M>(l0BufferIndex);
        WaitFlag<HardEvent::MTE1_M>(l0BufferIndex);

        Compute(currentC, currentA2, currentB2, m, n, currentKL0, initC && offsetKL0 == 0U);
        SetFlag<HardEvent::M_MTE1>(CUBE_L0_EVENT_BASE + l0BufferIndex);
        ++l0PingPongID_;
    }
}

__aicore__ inline void MhcPreSinkhornBackwardCubeCompute::PreloadB1(const MMConfig& config,
                                                                    const GlobalTensor<float>& b, uint32_t n,
                                                                    uint32_t k, uint32_t bGmStride)
{
    for (uint32_t offsetKL1 = 0; offsetKL1 < k; offsetKL1 += config.l1K) {
        uint32_t remainKL1 = k - offsetKL1;
        uint32_t currentKL1 = remainKL1 < config.l1K ? remainKL1 : config.l1K;
        uint32_t l1BufferIndex = (offsetKL1 / config.l1K) % config.depthB1;
        LocalTensor<float> currentB1 = bL1Local_[l1BufferIndex * l1SingleBufferElems_];

        WaitFlag<HardEvent::MTE1_MTE2>(l1BufferIndex);
        CopyInB1(currentB1, b[offsetKL1 * bGmStride], n, currentKL1, bGmStride);
        SetFlag<HardEvent::MTE2_MTE1>(l1BufferIndex);
        WaitFlag<HardEvent::MTE2_MTE1>(l1BufferIndex);
        SetFlag<HardEvent::MTE1_MTE2>(l1BufferIndex);
    }
}

__aicore__ inline void MhcPreSinkhornBackwardCubeCompute::CopyInA1(const LocalTensor<float>& dst,
                                                                   const GlobalTensor<float>& src, uint32_t m,
                                                                   uint32_t k, uint32_t gmStride)
{
    uint32_t alignedM = BasicCubeAlign(m, BLOCK_CUBE);
    uint32_t alignedK = BasicCubeAlign(k, AuxGetC0Size<float>());
    if (alignedM != m || alignedK != k) {
        ClearL1Buffer(dst, alignedM * alignedK);
    }
    Nd2NzParams p;
    p.ndNum = 1;
    p.nValue = m;
    p.dValue = k;
    p.srcNdMatrixStride = 1;
    p.srcDValue = gmStride;
    p.dstNzC0Stride = BasicCubeAlign(m, BLOCK_CUBE);
    p.dstNzNStride = 1;
    p.dstNzMatrixStride = 1;
    DataCopy(dst, src, p);
}

__aicore__ inline void MhcPreSinkhornBackwardCubeCompute::CopyInA1Trans(const LocalTensor<float>& dst,
                                                                        const GlobalTensor<float>& src, uint32_t m,
                                                                        uint32_t k, uint32_t gmStride)
{
    uint32_t alignedM = BasicCubeAlign(m, BLOCK_CUBE);
    uint32_t alignedK = BasicCubeAlign(k, BLOCK_CUBE);
    if (alignedM != m || alignedK != k) {
        ClearL1Buffer(dst, alignedM * alignedK);
    }
    Nd2NzParams p;
    p.ndNum = 1;
    p.nValue = k;
    p.dValue = m;
    p.srcNdMatrixStride = 1;
    p.srcDValue = gmStride;
    p.dstNzC0Stride = BasicCubeAlign(k, BLOCK_CUBE);
    p.dstNzNStride = 1;
    p.dstNzMatrixStride = 1;
    DataCopy(dst, src, p);
}

__aicore__ inline void MhcPreSinkhornBackwardCubeCompute::CopyInB1(const LocalTensor<float>& dst,
                                                                   const GlobalTensor<float>& src, uint32_t n,
                                                                   uint32_t k, uint32_t gmStride)
{
    uint32_t alignedN = BasicCubeAlign(n, BLOCK_CUBE);
    uint32_t alignedK = BasicCubeAlign(k, BLOCK_CUBE);
    if (alignedN != n || alignedK != k) {
        ClearL1Buffer(dst, alignedN * alignedK);
    }
    Nd2NzParams p;
    p.ndNum = 1;
    p.nValue = k;
    p.dValue = n;
    p.srcNdMatrixStride = 1;
    p.srcDValue = gmStride;
    p.dstNzC0Stride = BasicCubeAlign(k, BLOCK_CUBE);
    p.dstNzNStride = 1;
    p.dstNzMatrixStride = 1;
    DataCopy(dst, src, p);
}

__aicore__ inline void MhcPreSinkhornBackwardCubeCompute::CopyInA2(const LocalTensor<float>& dst,
                                                                   const LocalTensor<float>& src, uint32_t m,
                                                                   uint32_t k, uint32_t rowStride)
{
    LoadData2DParamsV2 p;
    p.mStartPosition = 0;
    p.kStartPosition = 0;
    p.mStep = BasicCubeCeilDiv(m, BLOCK_CUBE);
    p.kStep = BasicCubeCeilDiv(k, AuxGetC0Size<float>());
    p.srcStride = BasicCubeCeilDiv(rowStride, BLOCK_CUBE);
    p.dstStride = p.mStep;
    p.ifTranspose = false;
    LoadData<float>(dst, src, p);
}

__aicore__ inline void MhcPreSinkhornBackwardCubeCompute::CopyInA2Trans(const LocalTensor<float>& dst,
                                                                        const LocalTensor<float>& src, uint32_t m,
                                                                        uint32_t k, uint32_t kL1)
{
    LoadData2DParamsV2 p;
    p.mStartPosition = 0;
    p.kStartPosition = 0;
    p.mStep = BasicCubeCeilDiv(k, BLOCK_CUBE);
    p.kStep = BasicCubeCeilDiv(m, BLOCK_CUBE) * NUMBER_TWO;
    p.dstStride = p.kStep >> 1U;
    p.srcStride = BasicCubeCeilDiv(kL1, BLOCK_CUBE);
    p.ifTranspose = true;
    LoadData<float>(dst, src, p);
}

__aicore__ inline void MhcPreSinkhornBackwardCubeCompute::CopyInB2(const LocalTensor<float>& dst,
                                                                   const LocalTensor<float>& src, uint32_t n,
                                                                   uint32_t k, uint32_t kL1)
{
    LoadData2DParamsV2 p;
    p.mStartPosition = 0;
    p.kStartPosition = 0;
    p.mStep = BasicCubeCeilDiv(k, BLOCK_CUBE);
    p.kStep = BasicCubeCeilDiv(n, BLOCK_CUBE) * NUMBER_TWO;
    p.dstStride = p.kStep >> 1U;
    p.srcStride = BasicCubeCeilDiv(kL1, BLOCK_CUBE);
    p.ifTranspose = true;
    LoadData<float>(dst, src, p);
}

__aicore__ inline void MhcPreSinkhornBackwardCubeCompute::Compute(const LocalTensor<float>& c,
                                                                  const LocalTensor<float>& a,
                                                                  const LocalTensor<float>& b, uint32_t m, uint32_t n,
                                                                  uint32_t k, bool initC)
{
    MmadParams p;
    p.m = m;
    p.n = n;
    p.k = k;
    p.disableGemv = true;
    p.cmatrixInitVal = initC;
    p.cmatrixSource = false;
    // unitFlag=0，使用 M_FIX/FIX_M 事件管理 L0C。
    p.unitFlag = 0;
    Mmad(c, a, b, p);
}

__aicore__ inline void MhcPreSinkhornBackwardCubeCompute::CopyOut(const GlobalTensor<float>& dst,
                                                                  const LocalTensor<float>& src, uint32_t m, uint32_t n,
                                                                  uint32_t dstStride)
{
    DataCopyCO12DstParams p;
    p.nSize = n;
    p.mSize = m;
    p.dstStride = dstStride;
    p.srcStride = BasicCubeAlign(m, BLOCK_CUBE);
    p.quantPre = QuantMode_t::NoQuant;
    p.nz2ndEn = true;
    p.unitFlag = 0;
    SetFixpipeNz2ndFlag(1, 1, 1);
    DataCopy(dst, src, p);
}
} // namespace MhcPreSinkhornBackwardNs
#endif
