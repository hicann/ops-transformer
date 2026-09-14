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
#ifndef MHC_PRE_BACKWARD_CUBE_COMPUTE_H
#define MHC_PRE_BACKWARD_CUBE_COMPUTE_H
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "mhc_pre_backward_utils.h"

namespace MhcPreBackward {
using namespace AscendC;
using namespace MhcPreBackwardUtils;

constexpr uint32_t CUBE_L0_EVENT_BASE = 3U;
constexpr uint32_t CUBE_BLOCK_BYTES = 32U;

__aicore__ inline uint64_t BasicCubeCeilDiv(uint64_t x, uint64_t n)
{
    return (x + n - 1U) / n;
}

__aicore__ inline uint64_t BasicCubeAlign(uint64_t x, uint64_t n)
{
    return BasicCubeCeilDiv(x, n) * n;
}

class MhcPreBackwardCubeCompute {
public:
    __aicore__ inline void Init(TPipe *pipe, const MhcPreBackwardTilingData *tiling);

    __aicore__ inline void ProcessC0(const MMConfig &config, const GlobalTensor<float> &a, const GlobalTensor<float> &b,
                                     const GlobalTensor<float> &c, uint32_t m, uint32_t n, uint32_t k,
                                     uint32_t bGmStride);

    __aicore__ inline void ProcessC1(const MMConfig &config, const GlobalTensor<float> &a, const GlobalTensor<float> &b,
                                     const GlobalTensor<float> &c, uint32_t m, uint32_t n, uint32_t k,
                                     uint32_t aGmStride, uint32_t bGmStride, uint32_t cGmStride, bool atomicAdd);

    __aicore__ inline void End();

private:
    __aicore__ inline void ComputeTileC0(const MMConfig &config, const GlobalTensor<float> &a,
                                         const GlobalTensor<float> &c, uint32_t m, uint32_t n, uint32_t k,
                                         uint32_t aGmStride, uint32_t cGmStride);

    __aicore__ inline void ComputeTileC1(const MMConfig &config, const GlobalTensor<float> &a,
                                         const GlobalTensor<float> &b, const GlobalTensor<float> &c, uint32_t m,
                                         uint32_t n, uint32_t k, uint32_t aGmStride, uint32_t bGmStride,
                                         uint32_t cGmStride, bool atomicAdd);

    __aicore__ inline void ProcessKRangeC0(const MMConfig &config, const LocalTensor<float> &currentC,
                                           const GlobalTensor<float> &a, uint32_t m, uint32_t n, uint32_t k,
                                           uint32_t aGmStride);

    __aicore__ inline void ProcessKRangeC1(const MMConfig &config, const LocalTensor<float> &currentC,
                                           const GlobalTensor<float> &a, const GlobalTensor<float> &b, uint32_t m,
                                           uint32_t n, uint32_t k, uint32_t aGmStride, uint32_t bGmStride);

    __aicore__ inline void ProcessL0KRange(const MMConfig &config, const LocalTensor<float> &currentC,
                                           const LocalTensor<float> &currentA1, const LocalTensor<float> &currentB1,
                                           uint32_t m, uint32_t n, uint32_t currentKL1, bool transA, bool initC);

    __aicore__ inline void FinishTile(const GlobalTensor<float> &c, const LocalTensor<float> &currentC, uint32_t m,
                                      uint32_t n, uint32_t cGmStride, bool atomicAdd, uint32_t cBufferIndex);

    __aicore__ inline void PreloadB1(const MMConfig &config, const GlobalTensor<float> &b, uint32_t n, uint32_t k,
                                     uint32_t bGmStride);

    __aicore__ inline void CopyInA1(const LocalTensor<float> &dst, const GlobalTensor<float> &src, uint32_t m,
                                    uint32_t k, uint32_t gmStride);

    __aicore__ inline void CopyInA1Trans(const LocalTensor<float> &dst, const GlobalTensor<float> &src, uint32_t m,
                                         uint32_t k, uint32_t gmStride);

    __aicore__ inline void CopyInB1(const LocalTensor<float> &dst, const GlobalTensor<float> &src, uint32_t n,
                                    uint32_t k, uint32_t gmStride);

    __aicore__ inline void CopyInA2(const LocalTensor<float> &dst, const LocalTensor<float> &src, uint32_t m,
                                    uint32_t k);

    __aicore__ inline void CopyInA2Trans(const LocalTensor<float> &dst, const LocalTensor<float> &src, uint32_t m,
                                         uint32_t k, uint32_t kL1);

    __aicore__ inline void CopyInB2(const LocalTensor<float> &dst, const LocalTensor<float> &src, uint32_t n,
                                    uint32_t k, uint32_t kL1);

    __aicore__ inline void Compute(const LocalTensor<float> &c, const LocalTensor<float> &a,
                                   const LocalTensor<float> &b, uint32_t m, uint32_t n, uint32_t k, bool initC);

    __aicore__ inline void CopyOut(const GlobalTensor<float> &dst, const LocalTensor<float> &src, uint32_t m,
                                   uint32_t n, uint32_t dstStride);

    TBuf<TPosition::A1> l1Buf_;
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

__aicore__ inline void MhcPreBackwardCubeCompute::Init(TPipe *pipe, const MhcPreBackwardTilingData *tiling)
{
    maxBufferDepth_ = tiling->maxBufferDepth;
    l1SingleBufferElems_ = tiling->l1SingleBufferElems;
    l0ABSingleBufferElems_ = tiling->l0ABSingleBufferElems;
    l0CSingleBufferElems_ = tiling->l0CSingleBufferElems;
    pipe->InitBuffer(l1Buf_, tiling->l1UsedBytes);
    aL1Local_ = l1Buf_.Get<float>();
    bL1Local_ = aL1Local_[tiling->l1BOffsetElems];
    aL0Local_ = LocalTensor<float>(TPosition::A2, 0, tiling->l0ABUsedBytes / sizeof(float));
    bL0Local_ = LocalTensor<float>(TPosition::B2, 0, tiling->l0ABUsedBytes / sizeof(float));
    cL0Local_ = LocalTensor<float>(TPosition::CO1, 0, tiling->l0CUsedBytes / sizeof(float));

    for (uint32_t bufferIndex = 0; bufferIndex < maxBufferDepth_; ++bufferIndex) {
        // 三组初始令牌分别表示L1、L0A/B和L0C对应缓冲区可写。
        SetFlag<HardEvent::MTE1_MTE2>(bufferIndex);
        SetFlag<HardEvent::M_MTE1>(CUBE_L0_EVENT_BASE + bufferIndex);
        SetFlag<HardEvent::FIX_M>(bufferIndex);
    }
}

__aicore__ inline void MhcPreBackwardCubeCompute::ProcessC0(const MMConfig &config, const GlobalTensor<float> &a,
                                                            const GlobalTensor<float> &b, const GlobalTensor<float> &c,
                                                            uint32_t m, uint32_t n, uint32_t k, uint32_t bGmStride)
{
    // C0为[M=BS块,K=fusionSize]×[K=fusionSize,N=ND块]；头数4/6/8对应K=24/48/80。
    // C0按tiling给出的B1驻留策略预载，C1仍使用自身的分段搬运路径。
    PreloadB1(config, b, n, k, bGmStride);
    for (uint32_t offsetM = 0; offsetM < m; offsetM += config.baseM) {
        uint32_t remainM = m - offsetM;
        uint32_t currentM = remainM < config.baseM ? remainM : config.baseM;
        for (uint32_t offsetN = 0; offsetN < n; offsetN += config.baseN) {
            uint32_t remainN = n - offsetN;
            uint32_t currentN = remainN < config.baseN ? remainN : config.baseN;
            ComputeTileC0(config, a[offsetM * k], c[offsetM * ND_BLOCK_SIZE + offsetN], currentM, currentN, k, k,
                          ND_BLOCK_SIZE);
        }
        CrossCoreSetFlag<CROSS_CORE_FLAG_INDEX, PIPE_FIX>(CROSS_CORE_WAIT_FLAG_C0);
    }
}

__aicore__ inline void MhcPreBackwardCubeCompute::ProcessC1(const MMConfig &config, const GlobalTensor<float> &a,
                                                            const GlobalTensor<float> &b, const GlobalTensor<float> &c,
                                                            uint32_t m, uint32_t n, uint32_t k, uint32_t aGmStride,
                                                            uint32_t bGmStride, uint32_t cGmStride, bool atomicAdd)
{
    for (uint32_t offsetM = 0; offsetM < m; offsetM += config.baseM) {
        uint32_t remainM = m - offsetM;
        uint32_t currentM = remainM < config.baseM ? remainM : config.baseM;
        for (uint32_t offsetN = 0; offsetN < n; offsetN += config.baseN) {
            uint32_t remainN = n - offsetN;
            uint32_t currentN = remainN < config.baseN ? remainN : config.baseN;
            ComputeTileC1(config, a[offsetM], b[offsetN], c[offsetM * cGmStride + offsetN], currentM, currentN, k,
                          aGmStride, bGmStride, cGmStride, atomicAdd);
        }
    }
}

__aicore__ inline void MhcPreBackwardCubeCompute::End()
{
    for (uint32_t bufferIndex = 0; bufferIndex < maxBufferDepth_; ++bufferIndex) {
        WaitFlag<HardEvent::MTE1_MTE2>(bufferIndex);
        WaitFlag<HardEvent::M_MTE1>(CUBE_L0_EVENT_BASE + bufferIndex);
        WaitFlag<HardEvent::FIX_M>(bufferIndex);
    }
}

__aicore__ inline void MhcPreBackwardCubeCompute::ComputeTileC0(const MMConfig &config, const GlobalTensor<float> &a,
                                                                const GlobalTensor<float> &c, uint32_t m, uint32_t n,
                                                                uint32_t k, uint32_t aGmStride, uint32_t cGmStride)
{
    uint32_t cDepth = config.dbL0C;
    uint32_t cBufferIndex = cl0PingPongID_ % cDepth;
    LocalTensor<float> currentC = cL0Local_[cBufferIndex * l0CSingleBufferElems_];
    WaitFlag<HardEvent::FIX_M>(cBufferIndex);

    ProcessKRangeC0(config, currentC, a, m, n, k, aGmStride);
    FinishTile(c, currentC, m, n, cGmStride, false, cBufferIndex);
}

__aicore__ inline void MhcPreBackwardCubeCompute::ComputeTileC1(const MMConfig &config, const GlobalTensor<float> &a,
                                                                const GlobalTensor<float> &b,
                                                                const GlobalTensor<float> &c, uint32_t m, uint32_t n,
                                                                uint32_t k, uint32_t aGmStride, uint32_t bGmStride,
                                                                uint32_t cGmStride, bool atomicAdd)
{
    uint32_t cDepth = config.dbL0C;
    uint32_t cBufferIndex = cl0PingPongID_ % cDepth;
    LocalTensor<float> currentC = cL0Local_[cBufferIndex * l0CSingleBufferElems_];
    WaitFlag<HardEvent::FIX_M>(cBufferIndex);

    ProcessKRangeC1(config, currentC, a, b, m, n, k, aGmStride, bGmStride);
    FinishTile(c, currentC, m, n, cGmStride, atomicAdd, cBufferIndex);
}

__aicore__ inline void MhcPreBackwardCubeCompute::FinishTile(const GlobalTensor<float> &c,
                                                             const LocalTensor<float> &currentC, uint32_t m, uint32_t n,
                                                             uint32_t cGmStride, bool atomicAdd, uint32_t cBufferIndex)
{
    SetFlag<HardEvent::M_FIX>(cBufferIndex);
    WaitFlag<HardEvent::M_FIX>(cBufferIndex);
    if (atomicAdd) {
        // 后续C1分段写同一gradPhi前仅等待上一条Fixpipe完成，固定AtomicAdd顺序；
        // 等待发生在当前MMAD结束后，继续保留L0C双槽和MMAD/Fixpipe重叠。
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

__aicore__ inline void MhcPreBackwardCubeCompute::ProcessKRangeC0(const MMConfig &config,
                                                                  const LocalTensor<float> &currentC,
                                                                  const GlobalTensor<float> &a, uint32_t m, uint32_t n,
                                                                  uint32_t k, uint32_t aGmStride)
{
    uint32_t l1K = config.l1K;
    uint32_t l1Depth = config.depthA1;
    for (uint32_t offsetKL1 = 0; offsetKL1 < k; offsetKL1 += l1K) {
        uint32_t remainKL1 = k - offsetKL1;
        uint32_t currentKL1 = remainKL1 < l1K ? remainKL1 : l1K;
        uint32_t l1BufferIndex = (offsetKL1 / l1K) % l1Depth;
        LocalTensor<float> currentA1 = aL1Local_[l1BufferIndex * l1SingleBufferElems_];
        LocalTensor<float> currentB1 = bL1Local_[l1BufferIndex * l1SingleBufferElems_];

        WaitFlag<HardEvent::MTE1_MTE2>(l1BufferIndex);
        CopyInA1(currentA1, a[offsetKL1], m, currentKL1, aGmStride);
        SetFlag<HardEvent::MTE2_MTE1>(l1BufferIndex);
        WaitFlag<HardEvent::MTE2_MTE1>(l1BufferIndex);

        ProcessL0KRange(config, currentC, currentA1, currentB1, m, n, currentKL1, false, offsetKL1 == 0U);

        // MTE1完成该L1半区的最后一次读取后，允许MTE2复用此半区。
        SetFlag<HardEvent::MTE1_MTE2>(l1BufferIndex);
        ++l1PingPongID_;
    }
}

__aicore__ inline void MhcPreBackwardCubeCompute::ProcessKRangeC1(const MMConfig &config,
                                                                  const LocalTensor<float> &currentC,
                                                                  const GlobalTensor<float> &a,
                                                                  const GlobalTensor<float> &b, uint32_t m, uint32_t n,
                                                                  uint32_t k, uint32_t aGmStride, uint32_t bGmStride)
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
        CopyInA1Trans(currentA1, a[offsetKL1 * aGmStride], m, currentKL1, aGmStride);
        CopyInB1(currentB1, b[offsetKL1 * bGmStride], n, currentKL1, bGmStride);
        SetFlag<HardEvent::MTE2_MTE1>(l1BufferIndex);
        WaitFlag<HardEvent::MTE2_MTE1>(l1BufferIndex);

        ProcessL0KRange(config, currentC, currentA1, currentB1, m, n, currentKL1, true, offsetKL1 == 0U);

        SetFlag<HardEvent::MTE1_MTE2>(l1BufferIndex);
        ++l1PingPongID_;
    }
}

__aicore__ inline void MhcPreBackwardCubeCompute::ProcessL0KRange(
    const MMConfig &config, const LocalTensor<float> &currentC, const LocalTensor<float> &currentA1,
    const LocalTensor<float> &currentB1, uint32_t m, uint32_t n, uint32_t currentKL1, bool transA, bool initC)
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
            CopyInA2Trans(currentA2, currentA1[offsetKL0 * AuxGetC0Size<float>()], m, currentKL0, currentKL1);
        } else {
            CopyInA2(currentA2, currentA1[offsetKL0 * BasicCubeAlign(m, BLOCK_CUBE)], m, currentKL0);
        }
        CopyInB2(currentB2, currentB1[offsetKL0 * AuxGetC0Size<float>()], n, currentKL0, currentKL1);
        SetFlag<HardEvent::MTE1_M>(l0BufferIndex);
        WaitFlag<HardEvent::MTE1_M>(l0BufferIndex);

        Compute(currentC, currentA2, currentB2, m, n, currentKL0, initC && offsetKL0 == 0U);
        SetFlag<HardEvent::M_MTE1>(CUBE_L0_EVENT_BASE + l0BufferIndex);
        ++l0PingPongID_;
    }
}

__aicore__ inline void MhcPreBackwardCubeCompute::PreloadB1(const MMConfig &config, const GlobalTensor<float> &b,
                                                            uint32_t n, uint32_t k, uint32_t bGmStride)
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

__aicore__ inline void MhcPreBackwardCubeCompute::CopyInA1(const LocalTensor<float> &dst,
                                                           const GlobalTensor<float> &src, uint32_t m, uint32_t k,
                                                           uint32_t gmStride)
{
    uint32_t alignedM = BasicCubeAlign(m, BLOCK_CUBE);
    uint32_t alignedK = BasicCubeAlign(k, AuxGetC0Size<float>());
    if (alignedM != m || alignedK != k) {
        uint16_t blockNum = static_cast<uint16_t>(alignedM * alignedK * sizeof(float) / CUBE_BLOCK_BYTES);
        InitConstValueParams<float> zeroParams(1, blockNum, 0, 0.0f);
        InitConstValue(dst, zeroParams);
        PipeBarrier<PIPE_MTE2>();
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

__aicore__ inline void MhcPreBackwardCubeCompute::CopyInA1Trans(const LocalTensor<float> &dst,
                                                                const GlobalTensor<float> &src, uint32_t m, uint32_t k,
                                                                uint32_t gmStride)
{
    uint32_t alignedM = BasicCubeAlign(m, BLOCK_CUBE);
    uint32_t alignedK = BasicCubeAlign(k, BLOCK_CUBE);
    if (alignedM != m || alignedK != k) {
        uint16_t blockNum = static_cast<uint16_t>(alignedM * alignedK * sizeof(float) / CUBE_BLOCK_BYTES);
        InitConstValueParams<float> zeroParams(1, blockNum, 0, 0.0f);
        InitConstValue(dst, zeroParams);
        PipeBarrier<PIPE_MTE2>();
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

__aicore__ inline void MhcPreBackwardCubeCompute::CopyInB1(const LocalTensor<float> &dst,
                                                           const GlobalTensor<float> &src, uint32_t n, uint32_t k,
                                                           uint32_t gmStride)
{
    uint32_t alignedN = BasicCubeAlign(n, BLOCK_CUBE);
    uint32_t alignedK = BasicCubeAlign(k, BLOCK_CUBE);
    if (alignedN != n || alignedK != k) {
        uint16_t blockNum = static_cast<uint16_t>(alignedN * alignedK * sizeof(float) / CUBE_BLOCK_BYTES);
        InitConstValueParams<float> zeroParams(1, blockNum, 0, 0.0f);
        InitConstValue(dst, zeroParams);
        PipeBarrier<PIPE_MTE2>();
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

__aicore__ inline void MhcPreBackwardCubeCompute::CopyInA2(const LocalTensor<float> &dst, const LocalTensor<float> &src,
                                                           uint32_t m, uint32_t k)
{
    LoadData2DParamsV2 p;
    p.mStartPosition = 0;
    p.kStartPosition = 0;
    p.mStep = BasicCubeCeilDiv(m, BLOCK_CUBE);
    p.kStep = BasicCubeCeilDiv(k, AuxGetC0Size<float>());
    p.srcStride = BasicCubeCeilDiv(m, BLOCK_CUBE);
    p.dstStride = p.mStep;
    p.ifTranspose = false;
    LoadData<float>(dst, src, p);
}

__aicore__ inline void MhcPreBackwardCubeCompute::CopyInA2Trans(const LocalTensor<float> &dst,
                                                                const LocalTensor<float> &src, uint32_t m, uint32_t k,
                                                                uint32_t kL1)
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

__aicore__ inline void MhcPreBackwardCubeCompute::CopyInB2(const LocalTensor<float> &dst, const LocalTensor<float> &src,
                                                           uint32_t n, uint32_t k, uint32_t kL1)
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

__aicore__ inline void MhcPreBackwardCubeCompute::Compute(const LocalTensor<float> &c, const LocalTensor<float> &a,
                                                          const LocalTensor<float> &b, uint32_t m, uint32_t n,
                                                          uint32_t k, bool initC)
{
    MmadParams p;
    p.m = m;
    p.n = n;
    p.k = k;
    p.disableGemv = true;
    p.cmatrixInitVal = initC;
    p.cmatrixSource = false;
    // unitFlag=0，L0C的生产/消费完全由M_FIX与FIX_M事件管理。
    p.unitFlag = 0;
    Mmad(c, a, b, p);
}

__aicore__ inline void MhcPreBackwardCubeCompute::CopyOut(const GlobalTensor<float> &dst, const LocalTensor<float> &src,
                                                          uint32_t m, uint32_t n, uint32_t dstStride)
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
} // namespace MhcPreBackward
#endif
