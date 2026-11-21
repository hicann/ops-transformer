/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
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
constexpr FixpipeConfig C0_ROW_MAJOR_UB = {CO2Layout::ROW_MAJOR, true};

template <bool USE_HF32>
class MhcPreBackwardCubeCompute {
public:
    __aicore__ inline void Init(TPipe* pipe, const MhcPreBackwardTilingData* tiling, bool allC1BFromSharedL1,
                                bool rowPartitionV2);

    __aicore__ inline void ProcessC0(const MMConfig& config, const GlobalTensor<float>& a, const GlobalTensor<float>& b,
                                     const GlobalTensor<float>& c, uint32_t m, uint32_t n, uint32_t k,
                                     uint32_t bGmStride, bool packedOutput, bool directUb = false,
                                     bool explicitSync = false, bool reuseWaveA = false, uint32_t waveTile = 0U,
                                     bool rowPartitionTile = false);

    __aicore__ inline void ProcessC1(const MMConfig& config, const GlobalTensor<float>& a, const GlobalTensor<float>& b,
                                     const GlobalTensor<float>& c, uint32_t m, uint32_t n, uint32_t k,
                                     uint32_t aGmStride, uint32_t bGmStride, uint32_t cGmStride, bool atomicAdd,
                                     bool directB, uint32_t directBOffset, bool streamB, bool sliceRecycle = false,
                                     bool lastBS = true, bool independentBS = false, uint32_t waveTile = 0U,
                                     uint32_t waveTiles = 0U, bool retainC1Accumulator = true,
                                     bool precisionPartials = false);

    __aicore__ inline void End();

private:
    uint64_t c1PartialPlaneElements_{0U};

    __aicore__ inline void InitPhiWaveState(const MhcPreBackwardTilingData* tiling, bool allC1BFromSharedL1,
                                            uint32_t bOffset);

    __aicore__ inline void InitCubeBuffers(const MhcPreBackwardTilingData* tiling);

    __aicore__ inline void InitEventTokens();

    __aicore__ inline void PrepareHmixCache(const GlobalTensor<float>& a, bool reuseWaveA);

    __aicore__ inline void PreparePhiForC0(const MMConfig& config, const GlobalTensor<float>& b, uint32_t n, uint32_t k,
                                           uint32_t bGmStride, bool reuseWaveA, uint32_t waveTile);

    __aicore__ inline void ProcessC0Tiles(const MMConfig& config, const GlobalTensor<float>& a,
                                          const GlobalTensor<float>& c, uint32_t m, uint32_t n, uint32_t k,
                                          bool packedOutput);

    __aicore__ inline void EmitDirectC0Tile(const LocalTensor<float>& currentC, uint32_t m, uint32_t n,
                                            uint32_t cBufferIndex);

    __aicore__ inline void ComputeTileC0(const MMConfig& config, const GlobalTensor<float>& a,
                                         const GlobalTensor<float>& c, uint32_t m, uint32_t n, uint32_t k,
                                         uint32_t aGmStride, uint32_t cGmStride, bool packedOutput);

    __aicore__ inline void ComputeTileC1(const MMConfig& config, const GlobalTensor<float>& a,
                                         const GlobalTensor<float>& b, const GlobalTensor<float>& c, uint32_t m,
                                         uint32_t n, uint32_t k, uint32_t aGmStride, uint32_t bGmStride,
                                         uint32_t cGmStride, bool atomicAdd, bool directB, uint32_t directBOffset,
                                         bool streamB, bool accumulateC1, bool lastBS, uint32_t accumulatorOffset,
                                         bool waveAccumulate);

    __aicore__ inline void ProcessKRangeC0(const MMConfig& config, const LocalTensor<float>& currentC,
                                           const GlobalTensor<float>& a, uint32_t m, uint32_t n, uint32_t k,
                                           uint32_t aGmStride);

    __aicore__ inline void ProcessKRangeC1(const MMConfig& config, const LocalTensor<float>& currentC,
                                           const GlobalTensor<float>& a, const GlobalTensor<float>& b, uint32_t m,
                                           uint32_t n, uint32_t k, uint32_t aGmStride, uint32_t bGmStride, bool directB,
                                           uint32_t directBOffset, bool streamB, bool initC,
                                           const GlobalTensor<float>& c, uint32_t cGmStride, bool atomicAdd,
                                           uint32_t cBufferIndex, bool splitKOutput);

    __aicore__ inline void ProcessL0KRange(const MMConfig& config, const LocalTensor<float>& currentC,
                                           const LocalTensor<float>& currentA1, const LocalTensor<float>& currentB1,
                                           uint32_t m, uint32_t n, uint32_t currentKL1, bool transA, bool initC,
                                           bool unitSync = false, bool lastL1 = false);

    __aicore__ inline void FinishTile(const GlobalTensor<float>& c, const LocalTensor<float>& currentC, uint32_t m,
                                      uint32_t n, uint32_t cGmStride, bool atomicAdd, uint32_t cBufferIndex,
                                      bool packedOutput = false);

    __aicore__ inline void PreloadB1(const MMConfig& config, const GlobalTensor<float>& b, uint32_t n, uint32_t k,
                                     uint32_t bGmStride);

    __aicore__ inline void CopyInA1(const LocalTensor<float>& dst, const GlobalTensor<float>& src, uint32_t m,
                                    uint32_t k, uint32_t gmStride);

    __aicore__ inline void CopyInL1Nz(const LocalTensor<float>& dst, const GlobalTensor<float>& src, uint32_t cols,
                                      uint32_t k, uint32_t gmStride);

    __aicore__ inline void CopyInA2(const LocalTensor<float>& dst, const LocalTensor<float>& src, uint32_t m,
                                    uint32_t k);

    __aicore__ inline void CopyInL0Trans(const LocalTensor<float>& dst, const LocalTensor<float>& src, uint32_t cols,
                                         uint32_t k, uint32_t kL1);

    __aicore__ inline void Compute(const LocalTensor<float>& c, const LocalTensor<float>& a,
                                   const LocalTensor<float>& b, uint32_t m, uint32_t n, uint32_t k, bool initC,
                                   uint8_t unitFlag = 0);

    __aicore__ inline void CopyOut(const GlobalTensor<float>& dst, const LocalTensor<float>& src, uint32_t m,
                                   uint32_t n, uint32_t dstStride);

    bool rowPartitionV2_{false};
    bool c1AccEnabled_{false};
    uint32_t c1AccSlot_{0};
    uint32_t c1AccPoolElems_{0};
    bool hmixCacheAllowed_{false};
    bool hmixCacheC0_{false};
    bool hmixCacheC0Hit_{false};
    bool hmixCacheC1_{false};
    uint32_t hmixCachePitch_{0};
    uint32_t hmixCacheDepth_{0};
    uint32_t hmixCacheNext_{0};
    uint32_t hmixCacheWrite_{0};
    uint32_t hmixCacheRead_{0};
    uint64_t hmixCacheAddress_[2]{0, 0};
    bool hmixCacheValid_[2]{false, false};
    bool recycleC1Slices_{false};
    bool directC0_{false};
    bool explicitC0Sync_{false};
    uint32_t directSlot_{0};
    bool phiResident_{false};
    uint64_t phiAddress_{0};
    uint32_t phiN_{0};
    uint32_t phiK_{0};
    uint32_t phiStride_{0};
    uint32_t phiL1K_{0};
    uint32_t phiDepth_{0};
    uint32_t phiWaveSlots_{0};
    uint32_t phiWaveTileElems_{0};
    uint32_t phiWaveOffset_{0};
    uint64_t phiWaveValid_{0};
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

template <bool USE_HF32>
__aicore__ inline void MhcPreBackwardCubeCompute<USE_HF32>::Init(TPipe* pipe, const MhcPreBackwardTilingData* tiling,
                                                                 bool allC1BFromSharedL1, bool rowPartitionV2)
{
    rowPartitionV2_ = rowPartitionV2;
    maxBufferDepth_ = tiling->maxBufferDepth;
    l1SingleBufferElems_ = tiling->l1SingleBufferElems;
    hmixCachePitch_ = MhcPreBackwardUtils::CeilAlign<uint64_t>(static_cast<uint32_t>(tiling->fusionSize), BLOCK_CUBE);
    hmixCacheDepth_ = tiling->mmConfigC0.depthA1;
    uint64_t cacheElems = static_cast<uint64_t>(DOUBLE_BUFFER) * BS_BLOCK_SIZE * hmixCachePitch_;
    uint64_t phiElems = static_cast<uint64_t>(tiling->mmConfigC0.baseN) * tiling->mmConfigC0.l1K;
    // 所有 C1 的 B 分块都位于独立的共享区域。C0 只需要一个 phi
    // K 面板，因此普通 A/B 分区可以同时容纳两个 Gmix bank 和 phi。
    bool repartitionL1 = allC1BFromSharedL1 && tiling->nD % ND_BLOCK_SIZE == 0U &&
                         cacheElems > tiling->l1BOffsetElems &&
                         cacheElems + phiElems <= tiling->l1UsedBytes / sizeof(float);
    hmixCacheAllowed_ =
        tiling->mmConfigC0.baseM == MATMUL_WRITE_OFFSET_M && tiling->mmConfigC1.l1K == MATMUL_WRITE_OFFSET_M &&
        tiling->fusionSize <= tiling->mmConfigC1.baseM && tiling->fusionSize <= tiling->mmConfigC0.l1K &&
        tiling->mmConfigC0.depthA1 == tiling->mmConfigC1.depthA1 && tiling->totalLength % BS_BLOCK_SIZE == 0U &&
        (cacheElems <= tiling->l1BOffsetElems || repartitionL1);
    l0ABSingleBufferElems_ = tiling->l0ABSingleBufferElems;
    l0CSingleBufferElems_ = tiling->l0CSingleBufferElems;
    pipe->InitBuffer(l1Buf_, tiling->l1UsedBytes);
    aL1Local_ = l1Buf_.Get<float>();
    uint32_t bOffset = hmixCacheAllowed_ && repartitionL1 ? static_cast<uint32_t>(cacheElems) : tiling->l1BOffsetElems;
    bL1Local_ = aL1Local_[bOffset];
    InitPhiWaveState(tiling, allC1BFromSharedL1, bOffset);
    InitCubeBuffers(tiling);
    InitEventTokens();
}

template <bool USE_HF32>
__aicore__ inline void MhcPreBackwardCubeCompute<USE_HF32>::InitPhiWaveState(const MhcPreBackwardTilingData* tiling,
                                                                             bool allC1BFromSharedL1, uint32_t bOffset)
{
    if constexpr (!USE_HF32) {
        // 以 BS 为主序的 wave 会重复访问本核的全部 phi 分块；仅当普通 B1
        // 有足够空间且 C1 不会覆盖时，才固定保存完整工作集。
        uint32_t ndPerCube = tiling->nD / GetBlockNum();
        if (ndPerCube == 0U) {
            ndPerCube = tiling->nD;
        } else {
            const uint32_t ndAlign =
                tiling->nD > GetBlockNum() * ND_BLOCK_SIZE ? ND_BLOCK_SIZE / NUMBER_TWO : ND_BLOCK_SIZE;
            ndPerCube = MhcPreBackwardUtils::CeilAlign<uint64_t>(ndPerCube, ndAlign);
        }
        const uint32_t tiles = MhcPreBackwardUtils::CeilDiv<uint64_t>(ndPerCube, tiling->mmConfigC0.baseN);
        // CopyInB1 按实际 K 分片打包，而不是按配置的最大 l1K 打包。
        phiWaveTileElems_ =
            tiling->mmConfigC0.baseN * MhcPreBackwardUtils::CeilAlign<uint64_t>(tiling->fusionSize, BLOCK_CUBE);
        if (allC1BFromSharedL1 && tiling->fusionSize <= tiling->mmConfigC0.l1K && tiles > 0U && tiles <= 64U &&
            static_cast<uint64_t>(tiles) * phiWaveTileElems_ <= tiling->l1UsedBytes / sizeof(float) - bOffset) {
            phiWaveSlots_ = tiles;
        }
    }
}

template <bool USE_HF32>
__aicore__ inline void MhcPreBackwardCubeCompute<USE_HF32>::InitCubeBuffers(const MhcPreBackwardTilingData* tiling)
{
    aL0Local_ = LocalTensor<float>(TPosition::A2, 0, tiling->l0ABUsedBytes / sizeof(float));
    bL0Local_ = LocalTensor<float>(TPosition::B2, 0, tiling->l0ABUsedBytes / sizeof(float));
    cL0Local_ = LocalTensor<float>(TPosition::CO1, 0, tiling->l0CUsedBytes / sizeof(float));
    c1AccSlot_ =
        tiling->mmConfigC0.dbL0C > tiling->mmConfigC1.dbL0C ? tiling->mmConfigC0.dbL0C : tiling->mmConfigC1.dbL0C;
    c1AccEnabled_ = c1AccSlot_ >= maxBufferDepth_ &&
                    tiling->l0CUsedBytes / sizeof(float) >= (c1AccSlot_ + 1U) * l0CSingleBufferElems_;
    c1AccPoolElems_ = tiling->l0CUsedBytes / sizeof(float) - c1AccSlot_ * l0CSingleBufferElems_;
    if (c1AccEnabled_) {
        SetFlag<HardEvent::FIX_M>(c1AccSlot_);
    }
}

template <bool USE_HF32>
__aicore__ inline void MhcPreBackwardCubeCompute<USE_HF32>::InitEventTokens()
{
    for (uint32_t bufferIndex = 0; bufferIndex < maxBufferDepth_; ++bufferIndex) {
        // 三组初始令牌分别表示L1、L0A/B和L0C对应缓冲区可写。
        SetFlag<HardEvent::MTE1_MTE2>(bufferIndex);
        SetFlag<HardEvent::M_MTE1>(CUBE_L0_EVENT_BASE + bufferIndex);
        SetFlag<HardEvent::FIX_M>(bufferIndex);
    }
}

template <bool USE_HF32>
__aicore__ inline void MhcPreBackwardCubeCompute<USE_HF32>::ProcessC0(
    const MMConfig& config, const GlobalTensor<float>& a, const GlobalTensor<float>& b, const GlobalTensor<float>& c,
    uint32_t m, uint32_t n, uint32_t k, uint32_t bGmStride, bool packedOutput, bool directUb, bool explicitSync,
    bool reuseWaveA, uint32_t waveTile, bool rowPartitionTile)
{
    // C0和V2按有效列宽选择同一分行布局，跨N分组块保留原路径。
    rowPartitionV2_ = rowPartitionTile;
    directC0_ = directUb;
    explicitC0Sync_ = explicitSync;
    hmixCacheC0_ = hmixCacheAllowed_ && m == BS_BLOCK_SIZE;
    PrepareHmixCache(a, reuseWaveA);
    PreparePhiForC0(config, b, n, k, bGmStride, reuseWaveA, waveTile);
    ProcessC0Tiles(config, a, c, m, n, k, packedOutput);
}

template <bool USE_HF32>
__aicore__ inline void MhcPreBackwardCubeCompute<USE_HF32>::PrepareHmixCache(const GlobalTensor<float>& a,
                                                                             bool reuseWaveA)
{
    hmixCacheC0Hit_ = false;
    if (hmixCacheC0_ && reuseWaveA) {
        const uint64_t address = reinterpret_cast<uint64_t>(a.GetPhyAddr());
        for (uint32_t slot = 0U; slot < DOUBLE_BUFFER; ++slot) {
            if (hmixCacheValid_[slot] && hmixCacheAddress_[slot] == address) {
                hmixCacheWrite_ = slot;
                hmixCacheC0Hit_ = true;
                break;
            }
        }
    }
    if (hmixCacheC0_ && !hmixCacheC0Hit_) {
        // 两个原有的 L1 A 通道在复用前都必须完成最后一次 LoadData。
        for (uint32_t channel = 0; channel < hmixCacheDepth_; ++channel) {
            WaitFlag<HardEvent::MTE1_MTE2>(channel);
        }
        // 完成依赖闭环后再重新发出槽位 token。
        SetFlag<HardEvent::MTE2_MTE1>(0);
        WaitFlag<HardEvent::MTE2_MTE1>(0);
        for (uint32_t channel = 0; channel < hmixCacheDepth_; ++channel) {
            SetFlag<HardEvent::MTE1_MTE2>(channel);
        }
        hmixCacheWrite_ = hmixCacheNext_++ % DOUBLE_BUFFER;
        hmixCacheAddress_[hmixCacheWrite_] = reinterpret_cast<uint64_t>(a.GetPhyAddr());
        hmixCacheValid_[hmixCacheWrite_] = true;
    }
}

template <bool USE_HF32>
__aicore__ inline void MhcPreBackwardCubeCompute<USE_HF32>::PreparePhiForC0(const MMConfig& config,
                                                                            const GlobalTensor<float>& b, uint32_t n,
                                                                            uint32_t k, uint32_t bGmStride,
                                                                            bool reuseWaveA, uint32_t waveTile)
{
    // C0为[M=BS块,K=fusionSize]×[K=fusionSize,N=ND块]；仅在工作集可驻留时复用 phi。
    uint64_t phiAddress = reinterpret_cast<uint64_t>(b.GetPhyAddr());
    phiWaveOffset_ = 0U;
    const bool wavePhi = reuseWaveA && waveTile < phiWaveSlots_ && n <= config.baseN && k <= config.l1K;
    if (wavePhi) {
        const uint32_t slot = waveTile;
        phiWaveOffset_ = slot * phiWaveTileElems_;
        const uint64_t mask = 1ULL << slot;
        if ((phiWaveValid_ & mask) == 0U) {
            PreloadB1(config, b, n, k, bGmStride);
            phiWaveValid_ |= mask;
        }
        phiResident_ = false;
        return;
    }

    phiWaveValid_ = 0U;
    bool fitsB1 = static_cast<uint64_t>(k) <= static_cast<uint64_t>(config.l1K) * config.depthB1;
    bool samePhi = phiResident_ && phiAddress_ == phiAddress && phiN_ == n && phiK_ == k && phiStride_ == bGmStride &&
                   phiL1K_ == config.l1K && phiDepth_ == config.depthB1;
    if (!fitsB1 || !samePhi) {
        PreloadB1(config, b, n, k, bGmStride);
        phiAddress_ = phiAddress;
        phiN_ = n;
        phiK_ = k;
        phiStride_ = bGmStride;
        phiL1K_ = config.l1K;
        phiDepth_ = config.depthB1;
    }
    phiResident_ = fitsB1;
}

template <bool USE_HF32>
__aicore__ inline void MhcPreBackwardCubeCompute<USE_HF32>::ProcessC0Tiles(const MMConfig& config,
                                                                           const GlobalTensor<float>& a,
                                                                           const GlobalTensor<float>& c, uint32_t m,
                                                                           uint32_t n, uint32_t k, bool packedOutput)
{
    for (uint32_t offsetM = 0; offsetM < m; offsetM += config.baseM) {
        uint32_t remainM = m - offsetM;
        uint32_t currentM = remainM < config.baseM ? remainM : config.baseM;
        for (uint32_t offsetN = 0; offsetN < n; offsetN += config.baseN) {
            uint32_t remainN = n - offsetN;
            uint32_t currentN = remainN < config.baseN ? remainN : config.baseN;
            directSlot_ = offsetM;
            uint32_t dstRowStride = packedOutput ? ND_BLOCK_SIZE / 2U : ND_BLOCK_SIZE;
            ComputeTileC0(config, a[offsetM * k], c[offsetM * dstRowStride + offsetN], currentM, currentN, k, k,
                          dstRowStride, packedOutput);
        }
        if (!directC0_) {
            if (explicitC0Sync_) {
                C0PublishReady<PIPE_FIX>();
            } else {
                CrossCoreSetFlag<CROSS_CORE_FLAG_INDEX, PIPE_FIX>(CROSS_CORE_WAIT_FLAG_C0);
            }
        }
    }
}

template <bool USE_HF32>
__aicore__ inline void MhcPreBackwardCubeCompute<USE_HF32>::ProcessC1(
    const MMConfig& config, const GlobalTensor<float>& a, const GlobalTensor<float>& b, const GlobalTensor<float>& c,
    uint32_t m, uint32_t n, uint32_t k, uint32_t aGmStride, uint32_t bGmStride, uint32_t cGmStride, bool atomicAdd,
    bool directB, uint32_t directBOffset, bool streamB, bool sliceRecycle, bool lastBS, bool independentBS,
    uint32_t waveTile, uint32_t waveTiles, bool retainC1Accumulator, bool precisionPartials)
{
    c1PartialPlaneElements_ = !USE_HF32 && precisionPartials ? uint64_t(m) * cGmStride : 0U;
    recycleC1Slices_ = sliceRecycle;
    hmixCacheC1_ = false;
    if (hmixCacheAllowed_ && k == BS_BLOCK_SIZE && config.l1K == MATMUL_WRITE_OFFSET_M) {
        uint64_t address = reinterpret_cast<uint64_t>(a.GetPhyAddr());
        for (uint32_t slot = 0; slot < DOUBLE_BUFFER; ++slot) {
            if (hmixCacheValid_[slot] && address == hmixCacheAddress_[slot]) {
                hmixCacheRead_ = slot;
                hmixCacheC1_ = true;
                break;
            }
        }
    }
    if (!hmixCacheC1_) {
        hmixCacheValid_[0] = false;
        hmixCacheValid_[1] = false;
    }
    if (!directB) {
        // GM C1 路径会覆盖存放 phi 的普通 B1 槽位。
        phiResident_ = false;
        phiWaveValid_ = 0U;
    }
    // 每个 ND 输出在 BS wave 期间使用独立的 compact L0C 区域。
    // 如果本核完整工作集无法容纳，则保留 GM 部分输出。
    const uint32_t compactStride =
        static_cast<uint32_t>(MhcPreBackwardUtils::CeilAlign<uint64_t>(m, BLOCK_CUBE)) * config.baseN;
    const bool waveAccumulate = retainC1Accumulator && independentBS && c1AccEnabled_ && m <= config.baseM &&
                                n <= config.baseN && waveTiles > 0U && waveTile < waveTiles &&
                                static_cast<uint64_t>(compactStride) * waveTiles <= c1AccPoolElems_;
    // 长序列按 BS 分段输出部分和，再合并各段结果。
    bool accumulateC1 = retainC1Accumulator && c1AccEnabled_ && (!independentBS || waveAccumulate) &&
                        m <= config.baseM && n <= config.baseN;
    const uint32_t accumulatorOffset = waveAccumulate ? waveTile * compactStride : 0U;
    for (uint32_t offsetM = 0; offsetM < m; offsetM += config.baseM) {
        uint32_t remainM = m - offsetM;
        uint32_t currentM = remainM < config.baseM ? remainM : config.baseM;
        for (uint32_t offsetN = 0; offsetN < n; offsetN += config.baseN) {
            uint32_t remainN = n - offsetN;
            uint32_t currentN = remainN < config.baseN ? remainN : config.baseN;
            ComputeTileC1(config, a[offsetM], b[offsetN], c[offsetM * cGmStride + offsetN], currentM, currentN, k,
                          aGmStride, bGmStride, cGmStride, atomicAdd, directB, directBOffset, streamB, accumulateC1,
                          lastBS, accumulatorOffset, waveAccumulate);
        }
    }
}

template <bool USE_HF32>
__aicore__ inline void MhcPreBackwardCubeCompute<USE_HF32>::End()
{
    if (c1AccEnabled_) {
        WaitFlag<HardEvent::FIX_M>(c1AccSlot_);
    }
    for (uint32_t bufferIndex = 0; bufferIndex < maxBufferDepth_; ++bufferIndex) {
        WaitFlag<HardEvent::MTE1_MTE2>(bufferIndex);
        WaitFlag<HardEvent::M_MTE1>(CUBE_L0_EVENT_BASE + bufferIndex);
        WaitFlag<HardEvent::FIX_M>(bufferIndex);
    }
}

template <bool USE_HF32>
__aicore__ inline void MhcPreBackwardCubeCompute<USE_HF32>::ComputeTileC0(const MMConfig& config,
                                                                          const GlobalTensor<float>& a,
                                                                          const GlobalTensor<float>& c, uint32_t m,
                                                                          uint32_t n, uint32_t k, uint32_t aGmStride,
                                                                          uint32_t cGmStride, bool packedOutput)
{
    uint32_t cDepth = config.dbL0C;
    uint32_t cBufferIndex = cl0PingPongID_ % cDepth;
    LocalTensor<float> currentC = cL0Local_[cBufferIndex * l0CSingleBufferElems_];
    WaitFlag<HardEvent::FIX_M>(cBufferIndex);

    ProcessKRangeC0(config, currentC, a, m, n, k, aGmStride);
    if (directC0_) {
        EmitDirectC0Tile(currentC, m, n, cBufferIndex);
        return;
    }
    FinishTile(c, currentC, m, n, cGmStride, false, cBufferIndex, packedOutput);
}

template <bool USE_HF32>
__aicore__ inline void MhcPreBackwardCubeCompute<USE_HF32>::EmitDirectC0Tile(const LocalTensor<float>& currentC,
                                                                             uint32_t m, uint32_t n,
                                                                             uint32_t cBufferIndex)
{
    if (rowPartitionV2_ && !USE_HF32) {
        // FP32 两次输出等待完整 L0C；HF32 的 M 双目标搬运使用 unitFlag。
        SetFlag<HardEvent::M_FIX>(cBufferIndex);
        WaitFlag<HardEvent::M_FIX>(cBufferIndex);
    }
    uint32_t tileRows = C0UbTileRows();
    uint32_t firstSlot = (directSlot_ / tileRows) % C0UbSlotCount();
    // MMAD 通过 unitFlag 发布 L0C 就绪状态。Fixpipe 可以消费已就绪的部分。
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> p;
    p.nSize = n;
    p.mSize = tileRows;
    p.srcStride = MhcPreBackwardUtils::CeilAlign<uint64_t>(m, BLOCK_CUBE);
    p.dstStride = ND_BLOCK_SIZE / 2U;
    // 硬件在成对的 AIV 之间拆分 N，与 V2 的列拆分保持一致。
    p.dualDstCtl = n == ND_BLOCK_SIZE ? static_cast<uint8_t>(McgShfMode::DUAL_DST_SPLIT_N) : 0;
    p.quantPre = QuantMode_t::NoQuant;
    p.unitFlag = 3;
    p.params.ndNum = 1;
    p.params.srcNdStride = 0;
    p.params.dstNdStride = 0;

    for (uint32_t slot = 0; slot < m / tileRows; ++slot) {
        uint32_t ringSlot = (firstSlot + slot) % C0UbSlotCount();
        C0WaitFreeSlot<PIPE_FIX>(ringSlot);
        LocalTensor<float> dst(TPosition::VECIN, C0UbSlotOffset(ringSlot), C0UbSlotBytes() / sizeof(float));
        // L0C NZ 为每组行保存一个 16 列片段。
        uint32_t rowOffset = slot * tileRows * BLOCK_CUBE;
        // 64 列尾块由一个 AIV 负责：前后各处理 256 行。
        p.subBlockId = n != ND_BLOCK_SIZE && directSlot_ + slot * tileRows >= BS_BLOCK_SIZE / NUMBER_TWO;
        if (rowPartitionV2_) {
            if constexpr (USE_HF32) {
                // M 双目标一次搬运，将两半行分别写入两个 AIV；V2 继续消费相同的行分片。
                p.dualDstCtl = static_cast<uint8_t>(McgShfMode::DUAL_DST_SPLIT_M);
                p.mSize = tileRows;
                p.dstStride = n;
                p.subBlockId = 0;
                p.unitFlag = 3;
                Fixpipe<float, float, C0_ROW_MAJOR_UB>(dst, currentC[rowOffset], p);
            } else {
                // FP32 保留原有的两次按行输出路径。
                p.dualDstCtl = 0;
                p.mSize = tileRows / NUMBER_TWO;
                p.dstStride = n;
                p.subBlockId = 0;
                p.unitFlag = 0;
                Fixpipe<float, float, C0_ROW_MAJOR_UB>(dst, currentC[rowOffset], p);
                p.subBlockId = 1;
                p.unitFlag = 0;
                Fixpipe<float, float, C0_ROW_MAJOR_UB>(dst, currentC[rowOffset + (tileRows / NUMBER_TWO) * BLOCK_CUBE],
                                                       p);
            }
        } else {
            Fixpipe<float, float, C0_ROW_MAJOR_UB>(dst, currentC[rowOffset], p);
        }
        C0PublishReady<PIPE_FIX>();
    }
    SetFlag<HardEvent::FIX_M>(cBufferIndex);
    ++cl0PingPongID_;
}

template <bool USE_HF32>
__aicore__ inline void MhcPreBackwardCubeCompute<USE_HF32>::ComputeTileC1(
    const MMConfig& config, const GlobalTensor<float>& a, const GlobalTensor<float>& b, const GlobalTensor<float>& c,
    uint32_t m, uint32_t n, uint32_t k, uint32_t aGmStride, uint32_t bGmStride, uint32_t cGmStride, bool atomicAdd,
    bool directB, uint32_t directBOffset, bool streamB, bool accumulateC1, bool lastBS, uint32_t accumulatorOffset,
    bool waveAccumulate)
{
    uint32_t cDepth = config.dbL0C;
    uint32_t cBufferIndex = accumulateC1 ? c1AccSlot_ : cl0PingPongID_ % cDepth;
    LocalTensor<float> currentC = cL0Local_[cBufferIndex * l0CSingleBufferElems_ + accumulatorOffset];
    if (!waveAccumulate && (!accumulateC1 || !atomicAdd)) {
        WaitFlag<HardEvent::FIX_M>(cBufferIndex);
    }

    // FP32 的每个 L1 K 分片独立求和，再合并分片结果。
    const bool splitKOutput = !USE_HF32 && !accumulateC1 && k > config.l1K;
    ProcessKRangeC1(config, currentC, a, b, m, n, k, aGmStride, bGmStride, directB, directBOffset, streamB,
                    !accumulateC1 || !atomicAdd, c, cGmStride, atomicAdd, cBufferIndex, splitKOutput);
    if (!splitKOutput && (!accumulateC1 || lastBS)) {
        if (waveAccumulate) {
            // 共享事件用于保证最后的 Fixpipe 写入；每个 ND 累加器
            // 存储互不重叠，最后一个 wave 前没有 Fixpipe 读取方。
            WaitFlag<HardEvent::FIX_M>(cBufferIndex);
        }
        FinishTile(c, currentC, m, n, cGmStride, accumulateC1 ? false : atomicAdd, cBufferIndex);
    }
}

template <bool USE_HF32>
__aicore__ inline void MhcPreBackwardCubeCompute<USE_HF32>::FinishTile(const GlobalTensor<float>& c,
                                                                       const LocalTensor<float>& currentC, uint32_t m,
                                                                       uint32_t n, uint32_t cGmStride, bool atomicAdd,
                                                                       uint32_t cBufferIndex, bool packedOutput)
{
    SetFlag<HardEvent::M_FIX>(cBufferIndex);
    WaitFlag<HardEvent::M_FIX>(cBufferIndex);
    if (atomicAdd) {
        // 后续C1分段写同一gradPhi前仅等待上一条Fixpipe完成，固定AtomicAdd顺序；
        // 等待发生在当前MMAD结束后，继续保留L0C双槽和MMAD/Fixpipe重叠。
        PipeBarrier<PIPE_FIX>();
        SetAtomicAdd<float>();
    }
    if (packedOutput) {
        constexpr uint32_t HALF_N = ND_BLOCK_SIZE / 2U;
        CopyOut(c, currentC, m, HALF_N, HALF_N);
        CopyOut(c[BS_BLOCK_SIZE * HALF_N], currentC[MhcPreBackwardUtils::CeilAlign<uint64_t>(m, BLOCK_CUBE) * HALF_N],
                m, HALF_N, HALF_N);
    } else {
        CopyOut(c, currentC, m, n, cGmStride);
    }
    if (atomicAdd) {
        SetAtomicNone();
    }
    SetFlag<HardEvent::FIX_M>(cBufferIndex);
    ++cl0PingPongID_;
}

template <bool USE_HF32>
__aicore__ inline void MhcPreBackwardCubeCompute<USE_HF32>::ProcessKRangeC0(const MMConfig& config,
                                                                            const LocalTensor<float>& currentC,
                                                                            const GlobalTensor<float>& a, uint32_t m,
                                                                            uint32_t n, uint32_t k, uint32_t aGmStride)
{
    uint32_t l1K = config.l1K;
    uint32_t l1Depth = config.depthA1;
    for (uint32_t offsetKL1 = 0; offsetKL1 < k; offsetKL1 += l1K) {
        uint32_t remainKL1 = k - offsetKL1;
        uint32_t currentKL1 = remainKL1 < l1K ? remainKL1 : l1K;
        uint32_t l1BufferIndex = (offsetKL1 / l1K) % l1Depth;
        LocalTensor<float> currentA1 = aL1Local_[l1BufferIndex * l1SingleBufferElems_];
        LocalTensor<float> currentB1 = bL1Local_[phiWaveOffset_ + l1BufferIndex * l1SingleBufferElems_];

        WaitFlag<HardEvent::MTE1_MTE2>(l1BufferIndex);
        if (hmixCacheC0_) {
            currentA1 = aL1Local_[(hmixCacheWrite_ * BS_BLOCK_SIZE + directSlot_) * hmixCachePitch_];
            // 相同的 [128,F] ND 输入同时按 C1 转置 LoadData 的要求补齐。
            if (!hmixCacheC0Hit_) {
                CopyInL1Nz(currentA1, a[offsetKL1], currentKL1, m, aGmStride);
            }
        } else {
            CopyInA1(currentA1, a[offsetKL1], m, currentKL1, aGmStride);
        }
        // 命中同一 wave 的 A-cache 时，本次迭代无需执行 MTE2 搬运。
        // PreloadB1 已完成 phi 的就绪，下面保留 MTE1_MTE2 的占用令牌，
        // 确保后续 phi/A 刷新仍等待当前读取完成。
        if (!hmixCacheC0Hit_) {
            SetFlag<HardEvent::MTE2_MTE1>(l1BufferIndex);
            WaitFlag<HardEvent::MTE2_MTE1>(l1BufferIndex);
        }

        // HF32 的 M 双目标 Fixpipe 依赖 Mmad 的 unitFlag 细粒度就绪；FP32 两次搬运继续等待完整 L0C。
        ProcessL0KRange(config, currentC, currentA1, currentB1, m, n, currentKL1, false, offsetKL1 == 0U,
                        directC0_ && (!rowPartitionV2_ || USE_HF32), offsetKL1 + currentKL1 == k);

        // MTE1完成该L1半区的最后一次读取后，允许MTE2复用此半区。
        SetFlag<HardEvent::MTE1_MTE2>(l1BufferIndex);
        ++l1PingPongID_;
    }
}

template <bool USE_HF32>
__aicore__ inline void MhcPreBackwardCubeCompute<USE_HF32>::ProcessKRangeC1(
    const MMConfig& config, const LocalTensor<float>& currentC, const GlobalTensor<float>& a,
    const GlobalTensor<float>& b, uint32_t m, uint32_t n, uint32_t k, uint32_t aGmStride, uint32_t bGmStride,
    bool directB, uint32_t directBOffset, bool streamB, bool initC, const GlobalTensor<float>& c, uint32_t cGmStride,
    bool atomicAdd, uint32_t cBufferIndex, bool splitKOutput)
{
    uint32_t l1K = config.l1K;
    uint32_t l1Depth = config.depthA1;
    for (uint32_t offsetKL1 = 0; offsetKL1 < k; offsetKL1 += l1K) {
        if (streamB) {
            CrossCoreWaitFlag(CROSS_CORE_WAIT_FLAG_C1);
        }
        uint32_t remainKL1 = k - offsetKL1;
        uint32_t currentKL1 = remainKL1 < l1K ? remainKL1 : l1K;
        uint32_t l1BufferIndex = l1PingPongID_ % l1Depth;
        LocalTensor<float> currentA1 = aL1Local_[l1BufferIndex * l1SingleBufferElems_];
        LocalTensor<float> currentB1;
        if (directB) {
            // V2 已在 MM L1 缓冲区之后将该 K 分片放置为 NZ 布局。
            // UB→L1 NZ 为 64 列 ND 尾块保留物理 128 列步长。
            currentB1 = LocalTensor<float>(TPosition::A1, directBOffset + offsetKL1 * ND_BLOCK_SIZE * sizeof(float),
                                           currentKL1 * n);
        } else {
            currentB1 = bL1Local_[l1BufferIndex * l1SingleBufferElems_];
        }

        // 两个操作数都常驻时，本次迭代既不写入也不借用
        // 普通 L1 ping-pong 槽位，因此保持其事件 token 不变。
        // 缓存的 A 已由 C0 准备就绪；direct B 由 C1 生产者标志保护。
        bool residentInputs = hmixCacheC1_ && directB;
        if (!residentInputs) {
            WaitFlag<HardEvent::MTE1_MTE2>(l1BufferIndex);
        }
        if (hmixCacheC1_) {
            currentA1 = aL1Local_[(hmixCacheRead_ * BS_BLOCK_SIZE + offsetKL1) * hmixCachePitch_];
        } else {
            CopyInL1Nz(currentA1, a[offsetKL1 * aGmStride], m, currentKL1, aGmStride);
        }
        if (!directB) {
            CopyInL1Nz(currentB1, b[offsetKL1 * bGmStride], n, currentKL1, bGmStride);
        }
        if (!residentInputs) {
            SetFlag<HardEvent::MTE2_MTE1>(l1BufferIndex);
            WaitFlag<HardEvent::MTE2_MTE1>(l1BufferIndex);
        }

        // 部分和交替使用现有 L0C 双槽，让本次 MMAD 与上次 Fixpipe 重叠。
        uint32_t partialSlot = cBufferIndex;
        LocalTensor<float> partialC = currentC;
        if (splitKOutput && offsetKL1 != 0U) {
            partialSlot = cl0PingPongID_ % config.dbL0C;
            partialC = cL0Local_[partialSlot * l0CSingleBufferElems_];
            WaitFlag<HardEvent::FIX_M>(partialSlot);
        }
        ProcessL0KRange(config, partialC, currentA1, currentB1, m, n, currentKL1, true,
                        splitKOutput || (initC && offsetKL1 == 0U));
        if (directB && recycleC1Slices_) {
            // 该 L1 分片的所有 LoadData 读取都已在 MTE1 上发出。
            // 生产者获准覆盖该分片后，MMAD 仍可继续执行。
            C1PublishSliceFree<PIPE_MTE1>(offsetKL1 / l1K);
        }

        if (!residentInputs) {
            SetFlag<HardEvent::MTE1_MTE2>(l1BufferIndex);
        }
        if (splitKOutput) {
            if (c1PartialPlaneElements_ != 0U) {
                FinishTile(c[(offsetKL1 / l1K) * c1PartialPlaneElements_], partialC, m, n, cGmStride, false,
                           partialSlot);
            } else {
                FinishTile(c, partialC, m, n, cGmStride, atomicAdd || offsetKL1 != 0U, partialSlot);
            }
        }
        ++l1PingPongID_;
    }
}

template <bool USE_HF32>
__aicore__ inline void MhcPreBackwardCubeCompute<USE_HF32>::ProcessL0KRange(
    const MMConfig& config, const LocalTensor<float>& currentC, const LocalTensor<float>& currentA1,
    const LocalTensor<float>& currentB1, uint32_t m, uint32_t n, uint32_t currentKL1, bool transA, bool initC,
    bool unitSync, bool lastL1)
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
            CopyInL0Trans(currentA2, currentA1[offsetKL0 * AuxGetC0Size<float>()], m, currentKL0, currentKL1);
        } else {
            CopyInA2(currentA2, currentA1[offsetKL0 * MhcPreBackwardUtils::CeilAlign<uint64_t>(m, BLOCK_CUBE)], m,
                     currentKL0);
        }
        CopyInL0Trans(currentB2, currentB1[offsetKL0 * AuxGetC0Size<float>()], n, currentKL0, currentKL1);
        SetFlag<HardEvent::MTE1_M>(l0BufferIndex);
        WaitFlag<HardEvent::MTE1_M>(l0BufferIndex);

        uint8_t unitFlag = unitSync ? (lastL1 && offsetKL0 + currentKL0 == currentKL1 ? 3U : 2U) : 0U;
        Compute(currentC, currentA2, currentB2, m, n, currentKL0, initC && offsetKL0 == 0U, unitFlag);
        SetFlag<HardEvent::M_MTE1>(CUBE_L0_EVENT_BASE + l0BufferIndex);
        ++l0PingPongID_;
    }
}

template <bool USE_HF32>
__aicore__ inline void MhcPreBackwardCubeCompute<USE_HF32>::PreloadB1(const MMConfig& config,
                                                                      const GlobalTensor<float>& b, uint32_t n,
                                                                      uint32_t k, uint32_t bGmStride)
{
    for (uint32_t offsetKL1 = 0; offsetKL1 < k; offsetKL1 += config.l1K) {
        uint32_t remainKL1 = k - offsetKL1;
        uint32_t currentKL1 = remainKL1 < config.l1K ? remainKL1 : config.l1K;
        uint32_t l1BufferIndex = (offsetKL1 / config.l1K) % config.depthB1;
        LocalTensor<float> currentB1 = bL1Local_[phiWaveOffset_ + l1BufferIndex * l1SingleBufferElems_];

        WaitFlag<HardEvent::MTE1_MTE2>(l1BufferIndex);
        CopyInL1Nz(currentB1, b[offsetKL1 * bGmStride], n, currentKL1, bGmStride);
        SetFlag<HardEvent::MTE2_MTE1>(l1BufferIndex);
        WaitFlag<HardEvent::MTE2_MTE1>(l1BufferIndex);
        SetFlag<HardEvent::MTE1_MTE2>(l1BufferIndex);
    }
}

template <bool USE_HF32>
__aicore__ inline void MhcPreBackwardCubeCompute<USE_HF32>::CopyInA1(const LocalTensor<float>& dst,
                                                                     const GlobalTensor<float>& src, uint32_t m,
                                                                     uint32_t k, uint32_t gmStride)
{
    uint32_t alignedM = MhcPreBackwardUtils::CeilAlign<uint64_t>(m, BLOCK_CUBE);
    uint32_t alignedK = MhcPreBackwardUtils::CeilAlign<uint64_t>(k, AuxGetC0Size<float>());
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
    p.dstNzC0Stride = MhcPreBackwardUtils::CeilAlign<uint64_t>(m, BLOCK_CUBE);
    p.dstNzNStride = 1;
    p.dstNzMatrixStride = 1;
    DataCopy(dst, src, p);
}

template <bool USE_HF32>
__aicore__ inline void MhcPreBackwardCubeCompute<USE_HF32>::CopyInL1Nz(const LocalTensor<float>& dst,
                                                                       const GlobalTensor<float>& src, uint32_t cols,
                                                                       uint32_t k, uint32_t gmStride)
{
    uint32_t alignedCols = MhcPreBackwardUtils::CeilAlign<uint64_t>(cols, BLOCK_CUBE);
    uint32_t alignedK = MhcPreBackwardUtils::CeilAlign<uint64_t>(k, BLOCK_CUBE);
    if (alignedCols != cols || alignedK != k) {
        uint16_t blockNum = static_cast<uint16_t>(alignedCols * alignedK * sizeof(float) / CUBE_BLOCK_BYTES);
        InitConstValueParams<float> zeroParams(1, blockNum, 0, 0.0f);
        InitConstValue(dst, zeroParams);
        PipeBarrier<PIPE_MTE2>();
    }
    Nd2NzParams p;
    p.ndNum = 1;
    p.nValue = k;
    p.dValue = cols;
    p.srcNdMatrixStride = 1;
    p.srcDValue = gmStride;
    p.dstNzC0Stride = MhcPreBackwardUtils::CeilAlign<uint64_t>(k, BLOCK_CUBE);
    p.dstNzNStride = 1;
    p.dstNzMatrixStride = 1;
    DataCopy(dst, src, p);
}

template <bool USE_HF32>
__aicore__ inline void MhcPreBackwardCubeCompute<USE_HF32>::CopyInA2(const LocalTensor<float>& dst,
                                                                     const LocalTensor<float>& src, uint32_t m,
                                                                     uint32_t k)
{
    LoadData2DParamsV2 p;
    p.mStartPosition = 0;
    p.kStartPosition = 0;
    p.mStep = MhcPreBackwardUtils::CeilDiv<uint64_t>(m, BLOCK_CUBE);
    p.kStep = MhcPreBackwardUtils::CeilDiv<uint64_t>(k, AuxGetC0Size<float>());
    p.srcStride = MhcPreBackwardUtils::CeilDiv<uint64_t>(m, BLOCK_CUBE);
    p.dstStride = p.mStep;
    p.ifTranspose = false;
    LoadData<float>(dst, src, p);
}

template <bool USE_HF32>
__aicore__ inline void MhcPreBackwardCubeCompute<USE_HF32>::CopyInL0Trans(const LocalTensor<float>& dst,
                                                                          const LocalTensor<float>& src, uint32_t cols,
                                                                          uint32_t k, uint32_t kL1)
{
    LoadData2DParamsV2 p;
    p.mStartPosition = 0;
    p.kStartPosition = 0;
    p.mStep = MhcPreBackwardUtils::CeilDiv<uint64_t>(k, BLOCK_CUBE);
    p.kStep = MhcPreBackwardUtils::CeilDiv<uint64_t>(cols, BLOCK_CUBE) * NUMBER_TWO;
    p.dstStride = p.kStep >> 1U;
    p.srcStride = MhcPreBackwardUtils::CeilDiv<uint64_t>(kL1, BLOCK_CUBE);
    p.ifTranspose = true;
    LoadData<float>(dst, src, p);
}

template <bool USE_HF32>
__aicore__ inline void MhcPreBackwardCubeCompute<USE_HF32>::Compute(const LocalTensor<float>& c,
                                                                    const LocalTensor<float>& a,
                                                                    const LocalTensor<float>& b, uint32_t m, uint32_t n,
                                                                    uint32_t k, bool initC, uint8_t unitFlag)
{
    MmadParams p;
    p.m = m;
    p.n = n;
    p.k = k;
    p.disableGemv = true;
    p.cmatrixInitVal = initC;
    p.cmatrixSource = false;
    // direct C0 对中间 K 使用 2，对最终累加使用 3。
    // C1 和 GM 回退路径保留原有的显式事件协议。
    p.unitFlag = unitFlag;
    Mmad(c, a, b, p);
}

template <bool USE_HF32>
__aicore__ inline void MhcPreBackwardCubeCompute<USE_HF32>::CopyOut(const GlobalTensor<float>& dst,
                                                                    const LocalTensor<float>& src, uint32_t m,
                                                                    uint32_t n, uint32_t dstStride)
{
    DataCopyCO12DstParams p;
    p.nSize = n;
    p.mSize = m;
    p.dstStride = dstStride;
    p.srcStride = MhcPreBackwardUtils::CeilAlign<uint64_t>(m, BLOCK_CUBE);
    p.quantPre = QuantMode_t::NoQuant;
    p.nz2ndEn = true;
    p.unitFlag = 0;
    SetFixpipeNz2ndFlag(1, 1, 1);
    DataCopy(dst, src, p);
}
} // namespace MhcPreBackward
#endif
