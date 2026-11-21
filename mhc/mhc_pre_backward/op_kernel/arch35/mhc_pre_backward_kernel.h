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
 * \file mhc_pre_backward_kernel.h
 * \brief
 */

#ifndef __mhc_pre_backward_KERNEL_H_
#define __mhc_pre_backward_KERNEL_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "mhc_pre_backward_utils.h"
#include "mhc_pre_backward_tiling_key.h"
#include "mhc_pre_backward_cube_compute.h"

namespace MhcPreBackward {

using namespace AscendC;
using namespace MhcPreBackwardUtils;

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
class MhcPreBackwardKernel {
    static constexpr bool STREAM_C1 = TILING_MODE != MHC_PRE_BACKWARD_DEFAULT;
    static constexpr bool STREAM_TAIL = TILING_MODE == MHC_PRE_BACKWARD_C1_STREAM_TAIL;

public:
    __aicore__ inline MhcPreBackwardKernel() = default;
    __aicore__ inline void Init(InitParams initParams);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitGlobalBuffersAndTiling(InitParams initParams);
    __aicore__ inline void InitUBAndAIVBuffers(InitParams initParams);
    __aicore__ inline void InitNdPartition();
    __aicore__ inline void ProcessBsWave();
    __aicore__ inline void ProduceBsWave(uint32_t wave);
    __aicore__ inline void ProduceBsWaveRows(uint32_t wave, uint32_t row);
    __aicore__ inline void PreProcessBsWave(LocalTensor<P>& output, uint64_t begin, uint64_t end);
    __aicore__ inline void ProcessBsWave4FP32(uint64_t bsStart);
    __aicore__ inline void PublishBsWave(uint32_t wave);
    __aicore__ inline bool CanPartitionV2Rows(uint32_t nd, uint32_t cols);
    __aicore__ inline void MergeGammaPartials();
    template <uint16_t cols>
    __aicore__ inline void PrefetchV2Rows(uint32_t nd, uint64_t row, uint32_t slot);
    template <uint16_t cols>
    __aicore__ inline void ProcessV2Rows(uint32_t nd, uint64_t bsStart, LocalTensor<P>& sum);

    // V0 流程 (向量计算)
    __aicore__ inline void ProcessV0Main();
    __aicore__ inline void ProcessV0MainLoop(uint64_t vecRunTimes, float alphaPre, float alphaPost, float alphaComb,
                                             LocalTensor<P>& sumBuf);
    __aicore__ inline void ProcessV0MainAlphaReduce(LocalTensor<P>& sumBuf, uint32_t coreId);
    __aicore__ inline void PreProcessV0Impl(LocalTensor<P>& hPreGradBuf, uint64_t runBSStart, uint64_t runBSEnd);
    __aicore__ inline void AllocV0V1Buffers(uint64_t runBSStart, uint64_t runBSEnd, V0V1Buffers<P>& buffers);
    __aicore__ inline void ProcessV0(uint64_t runBSStart, uint64_t runBSEnd, V0V1Buffers<P>& buffers, float alphaPre,
                                     float alphaPost, float alphaComb);
    __aicore__ inline void ProcessV1(uint64_t runBSStart, uint64_t runBSEnd, V0V1Buffers<P>& buffers,
                                     uint64_t vecRuntimesId, LocalTensor<P>& sumBuf);
    __aicore__ inline void AIV02Process(V0V1Buffers<P>& buffers, LocalTensor<P>& hMixGradBuf, uint64_t currentDealBsNum,
                                        float alphaPre, float alphaPost, float alphaComb);
    __aicore__ inline void AIV21Process(LocalTensor<P>& invRmsInBuf, LocalTensor<P>& invRmsUb,
                                        LocalTensor<P>& invRmsGradUb, uint32_t currentChunkSize);
    // Cube侧C0/C1流水
    __aicore__ inline void ProcessC0C1Pipeline();
    __aicore__ inline void SetCubeHf32Rounding(AscendC::HF32TransMode mode);
    __aicore__ inline void ProcessC0(uint32_t offsetND, uint32_t currentNDBlock, uint64_t offsetBS,
                                     uint32_t currentBSBlock, uint32_t buffId);
    __aicore__ inline void ProcessC1(uint32_t offsetND, uint32_t currentNDBlock, uint64_t offsetBS,
                                     uint32_t currentBSBlock, uint32_t buffId);

    // Vector侧V2流水
    __aicore__ inline void ProcessV2Pipeline();
    __aicore__ inline void ProcessV2(uint32_t offsetND, uint32_t copySizeND, uint64_t bsStart, uint64_t bsEnd,
                                     LocalTensor<P>& sumBuf, uint32_t buffId);
    __aicore__ inline void ProcessV2ChunkLoop(uint32_t offsetND, uint32_t copySizeND, uint64_t bsStart,
                                              uint64_t bsChunkStart, uint32_t currentChunkSize, uint32_t chunkNDSize,
                                              uint32_t currentN, uint32_t buffId, LocalTensor<P>& sumBuf);
    __aicore__ inline void ProcessV2ChunkLoadInvRms(uint64_t bsChunkStart, uint32_t currentChunkSize);
    __aicore__ inline void ProcessV2CompactFused(uint32_t offsetND, uint64_t bsStart, uint64_t bsChunkStart,
                                                 uint32_t currentN, uint32_t buffId, LocalTensor<P>& sumBuf);
    __aicore__ inline void CopyV2GammaOutputToC1(uint32_t copySizeND, uint64_t bsStart, uint64_t bsChunkStart,
                                                 uint32_t currentChunkSize, uint32_t buffId, bool directC1);
    __aicore__ inline void CopyV2PackedGammaToC1(uint64_t bsStart, uint64_t bsChunkStart);
    __aicore__ inline void DrainV2PackedGamma();
    __aicore__ inline void WaitV2PackedGammaSlot(uint32_t slot);
    __aicore__ inline void PrefetchV2FusedX(uint32_t offsetND, uint64_t bsChunkStart);
    __aicore__ inline void PrefetchV2FusedHInPost(uint32_t offsetND, uint64_t bsChunkStart);
    __aicore__ inline void ProcessV2ChunkLoadHInGrad(uint32_t offsetND, uint32_t copySizeND, uint64_t bsChunkStart,
                                                     uint32_t currentChunkSize, uint32_t currentN, uint64_t bsStart);
    __aicore__ inline void ProcessV2ChunkLoadXAndGamma(uint32_t offsetND, uint32_t copySizeND, uint64_t bsStart,
                                                       uint64_t bsChunkStart, uint32_t currentChunkSize,
                                                       uint32_t buffId, bool directC1);
    __aicore__ inline void ProcessV2ChunkProcessXRsGradMm(uint64_t bsStart, uint64_t bsChunkStart, uint32_t copySizeND,
                                                          uint32_t currentChunkSize, uint32_t buffId,
                                                          LocalTensor<P>& sumBuf);
    __aicore__ inline void ProcessV2ChunkOutputXGrad(uint32_t offsetND, uint32_t copySizeND, uint64_t bsChunkStart,
                                                     uint32_t currentChunkSize, uint32_t chunkNDSize);

    // V3 流程 (向量计算)
    __aicore__ inline void ProcessV3();
    __aicore__ inline void MergePhiPartials();
    __aicore__ inline void ProcessV3AlphaGrad();
    __aicore__ inline void ProcessV3BiasGrad();

    // 向量化计算函数 (VFDo*)
    __aicore__ inline void VFDoPreProcessV0(__ubuf__ P* hPreGradAddr, __ubuf__ T* xB16InAddr,
                                            __ubuf__ T* gradHInB16InAddr, uint16_t curLenN, uint32_t lenD, uint64_t i,
                                            uint32_t j, uint64_t runBSStart);
    __aicore__ inline void VFDoPreProcessV0Hf32(__ubuf__ P* dstAddr, __ubuf__ T* xB16InAddr,
                                                __ubuf__ T* gradHInB16InAddr, uint16_t curLenN, uint32_t lenD);
    __aicore__ inline void VFDoPreProcessV0Rows(__ubuf__ P* hPreGradAddr, __ubuf__ T* xB16InAddr,
                                                __ubuf__ T* gradHInB16InAddr, uint32_t rows, uint32_t lenD,
                                                uint32_t rowStart);
    __aicore__ inline void VFDoPreProcessV0RowsN4(__ubuf__ P* hPreGradAddr, __ubuf__ T* xB16InAddr,
                                                  __ubuf__ T* gradHInB16InAddr, uint32_t rows, uint32_t lenD,
                                                  uint32_t rowStart);
    __aicore__ inline void VFDoV0HPreGrad(__ubuf__ P* hPreBufS1Addr, __ubuf__ P* hPreAddr, __ubuf__ P* gradHInAddr,
                                          uint32_t totalElem);
    __aicore__ inline void VFDoV0ProcessGradHPost(__ubuf__ P* hPostIn, __ubuf__ P* gradHPostIn,
                                                  __ubuf__ P* gradHPostOut, uint32_t stepLen);
    __aicore__ inline void VFDoV1Process(__ubuf__ P* gradHMix, __ubuf__ P* gradInvRmsOut, __ubuf__ P* gradAlphaOut,
                                         __ubuf__ P* gatherIn, __ubuf__ P* hMixIn, __ubuf__ P* invRmsIn,
                                         uint32_t dealBSSize);
    __aicore__ inline void VFDoV1ProcessBiasGradForN4N6(__ubuf__ P* outBufDst, __ubuf__ P* gatherFusion,
                                                        uint32_t curBSSize);
    __aicore__ inline void VFDoV1ProcessBiasGradForN8(__ubuf__ P* outBufDst, __ubuf__ P* gatherFusion,
                                                      uint32_t curBSSize);
    __aicore__ inline void VFDoV2HInMulHPre(__ubuf__ P* xGradVec3BufAddr, __ubuf__ T* hInGradInBufAddr,
                                            __ubuf__ P* hPreUbAddr, uint32_t currentChunkSize, uint32_t copySizeND,
                                            uint32_t currentN, uint32_t bsGlobalOffset);

    template <bool hasGamma>
    __aicore__ inline void VFDoV2XCastAndMulGamma(__ubuf__ P* xRsFp32Addr, __ubuf__ P* gammaOutAddr,
                                                  __ubuf__ T* bf16InputAddr, __ubuf__ P* gammaInAddr,
                                                  uint32_t currentChunkSize, uint32_t copySizeND);
    __aicore__ inline void VFDoV2XGradWithoutGamma(__ubuf__ P* xGradVec3Addr, __ubuf__ P* xRsGradMmAddr,
                                                   __ubuf__ P* xRsFp32Addr, __ubuf__ P* invRmsAddr,
                                                   uint32_t currentChunkSize, uint32_t copySizeND);

    template <bool hasGradXPost>
    __aicore__ inline void VFDoV2AddGradXPostAndCast(__ubuf__ T* outputAddr, __ubuf__ P* xGradAddr,
                                                     __ubuf__ T* gradXPostAddr, uint32_t totalElem);

    template <bool FULL_CHUNK = true>
    __aicore__ inline void VFDoV2FullChunkGammaReduce(__ubuf__ P* xGradAddr, __ubuf__ P* c0Addr, __ubuf__ P* xAddr,
                                                      __ubuf__ P* invRmsAddr, __ubuf__ P* gammaAddr,
                                                      __ubuf__ P* sumAddr, uint32_t rows = C0_UB_ROW_STEP,
                                                      uint32_t cols = ND_BLOCK_SIZE / NUMBER_TWO,
                                                      uint32_t stride = ND_BLOCK_SIZE / NUMBER_TWO);
    template <bool hasGradXPost, bool packedC1 = false, uint16_t rows = C0_UB_ROW_STEP,
              uint16_t cols = ND_BLOCK_SIZE / NUMBER_TWO>
    __aicore__ inline void VFDoV2CompactFused(__ubuf__ T* outputAddr, __ubuf__ P* gammaOutAddr, __ubuf__ T* xAddr,
                                              __ubuf__ T* hInAddr, __ubuf__ T* postAddr, __ubuf__ P* c0Addr,
                                              __ubuf__ P* hPreAddr, __ubuf__ P* invRmsAddr, __ubuf__ P* gammaAddr,
                                              __ubuf__ P* sumAddr);

    __aicore__ inline void VFDoV3ProcessAlphaGrad(__ubuf__ P* alphaGradOut, __ubuf__ P* alphaGradIn);

    __aicore__ inline bool CanStreamC1(uint32_t currentBSBlock) const
    {
        return STREAM_C1 && mmConfigC1_.l1K == MATMUL_WRITE_OFFSET_M &&
               (currentBSBlock == BS_BLOCK_SIZE || STREAM_TAIL);
    }

    __aicore__ inline bool UseDirectC1Block(uint64_t bsStart) const
    {
        return totalLength_ > BS_BLOCK_SIZE && (recycleC1Slices_ || (bsStart / BS_BLOCK_SIZE) % DOUBLE_BUFFER == 0);
    }

    __aicore__ inline bool UsePhiPartials() const
    {
        return !USE_HF32 && CeilDiv(totalLength_, uint64_t(mmConfigC1_.l1K)) > MAX_PHI_ACCUMULATION_SLICES;
    }
    __aicore__ inline event_t GetPackedGammaEvent(uint32_t slot) const
    {
        return slot == 0U ? EVENT_ID4 : EVENT_ID5;
    }
    MhcPreBackwardCubeCompute<USE_HF32> cubeCompute_;
    MMConfig mmConfigC0_;
    MMConfig mmConfigC1_;
    uint32_t coreNum_;
    uint64_t totalLength_;
    uint64_t vecDealBSPeCore_;
    uint32_t vecCoreNum_;
    uint64_t dealStartBS_;
    uint64_t dealEndBS_;
    uint32_t usedVecCoreNum_;
    uint32_t D_;
    uint32_t N_;
    uint32_t fusionSize_;
    float hcEps_;

    uint32_t blockIdx_;
    uint32_t nD_;
    uint32_t v1UsedCubeCoreNum_;
    uint32_t cubeDealnDPeCore_;
    uint32_t dealStartND_;
    uint32_t dealEndND_;
    uint32_t vecColumnOffset_;
    bool splitRowTail_;
    bool compactC0Route_{false};
    bool directC0Chunk_{false};
    bool packedC0Read_{false};

    GlobalTensor<T> xGm_;                 // 输入 x
    GlobalTensor<P> phiGm_;               // 输入 phi
    GlobalTensor<P> alphaGm_;             // 输入 alpha
    GlobalTensor<P> gammaGm_;             // 输入 gamma
    GlobalTensor<T> gradHInGm_;           // 输入 grad_h_in
    GlobalTensor<P> gradHPostGm_;         // 输入 grad_h_post
    GlobalTensor<P> gradHResGm_;          // 输入 grad_h_res
    GlobalTensor<T> gradXPostOptionalGm_; // 可选输入 grad_x_post_optional（GM）
    GlobalTensor<P> invRmsGm_;            // 前向预计算 inv_rms
    GlobalTensor<P> hMixGm_;              // 前向预计算 h_mix
    GlobalTensor<P> hPreGm_;              // 前向预计算 h_pre
    GlobalTensor<P> hPostGm_;             // 前向输出 h_post
    GlobalTensor<P> workSpaceGm_;
    WorkspaceBuffer workspaceBuf_; // Workspace 管理接口

    GlobalTensor<T> xGradGm_;     // 输出 grad_x
    GlobalTensor<P> gradPhiGm_;   // 输出 grad_phi
    GlobalTensor<P> alphaGradGm_; // 输出 grad_alpha
    GlobalTensor<P> gradBiasGm_;  // 输出 grad_bias
    GlobalTensor<P> gammaGradGm_; // 输出 grad_gamma

    // ProcessV2 chunk循环使用的buffer
    LocalTensor<P> invRmsInBuf_;
    LocalTensor<P> invRmsGradBuf_;
    LocalTensor<T> gradHInInBuf_;
    LocalTensor<T> xInputBuf_;
    LocalTensor<P> gammaOutUb_;
    LocalTensor<P> xRsGradMmInBuf_;
    LocalTensor<T> xOutputBuf_;
    LocalTensor<T> gradXPostBuf_;

    // ProcessV2 chunk循环的buffer (由ProcessV2分配)
    LocalTensor<P> gammaIn_;
    LocalTensor<P> xRsFp32Buf_;
    LocalTensor<P> xGradVec3Buf_;
    LocalTensor<P> invRmsUb_;
    LocalTensor<P> hPreUb_;

    TPipe* pipe_;
    const MhcPreBackwardTilingData* tiling_;
    TQue<QuePosition::VECIN, 0> vecInQueue0_;
    TQue<QuePosition::VECOUT, 1> vecOutQueue1_;
    TQue<QuePosition::VECIN, 1> vecInQueue1_;
    TQue<QuePosition::VECOUT, 1> vecOutQueueSmall_;

    TBuf<TPosition::VECCALC> fp32TBuf_;
    TBuf<TPosition::VECCALC> c0ResultTBuf_;
    bool packedGammaPending_[DOUBLE_BUFFER]{false, false};
    bool packedGammaDoubleBuffer_{false};
    uint32_t packedGammaSlot_{0};
    bool fusedPrefetchX_{false};
    bool rowPartitionV2_{false};
    bool rowPartitionTile_{false};
    uint32_t rowPartitionCols_{ND_BLOCK_SIZE};
    bool fusedInputPipeline_{false};
    LocalTensor<T> fusedHInPostSlots_[DOUBLE_BUFFER];
    uint64_t fusedPrefetchXEnd_{0};

    uint32_t hPreMaxBufLen_;
    uint32_t hPostMaxBufLen_;
    uint32_t gradHResBufLen_;
    uint32_t gradHPostOffset_;
    uint32_t hMixOffset_;
    uint32_t hFusionBufLen_;
    uint32_t globalUbOffset_;
    uint32_t vecDealChunk_;
    uint32_t v0XQueueBytes_{INOUT_QUEUE_SIZE};
    uint32_t v0GradQueueBytes_{INOUT_QUEUE_SIZE};

    uint16_t eleNumPerVf_;
    bool fullWaveRoute_{false};
    uint32_t prologueWaveIndex_{0};
    float waveAlphaPre_{0};
    float waveAlphaPost_{0};
    float waveAlphaComb_{0};
    bool withGamma_;
    bool withGradXPost_;
    bool packedC0Output_;
    bool recycleC1Slices_{false};
    float scaleMean_;
};

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::Init(InitParams initParams)
{
    InitGlobalBuffersAndTiling(initParams);
    InitUBAndAIVBuffers(initParams);
    InitNdPartition();
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::InitGlobalBuffersAndTiling(
    InitParams initParams)
{
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(initParams.x));
    xGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    phiGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(initParams.phi));
    phiGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_PERSISTENT);
    alphaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(initParams.alpha));
    gradHInGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(initParams.grad_h_in));
    gradHPostGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(initParams.grad_h_post));
    gradHResGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(initParams.grad_h_res));
    invRmsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(initParams.inv_rms));
    hMixGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(initParams.h_mix));
    hPreGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(initParams.h_pre));
    hPostGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(initParams.h_post));
    xGradGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(initParams.grad_x));
    xGradGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    gradPhiGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(initParams.grad_phi));
    alphaGradGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(initParams.grad_alpha));
    gradBiasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(initParams.grad_bias));
    workSpaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(initParams.workspace));
    withGamma_ = (initParams.gamma != nullptr);
    withGradXPost_ = (initParams.grad_x_post_optional != nullptr);
    if (withGradXPost_) {
        gradXPostOptionalGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(initParams.grad_x_post_optional));
        gradXPostOptionalGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    }
    if (withGamma_) {
        gammaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(initParams.gamma));
        gammaGradGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(initParams.grad_gamma));
    }
    tiling_ = initParams.tilingData;
    // 两种计算模式共用经过资源检查的 V2 融合和直达路径。
    mmConfigC0_ = tiling_->mmConfigC0;
    mmConfigC1_ = tiling_->mmConfigC1;
    nD_ = static_cast<uint32_t>(tiling_->nD);
    D_ = static_cast<uint32_t>(tiling_->D);
    N_ = static_cast<uint32_t>(tiling_->N);
    fusionSize_ = static_cast<uint32_t>(tiling_->fusionSize);
    coreNum_ = tiling_->coreNum;
    totalLength_ = tiling_->totalLength;
    const uint64_t hPreBytes = static_cast<uint64_t>(BS_BLOCK_SIZE) * N_ * sizeof(P);
    const uint64_t v2WorkBytes = PROCESS_V2_CHUNK_SIZE * sizeof(uint32_t) + (ND_BLOCK_SIZE / NUMBER_TWO) * sizeof(P) +
                                 NUMBER_TWO * C0_UB_ROW_STEP * (ND_BLOCK_SIZE / NUMBER_TWO) * sizeof(P) +
                                 BS_BLOCK_SIZE * sizeof(P) + hPreBytes;
    // 只有所有完整 BS 分块完成后，最终 ND 尾块才可以借用 C0 区域。
    // BS 尾块可能与下一个 ND 分块对该区域的 C0 写入重叠。
    compactC0Route_ = sizeof(T) == 2U && totalLength_ >= BS_BLOCK_SIZE && totalLength_ % BS_BLOCK_SIZE == 0U &&
                      nD_ % ND_BLOCK_SIZE == 0U && mmConfigC0_.baseM == MATMUL_WRITE_OFFSET_M &&
                      mmConfigC0_.baseN == MATMUL_WRITE_OFFSET_N && v2WorkBytes <= FP32_BUF_SIZE;
    // 两个带 padding 的 65x64 分块需要在原有两个 64x64
    // 中间缓冲之外额外占用 512 字节；如果没有余量则保留现有 compact 路径。
    const uint64_t packedGammaWorkBytes = v2WorkBytes + NUMBER_TWO * (ND_BLOCK_SIZE / NUMBER_TWO) * sizeof(P);
    packedGammaDoubleBuffer_ = compactC0Route_ && packedGammaWorkBytes <= FP32_BUF_SIZE;
    packedC0Output_ = totalLength_ % BS_BLOCK_SIZE == 0U && nD_ % ND_BLOCK_SIZE == 0U;
    // L1 生命周期取决于 C1 读取调度，与 VF 融合相互独立。
    recycleC1Slices_ = compactC0Route_ && totalLength_ > BS_BLOCK_SIZE && mmConfigC1_.l1K == MATMUL_WRITE_OFFSET_M &&
                       fusionSize_ <= mmConfigC1_.baseM;
    hcEps_ = tiling_->hcEps;
    vecCoreNum_ = tiling_->vecCoreNum;
    scaleMean_ = (nD_ > 0) ? (1.0f / nD_) : 0.0f;
    eleNumPerVf_ = GetVRegSize() / sizeof(P);
    workspaceBuf_.Init(totalLength_, fusionSize_, vecCoreNum_, coreNum_);
    blockIdx_ = GetBlockIdx();
    // 共享 FP32/HF32 的容量与路径检查，不再维护形状列表。
    const uint32_t producerRows = vecCoreNum_ > 0U ? BS_BLOCK_SIZE / vecCoreNum_ : 0U;
    const uint64_t producerScratch =
        static_cast<uint64_t>(producerRows) * fusionSize_ * sizeof(P) * 4U +
        MhcPreBackwardUtils::CeilAlign(producerRows * N_, uint32_t(CEIL_ALIGN_DEFAULT)) * sizeof(P) +
        MhcPreBackwardUtils::CeilAlign(uint32_t(fusionSize_ * sizeof(P)), uint32_t(CEIL_ALIGN_DEFAULT));
    // 128 列布局：前 1024 字节预留给归约及 gamma，两块 NZ 工作区各有 32 行加一行 padding。
    constexpr uint32_t ROW_PARTITION_PREFIX_BYTES = 1024U;
    constexpr uint32_t ROW_PARTITION_PADDED_ROWS = 33U;
    constexpr uint32_t ROW_PARTITION_COLS = 128U;
    const uint64_t rowPartitionBytes = ROW_PARTITION_PREFIX_BYTES +
                                       2U * ROW_PARTITION_PADDED_ROWS * ROW_PARTITION_COLS * sizeof(P) +
                                       BS_BLOCK_SIZE * (N_ + 1U) * sizeof(P);
    const bool rowPartitionFits = rowPartitionBytes <= FP32_BUF_SIZE;
    rowPartitionV2_ = recycleC1Slices_ && withGamma_ && D_ % (ND_BLOCK_SIZE / NUMBER_TWO) == 0U && rowPartitionFits;
    constexpr uint32_t WAVE_INPUT_ROW_LIMIT_BYTES = 16U * KIBIBYTE;
    constexpr uint32_t WAVE_OUTPUT_TILE_LIMIT_BYTES = 8U * KIBIBYTE;
    constexpr uint32_t WAVE_SCRATCH_RESERVE_BYTES = 16U * KIBIBYTE;
    fullWaveRoute_ = !USE_HF32 && recycleC1Slices_ && withGamma_ && producerRows > 0U &&
                     BS_BLOCK_SIZE % vecCoreNum_ == 0U && D_ % (ND_BLOCK_SIZE / NUMBER_TWO) == 0U &&
                     D_ * sizeof(T) <= WAVE_INPUT_ROW_LIMIT_BYTES &&
                     producerRows * fusionSize_ * sizeof(P) <= WAVE_OUTPUT_TILE_LIMIT_BYTES &&
                     producerScratch <= FP32_BUF_SIZE - WAVE_SCRATCH_RESERVE_BYTES;
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::InitUBAndAIVBuffers(InitParams initParams)
{
    vecDealChunk_ = N_ > LARGE_N_THRESHOLD ? VEC_DEAL_CHUNK_LARGE_N : VEC_DEAL_CHUNK_SMALL_N;
    if (fullWaveRoute_) {
        // 保持打包后的 hPre/hPost 平面之间满足 32 字节边界。
        vecDealChunk_ = 1U;
        while ((vecDealChunk_ * N_ * sizeof(P)) % 32U != 0U) {
            vecDealChunk_ *= 2U;
        }
        if (vecDealChunk_ * 2U <= BS_BLOCK_SIZE / vecCoreNum_) {
            vecDealChunk_ *= 2U;
        }
    }

    hPreMaxBufLen_ = vecDealChunk_ * N_;
    hPostMaxBufLen_ = hPreMaxBufLen_;
    hMixOffset_ = (hPreMaxBufLen_ + hPostMaxBufLen_) * sizeof(P);
    gradHPostOffset_ = hPreMaxBufLen_ * sizeof(P);
    gradHResBufLen_ = vecDealChunk_ * N_ * N_;
    hFusionBufLen_ = hPreMaxBufLen_ + hPostMaxBufLen_ + gradHResBufLen_;
    pipe_ = initParams.tPipeIn;
    if ASCEND_IS_AIC {
        cubeCompute_.Init(pipe_, tiling_, recycleC1Slices_, rowPartitionV2_);
    }
    if ASCEND_IS_AIV {
        v0XQueueBytes_ = fullWaveRoute_ ? 16U * KIBIBYTE : INOUT_QUEUE_SIZE;
        v0GradQueueBytes_ = fullWaveRoute_ ? 8U * KIBIBYTE : INOUT_QUEUE_SIZE;
        const uint32_t outputSlot = fullWaveRoute_ ? 8U * KIBIBYTE : INOUT_QUEUE_SIZE;
        pipe_->InitBuffer(vecInQueue0_, DOUBLE_BUFFER, v0XQueueBytes_);
        pipe_->InitBuffer(vecOutQueue1_, DOUBLE_BUFFER, outputSlot);
        pipe_->InitBuffer(vecInQueue1_, DOUBLE_BUFFER, v0GradQueueBytes_);
        if (fullWaveRoute_) {
            // 队列占用 0~64 KiB，C0 结果占用 64~192 KiB，small/TBuf 占用 192~248 KiB。
            // 沿用原 compact V2 布局，适用于所有 wave。
            pipe_->InitBuffer(c0ResultTBuf_, BS_BLOCK_SIZE * (ND_BLOCK_SIZE / NUMBER_TWO) * sizeof(P));
        }
        pipe_->InitBuffer(vecOutQueueSmall_, 1, SMALL_QUEUE_SIZE);
        pipe_->InitBuffer(fp32TBuf_, FP32_BUF_SIZE);
        globalUbOffset_ = 0;
        vecDealBSPeCore_ = totalLength_ / vecCoreNum_;
        if (vecDealBSPeCore_ == 0) {
            vecDealBSPeCore_ = totalLength_;
            usedVecCoreNum_ = 1;
        } else {
            vecDealBSPeCore_ = MhcPreBackwardUtils::CeilAlign(vecDealBSPeCore_, static_cast<uint64_t>(CEIL_ALIGN_16));
            uint64_t usedVecCoreNum = CeilDiv(totalLength_, vecDealBSPeCore_);
            usedVecCoreNum_ = usedVecCoreNum > vecCoreNum_ ? vecCoreNum_ : static_cast<uint32_t>(usedVecCoreNum);
        }
        dealStartBS_ = vecDealBSPeCore_ * blockIdx_;
        dealEndBS_ = dealStartBS_ + vecDealBSPeCore_;
        if (dealEndBS_ > totalLength_ || blockIdx_ == usedVecCoreNum_ - 1) {
            dealEndBS_ = totalLength_;
        }
    }
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::InitNdPartition()
{
    v1UsedCubeCoreNum_ = 0;
    cubeDealnDPeCore_ = nD_ / GetBlockNum();
    if (cubeDealnDPeCore_ == 0) {
        cubeDealnDPeCore_ = nD_;
        v1UsedCubeCoreNum_ = 1;
    } else {
        // 当每个 AIC 拥有多个 128 列 ND 分块时，按完整分块分配
        // 可能导致部分 AIC 空闲。该负载均衡规则不依赖 N。
        // 允许每个 AIC 处理一个最终 64 列分块，以平衡所有核的工作量。
        // 在 FP32 下，单 BS 的 N4/D512 场景只有 16 个完整分块；将其拆分
        // 到全部 32 个 AIC。HF32 保留原有的完整分块路径。
        const bool hasMultipleNdTilesPerCore = nD_ > GetBlockNum() * ND_BLOCK_SIZE;
        const bool useHalfTileForSingleBs = N_ == 4 && nD_ == GetBlockNum() * (ND_BLOCK_SIZE / NUMBER_TWO) &&
                                            totalLength_ == BS_BLOCK_SIZE && !USE_HF32;
        const bool useHalfNdTile =
            totalLength_ >= BS_BLOCK_SIZE && (hasMultipleNdTilesPerCore || useHalfTileForSingleBs);
        uint32_t ndCoreAlign = useHalfNdTile ? ND_BLOCK_SIZE / NUMBER_TWO : CEIL_ALIGN_128;
        cubeDealnDPeCore_ = MhcPreBackwardUtils::CeilAlign(cubeDealnDPeCore_, ndCoreAlign);
        v1UsedCubeCoreNum_ = CeilDiv(nD_, cubeDealnDPeCore_);
    }
    uint32_t cubeId = blockIdx_;
    if ASCEND_IS_AIV {
        cubeId = uint32_t(blockIdx_ / VEC_DEAL_VECIDX_DIV);
    }
    dealStartND_ = cubeDealnDPeCore_ * cubeId;
    dealEndND_ = dealStartND_ + cubeDealnDPeCore_;
    if (dealEndND_ > nD_) {
        dealEndND_ = nD_;
    }
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::Process()
{
    if (fullWaveRoute_) {
        ProcessBsWave();
        return;
    }
    if ASCEND_IS_AIV {
        // 对每个 head 和每种分块模式，先累加 D 通道再做归约。
        ProcessV0Main();
        // 原有的全核同步也会保证成对 AIV 在尾部初始化后进行的
        // Gamma 原子归约按序执行。
        if (rowPartitionV2_ && withGamma_ && blockIdx_ < NUMBER_TWO * v1UsedCubeCoreNum_ &&
            blockIdx_ % NUMBER_TWO == 0U) {
            auto zero = vecOutQueueSmall_.AllocTensor<P>();
            Duplicate(zero, static_cast<P>(0), ND_BLOCK_SIZE);
            vecOutQueueSmall_.EnQue(zero);
            zero = vecOutQueueSmall_.DeQue<P>();
            DataCopyExtParams params;
            params.blockCount = 1;
            params.blockLen = ND_BLOCK_SIZE * sizeof(P);
            params.srcStride = 0;
            params.dstStride = 0;
            // 按全局完整块分配清零任务，避免两个 AIV 分写同一缓存行。
            const uint32_t startND = MhcPreBackwardUtils::CeilAlign(dealStartND_, uint32_t(ND_BLOCK_SIZE));
            for (uint32_t nd = startND; nd < dealEndND_; nd += ND_BLOCK_SIZE) {
                params.blockLen = Min(uint32_t(ND_BLOCK_SIZE), nD_ - nd) * sizeof(P);
                DataCopyPad(gammaGradGm_[nd], zero, params);
            }
            PipeBarrier<PIPE_MTE3>();
            vecOutQueueSmall_.FreeTensor(zero);
        }
        bool splitTailRoute = compactC0Route_ && blockIdx_ < NUMBER_TWO * v1UsedCubeCoreNum_ &&
                              (dealEndND_ - dealStartND_) % ND_BLOCK_SIZE == ND_BLOCK_SIZE / NUMBER_TWO;
        if (splitTailRoute && withGamma_ && !rowPartitionV2_ && blockIdx_ % VEC_CORE_VECIDX_MOD == 0) {
            LocalTensor<P> zeroBuf = vecOutQueueSmall_.AllocTensor<P>();
            Duplicate(zeroBuf, static_cast<P>(0), ND_BLOCK_SIZE / NUMBER_TWO);
            vecOutQueueSmall_.EnQue(zeroBuf);
            zeroBuf = vecOutQueueSmall_.DeQue<P>();
            DataCopyParams zeroParams;
            zeroParams.blockCount = 1;
            zeroParams.blockLen = ND_BLOCK_SIZE / NUMBER_TWO * sizeof(P);
            zeroParams.srcStride = 0;
            zeroParams.dstStride = 0;
            DataCopyPad(gammaGradGm_[dealEndND_ - ND_BLOCK_SIZE / NUMBER_TWO], zeroBuf, zeroParams);
            PipeBarrier<PIPE_MTE3>();
            vecOutQueueSmall_.FreeTensor(zeroBuf);
        }
    }

    SyncAll<false>();

    if ASCEND_IS_AIC {
        ProcessC0C1Pipeline();
    }

    if ASCEND_IS_AIV {
        ProcessV2Pipeline();
    }

    if constexpr (!USE_HF32) {
        if (rowPartitionV2_ || UsePhiPartials()) {
            PipeBarrier<PIPE_ALL>();
            SyncAll<false>();
            if ASCEND_IS_AIV {
                if (UsePhiPartials()) {
                    MergePhiPartials();
                }
                if (rowPartitionV2_) {
                    MergeGammaPartials();
                }
            }
        }
    }

    if ASCEND_IS_AIV {
        if (GetBlockIdx() == (GetBlockNum() - 1)) {
            ProcessV3();
        }
    }
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::VFDoPreProcessV0(
    __ubuf__ P* hPreGradAddr, __ubuf__ T* xB16InAddr, __ubuf__ T* gradHInB16InAddr, uint16_t curLenN, uint32_t lenD,
    uint64_t i, uint32_t j, uint64_t runBSStart)
{
    uint16_t eleNumPerVf = 64;
    uint16_t dLoopCnt = (lenD + eleNumPerVf - 1) / eleNumPerVf;
    __VEC_SCOPE__
    {
        Reg::RegTensor<P> sumReg;
        uint32_t offset = static_cast<uint32_t>((i - runBSStart) * N_ + j);
        for (uint16_t offsetN = 0; offsetN < curLenN; offsetN++) { // x [curLenN, lenD],  hinGrad [1, lenD]
            Reg::Duplicate(sumReg, 0.0f);
            uint32_t curLenD = lenD;
            // x[1, lenD]  hinGrad [1, lenD]   res[1, lenD]
            Reg::RegTensor<T> xInB16Reg, gradHInB16InReg;
            Reg::RegTensor<P> xFp32Reg, gradHInFp32Reg, mulReg, tmpSumPerVfReg;
            for (uint16_t vfBlockIdx = 0; vfBlockIdx < dLoopCnt; vfBlockIdx++) {
                Reg::MaskReg mask = Reg::UpdateMask<P>(curLenD);

                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(
                    xInB16Reg, xB16InAddr + offsetN * lenD + vfBlockIdx * eleNumPerVf);
                Reg::Cast<float, T, ctHalf2Fp32Zero>(xFp32Reg, xInB16Reg, mask);

                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(gradHInB16InReg,
                                                                  gradHInB16InAddr + vfBlockIdx * eleNumPerVf);
                Reg::Cast<float, T, ctHalf2Fp32Zero>(gradHInFp32Reg, gradHInB16InReg, mask);

                Reg::Mul(mulReg, xFp32Reg, gradHInFp32Reg, mask); // mulReg[1,lenD] 0-63
                Reg::Reduce<Reg::ReduceType::SUM>(tmpSumPerVfReg, mulReg,
                                                  mask); // 每个vfBlockIdx循环的临时求和，每64个元素的和
                Reg::Add(sumReg, sumReg, tmpSumPerVfReg, mask); // D方向上的reduceSum
            }
            uint32_t bufIdx = offset + offsetN;
            Reg::Store(hPreGradAddr + bufIdx, sumReg, 1);
        }
    }
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::VFDoPreProcessV0Hf32(
    __ubuf__ P* dstAddr, __ubuf__ T* xB16InAddr, __ubuf__ T* gradHInB16InAddr, uint16_t curLenN, uint32_t lenD)
{
    constexpr uint16_t elementsPerVector = 64U;
    uint16_t dLoopCnt = (lenD + elementsPerVector - 1U) / elementsPerVector;
    uint16_t pairCount = curLenN / NUMBER_TWO;
    __VEC_SCOPE__
    {
        Reg::RegTensor<P> sum0, sum1, sumReg, xReg, gradReg;
        Reg::RegTensor<T> inputReg;
        // 两个独立累加器复用每次 grad_h_in 的加载和类型转换。
        for (uint16_t pair = 0; pair < pairCount; ++pair) {
            Reg::Duplicate(sum0, 0.0f);
            Reg::Duplicate(sum1, 0.0f);
            uint32_t remainingD = lenD;
            uint32_t head = pair * NUMBER_TWO;
            for (uint16_t dBlock = 0; dBlock < dLoopCnt; ++dBlock) {
                Reg::MaskReg mask = Reg::UpdateMask<P>(remainingD);
                uint32_t dOffset = dBlock * elementsPerVector;
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(inputReg, gradHInB16InAddr + dOffset);
                Reg::Cast<float, T, ctHalf2Fp32Zero>(gradReg, inputReg, mask);
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(inputReg, xB16InAddr + head * lenD + dOffset);
                Reg::Cast<float, T, ctHalf2Fp32Zero>(xReg, inputReg, mask);
                Reg::MulAddDst(sum0, xReg, gradReg, mask);
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(inputReg, xB16InAddr + (head + 1U) * lenD + dOffset);
                Reg::Cast<float, T, ctHalf2Fp32Zero>(xReg, inputReg, mask);
                Reg::MulAddDst(sum1, xReg, gradReg, mask);
            }
            uint32_t reduceCount = elementsPerVector;
            Reg::MaskReg reduceMask = Reg::UpdateMask<P>(reduceCount);
            Reg::Reduce<Reg::ReduceType::SUM>(sumReg, sum0, reduceMask);
            Reg::Store(dstAddr + head, sumReg, 1);
            Reg::Reduce<Reg::ReduceType::SUM>(sumReg, sum1, reduceMask);
            Reg::Store(dstAddr + head + 1U, sumReg, 1);
        }
        // 剩余 head 数为零或一时，避免读取已暂存 UB 分组之外的数据。
        for (uint16_t head = pairCount * NUMBER_TWO; head < curLenN; ++head) {
            Reg::Duplicate(sum0, 0.0f);
            uint32_t remainingD = lenD;
            for (uint16_t dBlock = 0; dBlock < dLoopCnt; ++dBlock) {
                Reg::MaskReg mask = Reg::UpdateMask<P>(remainingD);
                uint32_t dOffset = dBlock * elementsPerVector;
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(inputReg, gradHInB16InAddr + dOffset);
                Reg::Cast<float, T, ctHalf2Fp32Zero>(gradReg, inputReg, mask);
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(inputReg, xB16InAddr + head * lenD + dOffset);
                Reg::Cast<float, T, ctHalf2Fp32Zero>(xReg, inputReg, mask);
                Reg::MulAddDst(sum0, xReg, gradReg, mask);
            }
            uint32_t reduceCount = elementsPerVector;
            Reg::MaskReg reduceMask = Reg::UpdateMask<P>(reduceCount);
            Reg::Reduce<Reg::ReduceType::SUM>(sumReg, sum0, reduceMask);
            Reg::Store(dstAddr + head, sumReg, 1);
        }
    }
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::VFDoPreProcessV0Rows(
    __ubuf__ P* hPreGradAddr, __ubuf__ T* xB16InAddr, __ubuf__ T* gradHInB16InAddr, uint32_t rows, uint32_t lenD,
    uint32_t rowStart)
{
    constexpr uint16_t elementsPerVector = 64U;
    uint16_t dLoopCnt = CeilDiv(lenD, static_cast<uint32_t>(elementsPerVector));
    __VEC_SCOPE__
    {
        Reg::RegTensor<P> sumReg, sumVec;
        for (uint16_t row = 0; row < rows; ++row) {
            uint32_t xRowOffset = row * N_ * lenD;
            uint32_t gradRowOffset = row * lenD;
            for (uint16_t head = 0; head < N_; ++head) {
                Reg::Duplicate(sumVec, 0.0f);

                uint32_t curLenD = lenD;
                Reg::RegTensor<T> xInB16Reg, gradHInB16InReg;
                Reg::RegTensor<P> xFp32Reg, gradHInFp32Reg;
                for (uint16_t vfBlockIdx = 0; vfBlockIdx < dLoopCnt; ++vfBlockIdx) {
                    Reg::MaskReg mask = Reg::UpdateMask<P>(curLenD);
                    uint32_t dOffset = vfBlockIdx * elementsPerVector;
                    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(xInB16Reg,
                                                                      xB16InAddr + xRowOffset + head * lenD + dOffset);
                    Reg::Cast<float, T, ctHalf2Fp32Zero>(xFp32Reg, xInB16Reg, mask);
                    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(gradHInB16InReg,
                                                                      gradHInB16InAddr + gradRowOffset + dOffset);
                    Reg::Cast<float, T, ctHalf2Fp32Zero>(gradHInFp32Reg, gradHInB16InReg, mask);

                    Reg::MulAddDst(sumVec, xFp32Reg, gradHInFp32Reg, mask);
                }

                uint32_t reduceCount = elementsPerVector;
                Reg::MaskReg reduceMask = Reg::UpdateMask<P>(reduceCount);
                Reg::Reduce<Reg::ReduceType::SUM>(sumReg, sumVec, reduceMask);

                Reg::Store(hPreGradAddr + (rowStart + row) * N_ + head, sumReg, 1);
            }
        }
    }
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::VFDoPreProcessV0RowsN4(
    __ubuf__ P* hPreGradAddr, __ubuf__ T* xB16InAddr, __ubuf__ T* gradHInB16InAddr, uint32_t rows, uint32_t lenD,
    uint32_t rowStart)
{
    if constexpr (USE_HF32) {
        constexpr uint16_t elementsPerVector = 64U;
        uint16_t dLoopCnt = CeilDiv(lenD, static_cast<uint32_t>(elementsPerVector));
        __VEC_SCOPE__
        {
            Reg::RegTensor<P> sumReg;
            for (uint16_t row = 0; row < rows; ++row) {
                for (uint16_t head = 0; head < 4U; ++head) {
                    Reg::Duplicate(sumReg, 0.0f);
                    uint32_t remainingD = lenD;
                    Reg::RegTensor<T> xInB16Reg, gradHInB16InReg;
                    Reg::RegTensor<P> xFp32Reg, gradHInFp32Reg, productReg, blockSumReg;
                    for (uint16_t dBlock = 0; dBlock < dLoopCnt; ++dBlock) {
                        Reg::MaskReg mask = Reg::UpdateMask<P>(remainingD);
                        uint32_t dOffset = dBlock * elementsPerVector;
                        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(
                            xInB16Reg, xB16InAddr + (row * 4U + head) * lenD + dOffset);
                        Reg::Cast<float, T, ctHalf2Fp32Zero>(xFp32Reg, xInB16Reg, mask);
                        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(gradHInB16InReg,
                                                                          gradHInB16InAddr + row * lenD + dOffset);
                        Reg::Cast<float, T, ctHalf2Fp32Zero>(gradHInFp32Reg, gradHInB16InReg, mask);
                        Reg::Mul(productReg, xFp32Reg, gradHInFp32Reg, mask);
                        Reg::Reduce<Reg::ReduceType::SUM>(blockSumReg, productReg, mask);
                        Reg::Add(sumReg, sumReg, blockSumReg, mask);
                    }
                    Reg::Store(hPreGradAddr + (rowStart + row) * 4U + head, sumReg, 1);
                }
            }
        }
    } else {
        constexpr uint16_t elementsPerVector = 64U;
        uint16_t dLoopCnt = CeilDiv(lenD, static_cast<uint32_t>(elementsPerVector));
        __VEC_SCOPE__
        {
            Reg::RegTensor<P> sum0, sum1, sum2, sum3, sumReg;
            Reg::RegTensor<T> xInB16Reg, gradHInB16InReg;
            Reg::RegTensor<P> xFp32Reg, gradHInFp32Reg;
            for (uint16_t row = 0; row < rows; ++row) {
                Reg::Duplicate(sum0, 0.0f);
                Reg::Duplicate(sum1, 0.0f);
                Reg::Duplicate(sum2, 0.0f);
                Reg::Duplicate(sum3, 0.0f);
                uint32_t remainingD = lenD;
                uint32_t xRowOffset = row * 4U * lenD;
                for (uint16_t dBlock = 0; dBlock < dLoopCnt; ++dBlock) {
                    Reg::MaskReg mask = Reg::UpdateMask<P>(remainingD);
                    uint32_t dOffset = dBlock * elementsPerVector;
                    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(gradHInB16InReg,
                                                                      gradHInB16InAddr + row * lenD + dOffset);
                    Reg::Cast<float, T, ctHalf2Fp32Zero>(gradHInFp32Reg, gradHInB16InReg, mask);

                    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(xInB16Reg, xB16InAddr + xRowOffset + dOffset);
                    Reg::Cast<float, T, ctHalf2Fp32Zero>(xFp32Reg, xInB16Reg, mask);
                    Reg::MulAddDst(sum0, xFp32Reg, gradHInFp32Reg, mask);
                    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(xInB16Reg,
                                                                      xB16InAddr + xRowOffset + lenD + dOffset);
                    Reg::Cast<float, T, ctHalf2Fp32Zero>(xFp32Reg, xInB16Reg, mask);
                    Reg::MulAddDst(sum1, xFp32Reg, gradHInFp32Reg, mask);
                    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(xInB16Reg,
                                                                      xB16InAddr + xRowOffset + 2U * lenD + dOffset);
                    Reg::Cast<float, T, ctHalf2Fp32Zero>(xFp32Reg, xInB16Reg, mask);
                    Reg::MulAddDst(sum2, xFp32Reg, gradHInFp32Reg, mask);
                    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(xInB16Reg,
                                                                      xB16InAddr + xRowOffset + 3U * lenD + dOffset);
                    Reg::Cast<float, T, ctHalf2Fp32Zero>(xFp32Reg, xInB16Reg, mask);
                    Reg::MulAddDst(sum3, xFp32Reg, gradHInFp32Reg, mask);
                }
                uint32_t reduceCount = elementsPerVector;
                Reg::MaskReg reduceMask = Reg::UpdateMask<P>(reduceCount);
                uint32_t outOffset = (rowStart + row) * 4U;
                Reg::Reduce<Reg::ReduceType::SUM>(sumReg, sum0, reduceMask);
                Reg::Store(hPreGradAddr + outOffset, sumReg, 1);
                Reg::Reduce<Reg::ReduceType::SUM>(sumReg, sum1, reduceMask);
                Reg::Store(hPreGradAddr + outOffset + 1U, sumReg, 1);
                Reg::Reduce<Reg::ReduceType::SUM>(sumReg, sum2, reduceMask);
                Reg::Store(hPreGradAddr + outOffset + 2U, sumReg, 1);
                Reg::Reduce<Reg::ReduceType::SUM>(sumReg, sum3, reduceMask);
                Reg::Store(hPreGradAddr + outOffset + 3U, sumReg, 1);
            }
        }
    }
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::PreProcessV0Impl(LocalTensor<P>& hPreGradBuf,
                                                                                           uint64_t runBSStart,
                                                                                           uint64_t runBSEnd)
{
    uint32_t maxDLen = MAX_D_LEN;
    uint32_t queueCopySize = v0XQueueBytes_ / sizeof(T);
    uint32_t copySizeD = D_ <= maxDLen ? D_ : maxDLen;
    uint32_t lenN = queueCopySize / copySizeD > N_ ? N_ : queueCopySize / copySizeD;
    AscendC::LocalTensor<T> bf16InputBuf;
    DataCopyParams dataCopyParams;
    DataCopyPadParams dataCopyPadParams;
    dataCopyPadParams.isPad = false;

    // 常见 D 下完整的 x 行可以放入 VECIN0。尽可能加载相邻的
    // 多行到一个队列槽位，并复用同一组 DMA 描述符
    // 完成所有 head 归约。D 较大时仍使用下面的按 head 拆分路径。
    uint32_t rowElements = N_ * copySizeD;
    if (rowElements <= queueCopySize) {
        uint32_t rowsPerCopy = Min(queueCopySize / rowElements, v0GradQueueBytes_ / uint32_t(sizeof(T)) / copySizeD);
        for (uint64_t i = runBSStart; i < runBSEnd; i += rowsPerCopy) {
            uint32_t rows = static_cast<uint32_t>(Min(static_cast<uint64_t>(rowsPerCopy), runBSEnd - i));
            LocalTensor<T> gradHInInput = vecInQueue1_.AllocTensor<T>();
            dataCopyParams.blockCount = 1;
            dataCopyParams.blockLen = rows * copySizeD * sizeof(T);
            dataCopyParams.srcStride = 0;
            dataCopyParams.dstStride = 0;
            DataCopyPad(gradHInInput, gradHInGm_[i * copySizeD], dataCopyParams, dataCopyPadParams);
            vecInQueue1_.EnQue(gradHInInput);

            LocalTensor<T> xInput;
            vecInQueue0_.AllocTensor<T>(xInput);
            dataCopyParams.blockLen = rows * rowElements * sizeof(T);
            DataCopyPad(xInput, xGm_[i * rowElements], dataCopyParams, dataCopyPadParams);
            vecInQueue0_.EnQue(xInput);

            gradHInInput = vecInQueue1_.DeQue<T>();
            vecInQueue0_.DeQue<T>(xInput);
            __ubuf__ T* gradAddr = (__ubuf__ T*)gradHInInput.GetPhyAddr();
            __ubuf__ T* xAddr = (__ubuf__ T*)xInput.GetPhyAddr();
            __ubuf__ P* dstAddr = (__ubuf__ P*)hPreGradBuf.GetPhyAddr();

            if (N_ == 4U) {
                VFDoPreProcessV0RowsN4(dstAddr, xAddr, gradAddr, rows, copySizeD,
                                       static_cast<uint32_t>(i - runBSStart));
            } else {
                VFDoPreProcessV0Rows(dstAddr, xAddr, gradAddr, rows, copySizeD, static_cast<uint32_t>(i - runBSStart));
            }

            vecInQueue0_.FreeTensor(xInput);
            vecInQueue1_.FreeTensor(gradHInInput);
        }
        return;
    }

    for (uint64_t i = runBSStart; i < runBSEnd; i++) {
        uint64_t inputOffset = i * N_ * copySizeD;
        dataCopyParams.blockLen = copySizeD * sizeof(T);
        dataCopyParams.blockCount = 1;
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        AscendC::LocalTensor<T> gradHInInput = vecInQueue1_.AllocTensor<T>();
        DataCopyPad(gradHInInput, gradHInGm_[i * copySizeD], dataCopyParams, dataCopyPadParams);
        vecInQueue1_.EnQue(gradHInInput);
        LocalTensor<T> gradHInInputBuf = vecInQueue1_.DeQue<T>();

        for (uint32_t j = 0; j < N_; j += lenN) {
            uint32_t curLenN = j + lenN > N_ ? N_ - j : lenN;
            dataCopyParams.blockLen = copySizeD * sizeof(T);
            dataCopyParams.blockCount = curLenN;
            dataCopyParams.srcStride = 0;
            dataCopyParams.dstStride = 0;

            vecInQueue0_.AllocTensor<T>(bf16InputBuf);
            DataCopyPad(bf16InputBuf, xGm_[inputOffset], dataCopyParams, dataCopyPadParams);
            vecInQueue0_.EnQue(bf16InputBuf);
            vecInQueue0_.DeQue<T>(bf16InputBuf);

            __ubuf__ T* xB16InAddr = (__ubuf__ T*)bf16InputBuf.GetPhyAddr();
            __ubuf__ T* gradHInB16InAddr = (__ubuf__ T*)gradHInInputBuf.GetPhyAddr();
            __ubuf__ P* hPreGradAddr = (__ubuf__ P*)hPreGradBuf.GetPhyAddr();
            if constexpr (USE_HF32) {
                auto dstAddr = hPreGradAddr + static_cast<uint32_t>((i - runBSStart) * N_ + j);
                VFDoPreProcessV0Hf32(dstAddr, xB16InAddr, gradHInB16InAddr, curLenN, copySizeD);
            } else {
                VFDoPreProcessV0(hPreGradAddr, xB16InAddr, gradHInB16InAddr, curLenN, copySizeD, i, j, runBSStart);
            }

            inputOffset += copySizeD * curLenN;
            vecInQueue0_.FreeTensor(bf16InputBuf);
        }
        vecInQueue1_.FreeTensor(gradHInInputBuf);
    }
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::ProcessV0Main()
{
    if (blockIdx_ >= usedVecCoreNum_) {
        return;
    }
    uint64_t dealBsNum = dealEndBS_ - dealStartBS_;
    uint64_t vecRunTimes = CeilDiv(dealBsNum, static_cast<uint64_t>(vecDealChunk_));
    float alphaPre = fullWaveRoute_ ? waveAlphaPre_ : alphaGm_.GetValue(0);
    float alphaPost = fullWaveRoute_ ? waveAlphaPost_ : alphaGm_.GetValue(1);
    float alphaComb = fullWaveRoute_ ? waveAlphaComb_ : alphaGm_.GetValue(2);
    LocalTensor<P> sumBuf = fp32TBuf_.GetWithOffset<P>(fusionSize_, globalUbOffset_);
    globalUbOffset_ += MhcPreBackwardUtils::CeilAlign(uint32_t(fusionSize_ * sizeof(P)), uint32_t(CEIL_ALIGN_DEFAULT));
    ProcessV0MainLoop(vecRunTimes, alphaPre, alphaPost, alphaComb, sumBuf);
    // V2 scratch 偏移之前的区域会跨生产者 tick 持续有效。
    // 每个 wave 归约并发布一次 alpha，而不是每个 micro-batch 发布一次。
    if (!fullWaveRoute_ ||
        (prologueWaveIndex_ % (BS_BLOCK_SIZE / vecCoreNum_)) + vecDealChunk_ == BS_BLOCK_SIZE / vecCoreNum_) {
        ProcessV0MainAlphaReduce(sumBuf, GetBlockIdx());
    }
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::ProcessV0MainLoop(
    uint64_t vecRunTimes, float alphaPre, float alphaPost, float alphaComb, LocalTensor<P>& sumBuf)
{
    for (uint64_t cIdx = 0; cIdx < vecRunTimes; cIdx++) {
        uint64_t runBSStart = cIdx * vecDealChunk_ + dealStartBS_;
        uint64_t runBSEnd = runBSStart + vecDealChunk_;
        if (runBSEnd > dealEndBS_) {
            runBSEnd = dealEndBS_;
        }
        uint32_t currentDealBsNum = static_cast<uint32_t>(runBSEnd - runBSStart);
        V0V1Buffers<P> buffers;
        AllocV0V1Buffers(runBSStart, runBSEnd, buffers);
        if (fullWaveRoute_) {
            PreProcessBsWave(buffers.hPreGradBuf, runBSStart, runBSEnd);
        } else {
            PreProcessV0Impl(buffers.hPreGradBuf, runBSStart, runBSEnd);
        }
        ProcessV0(runBSStart, runBSEnd, buffers, alphaPre, alphaPost, alphaComb);
        ProcessV1(runBSStart, runBSEnd, buffers, cIdx + prologueWaveIndex_, sumBuf);
    }
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::ProcessV0MainAlphaReduce(
    LocalTensor<P>& sumBuf, uint32_t coreId)
{
    AscendC::LocalTensor<P> alphaOutBuf = vecOutQueue1_.AllocTensor<P>();
    Duplicate(alphaOutBuf, static_cast<P>(0), ALPHA_GRAD_PADDING);
    PipeBarrier<PIPE_V>();
    uint32_t srcShape1[] = {1, N_};
    uint32_t srcShape2[] = {1, 2 * N_};
    uint32_t srcShape3[] = {1, fusionSize_};
    ReduceSum<float, Pattern::Reduce::AR, false>(alphaOutBuf, sumBuf, srcShape1, true);
    PipeBarrier<PIPE_V>();
    Muls(sumBuf, sumBuf, ZERO, N_);
    PipeBarrier<PIPE_V>();
    ReduceSum<float, Pattern::Reduce::AR, false>(alphaOutBuf[8], sumBuf, srcShape2, true);
    PipeBarrier<PIPE_V>();
    if (N_ == 4U) {
        // N=4 时 N*N 尾块从对齐的 32 字节偏移开始。
        // 直接归约该尾块，不再先将前 2N 个元素清零。
        uint32_t tailShape[] = {1, N_ * N_};
        ReduceSum<float, Pattern::Reduce::AR, true>(alphaOutBuf[16], sumBuf[2 * N_], tailShape, true);
    } else {
        Muls(sumBuf, sumBuf, ZERO, N_ * 2);
        PipeBarrier<PIPE_V>();
        ReduceSum<float, Pattern::Reduce::AR, true>(alphaOutBuf[16], sumBuf, srcShape3, true);
    }
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
    vecOutQueue1_.EnQue(alphaOutBuf);
    alphaOutBuf = vecOutQueue1_.DeQue<P>();
    DataCopyParams bsCopyParams;
    bsCopyParams.blockCount = 1;
    bsCopyParams.blockLen = ALPHA_GRAD_PADDING * sizeof(P);
    bsCopyParams.srcStride = 0;
    bsCopyParams.dstStride = 0;
    if (fullWaveRoute_ && prologueWaveIndex_ >= BS_BLOCK_SIZE / vecCoreNum_) {
        SetAtomicAdd<float>();
    }
    DataCopyPad(workSpaceGm_[workspaceBuf_.GetAlphaGradOffset(coreId)], alphaOutBuf, bsCopyParams);
    if (fullWaveRoute_ && prologueWaveIndex_ >= BS_BLOCK_SIZE / vecCoreNum_) {
        PipeBarrier<PIPE_MTE3>();
        SetAtomicNone();
    }
    vecOutQueue1_.FreeTensor(alphaOutBuf);
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::AllocV0V1Buffers(uint64_t runBSStart,
                                                                                           uint64_t runBSEnd,
                                                                                           V0V1Buffers<P>& buffers)
{
    buffers.stepLength = static_cast<uint32_t>((runBSEnd - runBSStart) * N_);
    uint32_t hMixLength = static_cast<uint32_t>((runBSEnd - runBSStart) * N_ * N_);
    buffers.gatherLength = buffers.stepLength * 2 + hMixLength;
    uint32_t stepOffset1 =
        MhcPreBackwardUtils::CeilAlign(uint32_t(buffers.stepLength * sizeof(P)), uint32_t(CEIL_ALIGN_DEFAULT));

    // 先分配 hPreGradBuf。
    uint32_t hpreGradTotlenLen = MhcPreBackwardUtils::CeilAlign(buffers.stepLength, uint32_t(CEIL_ALIGN_DEFAULT));
    buffers.hPreGradBuf = fp32TBuf_.GetWithOffset<P>(hpreGradTotlenLen, globalUbOffset_);

    uint32_t ubOffset = globalUbOffset_ + hpreGradTotlenLen * sizeof(P);
    uint32_t fusionOffset = ubOffset;

    // V0 缓冲区。
    buffers.hPreBufS1 = fp32TBuf_.GetWithOffset<P>(hPreMaxBufLen_, ubOffset);
    buffers.hPostBufS1 = fp32TBuf_.GetWithOffset<P>(hPostMaxBufLen_, ubOffset + gradHPostOffset_);
    buffers.hResBufS1 = fp32TBuf_.GetWithOffset<P>(gradHResBufLen_, ubOffset + hMixOffset_);
    buffers.hFusionBuf = fp32TBuf_.GetWithOffset<P>(hFusionBufLen_, fusionOffset);

    ubOffset += hFusionBufLen_ * sizeof(P);
    fusionOffset = ubOffset;

    buffers.hPreBufS2 = fp32TBuf_.GetWithOffset<P>(buffers.stepLength, ubOffset);
    ubOffset += stepOffset1;
    buffers.hPostBufS2 = fp32TBuf_.GetWithOffset<P>(buffers.stepLength, ubOffset);
    buffers.gatherFusionOutBuf = fp32TBuf_.GetWithOffset<P>(hFusionBufLen_, fusionOffset);

    // V1 缓冲区。
    ubOffset += hFusionBufLen_ * sizeof(P);
    buffers.invRmsBuf = fp32TBuf_.GetWithOffset<P>(hFusionBufLen_, ubOffset);
    ubOffset += hFusionBufLen_ * sizeof(P);
    buffers.calcTmpBuf = fp32TBuf_.GetWithOffset<P>(hFusionBufLen_, ubOffset);
    ubOffset += hFusionBufLen_ * sizeof(P);
    buffers.brcbTmpBuf = fp32TBuf_.GetWithOffset<uint8_t>(FP32_BUF_SIZE - ubOffset, ubOffset);
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::VFDoV0HPreGrad(__ubuf__ P* hPreBufS1Addr,
                                                                                         __ubuf__ P* hPreAddr,
                                                                                         __ubuf__ P* gradHInAddr,
                                                                                         uint32_t totalElem)
{
    uint32_t eleNumPerVf = 64;
    uint16_t loopCnt = Ceil(totalElem, eleNumPerVf);
    uint32_t curElemCnt = totalElem;
    __VEC_SCOPE__
    {
        Reg::RegTensor<P> hPreReg, hInGradReg;
        Reg::RegTensor<P> s1Reg, s2Reg, mulReg, resReg;
        Reg::RegTensor<P> oneReg;
        Reg::MaskReg mask = Reg::CreateMask<P, Reg::MaskPattern::ALL>();
        Reg::Duplicate(oneReg, 1.0f, mask);
        for (uint16_t vfBlockIdx = 0; vfBlockIdx < loopCnt; vfBlockIdx++) {
            mask = Reg::UpdateMask<P>(curElemCnt);
            Reg::LoadAlign(hPreReg, hPreAddr + vfBlockIdx * eleNumPerVf);
            Reg::LoadAlign(hInGradReg, gradHInAddr + vfBlockIdx * eleNumPerVf);
            Reg::Adds(s1Reg, hPreReg, -hcEps_, mask);
            Reg::Sub(s2Reg, oneReg, s1Reg, mask);
            // 先乘上游梯度，再乘 sigmoid 的另一个因子，保持参考公式的舍入顺序。
            Reg::Mul(mulReg, hInGradReg, s1Reg, mask);
            Reg::Mul(resReg, mulReg, s2Reg, mask);

            Reg::StoreAlign(hPreBufS1Addr + vfBlockIdx * eleNumPerVf, resReg, mask);
        }
    }
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::ProcessV0(
    uint64_t runBSStart, uint64_t runBSEnd, V0V1Buffers<P>& buffers, float alphaPre, float alphaPost, float alphaComb)
{
    AscendC::LocalTensor<P> fp32InputBuf = vecInQueue1_.AllocTensor<P>();
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    DataCopyPadParams dataCopyPadParams;
    dataCopyPadParams.isPad = false;

    auto hMixGradBuf = buffers.calcTmpBuf;
    dataCopyParams.blockLen = buffers.stepLength * sizeof(P);

    PipeBarrier<PIPE_MTE2>();
    DataCopyPad(fp32InputBuf, hPreGm_[runBSStart * N_], dataCopyParams, dataCopyPadParams);
    vecInQueue1_.EnQue(fp32InputBuf);
    LocalTensor<P> hPre = vecInQueue1_.DeQue<P>();

    __ubuf__ P* hPreAddr = (__ubuf__ P*)hPre.GetPhyAddr();
    __ubuf__ P* gradHInAddr = (__ubuf__ P*)buffers.hPreGradBuf.GetPhyAddr();
    __ubuf__ P* hPreBufS1Addr = (__ubuf__ P*)buffers.hPreBufS1.GetPhyAddr();
    VFDoV0HPreGrad(hPreBufS1Addr, hPreAddr, gradHInAddr, buffers.stepLength);
    vecInQueue1_.FreeTensor(hPre);

    fp32InputBuf = vecInQueue1_.AllocTensor<P>();
    DataCopyPad(fp32InputBuf, hPostGm_[runBSStart * N_], dataCopyParams, dataCopyPadParams);
    vecInQueue1_.EnQue(fp32InputBuf);
    AscendC::LocalTensor<P> gradHPost;
    vecInQueue0_.AllocTensor<P>(gradHPost);
    DataCopyPad(gradHPost, gradHPostGm_[runBSStart * N_], dataCopyParams, dataCopyPadParams);
    vecInQueue0_.EnQue(gradHPost);

    LocalTensor<P> hPost = vecInQueue1_.DeQue<P>();
    vecInQueue0_.DeQue<P>(gradHPost);

    AscendC::LocalTensor<P> gradHResBuf;
    vecInQueue0_.AllocTensor<P>(gradHResBuf);
    dataCopyParams.blockLen = buffers.stepLength * N_ * sizeof(P);
    DataCopyPad(gradHResBuf, gradHResGm_[runBSStart * N_ * N_], dataCopyParams, dataCopyPadParams);
    vecInQueue0_.EnQue(gradHResBuf);

    VFDoV0ProcessGradHPost((__ubuf__ P*)hPost.GetPhyAddr(), (__ubuf__ P*)gradHPost.GetPhyAddr(),
                           (__ubuf__ P*)buffers.hPostBufS1.GetPhyAddr(), buffers.stepLength);
    vecInQueue1_.FreeTensor(hPost);
    vecInQueue0_.FreeTensor(gradHPost);

    vecInQueue0_.DeQue<P>(gradHResBuf);

    Muls(buffers.hResBufS1, gradHResBuf, ONE, buffers.stepLength * N_);
    PipeBarrier<PIPE_V>();
    vecInQueue0_.FreeTensor(gradHResBuf);

    uint32_t currentDealBsNum = static_cast<uint32_t>(runBSEnd - runBSStart);
    AIV02Process(buffers, hMixGradBuf, currentDealBsNum, alphaPre, alphaPost, alphaComb);
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::VFDoV0ProcessGradHPost(
    __ubuf__ P* hPostIn, __ubuf__ P* gradHPostIn, __ubuf__ P* gradHPostOut, uint32_t stepLen)
{
    uint16_t loopCnt = CeilDiv(stepLen, uint32_t(eleNumPerVf_));
    uint32_t curLen = stepLen;
    __VEC_SCOPE__
    {
        Reg::MaskReg mask;
        Reg::RegTensor<P> gradHPostReg, hPostReg, tmpReg;
        for (uint16_t vfBlockIdx = 0; vfBlockIdx < loopCnt; vfBlockIdx++) {
            mask = Reg::UpdateMask<P>(curLen);
            Reg::LoadAlign(hPostReg, hPostIn + vfBlockIdx * eleNumPerVf_);
            Reg::Muls(tmpReg, hPostReg, NEG_HALF, mask);
            Reg::Adds(tmpReg, tmpReg, ONE, mask);
            Reg::LoadAlign(gradHPostReg, gradHPostIn + vfBlockIdx * eleNumPerVf_);
            // 与 grad_h_post * h_post * (1 - h_post / 2) 保持相同顺序。
            Reg::Mul(gradHPostReg, gradHPostReg, hPostReg, mask);
            Reg::Mul(gradHPostReg, gradHPostReg, tmpReg, mask);
            Reg::StoreAlign(gradHPostOut + vfBlockIdx * eleNumPerVf_, gradHPostReg, mask);
        }
    }
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::AIV02Process(V0V1Buffers<P>& buffers,
                                                                                       LocalTensor<P>& hMixGradBuf,
                                                                                       uint64_t currentDealBsNum,
                                                                                       float alphaPre, float alphaPost,
                                                                                       float alphaComb)
{
    __ubuf__ P* hPreBufAddr = (__ubuf__ P*)buffers.hPreBufS1.GetPhyAddr();
    __ubuf__ P* hPostBufAddr = (__ubuf__ P*)buffers.hPostBufS1.GetPhyAddr();
    __ubuf__ P* hResBufAddr = (__ubuf__ P*)buffers.hResBufS1.GetPhyAddr();
    __ubuf__ P* gatherFusionOutBufAddr = (__ubuf__ P*)buffers.gatherFusionOutBuf.GetPhyAddr();
    __ubuf__ P* hMixGradBufAddr = (__ubuf__ P*)hMixGradBuf.GetPhyAddr();

    uint32_t blockLenPre = N_;
    uint32_t blockLenPost = N_;
    uint32_t blockLenComb = N_ * N_;
    uint32_t totalBlockLen = blockLenPre + blockLenPost + blockLenComb;

    __VEC_SCOPE__
    {
        Reg::RegTensor<P> gatherReg;
        Reg::RegTensor<P> hMixGradReg;
        Reg::RegTensor<P> alphaPreReg;
        Reg::RegTensor<P> alphaPostReg;
        Reg::RegTensor<P> alphaCombReg;
        Reg::MaskReg maskPre = Reg::CreateMask<P, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskPost = Reg::CreateMask<P, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskComb = Reg::CreateMask<P, Reg::MaskPattern::ALL>();
        Reg::Duplicate<P>(alphaPreReg, alphaPre, maskPre);
        Reg::Duplicate<P>(alphaPostReg, alphaPost, maskPost);
        Reg::Duplicate<P>(alphaCombReg, alphaComb, maskComb);

        for (uint16_t bsIdx = 0; bsIdx < static_cast<uint16_t>(currentDealBsNum); bsIdx++) {
            uint32_t curLenPre = blockLenPre;
            uint32_t curLenPost = blockLenPost;
            uint32_t curLenComb = blockLenComb;
            uint32_t bsFusionOffset = bsIdx * totalBlockLen;

            uint32_t srcOffset = bsIdx * blockLenPre;
            uint32_t dstOffset = bsFusionOffset;
            maskPre = Reg::UpdateMask<P>(curLenPre);
            Reg::Load<P>(gatherReg, hPreBufAddr + srcOffset);
            Reg::Store<P>(gatherFusionOutBufAddr + dstOffset, gatherReg);
            Reg::Mul(hMixGradReg, gatherReg, alphaPreReg, maskPre);
            Reg::Store<P>(hMixGradBufAddr + dstOffset, hMixGradReg);

            uint32_t srcOffset1 = bsIdx * blockLenPre;
            uint32_t dstOffset1 = bsFusionOffset;
            maskPost = Reg::UpdateMask<P>(curLenPost);
            Reg::Load<P>(gatherReg, hPostBufAddr + srcOffset1);
            Reg::Store<P>(gatherFusionOutBufAddr + dstOffset1 + blockLenPre, gatherReg);
            Reg::Mul(hMixGradReg, gatherReg, alphaPostReg, maskPost);
            Reg::Store<P>(hMixGradBufAddr + dstOffset1 + blockLenPre, hMixGradReg);

            uint32_t srcOffset2 = bsIdx * blockLenComb;
            uint32_t dstOffset2 = bsFusionOffset + blockLenPre + blockLenPost;
            maskComb = Reg::UpdateMask<P>(curLenComb);
            Reg::Load<P>(gatherReg, hResBufAddr + srcOffset2);
            Reg::Store<P>(gatherFusionOutBufAddr + dstOffset2, gatherReg);
            Reg::Mul(hMixGradReg, gatherReg, alphaCombReg, maskComb);
            Reg::Store<P>(hMixGradBufAddr + dstOffset2, hMixGradReg);
        }
    }
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::ProcessV1(
    uint64_t runBSStart, uint64_t runBSEnd, V0V1Buffers<P>& buffers, uint64_t vecRuntimesId, LocalTensor<P>& sumBuf)
{
    uint32_t curBSSize = static_cast<uint32_t>(runBSEnd - runBSStart);
    constexpr bool isReuse = false;
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    DataCopyPadParams dataCopyPadParams;
    dataCopyPadParams.isPad = false;

    AscendC::LocalTensor<P> fp32OutBuf = vecOutQueue1_.AllocTensor<P>();
    AscendC::LocalTensor<P> invRmsGradUb = vecOutQueueSmall_.AllocTensor<P>();
    if (N_ == 8) {
        VFDoV1ProcessBiasGradForN8((__ubuf__ P*)fp32OutBuf.GetPhyAddr(),
                                   (__ubuf__ P*)buffers.gatherFusionOutBuf.GetPhyAddr(), curBSSize);
    } else {
        VFDoV1ProcessBiasGradForN4N6((__ubuf__ P*)fp32OutBuf.GetPhyAddr(),
                                     (__ubuf__ P*)buffers.gatherFusionOutBuf.GetPhyAddr(), curBSSize);
    }
    vecOutQueue1_.EnQue(fp32OutBuf);
    LocalTensor<P> BiasOutBuf = vecOutQueue1_.DeQue<P>();

    uint32_t coreId = GetBlockIdx();
    dataCopyParams.blockLen = fusionSize_ * sizeof(P);
    if (vecRuntimesId != 0) {
        SetAtomicAdd<float>();
        DataCopyPad(workSpaceGm_[workspaceBuf_.GetBiasGradOffset(coreId)], BiasOutBuf, dataCopyParams);
        PipeBarrier<PIPE_MTE3>();
        SetAtomicNone();
    } else {
        DataCopyPad(workSpaceGm_[workspaceBuf_.GetBiasGradOffset(coreId)], BiasOutBuf, dataCopyParams);
        PipeBarrier<PIPE_MTE3>();
    }
    vecOutQueue1_.FreeTensor(BiasOutBuf);

    auto hMixGradBuf = buffers.calcTmpBuf;
    fp32OutBuf = vecOutQueue1_.AllocTensor<P>();
    dataCopyParams.blockLen = curBSSize * sizeof(P);
    AscendC::LocalTensor<P> fp32InputBuf = vecInQueue1_.AllocTensor<P>();
    DataCopyPad(fp32InputBuf, invRmsGm_[runBSStart], dataCopyParams, dataCopyPadParams);
    vecInQueue1_.EnQue(fp32InputBuf);
    LocalTensor<P> invRmsBufLocal = vecInQueue1_.DeQue<P>();

    const uint32_t xRowSumBroadCastDst[2] = {curBSSize, fusionSize_};
    const uint32_t xRowSumBroadCastSrc[2] = {curBSSize, 1};
    BroadCast<float, 2, 1>(buffers.invRmsBuf, invRmsBufLocal, xRowSumBroadCastDst, xRowSumBroadCastSrc,
                           buffers.brcbTmpBuf);
    PipeBarrier<PIPE_V>();
    vecInQueue1_.FreeTensor(invRmsBufLocal);

    PipeBarrier<PIPE_V>();
    LocalTensor<P> hMixBuf;
    vecInQueue0_.AllocTensor<P>(hMixBuf);
    dataCopyParams.blockLen = curBSSize * fusionSize_ * sizeof(P);
    DataCopyPad(hMixBuf, hMixGm_[runBSStart * fusionSize_], dataCopyParams, dataCopyPadParams);
    vecInQueue0_.EnQue(hMixBuf);
    vecInQueue0_.DeQue<P>(hMixBuf);

    PipeBarrier<PIPE_V>();
    VFDoV1Process((__ubuf__ P*)fp32OutBuf.GetPhyAddr(), (__ubuf__ P*)hMixGradBuf.GetPhyAddr(),
                  (__ubuf__ P*)buffers.hFusionBuf.GetPhyAddr(), (__ubuf__ P*)buffers.gatherFusionOutBuf.GetPhyAddr(),
                  (__ubuf__ P*)hMixBuf.GetPhyAddr(), (__ubuf__ P*)buffers.invRmsBuf.GetPhyAddr(), curBSSize);

    vecOutQueue1_.EnQue(fp32OutBuf);
    LocalTensor<P> hMixGradOutBuf = vecOutQueue1_.DeQue<P>();
    dataCopyParams.blockLen = curBSSize * fusionSize_ * sizeof(P);
    DataCopyPad(workSpaceGm_[workspaceBuf_.GetHMixGradOffset(runBSStart)], hMixGradOutBuf, dataCopyParams);
    PipeBarrier<PIPE_V>();

    uint32_t shape[] = {curBSSize, fusionSize_};
    ReduceSum<float, AscendC::Pattern::Reduce::AR, isReuse>(invRmsGradUb, hMixGradBuf, buffers.brcbTmpBuf, shape, true);
    vecOutQueueSmall_.EnQue(invRmsGradUb);
    LocalTensor<P> invRmsGradUbtBuf = vecOutQueueSmall_.DeQue<P>();

    dataCopyParams.blockLen = curBSSize * sizeof(P);
    DataCopyPad(workSpaceGm_[workspaceBuf_.GetInvRmsGradOffset(runBSStart)], invRmsGradUbtBuf, dataCopyParams);
    ReduceSum<float, AscendC::Pattern::Reduce::RA, isReuse>(hMixGradBuf, buffers.hFusionBuf, buffers.brcbTmpBuf, shape,
                                                            true);
    PipeBarrier<PIPE_V>();
    vecOutQueue1_.FreeTensor(hMixGradOutBuf);
    vecOutQueueSmall_.FreeTensor(invRmsGradUbtBuf);
    vecInQueue0_.FreeTensor(hMixBuf);

    if ((fullWaveRoute_ && vecRuntimesId % (BS_BLOCK_SIZE / vecCoreNum_) == 0U) ||
        (!fullWaveRoute_ && vecRuntimesId == 0U)) {
        Adds(sumBuf, hMixGradBuf, ZERO, fusionSize_);
    } else {
        Add(sumBuf, hMixGradBuf, sumBuf, fusionSize_);
    }
    PipeBarrier<PIPE_V>();
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::VFDoV1Process(
    __ubuf__ P* gradHMixOut, __ubuf__ P* hMixGradBuf, __ubuf__ P* gradAlphaOut, __ubuf__ P* gatherIn,
    __ubuf__ P* hMixIn, __ubuf__ P* invRmsIn, uint32_t curBSSize)
{
    uint32_t totalElem = curBSSize * fusionSize_;
    uint16_t nLoopCnt = Ceil(totalElem, eleNumPerVf_);
    uint32_t curElemCnt = totalElem;

    __VEC_SCOPE__
    {
        for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt; ++vfBlockIdx) {
            uint32_t elemOffset = vfBlockIdx * eleNumPerVf_;
            Reg::MaskReg mask = Reg::UpdateMask<P>(curElemCnt);

            Reg::RegTensor<P> h1GradReg, invRmsReg, hMixReg, gatherReg;
            Reg::RegTensor<P> hMixGradReg, invRmsGradReg;
            Reg::RegTensor<P> hMul1Reg, hMul2Reg, hMul3Reg;

            Reg::LoadAlign(h1GradReg, hMixGradBuf + elemOffset);
            Reg::LoadAlign(invRmsReg, invRmsIn + elemOffset);

            Reg::Mul(hMixGradReg, h1GradReg, invRmsReg, mask);
            Reg::StoreAlign(gradHMixOut + elemOffset, hMixGradReg, mask);

            Reg::LoadAlign(hMixReg, hMixIn + elemOffset);
            Reg::Mul(hMul1Reg, h1GradReg, hMixReg, mask);
            Reg::StoreAlign(hMixGradBuf + elemOffset, hMul1Reg, mask);

            Reg::LoadAlign(gatherReg, gatherIn + elemOffset);
            Reg::Mul(hMul2Reg, invRmsReg, hMixReg, mask);
            Reg::Mul(hMul3Reg, hMul2Reg, gatherReg, mask);
            Reg::StoreAlign(gradAlphaOut + elemOffset, hMul3Reg, mask);
        }
    }
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::VFDoV1ProcessBiasGradForN8(
    __ubuf__ P* outBufDst, __ubuf__ P* gatherFusion, uint32_t curBSSize)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<P> sumReg1, sumReg2;
        Reg::Duplicate(sumReg1, 0);
        Reg::Duplicate(sumReg2, 0);
        uint32_t dealMask1 = eleNumPerVf_;
        uint32_t dealMask2 = fusionSize_ - eleNumPerVf_;
        Reg::MaskReg mask1 = Reg::UpdateMask<P>(dealMask1);
        Reg::MaskReg mask2 = Reg::UpdateMask<P>(dealMask2);
        for (uint16_t bsIdx = 0; bsIdx < static_cast<uint16_t>(curBSSize); ++bsIdx) {
            uint32_t elemOffset1 = bsIdx * fusionSize_;
            uint32_t elemOffset2 = bsIdx * fusionSize_ + eleNumPerVf_;
            Reg::RegTensor<P> gatherReg1, gatherReg2;

            Reg::LoadAlign(gatherReg1, gatherFusion + elemOffset1);
            Reg::LoadAlign(gatherReg2, gatherFusion + elemOffset2);

            Reg::Add(sumReg1, sumReg1, gatherReg1, mask1);
            Reg::Add(sumReg2, sumReg2, gatherReg2, mask2);
        }
        Reg::StoreAlign(outBufDst, sumReg1, mask1);
        Reg::StoreAlign(outBufDst + eleNumPerVf_, sumReg2, mask2);
    }
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::VFDoV1ProcessBiasGradForN4N6(
    __ubuf__ P* outBufDst, __ubuf__ P* gatherFusion, uint32_t curBSSize)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<P> sumReg;
        Reg::Duplicate(sumReg, 0);
        uint32_t dealMask = fusionSize_;
        Reg::MaskReg mask = Reg::UpdateMask<P>(dealMask);
        for (uint16_t bsIdx = 0; bsIdx < static_cast<uint16_t>(curBSSize); ++bsIdx) {
            uint32_t elemOffset1 = bsIdx * fusionSize_;
            Reg::RegTensor<P> gatherReg;

            Reg::LoadAlign(gatherReg, gatherFusion + elemOffset1);

            Reg::Add(sumReg, sumReg, gatherReg, mask);
        }
        Reg::StoreAlign(outBufDst, sumReg, mask);
    }
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::ProcessC0C1Pipeline()
{
    if (GetBlockIdx() >= v1UsedCubeCoreNum_) {
        // Init会在所有AIC上设置缓冲区令牌，未参与计算的AIC也必须回收，避免残留事件影响后续下发。
        cubeCompute_.End();
        return;
    }
    if (recycleC1Slices_) {
        for (uint32_t slice = 0; slice < C1_L1_SLICE_COUNT; ++slice) {
            C1PublishSliceFree<PIPE_MTE1>(slice);
        }
    } else if (compactC0Route_ && totalLength_ > BS_BLOCK_SIZE) {
        C1PublishL1Free<PIPE_MTE1>();
    }
    int64_t oriHf32Mode = 0;
    int64_t oriHf32TransMode = 0;
    if ASCEND_IS_AIC {
        if constexpr (USE_HF32) {
            oriHf32Mode = AscendC::GetCtrlSpr<AscendC::HF32_MODE_BIT, AscendC::HF32_MODE_BIT>();
            oriHf32TransMode = AscendC::GetCtrlSpr<AscendC::HF32_TRANS_MODE_BIT, AscendC::HF32_TRANS_MODE_BIT>();
            AscendC::SetHF32Mode(1);
        }
    }

    uint32_t currentNDBlock = Min(ND_BLOCK_SIZE, dealEndND_ - dealStartND_);
    uint32_t currentBSBlock = static_cast<uint32_t>(Min(static_cast<uint64_t>(BS_BLOCK_SIZE), totalLength_));
    uint32_t buffId = 0;
    // 预执行首块C0，为C0/V2/C1双缓冲流水建立初始槽。
    ProcessC0(dealStartND_, currentNDBlock, 0, currentBSBlock, buffId);

    for (uint32_t offsetND = dealStartND_; offsetND < dealEndND_; offsetND += ND_BLOCK_SIZE) {
        currentNDBlock = Min(ND_BLOCK_SIZE, dealEndND_ - offsetND);

        // 先发射下一块C0，再等待当前块V2完成并执行C1。
        for (uint64_t offsetBS = 0; offsetBS < totalLength_; offsetBS += BS_BLOCK_SIZE) {
            currentBSBlock = static_cast<uint32_t>(Min(static_cast<uint64_t>(BS_BLOCK_SIZE), totalLength_ - offsetBS));

            uint64_t nextOffsetBS = offsetBS + BS_BLOCK_SIZE;
            if (nextOffsetBS < totalLength_) {
                uint32_t nextBSBlock =
                    static_cast<uint32_t>(Min(static_cast<uint64_t>(BS_BLOCK_SIZE), totalLength_ - nextOffsetBS));
                ProcessC0(offsetND, currentNDBlock, nextOffsetBS, nextBSBlock, buffId + 1);
            } else {
                uint32_t nextOffsetND = offsetND + ND_BLOCK_SIZE;
                if (nextOffsetND < dealEndND_) {
                    uint32_t nextNDBlock = Min(ND_BLOCK_SIZE, dealEndND_ - nextOffsetND);
                    nextOffsetBS = 0;
                    uint32_t nextBSBlock =
                        static_cast<uint32_t>(Min(static_cast<uint64_t>(BS_BLOCK_SIZE), totalLength_));
                    ProcessC0(nextOffsetND, nextNDBlock, nextOffsetBS, nextBSBlock, buffId + 1);
                }
            }

            ProcessC1(offsetND, currentNDBlock, offsetBS, currentBSBlock, buffId);
            buffId++;
        }
    }
    if (compactC0Route_) {
        for (uint32_t slot = 0; slot < C0UbSlotCount(); ++slot) {
            C0WaitFreeSlot<PIPE_FIX>(slot);
        }
    }
    cubeCompute_.End();
    if (compactC0Route_) {
        // 恢复模式并返回前，先清空最后一轮异步槽位等待。
        PipeBarrier<PIPE_ALL>();
    }

    if ASCEND_IS_AIC {
        if constexpr (USE_HF32) {
            AscendC::SetHF32TransMode(oriHf32TransMode);
            AscendC::SetHF32Mode(oriHf32Mode);
        }
    }
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::SetCubeHf32Rounding(
    AscendC::HF32TransMode mode)
{
    if constexpr (USE_HF32) {
        // 切换舍入前完成已发MMAD，保留L0C累加器内容。
        event_t event = static_cast<event_t>(pipe_->FetchEventID(HardEvent::M_S));
        SetFlag<HardEvent::M_S>(event);
        WaitFlag<HardEvent::M_S>(event);
        AscendC::SetHF32TransMode(mode);
    }
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::ProcessC0(
    uint32_t offsetND, uint32_t currentNDBlock, uint64_t offsetBS, uint32_t currentBSBlock, uint32_t buffIdx)
{
    SetCubeHf32Rounding(AscendC::HF32TransMode::NEAREST_ZERO);
    buffIdx = buffIdx % VEC_CORE_VECIDX_MOD;
    uint64_t writeOffset = workspaceBuf_.GetXRsGradOffset(GetBlockIdx(), buffIdx);
    bool directC0 = compactC0Route_ && currentBSBlock == BS_BLOCK_SIZE &&
                    (currentNDBlock == ND_BLOCK_SIZE || currentNDBlock == ND_BLOCK_SIZE / NUMBER_TWO);
    // 保留原有的 128 行 Cube 分块；四个 UB 槽位覆盖所有完整分块。
    cubeCompute_.ProcessC0(mmConfigC0_, workSpaceGm_[workspaceBuf_.GetHMixGradOffset(offsetBS)], phiGm_[offsetND],
                           workSpaceGm_[writeOffset], currentBSBlock, currentNDBlock, fusionSize_, nD_,
                           packedC0Output_ && currentNDBlock == ND_BLOCK_SIZE, directC0, compactC0Route_,
                           fullWaveRoute_, (offsetND - dealStartND_) / ND_BLOCK_SIZE,
                           CanPartitionV2Rows(offsetND, currentNDBlock));
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::ProcessC1(
    uint32_t offsetND, uint32_t currentNDBlock, uint64_t offsetBS, uint32_t currentBSBlock, uint32_t buffIdx)
{
    // L1K 为 128 且 ND 分块满足列宽条件时，按 BS 完整块或尾块策略逐片启动 C1。
    bool streamC1 = CanStreamC1(currentBSBlock) && currentNDBlock >= ND_BLOCK_SIZE / NUMBER_TWO;
    if (!streamC1) {
        AscendC::CrossCoreWaitFlag(CROSS_CORE_WAIT_FLAG_C1);
    }
    SetCubeHf32Rounding(AscendC::HF32TransMode::NEAREST_EVEN);
    // 两个Vector核写入同一xRs槽的前后半个ND块，C1按currentNDBlock整体读取。
    buffIdx = buffIdx % VEC_CORE_VECIDX_MOD;
    uint64_t offsetXs = workspaceBuf_.GetXRsOffset(GetBlockIdx(), buffIdx);
    const bool separatePhi = UsePhiPartials();
    auto phiOutput = gradPhiGm_[offsetND];
    if (separatePhi) {
        const uint64_t partialBase = workspaceBuf_.GetPhiPartialBaseOffset(nD_);
        phiOutput = workSpaceGm_[partialBase + (offsetBS / mmConfigC1_.l1K) * fusionSize_ * nD_ + offsetND];
    }
    bool enAtomic = !separatePhi && (offsetBS != 0);
    // 在中间经过一个 GM 分块后复用一个 L1 分块。每个
    // 分块都等待 C1 会使 V2 与 C1 串行，失去已有的 C0/C1 重叠。
    bool directC1 =
        UseDirectC1Block(offsetBS) && currentNDBlock >= ND_BLOCK_SIZE / NUMBER_TWO && currentBSBlock == BS_BLOCK_SIZE;
    cubeCompute_.ProcessC1(
        mmConfigC1_, workSpaceGm_[workspaceBuf_.GetHMixGradOffset(offsetBS)], workSpaceGm_[offsetXs], phiOutput,
        fusionSize_, currentNDBlock, currentBSBlock, fusionSize_, ND_BLOCK_SIZE, nD_, enAtomic, directC1,
        tiling_->l1UsedBytes, streamC1, recycleC1Slices_, offsetBS + currentBSBlock == totalLength_, fullWaveRoute_,
        (offsetND - dealStartND_) / ND_BLOCK_SIZE, CeilDiv(dealEndND_ - dealStartND_, uint32_t(ND_BLOCK_SIZE)),
        totalLength_ <= BS_BLOCK_SIZE, separatePhi);
    if (compactC0Route_ && directC1 && !recycleC1Slices_) {
        // 只需等待 MTE1 完成 L1 读取；MMAD/Fixpipe 可以继续执行。
        C1PublishL1Free<PIPE_MTE1>();
    }
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::ProcessV2Pipeline()
{
    // 修改队列大小前，每个本核负责的分块都必须完成完整融合路径；
    // 部分列和无 gamma 路径保留原有队列。
    fusedInputPipeline_ = compactC0Route_ && withGamma_ && D_ % (ND_BLOCK_SIZE / NUMBER_TWO) == 0U &&
                          dealStartND_ % (ND_BLOCK_SIZE / NUMBER_TWO) == 0U &&
                          (dealEndND_ - dealStartND_) % (ND_BLOCK_SIZE / NUMBER_TWO) == 0U;
    if (compactC0Route_) {
        // V0/V1 在原有 SyncAll 处完成。保留双缓冲，
        // 但按 V2 的 64x64 分块设置 BF16 输入大小，并预留完整 C0 分块。
        PipeBarrier<PIPE_ALL>();
        pipe_->Reset();
        pipe_->InitBuffer(vecInQueue0_, DOUBLE_BUFFER, (fusedInputPipeline_ ? 16U : 8U) * KIBIBYTE);
        pipe_->InitBuffer(vecOutQueue1_, DOUBLE_BUFFER, (fusedInputPipeline_ ? 8U : 16U) * KIBIBYTE);
        pipe_->InitBuffer(vecInQueue1_, DOUBLE_BUFFER, 8U * KIBIBYTE);
        pipe_->InitBuffer(c0ResultTBuf_, BS_BLOCK_SIZE * (ND_BLOCK_SIZE / NUMBER_TWO) * sizeof(P));
        pipe_->InitBuffer(vecOutQueueSmall_, 1, SMALL_QUEUE_SIZE);
        pipe_->InitBuffer(fp32TBuf_, FP32_BUF_SIZE);
    }
    if (blockIdx_ >= 2 * v1UsedCubeCoreNum_) {
        return;
    }
    if (compactC0Route_) {
        for (uint32_t slot = 0; slot < C0UbSlotCount(); ++slot) {
            CrossCoreSetFlag<4, PIPE_V>(C0UbFreeFlag(slot));
        }
    }
    // 保留原 UB 预留空间，确保后续 tensor 地址不变。
    // 原 gather-offset 表在所有 V2 路径上都没有读取方。
    globalUbOffset_ = PROCESS_V2_CHUNK_SIZE * sizeof(uint32_t); // V2部分开始分配fp32TBuf_

    // 确定处理范围 （1C2V 模式）
    uint32_t vecIdx = blockIdx_ % VEC_CORE_VECIDX_MOD;
    uint32_t buffId = 0;
    bool splitTailRoute = compactC0Route_ && (dealEndND_ - dealStartND_) % ND_BLOCK_SIZE == ND_BLOCK_SIZE / NUMBER_TWO;
    // ND方向切分：每个ND block大小为nDBlockSize_，在1C2V模式下，每个V核处理一半
    for (uint32_t offsetNDBase = dealStartND_; offsetNDBase < dealEndND_; offsetNDBase += ND_BLOCK_SIZE) {
        // 根据vecIdx选择处理ND block的前半部分(0)或后半部分(1)
        uint32_t currentNDBlock = Min(ND_BLOCK_SIZE, dealEndND_ - offsetNDBase);
        rowPartitionCols_ = currentNDBlock;
        rowPartitionTile_ = CanPartitionV2Rows(offsetNDBase, currentNDBlock);
        packedC0Read_ = packedC0Output_ && currentNDBlock == ND_BLOCK_SIZE;
        splitRowTail_ = splitTailRoute && currentNDBlock == ND_BLOCK_SIZE / NUMBER_TWO;
        vecColumnOffset_ = splitRowTail_ ? 0 : (ND_BLOCK_SIZE / NUMBER_TWO) * vecIdx;
        uint32_t offsetND = offsetNDBase + vecColumnOffset_;
        if (offsetND >= dealEndND_) {
            // 成对的 AIV 仍需消费 C0 的行就绪事件，并
            // 为 64 列尾块贡献 C1 的 1C2V 完成事件。
            for (uint64_t offsetBS = 0; offsetBS < totalLength_; offsetBS += BS_BLOCK_SIZE) {
                uint32_t currentBSBlock =
                    static_cast<uint32_t>(Min(static_cast<uint64_t>(BS_BLOCK_SIZE), totalLength_ - offsetBS));
                if (!recycleC1Slices_ && compactC0Route_ && totalLength_ > BS_BLOCK_SIZE &&
                    currentNDBlock >= ND_BLOCK_SIZE / NUMBER_TWO && currentBSBlock == BS_BLOCK_SIZE &&
                    (offsetBS / BS_BLOCK_SIZE) % DOUBLE_BUFFER == 0) {
                    // 该 AIV 没有列数据，但仍必须消费自己的 L1 token。
                    CrossCoreWaitFlag<4, PIPE_MTE3>(C1_L1_FREE_FLAG);
                }
                bool streamC1 = CanStreamC1(currentBSBlock);
                for (uint32_t row = 0; row < currentBSBlock; row += MATMUL_WRITE_OFFSET_M) {
                    if (recycleC1Slices_) {
                        CrossCoreWaitFlag<4, PIPE_MTE3>(row / MATMUL_WRITE_OFFSET_M);
                    }
                    if (compactC0Route_) {
                        CrossCoreWaitFlag<4, PIPE_MTE2>(CROSS_CORE_WAIT_FLAG_C0);
                    } else {
                        AscendC::CrossCoreWaitFlag(CROSS_CORE_WAIT_FLAG_C0);
                    }
                    if (streamC1) {
                        AscendC::CrossCoreSetFlag<CROSS_CORE_FLAG_INDEX, PIPE_MTE3>(CROSS_CORE_WAIT_FLAG_C1);
                    }
                }
                if (!streamC1) {
                    AscendC::CrossCoreSetFlag<CROSS_CORE_FLAG_INDEX, PIPE_MTE3>(CROSS_CORE_WAIT_FLAG_C1);
                }
            }
            continue;
        }
        uint32_t copySizeND = Min(ND_BLOCK_SIZE / 2, dealEndND_ - offsetND);

        directC0Chunk_ =
            compactC0Route_ && (currentNDBlock == ND_BLOCK_SIZE || currentNDBlock == ND_BLOCK_SIZE / NUMBER_TWO);
        // BS方向切分：与C0保持一致，按BS_BLOCK_SIZE切分
        LocalTensor<P> sumBuf = vecOutQueueSmall_.AllocTensor<P>();
        for (uint64_t offsetBS = 0; offsetBS < totalLength_; offsetBS += BS_BLOCK_SIZE) {
            uint32_t currentBSBlock =
                static_cast<uint32_t>(Min(static_cast<uint64_t>(BS_BLOCK_SIZE), totalLength_ - offsetBS));
            // 执行当前的V2
            ProcessV2(offsetND, copySizeND, offsetBS, offsetBS + currentBSBlock, sumBuf, buffId);
            buffId++;
            bool streamC1 = CanStreamC1(currentBSBlock) && copySizeND == ND_BLOCK_SIZE / NUMBER_TWO;
            if (!streamC1) {
                AscendC::CrossCoreSetFlag<CROSS_CORE_FLAG_INDEX, PIPE_MTE3>(CROSS_CORE_WAIT_FLAG_C1);
            }
        }
        if (withGamma_) {
            vecOutQueueSmall_.EnQue(sumBuf);
            sumBuf = vecOutQueueSmall_.DeQue<P>();
            DataCopyExtParams fp32ExtCopyParams;
            fp32ExtCopyParams.blockCount = 1;
            fp32ExtCopyParams.blockLen = (rowPartitionTile_ ? rowPartitionCols_ : copySizeND) * sizeof(P);
            fp32ExtCopyParams.srcStride = 0;
            fp32ExtCopyParams.dstStride = 0;
            if (rowPartitionTile_ && !USE_HF32) {
                DataCopyPad(workSpaceGm_[workspaceBuf_.GetGammaPartialOffset(vecIdx, offsetNDBase, nD_)], sumBuf,
                            fp32ExtCopyParams);
            } else if (splitRowTail_ || rowPartitionTile_) {
                SetAtomicAdd<float>();
                DataCopyPad(gammaGradGm_[offsetND - (rowPartitionTile_ ? vecColumnOffset_ : 0U)], sumBuf,
                            fp32ExtCopyParams);
                PipeBarrier<PIPE_MTE3>();
                SetAtomicNone();
            } else {
                DataCopyPad(gammaGradGm_[offsetND], sumBuf, fp32ExtCopyParams);
            }
        }
        vecOutQueueSmall_.FreeTensor(sumBuf);
    }
    if (recycleC1Slices_) {
        for (uint32_t slice = 0; slice < C1_L1_SLICE_COUNT; ++slice) {
            CrossCoreWaitFlag<4, PIPE_MTE3>(slice);
        }
        PipeBarrier<PIPE_ALL>();
    } else if (compactC0Route_ && totalLength_ > BS_BLOCK_SIZE) {
        // 消费最后返回的 token，或者未使用的初始 token。
        CrossCoreWaitFlag<4, PIPE_MTE3>(C1_L1_FREE_FLAG);
        PipeBarrier<PIPE_ALL>();
    }
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
template <bool hasGamma>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::VFDoV2XCastAndMulGamma(
    __ubuf__ P* xRsFp32Addr, __ubuf__ P* gammaOutAddr, __ubuf__ T* bf16InputAddr, __ubuf__ P* gammaInAddr,
    uint32_t currentChunkSize, uint32_t copySizeND)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<T> xInB16Reg;
        Reg::RegTensor<P> xFp32Reg, gammaReg, resultReg;
        if constexpr (hasGamma) {
            // copySizeND最大为ND_BLOCK_SIZE的一半，即64个元素，可完整驻留在一个向量寄存器中。
            Reg::LoadAlign(gammaReg, gammaInAddr);
        }
        for (uint16_t tIdx = 0; tIdx < (uint16_t)currentChunkSize; tIdx++) {
            // copySizeND最大为64，一个向量指令即可处理整行，无需额外的VF分块循环。
            uint32_t lenND = copySizeND;
            Reg::MaskReg mask = Reg::UpdateMask<P>(lenND);
            uint32_t offset = tIdx * copySizeND;

            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(xInB16Reg, bf16InputAddr + offset);
            Reg::Cast<float, T, ctHalf2Fp32Zero>(xFp32Reg, xInB16Reg, mask);

            Reg::StoreAlign(xRsFp32Addr + offset, xFp32Reg, mask); // 后续其他计算需要xFp32

            if constexpr (hasGamma) {
                Reg::Mul(resultReg, gammaReg, xFp32Reg, mask); // 逐元素乘：gamma * x
                Reg::StoreAlign(gammaOutAddr + offset, resultReg, mask);
            } else {
                Reg::StoreAlign(gammaOutAddr + offset, xFp32Reg, mask);
            }
        }
    }
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::VFDoV2XGradWithoutGamma(
    __ubuf__ P* xGradVec3Addr, __ubuf__ P* xRsGradMmAddr, __ubuf__ P* xRsFp32Addr, __ubuf__ P* invRmsAddr,
    uint32_t currentChunkSize, uint32_t copySizeND)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<P> xRsFp32Reg, xRsGradMmReg, xRsGradReg, xGradReg;
        Reg::RegTensor<P> invRmsReg;

        for (uint16_t tIdx = 0; tIdx < (uint16_t)currentChunkSize; tIdx++) {
            uint32_t lenND = copySizeND;
            // invRms每行只使用一个FP32标量，直接广播加载，减少标量读取和Duplicate指令。
            Reg::DataCopy<P, Reg::LoadDist::DIST_BRC_B32>(invRmsReg, invRmsAddr + tIdx);
            // copySizeND最大为64，当前T行只需一次向量处理，删除恒单次的内层循环控制。
            Reg::MaskReg mask = Reg::UpdateMask<P>(lenND);
            uint32_t offset = tIdx * copySizeND;
            Reg::LoadAlign(xRsGradMmReg, xRsGradMmAddr + offset);
            Reg::LoadAlign(xGradReg, xGradVec3Addr + offset);
            Reg::LoadAlign(xRsFp32Reg, xRsFp32Addr + offset);

            Reg::Mul(xRsGradReg, xRsFp32Reg, invRmsReg, mask);
            Reg::Add(xGradReg, xGradReg, xRsGradReg, mask);

            Reg::Add(xGradReg, xGradReg, xRsGradMmReg, mask);

            Reg::StoreAlign(xGradVec3Addr + offset, xGradReg, mask);
        }
    }
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::VFDoV2HInMulHPre(
    __ubuf__ P* xGradVec3BufAddr, __ubuf__ T* hInGradInBufAddr, __ubuf__ P* hPreUbAddr, uint32_t currentChunkSize,
    uint32_t copySizeND, uint32_t currentN, uint32_t bsGlobalOffset)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<T> hInReg;
        Reg::RegTensor<P> hInFP32Reg, hPreReg, hPreFullReg;
        Reg::MaskReg mask;
        uint32_t hPreOffset = bsGlobalOffset * N_ + currentN;
        for (uint16_t tIdx = 0; tIdx < (uint16_t)currentChunkSize; tIdx++) {
            uint32_t lenND = copySizeND;
            uint32_t hInOffset = tIdx * copySizeND;

            Reg::DataCopy<P, Reg::LoadDist::DIST_BRC_B32>(hPreFullReg, hPreUbAddr + hPreOffset);
            // copySizeND最大为64，当前T行只需一次向量处理，删除恒单次的内层循环控制。
            mask = Reg::UpdateMask<P>(lenND);
            // hIn B16 -> B32
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(hInReg, hInGradInBufAddr + hInOffset);
            Reg::Cast<float, T, ctHalf2Fp32Zero>(hInFP32Reg, hInReg, mask);
            Reg::Mul(hInFP32Reg, hInFP32Reg, hPreFullReg, mask);
            Reg::StoreAlign(xGradVec3BufAddr + hInOffset, hInFP32Reg, mask);
            hPreOffset += N_;
        }
    }
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
template <bool FULL_CHUNK>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::VFDoV2FullChunkGammaReduce(
    __ubuf__ P* xGradAddr, __ubuf__ P* c0Addr, __ubuf__ P* xAddr, __ubuf__ P* invRmsAddr, __ubuf__ P* gammaAddr,
    __ubuf__ P* sumAddr, uint32_t rows, uint32_t cols, uint32_t stride)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<P> gammaReg, sumReg, correctionReg, adjustedReg, nextReg, errorReg, c0Reg, xReg, gradReg, invReg,
            tmpReg;
        Reg::MaskReg mask = Reg::CreateMask<P, Reg::MaskPattern::ALL>();
        if constexpr (!FULL_CHUNK) {
            mask = Reg::UpdateMask<P>(cols);
        }
        // 在循环前提取所有不变的 UB 基址。尤其是提前加载
        // sumAddr，而不是在循环结束后首次引入其 VAG。
        Reg::LoadAlign(gammaReg, gammaAddr);
        Reg::LoadAlign(sumReg, sumAddr);
        Reg::LoadAlign(correctionReg, sumAddr + ND_BLOCK_SIZE / NUMBER_TWO);
        for (uint16_t row = 0; row < (FULL_CHUNK ? C0_UB_ROW_STEP : rows); ++row) {
            uint32_t offset = row * (FULL_CHUNK ? ND_BLOCK_SIZE / NUMBER_TWO : stride);
            Reg::LoadAlign(c0Reg, c0Addr + offset);
            Reg::LoadAlign(xReg, xAddr + offset);
            Reg::LoadAlign(gradReg, xGradAddr + offset);
            Reg::DataCopy<P, Reg::LoadDist::DIST_BRC_B32>(invReg, invRmsAddr + row);
            Reg::Mul(tmpReg, xReg, invReg, mask);
            Reg::Add(gradReg, gradReg, tmpReg, mask);
            Reg::Mul(tmpReg, gammaReg, c0Reg, mask);
            Reg::Add(gradReg, gradReg, tmpReg, mask);
            Reg::StoreAlign(xGradAddr + offset, gradReg, mask);
            // 保持 FP32 乘法和加法分离；只调整求和顺序。
            Reg::Mul(tmpReg, xReg, c0Reg, mask);
            Reg::Sub(adjustedReg, tmpReg, correctionReg, mask);
            Reg::Add(nextReg, sumReg, adjustedReg, mask);
            Reg::Sub(errorReg, nextReg, sumReg, mask);
            Reg::Sub(correctionReg, errorReg, adjustedReg, mask);
            Reg::Adds(sumReg, nextReg, ZERO, mask);
        }
        // 补偿项跨 VF 保存，与部分和一起参与后续归约。
        Reg::StoreAlign(sumAddr + ND_BLOCK_SIZE / NUMBER_TWO, correctionReg, mask);
        Reg::StoreAlign(sumAddr, sumReg, mask);
    }
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::ProcessV2(
    uint32_t offsetND, uint32_t copySizeND, uint64_t bsStart, uint64_t bsEnd, LocalTensor<P>& sumBuf, uint32_t buffId)
{
    if (rowPartitionTile_) {
        if (rowPartitionCols_ == ND_BLOCK_SIZE / NUMBER_TWO) {
            ProcessV2Rows<ND_BLOCK_SIZE / NUMBER_TWO>(offsetND - vecColumnOffset_, bsStart, sumBuf);
        } else {
            ProcessV2Rows<ND_BLOCK_SIZE>(offsetND - vecColumnOffset_, bsStart, sumBuf);
        }
        return;
    }
    // 初始化
    uint32_t currentN = offsetND / D_;
    // 统一使用64行chunk，使N=4时两个FP32主工作区各由32KB降为16KB，
    // 从而压缩V2的TBuf峰值，并为后续输出队列双缓冲腾出UB空间。
    uint32_t chunkSize = PROCESS_V2_CHUNK_SIZE / NUMBER_TWO;
    uint32_t currentBsSize = static_cast<uint32_t>(bsEnd - bsStart);
    buffId = buffId % VEC_CORE_VECIDX_MOD;
    if (!recycleC1Slices_ && compactC0Route_ && totalLength_ > BS_BLOCK_SIZE &&
        copySizeND == ND_BLOCK_SIZE / NUMBER_TWO && currentBsSize == BS_BLOCK_SIZE &&
        (bsStart / BS_BLOCK_SIZE) % DOUBLE_BUFFER == 0) {
        // 在 C1 上一次 MTE1 读取完成前保护共享 L1 输入。
        CrossCoreWaitFlag<4, PIPE_MTE3>(C1_L1_FREE_FLAG);
    }
    // 按最大 CHUNK 大小分配缓冲区（保存到全局变量）
    uint32_t ubOffset = globalUbOffset_; // sumBuf 偏移
    uint32_t maxChunkNDSize = chunkSize * copySizeND;
    if (withGamma_) {
        gammaIn_ = fp32TBuf_.GetWithOffset<P>(copySizeND, ubOffset);
        ubOffset =
            MhcPreBackwardUtils::CeilAlign(uint32_t(ubOffset + copySizeND * sizeof(P)), uint32_t(CEIL_ALIGN_DEFAULT));
    }

    xRsFp32Buf_ = fp32TBuf_.GetWithOffset<P>(maxChunkNDSize, ubOffset);
    ubOffset =
        MhcPreBackwardUtils::CeilAlign(uint32_t(ubOffset + maxChunkNDSize * sizeof(P)), uint32_t(CEIL_ALIGN_DEFAULT));

    xGradVec3Buf_ = fp32TBuf_.GetWithOffset<P>(maxChunkNDSize, ubOffset);
    ubOffset =
        MhcPreBackwardUtils::CeilAlign(uint32_t(ubOffset + maxChunkNDSize * sizeof(P)), uint32_t(CEIL_ALIGN_DEFAULT));

    if (packedGammaDoubleBuffer_ && withGamma_ && copySizeND == ND_BLOCK_SIZE / NUMBER_TWO) {
        ubOffset += NUMBER_TWO * (ND_BLOCK_SIZE / NUMBER_TWO) * sizeof(P);
    }

    invRmsUb_ = fp32TBuf_.GetWithOffset<P>(currentBsSize, ubOffset);
    ubOffset =
        MhcPreBackwardUtils::CeilAlign(uint32_t(ubOffset + currentBsSize * sizeof(P)), uint32_t(CEIL_ALIGN_DEFAULT));

    uint32_t hPreUbLen =
        MhcPreBackwardUtils::CeilAlign(uint32_t(currentBsSize * N_ * sizeof(P)), uint32_t(CEIL_ALIGN_DEFAULT)) /
        sizeof(P);
    hPreUb_ = fp32TBuf_.GetWithOffset<P>(hPreUbLen, ubOffset);
    ubOffset = MhcPreBackwardUtils::CeilAlign(uint32_t(ubOffset + currentBsSize * N_ * sizeof(P)),
                                              uint32_t(CEIL_ALIGN_DEFAULT));
    // 参数初始化和hPre预加载
    DataCopyPadParams dataCopyPadParams;
    dataCopyPadParams.isPad = false;

    DataCopyParams bsCopyParams;
    bsCopyParams.blockCount = 1;
    bsCopyParams.srcStride = 0;
    bsCopyParams.dstStride = 0;
    SetFlag<HardEvent::V_MTE2>(EVENT_ID3);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID3);
    if (withGamma_) {
        bsCopyParams.blockLen = copySizeND * sizeof(P);
        DataCopyPad(gammaIn_, gammaGm_[offsetND], bsCopyParams, dataCopyPadParams);
    }
    bsCopyParams.blockLen = currentBsSize * N_ * sizeof(P);
    DataCopyPad(hPreUb_, hPreGm_[bsStart * N_], bsCopyParams, dataCopyPadParams);

    // 当前BS块的invRms系数一次搬入并计算，后续各chunk按行偏移复用已有UB空间。
    ProcessV2ChunkLoadInvRms(bsStart, currentBsSize);
    // 该路径上的所有分块都使用相同的 64x64 输入布局。利用
    // input1 中另一个已有队列缓冲保存下一块 X，同时当前 VF 执行。
    fusedPrefetchX_ = compactC0Route_ && withGamma_ && copySizeND == ND_BLOCK_SIZE / NUMBER_TWO &&
                      currentBsSize % C0_UB_ROW_STEP == 0;
    uint64_t firstXChunk = bsStart;
    fusedPrefetchXEnd_ = bsEnd;
    if (splitRowTail_) {
        uint64_t middle = Min(bsEnd, bsStart + BS_BLOCK_SIZE / NUMBER_TWO);
        if (blockIdx_ % NUMBER_TWO == 0) {
            fusedPrefetchXEnd_ = middle;
        } else {
            firstXChunk = middle;
        }
    }
    fusedPrefetchX_ = fusedPrefetchX_ && firstXChunk < fusedPrefetchXEnd_;
    if (fusedPrefetchX_) {
        if (fusedInputPipeline_) {
            PrefetchV2FusedHInPost(offsetND, firstXChunk);
        }
        PrefetchV2FusedX(offsetND, firstXChunk);
    }
    // 主循环: 按chunk=64切分BS方向
    if constexpr (STREAM_C1 && !USE_HF32) {
        if (directC0Chunk_ && withGamma_ && copySizeND == ND_BLOCK_SIZE / NUMBER_TWO &&
            currentBsSize == BS_BLOCK_SIZE && mmConfigC1_.l1K == MATMUL_WRITE_OFFSET_M) {
            // 一个 C0 就绪令牌和一次 C1 发布对应两个 64 行 VF。
            // 保持原有 DMA/Vector 依赖关系和 UB 释放顺序。
            for (uint32_t wave = 0; wave < BS_BLOCK_SIZE / MATMUL_WRITE_OFFSET_M; ++wave) {
                const uint32_t row = wave * MATMUL_WRITE_OFFSET_M;
                if (recycleC1Slices_) {
                    CrossCoreWaitFlag<4, PIPE_MTE3>(wave);
                }
                CrossCoreWaitFlag<4, PIPE_V>(CROSS_CORE_WAIT_FLAG_C0);
                const bool ownsRows =
                    !splitRowTail_ || ((row < BS_BLOCK_SIZE / NUMBER_TWO) == (blockIdx_ % NUMBER_TWO == 0));
                if (ownsRows) {
                    ProcessV2CompactFused(offsetND, bsStart, bsStart + row, currentN, buffId, sumBuf);
                    ProcessV2CompactFused(offsetND, bsStart, bsStart + row + C0_UB_ROW_STEP, currentN, buffId, sumBuf);
                } else {
                    // 成对的 AIV 仍参与同一组令牌同步。
                    CrossCoreSetFlag<4, PIPE_V>(C0UbFreeFlag(wave));
                }
                AscendC::CrossCoreSetFlag<CROSS_CORE_FLAG_INDEX, PIPE_MTE3>(CROSS_CORE_WAIT_FLAG_C1);
            }
            DrainV2PackedGamma();
            return;
        }
    }
    uint32_t c0ReadyRows = directC0Chunk_ && currentBsSize == BS_BLOCK_SIZE ? C0UbTileRows() : MATMUL_WRITE_OFFSET_M;
    for (uint64_t bsChunkStart = bsStart; bsChunkStart < bsEnd; bsChunkStart += chunkSize) {
        uint64_t bsChunkEnd = Min(bsChunkStart + chunkSize, bsEnd);
        uint32_t currentChunkSize = static_cast<uint32_t>(bsChunkEnd - bsChunkStart);
        uint32_t chunkNDSize = currentChunkSize * copySizeND;

        if (recycleC1Slices_ && (bsChunkStart - bsStart) % MATMUL_WRITE_OFFSET_M == 0) {
            // 只保护即将发生的 MTE3 写入，同时保留 Vector/C0 重叠。
            CrossCoreWaitFlag<4, PIPE_MTE3>((bsChunkStart - bsStart) / MATMUL_WRITE_OFFSET_M);
        }
        // C0每完成128行发布一次；两个64行chunk共同消费一块C0结果。
        if ((bsChunkStart - bsStart) % c0ReadyRows == 0) {
            if (directC0Chunk_ && currentBsSize == BS_BLOCK_SIZE) {
                // 保留原有依赖顺序，但允许独立的
                // DMA 在 Vector 流水等待 C0 时发出。
                CrossCoreWaitFlag<4, PIPE_V>(CROSS_CORE_WAIT_FLAG_C0);
            } else if (compactC0Route_) {
                // GM 回退路径由 MTE2 消费，并在 Vector 阶段之前完成。
                CrossCoreWaitFlag<4, PIPE_MTE2>(CROSS_CORE_WAIT_FLAG_C0);
            } else {
                AscendC::CrossCoreWaitFlag(CROSS_CORE_WAIT_FLAG_C0);
            }
        }
        bool ownsRows =
            !splitRowTail_ || ((bsChunkStart - bsStart < BS_BLOCK_SIZE / NUMBER_TWO) == (blockIdx_ % NUMBER_TWO == 0));
        if (ownsRows) {
            ProcessV2ChunkLoop(offsetND, copySizeND, bsStart, bsChunkStart, currentChunkSize, chunkNDSize, currentN,
                               buffId, sumBuf);
        }
        if (!ownsRows && directC0Chunk_ && currentBsSize == BS_BLOCK_SIZE &&
            (bsChunkEnd - bsStart) % C0UbTileRows() == 0) {
            CrossCoreSetFlag<4, PIPE_V>(C0UbFreeFlag(((bsChunkStart - bsStart) / C0UbTileRows()) % C0UbSlotCount()));
        }
        // 两个 Vector 核在各自第二个 64 行分块后发布一个 K 分片。
        if (CanStreamC1(currentBsSize) && copySizeND == ND_BLOCK_SIZE / NUMBER_TWO &&
            (bsChunkEnd - bsStart) % MATMUL_WRITE_OFFSET_M == 0) {
            AscendC::CrossCoreSetFlag<CROSS_CORE_FLAG_INDEX, PIPE_MTE3>(CROSS_CORE_WAIT_FLAG_C1);
        }
    }
    if (STREAM_TAIL && mmConfigC1_.l1K == MATMUL_WRITE_OFFSET_M && copySizeND == ND_BLOCK_SIZE / NUMBER_TWO &&
        currentBsSize % MATMUL_WRITE_OFFSET_M != 0) {
        AscendC::CrossCoreSetFlag<CROSS_CORE_FLAG_INDEX, PIPE_MTE3>(CROSS_CORE_WAIT_FLAG_C1);
    }
    DrainV2PackedGamma();
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::ProcessV2ChunkLoop(
    uint32_t offsetND, uint32_t copySizeND, uint64_t bsStart, uint64_t bsChunkStart, uint32_t currentChunkSize,
    uint32_t chunkNDSize, uint32_t currentN, uint32_t buffId, LocalTensor<P>& sumBuf)
{
    if (compactC0Route_ && withGamma_ && copySizeND == ND_BLOCK_SIZE / NUMBER_TWO &&
        currentChunkSize == C0_UB_ROW_STEP) {
        ProcessV2CompactFused(offsetND, bsStart, bsChunkStart, currentN, buffId, sumBuf);
        return;
    }
    DrainV2PackedGamma();
    // 2. 加载hInGrad并进行向量计算
    ProcessV2ChunkLoadHInGrad(offsetND, copySizeND, bsChunkStart, currentChunkSize, currentN, bsStart);

    // 3. 加载x和gamma，计算xRs
    bool directC1 = UseDirectC1Block(bsStart) && copySizeND == ND_BLOCK_SIZE / NUMBER_TWO &&
                    Min(static_cast<uint64_t>(BS_BLOCK_SIZE), totalLength_ - bsStart) == BS_BLOCK_SIZE;
    ProcessV2ChunkLoadXAndGamma(offsetND, copySizeND, bsStart, bsChunkStart, currentChunkSize, buffId, directC1);

    // 4. 融合计算xRsGrad并处理xRsGradMm
    ProcessV2ChunkProcessXRsGradMm(bsStart, bsChunkStart, copySizeND, currentChunkSize, buffId, sumBuf);

    // 5. 融合处理gradXPost并输出grad_x
    ProcessV2ChunkOutputXGrad(offsetND, copySizeND, bsChunkStart, currentChunkSize, chunkNDSize);
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
template <bool hasGradXPost, bool packedC1, uint16_t rows, uint16_t cols>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::VFDoV2CompactFused(
    __ubuf__ T* outputAddr, __ubuf__ P* gammaOutAddr, __ubuf__ T* xAddr, __ubuf__ T* hInAddr, __ubuf__ T* postAddr,
    __ubuf__ P* c0Addr, __ubuf__ P* hPreAddr, __ubuf__ P* invRmsAddr, __ubuf__ P* gammaAddr, __ubuf__ P* sumAddr)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<T> inputReg, outputReg;
        Reg::RegTensor<P> xReg, gradReg, c0Reg, gammaReg, sumReg, correctionReg, adjustedReg, nextReg, errorReg,
            scalarReg, tmpReg;
        Reg::MaskReg mask = Reg::CreateMask<P, Reg::MaskPattern::ALL>();
        static constexpr Reg::CastTrait castFp32ToB16 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                         Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        for (uint16_t col = 0; col < cols; col += ND_BLOCK_SIZE / NUMBER_TWO) {
            Reg::LoadAlign(gammaReg, gammaAddr + col);
            Reg::LoadAlign(sumReg, sumAddr + col);
            Reg::LoadAlign(correctionReg, sumAddr + cols + col);
            for (uint16_t row = 0; row < rows; ++row) {
                uint32_t offset = row * cols + col;
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(inputReg, hInAddr + offset);
                Reg::Cast<P, T, ctHalf2Fp32Zero>(gradReg, inputReg, mask);
                Reg::DataCopy<P, Reg::LoadDist::DIST_BRC_B32>(scalarReg, hPreAddr + row * N_);
                Reg::Mul(gradReg, gradReg, scalarReg, mask);
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(inputReg, xAddr + offset);
                Reg::Cast<P, T, ctHalf2Fp32Zero>(xReg, inputReg, mask);
                Reg::Mul(tmpReg, gammaReg, xReg, mask);
                if constexpr (packedC1) {
                    // 按 NZ[c/8, row, c%8] 布局，每组额外保留一行物理行。
                    // 相邻 32B 列块间隔为 32*(64+1) 字节。
                    Reg::StoreAlign<P, Reg::DataCopyMode::DATA_BLOCK_COPY>(
                        gammaOutAddr + (col / 8U) * (rows + 1U) * 8U + row * 8U, tmpReg, rows + 1U, mask);
                } else {
                    Reg::StoreAlign(gammaOutAddr + offset, tmpReg, mask);
                }
                Reg::DataCopy<P, Reg::LoadDist::DIST_BRC_B32>(scalarReg, invRmsAddr + row);
                Reg::MulAddDst(gradReg, xReg, scalarReg, mask);
                Reg::LoadAlign(c0Reg, c0Addr + offset);
                Reg::MulAddDst(gradReg, gammaReg, c0Reg, mask);
                Reg::Mul(tmpReg, xReg, c0Reg, mask);
                Reg::Sub(adjustedReg, tmpReg, correctionReg, mask);
                Reg::Add(nextReg, sumReg, adjustedReg, mask);
                Reg::Sub(errorReg, nextReg, sumReg, mask);
                Reg::Sub(correctionReg, errorReg, adjustedReg, mask);
                Reg::Adds(sumReg, nextReg, ZERO, mask);
                if constexpr (hasGradXPost) {
                    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(inputReg, postAddr + offset);
                    Reg::Cast<P, T, ctHalf2Fp32Zero>(tmpReg, inputReg, mask);
                    Reg::Add(gradReg, gradReg, tmpReg, mask);
                }
                Reg::Cast<T, P, castFp32ToB16>(outputReg, gradReg, mask);
                Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(outputAddr + offset, outputReg, mask);
            }
            // 最大128列，和及补偿项合计占用1KiB。
            Reg::StoreAlign(sumAddr + cols + col, correctionReg, mask);
            Reg::StoreAlign(sumAddr + col, sumReg, mask);
        }
    }
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::PrefetchV2FusedX(uint32_t offsetND,
                                                                                           uint64_t bsChunkStart)
{
    constexpr uint32_t COLS = ND_BLOCK_SIZE / NUMBER_TWO;
    LocalTensor<T> nextX = vecInQueue1_.AllocTensor<T>();
    DataCopyExtParams params;
    DataCopyPadExtParams<T> padding;
    params.blockCount = C0_UB_ROW_STEP;
    params.blockLen = COLS * sizeof(T);
    params.srcStride = (nD_ - COLS) * sizeof(T);
    params.dstStride = 0;
    DataCopyPad(nextX, xGm_[bsChunkStart * nD_ + offsetND], params, padding);
    vecInQueue1_.EnQue(nextX);
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::PrefetchV2FusedHInPost(uint32_t offsetND,
                                                                                                 uint64_t bsChunkStart)
{
    constexpr uint32_t COLS = ND_BLOCK_SIZE / NUMBER_TWO;
    LocalTensor<T>& inputs = fusedHInPostSlots_[(bsChunkStart / C0_UB_ROW_STEP) % DOUBLE_BUFFER];
    vecInQueue0_.AllocTensor<T>(inputs);
    DataCopyExtParams params;
    DataCopyPadExtParams<T> padding;
    params.blockCount = C0_UB_ROW_STEP;
    params.blockLen = COLS * sizeof(T);
    params.srcStride = (D_ - COLS) * sizeof(T);
    params.dstStride = 0;
    DataCopyPad(inputs, gradHInGm_[bsChunkStart * D_ + offsetND % D_], params, padding);
    if (withGradXPost_) {
        LocalTensor<T> post = inputs[C0_UB_ROW_STEP * COLS];
        params.srcStride = (nD_ - COLS) * sizeof(T);
        DataCopyPad(post, gradXPostOptionalGm_[bsChunkStart * nD_ + offsetND], params, padding);
    }
    vecInQueue0_.EnQue(inputs);
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::ProcessV2CompactFused(
    uint32_t offsetND, uint64_t bsStart, uint64_t bsChunkStart, uint32_t currentN, uint32_t buffId,
    LocalTensor<P>& sumBuf)
{
    constexpr uint32_t ROWS = C0_UB_ROW_STEP;
    constexpr uint32_t COLS = ND_BLOCK_SIZE / NUMBER_TWO;
    DataCopyExtParams inputParams;
    DataCopyPadExtParams<T> inputPad;
    inputParams.blockCount = ROWS;
    inputParams.blockLen = COLS * sizeof(T);
    inputParams.srcStride = (D_ - COLS) * sizeof(T);
    inputParams.dstStride = 0;
    if (!fusedInputPipeline_) {
        vecInQueue0_.AllocTensor<T>(gradHInInBuf_);
        DataCopyPad(gradHInInBuf_, gradHInGm_[bsChunkStart * D_ + offsetND % D_], inputParams, inputPad);
        vecInQueue0_.EnQue(gradHInInBuf_);
    }
    inputParams.srcStride = (nD_ - COLS) * sizeof(T);
    if (!fusedPrefetchX_) {
        PrefetchV2FusedX(offsetND, bsChunkStart);
    }
    if (withGradXPost_ && !fusedInputPipeline_) {
        // 第二个 8 KiB input0 槽位一直保留到本次 VF 完成。
        vecInQueue0_.AllocTensor<T>(gradXPostBuf_);
        DataCopyPad(gradXPostBuf_, gradXPostOptionalGm_[bsChunkStart * nD_ + offsetND], inputParams, inputPad);
        vecInQueue0_.EnQue(gradXPostBuf_);
    }
    uint32_t row = static_cast<uint32_t>(bsChunkStart - bsStart);
    bool directRead = directC0Chunk_;
    if (directRead) {
        xRsGradMmInBuf_ = LocalTensor<P>(
            TPosition::VECIN,
            C0UbSlotOffset((row / C0UbTileRows()) % C0UbSlotCount()) + (row % C0UbTileRows()) * COLS * sizeof(P),
            ROWS * COLS);
    } else {
        // 与现有 compact 路径相同，最终 ND 尾块借用该槽位。
        xRsGradMmInBuf_ = c0ResultTBuf_.Get<P>();
        SetFlag<HardEvent::V_MTE2>(EVENT_ID3);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID3);
        DataCopyExtParams c0Params;
        DataCopyPadExtParams<P> c0Pad;
        c0Params.blockCount = ROWS;
        c0Params.blockLen = COLS * sizeof(P);
        c0Params.srcStride = (ND_BLOCK_SIZE - COLS) * sizeof(P);
        c0Params.dstStride = 0;
        uint64_t offset = workspaceBuf_.GetXRsGradOffset(GetBlockIdx() / VEC_DEAL_VECIDX_DIV, buffId);
        DataCopyPad(xRsGradMmInBuf_, workSpaceGm_[offset + row * ND_BLOCK_SIZE + vecColumnOffset_], c0Params, c0Pad);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID3);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID3);
    }
    if (fusedInputPipeline_) {
        // 深度为零的 input0 将就绪状态绑定到已分配的 tensor 句柄。
        gradHInInBuf_ = fusedHInPostSlots_[(bsChunkStart / ROWS) % DOUBLE_BUFFER];
    }
    vecInQueue0_.DeQue<T>(gradHInInBuf_);
    xInputBuf_ = vecInQueue1_.DeQue<T>();
    if (withGradXPost_) {
        if (fusedInputPipeline_) {
            gradXPostBuf_ = gradHInInBuf_[ROWS * COLS];
        } else {
            vecInQueue0_.DeQue<T>(gradXPostBuf_);
        }
    }
    if (fusedPrefetchX_ && bsChunkStart + ROWS < fusedPrefetchXEnd_) {
        if (fusedInputPipeline_) {
            PrefetchV2FusedHInPost(offsetND, bsChunkStart + ROWS);
        }
        PrefetchV2FusedX(offsetND, bsChunkStart + ROWS);
    }
    bool directC1 = UseDirectC1Block(bsStart);
    bool packedC1 = directC1 && mmConfigC1_.l1K >= ROWS && mmConfigC1_.l1K % ROWS == 0;
    if (packedC1 || fusedInputPipeline_) {
        packedGammaSlot_ = packedGammaDoubleBuffer_ ? (row / ROWS) % DOUBLE_BUFFER : 0U;
        WaitV2PackedGammaSlot(packedGammaSlot_);
        // 融合 VF 不使用两个 16 KiB 的 FP32 中间区域。
        // 容量允许时交替使用带 padding 的分块，仅在复用时等待。
        gammaOutUb_ = fp32TBuf_.GetWithOffset<P>(
            (ROWS + 1U) * COLS, globalUbOffset_ + COLS * sizeof(P) + packedGammaSlot_ * (ROWS + 1U) * COLS * sizeof(P));
    } else {
        DrainV2PackedGamma();
        gammaOutUb_ = vecOutQueue1_.AllocTensor<P>();
    }
    xOutputBuf_ = vecOutQueue1_.AllocTensor<T>();
    if (bsChunkStart == 0 ||
        (splitRowTail_ && blockIdx_ % NUMBER_TWO == 1 && bsChunkStart == BS_BLOCK_SIZE / NUMBER_TWO)) {
        Duplicate(sumBuf, static_cast<P>(0), NUMBER_TWO * COLS);
        PipeBarrier<PIPE_V>();
    }
    __ubuf__ T* out = (__ubuf__ T*)xOutputBuf_.GetPhyAddr();
    __ubuf__ P* gammaOut = (__ubuf__ P*)gammaOutUb_.GetPhyAddr();
    __ubuf__ T* x = (__ubuf__ T*)xInputBuf_.GetPhyAddr();
    __ubuf__ T* hIn = (__ubuf__ T*)gradHInInBuf_.GetPhyAddr();
    __ubuf__ P* c0 = (__ubuf__ P*)xRsGradMmInBuf_.GetPhyAddr();
    __ubuf__ P* hPre = (__ubuf__ P*)hPreUb_.GetPhyAddr() + row * N_ + currentN;
    __ubuf__ P* inv = (__ubuf__ P*)invRmsUb_.GetPhyAddr() + row;
    __ubuf__ P* gamma = (__ubuf__ P*)gammaIn_.GetPhyAddr();
    __ubuf__ P* sum = (__ubuf__ P*)sumBuf.GetPhyAddr();
    if (packedC1 && withGradXPost_) {
        VFDoV2CompactFused<true, true>(out, gammaOut, x, hIn, (__ubuf__ T*)gradXPostBuf_.GetPhyAddr(), c0, hPre, inv,
                                       gamma, sum);
    } else if (packedC1) {
        VFDoV2CompactFused<false, true>(out, gammaOut, x, hIn, x, c0, hPre, inv, gamma, sum);
    } else if (withGradXPost_) {
        VFDoV2CompactFused<true>(out, gammaOut, x, hIn, (__ubuf__ T*)gradXPostBuf_.GetPhyAddr(), c0, hPre, inv, gamma,
                                 sum);
    } else {
        VFDoV2CompactFused<false>(out, gammaOut, x, hIn, x, c0, hPre, inv, gamma, sum);
    }
    vecInQueue0_.FreeTensor(gradHInInBuf_);
    vecInQueue1_.FreeTensor(xInputBuf_);
    if (withGradXPost_ && !fusedInputPipeline_) {
        vecInQueue0_.FreeTensor(gradXPostBuf_);
    }
    if (directRead && (row + ROWS) % C0UbTileRows() == 0) {
        CrossCoreSetFlag<4, PIPE_V>(C0UbFreeFlag((row / C0UbTileRows()) % C0UbSlotCount()));
    }
    if (packedC1) {
        CopyV2PackedGammaToC1(bsStart, bsChunkStart);
    } else if (fusedInputPipeline_) {
        const event_t event = GetPackedGammaEvent(packedGammaSlot_);
        SetFlag<HardEvent::V_MTE3>(event);
        WaitFlag<HardEvent::V_MTE3>(event);
        CopyV2GammaOutputToC1(COLS, bsStart, bsChunkStart, ROWS, buffId, directC1);
        SetFlag<HardEvent::MTE3_V>(event);
        packedGammaPending_[packedGammaSlot_] = true;
    } else {
        vecOutQueue1_.EnQue(gammaOutUb_);
        gammaOutUb_ = vecOutQueue1_.DeQue<P>();
        CopyV2GammaOutputToC1(COLS, bsStart, bsChunkStart, ROWS, buffId, directC1);
    }
    vecOutQueue1_.EnQue(xOutputBuf_);
    xOutputBuf_ = vecOutQueue1_.DeQue<T>();
    DataCopyExtParams outputParams;
    outputParams.blockCount = ROWS;
    outputParams.blockLen = COLS * sizeof(T);
    outputParams.srcStride = 0;
    outputParams.dstStride = (nD_ - COLS) * sizeof(T);
    DataCopyPad(xGradGm_[bsChunkStart * nD_ + offsetND], xOutputBuf_, outputParams);
    vecOutQueue1_.FreeTensor(xOutputBuf_);
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::ProcessV2ChunkLoadInvRms(
    uint64_t bsChunkStart, uint32_t currentChunkSize)
{
    DataCopyPadParams dataCopyPadParams;
    dataCopyPadParams.isPad = false;
    DataCopyParams bsCopyParams;
    bsCopyParams.blockCount = 1;
    bsCopyParams.blockLen = currentChunkSize * sizeof(P);
    bsCopyParams.srcStride = 0;
    bsCopyParams.dstStride = 0;

    vecInQueue0_.AllocTensor<P>(invRmsInBuf_);
    DataCopyPad(invRmsInBuf_, invRmsGm_[bsChunkStart], bsCopyParams, dataCopyPadParams);
    vecInQueue0_.EnQue(invRmsInBuf_);
    vecInQueue0_.DeQue<P>(invRmsInBuf_);

    invRmsGradBuf_ = vecInQueue1_.AllocTensor<P>();
    DataCopyPad(invRmsGradBuf_, workSpaceGm_[workspaceBuf_.GetInvRmsGradOffset(bsChunkStart)], bsCopyParams,
                dataCopyPadParams);
    vecInQueue1_.EnQue(invRmsGradBuf_);
    LocalTensor<P> invRmsGradUb = vecInQueue1_.DeQue<P>();

    AIV21Process(invRmsInBuf_, invRmsUb_, invRmsGradUb, currentChunkSize);
    vecInQueue0_.FreeTensor(invRmsInBuf_);
    vecInQueue1_.FreeTensor(invRmsGradUb);
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::ProcessV2ChunkLoadHInGrad(
    uint32_t offsetND, uint32_t copySizeND, uint64_t bsChunkStart, uint32_t currentChunkSize, uint32_t currentN,
    uint64_t bsStart)
{
    DataCopyPadParams dataCopyPadParams;
    dataCopyPadParams.isPad = false;
    DataCopyParams blockCopyParams;
    blockCopyParams.blockCount = currentChunkSize;
    blockCopyParams.blockLen = copySizeND * sizeof(T);
    blockCopyParams.srcStride = (D_ - copySizeND) * sizeof(T);
    blockCopyParams.dstStride = 0;
    vecInQueue0_.AllocTensor<T>(gradHInInBuf_);
    DataCopyPad(gradHInInBuf_, gradHInGm_[bsChunkStart * D_ + (offsetND % D_)], blockCopyParams, dataCopyPadParams);
    vecInQueue0_.EnQue(gradHInInBuf_);

    vecInQueue0_.DeQue<T>(gradHInInBuf_);

    __ubuf__ T* gradHInInBufAddr = (__ubuf__ T*)gradHInInBuf_.GetPhyAddr();
    __ubuf__ P* hPreUbAddr = (__ubuf__ P*)hPreUb_.GetPhyAddr();
    __ubuf__ P* xGradVec3BufAddr = (__ubuf__ P*)xGradVec3Buf_.GetPhyAddr();
    VFDoV2HInMulHPre(xGradVec3BufAddr, gradHInInBufAddr, hPreUbAddr, currentChunkSize, copySizeND, currentN,
                     (bsChunkStart - bsStart));
    vecInQueue0_.FreeTensor(gradHInInBuf_);
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::ProcessV2ChunkLoadXAndGamma(
    uint32_t offsetND, uint32_t copySizeND, uint64_t bsStart, uint64_t bsChunkStart, uint32_t currentChunkSize,
    uint32_t buffId, bool directC1)
{
    DataCopyExtParams xCopyParams;
    DataCopyPadExtParams<T> xCopyPadParams;
    xCopyParams.blockCount = currentChunkSize;
    xCopyParams.blockLen = copySizeND * sizeof(T);
    xCopyParams.srcStride = (nD_ - copySizeND) * sizeof(T);
    xCopyParams.dstStride = 0;
    xInputBuf_ = vecInQueue1_.AllocTensor<T>();
    DataCopyPad(xInputBuf_, xGm_[bsChunkStart * nD_ + offsetND], xCopyParams, xCopyPadParams);
    vecInQueue1_.EnQue(xInputBuf_);

    xInputBuf_ = vecInQueue1_.DeQue<T>();

    // 计算xRs和gamma
    __ubuf__ T* bf16InputAddr = (__ubuf__ T*)xInputBuf_.GetPhyAddr();
    __ubuf__ P* xRsFp32Addr = (__ubuf__ P*)xRsFp32Buf_.GetPhyAddr();
    gammaOutUb_ = vecOutQueue1_.AllocTensor<P>();
    __ubuf__ P* gammaOutAddr = (__ubuf__ P*)gammaOutUb_.GetPhyAddr();
    if (withGamma_) {
        __ubuf__ P* gammaInAddr = (__ubuf__ P*)gammaIn_.GetPhyAddr();
        VFDoV2XCastAndMulGamma<true>(xRsFp32Addr, gammaOutAddr, bf16InputAddr, gammaInAddr, currentChunkSize,
                                     copySizeND);
    } else {
        VFDoV2XCastAndMulGamma<false>(xRsFp32Addr, gammaOutAddr, bf16InputAddr, nullptr, currentChunkSize, copySizeND);
    }
    vecInQueue1_.FreeTensor(xInputBuf_);
    vecOutQueue1_.EnQue(gammaOutUb_);
    gammaOutUb_ = vecOutQueue1_.DeQue<P>();

    CopyV2GammaOutputToC1(copySizeND, bsStart, bsChunkStart, currentChunkSize, buffId, directC1);
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::DrainV2PackedGamma()
{
    for (uint32_t slot = 0; slot < DOUBLE_BUFFER; ++slot) {
        WaitV2PackedGammaSlot(slot);
    }
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::WaitV2PackedGammaSlot(uint32_t slot)
{
    if (packedGammaPending_[slot]) {
        const event_t event = GetPackedGammaEvent(slot);
        WaitFlag<HardEvent::MTE3_V>(event);
        packedGammaPending_[slot] = false;
    }
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::CopyV2PackedGammaToC1(uint64_t bsStart,
                                                                                                uint64_t bsChunkStart)
{
    constexpr uint32_t ROWS = C0_UB_ROW_STEP;
    constexpr uint32_t COLS = ND_BLOCK_SIZE / NUMBER_TWO;
    constexpr uint32_t NZ_BLOCK = 32U / sizeof(P);
    uint32_t l1K = mmConfigC1_.l1K;
    uint32_t row = static_cast<uint32_t>(bsChunkStart - bsStart);
    uint32_t l1Offset =
        tiling_->l1UsedBytes +
        ((row / l1K) * l1K * ND_BLOCK_SIZE + vecColumnOffset_ * l1K + (row % l1K) * NZ_BLOCK) * sizeof(P);
    // NZ 各列块之间存在 L1 K 步长，描述范围必须覆盖最后一块的实际末端。
    const uint32_t l1SpanElems = ((COLS / NZ_BLOCK - 1U) * l1K + ROWS) * NZ_BLOCK;
    LocalTensor<P> destination(TPosition::A1, l1Offset, l1SpanElems);
    // 跳过额外的 UB padding 行；保留原有 L1 NZ 步长和就绪事件。
    const event_t event = GetPackedGammaEvent(packedGammaSlot_);
    SetFlag<HardEvent::V_MTE3>(event);
    WaitFlag<HardEvent::V_MTE3>(event);
    DataCopyParams params{static_cast<uint16_t>(COLS / NZ_BLOCK), static_cast<uint16_t>(ROWS), 1U,
                          static_cast<uint16_t>(l1K - ROWS)};
    DataCopy(destination, gammaOutUb_, params);
    SetFlag<HardEvent::MTE3_V>(event);
    packedGammaPending_[packedGammaSlot_] = true;
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::CopyV2GammaOutputToC1(
    uint32_t copySizeND, uint64_t bsStart, uint64_t bsChunkStart, uint32_t currentChunkSize, uint32_t buffId,
    bool directC1)
{
    // 输出gamma到workspace
    DataCopyExtParams fp32ExtCopyParams;
    fp32ExtCopyParams.blockCount = currentChunkSize;
    fp32ExtCopyParams.blockLen = copySizeND * sizeof(P);
    fp32ExtCopyParams.srcStride = 0;
    fp32ExtCopyParams.dstStride = (ND_BLOCK_SIZE - copySizeND) * sizeof(P);

    uint64_t offset = workspaceBuf_.GetXRsOffset(GetBlockIdx() / VEC_DEAL_VECIDX_DIV, buffId);
    offset += (bsChunkStart - bsStart) * ND_BLOCK_SIZE;
    offset += vecColumnOffset_;

    if (directC1) {
        // UB→L1 NZ 使用 32 字节 C0 分块：每行 8 个 FP32 元素。将
        // 每个 64 行 V2 分块按该网格放入 C1 的 64/128 行 K 分片。
        uint32_t l1K = mmConfigC1_.l1K;
        uint32_t rowOffset = static_cast<uint32_t>(bsChunkStart - bsStart);
        uint32_t kTileIndex = rowOffset / l1K;
        uint32_t rowInKTile = rowOffset % l1K;
        uint32_t l1Offset =
            tiling_->l1UsedBytes +
            (kTileIndex * l1K * ND_BLOCK_SIZE + vecColumnOffset_ * l1K + rowInKTile * (32U / sizeof(P))) * sizeof(P);
        constexpr uint32_t NZ_BLOCK = 32U / sizeof(P);
        const uint32_t l1SpanElems = ((copySizeND / NZ_BLOCK - 1U) * l1K + currentChunkSize) * NZ_BLOCK;
        LocalTensor<P> dst(TPosition::A1, l1Offset, l1SpanElems);
        Nd2NzParams p;
        p.ndNum = 1;
        p.nValue = currentChunkSize;
        p.dValue = copySizeND;
        p.srcNdMatrixStride = 1;
        p.srcDValue = copySizeND;
        p.dstNzC0Stride = l1K;
        p.dstNzNStride = 1;
        p.dstNzMatrixStride = 1;
        AscendC::DataCopyUB2L1ND2NZImplV2((__cbuf__ P*)dst.GetPhyAddr(), (__ubuf__ P*)gammaOutUb_.GetPhyAddr(), p);
    } else {
        DataCopyPad(workSpaceGm_[offset], gammaOutUb_, fp32ExtCopyParams);
    }
    if (!fusedInputPipeline_) {
        vecOutQueue1_.FreeTensor(gammaOutUb_);
    }
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::ProcessV2ChunkProcessXRsGradMm(
    uint64_t bsStart, uint64_t bsChunkStart, uint32_t copySizeND, uint32_t currentChunkSize, uint32_t buffId,
    LocalTensor<P>& sumBuf)
{
    bool directRead =
        directC0Chunk_ && Min(static_cast<uint64_t>(BS_BLOCK_SIZE), totalLength_ - bsStart) == BS_BLOCK_SIZE;
    if (directRead) {
        uint32_t row = static_cast<uint32_t>(bsChunkStart - bsStart);
        uint32_t tileRows = C0UbTileRows();
        xRsGradMmInBuf_ = LocalTensor<P>(
            TPosition::VECIN,
            C0UbSlotOffset((row / tileRows) % C0UbSlotCount()) + (row % tileRows) * copySizeND * sizeof(P),
            currentChunkSize * copySizeND);
    } else {
        DataCopyPadParams dataCopyPadParams{};
        dataCopyPadParams.isPad = false;
        uint64_t offsetXRsGradMm = workspaceBuf_.GetXRsGradOffset(GetBlockIdx() / VEC_DEAL_VECIDX_DIV, buffId);
        bool packedRead = packedC0Read_ && !splitRowTail_;
        if (packedRead) {
            // 每个成对 AIV 负责一个连续的 512x64 FP32 平面。
            offsetXRsGradMm +=
                (bsChunkStart - bsStart) * (ND_BLOCK_SIZE / NUMBER_TWO) +
                (vecColumnOffset_ / (ND_BLOCK_SIZE / NUMBER_TWO)) * BS_BLOCK_SIZE * (ND_BLOCK_SIZE / NUMBER_TWO);
        } else {
            offsetXRsGradMm += (bsChunkStart - bsStart) * ND_BLOCK_SIZE + vecColumnOffset_;
        }
        if (compactC0Route_) {
            // ND 尾块是本核最后的分块，因此 C0 不会再次写入 UB。
            // 借用一个 16 KiB 结果槽位，替代一个 8 KiB BF16 输入槽位。
            xRsGradMmInBuf_ = c0ResultTBuf_.Get<P>();
            SetFlag<HardEvent::V_MTE2>(EVENT_ID3);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID3);
        } else {
            vecInQueue0_.AllocTensor<P>(xRsGradMmInBuf_);
        }
        DataCopyParams blockCopyParams{};
        blockCopyParams.blockCount = currentChunkSize;
        blockCopyParams.blockLen = copySizeND * sizeof(P);
        blockCopyParams.srcStride = packedRead ? 0U : (ND_BLOCK_SIZE - copySizeND) * sizeof(P);
        blockCopyParams.dstStride = 0;
        DataCopyPad(xRsGradMmInBuf_, workSpaceGm_[offsetXRsGradMm], blockCopyParams, dataCopyPadParams);
        if (compactC0Route_) {
            SetFlag<HardEvent::MTE2_V>(EVENT_ID3);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID3);
        } else {
            vecInQueue0_.EnQue(xRsGradMmInBuf_);
            vecInQueue0_.DeQue<P>(xRsGradMmInBuf_);
        }
    }

    __ubuf__ P* xRsGradMmInBufAddr = (__ubuf__ P*)xRsGradMmInBuf_.GetPhyAddr();
    __ubuf__ P* xGradVec3Addr = (__ubuf__ P*)xGradVec3Buf_.GetPhyAddr();
    __ubuf__ P* xRsFp32Addr = (__ubuf__ P*)xRsFp32Buf_.GetPhyAddr();
    __ubuf__ P* invRmsAddr = (__ubuf__ P*)invRmsUb_.GetPhyAddr() + (bsChunkStart - bsStart);
    if (withGamma_) {
        __ubuf__ P* gammaInAddr = (__ubuf__ P*)gammaIn_.GetPhyAddr();
        if (bsChunkStart == 0 ||
            (splitRowTail_ && blockIdx_ % NUMBER_TWO == 1 && bsChunkStart == BS_BLOCK_SIZE / NUMBER_TWO)) {
            Duplicate(sumBuf, static_cast<P>(0), ND_BLOCK_SIZE);
            PipeBarrier<PIPE_V>();
        }
        if (compactC0Route_) {
            VFDoV2FullChunkGammaReduce(xGradVec3Addr, xRsGradMmInBufAddr, xRsFp32Addr, invRmsAddr, gammaInAddr,
                                       (__ubuf__ P*)sumBuf.GetPhyAddr());
        } else {
            // 和完整块共用融合归约；sum与补偿各保留64个FP32元素，跨BS块持续累加。
            VFDoV2FullChunkGammaReduce<false>(xGradVec3Addr, xRsGradMmInBufAddr, xRsFp32Addr, invRmsAddr, gammaInAddr,
                                              (__ubuf__ P*)sumBuf.GetPhyAddr(), currentChunkSize, copySizeND,
                                              copySizeND);
        }
    } else {
        VFDoV2XGradWithoutGamma(xGradVec3Addr, xRsGradMmInBufAddr, xRsFp32Addr, invRmsAddr, currentChunkSize,
                                copySizeND);
    }

    if (directRead && (bsChunkStart - bsStart + currentChunkSize) % C0UbTileRows() == 0) {
        // 融合 VF 和 gamma 归约是 C0 槽位的最后读取者。
        // grad-x 输出使用 xGradVec3Buf_，因此不需要继续占用该槽位。
        CrossCoreSetFlag<4, PIPE_V>(
            C0UbFreeFlag(static_cast<uint32_t>((bsChunkStart - bsStart) / C0UbTileRows()) % C0UbSlotCount()));
    } else if (!directRead && !compactC0Route_) {
        vecInQueue0_.FreeTensor(xRsGradMmInBuf_);
    }

    PipeBarrier<PIPE_V>();
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
template <bool hasGradXPost>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::VFDoV2AddGradXPostAndCast(
    __ubuf__ T* outputAddr, __ubuf__ P* xGradAddr, __ubuf__ T* gradXPostAddr, uint32_t totalElem)
{
    uint16_t vfLoopCnt = CeilDiv(totalElem, static_cast<uint32_t>(eleNumPerVf_));
    __VEC_SCOPE__
    {
        Reg::RegTensor<P> xGradReg, gradXPostFp32Reg;
        Reg::RegTensor<T> gradXPostReg, outputReg;
        static constexpr Reg::CastTrait castFp32ToB16 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                         Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        uint32_t len = totalElem;
        for (uint16_t vfBlockIdx = 0; vfBlockIdx < vfLoopCnt; vfBlockIdx++) {
            Reg::MaskReg mask = Reg::UpdateMask<P>(len);
            uint32_t offset = vfBlockIdx * eleNumPerVf_;
            Reg::LoadAlign(xGradReg, xGradAddr + offset);
            if constexpr (hasGradXPost) {
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(gradXPostReg, gradXPostAddr + offset);
                Reg::Cast<P, T, ctHalf2Fp32Zero>(gradXPostFp32Reg, gradXPostReg, mask);
                Reg::Add(xGradReg, xGradReg, gradXPostFp32Reg, mask);
            }
            Reg::Cast<T, P, castFp32ToB16>(outputReg, xGradReg, mask);
            Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(outputAddr + offset, outputReg, mask);
        }
    }
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::ProcessV2ChunkOutputXGrad(
    uint32_t offsetND, uint32_t copySizeND, uint64_t bsChunkStart, uint32_t currentChunkSize, uint32_t chunkNDSize)
{
    xOutputBuf_ = vecOutQueue1_.AllocTensor<T>();
    __ubuf__ T* outputAddr = (__ubuf__ T*)xOutputBuf_.GetPhyAddr();
    __ubuf__ P* xGradAddr = (__ubuf__ P*)xGradVec3Buf_.GetPhyAddr();
    if (withGradXPost_) {
        DataCopyExtParams gradXPostCopyParams;
        DataCopyPadExtParams<T> gradXPostPadParams;
        gradXPostCopyParams.blockCount = currentChunkSize;
        gradXPostCopyParams.blockLen = copySizeND * sizeof(T);
        gradXPostCopyParams.srcStride = (nD_ - copySizeND) * sizeof(T);
        gradXPostCopyParams.dstStride = 0;
        gradXPostBuf_ = vecInQueue1_.AllocTensor<T>();
        DataCopyPad(gradXPostBuf_, gradXPostOptionalGm_[bsChunkStart * nD_ + offsetND], gradXPostCopyParams,
                    gradXPostPadParams);
        vecInQueue1_.EnQue(gradXPostBuf_);
        gradXPostBuf_ = vecInQueue1_.DeQue<T>();
        __ubuf__ T* gradXPostAddr = (__ubuf__ T*)gradXPostBuf_.GetPhyAddr();
        VFDoV2AddGradXPostAndCast<true>(outputAddr, xGradAddr, gradXPostAddr, chunkNDSize);
    } else {
        VFDoV2AddGradXPostAndCast<false>(outputAddr, xGradAddr, nullptr, chunkNDSize);
    }
    PipeBarrier<PIPE_V>();
    if (withGradXPost_) {
        vecInQueue1_.FreeTensor(gradXPostBuf_);
    }
    vecOutQueue1_.EnQue(xOutputBuf_);
    xOutputBuf_ = vecOutQueue1_.DeQue<T>();

    DataCopyExtParams bf16ExtCopyParams;
    bf16ExtCopyParams.blockCount = currentChunkSize;
    bf16ExtCopyParams.blockLen = copySizeND * sizeof(T);
    bf16ExtCopyParams.srcStride = 0;
    bf16ExtCopyParams.dstStride = (nD_ - copySizeND) * sizeof(T);
    DataCopyPad(xGradGm_[bsChunkStart * nD_ + offsetND], xOutputBuf_, bf16ExtCopyParams);
    vecOutQueue1_.FreeTensor(xOutputBuf_);
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::AIV21Process(LocalTensor<P>& invRmsInBuf,
                                                                                       LocalTensor<P>& invRmsUb,
                                                                                       LocalTensor<P>& invRmsGradUb,
                                                                                       uint32_t currentChunkSize)
{
    __ubuf__ P* invRmsInBufAddr = (__ubuf__ P*)invRmsInBuf.GetPhyAddr();
    __ubuf__ P* invRmsUbAddr = (__ubuf__ P*)invRmsUb.GetPhyAddr();
    __ubuf__ P* invRmsGradUbAddr = (__ubuf__ P*)invRmsGradUb.GetPhyAddr();

    uint32_t vfloopCnt = Ceil(currentChunkSize, eleNumPerVf_);
    uint32_t curLen = currentChunkSize;

    __VEC_SCOPE__
    {
        Reg::RegTensor<P> invRmsInReg;
        Reg::RegTensor<P> invRmsTempReg;
        Reg::RegTensor<P> invRmsUbReg;
        Reg::RegTensor<P> invRmsGradUb;
        Reg::RegTensor<P> scaleMeanReg;
        Reg::MaskReg maskAll, mask;
        maskAll = Reg::CreateMask<P, Reg::MaskPattern::ALL>();
        Reg::Duplicate<P>(scaleMeanReg, (-scaleMean_), maskAll);
        for (uint16_t vfBlockIdx = 0; vfBlockIdx < static_cast<uint16_t>(vfloopCnt); vfBlockIdx++) {
            uint32_t elemOffset = vfBlockIdx * eleNumPerVf_;
            mask = Reg::UpdateMask<P>(curLen);
            Reg::LoadAlign(invRmsInReg, invRmsInBufAddr + elemOffset);
            Reg::Mul(invRmsTempReg, invRmsInReg, invRmsInReg, mask);
            Reg::Mul(invRmsUbReg, invRmsTempReg, invRmsInReg, mask);
            Reg::LoadAlign(invRmsGradUb, invRmsGradUbAddr + elemOffset);
            Reg::Mul(invRmsUbReg, invRmsUbReg, invRmsGradUb, mask);
            Reg::Mul(invRmsUbReg, invRmsUbReg, scaleMeanReg, mask);
            Reg::StoreAlign(invRmsUbAddr + elemOffset, invRmsUbReg, mask);
        }
    }
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::MergeGammaPartials()
{
    if (blockIdx_ >= NUMBER_TWO * v1UsedCubeCoreNum_ || blockIdx_ % NUMBER_TWO != 0U) {
        return;
    }
    for (uint32_t nd = dealStartND_; nd < dealEndND_; nd += ND_BLOCK_SIZE) {
        uint32_t cols = Min(ND_BLOCK_SIZE, dealEndND_ - nd);
        if (!CanPartitionV2Rows(nd, cols)) {
            continue;
        }
        // 只由配对 AIV 的偶数核合并两个平面，固定加法顺序。
        LocalTensor<P> left;
        vecInQueue0_.AllocTensor<P>(left);
        auto right = vecInQueue1_.AllocTensor<P>();
        DataCopyParams params{};
        params.blockCount = 1;
        params.blockLen = cols * sizeof(P);
        params.srcStride = 0;
        params.dstStride = 0;
        DataCopyPadParams pad{false, 0, 0, 0};
        DataCopyPad(left, workSpaceGm_[workspaceBuf_.GetGammaPartialOffset(0U, nd, nD_)], params, pad);
        DataCopyPad(right, workSpaceGm_[workspaceBuf_.GetGammaPartialOffset(1U, nd, nD_)], params, pad);
        vecInQueue0_.EnQue(left);
        vecInQueue1_.EnQue(right);
        vecInQueue0_.DeQue<P>(left);
        right = vecInQueue1_.DeQue<P>();
        auto sum = vecOutQueueSmall_.AllocTensor<P>();
        Add(sum, left, right, cols);
        vecOutQueueSmall_.EnQue(sum);
        sum = vecOutQueueSmall_.DeQue<P>();
        DataCopyPad(gammaGradGm_[nd], sum, params);
        vecInQueue0_.FreeTensor(left);
        vecInQueue1_.FreeTensor(right);
        vecOutQueueSmall_.FreeTensor(sum);
    }
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::MergePhiPartials()
{
    // 主流水已经结束，恢复完整队列后批量读取部分和，降低逐小片搬运的启动开销。
    PipeBarrier<PIPE_ALL>();
    pipe_->Reset();
    pipe_->InitBuffer(vecInQueue0_, DOUBLE_BUFFER, INOUT_QUEUE_SIZE);
    pipe_->InitBuffer(vecOutQueue1_, DOUBLE_BUFFER, INOUT_QUEUE_SIZE);
    pipe_->InitBuffer(vecInQueue1_, DOUBLE_BUFFER, INOUT_QUEUE_SIZE);
    pipe_->InitBuffer(vecOutQueueSmall_, 1, SMALL_QUEUE_SIZE);
    pipe_->InitBuffer(fp32TBuf_, FP32_BUF_SIZE);
    constexpr uint32_t columns = 1024U;
    constexpr uint32_t batchPlanes = 8U;
    const uint64_t planeElements = fusionSize_ * nD_;
    const uint64_t partialBase = workspaceBuf_.GetPhiPartialBaseOffset(nD_);
    const uint32_t planes = static_cast<uint32_t>(CeilDiv(totalLength_, uint64_t(mmConfigC1_.l1K)));
    auto state = fp32TBuf_.Get<P>();
    for (uint64_t offset = uint64_t(blockIdx_) * columns; offset < planeElements;
         offset += uint64_t(vecCoreNum_) * columns) {
        Duplicate(state, static_cast<P>(0), NUMBER_TWO * columns);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID6);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID6);
        for (uint32_t first = 0; first < planes; first += batchPlanes) {
            const uint32_t count = Min(batchPlanes, planes - first);
            LocalTensor<P> input;
            vecInQueue0_.AllocTensor<P>(input);
            DataCopyExtParams params{};
            params.blockCount = count;
            params.blockLen = columns * sizeof(P);
            params.srcStride = (planeElements - columns) * sizeof(P);
            DataCopyPadExtParams<P> padding{false, 0, 0, 0};
            DataCopyPad(input, workSpaceGm_[partialBase + uint64_t(first) * planeElements + offset], params, padding);
            vecInQueue0_.EnQue(input);
            vecInQueue0_.DeQue<P>(input);
            __ubuf__ P* inputAddr = (__ubuf__ P*)input.GetPhyAddr();
            __ubuf__ P* stateAddr = (__ubuf__ P*)state.GetPhyAddr();
            __VEC_SCOPE__
            {
                Reg::MaskReg mask = Reg::CreateMask<P, Reg::MaskPattern::ALL>();
                Reg::RegTensor<P> sum, correction, value, adjusted, next, error;
                for (uint16_t col = 0U; col < columns; col += 64U) {
                    Reg::LoadAlign(sum, stateAddr + col);
                    Reg::LoadAlign(correction, stateAddr + columns + col);
                    for (uint16_t plane = 0; plane < static_cast<uint16_t>(count); ++plane) {
                        Reg::LoadAlign(value, inputAddr + plane * columns + col);
                        Reg::Sub(adjusted, value, correction, mask);
                        Reg::Add(next, sum, adjusted, mask);
                        Reg::Sub(error, next, sum, mask);
                        Reg::Sub(correction, error, adjusted, mask);
                        Reg::Adds(sum, next, ZERO, mask);
                    }
                    Reg::StoreAlign(stateAddr + col, sum, mask);
                    Reg::StoreAlign(stateAddr + columns + col, correction, mask);
                }
            }
            vecInQueue0_.FreeTensor(input);
        }
        // 不同 AIV 写不重叠的连续片段，无需原子加。
        SetFlag<HardEvent::V_MTE3>(EVENT_ID6);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID6);
        DataCopyExtParams output{};
        output.blockCount = 1;
        output.blockLen = columns * sizeof(P);
        DataCopyPad(gradPhiGm_[offset], state, output);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID6);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID6);
    }
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::ProcessV3()
{
    uint32_t v3Cols = fusionSize_ > ALPHA_GRAD_PADDING ? fusionSize_ : ALPHA_GRAD_PADDING;
    if (compactC0Route_ && usedVecCoreNum_ * v3Cols * sizeof(P) > 8U * KIBIBYTE) {
        // V2 的 BF16 输入占用 8 KiB。V3 则从所有 Vector 核
        // 汇聚 FP32 归约结果，因此在需要汇聚时恢复完整队列。
        PipeBarrier<PIPE_ALL>();
        pipe_->Reset();
        pipe_->InitBuffer(vecInQueue0_, DOUBLE_BUFFER, INOUT_QUEUE_SIZE);
        pipe_->InitBuffer(vecOutQueue1_, DOUBLE_BUFFER, INOUT_QUEUE_SIZE);
        pipe_->InitBuffer(vecInQueue1_, DOUBLE_BUFFER, INOUT_QUEUE_SIZE);
        pipe_->InitBuffer(vecOutQueueSmall_, 1, SMALL_QUEUE_SIZE);
        pipe_->InitBuffer(fp32TBuf_, FP32_BUF_SIZE);
    }
    ProcessV3AlphaGrad();
    ProcessV3BiasGrad();
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::ProcessV3AlphaGrad()
{
    LocalTensor<P> alphaGradInLocal = vecInQueue1_.AllocTensor<P>();
    LocalTensor<P> alphaGradOutLocal = vecOutQueueSmall_.AllocTensor<P>();
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = usedVecCoreNum_ * ALPHA_GRAD_PADDING * sizeof(P);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    DataCopyPadParams dataCopyPadParams;
    dataCopyPadParams.isPad = false;
    DataCopyPad(alphaGradInLocal, workSpaceGm_[workspaceBuf_.GetAlphaGradOffset(0)], dataCopyParams, dataCopyPadParams);
    vecInQueue1_.EnQue(alphaGradInLocal);
    alphaGradInLocal = vecInQueue1_.DeQue<P>();
    VFDoV3ProcessAlphaGrad((__ubuf__ P*)alphaGradOutLocal.GetPhyAddr(), (__ubuf__ P*)alphaGradInLocal.GetPhyAddr());
    SetFlag<HardEvent::V_S>(EVENT_ID2);
    WaitFlag<HardEvent::V_S>(EVENT_ID2);
    alphaGradOutLocal.SetValue(1, alphaGradOutLocal.GetValue(ALPHA_GRAD_SHAPE_2_OFFSET));
    alphaGradOutLocal.SetValue(2, alphaGradOutLocal.GetValue(ALPHA_GRAD_SHAPE_3_OFFSET));
    vecOutQueueSmall_.EnQue(alphaGradOutLocal);
    alphaGradOutLocal = vecOutQueueSmall_.DeQue<P>();
    PipeBarrier<PIPE_V>();
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = ALPHA_GRAD_LAST_DIM_SIZE * sizeof(P);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    DataCopyPad(alphaGradGm_, alphaGradOutLocal, dataCopyParams);
    vecInQueue1_.FreeTensor(alphaGradInLocal);
    vecOutQueueSmall_.FreeTensor(alphaGradOutLocal);
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::ProcessV3BiasGrad()
{
    LocalTensor<uint8_t> tmpLocal = fp32TBuf_.GetWithOffset<uint8_t>(hFusionBufLen_ / 4, 0);
    constexpr bool isReuse = false;
    LocalTensor<P> biasGradInLocal = vecInQueue1_.AllocTensor<P>();
    LocalTensor<P> biasGradOutLocal = vecOutQueueSmall_.AllocTensor<P>();
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = fusionSize_ * usedVecCoreNum_ * sizeof(P);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    DataCopyPadParams dataCopyPadParams;
    dataCopyPadParams.isPad = false;
    DataCopyPad(biasGradInLocal, workSpaceGm_[workspaceBuf_.GetBiasGradOffset(0)], dataCopyParams, dataCopyPadParams);
    vecInQueue1_.EnQue(biasGradInLocal);
    biasGradInLocal = vecInQueue1_.DeQue<P>();
    uint32_t biasGradShapeSrc[] = {usedVecCoreNum_, fusionSize_};
    ReduceSum<P, AscendC::Pattern::Reduce::RA, isReuse>(biasGradOutLocal, biasGradInLocal, tmpLocal, biasGradShapeSrc,
                                                        true);
    vecOutQueueSmall_.EnQue(biasGradOutLocal);
    biasGradOutLocal = vecOutQueueSmall_.DeQue<P>();
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = fusionSize_ * sizeof(P);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    DataCopyPad(gradBiasGm_[0], biasGradOutLocal, dataCopyParams);
    vecInQueue1_.FreeTensor(biasGradInLocal);
    vecOutQueueSmall_.FreeTensor(biasGradOutLocal);
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::VFDoV3ProcessAlphaGrad(
    __ubuf__ P* alphaGradOut, __ubuf__ P* alphaGradIn)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<P> sumReg, correctionReg, adjustedReg, nextReg, errorReg;
        Reg::Duplicate(sumReg, 0);
        Reg::Duplicate(correctionReg, 0);
        uint32_t dealMask = ALPHA_GRAD_PADDING;
        Reg::MaskReg mask = Reg::UpdateMask<P>(dealMask);
        for (uint16_t vcIdx = 0; vcIdx < static_cast<uint16_t>(usedVecCoreNum_); ++vcIdx) {
            uint32_t elemOffset = vcIdx * ALPHA_GRAD_PADDING;
            Reg::RegTensor<P> alphaGradInReg;
            Reg::LoadAlign(alphaGradInReg, alphaGradIn + elemOffset);
            // 按固定遍历顺序对跨核部分和执行补偿求和。
            Reg::Sub(adjustedReg, alphaGradInReg, correctionReg, mask);
            Reg::Add(nextReg, sumReg, adjustedReg, mask);
            Reg::Sub(errorReg, nextReg, sumReg, mask);
            Reg::Sub(correctionReg, errorReg, adjustedReg, mask);
            Reg::Adds(sumReg, nextReg, ZERO, mask);
        }
        Reg::StoreAlign(alphaGradOut, sumReg, mask);
    }
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
__aicore__ inline bool MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::CanPartitionV2Rows(uint32_t nd, uint32_t cols)
{
    // ND循环只产生非空且不超过128列的块，按64列粒度处理，避免跨N分组读取行系数。
    return rowPartitionV2_ && cols % (ND_BLOCK_SIZE / NUMBER_TWO) == 0U && nd % D_ + cols <= D_;
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
template <uint16_t cols>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::PrefetchV2Rows(uint32_t nd, uint64_t row,
                                                                                         uint32_t slot)
{
    constexpr uint32_t COLS = cols;
    constexpr uint32_t ROWS = 4096U / COLS;
    auto& input = fusedHInPostSlots_[slot];
    vecInQueue0_.AllocTensor<T>(input);
    DataCopyExtParams params;
    DataCopyPadExtParams<T> pad;
    params.blockCount = ROWS;
    params.blockLen = COLS * sizeof(T);
    params.srcStride = (D_ - COLS) * sizeof(T);
    params.dstStride = 0;
    DataCopyPad(input, gradHInGm_[row * D_ + nd % D_], params, pad);
    if (withGradXPost_) {
        auto post = input[ROWS * COLS];
        params.srcStride = (nD_ - COLS) * sizeof(T);
        DataCopyPad(post, gradXPostOptionalGm_[row * nD_ + nd], params, pad);
    }
    vecInQueue0_.EnQue(input);
    auto x = vecInQueue1_.AllocTensor<T>();
    params.srcStride = (nD_ - COLS) * sizeof(T);
    DataCopyPad(x, xGm_[row * nD_ + nd], params, pad);
    vecInQueue1_.EnQue(x);
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
template <uint16_t cols>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::ProcessV2Rows(uint32_t nd, uint64_t bsStart,
                                                                                        LocalTensor<P>& sum)
{
    constexpr uint32_t COLS = cols;
    constexpr uint32_t ROWS = 4096U / COLS;
    constexpr uint32_t GAMMA_BYTES = (ROWS + 1U) * COLS * sizeof(P);
    const uint32_t rowHalf = (blockIdx_ % NUMBER_TWO) * 64U;
    // 两种列宽共用4096元素分块和双槽；NZ保留一行填充，行系数区不交叠。
    constexpr uint32_t PARTS = 64U / ROWS;
    constexpr uint32_t CHUNKS = (BS_BLOCK_SIZE / MATMUL_WRITE_OFFSET_M) * PARTS;
    gammaIn_ = fp32TBuf_.GetWithOffset<P>(COLS, 512U);
    invRmsUb_ = fp32TBuf_.GetWithOffset<P>(BS_BLOCK_SIZE, 1024U + DOUBLE_BUFFER * GAMMA_BYTES);
    hPreUb_ =
        fp32TBuf_.GetWithOffset<P>(BS_BLOCK_SIZE * N_, 1024U + DOUBLE_BUFFER * GAMMA_BYTES + BS_BLOCK_SIZE * sizeof(P));
    SetFlag<HardEvent::V_MTE2>(EVENT_ID3);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID3);
    DataCopyParams coeff;
    DataCopyPadParams pad;
    pad.isPad = false;
    coeff.blockCount = 1;
    coeff.blockLen = COLS * sizeof(P);
    coeff.srcStride = 0;
    coeff.dstStride = 0;
    DataCopyPad(gammaIn_, gammaGm_[nd], coeff, pad);
    coeff.blockLen = BS_BLOCK_SIZE * N_ * sizeof(P);
    DataCopyPad(hPreUb_, hPreGm_[bsStart * N_], coeff, pad);
    ProcessV2ChunkLoadInvRms(bsStart, BS_BLOCK_SIZE);
    if (bsStart == 0U) {
        Duplicate(sum, static_cast<P>(0), NUMBER_TWO * COLS);
        PipeBarrier<PIPE_V>();
    }
    PrefetchV2Rows<cols>(nd, bsStart + rowHalf, 0U);
    for (uint32_t chunk = 0; chunk < CHUNKS; ++chunk) {
        const uint32_t tile = chunk / PARTS;
        const uint32_t half = chunk % PARTS;
        const uint32_t slot = chunk % DOUBLE_BUFFER;
        const uint32_t row = tile * MATMUL_WRITE_OFFSET_M + rowHalf + half * ROWS;
        if (half == 0U) {
            CrossCoreWaitFlag<4, PIPE_MTE3>(tile);
            CrossCoreWaitFlag<4, PIPE_V>(CROSS_CORE_WAIT_FLAG_C0);
        }
        auto& input = fusedHInPostSlots_[slot];
        vecInQueue0_.DeQue<T>(input);
        auto x = vecInQueue1_.DeQue<T>();
        if (chunk + 1U < CHUNKS) {
            const uint32_t next = chunk + 1U;
            PrefetchV2Rows<cols>(nd, bsStart + (next / PARTS) * MATMUL_WRITE_OFFSET_M + rowHalf + (next % PARTS) * ROWS,
                                 next % DOUBLE_BUFFER);
        }
        WaitV2PackedGammaSlot(slot);
        auto gammaOut = fp32TBuf_.GetWithOffset<P>((ROWS + 1U) * COLS, 1024U + slot * GAMMA_BYTES);
        auto output = vecOutQueue1_.AllocTensor<T>();
        LocalTensor<P> c0(TPosition::VECIN, C0UbSlotOffset(tile % C0UbSlotCount()) + half * ROWS * COLS * sizeof(P),
                          ROWS * COLS);
        auto post = input[ROWS * COLS];
        auto hpre = hPreUb_[row * N_ + nd / D_];
        auto inv = invRmsUb_[row];
        if (withGradXPost_) {
            VFDoV2CompactFused<true, true, ROWS, COLS>(
                (__ubuf__ T*)output.GetPhyAddr(), (__ubuf__ P*)gammaOut.GetPhyAddr(), (__ubuf__ T*)x.GetPhyAddr(),
                (__ubuf__ T*)input.GetPhyAddr(), (__ubuf__ T*)post.GetPhyAddr(), (__ubuf__ P*)c0.GetPhyAddr(),
                (__ubuf__ P*)hpre.GetPhyAddr(), (__ubuf__ P*)inv.GetPhyAddr(), (__ubuf__ P*)gammaIn_.GetPhyAddr(),
                (__ubuf__ P*)sum.GetPhyAddr());
        } else {
            VFDoV2CompactFused<false, true, ROWS, COLS>(
                (__ubuf__ T*)output.GetPhyAddr(), (__ubuf__ P*)gammaOut.GetPhyAddr(), (__ubuf__ T*)x.GetPhyAddr(),
                (__ubuf__ T*)input.GetPhyAddr(), (__ubuf__ T*)input.GetPhyAddr(), (__ubuf__ P*)c0.GetPhyAddr(),
                (__ubuf__ P*)hpre.GetPhyAddr(), (__ubuf__ P*)inv.GetPhyAddr(), (__ubuf__ P*)gammaIn_.GetPhyAddr(),
                (__ubuf__ P*)sum.GetPhyAddr());
        }
        vecInQueue0_.FreeTensor(input);
        vecInQueue1_.FreeTensor(x);
        if (half + 1U == PARTS) {
            CrossCoreSetFlag<4, PIPE_V>(C0UbFreeFlag(tile % C0UbSlotCount()));
        }
        const event_t event = GetPackedGammaEvent(slot);
        SetFlag<HardEvent::V_MTE3>(event);
        WaitFlag<HardEvent::V_MTE3>(event);
        const uint32_t l1K = mmConfigC1_.l1K;
        constexpr uint32_t NZ_BLOCK = 32U / sizeof(P);
        const uint32_t l1SpanElems = ((COLS / NZ_BLOCK - 1U) * l1K + ROWS) * NZ_BLOCK;
        LocalTensor<P> dst(TPosition::A1,
                           tiling_->l1UsedBytes + ((row / l1K) * l1K * ND_BLOCK_SIZE + (row % l1K) * 8U) * sizeof(P),
                           l1SpanElems);
        DataCopyParams copy;
        copy.blockCount = COLS / 8U;
        copy.blockLen = ROWS;
        copy.srcStride = 1;
        copy.dstStride = l1K - ROWS;
        DataCopy(dst, gammaOut, copy);
        SetFlag<HardEvent::MTE3_V>(event);
        packedGammaPending_[slot] = true;
        vecOutQueue1_.EnQue(output);
        output = vecOutQueue1_.DeQue<T>();
        DataCopyExtParams params;
        params.blockCount = ROWS;
        params.blockLen = COLS * sizeof(T);
        params.srcStride = 0;
        params.dstStride = (nD_ - COLS) * sizeof(T);
        DataCopyPad(xGradGm_[(bsStart + row) * nD_ + nd], output, params);
        vecOutQueue1_.FreeTensor(output);
        if (half + 1U == PARTS) {
            CrossCoreSetFlag<2, PIPE_MTE3>(CROSS_CORE_WAIT_FLAG_C1);
        }
    }
    DrainV2PackedGamma();
}

} // namespace MhcPreBackward

#include "mhc_pre_backward_schedule_fp32_wave.h"

#endif // __mhc_pre_backward_KERNEL_H_
