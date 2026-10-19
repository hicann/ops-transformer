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
 * \file kernel_mla_prolog_split_n_arch92.h
 * \brief
 */

#ifndef KERNEL_MLA_PROLOG_SPLIT_N_arch92_H
#define KERNEL_MLA_PROLOG_SPLIT_N_arch92_H

#include "mla_prolog_comm_arch92.h"
#include "mla_prolog_vector_comm_arch92.h"
#include "service_matmul_arch92.h"
#include "service_rms_norm_arch92.h"
#include "service_gather_sin_cos_arch92.h"
#include "service_rotary_position_embedding_arch92.h"
#include "service_scatter_cache_arch92.h"
#include "service_dequant_arch92.h"
#include "service_dynamic_quant_qn_mul_qr_arch92.h"
#include "../mla_prolog_tiling_data.h"
#include "../mla_prolog_template_tiling_key.h"

namespace MlaProlog {
template <typename MLAPT>
class MlaPrologVecS1CubS2 {
public:
    static constexpr bool isPertile = MLAPT::isPertile;

    using mmInputType = typename MLAPT::mmInputType;
    using mmQcQrInputType = typename MLAPT::mmQcQrInputType;
    using mmQnInputType = typename MLAPT::mmQnInputType;
    using mmCqOutputType = typename MLAPT::mmCqOutputType;
    using mmCkvKrOutputType = typename MLAPT::mmCkvKrOutputType;
    using mmQcQrOutputType = typename MLAPT::mmQcQrOutputType;
    using mmQnOutputType = typename MLAPT::mmQnOutputType;
    using rmsNormGammaType = typename MLAPT::rmsNormGammaType;
    using rmsNormComputType = typename MLAPT::rmsNormComputType;
    using rmsNormCqOutputType = typename MLAPT::rmsNormCqOutputType;
    using rmsNormCkvOutputType = typename MLAPT::rmsNormCkvOutputType;
    using ropeComputType = typename MLAPT::ropeComputType;
    using ropeSinCosType = typename MLAPT::ropeSinCosType;
    using ropeOutputType = typename MLAPT::ropeOutputType;
    using kvCacheType = typename MLAPT::kvCacheType;
    using krCacheType = typename MLAPT::krCacheType;
    using queryOutputType = typename std::conditional<isPertile, bfloat16_t, typename MLAPT::kvCacheType>::type;
    using dequantScaleQNopeType = typename MLAPT::dequantScaleQNopeType;
    using dequantScaleQNormType = typename MLAPT::dequantScaleQNormType;
    using dequantScaleType = typename MLAPT::dequantScaleType;

    MMParams mmCqParam_;
    MMParams mmCkvKrParam_;
    MMParams mmQcQrParam_;
    MMParams mmQnParam_;

    __aicore__ inline MlaPrologVecS1CubS2(TPipe* pipe, const optiling::MlaPrologTilingData* __restrict tilingData,
                                          const optiling::MlaPrologBaseParams* __restrict baseParams)
        : pipe_(pipe),
          tilingData_(tilingData),
          baseParams_(baseParams)
    {}

    __aicore__ inline void Init(__gm__ uint8_t* tokenX, __gm__ uint8_t* weightDq, __gm__ uint8_t* weightUqQr,
                                __gm__ uint8_t* weightUk, __gm__ uint8_t* weightDkvKr, __gm__ uint8_t* rmsnormGammaCq,
                                __gm__ uint8_t* rmsnormGammaCkv, __gm__ uint8_t* ropeSin, __gm__ uint8_t* ropeCos,
                                __gm__ uint8_t* cacheIndex, __gm__ uint8_t* kvCache, __gm__ uint8_t* krCache,
                                __gm__ uint8_t* dequantScaleX, __gm__ uint8_t* dequantScaleWDq,
                                __gm__ uint8_t* deqScaleQcQrW, __gm__ uint8_t* dequantScaleWDkvkr,
                                __gm__ uint8_t* quantScaleCkv, __gm__ uint8_t* quantScaleCkr,
                                __gm__ uint8_t* smoothScaleCq, __gm__ uint8_t* actualSeqLen,
                                __gm__ uint8_t* kNopeClipAlpha, __gm__ uint8_t* queryOut, __gm__ uint8_t* queryRopeOut,
                                __gm__ uint8_t* dequantScaleQNopeOut, __gm__ uint8_t* queryNormOut,
                                __gm__ uint8_t* dequantScaleQNormOut, __gm__ uint8_t* workspace);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyGlobalParams();
    __aicore__ inline void OutputInit(__gm__ uint8_t* queryOut, __gm__ uint8_t* queryRopeOut,
                                      __gm__ uint8_t* dequantScaleQNopeOut, __gm__ uint8_t* queryNormOut,
                                      __gm__ uint8_t* dequantScaleQNormOut);
    __aicore__ inline void ScaleInit(__gm__ uint8_t* dequantScaleX, __gm__ uint8_t* dequantScaleWDq,
                                     __gm__ uint8_t* deqScaleQcQrW, __gm__ uint8_t* dequantScaleWDkvkr,
                                     __gm__ uint8_t* quantScaleCkv, __gm__ uint8_t* quantScaleCkr,
                                     __gm__ uint8_t* smoothScaleCq, __gm__ uint8_t* kNopeClipAlpha);
    __aicore__ inline void WorkspaceInit(__gm__ uint8_t* workspace);
    __aicore__ inline void MmParamInit();
    __aicore__ inline void MmCqParamInit();
    __aicore__ inline void MmCkvKrParamInit();
    __aicore__ inline void MmQcQrParamInit();
    __aicore__ inline void MmQnParamInit();
    __aicore__ inline void CubeBufferInit();
    __aicore__ inline void VectorBufferInit();
    __aicore__ inline void UpdateStepBatchParams(int64_t curStepBatchSize);
    __aicore__ inline void ComputeAicOffset(AicOffset& aicOffset, int64_t numHeadOffset);
    __aicore__ inline void ComputeBlkScatterOffsets(GlobalTensor<int64_t> indexGm, int64_t tokenIndex, int64_t rows,
                                                    CkvkrParams& rmsNormAndScatterCkvParams,
                                                    CkvkrParams& ropeAndScatterKrParams);
    __aicore__ inline void ComputeAivOffset(AivOffset& aivOffset, int64_t batchOffset);
    template <bool needQnDynamicQuant>
    __aicore__ inline void AicProcess(AicOffset& aicOffset, int64_t batchOffset, int64_t mmQnLoops);
    template <bool needQnDynamicQuant>
    __aicore__ inline void AivProcess(AivOffset& aivOffset, int64_t batchOffset, int64_t curStepBatchSize,
                                      int64_t numHeadOffset, int64_t mmQnLoops);
    template <typename T, typename O, typename S, bool needCheckEmptyTensor = false, bool needCheckAFullLoad = false,
              bool isContinuousCopy = true>
    __aicore__ inline void MatmulSplitN(const GlobalTensor<O>& tensorResGm, const GlobalTensor<T>& tensorAGm,
                                        const GlobalTensor<T>& tensorBGm, const MMParams& mmPara,
                                        const UsedBlockParams& mmBlockParams,
                                        const GlobalTensor<S>& tensorAScaleGm = {},
                                        const GlobalTensor<S>& tensorBScaleGm = {}, const bool b0BPre = false);
    __aicore__ inline void MatmulAndSyncQcQr(AicOffset& aicOffset, bool b0BPre = false);
    __aicore__ inline void MatmulQcQr(AicOffset& aicOffset, bool b0BPre = false);
    __aicore__ inline void MatmulQcQrLoadA(AicOffset& aicOffset);
    __aicore__ inline bool MatmulQcQrPreloadB0(AicOffset& aicOffset);
    __aicore__ inline bool MatmulMm1B0Preload(AicOffset& aicOffset);
    __aicore__ inline bool MatmulMm2B0Preload(AicOffset& aicOffset);
    __aicore__ inline void MatmulQcQrSplitN(AicOffset& aicOffset, bool isAFullLoad, uint32_t nInput,
                                            uint32_t nL1SplitSize, uint32_t nL1loops, bool b0BPre = false,
                                            bool isAResident = false);
    __aicore__ inline void PreloadQnAndSync(AicOffset& aicOffset, int64_t mmQnLoops);
    __aicore__ inline void MatmulQnWeightPreload(int64_t weightUkOffset);
    __aicore__ inline void MatmulQnHeadUb(const GlobalTensor<mmQnInputType>& tensorAGm,
                                          const GlobalTensor<mmQnInputType>& tensorBGm, const bool bPreloaded);
    template <bool needQnDynamicQuant>
    __aicore__ inline void MatmulQnSyncDynamicQuantAndMulQr(int64_t qcOffset, int64_t weightUkOffset,
                                                            int64_t qnResOffset, int64_t mmQnLoops);
    __aicore__ inline void CopyInSinCos(int64_t tokenIndex, int64_t curVecToken, int64_t batchOffset,
                                        int64_t curStepBatchSize);
    __aicore__ inline void RmsNormCq(int64_t tokenIndex, int64_t rmsNormCqOffset, int64_t rmsNormCqResOffset,
                                     int64_t curVecToken, int64_t curBlockTokenOffset);
    __aicore__ inline void RmsNormCqProcess(LocalTensor<rmsNormCqOutputType>& outputLocal,
                                            LocalTensor<dequantScaleType>& dequantScaleQcQr,
                                            LocalTensor<float>& dequantScaleXLocal, LocalTensor<uint8_t>& shareTmpUb,
                                            int64_t tokenIndex, int64_t rmsNormCqOffset, int64_t rmsNormCqResOffset,
                                            int64_t curVecTokenIdx, uint64_t dequantScaleCqElementNum,
                                            const SplitKnAccumParams& splitKnParams);
    __aicore__ inline void CopyDequantScaleCq(uint64_t dequantScaleCqElementNum,
                                              LocalTensor<dequantScaleType>& dequantScaleQcQr, int64_t curVecToken,
                                              int64_t curBlockTokenOffset);
    __aicore__ inline void CopyQueryNormScale(int64_t stepTokenIndex, LocalTensor<dequantScaleType>& dequantScaleQcQr,
                                              int64_t curVecToken);
    __aicore__ inline void RopeAndScatterKr(LocalTensor<float>& dequantScaleXLocal, LocalTensor<uint8_t>& shareTmpUb,
                                            LocalTensor<typename MLAPT::ropeComputType>& cosLocalCkvKr,
                                            LocalTensor<typename MLAPT::ropeComputType>& sinLocalCkvKr,
                                            CkvkrParams ropeAndScatterKrParams, const SplitKnAccumParams& splitKnParams,
                                            bool batch2Layout = false, bool doScatter = true);
    __aicore__ inline void ScatterKr(LocalTensor<krCacheType>& outputKrLocal, CkvkrParams ropeAndScatterKrParams);
    __aicore__ inline void RmsNormAndScatterCkv(LocalTensor<float>& dequantScaleXLocal,
                                                LocalTensor<uint8_t>& shareTmpUb,
                                                LocalTensor<typename MLAPT::ropeComputType>& cosLocalCkvKr,
                                                LocalTensor<typename MLAPT::ropeComputType>& sinLocalCkvKr,
                                                CkvkrParams rmsNormAndScatterCkvParams,
                                                const SplitKnAccumParams& splitKnParams, bool batch2Layout = false,
                                                bool doScatter = true);
    __aicore__ inline void EmitCkvkrWrites();
    __aicore__ inline void RmsNormAndQuantizeCkv(LocalTensor<kvCacheType>& ckvOutLocal,
                                                 LocalTensor<uint8_t>& ckvNormTmpUb,
                                                 LocalTensor<float>& dequantScaleXLocal, RmsNormParam ckvNormParams,
                                                 CkvkrParams rmsNormAndScatterCkvParams,
                                                 const SplitKnAccumParams& splitKnParams);
    __aicore__ inline void ScatterCkv(LocalTensor<kvCacheType>& outputLocal, CkvkrParams rmsNormAndScatterCkvParams);
    __aicore__ inline void RmsNormRopeScatterCkvKr(int64_t tokenIndex, int64_t rmsNormCkvOffset, int64_t ropeKrOffset,
                                                   int64_t curVecToken, int64_t batchOffset);
    __aicore__ inline void RopeQr(int64_t ropeQrOffset, int64_t ropeQrResOffset, int64_t curVecToken,
                                  int64_t curBlockTokenOffset);
    __aicore__ inline void DequantQc(int64_t mmQnPreDequantOffset, int64_t mmQnPreDequantResOffset, int64_t curVecToken,
                                     int64_t curBlockTokenOffset);
    __aicore__ inline void CastQc(int64_t mmQnPreCastOffset, int64_t mmQnPreCastResOffset, int64_t curVecToken);
    // 低时延算力分组场景
    template <bool needQnDynamicQuant>
    __aicore__ inline void DequantQcAndRopeQc(AivOffset& aivOffset, int64_t batchOffset, int64_t curStepBatchSize,
                                              int64_t numHeadOffset, int64_t mmQnLoops);
    __aicore__ inline void DequantQcSplitNGroupCase(int64_t mmQnPreDequantOffset, int64_t mmQnPreDequantResOffset,
                                                    int64_t qcQrScaleOffset);
    __aicore__ inline void RopeQrSplitNGroupCase(int64_t ropeQrOffset, int64_t ropeQrResOffset);
    __aicore__ inline void DequantQcQrSplitN(
        const DequantQcQrSplitNParams& dequantQcQrSplitN,
        const LocalTensor<mmQcQrOutputType>& waveUb = LocalTensor<mmQcQrOutputType>(), const uint32_t wavePitch = 0);
    __aicore__ inline void CastQcQrSplitN(const CastQcQrSplitNParams& castQcQrSplitN,
                                          const LocalTensor<mmQcQrOutputType>& waveUb = LocalTensor<mmQcQrOutputType>(),
                                          const uint32_t wavePitch = 0);
    __aicore__ inline void RopeQrSplitN(const RopeQrSplitNParams& ropeQrSplitNParams,
                                        const LocalTensor<mmQcQrOutputType>& waveUbQr = LocalTensor<mmQcQrOutputType>(),
                                        const uint32_t wavePitch = 0);
    __aicore__ inline void DequantAndRopeSplitNSyncMMQcQr(int64_t mmQnPreDequantOffset, int64_t mmQnPreDequantResOffset,
                                                          int64_t ropeQrOffset, int64_t ropeQrResOffset);
    __aicore__ inline void DequantRopeSplitNDequantLoop(
        int64_t mmQnPreDequantOffset, int64_t mmQnPreDequantResOffset, uint32_t colQc, uint32_t colQr,
        uint32_t colOffsetCube, uint32_t srcStride, uint32_t dstStride, uint32_t& inputOffset, uint32_t& outputOffset,
        uint32_t totalDequantLoops, bool needSparseSync, uint32_t& colOffsetVec, uint32_t& dequantLoopCount,
        const LocalTensor<mmQcQrOutputType>& waveUb = LocalTensor<mmQcQrOutputType>(), const uint32_t wavePitch = 0);
    __aicore__ inline void CopyQcWaveSplitNLoop(int64_t mmQnPreDequantResOffset, uint32_t colQc, uint32_t colQr,
                                                uint32_t colOffsetCube, uint32_t dstStride, uint32_t& inputOffset,
                                                uint32_t& outputOffset, uint32_t totalDequantLoops, bool needSparseSync,
                                                uint32_t& colOffsetVec, uint32_t& dequantLoopCount,
                                                const LocalTensor<mmQcQrOutputType>& waveUb, const uint32_t wavePitch);
    __aicore__ inline void DequantRopeSplitNRopeLoop(
        int64_t ropeQrOffset, int64_t ropeQrResOffset, uint32_t colQc, uint32_t colQr, uint32_t colOffsetCube,
        uint32_t ropeDstStride, uint32_t ropeCnt, int64_t ropeStride, uint32_t& inputOffsetRope,
        uint32_t& outputOffsetRope, uint32_t& deqScaleOffset, uint32_t deQuantScaleCqOffset, uint32_t& colOffsetRope,
        const LocalTensor<mmQcQrOutputType>& waveUb = LocalTensor<mmQcQrOutputType>(), const uint32_t wavePitch = 0);
    __aicore__ inline void DynamicQuantQnAndMulQrSyncMMQn(int64_t batchOffset, int64_t curStepBatchSize,
                                                          int64_t numHeadOffset, int64_t mmQnLoops);

    TPipe* pipe_;
    const optiling::MlaPrologTilingData* __restrict tilingData_;
    const optiling::MlaPrologBaseParams* __restrict baseParams_;
    uint32_t blockIdx_ = 0U;
    uint32_t cubeBlockIdx_ = 0U; // AIV上使用AIC的blockIdx
    int64_t vectorRow_ = 1;
    int64_t curVectorBlockNum_;
    int64_t vectorCoreNum_;
    uint64_t dequantScaleCqSize_ = 1;
    uint32_t curStepVecFrontToken_;
    uint32_t curStepVecFrontListNum_;
    uint32_t curStepVecBackToken_;
    uint32_t curVecTokenMax_;
    bool enableSmoothScalesCq_;
    static constexpr uint32_t cvMode = MLAPT::cvRatio;
    static constexpr bool isFp8E8m0 = std::is_same<dequantScaleType, FP8E8M0>::value;

    static constexpr bool nqWaveMode =
        MLA_NQ_WAVE_ENABLE && MLA_USE_UB_DIRECT && std::is_same<mmInputType, bfloat16_t>::value &&
        std::is_same<mmQcQrInputType, bfloat16_t>::value && std::is_same<dequantScaleType, float>::value;

    uint32_t cvRatio_ = 2U;
    uint32_t ubShareBytes_ = 0U;

    TBuf<TPosition::VECCALC> cvSharedUb_;
    uint32_t mm3WaveSeq_ = 0;
    uint32_t mm3WaveSeqAiv_ = 0;

    bool ckvkrBatchDrainPending_ = false;

    CkvkrParams deferredCkvParams_[2] = {};
    CkvkrParams deferredKrParams_[2] = {};

    LocalTensor<uint8_t> ckvkrDeferBase_;
    uint32_t ckvkrDeferNum_ = 0U;
    bool ckvkrDeferArmed_ = false;

    TBuf<TPosition::VECCALC> qcqrScaleBuf_;
    LocalTensor<float> qcqrScaleLocal_;

    __aicore__ inline uint32_t CkvkrWriteSpan()
    {
        const uint32_t rowSize = static_cast<uint32_t>(vectorRow_ * baseParams_->headSizeCkv * sizeof(kvCacheType));
        return static_cast<uint32_t>(Align(rowSize * 4, static_cast<uint32_t>(ALIGN_BLOCK_SIZE)));
    }
    __aicore__ inline uint32_t PostCkvkrShareBase()
    {
        if constexpr (MLA_CKVKR_BATCH2) {
            const uint32_t preamble =
                static_cast<uint32_t>(FP32_BLOCK_ELEMENT_NUM * sizeof(float) +
                                      2 * baseParams_->dimHeadRope * curVecTokenMax_ * sizeof(ropeComputType));
            return static_cast<uint32_t>(Align(preamble + CkvkrWriteSpan(), static_cast<uint32_t>(ALIGN_BLOCK_SIZE)));
        }
        return 0;
    }

    __aicore__ inline bool IsMxWaveCapable() const
    {
        if (mmQcQrParam_.baseN != (baseParams_->dimHeadSizeQc + baseParams_->dimHeadRope)) {
            return false;
        }
        const uint32_t m = baseParams_->stepBatchSize;
        const uint32_t colQcW = baseParams_->dimHeadSizeQc;
        const uint32_t colQrW = baseParams_->dimHeadRope;
        const uint64_t dequantPhase = static_cast<uint64_t>(m) * colQcW * 4U + 2048U; // Cast 的 fp32 输入 + 槽余量
        const uint64_t ropePhase = static_cast<uint64_t>(m) * colQcW * 2U +
                                   static_cast<uint64_t>(m) * colQrW * 6U; // 2×sFp(4B)+2×sOut(2B), ropeCnt=m/2
        uint64_t scratchPeak = (dequantPhase > ropePhase) ? dequantPhase : ropePhase;
        if (baseParams_->mm1KGroup > 1U && baseParams_->mm1KGroup <= 4U) {
            const uint64_t gaPhase = 2ULL * (baseParams_->mm1KGroup + 1U) * baseParams_->headSizeCq * 4U + 6144U;
            if (gaPhase > scratchPeak) {
                scratchPeak = gaPhase;
            }
        }
        const uint64_t cvUbBytes = 2ULL * m * mmQcQrParam_.baseN * sizeof(mmQcQrOutputType);
        const uint64_t fixedEst = 2ULL * colQrW * 4U * m + 2ULL * (baseParams_->headSizeCq + baseParams_->headSizeCkv) +
                                  static_cast<uint64_t>(m + 7U) * 32U + 8192U;
        return cvUbBytes + scratchPeak + fixedEst <= MAX_UB_SIZE;
    }

    struct DequantTool {
        GlobalTensor<dequantScaleType> deQuantScaleCqGm_;
        TBuf<TPosition::VECCALC> deQuantScaleCqBuffer_; // 用于临时存储每一行的Scale，以及汇总最终每一行的Scale参数
        LocalTensor<dequantScaleType> deQuantScaleCqLocal_;
        __aicore__ inline DequantTool() {}
    };

    // 算子分组开关
    DequantTool dequantTool_;

    // GM
    GlobalTensor<mmInputType> tokenXGm_;
    GlobalTensor<mmInputType> weightDqGm_;
    GlobalTensor<mmQcQrInputType> weightUqQrGm_;
    GlobalTensor<mmQnInputType> weightUkGm_;
    GlobalTensor<mmInputType> weightDkvKrGm_;
    GlobalTensor<int64_t> cacheIndexGm_;
    GlobalTensor<rmsNormGammaType> rmsnormGammaCqGm_;
    GlobalTensor<rmsNormGammaType> rmsnormGammaCkvGm_;
    GlobalTensor<ropeSinCosType> ropeSinGm_;
    GlobalTensor<ropeSinCosType> ropeCosGm_;
    GlobalTensor<kvCacheType> kvCacheGm_;
    GlobalTensor<krCacheType> krCacheGm_;
    GlobalTensor<ropeOutputType> qrOutGm_;

    GlobalTensor<dequantScaleType> dequantScaleXGm_;
    GlobalTensor<dequantScaleType> dequantScaleWDqGm_;
    GlobalTensor<dequantScaleType> dequantScaleWDkvkrGm_;
    GlobalTensor<dequantScaleType> deqScaleQcQrW_; // per-channel反量化参数
    GlobalTensor<float> smoothScaleCqGm_;
    GlobalTensor<float> quantScaleCkvGm_;
    GlobalTensor<float> quantScaleCkrGm_;

    GlobalTensor<int32_t> actualSeqLenGm_;
    GlobalTensor<float> kNopeClipAlphaGm_;

    GlobalTensor<rmsNormCqOutputType> rmsNormCqResGm_;
    GlobalTensor<mmCqOutputType> mmCqResGm_;
    GlobalTensor<mmCkvKrOutputType> mmCkvKrResGm_;
    GlobalTensor<float> mmCqResFloatGm_;
    GlobalTensor<float> mmCkvKrResFloatGm_;
    GlobalTensor<mmQcQrOutputType> mmQcQrResGm_;
    GlobalTensor<mmQnInputType> mmQcQrResDequantGm_;
    GlobalTensor<mmQnOutputType> mmQnResGm_;
    GlobalTensor<queryOutputType> queryOutGm_;
    GlobalTensor<dequantScaleQNopeType> dequantScaleQNopeGm_;
    GlobalTensor<dequantScaleQNormType> dequantScaleQNormGm_;

    bool nqWaveTokenPending_ = false;
    bool mm1B0PreNext_ = false;

    // UB
    TBuf<TPosition::VECCALC> sincosBuffer_;
    TBuf<TPosition::VECCALC> shareBuffer_;
    TBuf<TPosition::VECCALC> dequantScaleWDqBuffer_;
    TBuf<TPosition::VECCALC> dequantScaleWDkvKrBuffer_;
    TBuf<TPosition::VECCALC> rmsnormGammaCqBuffer_;
    TBuf<TPosition::VECCALC> rmsnormGammaCkvBuffer_;
    TBuf<TPosition::VECCALC> smoothScaleCqBuffer_;
    TBuf<TPosition::VECCALC> quantScaleCkrBuffer_;
    TBuf<TPosition::VECCALC> quantScaleCkvBuffer_;
    TBuf<TPosition::VECCALC> stepActualSeqBuffer_;

    LocalTensor<ropeComputType> cosLocal_;
    LocalTensor<ropeComputType> sinLocal_;
    LocalTensor<float> dequantScaleWDkvKrLocal_;
    LocalTensor<float> dequantScaleWDqLocal_;
    LocalTensor<rmsNormGammaType> rmsnormGammaCqLocal_;
    LocalTensor<rmsNormGammaType> rmsnormGammaCkvLocal_;
    LocalTensor<float> smoothScaleCqLocal_;
    LocalTensor<float> quantScaleCkvLocal_;
    LocalTensor<float> quantScaleCkrLocal_;
    LocalTensor<int64_t> stepActualSeqLocal_;

    struct ActSeqState {
        int64_t curIndexOffset = 0;  // prefix sum of blocks up to curBatch
        int64_t prevIndexOffset = 0; // prefix sum of blocks up to (curBatch - 1)
        int64_t curPrefix = 0;       // prefix sum of seq lengths up to curBatch (exclusive high bound)
        int64_t prevPrefix = 0;      // prefix sum of seq lengths up to (curBatch - 1)
        int64_t curBatch = 0;        // current batch index for the running token index
        bool inited = false;         // whether state is initialized for the run
    };
    ActSeqState actSeqState_;
    int64_t blockSpanPerBatch_ = 0; // blockNum * blockSize

    TBuf<TPosition::A1> aBufL1_;
    TBuf<TPosition::B1> bBufL1_;
    LocalTensor<mmInputType> aL1Tensor_;
    LocalTensor<mmInputType> bL1Tensor_;
    MMBufParams bufParam_;
    TBuf<TPosition::A2> aBufL0_;
    TBuf<TPosition::B2> bBufL0_;
    TBuf<TPosition::CO1> cBufL0_;
};

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::Init(
    __gm__ uint8_t* tokenX, __gm__ uint8_t* weightDq, __gm__ uint8_t* weightUqQr, __gm__ uint8_t* weightUk,
    __gm__ uint8_t* weightDkvKr, __gm__ uint8_t* rmsnormGammaCq, __gm__ uint8_t* rmsnormGammaCkv,
    __gm__ uint8_t* ropeSin, __gm__ uint8_t* ropeCos, __gm__ uint8_t* cacheIndex, __gm__ uint8_t* kvCache,
    __gm__ uint8_t* krCache, __gm__ uint8_t* dequantScaleX, __gm__ uint8_t* dequantScaleWDq,
    __gm__ uint8_t* deqScaleQcQrW, __gm__ uint8_t* dequantScaleWDkvkr, __gm__ uint8_t* quantScaleCkv,
    __gm__ uint8_t* quantScaleCkr, __gm__ uint8_t* smoothScaleCq, __gm__ uint8_t* actualSeqLen,
    __gm__ uint8_t* kNopeClipAlpha, __gm__ uint8_t* queryOut, __gm__ uint8_t* queryRopeOut,
    __gm__ uint8_t* dequantScaleQNopeOut, __gm__ uint8_t* queryNormOut, __gm__ uint8_t* dequantScaleQNormOut,
    __gm__ uint8_t* workspace)
{
    cvRatio_ = GetSubBlockNum();
    blockIdx_ = GetBlockIdx();
    if ASCEND_IS_AIC {
        cubeBlockIdx_ = blockIdx_;
    } else {
        cubeBlockIdx_ = blockIdx_ / cvRatio_;
    }
    curVectorBlockNum_ = static_cast<int64_t>(baseParams_->stepBatchSize);
    vectorCoreNum_ = static_cast<int64_t>(baseParams_->vectorBlockNum);
    if (cvMode == 1 && cvRatio_ == 2) {
        if (vectorCoreNum_ < curVectorBlockNum_) {
            vectorCoreNum_ = vectorCoreNum_ * 2;
        }
    }
    curVecTokenMax_ = (curVectorBlockNum_ + vectorCoreNum_ - 1) / vectorCoreNum_;
    enableSmoothScalesCq_ = smoothScaleCq == nullptr ? false : true;
    // GM
    tokenXGm_.SetGlobalBuffer((__gm__ mmInputType*)tokenX);
    weightDqGm_.SetGlobalBuffer((__gm__ mmInputType*)weightDq);         // NZ
    weightUqQrGm_.SetGlobalBuffer((__gm__ mmQcQrInputType*)weightUqQr); // NZ
    weightUkGm_.SetGlobalBuffer((__gm__ mmQnInputType*)weightUk);
    weightDkvKrGm_.SetGlobalBuffer((__gm__ mmInputType*)weightDkvKr); // NZ
    rmsnormGammaCqGm_.SetGlobalBuffer((__gm__ rmsNormGammaType*)rmsnormGammaCq);
    rmsnormGammaCkvGm_.SetGlobalBuffer((__gm__ rmsNormGammaType*)rmsnormGammaCkv);
    ropeCosGm_.SetGlobalBuffer((__gm__ ropeSinCosType*)ropeCos);
    ropeSinGm_.SetGlobalBuffer((__gm__ ropeSinCosType*)ropeSin);
    if constexpr (MLAPT::cacheMode != CACHE_MODE::ND) {
        cacheIndexGm_.SetGlobalBuffer((__gm__ int64_t*)cacheIndex);
    }
    krCacheGm_.SetGlobalBuffer((__gm__ krCacheType*)krCache);
    kvCacheGm_.SetGlobalBuffer((__gm__ kvCacheType*)kvCache);
    if constexpr (MLAPT::actualSeqMode == ACTUAL_SEQ_MODE::EN_Q_LEN) {
        actualSeqLenGm_.SetGlobalBuffer((__gm__ int32_t*)actualSeqLen);
    }

    OutputInit(queryOut, queryRopeOut, dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut);
    ScaleInit(dequantScaleX, dequantScaleWDq, deqScaleQcQrW, dequantScaleWDkvkr, quantScaleCkv, quantScaleCkr,
              smoothScaleCq, kNopeClipAlpha);
    MmParamInit();
    WorkspaceInit(workspace);
    if ASCEND_IS_AIV {
        VectorBufferInit();
    } else {
        CubeBufferInit();
    }
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::OutputInit(__gm__ uint8_t* queryOut, __gm__ uint8_t* queryRopeOut,
                                                              __gm__ uint8_t* dequantScaleQNopeOut,
                                                              __gm__ uint8_t* queryNormOut,
                                                              __gm__ uint8_t* dequantScaleQNormOut)
{
    if constexpr (((std::is_same<mmInputType, int8_t>::value && std::is_same<kvCacheType, int8_t>::value) ||
                   (std::is_same<mmInputType, FP8E4M3>::value && std::is_same<kvCacheType, FP8E4M3>::value) ||
                   (std::is_same<mmInputType, HIF8>::value && std::is_same<kvCacheType, HIF8>::value)) &&
                  !isPertile) {
        queryOutGm_.SetGlobalBuffer((__gm__ queryOutputType*)queryOut);
        dequantScaleQNopeGm_.SetGlobalBuffer((__gm__ dequantScaleQNopeType*)dequantScaleQNopeOut);
    } else {
        mmQnResGm_.SetGlobalBuffer((__gm__ mmQnOutputType*)queryOut);
    }
    qrOutGm_.SetGlobalBuffer((__gm__ ropeOutputType*)queryRopeOut);
    if (baseParams_->queryNormFlag == 1U) {
        if constexpr (IsFullQuantMode<mmQcQrInputType, dequantScaleType, false>()) {
            dequantScaleQNormGm_.SetGlobalBuffer((__gm__ dequantScaleQNormType*)dequantScaleQNormOut);
        }
        rmsNormCqResGm_.SetGlobalBuffer((__gm__ mmQcQrInputType*)queryNormOut);
    }
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::ScaleInit(
    __gm__ uint8_t* dequantScaleX, __gm__ uint8_t* dequantScaleWDq, __gm__ uint8_t* deqScaleQcQrW,
    __gm__ uint8_t* dequantScaleWDkvkr, __gm__ uint8_t* quantScaleCkv, __gm__ uint8_t* quantScaleCkr,
    __gm__ uint8_t* smoothScaleCq, __gm__ uint8_t* kNopeClipAlpha)
{
    if constexpr (IsFullQuantMode<mmInputType, dequantScaleType, false>()) {
        dequantScaleXGm_.SetGlobalBuffer((__gm__ dequantScaleType*)dequantScaleX);
        dequantScaleWDqGm_.SetGlobalBuffer((__gm__ dequantScaleType*)dequantScaleWDq);
        dequantScaleWDkvkrGm_.SetGlobalBuffer((__gm__ dequantScaleType*)dequantScaleWDkvkr);
    }

    if constexpr (IsFullQuantMode<mmQcQrInputType, dequantScaleType, true>()) {
        smoothScaleCqGm_.SetGlobalBuffer((__gm__ float*)smoothScaleCq);
        deqScaleQcQrW_.SetGlobalBuffer((__gm__ dequantScaleType*)deqScaleQcQrW);
        quantScaleCkvGm_.SetGlobalBuffer((__gm__ float*)quantScaleCkv);
        quantScaleCkrGm_.SetGlobalBuffer((__gm__ float*)quantScaleCkr);
    } else if constexpr (std::is_same<mmQcQrInputType, FP8E4M3>::value && isFp8E8m0) {
        deqScaleQcQrW_.SetGlobalBuffer((__gm__ dequantScaleType*)deqScaleQcQrW);
        quantScaleCkvGm_.SetGlobalBuffer((__gm__ float*)quantScaleCkv);
    }

    if constexpr (isPertile) {
        kNopeClipAlphaGm_.SetGlobalBuffer((__gm__ float*)kNopeClipAlpha);
    }
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::MmParamInit()
{
    MmCqParamInit();
    MmCkvKrParamInit();
    MmQcQrParamInit();
    MmQnParamInit();
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::MmCqParamInit()
{
    mmCqParam_.m = baseParams_->stepBatchSize; // 32
    mmCqParam_.needSetOrgShape = 1;
    mmCqParam_.baseK =
        (sizeof(mmInputType) == ONE_BYTE_TYPE_SIZE) ? 256 : 128; // 128KB / (128 max baseN * 4 stepK * sizeof(type))
    mmCqParam_.baseN = (std::is_same<mmInputType, FP8E4M3>::value && isFp8E8m0) ? 64 : 128;

    if (baseParams_->mm1KGroup > 1) {                         // MM1切KN: 部分和[kg][step][Hcq]
        uint32_t nGroup = blockIdx_ % baseParams_->mm1NGroup; // 主用例 0..3
        uint32_t kGroup = blockIdx_ / baseParams_->mm1NGroup; // 主用例 0..15
        uint32_t n = baseParams_->mm1SingleCoreN;
        if (nGroup == baseParams_->mm1NGroup - 1) { // 尾N组
            n = baseParams_->headSizeCq - baseParams_->mm1SingleCoreN * nGroup;
        }
        uint32_t k = baseParams_->mm1SingleCoreK;
        if (kGroup == baseParams_->mm1KGroup - 1) { // 尾K组(余数兜底)
            k = baseParams_->headSizeX - baseParams_->mm1SingleCoreK * kGroup;
        }
        mmCqParam_.n = n;
        mmCqParam_.k = k;
        mmCqParam_.kStart = kGroup * baseParams_->mm1SingleCoreK; // 本核 k 起点
        mmCqParam_.orgKa = baseParams_->headSizeX;
        mmCqParam_.orgKb = baseParams_->headSizeX;
    } else { // 原切N回退(host 门控不通过/搜索失败): K 全长不切, 原版行为
        if (cubeBlockIdx_ == baseParams_->mm1BlockNum - 1) {
            mmCqParam_.n = baseParams_->headSizeCq - baseParams_->mm1SingleCoreN * cubeBlockIdx_;
        } else {
            mmCqParam_.n = baseParams_->mm1SingleCoreN; // 1536 / 24 = 64
        }
        mmCqParam_.k = baseParams_->headSizeX; // 7168
        mmCqParam_.kStart = 0;
        mmCqParam_.orgKa = mmCqParam_.k;
        mmCqParam_.orgKb = mmCqParam_.k;
    }
    mmCqParam_.orgM = mmCqParam_.m;
    mmCqParam_.orgN = mmCqParam_.n;
    mmCqParam_.orgKc = baseParams_->headSizeCq;

    if (baseParams_->mm1KGroup > 1) {
        constexpr bool isMxL1BHalf = std::is_same<mmInputType, FP8E4M3>::value && isFp8E8m0;
        constexpr uint32_t l1BBudgetMx = isMxL1BHalf ? (L1_B_SIZE / 2) : L1_B_SIZE;
        uint32_t mSize = Align(mmCqParam_.m, BLOCK_CUBE_SIZE);
        uint32_t maxStepK_A = L1_A_SIZE / (mSize * mmCqParam_.baseK * sizeof(mmInputType));
        uint32_t maxStepK_B = l1BBudgetMx / (mmCqParam_.baseN * mmCqParam_.baseK * sizeof(mmInputType));
        uint32_t maxStepK = (maxStepK_A < maxStepK_B) ? maxStepK_A : maxStepK_B;
        uint32_t actStepK = CeilDiv(mmCqParam_.k, mmCqParam_.baseK);
        mmCqParam_.stepK = (maxStepK < actStepK) ? maxStepK : actStepK;
        if (mmCqParam_.stepK < 1) {
            mmCqParam_.stepK = 1;
        }
    } else {
        mmCqParam_.stepK = 4;
        if ((mmCqParam_.k / mmCqParam_.baseK) % mmCqParam_.stepK != 0) {
            mmCqParam_.stepK = 3; // support k = 7680, mmInputType int8, no tail
        }
    }
    mmCqParam_.kScale = mmCqParam_.k / FP8_E4M3_BLOCK_SIZE;
    mmCqParam_.kL1StepSize = mmCqParam_.baseK * mmCqParam_.stepK;
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::MmCkvKrParamInit()
{
    mmCkvKrParam_.m = baseParams_->stepBatchSize; // 128
    mmCkvKrParam_.needSetOrgShape = 1;
    mmCkvKrParam_.baseK =
        (sizeof(mmInputType) == ONE_BYTE_TYPE_SIZE) ? 256 : 128; // 128KB / (128 max baseN * 4 stepK * sizeof(type))
    mmCkvKrParam_.baseN = (std::is_same<mmInputType, FP8E4M3>::value && isFp8E8m0) ? 64 : 128;

    if (baseParams_->mm2KGroup > 1) {
        uint32_t nGroup = blockIdx_ % baseParams_->mm2NGroup;
        uint32_t kGroup = blockIdx_ / baseParams_->mm2NGroup;
        uint32_t n = baseParams_->mm2SingleCoreN;
        if (nGroup == baseParams_->mm2NGroup - 1) { // 尾N组
            n = baseParams_->headSizeCkv + baseParams_->headSizeKr - baseParams_->mm2SingleCoreN * nGroup;
        }
        uint32_t k = baseParams_->mm2SingleCoreK;
        if (kGroup == baseParams_->mm2KGroup - 1) { // 尾K组
            k = baseParams_->headSizeX - baseParams_->mm2SingleCoreK * kGroup;
        }
        mmCkvKrParam_.n = n;
        mmCkvKrParam_.k = k;
        mmCkvKrParam_.kStart = kGroup * baseParams_->mm2SingleCoreK; // 本核 K 起点(A: GM 直接加, B: NZ ×C0)
        mmCkvKrParam_.orgKa = baseParams_->headSizeX;
        mmCkvKrParam_.orgKb = baseParams_->headSizeX;
    } else { // 原切N路径(门控回退: int8/mxfp8/bf16; K 全长不切)
        if (cubeBlockIdx_ == baseParams_->mm2BlockNum - 1) {
            mmCkvKrParam_.n =
                baseParams_->headSizeCkv + baseParams_->headSizeKr - baseParams_->mm2SingleCoreN * cubeBlockIdx_;
        } else {
            mmCkvKrParam_.n = baseParams_->mm2SingleCoreN;
        }
        mmCkvKrParam_.k = baseParams_->headSizeX; // 7168
        mmCkvKrParam_.kStart = 0;
        mmCkvKrParam_.orgKa = mmCkvKrParam_.k;
        mmCkvKrParam_.orgKb = mmCkvKrParam_.k;
    }
    mmCkvKrParam_.orgM = mmCkvKrParam_.m;
    mmCkvKrParam_.orgN = mmCkvKrParam_.n;
    mmCkvKrParam_.orgKc = baseParams_->headSizeCkv + baseParams_->dimHeadRope; // 576: 部分和全行宽(FixPipe dstStride)

    if (baseParams_->mm2KGroup > 1) {
        constexpr bool isMxL1BHalf2 = std::is_same<mmInputType, FP8E4M3>::value && isFp8E8m0;
        constexpr uint32_t l1BBudgetMx2 = isMxL1BHalf2 ? (L1_B_SIZE / 2) : L1_B_SIZE;
        uint32_t mSize = Align(mmCkvKrParam_.m, BLOCK_CUBE_SIZE);
        uint32_t maxStepK_A = L1_A_SIZE / (mSize * mmCkvKrParam_.baseK * sizeof(mmInputType));
        uint32_t maxStepK_B = l1BBudgetMx2 / (mmCkvKrParam_.baseN * mmCkvKrParam_.baseK * sizeof(mmInputType));
        uint32_t maxStepK = (maxStepK_A < maxStepK_B) ? maxStepK_A : maxStepK_B;
        uint32_t actStepK = CeilDiv(mmCkvKrParam_.k, mmCkvKrParam_.baseK); // CeilDiv(224,256)=1
        mmCkvKrParam_.stepK = (maxStepK < actStepK) ? maxStepK : actStepK;
        if (mmCkvKrParam_.stepK < 1) {
            mmCkvKrParam_.stepK = 1;
        }
    } else { // 原启发式(k=7168: 28%4==0 → stepK=4, kL1StepSize=1024, kL1Loops=7)
        mmCkvKrParam_.stepK = 4;
        if ((mmCkvKrParam_.k / mmCkvKrParam_.baseK) % mmCkvKrParam_.stepK != 0) {
            mmCkvKrParam_.stepK = 3; // support k = 7680, mmInputType int8, no tail
        }
    }
    mmCkvKrParam_.kL1StepSize = mmCkvKrParam_.baseK * mmCkvKrParam_.stepK;
    mmCkvKrParam_.kScale = mmCkvKrParam_.k / FP8_E4M3_BLOCK_SIZE;
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::MmQcQrParamInit()
{
    mmQcQrParam_.m = baseParams_->stepBatchSize; // 32
    if constexpr (MLAPT::enableGroupComputeOpt) {
        // 算力分组仅考虑G8 Qc Qr(8+4核), n固定128, 这里不处理尾核
        mmQcQrParam_.n = baseParams_->mm3SingleCoreN;
    } else {
        if (cubeBlockIdx_ == baseParams_->mm3BlockNum - 1) {
            mmQcQrParam_.n =
                baseParams_->headSizeQc + baseParams_->headSizeQr - baseParams_->mm3SingleCoreN * cubeBlockIdx_;
        } else {
            mmQcQrParam_.n = baseParams_->mm3SingleCoreN;
        }
    }

    mmQcQrParam_.k = baseParams_->headSizeCq; // 1536
    mmQcQrParam_.needSetOrgShape = 1;
    mmQcQrParam_.orgM = mmQcQrParam_.m;
    mmQcQrParam_.orgN = mmQcQrParam_.n;
    mmQcQrParam_.orgKa = mmQcQrParam_.k;
    mmQcQrParam_.orgKb = mmQcQrParam_.k;
    mmQcQrParam_.orgKc = (baseParams_->headSizeQc + baseParams_->headSizeQr); // (128 * 32 + 64 * 32)
    mmQcQrParam_.baseK = (sizeof(mmQcQrInputType) == ONE_BYTE_TYPE_SIZE) ? 128 : 64;
    if constexpr (MLAPT::enableGroupComputeOpt) {
        mmQcQrParam_.baseN = 128;
    } else {
        if constexpr (std::is_same<mmInputType, FP8E4M3>::value && isFp8E8m0) {
            mmQcQrParam_.baseN = static_cast<uint32_t>(baseParams_->dimHeadSizeQc + baseParams_->dimHeadRope);
        } else {
            mmQcQrParam_.baseN = static_cast<uint32_t>(baseParams_->dimHeadSizeQc + baseParams_->dimHeadRope);
        }
    }
    if constexpr (std::is_same<mmInputType, FP8E4M3>::value && isFp8E8m0) {
        if constexpr (!MLAPT::enableGroupComputeOpt) {
            mmQcQrParam_.stepK = 2;
        } else {
            mmQcQrParam_.stepK = MLA_MM3_STEP_K;
        }
    } else {
        mmQcQrParam_.stepK = MLA_MM3_STEP_K;
    }
    mmQcQrParam_.kL1StepSize = mmQcQrParam_.baseK * mmQcQrParam_.stepK;
    mmQcQrParam_.kScale = mmQcQrParam_.k / FP8_E4M3_BLOCK_SIZE;
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::MmQnParamInit()
{
    mmQnParam_.m = baseParams_->stepBatchSize; // 32
    mmQnParam_.n = baseParams_->headSizeCkv;   // 512
    mmQnParam_.k = baseParams_->dimHeadSizeQc; // 128, 这里numHeadSize被分核，matmul设置里不体现
    mmQnParam_.needSetOrgShape = 1;
    mmQnParam_.orgN = mmQnParam_.n;
    mmQnParam_.orgM = mmQnParam_.m;
    if constexpr (std::is_same<mmQcQrOutputType, int32_t>::value || std::is_same<mmQcQrOutputType, float>::value) {
        mmQnParam_.orgKa = baseParams_->headSizeQc;
    } else {
        mmQnParam_.orgKa = (nqWaveMode && (cvRatio_ == 1)) ? baseParams_->headSizeQc :
                                                             (baseParams_->headSizeQc + baseParams_->headSizeQr);
    }
    mmQnParam_.orgKc = baseParams_->headSizeCkv * baseParams_->numHeadSize;
    mmQnParam_.orgKb = baseParams_->dimHeadSizeQc;
    mmQnParam_.baseK = 128;
    mmQnParam_.baseN = 128;
    mmQnParam_.stepK = 1;
    if ((mmQnParam_.k > mmQnParam_.baseK) && (mmQnParam_.k % mmQnParam_.baseK != 0)) {
        mmQnParam_.baseK = 64;
        mmQnParam_.stepK = 3; // support D = 192
    }
    mmQnParam_.kScale = mmQnParam_.k / FP8_E4M3_BLOCK_SIZE;
    mmQnParam_.kL1StepSize = mmQnParam_.baseK * mmQnParam_.stepK;
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::VectorBufferInit()
{
    uint64_t usedBytes = 0;
    if (MLA_USE_UB_DIRECT && (MLAPT::enableDequantOpt || nqWaveMode) && !MLAPT::enableGroupComputeOpt &&
        (!isFp8E8m0 || IsMxWaveCapable())) {
        uint64_t cvUbSize =
            2ULL * static_cast<uint64_t>(baseParams_->stepBatchSize) * mmQcQrParam_.baseN * sizeof(mmQcQrOutputType);
        pipe_->InitBuffer(cvSharedUb_, cvUbSize);
        usedBytes += cvUbSize;
    }
    if constexpr (MLA_PRELOAD_QCQR_SCALE && MLA_USE_UB_DIRECT && MLAPT::enableDequantOpt && !isFp8E8m0 &&
                  !MLAPT::enableGroupComputeOpt && IsFullQuantMode<mmQcQrInputType, dequantScaleType, true>()) {
        uint64_t qcqrScaleSize = static_cast<uint64_t>(mmQcQrParam_.n) * sizeof(float);
        pipe_->InitBuffer(qcqrScaleBuf_, qcqrScaleSize);
        qcqrScaleLocal_ = qcqrScaleBuf_.Get<float>();
        usedBytes += qcqrScaleSize;
    }
    if constexpr (IsFullQuantMode<mmInputType, dequantScaleType, true>()) {
        uint64_t dequantScaleWDqSize = baseParams_->headSizeCq * sizeof(float);
        pipe_->InitBuffer(dequantScaleWDqBuffer_, dequantScaleWDqSize); // [1, 1536]
        dequantScaleWDqLocal_ = dequantScaleWDqBuffer_.Get<float>();
        usedBytes += dequantScaleWDqSize;

        uint64_t dequantScaleWDkvKrSize = (baseParams_->headSizeCkv + baseParams_->dimHeadRope) * sizeof(float);
        pipe_->InitBuffer(dequantScaleWDkvKrBuffer_, dequantScaleWDkvKrSize); // [1, 512 + 64]
        dequantScaleWDkvKrLocal_ = dequantScaleWDkvKrBuffer_.Get<float>();
        usedBytes += dequantScaleWDkvKrSize;
    }
    uint64_t rmsnormGammaCqSize = baseParams_->headSizeCq * sizeof(rmsNormGammaType);
    pipe_->InitBuffer(rmsnormGammaCqBuffer_, rmsnormGammaCqSize); // [1, 1536] bf16
    rmsnormGammaCqLocal_ = rmsnormGammaCqBuffer_.Get<rmsNormGammaType>();
    usedBytes += rmsnormGammaCqSize;

    uint64_t rmsnormGammaCkvSize = baseParams_->headSizeCkv * sizeof(rmsNormGammaType);
    pipe_->InitBuffer(rmsnormGammaCkvBuffer_, rmsnormGammaCkvSize); // [1, 512] bf16
    rmsnormGammaCkvLocal_ = rmsnormGammaCkvBuffer_.Get<rmsNormGammaType>();
    usedBytes += rmsnormGammaCkvSize;

    if constexpr (IsFullQuantMode<mmQcQrInputType, dequantScaleType, true>()) {
        if (enableSmoothScalesCq_) {
            uint64_t smoothScaleCqSize = baseParams_->headSizeCq * sizeof(float);
            pipe_->InitBuffer(smoothScaleCqBuffer_, smoothScaleCqSize); // [1, 1536]
            smoothScaleCqLocal_ = smoothScaleCqBuffer_.Get<float>();
            usedBytes += smoothScaleCqSize;
        }
    }

    if constexpr (std::is_same<krCacheType, int8_t>::value) {
        uint64_t quantScaleCkrSize = baseParams_->dimHeadRope * sizeof(float);
        pipe_->InitBuffer(quantScaleCkrBuffer_, baseParams_->dimHeadRope * sizeof(float)); // [1, 64]
        quantScaleCkrLocal_ = quantScaleCkrBuffer_.Get<float>();
        usedBytes += quantScaleCkrSize;
    }

    if constexpr (IsFullQuantMode<rmsNormCkvOutputType, dequantScaleType, false>()) {
        uint64_t quantScaleCkvSize = 0;
        if constexpr (std::is_same<mmCkvKrOutputType, int32_t>::value ||
                      std::is_same<mmCkvKrOutputType, float>::value) {
            quantScaleCkvSize = ALIGN_BLOCK_SIZE;
            pipe_->InitBuffer(quantScaleCkvBuffer_, quantScaleCkvSize);
        } else { // per_channel量化场景走else分支
            quantScaleCkvSize = baseParams_->headSizeCkv * sizeof(float);
            pipe_->InitBuffer(quantScaleCkvBuffer_, quantScaleCkvSize); // [1, 512]
        }
        quantScaleCkvLocal_ = quantScaleCkvBuffer_.Get<float>();
        usedBytes += quantScaleCkvSize;
    }

    // 预留brcb的空间
    uint64_t deQuantScaleCqSize = (baseParams_->stepBatchSize + 7) * ALIGN_BLOCK_SIZE;
    pipe_->InitBuffer(dequantTool_.deQuantScaleCqBuffer_, deQuantScaleCqSize);
    dequantTool_.deQuantScaleCqLocal_ = dequantTool_.deQuantScaleCqBuffer_.template Get<dequantScaleType>();
    usedBytes += deQuantScaleCqSize;

    uint64_t sincosSize = 0;
    if constexpr (MLAPT::enableDequantOpt || nqWaveMode) {
        sincosSize = 2 * baseParams_->dimHeadRope * sizeof(ropeComputType) *
                     ((baseParams_->stepBatchSize + cvRatio_ - 1) / cvRatio_);
        pipe_->InitBuffer(sincosBuffer_, sincosSize);
    } else {
        sincosSize = 2 * baseParams_->dimHeadRope * sizeof(ropeComputType) * curVecTokenMax_;
        pipe_->InitBuffer(sincosBuffer_, sincosSize); // [2, 64] float
    }
    usedBytes += sincosSize;

    if constexpr (MLAPT::enableDequantOpt || nqWaveMode) {
        cosLocal_ = sincosBuffer_.Get<ropeComputType>();
        sinLocal_ = cosLocal_[baseParams_->dimHeadRope * ((baseParams_->stepBatchSize + cvRatio_ - 1) / cvRatio_)];
    } else {
        cosLocal_ = sincosBuffer_.Get<ropeComputType>();
        sinLocal_ = cosLocal_[baseParams_->dimHeadRope * curVecTokenMax_];
    }

    if constexpr (MLAPT::actualSeqMode == ACTUAL_SEQ_MODE::EN_Q_LEN) {
        uint64_t stepActualSeqSize = baseParams_->stepBatchSize * sizeof(int64_t);
        pipe_->InitBuffer(stepActualSeqBuffer_, stepActualSeqSize); // [1, stepBatchSize]
        stepActualSeqLocal_ = stepActualSeqBuffer_.Get<int64_t>();
        usedBytes += stepActualSeqSize;
    }
    pipe_->InitBuffer(shareBuffer_, MAX_UB_SIZE - usedBytes);
    ubShareBytes_ = static_cast<uint32_t>(MAX_UB_SIZE - usedBytes);
    CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_MTE3>(FINISH_VEC_CKVKR);
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::CubeBufferInit()
{
    // cube相关Buffer初始化
    pipe_->InitBuffer(aBufL1_, L1_A_SIZE * 2);
    pipe_->InitBuffer(bBufL1_, L1_B_SIZE * 2);

    if (MLA_USE_UB_DIRECT && (MLAPT::enableDequantOpt || nqWaveMode) && !MLAPT::enableGroupComputeOpt &&
        (!isFp8E8m0 || IsMxWaveCapable())) {
        uint64_t cvUbSize =
            2ULL * static_cast<uint64_t>(baseParams_->stepBatchSize) * mmQcQrParam_.baseN * sizeof(mmQcQrOutputType);
        pipe_->InitBuffer(cvSharedUb_, cvUbSize);
    }

    SetFlag<HardEvent::MTE1_MTE2>(A_EVENT0);
    SetFlag<HardEvent::MTE1_MTE2>(A_EVENT1);
    SetFlag<HardEvent::MTE1_MTE2>(B_EVENT0);
    SetFlag<HardEvent::MTE1_MTE2>(B_EVENT1);
    aL1Tensor_ = aBufL1_.Get<mmInputType>();
    bL1Tensor_ = bBufL1_.Get<mmInputType>();
    bufParam_.aL1BufAddr = aBufL1_.GetBufferAddr(aL1Tensor_.GetBufferHandle());
    bufParam_.bL1BufAddr = bBufL1_.GetBufferAddr(bL1Tensor_.GetBufferHandle());

    pipe_->InitBuffer(aBufL0_, L0A_PP_SIZE * 2); // 64K
    pipe_->InitBuffer(bBufL0_, L0B_PP_SIZE * 2); // 64K
    pipe_->InitBuffer(cBufL0_, L0C_PP_SIZE * 2); // 128K

    SetFlag<HardEvent::M_MTE1>(L0A_EVENT0);
    SetFlag<HardEvent::M_MTE1>(L0A_EVENT1);
    SetFlag<HardEvent::M_MTE1>(L0B_EVENT0);
    SetFlag<HardEvent::M_MTE1>(L0B_EVENT1);

    SetFlag<HardEvent::FIX_M>(L0C_EVENT0);
    SetFlag<HardEvent::FIX_M>(L0C_EVENT1);

    SetFlag<HardEvent::MTE1_MTE2>(SCALE_EVENT);

    bufParam_.aL0BufAddr = aBufL0_.GetBufferAddr(aBufL0_.Get<mmInputType>().GetBufferHandle());
    bufParam_.bL0BufAddr = bBufL0_.GetBufferAddr(bBufL0_.Get<mmInputType>().GetBufferHandle());
    bufParam_.cL0BufAddr = cBufL0_.GetBufferAddr(cBufL0_.Get<float>().GetBufferHandle());
}

/*
 * workspace管理
 * 1. 常驻：dequantTool_.deQuantScaleCqGm_ stepBs * 32 Byte
 * 2. 中间结果：
 *      tokenXGm_──────>mmCkvKrResGm_
 *          |           [stepBS, HCkv + Dr]
 *          |           (bf16 | int32)
 *          └─────────>mmCqResGm_──────>rmsNormCqResGm_──────>mmQcQrResGm_──────>mmQcQrResDequantGm_──────>mmQnResGm_
 *                     [stepBS, HCq]    [stepBS, HCq]     [stepBS, N1, D + Dr]   [stepBS, N1, D]           [stepBS, N1,
 * HCkv] (bf16 | int32)   (bf16 | int8)     (bf16 | int32)         (bf16)                    (bf16)
 */
template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::WorkspaceInit(__gm__ uint8_t* workspace)
{
    int64_t workspaceOffset = 0;
    if constexpr (std::is_same<rmsNormCqOutputType, FP8E4M3>::value && isFp8E8m0) {
        dequantScaleCqSize_ = baseParams_->headSizeCq / FP8_E4M3_BLOCK_SIZE;
    }
    dequantScaleCqSize_ = Align(dequantScaleCqSize_, BYTE_BLOCK);
    if constexpr (MLAPT::enableGroupComputeOpt || MLAPT::enableDequantOpt) {
        dequantTool_.deQuantScaleCqGm_.SetGlobalBuffer((__gm__ dequantScaleType*)(workspace + workspaceOffset));
        workspaceOffset += baseParams_->stepBatchSize * dequantScaleCqSize_;
    }
    const bool mm2SplitKnR1 = baseParams_->mm2KGroup > 1U;
    const int64_t mm2PartialBytes =
        mm2SplitKnR1 ? static_cast<int64_t>(sizeof(float)) : static_cast<int64_t>(sizeof(mmCkvKrOutputType));
    mmCkvKrResGm_.SetGlobalBuffer((__gm__ mmCkvKrOutputType*)(workspace + workspaceOffset));
    if (mm2SplitKnR1 && std::is_same<mmCkvKrOutputType, bfloat16_t>::value) {
        mmCkvKrResFloatGm_.SetGlobalBuffer((__gm__ float*)(workspace + workspaceOffset));
    }
    workspaceOffset += static_cast<int64_t>(baseParams_->stepBatchSize) *
                       static_cast<int64_t>(baseParams_->headSizeCkv + baseParams_->dimHeadRope) * mm2PartialBytes *
                       baseParams_->mm2KGroup;

    const bool mm1SplitKnR1 = baseParams_->mm1KGroup > 1U;
    const int64_t mm1PartialBytes =
        mm1SplitKnR1 ? static_cast<int64_t>(sizeof(float)) : static_cast<int64_t>(sizeof(mmCqOutputType));
    mmCqResGm_.SetGlobalBuffer((__gm__ mmCqOutputType*)(workspace + workspaceOffset));
    if (mm1SplitKnR1 && std::is_same<mmCqOutputType, bfloat16_t>::value) {
        mmCqResFloatGm_.SetGlobalBuffer((__gm__ float*)(workspace + workspaceOffset));
    }

    if (baseParams_->queryNormFlag == 0U) {
        if constexpr (!std::is_same<rmsNormCqOutputType, mmCqOutputType>::value) {
            workspaceOffset += static_cast<int64_t>(baseParams_->stepBatchSize) *
                               static_cast<int64_t>(baseParams_->headSizeCq) * mm1PartialBytes * baseParams_->mm1KGroup;
        } else if (mm1SplitKnR1) {
            workspaceOffset += static_cast<int64_t>(baseParams_->stepBatchSize) *
                               static_cast<int64_t>(baseParams_->headSizeCq) * mm1PartialBytes * baseParams_->mm1KGroup;
        }
        // 不返回queryNorm时，rmsNormCq结果放在workspace
        rmsNormCqResGm_.SetGlobalBuffer((__gm__ rmsNormCqOutputType*)(workspace + workspaceOffset));
        workspaceOffset += static_cast<int64_t>(baseParams_->stepBatchSize) *
                           static_cast<int64_t>(baseParams_->headSizeCq) * sizeof(rmsNormCqOutputType);
    } else {
        // 返回queryNorm时，rmsNormCq结果直接输出gm，此处需要偏移`mmCqResGm_`占用的大小
        workspaceOffset += static_cast<int64_t>(baseParams_->stepBatchSize) *
                           static_cast<int64_t>(baseParams_->headSizeCq) * mm1PartialBytes * baseParams_->mm1KGroup;
    }

    mmQcQrResGm_.SetGlobalBuffer((__gm__ mmQcQrOutputType*)(workspace + workspaceOffset));
    if constexpr (IsFullQuantMode<mmQcQrInputType, dequantScaleType, false>() || nqWaveMode) {
        workspaceOffset += static_cast<int64_t>(baseParams_->stepBatchSize) *
                           static_cast<int64_t>(baseParams_->headSizeQc + baseParams_->headSizeQr) *
                           static_cast<int64_t>(sizeof(mmQcQrOutputType));
    }

    mmQcQrResDequantGm_.SetGlobalBuffer((__gm__ mmQnInputType*)(workspace + workspaceOffset));
    if constexpr (((std::is_same<mmInputType, int8_t>::value && std::is_same<kvCacheType, int8_t>::value) ||
                   (std::is_same<mmInputType, FP8E4M3>::value && std::is_same<kvCacheType, FP8E4M3>::value) ||
                   (std::is_same<mmInputType, HIF8>::value && std::is_same<kvCacheType, HIF8>::value)) &&
                  !isPertile) {
        workspaceOffset += static_cast<int64_t>(baseParams_->stepBatchSize) *
                           static_cast<int64_t>(baseParams_->numHeadSize) *
                           static_cast<int64_t>(baseParams_->dimHeadSizeQc) * sizeof(mmQnInputType);
        mmQnResGm_.SetGlobalBuffer((__gm__ mmQnOutputType*)(workspace + workspaceOffset));
    }
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::UpdateStepBatchParams(int64_t curStepBatchSize)
{
    mmCqParam_.m = curStepBatchSize;
    mmCkvKrParam_.m = curStepBatchSize;
    mmQcQrParam_.m = curStepBatchSize;
    mmQnParam_.m = curStepBatchSize;
    curVectorBlockNum_ = curStepBatchSize;
}

/*
 * MlaProlog算子计算&CV流水同步流程
 *                    ┌───────────────── token_x ─────────────────┐
 *                    |                                           ▼
 *                    |                                      MatmulCkvKr
 *                    ▼                                           | wait mm CkvKr(0x1)
 *                MatmulCq                                        ▼
 *                    | wait mm Cq(0x1)                  ┌─────────────────┐
 *                    ▼                                  ▼                 ▼
 *               RmsNorm(Cq)                         RmsNorm(Ckv)       Rope(Kr)
 *                    | wait rmsNorm cq(0x1)             |                 |
 *                    ▼                                  ▼                 ▼
 *        ┌───────MatmulQcQr───────┐                 Scatter(Ckv)      Scatter(Kr)
 *        | wait mm Qc(0x1)        |                     |                 |
 *        ▼                        |                     ▼                 ▼
 *    DequantQc                    |                 kv_cache_out      kr_cache_out
 *        | wait dequant qc(0x1)   | wait mm Qr(0x2)
 *        ▼                        ▼
 *     MatmulQn                 Rope(Qr)
 *        | wait mm Qn(0x1)        |
 *   DynamicQuantQn──┐             |
 *        |          ▼             |
 *        ▼   dequant_scale_out    ▼
 *    query_out               query_rope_out
 * 注：仅为表明基本计算与CV同步流程，仅包含了影响CV同步的量化分支，其余量化分支应参考设计文档。
 */
template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::Process()
{
    constexpr bool needQnDynamicQuant =
        ((std::is_same<mmInputType, int8_t>::value && std::is_same<kvCacheType, int8_t>::value) ||
         (std::is_same<mmInputType, FP8E4M3>::value && std::is_same<kvCacheType, FP8E4M3>::value) ||
         (std::is_same<mmInputType, HIF8>::value && std::is_same<kvCacheType, HIF8>::value)) &&
        !isPertile;
    int64_t numHeadOffset = cubeBlockIdx_ * baseParams_->mm4SingleCoreBatch;
    int64_t mmQnLoops;
    if (cubeBlockIdx_ == baseParams_->mm4BlockNum - 1) {
        mmQnLoops = static_cast<int64_t>(baseParams_->numHeadSize) - numHeadOffset;
    } else {
        mmQnLoops = static_cast<int64_t>(baseParams_->mm4SingleCoreBatch);
    }

    // AIC的offset参数
    AicOffset aicOffset;
    ComputeAicOffset(aicOffset, numHeadOffset);

    // AIV的offset参数
    AivOffset aivOffset;
    ComputeAivOffset(aivOffset, 0);

    int64_t bsSize = static_cast<int64_t>(baseParams_->tokenSize);
    // 需要考虑BS合轴的尾块情况
    for (int64_t batchOffset = 0; batchOffset < bsSize;
         batchOffset += static_cast<int64_t>(baseParams_->stepBatchSize)) {
        if constexpr (MLAPT::actualSeqMode == ACTUAL_SEQ_MODE::EN_Q_LEN) {
            if (!actSeqState_.inited) {
                actSeqState_.curBatch = 0;
                actSeqState_.prevPrefix = 0;
                actSeqState_.curPrefix = (baseParams_->batchSize > 0) ? actualSeqLenGm_(0) : 0;
                actSeqState_.inited = true;
                actSeqState_.prevIndexOffset = 0;
                actSeqState_.curIndexOffset =
                    CeilDivT(actSeqState_.curPrefix, static_cast<int64_t>(baseParams_->blockSize));
            }
        }
        int64_t curStepBatchSize = bsSize - batchOffset;
        if (curStepBatchSize < static_cast<int64_t>(baseParams_->stepBatchSize)) {
            UpdateStepBatchParams(curStepBatchSize);
            if (batchOffset != 0) {
                ComputeAivOffset(aivOffset, batchOffset);
            }
        } else {
            curStepBatchSize = static_cast<int64_t>(baseParams_->stepBatchSize);
        }
        if ASCEND_IS_AIC {
            AicProcess<needQnDynamicQuant>(aicOffset, batchOffset, mmQnLoops);
        }
        if ASCEND_IS_AIV {
            AivProcess<needQnDynamicQuant>(aivOffset, batchOffset, curStepBatchSize, numHeadOffset, mmQnLoops);
        }
    }
    if ASCEND_IS_AIC {
        WaitFlag<HardEvent::MTE1_MTE2>(A_EVENT1);
        WaitFlag<HardEvent::MTE1_MTE2>(A_EVENT0);
        WaitFlag<HardEvent::MTE1_MTE2>(B_EVENT0);
        WaitFlag<HardEvent::MTE1_MTE2>(B_EVENT1);

        WaitFlag<HardEvent::M_MTE1>(L0A_EVENT0);
        WaitFlag<HardEvent::M_MTE1>(L0A_EVENT1);
        WaitFlag<HardEvent::M_MTE1>(L0B_EVENT1);
        WaitFlag<HardEvent::M_MTE1>(L0B_EVENT0);

        WaitFlag<HardEvent::FIX_M>(L0C_EVENT0);
        WaitFlag<HardEvent::FIX_M>(L0C_EVENT1);

        WaitFlag<HardEvent::MTE1_MTE2>(SCALE_EVENT);
        CrossCoreWaitFlag<SYNC_MODE_CUBE_VEC, PIPE_FIX>(FINISH_VEC_CKVKR);
        if ((MLAPT::enableDequantOpt || nqWaveMode) && !MLAPT::enableGroupComputeOpt &&
            (!isFp8E8m0 || IsMxWaveCapable())) {
            if (cvRatio_ == 1) {
                uint32_t drainCnt = (mm3WaveSeq_ < 2u) ? mm3WaveSeq_ : 2u;
                for (uint32_t dIter = 0; dIter < drainCnt; dIter++) {
                    CrossCoreWaitFlag<SYNC_MODE_CUBE_VEC, PIPE_FIX>(FINISH_VEC_QCQR_CONSUMED);
                }
            }
        }
    }
    if ASCEND_IS_AIV {
        if constexpr (MLA_CKVKR_BATCH2) {
            if (ckvkrBatchDrainPending_) {
                WaitFlag<HardEvent::MTE3_V>(CKVKR_STEP_DRAIN);
                ckvkrBatchDrainPending_ = false;
            }
        }
        if constexpr (nqWaveMode) {
            if (nqWaveTokenPending_) {
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
                nqWaveTokenPending_ = false;
            }
        }
    }
}

template <typename MLAPT>
template <bool needQnDynamicQuant>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::AicProcess(AicOffset& aicOffset, int64_t batchOffset,
                                                              int64_t mmQnLoops)
{
    int64_t tokenXOffset = batchOffset * static_cast<int64_t>(baseParams_->headSizeX);
    int64_t dequantScaleXOffset = batchOffset * static_cast<int64_t>(baseParams_->headSizeX) / 32;
    GlobalTensor<dequantScaleType> scaleAGm{};
    GlobalTensor<dequantScaleType> scaleBGmDq{};
    GlobalTensor<dequantScaleType> scaleBGmDkvKr{};
    if constexpr (std::is_same<mmInputType, FP8E4M3>::value && isFp8E8m0) {
        scaleAGm = dequantScaleXGm_[dequantScaleXOffset];
        scaleBGmDq = dequantScaleWDqGm_[aicOffset.dequantScaleWDqOffset];
        scaleBGmDkvKr = dequantScaleWDkvkrGm_[aicOffset.dequantScaleWDkvKrOffset];
    }
    // MatmulCq ──> RmsNorm(Cq)
    if constexpr (std::is_same<mmCqOutputType, bfloat16_t>::value) {
        if (baseParams_->mm1KGroup > 1U) {
            MatmulSplitN<mmInputType, float, dequantScaleType>(
                mmCqResFloatGm_[aicOffset.cqResOffset], tokenXGm_[tokenXOffset], weightDqGm_[aicOffset.weightDqOffset],
                mmCqParam_, UsedBlockParams{0, baseParams_->mm1BlockNum}, scaleAGm, scaleBGmDq, mm1B0PreNext_);
            mm1B0PreNext_ = false;
        } else {
            MatmulSplitN<mmInputType, mmCqOutputType, dequantScaleType>(
                mmCqResGm_[aicOffset.cqResOffset], tokenXGm_[tokenXOffset], weightDqGm_[aicOffset.weightDqOffset],
                mmCqParam_, UsedBlockParams{0, baseParams_->mm1BlockNum}, scaleAGm, scaleBGmDq);
        }
    } else {
        MatmulSplitN<mmInputType, mmCqOutputType, dequantScaleType>(
            mmCqResGm_[aicOffset.cqResOffset], tokenXGm_[tokenXOffset], weightDqGm_[aicOffset.weightDqOffset],
            mmCqParam_, UsedBlockParams{0, baseParams_->mm1BlockNum}, scaleAGm, scaleBGmDq);
    }
    CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_FIX>(FINISH_MM_CQ);
    // MatmulCkvKr ──> RmsNorm(Ckv)
    //            └──> Rope(Kr)
    const bool mm2B0Pre = MatmulMm2B0Preload(aicOffset);
    CrossCoreWaitFlag<SYNC_MODE_CUBE_VEC, PIPE_MTE2>(FINISH_VEC_CKVKR);
    if constexpr (std::is_same<mmCkvKrOutputType, bfloat16_t>::value) {
        if (baseParams_->mm2KGroup > 1U) {
            MatmulSplitN<mmInputType, float, dequantScaleType, true>(
                mmCkvKrResFloatGm_[aicOffset.ckvKrResOffset], tokenXGm_[tokenXOffset],
                weightDkvKrGm_[aicOffset.weightDkvKrOffset], mmCkvKrParam_,
                UsedBlockParams{0, baseParams_->mm2BlockNum}, scaleAGm, scaleBGmDkvKr, mm2B0Pre);
        } else {
            MatmulSplitN<mmInputType, mmCkvKrOutputType, dequantScaleType, true>(
                mmCkvKrResGm_[aicOffset.ckvKrResOffset], tokenXGm_[tokenXOffset],
                weightDkvKrGm_[aicOffset.weightDkvKrOffset], mmCkvKrParam_,
                UsedBlockParams{0, baseParams_->mm2BlockNum}, scaleAGm, scaleBGmDkvKr, mm2B0Pre);
        }
    } else {
        MatmulSplitN<mmInputType, mmCkvKrOutputType, dequantScaleType, true>(
            mmCkvKrResGm_[aicOffset.ckvKrResOffset], tokenXGm_[tokenXOffset],
            weightDkvKrGm_[aicOffset.weightDkvKrOffset], mmCkvKrParam_, UsedBlockParams{0, baseParams_->mm2BlockNum},
            scaleAGm, scaleBGmDkvKr, mm2B0Pre);
    }
    CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_FIX>(FINISH_MM_CKVKR);
    const bool mm3B0Pre = MatmulQcQrPreloadB0(aicOffset);
    CrossCoreWaitFlag<SYNC_MODE_CUBE_VEC, PIPE_MTE2>(FINISH_VEC_RMSNORM_CQ);

    if constexpr (std::is_same<mmInputType, FP8E4M3>::value && isFp8E8m0) {
        MatmulQcQr(aicOffset, mm3B0Pre);
    } else {
        MatmulAndSyncQcQr(aicOffset, mm3B0Pre);
    }
    if constexpr (std::is_same<mmQcQrInputType, FP8E4M3>::value && isFp8E8m0) {
        WaitFlag<HardEvent::MTE1_MTE2>(SCALE_EVENT); // FP8场景下Scale不做db，需要等scale用完才能做mmQn
    }
    PreloadQnAndSync(aicOffset, mmQnLoops);
    MatmulQnSyncDynamicQuantAndMulQr<needQnDynamicQuant>(aicOffset.qcOffset, aicOffset.weightUkOffset,
                                                         aicOffset.qnResOffset, mmQnLoops);
    if constexpr (std::is_same<mmQcQrInputType, FP8E4M3>::value && isFp8E8m0) {
        SetFlag<HardEvent::MTE1_MTE2>(SCALE_EVENT); // FP8场景下Scale不做db，需要mmQn用完才能做下一轮
    }
    if constexpr (!needQnDynamicQuant) {
        // MatmulQn的结果直接输出到 queryOut, qnOffset需要按Batch轴偏移
        aicOffset.qnResOffset += static_cast<int64_t>(baseParams_->stepBatchSize) *
                                 static_cast<int64_t>(baseParams_->headSizeCkv) *
                                 static_cast<int64_t>(baseParams_->numHeadSize);
    }
    if (unlikely(baseParams_->queryNormFlag == 1U)) {
        aicOffset.rmsNormCqResOffset +=
            static_cast<int64_t>(baseParams_->stepBatchSize) * static_cast<int64_t>(baseParams_->headSizeCq);
    }
    if (batchOffset + static_cast<int64_t>(baseParams_->stepBatchSize) < static_cast<int64_t>(baseParams_->tokenSize)) {
        mm1B0PreNext_ = MatmulMm1B0Preload(aicOffset);
    } else {
        mm1B0PreNext_ = false;
    }
}

template <typename MLAPT>
template <bool needQnDynamicQuant>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::AivProcess(AivOffset& aivOffset, int64_t batchOffset,
                                                              int64_t curStepBatchSize, int64_t numHeadOffset,
                                                              int64_t mmQnLoops)
{
    int64_t tokenIndex = batchOffset + aivOffset.curBlockTokenOffset;
    int64_t rmsNormCqResOffset = baseParams_->queryNormFlag == 1U ?
                                     tokenIndex * baseParams_->headSizeCq :
                                     static_cast<int64_t>(baseParams_->headSizeCq) * aivOffset.curBlockTokenOffset;

    if (batchOffset == 0) {
        // 只需要搬运一次
        CopyGlobalParams();
        if constexpr (MLA_PRELOAD_QCQR_SCALE && MLA_USE_UB_DIRECT && MLAPT::enableDequantOpt && !isFp8E8m0 &&
                      !MLAPT::enableGroupComputeOpt && IsFullQuantMode<mmQcQrInputType, dequantScaleType, true>()) {
            if (cvRatio_ == 1 && cubeBlockIdx_ < baseParams_->mm3BlockNum) {
                DataCopy(qcqrScaleLocal_, deqScaleQcQrW_[aivOffset.mmQnPreDequantOffset], mmQcQrParam_.n);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
            }
        }
    }
    CopyInSinCos(tokenIndex, aivOffset.curVecToken, batchOffset, curStepBatchSize);
    CrossCoreWaitFlag<SYNC_MODE_CUBE_VEC, PIPE_MTE2>(FINISH_MM_CQ);
    WaitAllCore<SYNC_MODE_ALL_VEC, PIPE_MTE2>(FINISH_VEC_ALL);
    RmsNormCq(tokenIndex, aivOffset.rmsNormCqOffset, rmsNormCqResOffset, aivOffset.curVecToken,
              aivOffset.curBlockTokenOffset);
    WaitAllCore<SYNC_MODE_ALL_VEC, PIPE_MTE3>(FINISH_VEC_ALL);
    if constexpr (MLA_FLAG06_EARLY && MLAPT::enableDequantOpt && !MLAPT::enableGroupComputeOpt && !isFp8E8m0 &&
                  IsFullQuantMode<rmsNormCqOutputType, dequantScaleType, true>()) {
        CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_MTE3>(FINISH_VEC_RMSNORM_CQ);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        if (aivOffset.curVecToken > 0) {
            DataCopy(dequantTool_.deQuantScaleCqGm_[aivOffset.curBlockTokenOffset * FP32_BLOCK_ELEMENT_NUM],
                     dequantTool_.deQuantScaleCqLocal_[aivOffset.curBlockTokenOffset * FP32_BLOCK_ELEMENT_NUM],
                     FP32_BLOCK_ELEMENT_NUM * aivOffset.curVecToken);
        }
        WaitAllCore<SYNC_MODE_ALL_VEC, PIPE_MTE3>(FINISH_VEC_ALL);
        DataCopy(dequantTool_.deQuantScaleCqLocal_, dequantTool_.deQuantScaleCqGm_,
                 ALIGN_BLOCK_SIZE / sizeof(float) * baseParams_->stepBatchSize);
    } else {
        // 聚合全部scale结果
        if constexpr ((MLAPT::enableDequantOpt || MLAPT::enableGroupComputeOpt) &&
                      IsFullQuantMode<rmsNormCqOutputType, dequantScaleType, true>()) {
            DataCopy(dequantTool_.deQuantScaleCqLocal_, dequantTool_.deQuantScaleCqGm_,
                     ALIGN_BLOCK_SIZE / sizeof(float) * baseParams_->stepBatchSize);
        }
        CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_MTE3>(FINISH_VEC_RMSNORM_CQ);
    }
    CrossCoreWaitFlag<SYNC_MODE_CUBE_VEC, PIPE_MTE2>(FINISH_MM_CKVKR);
    WaitAllCore<SYNC_MODE_ALL_VEC, PIPE_MTE2>(FINISH_VEC_ALL);
    RmsNormRopeScatterCkvKr(tokenIndex, aivOffset.rmsNormCkvOffset, aivOffset.ropeKrOffset, aivOffset.curVecToken,
                            batchOffset);
    CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_MTE3>(FINISH_VEC_CKVKR);

    DequantQcAndRopeQc<needQnDynamicQuant>(aivOffset, batchOffset, curStepBatchSize, numHeadOffset, mmQnLoops);
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::ComputeAicOffset(AicOffset& aicOffset, int64_t numHeadOffset)
{
    uint32_t nGroup = 0; // MM1/MM2 分组复用(切KN 分支内重新赋值; 原切N 分支不使用)
    uint32_t kGroup = 0;

    if (baseParams_->mm1KGroup > 1) { // MM1切KN: 部分和[kg][step][Hcq]
        nGroup = blockIdx_ % baseParams_->mm1NGroup;
        kGroup = blockIdx_ / baseParams_->mm1NGroup;
        int64_t M_N = static_cast<int64_t>(baseParams_->stepBatchSize) *
                      static_cast<int64_t>(baseParams_->headSizeCq); // 一个 K 组块大小(整幅)
        aicOffset.cqResOffset = kGroup * M_N + baseParams_->mm1SingleCoreN * nGroup;
        aicOffset.weightDqOffset =
            static_cast<int64_t>(baseParams_->headSizeX) * (nGroup * baseParams_->mm1SingleCoreN);
        if constexpr (std::is_same<mmInputType, FP8E4M3>::value && isFp8E8m0) {
            aicOffset.dequantScaleWDqOffset =
                static_cast<int64_t>(baseParams_->headSizeX) / 32 * (nGroup * baseParams_->mm1SingleCoreN);
        }
    } else {                                                             // MM1 原切N回退: 原版偏移公式
        aicOffset.cqResOffset = baseParams_->mm1SingleCoreN * blockIdx_; // 64 * idx
        aicOffset.weightDqOffset = static_cast<int64_t>(baseParams_->headSizeX) * aicOffset.cqResOffset;
        aicOffset.dequantScaleWDqOffset = static_cast<int64_t>(baseParams_->headSizeX) / 32 * aicOffset.cqResOffset;
    }

    if (baseParams_->mm2KGroup > 1) { // MM2切KN: 部分和 [kg][step][576]
        nGroup = blockIdx_ % baseParams_->mm2NGroup;
        kGroup = blockIdx_ / baseParams_->mm2NGroup;
        int64_t mN2 = static_cast<int64_t>(baseParams_->stepBatchSize) *
                      (static_cast<int64_t>(baseParams_->headSizeCkv) +
                       static_cast<int64_t>(baseParams_->headSizeKr)); // 一个 K 组块大小(整幅)
        aicOffset.ckvKrResOffset = kGroup * mN2 + baseParams_->mm2SingleCoreN * nGroup;
        aicOffset.weightDkvKrOffset =
            static_cast<int64_t>(baseParams_->headSizeX) * (baseParams_->mm2SingleCoreN * nGroup);
        aicOffset.dequantScaleWDkvKrOffset =
            static_cast<int64_t>(baseParams_->headSizeX) / 32 * (baseParams_->mm2SingleCoreN * nGroup);
    } else {                                                                // MM2 原切N路径
        aicOffset.ckvKrResOffset = baseParams_->mm2SingleCoreN * blockIdx_; //  (512 + 64) / 9 * idx  = 64 * idx
        aicOffset.weightDkvKrOffset = static_cast<int64_t>(baseParams_->headSizeX) * aicOffset.ckvKrResOffset;
        aicOffset.dequantScaleWDkvKrOffset =
            static_cast<int64_t>(baseParams_->headSizeX) / 32 * aicOffset.ckvKrResOffset;
    }

    if constexpr (MLAPT::enableGroupComputeOpt) {
        aicOffset.weightUqOffset = static_cast<int64_t>(baseParams_->headSizeCq) *
                                   (baseParams_->dimHeadSizeQc + baseParams_->dimHeadRope) *
                                   blockIdx_; // 1536 * (32 * 128 + 32 * 64) / 24 * idx = 1536 * 256 * idx
        aicOffset.weightQrOffset = static_cast<int64_t>(baseParams_->headSizeCq) *
                                       (baseParams_->dimHeadSizeQc + baseParams_->dimHeadRope) * 2 *
                                       (blockIdx_ - QC_CORE_NUM) +
                                   static_cast<int64_t>(baseParams_->headSizeCq) * baseParams_->dimHeadSizeQc;
        aicOffset.qCResOffset = baseParams_->dimHeadSizeQc * blockIdx_;
        aicOffset.qRResOffset =
            baseParams_->stepBatchSize * baseParams_->dimHeadSizeQc * QC_CORE_NUM +
            baseParams_->dimHeadRope * 2 * (blockIdx_ - QC_CORE_NUM); // m=baseParams_->stepBatchSize
    } else {
        aicOffset.qcQrResOffset = baseParams_->mm3SingleCoreN * blockIdx_; // 192 * idx
        aicOffset.weightUqQrOffset =
            static_cast<int64_t>(baseParams_->headSizeCq) *
            aicOffset.qcQrResOffset; // 1536 * (32 * 128 + 32 * 64) / 24 * idx = 1536 * 256 * idx
        aicOffset.dequantScaleWuqqrOffset = aicOffset.weightUqQrOffset / 32;
    }

    if (cubeBlockIdx_ < baseParams_->mm4BlockNum) {
        if constexpr (std::is_same<mmQcQrOutputType, int32_t>::value || std::is_same<mmQcQrOutputType, float>::value) {
            aicOffset.qcOffset = static_cast<int64_t>(baseParams_->dimHeadSizeQc) * numHeadOffset; // 128 * idx
        } else {
            aicOffset.qcOffset = (nqWaveMode && (cvRatio_ == 1)) ?
                                     (static_cast<int64_t>(baseParams_->dimHeadSizeQc) * numHeadOffset) :
                                     (static_cast<int64_t>(baseParams_->dimHeadSizeQc + baseParams_->dimHeadRope) *
                                      numHeadOffset); // (128 + 64) * idx
        }
        aicOffset.weightUkOffset = static_cast<int64_t>(baseParams_->dimHeadSizeQc) *
                                   static_cast<int64_t>(baseParams_->headSizeCkv) * numHeadOffset; // (128 * 512) * idx
        aicOffset.qnResOffset = static_cast<int64_t>(baseParams_->headSizeCkv) * numHeadOffset;    // BS, N, Dq
    }
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::ComputeAivOffset(AivOffset& aivOffset, int64_t batchOffset)
{
    curStepVecBackToken_ = curVectorBlockNum_ / static_cast<uint32_t>(vectorCoreNum_);
    curStepVecFrontListNum_ = curVectorBlockNum_ % static_cast<uint32_t>(vectorCoreNum_);
    curStepVecFrontToken_ = curStepVecFrontListNum_ == 0 ? curStepVecBackToken_ : curStepVecBackToken_ + 1;

    aivOffset.curVecToken = blockIdx_ < curStepVecFrontListNum_ ? curStepVecFrontToken_ : curStepVecBackToken_;
    aivOffset.curBlockTokenOffset = blockIdx_ < curStepVecFrontListNum_ ?
                                        blockIdx_ * aivOffset.curVecToken :
                                        blockIdx_ * aivOffset.curVecToken + curStepVecFrontListNum_;
    aivOffset.rmsNormCqOffset =
        static_cast<int64_t>(baseParams_->headSizeCq) * aivOffset.curBlockTokenOffset; // 1536 * idx
    aivOffset.rmsNormCkvOffset = static_cast<int64_t>(baseParams_->headSizeCkv + baseParams_->dimHeadRope) *
                                 aivOffset.curBlockTokenOffset;                     // (512 + 64) * idx
    aivOffset.ropeKrOffset = baseParams_->headSizeCkv + aivOffset.rmsNormCkvOffset; // 512 + (512 + 64) * idx
    if constexpr (!MLAPT::enableDequantOpt && !nqWaveMode) {
        aivOffset.mmQnPreDequantOffset =
            (baseParams_->headSizeQc + baseParams_->headSizeQr) * aivOffset.curBlockTokenOffset;
        aivOffset.mmQnPreDequantResOffset = baseParams_->headSizeQc * aivOffset.curBlockTokenOffset;
        aivOffset.ropeQrOffset = static_cast<int64_t>(baseParams_->dimHeadSizeQc) +
                                 static_cast<int64_t>(baseParams_->numHeadSize) *
                                     static_cast<int64_t>(baseParams_->dimHeadSizeQc + baseParams_->dimHeadRope) *
                                     aivOffset.curBlockTokenOffset; // 128 + 32 * (128 + 64) * idx
        aivOffset.ropeQrResOffset =
            static_cast<int64_t>(baseParams_->headSizeQr) *
            (aivOffset.curBlockTokenOffset + batchOffset); // 32 * 64 * idx;    // 按BS合轴切分step
    }

    // 以下的Offset均是与batchOffset无关的，仅初始化一次即可。
    if (batchOffset == 0) {
        if constexpr (MLAPT::enableDequantOpt || nqWaveMode) {
            aivOffset.mmQnPreDequantOffset = baseParams_->mm3SingleCoreN * cubeBlockIdx_; // qcQrResOffset
            aivOffset.mmQnPreDequantResOffset = baseParams_->mm3SingleCoreN /
                                                (baseParams_->dimHeadSizeQc + baseParams_->dimHeadRope) *
                                                baseParams_->dimHeadSizeQc * cubeBlockIdx_;
            aivOffset.ropeQrOffset = aivOffset.mmQnPreDequantOffset + baseParams_->dimHeadSizeQc;
            aivOffset.ropeQrResOffset =
                (baseParams_->mm3SingleCoreN / (baseParams_->dimHeadSizeQc + baseParams_->dimHeadRope)) *
                baseParams_->dimHeadRope * cubeBlockIdx_;
        }

        if constexpr (MLAPT::enableGroupComputeOpt) {
            aivOffset.qcScaleOffsetSplitN =
                (baseParams_->headSizeQc + baseParams_->headSizeQr) / QC_CORE_NUM * (blockIdx_ / cvRatio_);
            aivOffset.mmQnPreDequantResOffset = (baseParams_->headSizeQc) / QC_CORE_NUM * (blockIdx_ / cvRatio_);
            aivOffset.mmQnPreDequantOffset = aivOffset.mmQnPreDequantResOffset;
            aivOffset.ropeQrResSplitNOffset = (blockIdx_ - QC_CORE_NUM * cvRatio_) * baseParams_->dimHeadRope;
            aivOffset.ropeQrSplitNOffset = baseParams_->headSizeQc + aivOffset.ropeQrResSplitNOffset;
        }
    }
}

// Mlaprolog 支持int8进int32出以及mxfp8进fp32出, 参考MatmulQcQr
template <typename MLAPT>
template <typename T, typename O, typename S, bool needCheckEmptyTensor, bool needCheckAFullLoad, bool isContinuousCopy>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::MatmulSplitN(
    const GlobalTensor<O>& tensorResGm, const GlobalTensor<T>& tensorAGm, const GlobalTensor<T>& tensorBGm,
    const MMParams& mmPara, const UsedBlockParams& mmBlockParams, const GlobalTensor<S>& tensorAScaleGm,
    const GlobalTensor<S>& tensorBScaleGm, const bool b0BPre)
{
    if constexpr (needCheckEmptyTensor && MLAPT::emptyMode == EMPTY_TENSOR_MODE::EMPTY_CACHE) {
        return;
    }
    if (blockIdx_ < mmBlockParams.blockStartIdx || blockIdx_ >= mmBlockParams.blockEndIdx) {
        return;
    }
    // 用于enableGroupComputeOpt场景
    if constexpr (needCheckAFullLoad) {
        constexpr uint32_t fullLoadM =
            (sizeof(mmQcQrInputType) == sizeof(int8_t)) ? INT8_AFULLLOAD_MAX_MSIZE : BF16_AFULLLOAD_MAX_MSIZE;
        bool isAFullLoad = (mmQcQrParam_.m <= fullLoadM) ? true : false;
        if (isAFullLoad) {
            MatmulGroupComputeAFullLoad<T, O, S, isContinuousCopy>(tensorResGm, tensorAGm, tensorBGm, mmPara,
                                                                   bufParam_);
            return;
        }
    }
    uint32_t totalN = mmPara.n;
    uint32_t splitSizeN = mmPara.baseN;
    uint32_t nLoops = CeilDivT(totalN, splitSizeN);
    uint32_t tailN = splitSizeN;

    constexpr bool isE8M0Load = std::is_same<T, FP8E4M3>::value && std::is_same<S, FP8E8M0>::value;
    const bool aReuse = (CeilDivT(mmPara.k, mmPara.kL1StepSize) == 1) &&
                        (static_cast<uint64_t>(mmPara.m) * mmPara.k * sizeof(T) <= L1_A_SIZE) && !isE8M0Load;
    if (aReuse) {
        // 注意第 3/4 实参传全量: k = mmPara.k(单轮即全K), kInput = orgKa(GM 原行宽)
        LoadL1A(tensorAGm[mmPara.kStart], mmPara.m, mmPara.k, mmPara.orgKa, bufParam_);
        WaitFlag<HardEvent::MTE2_MTE1>(A_EVENT0 + (bufParam_.aL1BufIter & 1u)); // 确认 A 已搬完
    }
    const bool notE8M0 = !(std::is_same<T, FP8E4M3>::value && std::is_same<S, FP8E8M0>::value);
    constexpr bool kEnableSubBlockPrefetch = true; // A/B 单点: 关掉即回到"子块边界裸奔"
    for (int64_t nIdx = 0; nIdx < nLoops; nIdx++) {
        if (nIdx == nLoops - 1) {
            tailN = totalN - (nLoops - 1) * splitSizeN;
        }
        const bool hasNextN1 = kEnableSubBlockPrefetch && (nIdx + 1 < nLoops) && notE8M0;
        uint32_t nL1OffsetNext = 0;
        uint32_t nL1SizeNext = 0;
        if (nIdx + 1 < nLoops) {
            nL1OffsetNext = static_cast<uint32_t>((nIdx + 1) * splitSizeN);
            nL1SizeNext =
                static_cast<uint32_t>((nIdx + 2 == nLoops) ? (totalN - (nLoops - 1) * splitSizeN) : splitSizeN);
        }
        const bool b0Pre = kEnableSubBlockPrefetch && (nIdx > 0) && notE8M0; // 子块0 冷启动无人预取
        const bool b0BPreCur = b0BPre && (nIdx == 0);
        if (aReuse) {
            MatmulSplitK<T, O, S, true>(tensorResGm, tensorAGm, tensorBGm, mmPara, bufParam_, nIdx * splitSizeN, tailN,
                                        tensorAScaleGm, tensorBScaleGm, false, LocalTensor<O>(), hasNextN1,
                                        nL1OffsetNext, nL1SizeNext, b0Pre, b0BPreCur);
        } else {
            MatmulSplitK<T, O, S>(tensorResGm, tensorAGm, tensorBGm, mmPara, bufParam_, nIdx * splitSizeN, tailN,
                                  tensorAScaleGm, tensorBScaleGm, false, LocalTensor<O>(), hasNextN1, nL1OffsetNext,
                                  nL1SizeNext, b0Pre, b0BPreCur);
        }
    }
    if (aReuse) {
        SetFlag<HardEvent::MTE1_MTE2>(A_EVENT0 + (bufParam_.aL1BufIter & 1u));
        bufParam_.aL1BufIter++;
    }
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::MatmulAndSyncQcQr(AicOffset& aicOffset, bool b0BPre)
{
    if constexpr (MLAPT::enableGroupComputeOpt) {
        // MatmulQc
        // 复用mmCqResGm_ workspace
        MatmulSplitN<rmsNormCqOutputType, mmQcQrOutputType, dequantScaleType, false, true>(
            mmQcQrResGm_[aicOffset.qCResOffset], rmsNormCqResGm_[aicOffset.rmsNormCqResOffset],
            weightUqQrGm_[aicOffset.weightUqOffset], mmQcQrParam_, UsedBlockParams{0, QC_CORE_NUM});
        CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_FIX>(FINISH_MM_QC);
        // MatmulQr
        MatmulSplitN<rmsNormCqOutputType, mmQcQrOutputType, dequantScaleType, false, true, false>(
            mmQcQrResGm_[aicOffset.qRResOffset], rmsNormCqResGm_[aicOffset.rmsNormCqResOffset],
            weightUqQrGm_[aicOffset.weightQrOffset], mmQcQrParam_,
            UsedBlockParams{QC_CORE_NUM, QC_CORE_NUM + QR_CORE_NUM});
    } else {
        MatmulQcQr(aicOffset, b0BPre);
    }
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::MatmulQcQr(AicOffset& aicOffset, bool b0BPre)
{
    if (blockIdx_ >= baseParams_->mm3BlockNum) {
        return;
    }
    // RmsNorm(Cq) ──> MatmulQcQr ──> MatmulQn
    //                           └──> Rope(Qr)
    constexpr uint32_t mSize =
        (sizeof(mmQcQrInputType) == sizeof(int8_t)) ? INT8_AFULLLOAD_MAX_MSIZE : BF16_AFULLLOAD_MAX_MSIZE;
    bool isAFullLoad = (mmQcQrParam_.m <= mSize) ? true : false;

    uint32_t nInput = mmQcQrParam_.n;
    uint32_t nL1SplitSize = mmQcQrParam_.baseN;
    uint32_t nL1loops = CeilDivT(nInput, nL1SplitSize);
    uint32_t subNL1SplitSize = nL1SplitSize;
    if (isAFullLoad) {
        MatmulQcQrLoadA(aicOffset);
    }
    MatmulQcQrSplitN(aicOffset, isAFullLoad, nInput, nL1SplitSize, nL1loops, b0BPre);
    if (isAFullLoad) {
        SetFlag<HardEvent::MTE1_MTE2>(A_EVENT0 + (bufParam_.aL1BufIter & 1u));
        bufParam_.aL1BufIter++;
    }
}

template <typename MLAPT>
__aicore__ inline bool MlaPrologVecS1CubS2<MLAPT>::MatmulQcQrPreloadB0(AicOffset& aicOffset)
{
    constexpr bool notE8M0B = !(std::is_same<mmQcQrInputType, FP8E4M3>::value && isFp8E8m0);
    if constexpr (notE8M0B) {
        if constexpr (!MLAPT::enableGroupComputeOpt) {
            if (blockIdx_ >= baseParams_->mm3BlockNum) {
                return false;
            }
            const uint32_t nL1loops = CeilDivT(mmQcQrParam_.n, mmQcQrParam_.baseN);
            const uint32_t nSize0 = (nL1loops <= 1U) ? mmQcQrParam_.n : mmQcQrParam_.baseN;
            const uint32_t kLoadSize =
                (mmQcQrParam_.k < mmQcQrParam_.kL1StepSize) ? mmQcQrParam_.k : mmQcQrParam_.kL1StepSize;
            LoadL1B<mmQcQrInputType, DataFormat::NZ>(
                weightUqQrGm_[aicOffset.weightUqQrOffset +
                              mmQcQrParam_.kStart * GetC0Num<mmQcQrInputType>()], // 与 LoadL1AB 的 kStart 项同式
                nSize0, mmQcQrParam_.n, kLoadSize, mmQcQrParam_.orgKb, bufParam_);
            return true;
        }
    }
    return false;
}

template <typename MLAPT>
__aicore__ inline bool MlaPrologVecS1CubS2<MLAPT>::MatmulMm1B0Preload(AicOffset& aicOffset)
{
    if constexpr (MLA_NQ_MM1B0_XSTEP && !MLAPT::enableGroupComputeOpt && std::is_same<mmInputType, bfloat16_t>::value &&
                  std::is_same<mmQcQrInputType, bfloat16_t>::value) {
        if (baseParams_->mm1KGroup <= 1U) {
            return false;
        }
        if (blockIdx_ >= baseParams_->mm1BlockNum) {
            return false;
        }
        const uint32_t nL1loops = CeilDivT(mmCqParam_.n, mmCqParam_.baseN);
        const uint32_t nSize0 = (nL1loops <= 1U) ? mmCqParam_.n : mmCqParam_.baseN;
        const uint32_t kLoadSize = (mmCqParam_.k < mmCqParam_.kL1StepSize) ? mmCqParam_.k : mmCqParam_.kL1StepSize;
        LoadL1B<mmInputType, DataFormat::NZ>(
            weightDqGm_[aicOffset.weightDqOffset +
                        mmCqParam_.kStart * GetC0Num<mmInputType>()], // 与 LoadL1AB 的 kStart 项同式
            nSize0, mmCqParam_.n, kLoadSize, mmCqParam_.orgKb, bufParam_);
        return true;
    }
    return false;
}

template <typename MLAPT>
__aicore__ inline bool MlaPrologVecS1CubS2<MLAPT>::MatmulMm2B0Preload(AicOffset& aicOffset)
{
    if constexpr (MLA_NQ_MM2B0_PRE && !MLAPT::enableGroupComputeOpt && std::is_same<mmInputType, bfloat16_t>::value &&
                  std::is_same<mmQcQrInputType, bfloat16_t>::value &&
                  MLAPT::emptyMode != EMPTY_TENSOR_MODE::EMPTY_CACHE) {
        if (blockIdx_ >= baseParams_->mm2BlockNum) {
            return false;
        }
        const uint32_t nL1loops = CeilDivT(mmCkvKrParam_.n, mmCkvKrParam_.baseN);
        const uint32_t nSize0 = (nL1loops <= 1U) ? mmCkvKrParam_.n : mmCkvKrParam_.baseN;
        const uint32_t kLoadSize =
            (mmCkvKrParam_.k < mmCkvKrParam_.kL1StepSize) ? mmCkvKrParam_.k : mmCkvKrParam_.kL1StepSize;
        LoadL1B<mmInputType, DataFormat::NZ>(
            weightDkvKrGm_[aicOffset.weightDkvKrOffset +
                           mmCkvKrParam_.kStart * GetC0Num<mmInputType>()], // 与 LoadL1AB 的 kStart 项同式
            nSize0, mmCkvKrParam_.n, kLoadSize, mmCkvKrParam_.orgKb, bufParam_);
        return true;
    }
    return false;
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::MatmulQcQrLoadA(AicOffset& aicOffset)
{
    if constexpr (std::is_same<mmQcQrInputType, FP8E4M3>::value && isFp8E8m0) {
        uint32_t offsetL1B = L1_B_SIZE / 2 / sizeof(rmsNormCqOutputType); // 2表示scale起始地址固定从L1B上ping的64k开始
        WaitFlag<HardEvent::MTE1_MTE2>(SCALE_EVENT);
        LoadL1AAndScale<rmsNormCqOutputType, dequantScaleType, false, true>(
            rmsNormCqResGm_[aicOffset.rmsNormCqResOffset],
            dequantTool_.deQuantScaleCqGm_[aicOffset.dequantScaleCqOffset], mmQcQrParam_.m, mmQcQrParam_.k,
            mmQcQrParam_.k, mmQcQrParam_.kScale, offsetL1B, bufParam_);
        SetFlag<HardEvent::MTE1_MTE2>(SCALE_EVENT);
    } else {
        LoadL1A(rmsNormCqResGm_[aicOffset.rmsNormCqResOffset], mmQcQrParam_.m, mmQcQrParam_.k, mmQcQrParam_.k,
                bufParam_);
    }
    WaitFlag<HardEvent::MTE2_MTE1>(A_EVENT0 + (bufParam_.aL1BufIter & 1u));
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::MatmulQcQrSplitN(AicOffset& aicOffset, bool isAFullLoad,
                                                                    uint32_t nInput, uint32_t nL1SplitSize,
                                                                    uint32_t nL1loops, bool b0BPre, bool isAResident)
{
    const bool aLoaded = isAFullLoad || isAResident;
    uint32_t tailSplitSize = nL1SplitSize;
    bool ubPath = false;
    LocalTensor<mmQcQrOutputType> waveUb;
    constexpr bool notE8M0Q = !(std::is_same<mmQcQrInputType, FP8E4M3>::value && isFp8E8m0);
    const bool ubDirectSupported =
        (MLAPT::enableDequantOpt || nqWaveMode) && !MLAPT::enableGroupComputeOpt && (notE8M0Q || IsMxWaveCapable());
    const bool ubActive = ubDirectSupported && MLA_USE_UB_DIRECT;
    ubPath = ubActive && (cvRatio_ == 1);
    constexpr bool kEnableWavePrefetch = true;
    for (int64_t nIdx = 0; nIdx < nL1loops; nIdx++) {
        if (nIdx == nL1loops - 1) {
            tailSplitSize = nInput - (nL1loops - 1) * nL1SplitSize;
        }
        const bool hasNextN1 = kEnableWavePrefetch && (nIdx + 1 < nL1loops) && notE8M0Q;
        uint32_t nL1OffsetNext = 0;
        uint32_t nL1SizeNext = 0;
        if (nIdx + 1 < nL1loops) {
            nL1OffsetNext = static_cast<uint32_t>((nIdx + 1) * nL1SplitSize);
            nL1SizeNext =
                static_cast<uint32_t>((nIdx + 2 == nL1loops) ? (nInput - (nL1loops - 1) * nL1SplitSize) : nL1SplitSize);
        }
        const bool b0Pre = kEnableWavePrefetch && (nIdx > 0) && notE8M0Q;
        const bool b0BPreCur = b0BPre && (nIdx == 0);
        if (ubPath) {
            if (mm3WaveSeq_ >= 2) {
                CrossCoreWaitFlag<SYNC_MODE_CUBE_VEC, PIPE_FIX>(FINISH_VEC_QCQR_CONSUMED);
            }
            uint32_t waveElems = baseParams_->stepBatchSize * mmQcQrParam_.baseN;
            waveUb = cvSharedUb_.Get<mmQcQrOutputType>()[(mm3WaveSeq_ & 1u) * waveElems];
            mm3WaveSeq_++;
        }
        if constexpr (std::is_same<mmQcQrInputType, FP8E4M3>::value && isFp8E8m0) {
            if (isAFullLoad) {
                MatmulSplitK<rmsNormCqOutputType, mmQcQrOutputType, dequantScaleType, true, true>(
                    mmQcQrResGm_[aicOffset.qcQrResOffset], rmsNormCqResGm_[aicOffset.rmsNormCqResOffset],
                    weightUqQrGm_[aicOffset.weightUqQrOffset], mmQcQrParam_, bufParam_, nIdx * nL1SplitSize,
                    tailSplitSize, dequantTool_.deQuantScaleCqGm_[aicOffset.dequantScaleCqOffset],
                    deqScaleQcQrW_[aicOffset.dequantScaleWuqqrOffset], ubPath, waveUb);
            } else {
                MatmulSplitK<rmsNormCqOutputType, mmQcQrOutputType, dequantScaleType, false, true>(
                    mmQcQrResGm_[aicOffset.qcQrResOffset], rmsNormCqResGm_[aicOffset.rmsNormCqResOffset],
                    weightUqQrGm_[aicOffset.weightUqQrOffset], mmQcQrParam_, bufParam_, nIdx * nL1SplitSize,
                    tailSplitSize, dequantTool_.deQuantScaleCqGm_[aicOffset.dequantScaleCqOffset],
                    deqScaleQcQrW_[aicOffset.dequantScaleWuqqrOffset], ubPath, waveUb);
            }
        } else {
            if (aLoaded) {
                MatmulSplitK<rmsNormCqOutputType, mmQcQrOutputType, dequantScaleType, true>(
                    mmQcQrResGm_[aicOffset.qcQrResOffset], rmsNormCqResGm_[aicOffset.rmsNormCqResOffset],
                    weightUqQrGm_[aicOffset.weightUqQrOffset], mmQcQrParam_, bufParam_, nIdx * nL1SplitSize,
                    tailSplitSize, {}, {}, ubPath, waveUb, hasNextN1, nL1OffsetNext, nL1SizeNext, b0Pre, b0BPreCur);
            } else {
                MatmulSplitK<rmsNormCqOutputType, mmQcQrOutputType, dequantScaleType>(
                    mmQcQrResGm_[aicOffset.qcQrResOffset], rmsNormCqResGm_[aicOffset.rmsNormCqResOffset],
                    weightUqQrGm_[aicOffset.weightUqQrOffset], mmQcQrParam_, bufParam_, nIdx * nL1SplitSize,
                    tailSplitSize, {}, {}, ubPath, waveUb, hasNextN1, nL1OffsetNext, nL1SizeNext, b0Pre, b0BPreCur);
            }
        }
        if constexpr (MLAPT::enableDequantOpt || nqWaveMode) {
            CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_FIX>(FINISH_MM_QCQR_SPLIT_N);
        }
    }
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::PreloadQnAndSync(AicOffset& aicOffset, int64_t mmQnLoops)
{
    MatmulQnWeightPreload(aicOffset.weightUkOffset);
    if constexpr (MLAPT::enableGroupComputeOpt) {
        CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_FIX>(FINISH_MM_QR);
    } else if constexpr (MLAPT::enableDequantOpt || nqWaveMode) {
        return;
    } else {
        CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_FIX>(FINISH_MM_QCQR);
    }

    if constexpr (IsFullQuantMode<mmQcQrInputType, dequantScaleType, false>()) {
        CrossCoreWaitFlag<SYNC_MODE_CUBE_VEC, PIPE_MTE2>(FINISH_VEC_DEQUANT_QC);
    } else {
        WaitAllCore<SYNC_MODE_ALL_CUBE, PIPE_FIX>(FINISH_MM_ALL);
    }
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::MatmulQnWeightPreload(int64_t weightUkOffset)
{
    uint32_t maxBlockIdx = MLAPT::enableGroupComputeOpt ? QC_CORE_NUM : baseParams_->mm4BlockNum;
    if (blockIdx_ >= maxBlockIdx) {
        return;
    }
    if (mmQnParam_.k > mmQnParam_.baseK) {
        return;
    }
    int64_t weightOffset = weightUkOffset;
    LoadL1B<mmQnInputType, DataFormat::ND, false>(weightUkGm_[weightOffset], mmQnParam_.n, mmQnParam_.n, mmQnParam_.k,
                                                  mmQnParam_.k, bufParam_);
    WaitFlag<HardEvent::MTE2_MTE1>(B_EVENT0 + (bufParam_.bL1BufIter & 1u));
    weightOffset += static_cast<int64_t>(baseParams_->dimHeadSizeQc) * static_cast<int64_t>(baseParams_->headSizeCkv);
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::MatmulQnHeadUb(const GlobalTensor<mmQnInputType>& tensorAGm,
                                                                  const GlobalTensor<mmQnInputType>& tensorBGm,
                                                                  const bool bPreloaded)
{
    using O = mmQnOutputType; // bf16
    using O_L0C = float;      // mmQnInputType=bf16 → L0C float
    const MMParams& para = mmQnParam_;
    const uint32_t rows0 = (para.m + 1) / 2;
    const uint32_t rows1 = para.m - rows0;
    const uint32_t halfCnt = (rows1 > 0) ? 2 : 1;
    constexpr uint32_t aL1Offset = L1_A_SIZE / sizeof(mmQnInputType);
    constexpr uint32_t bL1Offset = L1_B_SIZE / sizeof(mmQnInputType);
    mmLocalTensors<mmQnInputType, O_L0C> localTensors;
    localTensors.Init(bufParam_);
    {
        LoadL1A(tensorAGm, rows0, para.k, para.orgKa, bufParam_);
        if (halfCnt == 2) {
            LoadL1A<mmQnInputType, true>(tensorAGm[static_cast<int64_t>(rows0) * static_cast<int64_t>(para.orgKa)],
                                         rows1, para.k, para.orgKa, bufParam_);
        }
    }
    if (!bPreloaded) {
        LoadL1B<mmQnInputType, DataFormat::ND, false>(tensorBGm, para.n, para.n, para.k, para.k, bufParam_);
        WaitFlag<HardEvent::MTE2_MTE1>(B_EVENT0 + (bufParam_.bL1BufIter & 1u));
    }
    LocalTensor<mmQnInputType> bL1 = localTensors.bL1Tensor[(bufParam_.bL1BufIter & 1u) * bL1Offset];
    const uint32_t cvPingElems = baseParams_->stepBatchSize * mmQcQrParam_.baseN * sizeof(mmQcQrOutputType) / sizeof(O);
    const uint32_t nloops = CeilDivT(para.n, para.baseN);
    for (uint32_t h = 0; h < halfCnt; h++) {
        const uint32_t rows = (h == 0) ? rows0 : rows1;
        LocalTensor<mmQnInputType> aL1h;
        {
            WaitFlag<HardEvent::MTE2_MTE1>(A_EVENT0 + ((bufParam_.aL1BufIter + h) & 1u));
            aL1h = localTensors.aL1Tensor[((bufParam_.aL1BufIter + h) & 1u) * aL1Offset];
        }
        if (mm3WaveSeq_ >= 2) {
            CrossCoreWaitFlag<SYNC_MODE_CUBE_VEC, PIPE_FIX>(FINISH_VEC_QCQR_CONSUMED);
        }
        LocalTensor<O> ubPing = cvSharedUb_.Get<O>()[(mm3WaveSeq_ & 1u) * cvPingElems];
        for (uint32_t n = 0; n < nloops; n++) {
            const uint32_t nAct = (n == nloops - 1) ? (para.n - (nloops - 1) * para.baseN) : para.baseN;
            MmadParams mmadParams(Align(rows, BLOCK_CUBE_SIZE), nAct, para.k, UNIT_FLAG_SET, false, true);
            WaitFlag<HardEvent::FIX_M>(L0C_EVENT0 + (bufParam_.cL0BufIter & 1u));
            LocalTensor<O_L0C> cL0 =
                localTensors.cL0Tensor[(bufParam_.cL0BufIter & 1u) * (L0C_PP_SIZE / sizeof(O_L0C))];
            MatmulL0<mmQnInputType, O_L0C, dequantScaleType>(bufParam_, aL1h, bL1[para.k * para.baseN * n],
                                                             localTensors.aL0Tensor, localTensors.bL0Tensor, cL0,
                                                             mmadParams, para.kL1StepSize);
            GetTensorCToUb<O, O_L0C, true>(ubPing[n * para.baseN], cL0, rows, nAct, Align(rows, BLOCK_CUBE_SIZE),
                                           para.n, bufParam_, QuantMode_t::F322BF16);
            SetFlag<HardEvent::FIX_M>(L0C_EVENT0 + (bufParam_.cL0BufIter & 1u));
            bufParam_.cL0BufIter++;
        }
        CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_FIX>(FINISH_MM_QN_SPLIT_N);
        mm3WaveSeq_++;
    }
    SetFlag<HardEvent::MTE1_MTE2>(A_EVENT0 + (bufParam_.aL1BufIter & 1u));
    if (halfCnt == 2) {
        SetFlag<HardEvent::MTE1_MTE2>(A_EVENT0 + ((bufParam_.aL1BufIter + 1) & 1u));
        bufParam_.aL1BufIter += 2;
    } else {
        bufParam_.aL1BufIter += 1;
    }
    SetFlag<HardEvent::MTE1_MTE2>(B_EVENT0 + (bufParam_.bL1BufIter & 1u));
    bufParam_.bL1BufIter++;
}

template <typename MLAPT>
template <bool needQnDynamicQuant>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::MatmulQnSyncDynamicQuantAndMulQr(int64_t qcOffset,
                                                                                    int64_t weightUkOffset,
                                                                                    int64_t qnResOffset,
                                                                                    int64_t subLoopTimes)
{
    uint32_t maxBlockIdx = MLAPT::enableGroupComputeOpt ? QC_CORE_NUM : baseParams_->mm4BlockNum;
    if (blockIdx_ >= maxBlockIdx) {
        return;
    }
    // MatmulQcQr ──> MatmulQn ──> query_out
    const bool mm4UbSupported = MLAPT::enableDequantOpt && !MLAPT::enableGroupComputeOpt &&
                                (!isFp8E8m0 || (MLA_MM4_UB_MXFP8_ENABLE && IsMxWaveCapable())) &&
                                (!std::is_same<mmInputType, int8_t>::value || MLA_MM4_UB_INT8_ENABLE);
    constexpr bool useMm4UbDirect = MLA_USE_MM4_UB;
    const bool mm4UbPathQn =
        mm4UbSupported && useMm4UbDirect && (cvRatio_ == 1) && (mmQnParam_.k <= mmQnParam_.baseK) && needQnDynamicQuant;
    bool needSparseSync = subLoopTimes > MAX_SYNC_FLAG_COUNT;
    for (int64_t i = 0; i < subLoopTimes; i++) {
        if constexpr (MLAPT::enableDequantOpt || nqWaveMode) {
            if (!needSparseSync || i % 2 == 0 || i == subLoopTimes - 1) {
                CrossCoreWaitFlag<SYNC_MODE_CUBE_VEC, PIPE_MTE2>(FINISH_VEC_DEQUANT_QC_SPLIT_N);
            }
        }
        if (unlikely(mmQnParam_.baseK < mmQnParam_.k)) {
            uint32_t nInput = baseParams_->headSizeCkv;
            uint32_t nL1SplitSize = mmQnParam_.baseN;
            uint32_t nL1loops = CeilDivT(nInput, nL1SplitSize);
            uint32_t subNL1SplitSize = nL1SplitSize;
            for (int64_t nL1 = 0; nL1 < nL1loops; nL1++) {
                if (nL1 == nL1loops - 1) {
                    subNL1SplitSize = nInput - (nL1loops - 1) * nL1SplitSize;
                }
                MatmulSplitK<mmQnInputType, mmQnOutputType, dequantScaleType, false, false, false, DataFormat::ND>(
                    mmQnResGm_[qnResOffset], mmQcQrResDequantGm_[qcOffset], weightUkGm_[weightUkOffset], mmQnParam_,
                    bufParam_, nL1 * nL1SplitSize, subNL1SplitSize);
            }
        } else {
            if (mm4UbPathQn) {
                MatmulQnHeadUb(mmQcQrResDequantGm_[qcOffset], weightUkGm_[weightUkOffset],
                               (i < 1) || MLA_MM4_NEXTB_PRELOAD);
                if constexpr (MLA_MM4_NEXTB_PRELOAD) {
                    if (i + 1 < subLoopTimes) {
                        LoadL1B<mmQnInputType, DataFormat::ND, false>(
                            weightUkGm_[weightUkOffset + static_cast<int64_t>(baseParams_->dimHeadSizeQc) *
                                                             static_cast<int64_t>(baseParams_->headSizeCkv)],
                            mmQnParam_.n, mmQnParam_.n, mmQnParam_.k, mmQnParam_.k, bufParam_);
                        WaitFlag<HardEvent::MTE2_MTE1>(B_EVENT0 + (bufParam_.bL1BufIter & 1u));
                    }
                }
            } else if (i < 1) {
                MatmulFullLoad<mmQnInputType, mmQnOutputType, dequantScaleType, true, true>(
                    mmQnResGm_[qnResOffset], mmQcQrResDequantGm_[qcOffset], weightUkGm_[weightUkOffset], mmQnParam_,
                    bufParam_);
            } else {
                MatmulFullLoad<mmQnInputType, mmQnOutputType, dequantScaleType, false, true>(
                    mmQnResGm_[qnResOffset], mmQcQrResDequantGm_[qcOffset], weightUkGm_[weightUkOffset], mmQnParam_,
                    bufParam_);
            }
        }
        if constexpr (std::is_same<mmQcQrOutputType, int32_t>::value || std::is_same<mmQcQrInputType, FP8E4M3>::value ||
                      std::is_same<mmQcQrOutputType, float>::value) {
            qcOffset += static_cast<int64_t>(baseParams_->dimHeadSizeQc);
        } else {
            qcOffset += (nqWaveMode && (cvRatio_ == 1)) ?
                            static_cast<int64_t>(baseParams_->dimHeadSizeQc) :
                            static_cast<int64_t>(baseParams_->dimHeadSizeQc + baseParams_->dimHeadRope);
        }
        weightUkOffset +=
            static_cast<int64_t>(baseParams_->dimHeadSizeQc) * static_cast<int64_t>(baseParams_->headSizeCkv);
        qnResOffset += static_cast<int64_t>(baseParams_->headSizeCkv);

        if constexpr (needQnDynamicQuant) {
            if (!mm4UbPathQn) {
                CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_FIX>(FINISH_MM_QN_SPLIT_N);
            }
        }
    }
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::CopyInSinCos(int64_t tokenIndex, int64_t curVecToken,
                                                                int64_t batchOffset, int64_t curStepBatchSize)
{
    if constexpr (!MLAPT::enableRope) {
        return;
    }
    LocalTensor<uint8_t> shareTmpUb = shareBuffer_.Get<uint8_t>();
    if constexpr (MLAPT::enableDequantOpt || nqWaveMode) {
        // 如果是切N场景，mm3的每个C核都会做rope
        if (cubeBlockIdx_ >= baseParams_->mm3BlockNum) {
            return;
        }
        // 如果curStepBatchSize是偶数，则两个核平分；如果curStepBatchSize是奇数，则奇数核比偶数核多分一个
        // >> 1 是将curStepBatchSize分到每个vec核上；
        uint32_t subBlockIdx_ = blockIdx_ % cvRatio_;
        int64_t offset = (curStepBatchSize / cvRatio_) * subBlockIdx_ + batchOffset;
        GatherSinCos<ropeSinCosType, ropeComputType>(cosLocal_, sinLocal_, ropeCosGm_, ropeSinGm_, offset,
                                                     (curStepBatchSize + cvRatio_ - 1) / cvRatio_, shareTmpUb,
                                                     vectorRow_, baseParams_->dimHeadRope);
    } else {
        if (blockIdx_ >= curVectorBlockNum_) {
            return;
        }
        GatherSinCos<ropeSinCosType, ropeComputType>(cosLocal_, sinLocal_, ropeCosGm_, ropeSinGm_, tokenIndex,
                                                     curVecToken, shareTmpUb, vectorRow_, baseParams_->dimHeadRope);
    }
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::CopyGlobalParams()
{
    // dequantScaleWDq
    if constexpr (IsFullQuantMode<mmInputType, dequantScaleType, true>()) {
        DataCopyExtParams dequantCopyParams{1, static_cast<uint32_t>(baseParams_->headSizeCq * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> dequantPadParams{false, 0, 0, 0};
        DataCopyPad(dequantScaleWDqLocal_, dequantScaleWDqGm_, dequantCopyParams, dequantPadParams);
    }

    // rmsnormGammaCq
    DataCopy(rmsnormGammaCqLocal_, rmsnormGammaCqGm_, baseParams_->headSizeCq);

    // rmsnormGammaCkv
    DataCopy(rmsnormGammaCkvLocal_, rmsnormGammaCkvGm_, baseParams_->headSizeCkv);

    // smoothScaleCq

    if constexpr (IsFullQuantMode<mmQcQrInputType, dequantScaleType, true>()) {
        if (enableSmoothScalesCq_) {
            DataCopyExtParams smoothCopyParams{1, static_cast<uint32_t>(baseParams_->headSizeCq * sizeof(float)), 0, 0,
                                               0};
            DataCopyPadExtParams<float> smoothPadParams{false, 0, 0, 0};
            DataCopyPad(smoothScaleCqLocal_, smoothScaleCqGm_, smoothCopyParams, smoothPadParams);
        }
    }

    // dequantScaleWDkvKr
    if constexpr (IsFullQuantMode<mmInputType, dequantScaleType, true>()) {
        DataCopyExtParams dequantCopyParams{
            1, static_cast<uint32_t>((baseParams_->headSizeCkv + baseParams_->dimHeadRope) * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> dequantPadParams{false, 0, 0, 0};
        DataCopyPad(dequantScaleWDkvKrLocal_, dequantScaleWDkvkrGm_, dequantCopyParams, dequantPadParams);
    }

    // quantScaleCkv

    if constexpr (IsFullQuantMode<rmsNormCkvOutputType, dequantScaleType, false>() && !isPertile) {
        if constexpr (std::is_same<mmCkvKrOutputType, int32_t>::value ||
                      std::is_same<mmCkvKrOutputType, float>::value) {
            DataCopyExtParams ckvCopyParams{1, sizeof(float), 0, 0, 0};
            DataCopyPadExtParams<float> ckvPadParams{false, 0, 0, 0};
            DataCopyPad(quantScaleCkvLocal_, quantScaleCkvGm_, ckvCopyParams, ckvPadParams);
        } else { // per_channel量化场景走else分支
            DataCopy(quantScaleCkvLocal_, quantScaleCkvGm_, baseParams_->headSizeCkv);
        }
    }

    // quantScaleCkr
    if constexpr (std::is_same<krCacheType, int8_t>::value) {
        DataCopyExtParams ckrCopyParams{1, static_cast<uint32_t>(baseParams_->dimHeadRope * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> ckrPadParams{false, 0, 0, 0};
        DataCopyPad(quantScaleCkrLocal_, quantScaleCkrGm_, ckrCopyParams, ckrPadParams);
    }
}

/**
 * @brief RmsNormCq流程，融合了dynamicquant
          内部所需空间约为 curVecToken(128) * 8*4 + 8*4 + (4*vectorRow_*baseParams_->headSizeCq + 8)*4 = 28.0625K
 */
template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::RmsNormCq(int64_t tokenIndex, int64_t rmsNormCqOffset,
                                                             int64_t rmsNormCqResOffset, int64_t curVecToken,
                                                             int64_t curBlockTokenOffset)
{
    if (curVecToken == 0) {
        return;
    }
    if (blockIdx_ >= curVectorBlockNum_) {
        return;
    }
    uint64_t dequantScaleXSize = 1;
    if constexpr (std::is_same<rmsNormCqOutputType, FP8E4M3>::value && isFp8E8m0) {
        dequantScaleXSize = baseParams_->headSizeX / FP8_E4M3_BLOCK_SIZE;
    }
    uint64_t dequantScaleCqElementNum = dequantScaleCqSize_ / sizeof(dequantScaleType);
    uint64_t dequantScaleXElementNum = Align(dequantScaleXSize, BYTE_BLOCK) / sizeof(float);
    int64_t stepTokenIndex = tokenIndex;
    LocalTensor<rmsNormCqOutputType> outputLocal = shareBuffer_.Get<rmsNormCqOutputType>();
    const bool isMxfp8ScaleX = std::is_same<rmsNormCqOutputType, FP8E4M3>::value && isFp8E8m0;
    constexpr bool gaSupported = std::is_same<mmCqOutputType, float>::value &&
                                 std::is_same<dequantScaleType, float>::value &&
                                 IsFullQuantMode<rmsNormCqOutputType, dequantScaleType, false>() &&
                                 !(std::is_same<rmsNormCqOutputType, FP8E4M3>::value && isFp8E8m0);
    const bool gaGate = gaSupported && (baseParams_->mm1KGroup > 1U) && (baseParams_->mm1KGroup <= 4U) &&
                        (curVecToken == 2) && (cvRatio_ == 1);
    constexpr bool gaSupportedInt8 = std::is_same<mmCqOutputType, int32_t>::value &&
                                     std::is_same<dequantScaleType, float>::value &&
                                     IsFullQuantMode<rmsNormCqOutputType, dequantScaleType, false>();
    const bool gaGateInt8 = gaSupportedInt8 && (baseParams_->mm1KGroup > 1U) && (baseParams_->mm1KGroup <= 4U) &&
                            (curVecToken == 2) && (cvRatio_ == 1);
    constexpr bool gaSupportedMx = isFp8E8m0 && std::is_same<mmCqOutputType, float>::value;
    const bool gaGateMx = gaSupportedMx && (baseParams_->mm1KGroup > 1U) && (baseParams_->mm1KGroup <= 4U) &&
                          (curVecToken == 2) && (cvRatio_ == 1);
    constexpr bool gaSupportedNq = std::is_same<mmInputType, bfloat16_t>::value &&
                                   std::is_same<mmQcQrInputType, bfloat16_t>::value &&
                                   std::is_same<dequantScaleType, float>::value &&
                                   !IsFullQuantMode<rmsNormCqOutputType, dequantScaleType, false>();
    const bool gaGateNq = gaSupportedNq && (baseParams_->mm1KGroup > 1U) && (baseParams_->mm1KGroup <= 4U) &&
                          (curVecToken == 2) && (cvRatio_ == 1);
    const uint64_t cqRowReserve = (gaGate || gaGateInt8 || gaGateMx || gaGateNq) ? 2ULL : 1ULL;
    const uint64_t scaleXSlotNum = ((gaGate || gaGateInt8) && !isMxfp8ScaleX) ? 2ULL : 1ULL;
    LocalTensor<dequantScaleType> dequantScaleQcQr =
        outputLocal[cqRowReserve * baseParams_->headSizeCq].template ReinterpretCast<dequantScaleType>();
    LocalTensor<float> dequantScaleXLocal =
        dequantScaleQcQr[curVecToken * dequantScaleCqElementNum].template ReinterpretCast<float>();
    LocalTensor<uint8_t> shareTmpUb =
        dequantScaleXLocal[scaleXSlotNum * dequantScaleXElementNum].template ReinterpretCast<uint8_t>();
    SplitKnAccumParams mm1SplitKnParams{0, 0};
    if (baseParams_->mm1KGroup > 1) {
        mm1SplitKnParams.kGroups = baseParams_->mm1KGroup;
        mm1SplitKnParams.partialStride =
            static_cast<uint64_t>(baseParams_->stepBatchSize) * baseParams_->headSizeCq; // 整幅(一个 K 组块)
    }
    int64_t batchedTokens = 0;
    if constexpr (gaSupported) {
        if (gaGate) {
            const uint32_t slotBytes = static_cast<uint32_t>(baseParams_->headSizeCq) * sizeof(float) *
                                       (1U + static_cast<uint32_t>(baseParams_->mm1KGroup));
            RmsNormParam gaParams = {
                baseParams_->reciprocalCq, baseParams_->epsilonCq, static_cast<uint32_t>(vectorRow_),
                baseParams_->headSizeCq,   baseParams_->qcQrScale, baseParams_->isQcQrScaleEnable};
            for (int64_t t = 0; t < 2; t++) {
                DataCopyPad(dequantScaleXLocal[t * dequantScaleXElementNum], dequantScaleXGm_[tokenIndex + t],
                            {1, sizeof(float), 0, 0}, {false, 0, 0, 0});
                RmsNormKnAccumIssue<mmCqOutputType>(
                    shareTmpUb[static_cast<uint32_t>(t) * slotBytes],
                    mmCqResGm_[rmsNormCqOffset + t * static_cast<int64_t>(baseParams_->headSizeCq)], 1U,
                    baseParams_->headSizeCq, mm1SplitKnParams, (t == 0) ? EVENT_ID0 : EVENT_ID1);
            }
            // ---- C 阶段: 等本槽读完成 → Dequant/RmsNorm/smooth/DQ → 写回 ----
            for (int64_t t = 0; t < 2; t++) {
                const int64_t outOff = t * static_cast<int64_t>(baseParams_->headSizeCq);
                RmsNormKnDqCompute<rmsNormGammaType, float, mmCqOutputType, rmsNormCqOutputType, dequantScaleType>(
                    outputLocal[outOff], dequantScaleQcQr[t * dequantScaleCqElementNum], rmsnormGammaCqLocal_,
                    smoothScaleCqLocal_, dequantScaleWDqLocal_, dequantScaleXLocal[t * dequantScaleXElementNum],
                    shareTmpUb[static_cast<uint32_t>(t) * slotBytes], gaParams, enableSmoothScalesCq_, mm1SplitKnParams,
                    (t == 0) ? EVENT_ID0 : EVENT_ID1);
                if constexpr (!MLA_RMSNORM_RES_ONE_WRITE) {
                    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                    DataCopy(rmsNormCqResGm_[rmsNormCqResOffset + outOff], outputLocal[outOff],
                             baseParams_->headSizeCq);
                }
            }
            if constexpr (MLA_RMSNORM_RES_ONE_WRITE) {
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID0); // 等两 token 的 C 阶段 V 全部排空
                DataCopy(rmsNormCqResGm_[rmsNormCqResOffset], outputLocal, 2 * baseParams_->headSizeCq);
            }
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            rmsNormCqOffset += 2 * static_cast<int64_t>(baseParams_->headSizeCq);
            rmsNormCqResOffset += 2 * static_cast<int64_t>(baseParams_->headSizeCq);
            tokenIndex += 2;
            batchedTokens = curVecToken; // 跳过下面的原 per-token 循环
        }
    }
    if constexpr (gaSupportedInt8) {
        if (gaGateInt8) {
            const uint32_t slotBytes = static_cast<uint32_t>(baseParams_->headSizeCq) * sizeof(int32_t) *
                                       (1U + static_cast<uint32_t>(baseParams_->mm1KGroup));
            RmsNormParam gaParams = {
                baseParams_->reciprocalCq, baseParams_->epsilonCq, static_cast<uint32_t>(vectorRow_),
                baseParams_->headSizeCq,   baseParams_->qcQrScale, baseParams_->isQcQrScaleEnable};
            for (int64_t t = 0; t < 2; t++) {
                DataCopyPad(dequantScaleXLocal[t * dequantScaleXElementNum], dequantScaleXGm_[tokenIndex + t],
                            {1, sizeof(float), 0, 0}, {false, 0, 0, 0});
                RmsNormKnAccumIssue<int32_t>(
                    shareTmpUb[static_cast<uint32_t>(t) * slotBytes],
                    mmCqResGm_[rmsNormCqOffset + t * static_cast<int64_t>(baseParams_->headSizeCq)], 1U,
                    baseParams_->headSizeCq, mm1SplitKnParams, (t == 0) ? EVENT_ID0 : EVENT_ID1);
            }
            for (int64_t t = 0; t < 2; t++) {
                const int64_t outOff = t * static_cast<int64_t>(baseParams_->headSizeCq);
                RmsNormKnDqComputeInt32<rmsNormGammaType, float, rmsNormCqOutputType, dequantScaleType>(
                    outputLocal[outOff], dequantScaleQcQr[t * dequantScaleCqElementNum], rmsnormGammaCqLocal_,
                    smoothScaleCqLocal_, dequantScaleWDqLocal_, dequantScaleXLocal[t * dequantScaleXElementNum],
                    shareTmpUb[static_cast<uint32_t>(t) * slotBytes], gaParams, enableSmoothScalesCq_, mm1SplitKnParams,
                    (t == 0) ? EVENT_ID0 : EVENT_ID1);
                if constexpr (!MLA_RMSNORM_RES_ONE_WRITE) {
                    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                    DataCopy(rmsNormCqResGm_[rmsNormCqResOffset + outOff], outputLocal[outOff],
                             baseParams_->headSizeCq);
                }
            }
            if constexpr (MLA_RMSNORM_RES_ONE_WRITE) {
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                DataCopy(rmsNormCqResGm_[rmsNormCqResOffset], outputLocal, 2 * baseParams_->headSizeCq);
            }
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            rmsNormCqOffset += 2 * static_cast<int64_t>(baseParams_->headSizeCq);
            rmsNormCqResOffset += 2 * static_cast<int64_t>(baseParams_->headSizeCq);
            tokenIndex += 2;
            batchedTokens = curVecToken;
        }
    }
    if constexpr (gaSupportedMx) {
        if (gaGateMx) {
            const uint32_t slotBytes = static_cast<uint32_t>(baseParams_->headSizeCq) * sizeof(float) *
                                       (1U + static_cast<uint32_t>(baseParams_->mm1KGroup));
            RmsNormParam gaParams = {
                baseParams_->reciprocalCq, baseParams_->epsilonCq, static_cast<uint32_t>(vectorRow_),
                baseParams_->headSizeCq,   baseParams_->qcQrScale, baseParams_->isQcQrScaleEnable};
            for (int64_t t = 0; t < 2; t++) {
                RmsNormKnAccumIssue<float>(
                    shareTmpUb[static_cast<uint32_t>(t) * slotBytes],
                    mmCqResGm_[rmsNormCqOffset + t * static_cast<int64_t>(baseParams_->headSizeCq)], 1U,
                    baseParams_->headSizeCq, mm1SplitKnParams, (t == 0) ? EVENT_ID0 : EVENT_ID1);
            }
            for (int64_t t = 0; t < 2; t++) {
                const int64_t outOff = t * static_cast<int64_t>(baseParams_->headSizeCq);
                RmsNormKnMxCompute<rmsNormGammaType, rmsNormCqOutputType, dequantScaleType>(
                    outputLocal[outOff], dequantScaleQcQr[t * dequantScaleCqElementNum], rmsnormGammaCqLocal_,
                    shareTmpUb[static_cast<uint32_t>(t) * slotBytes], gaParams, mm1SplitKnParams,
                    (t == 0) ? EVENT_ID0 : EVENT_ID1);
                if constexpr (!MLA_RMSNORM_RES_ONE_WRITE) {
                    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                    DataCopy(rmsNormCqResGm_[rmsNormCqResOffset + outOff], outputLocal[outOff],
                             baseParams_->headSizeCq);
                }
            }
            if constexpr (MLA_RMSNORM_RES_ONE_WRITE) {
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                DataCopy(rmsNormCqResGm_[rmsNormCqResOffset], outputLocal, 2 * baseParams_->headSizeCq);
            }
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            rmsNormCqOffset += 2 * static_cast<int64_t>(baseParams_->headSizeCq);
            rmsNormCqResOffset += 2 * static_cast<int64_t>(baseParams_->headSizeCq);
            tokenIndex += 2;
            batchedTokens = curVecToken;
        }
    }
    if constexpr (gaSupportedNq) {
        if (gaGateNq) {
            const uint32_t slotBytes = static_cast<uint32_t>(baseParams_->headSizeCq) * sizeof(float) *
                                       (1U + static_cast<uint32_t>(baseParams_->mm1KGroup));
            RmsNormParam gaParams = {
                baseParams_->reciprocalCq, baseParams_->epsilonCq, static_cast<uint32_t>(vectorRow_),
                baseParams_->headSizeCq,   baseParams_->qcQrScale, baseParams_->isQcQrScaleEnable};
            for (int64_t t = 0; t < 2; t++) {
                RmsNormKnAccumIssue<float>(
                    shareTmpUb[static_cast<uint32_t>(t) * slotBytes],
                    mmCqResFloatGm_[rmsNormCqOffset + t * static_cast<int64_t>(baseParams_->headSizeCq)], 1U,
                    baseParams_->headSizeCq, mm1SplitKnParams, (t == 0) ? EVENT_ID0 : EVENT_ID1);
            }
            for (int64_t t = 0; t < 2; t++) {
                const int64_t outOff = t * static_cast<int64_t>(baseParams_->headSizeCq);
                RmsNormKnNqCompute<rmsNormGammaType, rmsNormComputType, rmsNormCqOutputType>(
                    outputLocal[outOff], rmsnormGammaCqLocal_, shareTmpUb[static_cast<uint32_t>(t) * slotBytes],
                    gaParams, mm1SplitKnParams, (t == 0) ? EVENT_ID0 : EVENT_ID1);
                if constexpr (!MLA_RMSNORM_RES_ONE_WRITE) {
                    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                    DataCopy(rmsNormCqResGm_[rmsNormCqResOffset + outOff], outputLocal[outOff],
                             baseParams_->headSizeCq);
                }
            }
            if constexpr (MLA_RMSNORM_RES_ONE_WRITE) {
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                DataCopy(rmsNormCqResGm_[rmsNormCqResOffset], outputLocal, 2 * baseParams_->headSizeCq);
            }
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            rmsNormCqOffset += 2 * static_cast<int64_t>(baseParams_->headSizeCq);
            rmsNormCqResOffset += 2 * static_cast<int64_t>(baseParams_->headSizeCq);
            tokenIndex += 2;
            batchedTokens = curVecToken;
        }
    }
    for (int64_t curVecTokenIdx = batchedTokens; curVecTokenIdx < curVecToken; curVecTokenIdx++) {
        RmsNormCqProcess(outputLocal, dequantScaleQcQr, dequantScaleXLocal, shareTmpUb, tokenIndex, rmsNormCqOffset,
                         rmsNormCqResOffset, curVecTokenIdx, dequantScaleCqElementNum, mm1SplitKnParams);
        rmsNormCqOffset += static_cast<int64_t>(baseParams_->headSizeCq);
        rmsNormCqResOffset += static_cast<int64_t>(baseParams_->headSizeCq);
        tokenIndex++;
    }

    CopyDequantScaleCq(dequantScaleCqElementNum, dequantScaleQcQr, curVecToken, curBlockTokenOffset);
    CopyQueryNormScale(stepTokenIndex, dequantScaleQcQr, curVecToken);
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::RmsNormCqProcess(
    LocalTensor<rmsNormCqOutputType>& outputLocal, LocalTensor<dequantScaleType>& dequantScaleQcQr,
    LocalTensor<float>& dequantScaleXLocal, LocalTensor<uint8_t>& shareTmpUb, int64_t tokenIndex,
    int64_t rmsNormCqOffset, int64_t rmsNormCqResOffset, int64_t curVecTokenIdx, uint64_t dequantScaleCqElementNum,
    const SplitKnAccumParams& splitKnParams)
{
    // MatmulCq ──> RmsNorm(Cq) ──> MatmulQcQr
    SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID1); // wait for vector operations to finish
    // dequantScaleXGm_  [BS , 1] 每个每个token对应一个系数，此处扩展为一个DataBlock
    if constexpr (std::is_same<mmCqOutputType, int32_t>::value ||
                  (std::is_same<mmCqOutputType, float>::value && !isFp8E8m0)) {
        DataCopyPad(dequantScaleXLocal, dequantScaleXGm_[tokenIndex], {1, sizeof(float), 0, 0}, {false, 0, 0, 0});
    }
    uint64_t scaleOffset = curVecTokenIdx * dequantScaleCqElementNum;
    RmsNormParam rmsNormParams = {
        baseParams_->reciprocalCq,         // reciprocal
        baseParams_->epsilonCq,            // epsilon
        static_cast<uint32_t>(vectorRow_), // row
        baseParams_->headSizeCq,           // col
        baseParams_->qcQrScale,            // scale
        baseParams_->isQcQrScaleEnable,    // isScaleEnable
    };
    if constexpr (IsFullQuantMode<rmsNormCqOutputType, dequantScaleType, false>()) {
        RmsNormDynamicQuant<mmCqOutputType, rmsNormGammaType, float, rmsNormComputType, rmsNormCqOutputType,
                            dequantScaleType>(outputLocal, dequantScaleQcQr[scaleOffset], mmCqResGm_[rmsNormCqOffset],
                                              rmsnormGammaCqLocal_, smoothScaleCqLocal_, dequantScaleWDqLocal_,
                                              dequantScaleXLocal, shareTmpUb, rmsNormParams, enableSmoothScalesCq_,
                                              splitKnParams); // ★迁移(Phase B)
    } else {
        if (splitKnParams.NeedAccumulate() && !std::is_same<mmCqOutputType, float>::value) {
            RmsNormNormal<float, rmsNormGammaType, rmsNormComputType, rmsNormCqOutputType, dequantScaleType, true>(
                outputLocal, mmCqResFloatGm_[rmsNormCqOffset], rmsnormGammaCqLocal_, dequantScaleWDqLocal_,
                dequantScaleXLocal, shareTmpUb, rmsNormParams, splitKnParams);
        } else {
            RmsNormNormal<mmCqOutputType, rmsNormGammaType, rmsNormComputType, rmsNormCqOutputType, dequantScaleType>(
                outputLocal, mmCqResGm_[rmsNormCqOffset], rmsnormGammaCqLocal_, dequantScaleWDqLocal_,
                dequantScaleXLocal, shareTmpUb, rmsNormParams, splitKnParams);
        }
    }
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    // RmsNorm(Cq)的结果拷进mmCqResGm_中，用于MatmulQcQr的A矩阵
    DataCopy(rmsNormCqResGm_[rmsNormCqResOffset], outputLocal, baseParams_->headSizeCq);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::CopyDequantScaleCq(uint64_t dequantScaleCqElementNum,
                                                                      LocalTensor<dequantScaleType>& dequantScaleQcQr,
                                                                      int64_t curVecToken, int64_t curBlockTokenOffset)
{
    if constexpr (std::is_same<rmsNormCqOutputType, FP8E4M3>::value && isFp8E8m0) {
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(dequantTool_.deQuantScaleCqGm_[curBlockTokenOffset * dequantScaleCqElementNum], dequantScaleQcQr,
                 curVecToken * dequantScaleCqElementNum);
    } else {
        Brcb(dequantTool_.deQuantScaleCqLocal_[curBlockTokenOffset * FP32_BLOCK_ELEMENT_NUM], dequantScaleQcQr,
             curVecToken, {1, 1});
        if constexpr ((MLAPT::enableDequantOpt || MLAPT::enableGroupComputeOpt) &&
                      !(MLA_FLAG06_EARLY && MLAPT::enableDequantOpt && !MLAPT::enableGroupComputeOpt && !isFp8E8m0 &&
                        IsFullQuantMode<rmsNormCqOutputType, dequantScaleType, true>())) {
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopy(dequantTool_.deQuantScaleCqGm_[curBlockTokenOffset * FP32_BLOCK_ELEMENT_NUM],
                     dequantTool_.deQuantScaleCqLocal_[curBlockTokenOffset * FP32_BLOCK_ELEMENT_NUM],
                     FP32_BLOCK_ELEMENT_NUM * curVecToken);
        }
    }
}
template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::CopyQueryNormScale(int64_t stepTokenIndex,
                                                                      LocalTensor<dequantScaleType>& dequantScaleQcQr,
                                                                      int64_t curVecToken)
{
    if (unlikely(baseParams_->queryNormFlag == 1U)) {
        if constexpr (IsFullQuantMode<mmQcQrInputType, dequantScaleType, true>()) {
            DataCopyParams queryNormScaleParams{static_cast<uint16_t>(curVecToken),
                                                static_cast<uint16_t>(sizeof(dequantScaleQNormType)), 0, 0};
            DataCopyPad(dequantScaleQNormGm_[stepTokenIndex], dequantScaleQcQr, queryNormScaleParams);
        } else if constexpr (std::is_same<mmQcQrInputType, FP8E4M3>::value && isFp8E8m0) {
            const uint32_t mxfp8NormScaleBytes =
                sizeof(dequantScaleQNormType) * (baseParams_->headSizeCq / FP8_E4M3_BLOCK_SIZE);
            DataCopyParams queryNormScaleMxParams{static_cast<uint16_t>(curVecToken),
                                                  static_cast<uint16_t>(mxfp8NormScaleBytes), 0, 0};
            DataCopyPad(dequantScaleQNormGm_[stepTokenIndex *
                                             static_cast<uint16_t>((baseParams_->headSizeCq / FP8_E4M3_BLOCK_SIZE))],
                        dequantScaleQcQr, queryNormScaleMxParams);
        }
    }
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::ComputeBlkScatterOffsets(GlobalTensor<int64_t> indexGm,
                                                                            int64_t tokenIndex, int64_t rows,
                                                                            CkvkrParams& rmsNormAndScatterCkvParams,
                                                                            CkvkrParams& ropeAndScatterKrParams)
{
    int64_t tokenIdxInBatch = 0;
    int64_t batchSeqLen = 0;
    int64_t batchBlkOff = 0;
    int64_t curRows = 0;
    int64_t nextBlkId = 0;
    bool needSpill = false;

    if constexpr (MLAPT::actualSeqMode == ACTUAL_SEQ_MODE::DISABLED) {
        const int64_t batchId = tokenIndex / baseParams_->seq1Size;
        tokenIdxInBatch = tokenIndex % baseParams_->seq1Size;
        batchSeqLen = baseParams_->seq1Size;
        batchBlkOff =
            batchId * CeilDivT(batchSeqLen, static_cast<int64_t>(baseParams_->blockSize)); // cacheIndex的offset
    } else {
        while (actSeqState_.curBatch < static_cast<int64_t>(baseParams_->batchSize) &&
               tokenIndex >= actSeqState_.curPrefix) {
            actSeqState_.prevPrefix = actSeqState_.curPrefix;
            actSeqState_.prevIndexOffset = actSeqState_.curIndexOffset;
            actSeqState_.curPrefix = actualSeqLenGm_(actSeqState_.curBatch);
            actSeqState_.curBatch += 1;
            actSeqState_.curIndexOffset += CeilDivT(actSeqState_.curPrefix - actSeqState_.prevPrefix,
                                                    static_cast<int64_t>(baseParams_->blockSize));
        }
        tokenIdxInBatch = tokenIndex - actSeqState_.prevPrefix;
        batchSeqLen = actSeqState_.curPrefix - actSeqState_.prevPrefix;
        batchBlkOff = actSeqState_.prevIndexOffset;
    }

    int64_t blkIdxOffset = batchBlkOff + tokenIdxInBatch / baseParams_->blockSize;
    int64_t paBlkIdx = indexGm(blkIdxOffset); // 取cacheIdx

    int64_t tokenOffInPage = tokenIdxInBatch % baseParams_->blockSize;

    int64_t rowsLeftInPage = baseParams_->blockSize - tokenOffInPage;
    if (rowsLeftInPage >= rows) {
        curRows = rows;
        needSpill = false;
        nextBlkId = -1;
    } else {
        curRows = rows - rowsLeftInPage;
        needSpill = true;
        nextBlkId = indexGm(blkIdxOffset + 1);
    }

    // --- Materialize for RMSNorm/CKV ---
    MaterializeOffsetsWithHeadSize<kvCacheType, (MLAPT::cacheMode == CACHE_MODE::PA_BLK_NZ)>(
        paBlkIdx, tokenOffInPage, curRows, needSpill, nextBlkId, baseParams_->headSizeCkv, baseParams_->kvCacheStride0,
        rmsNormAndScatterCkvParams);

    MaterializeOffsetsWithHeadSize<krCacheType, (MLAPT::cacheMode == CACHE_MODE::PA_BLK_NZ)>(
        paBlkIdx, tokenOffInPage, curRows, needSpill, nextBlkId, baseParams_->dimHeadRope, baseParams_->krCacheStride0,
        ropeAndScatterKrParams);
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::RmsNormAndScatterCkv(
    LocalTensor<float>& dequantScaleXLocal, LocalTensor<uint8_t>& shareTmpUb,
    LocalTensor<ropeComputType>& cosLocalCkvKr, LocalTensor<ropeComputType>& sinLocalCkvKr,
    CkvkrParams rmsNormAndScatterCkvParams, const SplitKnAccumParams& splitKnParams, bool batch2Layout, bool doScatter)
{
    // MatmulCkvKr ──> RmsNorm(Ckv) ──> Scatter(Ckv)
    uint32_t ckvOutSize = vectorRow_ * baseParams_->headSizeCkv * sizeof(kvCacheType);
    if constexpr (isPertile) { // pertile量化场景，按照concat的最长长度申请内存
        uint32_t ckvTileNum = baseParams_->headSizeCkv / baseParams_->tileSize;
        ckvOutSize = vectorRow_ * (baseParams_->headSizeCkv * sizeof(kvCacheType) + ckvTileNum * sizeof(float));
        ckvOutSize = Align(ckvOutSize, static_cast<uint32_t>(BYTE_BLOCK));
    }
    LocalTensor<uint8_t> ckvOutBase = shareTmpUb;
    LocalTensor<kvCacheType> ckvOutLocal =
        batch2Layout ?
            ckvOutBase[rmsNormAndScatterCkvParams.curVecTokenIdx * ckvOutSize].template ReinterpretCast<kvCacheType>() :
            ckvOutBase.ReinterpretCast<kvCacheType>();
    LocalTensor<uint8_t> ckvNormTmpUb = batch2Layout ?
                                            shareTmpUb[CkvkrWriteSpan()].template ReinterpretCast<uint8_t>() :
                                            shareTmpUb[ckvOutSize].template ReinterpretCast<uint8_t>();

    RmsNormParam ckvNormParams = {
        baseParams_->reciprocalCkv,        // reciprocal
        baseParams_->epsilonCkv,           // epsilon
        static_cast<uint32_t>(vectorRow_), // row
        baseParams_->headSizeCkv,          // col
        baseParams_->kcScale,              // scale
        baseParams_->isKcScaleEnable,      // isScaleEnable
    };
    if constexpr (IsFullQuantMode<rmsNormCkvOutputType, dequantScaleType, false>()) {
        RmsNormAndQuantizeCkv(ckvOutLocal, ckvNormTmpUb, dequantScaleXLocal, ckvNormParams, rmsNormAndScatterCkvParams,
                              splitKnParams);
    } else {
        if (splitKnParams.NeedAccumulate() && std::is_same<mmCkvKrOutputType, bfloat16_t>::value) {
            RmsNormNormal<float, rmsNormGammaType, rmsNormComputType, rmsNormCkvOutputType, dequantScaleType, true>(
                ckvOutLocal, mmCkvKrResFloatGm_[rmsNormAndScatterCkvParams.offset], rmsnormGammaCkvLocal_,
                dequantScaleWDkvKrLocal_, dequantScaleXLocal, ckvNormTmpUb, ckvNormParams, splitKnParams);
        } else {
            RmsNormNormal<mmCkvKrOutputType, rmsNormGammaType, rmsNormComputType, rmsNormCkvOutputType,
                          dequantScaleType>(ckvOutLocal, mmCkvKrResGm_[rmsNormAndScatterCkvParams.offset],
                                            rmsnormGammaCkvLocal_, dequantScaleWDkvKrLocal_, dequantScaleXLocal,
                                            ckvNormTmpUb, ckvNormParams, splitKnParams);
        }
    }

    if (doScatter) {
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        // Scatter(Ckv)
        // RmsNorm(Ckv) ──> Scatter(Ckv) ──> kv_cache_out
        ScatterCkv(ckvOutLocal, rmsNormAndScatterCkvParams);
    }

    if (!batch2Layout) {
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::RmsNormAndQuantizeCkv(
    LocalTensor<kvCacheType>& ckvOutLocal, LocalTensor<uint8_t>& ckvNormTmpUb, LocalTensor<float>& dequantScaleXLocal,
    RmsNormParam ckvNormParams, CkvkrParams rmsNormAndScatterCkvParams, const SplitKnAccumParams& splitKnParams)
{
    // row = vectorRow_ = 1     col = baseParams_->headSizeCkv
    LocalTensor<float> normInLocal = ckvNormTmpUb.ReinterpretCast<float>();
    LocalTensor<uint8_t> normSharedBuf =
        normInLocal[vectorRow_ * baseParams_->headSizeCkv].template ReinterpretCast<uint8_t>();
    RmsNormNormal<mmCkvKrOutputType, rmsNormGammaType, rmsNormComputType, float, dequantScaleType>(
        normInLocal, mmCkvKrResGm_[rmsNormAndScatterCkvParams.offset], rmsnormGammaCkvLocal_, dequantScaleWDkvKrLocal_,
        dequantScaleXLocal, normSharedBuf, ckvNormParams, splitKnParams);

    if constexpr (isPertile) {
        float clipAlpha = isFp8E8m0 ? 1.0f : kNopeClipAlphaGm_.GetValue(0);
        PerTileQuantParams perTileParams = {
            static_cast<uint32_t>(baseParams_->tileSize),                            // baseParams_->tileSize
            static_cast<uint32_t>(baseParams_->headSizeCkv / baseParams_->tileSize), // tileNum
            clipAlpha,                                                               // alpha
            static_cast<uint32_t>(vectorRow_),                                       // row
            baseParams_->headSizeCkv                                                 // col
        };
        if constexpr (std::is_same<rmsNormCkvOutputType, int8_t>::value) {
            QuantPerTile(ckvOutLocal, normInLocal, normSharedBuf, perTileParams);
        } else {
            QuantPerTile8Bit(ckvOutLocal, normInLocal, perTileParams);
        }
    } else {
        Rectangle quantRect{
            static_cast<uint32_t>(vectorRow_),               // row
            static_cast<uint32_t>(baseParams_->headSizeCkv), // col
            static_cast<uint32_t>(baseParams_->headSizeCkv)  // columnStride
        };
        if constexpr (std::is_same<mmCkvKrOutputType, int32_t>::value ||
                      std::is_same<mmCkvKrOutputType, float>::value) {
            QuantPerTensor(ckvOutLocal, normInLocal, quantScaleCkvLocal_, normSharedBuf, quantRect);
        } else {
            QuantPerChannel(ckvOutLocal, normInLocal, quantScaleCkvLocal_, normSharedBuf, quantRect);
        }
    }
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::ScatterCkv(LocalTensor<kvCacheType>& outputLocal,
                                                              CkvkrParams rmsNormAndScatterCkvParams)
{
    if constexpr ((MLAPT::cacheMode == CACHE_MODE::PA_NZ) || (MLAPT::cacheMode == CACHE_MODE::PA_BSND) ||
                  (MLAPT::cacheMode == CACHE_MODE::ND)) {
        int64_t paTokenIdx;
        if constexpr (MLAPT::cacheMode == CACHE_MODE::ND) {
            paTokenIdx = rmsNormAndScatterCkvParams.tokenIndex;
        } else {
            paTokenIdx = cacheIndexGm_(rmsNormAndScatterCkvParams.tokenIndex);
        }
        ScatterCache<kvCacheType, (MLAPT::cacheMode == CACHE_MODE::PA_NZ)>(
            kvCacheGm_, outputLocal,
            ScatterCacheParams{baseParams_->blockSize, paTokenIdx, vectorRow_, baseParams_->headSizeCkv,
                               baseParams_->dtileSize, static_cast<int64_t>(baseParams_->kvCacheStride0)});
        // 刷新量化scale
        if (isPertile && baseParams_->quantScaleRepoMode == 1U) {
            // BSND:
            uint32_t tileNum = baseParams_->headSizeCkv / baseParams_->tileSize;
            LocalTensor<kvCacheType> quantScaleCkvInt8Tensor =
                outputLocal[vectorRow_ * baseParams_->headSizeCkv].template ReinterpretCast<kvCacheType>();
            int64_t dstOffset = 0;
            int64_t colOffsetStart = baseParams_->headSizeCkv;
            if (baseParams_->ckvkrRepoMode == 1U) {
                colOffsetStart += baseParams_->headSizeKr * sizeof(krCacheType);
            }
            if constexpr ((MLAPT::cacheMode != CACHE_MODE::PA_NZ)) {
                dstOffset = colOffsetStart;
            }

            ScatterCacheUnAligned<kvCacheType, (MLAPT::cacheMode == CACHE_MODE::PA_NZ)>(
                kvCacheGm_[dstOffset], quantScaleCkvInt8Tensor,
                ScatterCacheParams{baseParams_->blockSize, paTokenIdx, vectorRow_,
                                   static_cast<int64_t>(tileNum * sizeof(float)), baseParams_->dtileSize,
                                   static_cast<int64_t>(baseParams_->kvCacheStride0)});
        }
    } else {
        ScatterCacheMultiRows<kvCacheType, (MLAPT::cacheMode == CACHE_MODE::PA_BLK_NZ)>(
            kvCacheGm_, outputLocal,
            ScatterCacheParams{baseParams_->blockSize, rmsNormAndScatterCkvParams.cacheOffset, vectorRow_,
                               baseParams_->headSizeCkv, baseParams_->headSizeCkv,
                               static_cast<int64_t>(baseParams_->kvCacheStride0), baseParams_->seq1Size,
                               rmsNormAndScatterCkvParams.tokenIndex},
            rmsNormAndScatterCkvParams.rowsInCurBatch, rmsNormAndScatterCkvParams.cacheOffset,
            rmsNormAndScatterCkvParams.nextBatchOffset);
    }
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::RopeAndScatterKr(
    LocalTensor<float>& dequantScaleXLocal, LocalTensor<uint8_t>& shareTmpUb,
    LocalTensor<ropeComputType>& cosLocalCkvKr, LocalTensor<ropeComputType>& sinLocalCkvKr,
    CkvkrParams ropeAndScatterKrParams, const SplitKnAccumParams& splitKnParams, bool batch2Layout, bool doScatter)
{
    // MatmulCkvKr ──> Rope(Ckv) ──> Scatter(Kr)
    const uint32_t ckvkrRowSize = static_cast<uint32_t>(vectorRow_ * baseParams_->headSizeCkv * sizeof(kvCacheType));
    LocalTensor<uint8_t> krOutBase = shareTmpUb;
    LocalTensor<krCacheType> krOutLocal = batch2Layout ?
                                              krOutBase[(2 + ropeAndScatterKrParams.curVecTokenIdx) * ckvkrRowSize]
                                                  .template ReinterpretCast<krCacheType>() :
                                              krOutBase.ReinterpretCast<krCacheType>();
    LocalTensor<uint8_t> krRopeTmpUb = krOutLocal[baseParams_->dimHeadRope].template ReinterpretCast<uint8_t>();
    int64_t krStride = static_cast<int64_t>(baseParams_->headSizeCkv + baseParams_->headSizeKr); // 512 + 64

    LocalTensor<ropeComputType> cosKrLocal;
    LocalTensor<ropeComputType> sinKrLocal;
    if constexpr (MLAPT::enableDequantOpt || nqWaveMode) {
        cosKrLocal = cosLocalCkvKr[baseParams_->dimHeadRope * ropeAndScatterKrParams.curVecTokenIdx];
        sinKrLocal = sinLocalCkvKr[baseParams_->dimHeadRope * ropeAndScatterKrParams.curVecTokenIdx];
    } else {
        cosKrLocal = cosLocal_[baseParams_->dimHeadRope * ropeAndScatterKrParams.curVecTokenIdx];
        sinKrLocal = sinLocal_[baseParams_->dimHeadRope * ropeAndScatterKrParams.curVecTokenIdx];
    }
    Rectangle krRopeParams{
        static_cast<uint32_t>(vectorRow_),               // row
        static_cast<uint32_t>(baseParams_->dimHeadRope), // col
        static_cast<uint32_t>(krStride)                  // stride
    };
    if constexpr ((std::is_same<mmCkvKrOutputType, int32_t>::value ||
                   (std::is_same<mmCkvKrOutputType, float>::value && !isFp8E8m0)) &&
                  std::is_same<krCacheType, bfloat16_t>::value) {
        LocalTensor<uint8_t> krSharedBuf = krRopeTmpUb;
        RotaryPosEmbPerTensor<mmCkvKrOutputType, ropeComputType, krCacheType, true, MLAPT::enableRope>(
            krOutLocal, mmCkvKrResGm_[ropeAndScatterKrParams.offset], cosKrLocal, sinKrLocal, krSharedBuf, krRopeParams,
            dequantScaleWDkvKrLocal_[baseParams_->headSizeCkv], dequantScaleXLocal, splitKnParams);
    } else if constexpr (std::is_same<krCacheType, int8_t>::value) {
        LocalTensor<ropeSinCosType> krInLocal = krRopeTmpUb.ReinterpretCast<ropeSinCosType>();
        LocalTensor<uint8_t> krSharedBuf =
            krRopeTmpUb.ReinterpretCast<uint8_t>()[baseParams_->dimHeadRope * sizeof(ropeSinCosType)];
        RotaryPosEmbPerTensor<mmCkvKrOutputType, ropeComputType, ropeSinCosType,
                              !std::is_same<mmCkvKrOutputType, bfloat16_t>::value, MLAPT::enableRope>(
            krInLocal, mmCkvKrResGm_[ropeAndScatterKrParams.offset], cosKrLocal, sinKrLocal, krSharedBuf, krRopeParams,
            LocalTensor<float>(), LocalTensor<float>(), splitKnParams);
        RopePostQuantPerChannel(krOutLocal, krInLocal, quantScaleCkrLocal_, krSharedBuf,
                                vectorRow_ * baseParams_->dimHeadRope);
    } else {
        if (splitKnParams.NeedAccumulate() && !std::is_same<mmCkvKrOutputType, float>::value) {
            RotaryPosEmbPerTensor<float, ropeComputType, krCacheType, false, MLAPT::enableRope, true>(
                krOutLocal, mmCkvKrResFloatGm_[ropeAndScatterKrParams.offset], cosKrLocal, sinKrLocal, krRopeTmpUb,
                krRopeParams, LocalTensor<float>(), LocalTensor<float>(), splitKnParams);
        } else {
            RotaryPosEmbPerTensor<mmCkvKrOutputType, ropeComputType, krCacheType, false, MLAPT::enableRope>(
                krOutLocal, mmCkvKrResGm_[ropeAndScatterKrParams.offset], cosKrLocal, sinKrLocal, krRopeTmpUb,
                krRopeParams, LocalTensor<float>(), LocalTensor<float>(), splitKnParams);
        }
    }
    if (doScatter) {
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        // scatter(Kr)
        // Rope(Kr) ──> Scatter(Kr) ──> kr_cache_out
        ScatterKr(krOutLocal, ropeAndScatterKrParams);
    }

    if (!batch2Layout) {
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::ScatterKr(LocalTensor<krCacheType>& outputKrLocal,
                                                             CkvkrParams ropeAndScatterKrParams)
{
    int64_t paTokenIdx;
    if constexpr ((MLAPT::cacheMode == CACHE_MODE::PA_NZ) || (MLAPT::cacheMode == CACHE_MODE::PA_BSND) ||
                  (MLAPT::cacheMode == CACHE_MODE::ND)) {
        if constexpr (MLAPT::cacheMode == CACHE_MODE::ND) {
            paTokenIdx = ropeAndScatterKrParams.tokenIndex;
        } else {
            paTokenIdx = cacheIndexGm_(ropeAndScatterKrParams.tokenIndex);
        }
        if (isPertile && baseParams_->ckvkrRepoMode == 1U) {
            int64_t dstOffset;
            if constexpr ((MLAPT::cacheMode == CACHE_MODE::PA_NZ)) {
                constexpr uint8_t col0 = ALIGN_BLOCK_SIZE / sizeof(kvCacheType);
                dstOffset = CeilDiv(baseParams_->headSizeCkv, col0) * col0 * baseParams_->blockSize; // 列方向的偏移
            } else {
                dstOffset = baseParams_->headSizeCkv;
            }
            LocalTensor<kvCacheType> outputKrInt8Tensor = outputKrLocal.template ReinterpretCast<kvCacheType>();
            ScatterCache<kvCacheType, (MLAPT::cacheMode == CACHE_MODE::PA_NZ)>(
                kvCacheGm_[dstOffset], outputKrInt8Tensor,
                ScatterCacheParams{baseParams_->blockSize, paTokenIdx, vectorRow_,
                                   static_cast<int64_t>(baseParams_->dimHeadRope * sizeof(krCacheType)),
                                   baseParams_->dtileSize, static_cast<int64_t>(baseParams_->kvCacheStride0)});
        } else {
            ScatterCache<krCacheType, (MLAPT::cacheMode == CACHE_MODE::PA_NZ)>(
                krCacheGm_, outputKrLocal,
                ScatterCacheParams{baseParams_->blockSize, paTokenIdx, vectorRow_, baseParams_->dimHeadRope,
                                   baseParams_->dimHeadRope, static_cast<int64_t>(baseParams_->krCacheStride0)});
        }
    } else {
        ScatterCacheMultiRows<krCacheType, (MLAPT::cacheMode == CACHE_MODE::PA_BLK_NZ)>(
            krCacheGm_, outputKrLocal,
            ScatterCacheParams{baseParams_->blockSize, ropeAndScatterKrParams.cacheOffset, vectorRow_,
                               baseParams_->dimHeadRope, baseParams_->dimHeadRope,
                               static_cast<int64_t>(baseParams_->krCacheStride0), baseParams_->seq1Size,
                               ropeAndScatterKrParams.tokenIndex},
            ropeAndScatterKrParams.rowsInCurBatch, ropeAndScatterKrParams.cacheOffset,
            ropeAndScatterKrParams.nextBatchOffset);
    }
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::EmitCkvkrWrites()
{
    if (!ckvkrDeferArmed_) {
        return;
    }
    ckvkrDeferArmed_ = false;
    LocalTensor<uint8_t> shareTmpUb = ckvkrDeferBase_;
    const uint32_t slotSize = static_cast<uint32_t>(vectorRow_ * baseParams_->headSizeCkv * sizeof(kvCacheType));
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    bool kvMerged = false;
    if constexpr (MLA_CKVKR_MERGE_KV) {
        if (ckvkrDeferNum_ >= 2U) {
            if constexpr ((MLAPT::cacheMode == CACHE_MODE::PA_BSND) || (MLAPT::cacheMode == CACHE_MODE::ND)) {
                const int64_t blkSize = baseParams_->blockSize;
                if (blkSize > 0) {
                    int64_t idx0 = 0;
                    int64_t idx1 = 0;
                    if constexpr (MLAPT::cacheMode == CACHE_MODE::ND) {
                        idx0 = deferredCkvParams_[0].tokenIndex;
                        idx1 = deferredCkvParams_[1].tokenIndex;
                    } else {
                        idx0 = cacheIndexGm_(deferredCkvParams_[0].tokenIndex);
                        idx1 = cacheIndexGm_(deferredCkvParams_[1].tokenIndex);
                    }
                    if (idx0 >= 0 && idx1 == idx0 + 1 && (idx0 % blkSize) + 1 < blkSize) {
                        const int64_t off0 = GetCacheOffset(idx0, blkSize, baseParams_->dtileSize,
                                                            static_cast<int64_t>(baseParams_->kvCacheStride0));
                        const int64_t off1 = GetCacheOffset(idx1, blkSize, baseParams_->dtileSize,
                                                            static_cast<int64_t>(baseParams_->kvCacheStride0));
                        if (off1 == off0 + static_cast<int64_t>(baseParams_->headSizeCkv)) {
                            LocalTensor<kvCacheType> ckvPair = shareTmpUb.ReinterpretCast<kvCacheType>();
                            DataCopy(kvCacheGm_[off0], ckvPair, 2U * static_cast<uint32_t>(baseParams_->headSizeCkv));
                            kvMerged = true;
                        }
                    }
                }
            }
        }
    }
    if (!kvMerged) {
        for (uint32_t t = 0; t < ckvkrDeferNum_ && t < 2U; t++) {
            LocalTensor<kvCacheType> ckvSlot =
                shareTmpUb[deferredCkvParams_[t].curVecTokenIdx * slotSize].template ReinterpretCast<kvCacheType>();
            ScatterCkv(ckvSlot, deferredCkvParams_[t]);
        }
    }
    for (uint32_t t = 0; t < ckvkrDeferNum_ && t < 2U; t++) {
        LocalTensor<krCacheType> krSlot =
            shareTmpUb[(2 + deferredKrParams_[t].curVecTokenIdx) * slotSize].template ReinterpretCast<krCacheType>();
        ScatterKr(krSlot, deferredKrParams_[t]);
    }
    ckvkrDeferNum_ = 0U;
    SetFlag<HardEvent::MTE3_V>(CKVKR_STEP_DRAIN);
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::RmsNormRopeScatterCkvKr(int64_t tokenIndex, int64_t rmsNormCkvOffset,
                                                                           int64_t ropeKrOffset, int64_t curVecToken,
                                                                           int64_t batchOffset)
{
    if (curVecToken == 0) {
        return;
    }
    if (blockIdx_ >= curVectorBlockNum_) {
        return;
    }
    LocalTensor<float> xDeqScaleLocal = shareBuffer_.Get<float>();
    LocalTensor<ropeComputType> cosLocalKr =
        xDeqScaleLocal[FP32_BLOCK_ELEMENT_NUM].template ReinterpretCast<ropeComputType>();
    LocalTensor<ropeComputType> sinLocalKr = cosLocalKr[baseParams_->dimHeadRope * curVecToken];
    LocalTensor<uint8_t> krTmpUb =
        sinLocalKr[baseParams_->dimHeadRope * curVecToken].template ReinterpretCast<uint8_t>();
    if constexpr ((MLAPT::enableDequantOpt || nqWaveMode) && MLAPT::enableRope) {
        GatherSinCos<ropeSinCosType, ropeComputType>(cosLocalKr, sinLocalKr, ropeCosGm_, ropeSinGm_, tokenIndex,
                                                     curVecToken, krTmpUb, vectorRow_, baseParams_->dimHeadRope);
    }
    if constexpr (MLAPT::emptyMode == EMPTY_TENSOR_MODE::EMPTY_CACHE) {
        return;
    }

    SplitKnAccumParams mm2SplitKnParams{0, 0};
    if (baseParams_->mm2KGroup > 1) {
        mm2SplitKnParams.kGroups = baseParams_->mm2KGroup;
        mm2SplitKnParams.partialStride =
            static_cast<uint64_t>(baseParams_->stepBatchSize) * (baseParams_->headSizeCkv + baseParams_->dimHeadRope);
    }

    if constexpr (MLA_CKVKR_BATCH2) {
        const bool takeBatch2 = (curVecToken == 2) && !isPertile && (vectorRow_ == 1);
        if (takeBatch2) {
            if (batchOffset > 0) {
                WaitFlag<HardEvent::MTE3_V>(CKVKR_STEP_DRAIN);
            }
            constexpr bool kvMergeModeOk =
                (MLAPT::cacheMode == CACHE_MODE::PA_BSND) || (MLAPT::cacheMode == CACHE_MODE::ND);
            const bool mergeKvPair = MLA_CKVKR_MERGE_KV && kvMergeModeOk && (curVecToken == 2) && !isPertile &&
                                     !MLAPT::enableGroupComputeOpt && (baseParams_->blockSize > 0);
            const bool doScatterNow = !mergeKvPair;
            for (int64_t curVecTokenIdx = 0; curVecTokenIdx < curVecToken; curVecTokenIdx++) {
                SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                if constexpr (std::is_same<mmCqOutputType, int32_t>::value ||
                              (std::is_same<mmCqOutputType, float>::value && !isFp8E8m0)) {
                    DataCopyPad(xDeqScaleLocal, dequantScaleXGm_[tokenIndex], {1, sizeof(float), 0, 0},
                                {false, 0, 0, 0});
                }
                CkvkrParams rmsNormAndScatterCkvParams{tokenIndex, rmsNormCkvOffset, curVecTokenIdx, 0, 0, 0};
                CkvkrParams ropeAndScatterKrParams{tokenIndex, ropeKrOffset, curVecTokenIdx, 0, 0, 0};
                if constexpr ((MLAPT::cacheMode == CACHE_MODE::PA_BLK_BSND) ||
                              (MLAPT::cacheMode == CACHE_MODE::PA_BLK_NZ)) {
                    ComputeBlkScatterOffsets(cacheIndexGm_, tokenIndex, vectorRow_, rmsNormAndScatterCkvParams,
                                             ropeAndScatterKrParams);
                }
                RmsNormAndScatterCkv(xDeqScaleLocal, krTmpUb, cosLocalKr, sinLocalKr, rmsNormAndScatterCkvParams,
                                     mm2SplitKnParams, true, doScatterNow);
                SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                RopeAndScatterKr(xDeqScaleLocal, krTmpUb, cosLocalKr, sinLocalKr, ropeAndScatterKrParams,
                                 mm2SplitKnParams, true, doScatterNow);
                if (mergeKvPair && curVecTokenIdx < 2) {
                    deferredCkvParams_[curVecTokenIdx] = rmsNormAndScatterCkvParams;
                    deferredKrParams_[curVecTokenIdx] = ropeAndScatterKrParams;
                }
                tokenIndex += 1;
                rmsNormCkvOffset += static_cast<int64_t>(baseParams_->headSizeCkv + baseParams_->dimHeadRope);
                ropeKrOffset += static_cast<int64_t>(baseParams_->headSizeCkv + baseParams_->dimHeadRope);
            }
            if (mergeKvPair) {
                ckvkrDeferNum_ = static_cast<uint32_t>(curVecToken);
                ckvkrDeferBase_ = krTmpUb;
                ckvkrDeferArmed_ = true;
                EmitCkvkrWrites();
            } else {
                SetFlag<HardEvent::MTE3_V>(CKVKR_STEP_DRAIN);
            }
            ckvkrBatchDrainPending_ = true;
            return;
        }
    }
    for (int64_t curVecTokenIdx = 0; curVecTokenIdx < curVecToken; curVecTokenIdx++) {
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);

        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);

        if constexpr (std::is_same<mmCqOutputType, int32_t>::value ||
                      (std::is_same<mmCqOutputType, float>::value && !isFp8E8m0)) {
            DataCopyPad(xDeqScaleLocal, dequantScaleXGm_[tokenIndex], {1, sizeof(float), 0, 0}, {false, 0, 0, 0});
        }

        CkvkrParams ckvScatterParams{tokenIndex, rmsNormCkvOffset, curVecTokenIdx, 0, 0, 0};

        CkvkrParams krScatterParams{tokenIndex, ropeKrOffset, curVecTokenIdx, 0, 0, 0};

        if constexpr ((MLAPT::cacheMode == CACHE_MODE::PA_BLK_BSND) || (MLAPT::cacheMode == CACHE_MODE::PA_BLK_NZ)) {
            ComputeBlkScatterOffsets(cacheIndexGm_, tokenIndex, vectorRow_, ckvScatterParams, krScatterParams);
        }
        RmsNormAndScatterCkv(xDeqScaleLocal, krTmpUb, cosLocalKr, sinLocalKr, ckvScatterParams, mm2SplitKnParams);

        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);

        RopeAndScatterKr(xDeqScaleLocal, krTmpUb, cosLocalKr, sinLocalKr, krScatterParams, mm2SplitKnParams);

        tokenIndex += 1;
        rmsNormCkvOffset += static_cast<int64_t>(baseParams_->headSizeCkv + baseParams_->dimHeadRope);
        ropeKrOffset += static_cast<int64_t>(baseParams_->headSizeCkv + baseParams_->dimHeadRope);
    }
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::RopeQr(int64_t ropeQrOffset, int64_t ropeQrResOffset,
                                                          int64_t curVecToken, int64_t curBlockTokenOffset)
{
    if (blockIdx_ >= curVectorBlockNum_) {
        return;
    }
    uint64_t stride = static_cast<uint64_t>(baseParams_->dimHeadRope + baseParams_->dimHeadSizeQc);

    LocalTensor<ropeOutputType> outputLocal;
    LocalTensor<float> channelDeqScaleLocal =
        shareBuffer_.Get<uint8_t>()[PostCkvkrShareBase()].template ReinterpretCast<float>();
    if constexpr (IsFullQuantMode<mmQcQrInputType, dequantScaleType, true>()) {
        uint64_t row = baseParams_->numHeadSize;
        uint64_t col = baseParams_->dimHeadRope;
        DataCopyExtParams copyParams{static_cast<uint16_t>(row), static_cast<uint32_t>(col * sizeof(float)),
                                     static_cast<uint32_t>((stride - col) * sizeof(float)), 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        DataCopyPad(channelDeqScaleLocal, deqScaleQcQrW_[baseParams_->dimHeadSizeQc], copyParams,
                    padParams); // 复用内存
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);

        outputLocal = channelDeqScaleLocal[row * col].template ReinterpretCast<ropeOutputType>();
    } else {
        outputLocal = shareBuffer_.Get<uint8_t>()[PostCkvkrShareBase()].template ReinterpretCast<ropeOutputType>();
    }
    LocalTensor<uint8_t> ropeShareTmpUb = outputLocal[baseParams_->headSizeQr].template ReinterpretCast<uint8_t>();

    // row, col, stride
    Rectangle ropeParams{baseParams_->numHeadSize, baseParams_->dimHeadRope, static_cast<uint32_t>(stride)};

    for (int64_t curVecTokenIdx = 0; curVecTokenIdx < curVecToken; curVecTokenIdx++) {
        // MatmulQcQr ──> Rope(Qr) ──> query_rope_out
        if constexpr (IsFullQuantMode<mmQcQrInputType, dequantScaleType, true>()) {
            RotaryPosEmbPerTensor<mmQcQrOutputType, ropeComputType, ropeOutputType, true, MLAPT::enableRope>(
                outputLocal, mmQcQrResGm_[ropeQrOffset], cosLocal_[baseParams_->dimHeadRope * curVecTokenIdx],
                sinLocal_[baseParams_->dimHeadRope * curVecTokenIdx], ropeShareTmpUb, ropeParams, channelDeqScaleLocal,
                dequantTool_.deQuantScaleCqLocal_[(curBlockTokenOffset + curVecTokenIdx) * FP32_BLOCK_ELEMENT_NUM]);
        } else {
            RotaryPosEmbPerTensor<mmQcQrOutputType, ropeComputType, ropeOutputType, false, MLAPT::enableRope>(
                outputLocal, mmQcQrResGm_[ropeQrOffset], cosLocal_[baseParams_->dimHeadRope * curVecTokenIdx],
                sinLocal_[baseParams_->dimHeadRope * curVecTokenIdx], ropeShareTmpUb, ropeParams);
        }

        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

        DataCopy(qrOutGm_[ropeQrResOffset], outputLocal, baseParams_->headSizeQr);

        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);

        ropeQrOffset += static_cast<int64_t>(baseParams_->numHeadSize) *
                        static_cast<int64_t>(baseParams_->dimHeadSizeQc + baseParams_->dimHeadRope);
        ropeQrResOffset += static_cast<int64_t>(baseParams_->headSizeQr);
    }
}

// 用于enableGroupComputeOpt场景 BS = 1 ， G = 8
template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::RopeQrSplitNGroupCase(int64_t ropeQrOffset, int64_t ropeQrResOffset)
{
    if (blockIdx_ < QC_CORE_NUM * cvRatio_ || blockIdx_ >= (QC_CORE_NUM + QR_CORE_NUM) * cvRatio_) {
        return;
    }

    uint32_t row = baseParams_->numHeadSize / QC_CORE_NUM;
    uint32_t col = baseParams_->dimHeadRope;
    int64_t stride = col;
    int64_t strideScale = static_cast<int64_t>(baseParams_->dimHeadRope + baseParams_->dimHeadSizeQc);
    int64_t deqScaleOffset = baseParams_->dimHeadSizeQc + row *
                                                              (baseParams_->dimHeadSizeQc + baseParams_->dimHeadRope) *
                                                              (blockIdx_ - QC_CORE_NUM * cvRatio_);
    LocalTensor<float> channelDeqScaleLocal =
        shareBuffer_.Get<uint8_t>()[PostCkvkrShareBase()].template ReinterpretCast<float>();
    DataCopyExtParams copyParams{static_cast<uint16_t>(row), static_cast<uint32_t>(col * sizeof(float)),
                                 static_cast<uint32_t>((strideScale - col) * sizeof(float)), 0, 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    DataCopyPad(channelDeqScaleLocal, deqScaleQcQrW_[deqScaleOffset], copyParams, padParams); // 复用内存
    SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);

    LocalTensor<ropeOutputType> outputLocal =
        channelDeqScaleLocal[row * col].template ReinterpretCast<ropeOutputType>();
    LocalTensor<uint8_t> ropeShareTmpUb = outputLocal[baseParams_->headSizeQr].template ReinterpretCast<uint8_t>();

    Rectangle ropeParams{
        static_cast<uint32_t>(col),   // row
        static_cast<uint32_t>(col),   // col
        static_cast<uint32_t>(stride) // stride
    };

    for (int64_t curVecTokenIdx = 0; curVecTokenIdx < curVectorBlockNum_; curVecTokenIdx++) {
        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
        if constexpr (MLAPT::enableRope) {
            GatherSinCos<ropeSinCosType, ropeComputType>(cosLocal_, sinLocal_, ropeCosGm_, ropeSinGm_,
                                                         curVecTokenIdx * baseParams_->dimHeadRope, 1, ropeShareTmpUb,
                                                         vectorRow_, baseParams_->dimHeadRope);
        }

        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);

        RotaryPosEmbPerTensor<mmQcQrOutputType, ropeComputType, ropeOutputType, false, MLAPT::enableRope>(
            outputLocal, mmQcQrResGm_[ropeQrOffset], cosLocal_, sinLocal_, ropeShareTmpUb, ropeParams,
            channelDeqScaleLocal, dequantTool_.deQuantScaleCqLocal_);

        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(qrOutGm_[ropeQrResOffset], outputLocal, row * col);

        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);

        ropeQrOffset += static_cast<int64_t>(baseParams_->numHeadSize) * static_cast<int64_t>(baseParams_->dimHeadRope);
        ropeQrResOffset += static_cast<int64_t>(baseParams_->headSizeQr);
    }
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::RopeQrSplitN(const RopeQrSplitNParams& ropeQrSplitNParams,
                                                                const LocalTensor<mmQcQrOutputType>& waveUbQr,
                                                                const uint32_t wavePitch)
{
    uint32_t ropeCnt = ropeQrSplitNParams.ropeCnt;
    if (ropeCnt == 0U) {
        return;
    }
    uint32_t colQr = baseParams_->dimHeadRope;
    uint32_t deQuantScaleCqOffset = ropeQrSplitNParams.deQuantScaleCqOffset;
    DataCopyParams outputRopeParams{
        static_cast<uint16_t>(ropeCnt), static_cast<uint16_t>(colQr * sizeof(ropeOutputType) / ALIGN_BLOCK_SIZE), 0,
        static_cast<uint16_t>(ropeQrSplitNParams.ropeDstStride * sizeof(ropeOutputType) / ALIGN_BLOCK_SIZE)};

    Rectangle ropeParams{
        ropeCnt,                                             // row
        colQr,                                               // col
        static_cast<uint32_t>(ropeQrSplitNParams.ropeStride) // stride
    };

    GlobalTensor<mmQcQrOutputType> inputGmRope = mmQcQrResGm_[ropeQrSplitNParams.ropeQrOffset];
    GlobalTensor<ropeOutputType> outputGmRope = qrOutGm_[ropeQrSplitNParams.ropeQrResOffset];

    LocalTensor<uint8_t> shareTmpUb = shareBuffer_.Get<uint8_t>()[PostCkvkrShareBase()];
    LocalTensor<ropeOutputType> outputLocalRope = shareTmpUb.ReinterpretCast<ropeOutputType>();
    LocalTensor<uint8_t> ropeShareTmpUb = outputLocalRope[ropeCnt * colQr].template ReinterpretCast<uint8_t>();

    SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    if constexpr (!isFp8E8m0) {
        if (wavePitch > 0) {
            uint32_t cnt = ropeCnt * colQr;
            if constexpr (MLA_PRELOAD_QCQR_SCALE && IsFullQuantMode<mmQcQrInputType, dequantScaleType, true>()) {
                const uint32_t sOut = cnt * static_cast<uint32_t>(sizeof(ropeOutputType));
                const uint32_t sFp = cnt * static_cast<uint32_t>(sizeof(float));
                const uint32_t dequantOutBytes = static_cast<uint32_t>(mmQcQrParam_.m) *
                                                 (static_cast<uint32_t>(baseParams_->dimHeadSizeQc) / cvRatio_) *
                                                 static_cast<uint32_t>(sizeof(mmQnInputType));
                const uint32_t ropeBase =
                    (dequantOutBytes + ALIGN_BLOCK_SIZE - 1U) / ALIGN_BLOCK_SIZE * ALIGN_BLOCK_SIZE;
                const uint32_t outBase = ropeBase + 2U * sFp;
                outputLocalRope = shareTmpUb[outBase + ropeQrSplitNParams.scratchSlot * sOut]
                                      .template ReinterpretCast<ropeOutputType>();
                ropeShareTmpUb = shareTmpUb[ropeBase];
                LocalTensor<float> kFp32Local = ropeShareTmpUb.ReinterpretCast<float>(); // 紧凑输入[ropeCnt,colQr]
                LocalTensor<float> kOutFp32Local = kFp32Local[cnt];                      // rope计算输出
                LocalTensor<float> scaleChLocal = qcqrScaleLocal_[static_cast<uint32_t>(baseParams_->dimHeadSizeQc) +
                                                                  ropeQrSplitNParams.deqScaleOffset];
                Dequant(kFp32Local, waveUbQr, scaleChLocal, dequantTool_.deQuantScaleCqLocal_[deQuantScaleCqOffset],
                        Rectangle{ropeCnt, colQr, wavePitch}, colQr);
                RotaryPosEmb<ropeComputType>(kOutFp32Local, kFp32Local, cosLocal_[ropeQrSplitNParams.sinCosOffset],
                                             sinLocal_[ropeQrSplitNParams.sinCosOffset], ropeCnt, colQr, colQr);
                Cast(outputLocalRope, kOutFp32Local, RoundMode::CAST_RINT, cnt);
            } else if constexpr (std::is_same<mmQcQrInputType, bfloat16_t>::value) {
                const uint32_t colQcSingleR2 = baseParams_->dimHeadSizeQc / cvRatio_;
                const uint32_t sOut = cnt * static_cast<uint32_t>(sizeof(ropeOutputType));
                const uint32_t qcScratchBytes = static_cast<uint32_t>(mmQcQrParam_.m) * colQcSingleR2 *
                                                static_cast<uint32_t>(sizeof(mmQnInputType));
                const uint32_t ropeBase =
                    (qcScratchBytes + ALIGN_BLOCK_SIZE - 1U) / ALIGN_BLOCK_SIZE * ALIGN_BLOCK_SIZE;
                outputLocalRope = shareTmpUb[ropeBase + ropeQrSplitNParams.scratchSlot * sOut]
                                      .template ReinterpretCast<ropeOutputType>();
                LocalTensor<uint8_t> ropeScratchUb = shareTmpUb[ropeBase + 2U * sOut];
                LocalTensor<mmQcQrOutputType> kLocalUb = ropeScratchUb.template ReinterpretCast<mmQcQrOutputType>();
                LocalTensor<float> kFp32Local =
                    kLocalUb[cnt].template ReinterpretCast<float>(); // 紧凑输入[ropeCnt,colQr]
                LocalTensor<float> kOutFp32Local = kFp32Local[cnt];  // rope计算输出
                CopyStridedVf<mmQcQrOutputType>(kLocalUb, waveUbQr, ropeCnt, colQr, wavePitch, colQr);
                Cast(kFp32Local, kLocalUb, RoundMode::CAST_NONE, cnt);
                RotaryPosEmb<ropeComputType>(kOutFp32Local, kFp32Local, cosLocal_[ropeQrSplitNParams.sinCosOffset],
                                             sinLocal_[ropeQrSplitNParams.sinCosOffset], ropeCnt, colQr, colQr);
                Cast(outputLocalRope, kOutFp32Local, RoundMode::CAST_RINT, cnt);
            } else {
                LocalTensor<float> kFp32Local = ropeShareTmpUb.ReinterpretCast<float>(); // 紧凑输入[ropeCnt,colQr]
                LocalTensor<float> kOutFp32Local = kFp32Local[cnt];                      // rope计算输出
                LocalTensor<float> scaleChLocal = kOutFp32Local[cnt];                    // 通道scale[colQr]
                GlobalTensor<float> deqScaleRope = deqScaleQcQrW_[ropeQrSplitNParams.ropeQrOffset];
                DataCopyPad(scaleChLocal, deqScaleRope[ropeQrSplitNParams.deqScaleOffset],
                            DataCopyExtParams{static_cast<uint16_t>(1), static_cast<uint32_t>(colQr * sizeof(float)),
                                              static_cast<int64_t>(ropeQrSplitNParams.ropeStride - colQr) *
                                                  static_cast<int64_t>(sizeof(float)),
                                              static_cast<int64_t>(0), static_cast<uint32_t>(0)},
                            DataCopyPadExtParams<float>{false, 0, 0, 0});
                SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
                Dequant(kFp32Local, waveUbQr, scaleChLocal, dequantTool_.deQuantScaleCqLocal_[deQuantScaleCqOffset],
                        Rectangle{ropeCnt, colQr, wavePitch}, colQr);
                RotaryPosEmb<ropeComputType>(kOutFp32Local, kFp32Local, cosLocal_[ropeQrSplitNParams.sinCosOffset],
                                             sinLocal_[ropeQrSplitNParams.sinCosOffset], ropeCnt, colQr, colQr);
                Cast(outputLocalRope, kOutFp32Local, RoundMode::CAST_RINT, cnt);
            }
        } else {
            // ===== 原 GM 路径 =====
            if constexpr (IsFullQuantMode<mmQcQrInputType, dequantScaleType, true>()) {
                GlobalTensor<float> deqScaleRope = deqScaleQcQrW_[ropeQrSplitNParams.ropeQrOffset];
                RotaryPosEmbPerHead<mmQcQrOutputType, ropeComputType, ropeOutputType, true, MLAPT::enableRope>(
                    outputLocalRope, inputGmRope[ropeQrSplitNParams.inputOffsetRope],
                    cosLocal_[ropeQrSplitNParams.sinCosOffset], sinLocal_[ropeQrSplitNParams.sinCosOffset],
                    ropeShareTmpUb, ropeParams, ropeQrSplitNParams.ropeStride,
                    deqScaleRope[ropeQrSplitNParams.deqScaleOffset],
                    dequantTool_.deQuantScaleCqLocal_[deQuantScaleCqOffset]);
            } else {
                RotaryPosEmbPerHead<mmQcQrOutputType, ropeComputType, ropeOutputType, false, MLAPT::enableRope>(
                    outputLocalRope, inputGmRope[ropeQrSplitNParams.inputOffsetRope],
                    cosLocal_[ropeQrSplitNParams.sinCosOffset], sinLocal_[ropeQrSplitNParams.sinCosOffset],
                    ropeShareTmpUb, ropeParams, ropeQrSplitNParams.ropeStride);
            }
        }
    } else {
        if (wavePitch > 0) {
            const uint32_t cnt = ropeCnt * colQr;
            const uint32_t sOut = cnt * static_cast<uint32_t>(sizeof(ropeOutputType));
            const uint32_t sFp = cnt * static_cast<uint32_t>(sizeof(float));
            const uint32_t castOutBytes = static_cast<uint32_t>(mmQcQrParam_.m) *
                                          (static_cast<uint32_t>(baseParams_->dimHeadSizeQc) / cvRatio_) *
                                          static_cast<uint32_t>(sizeof(mmQnInputType));
            const uint32_t ropeBase = (castOutBytes + ALIGN_BLOCK_SIZE - 1U) / ALIGN_BLOCK_SIZE * ALIGN_BLOCK_SIZE;
            outputLocalRope = shareTmpUb[ropeBase + 2U * sFp + ropeQrSplitNParams.scratchSlot * sOut]
                                  .template ReinterpretCast<ropeOutputType>();
            LocalTensor<float> kFp32Local = shareTmpUb[ropeBase].template ReinterpretCast<float>(); // 紧凑输入
            LocalTensor<float> kOutFp32Local = kFp32Local[cnt];                                     // rope计算输出
            CopyStridedVf<mmQcQrOutputType>(kFp32Local, waveUbQr, ropeCnt, colQr, wavePitch, colQr);
            RotaryPosEmb<ropeComputType>(kOutFp32Local, kFp32Local, cosLocal_[ropeQrSplitNParams.sinCosOffset],
                                         sinLocal_[ropeQrSplitNParams.sinCosOffset], ropeCnt, colQr, colQr);
            Cast(outputLocalRope, kOutFp32Local, RoundMode::CAST_RINT, cnt);
        } else {
            if constexpr (IsFullQuantMode<mmQcQrInputType, dequantScaleType, true>()) {
                GlobalTensor<dequantScaleType> deqScaleRope = deqScaleQcQrW_[ropeQrSplitNParams.ropeQrOffset];
                RotaryPosEmbPerHead<mmQcQrOutputType, ropeComputType, ropeOutputType, true, MLAPT::enableRope>(
                    outputLocalRope, inputGmRope[ropeQrSplitNParams.inputOffsetRope],
                    cosLocal_[ropeQrSplitNParams.sinCosOffset], sinLocal_[ropeQrSplitNParams.sinCosOffset],
                    ropeShareTmpUb, ropeParams, ropeQrSplitNParams.ropeStride,
                    deqScaleRope[ropeQrSplitNParams.deqScaleOffset],
                    dequantTool_.deQuantScaleCqLocal_[deQuantScaleCqOffset]);
            } else {
                RotaryPosEmbPerHead<mmQcQrOutputType, ropeComputType, ropeOutputType, false, MLAPT::enableRope>(
                    outputLocalRope, inputGmRope[ropeQrSplitNParams.inputOffsetRope],
                    cosLocal_[ropeQrSplitNParams.sinCosOffset], sinLocal_[ropeQrSplitNParams.sinCosOffset],
                    ropeShareTmpUb, ropeParams, ropeQrSplitNParams.ropeStride);
            }
        }
    }

    SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);

    DataCopy(outputGmRope[ropeQrSplitNParams.outputOffsetRope], outputLocalRope, outputRopeParams);
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::CastQcQrSplitN(const CastQcQrSplitNParams& castQcQrSplitN,
                                                                  const LocalTensor<mmQcQrOutputType>& waveUb,
                                                                  const uint32_t wavePitch)
{
    uint32_t row = mmQcQrParam_.m;
    uint32_t colQc = baseParams_->dimHeadSizeQc;
    uint32_t colQcSingle = colQc / cvRatio_;
    uint32_t count = row * colQcSingle;

    DataCopyParams inputCopyParams{
        static_cast<uint16_t>(row), static_cast<uint16_t>(colQcSingle * sizeof(mmQcQrOutputType) / ALIGN_BLOCK_SIZE),
        static_cast<uint16_t>(castQcQrSplitN.srcStride * sizeof(mmQcQrOutputType) / ALIGN_BLOCK_SIZE), 0};

    DataCopyParams outputCopyParams{
        static_cast<uint16_t>(row), static_cast<uint16_t>(colQcSingle * sizeof(mmQnInputType) / ALIGN_BLOCK_SIZE), 0,
        static_cast<uint16_t>(castQcQrSplitN.dstStride * sizeof(mmQnInputType) / ALIGN_BLOCK_SIZE)};

    GlobalTensor<mmQcQrOutputType> inputGm = mmQcQrResGm_[castQcQrSplitN.mmQnPreCastOffset];
    GlobalTensor<mmQnInputType> outputGm = mmQcQrResDequantGm_[castQcQrSplitN.mmQnPreCastResOffset];

    LocalTensor<uint8_t> shareTmpUb = shareBuffer_.Get<uint8_t>()[PostCkvkrShareBase()];
    LocalTensor<mmQcQrOutputType> inputLocal = shareTmpUb.ReinterpretCast<mmQcQrOutputType>();
    LocalTensor<mmQnInputType> outputLocal = inputLocal.template ReinterpretCast<mmQnInputType>();

    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    if (wavePitch > 0) {
        CopyStridedVf<mmQcQrOutputType>(inputLocal, waveUb, row, colQcSingle, wavePitch, colQcSingle);
    } else {
        DataCopy(inputLocal, inputGm[castQcQrSplitN.inputOffset], inputCopyParams);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
    }
    // cast
    Cast(outputLocal, inputLocal, RoundMode::CAST_RINT, count);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
    // copy out
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
    DataCopy(outputGm[castQcQrSplitN.outputOffset], outputLocal, outputCopyParams);
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::DequantQcQrSplitN(const DequantQcQrSplitNParams& dequantQcQrSplitN,
                                                                     const LocalTensor<mmQcQrOutputType>& waveUb,
                                                                     const uint32_t wavePitch)
{
    uint32_t row = mmQcQrParam_.m;
    uint32_t colQc = baseParams_->dimHeadSizeQc;
    uint32_t colQcSingle = colQc / cvRatio_;
    uint32_t count = row * colQcSingle;

    DataCopyParams inputCopyParams{
        static_cast<uint16_t>(row), static_cast<uint16_t>(colQcSingle * sizeof(mmQcQrOutputType) / ALIGN_BLOCK_SIZE),
        static_cast<uint16_t>(dequantQcQrSplitN.srcStride * sizeof(mmQcQrOutputType) / ALIGN_BLOCK_SIZE), 0};

    DataCopyParams outputCopyParams{
        static_cast<uint16_t>(row), static_cast<uint16_t>(colQcSingle * sizeof(mmQnInputType) / ALIGN_BLOCK_SIZE), 0,
        static_cast<uint16_t>(dequantQcQrSplitN.dstStride * sizeof(mmQnInputType) / ALIGN_BLOCK_SIZE)};

    GlobalTensor<mmQcQrOutputType> inputGm = mmQcQrResGm_[dequantQcQrSplitN.mmQnPreDequantOffset];
    GlobalTensor<float> scale1Gm = deqScaleQcQrW_[dequantQcQrSplitN.mmQnPreDequantOffset];
    GlobalTensor<mmQnInputType> outputGm = mmQcQrResDequantGm_[dequantQcQrSplitN.mmQnPreDequantResOffset];

    LocalTensor<uint8_t> shareTmpUb = shareBuffer_.Get<uint8_t>()[PostCkvkrShareBase()];
    LocalTensor<float> scale2Local = dequantTool_.deQuantScaleCqLocal_;
    LocalTensor<mmQcQrOutputType> inputLocal = shareTmpUb.ReinterpretCast<mmQcQrOutputType>();
    LocalTensor<float> computeLocal = shareTmpUb.ReinterpretCast<float>();
    LocalTensor<float> scaleLocal = inputLocal[count + FP32_BLOCK_ELEMENT_NUM].template ReinterpretCast<float>();
    LocalTensor<mmQnInputType> outputLocal = inputLocal.template ReinterpretCast<mmQnInputType>();

    Rectangle dequantParams{row, colQcSingle, colQcSingle};
    if constexpr (MLA_CKVKR_BATCH2) {
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    } else {
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }

    if (wavePitch > 0) {
        if constexpr (MLA_PRELOAD_QCQR_SCALE && IsFullQuantMode<mmQcQrInputType, dequantScaleType, true>()) {
            Dequant(computeLocal, waveUb, qcqrScaleLocal_[dequantQcQrSplitN.inputOffset], scale2Local,
                    Rectangle{row, colQcSingle, wavePitch}, colQcSingle);
        } else {
            DataCopy(scaleLocal, scale1Gm[dequantQcQrSplitN.inputOffset], colQcSingle);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
            Dequant(computeLocal, waveUb, scaleLocal, scale2Local, Rectangle{row, colQcSingle, wavePitch}, colQcSingle);
        }
    } else {
        // ===== 原 GM 路径 =====
        DataCopy(inputLocal, inputGm[dequantQcQrSplitN.inputOffset], inputCopyParams);
        DataCopy(scaleLocal, scale1Gm[dequantQcQrSplitN.inputOffset], colQcSingle);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        Dequant(computeLocal, inputLocal, scaleLocal, scale2Local, dequantParams);
    }
    // cast
    Cast(outputLocal, computeLocal, RoundMode::CAST_RINT, count);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
    // copy out
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
    DataCopy(outputGm[dequantQcQrSplitN.outputOffset], outputLocal, outputCopyParams);
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::DequantAndRopeSplitNSyncMMQcQr(int64_t mmQnPreDequantOffset,
                                                                                  int64_t mmQnPreDequantResOffset,
                                                                                  int64_t ropeQrOffset,
                                                                                  int64_t ropeQrResOffset)
{
    if (cubeBlockIdx_ >= baseParams_->mm3BlockNum) {
        return;
    }
    uint32_t subBlockIdx_ = blockIdx_ % cvRatio_;
    uint32_t colCube = mmQcQrParam_.baseN;
    uint32_t colQc = baseParams_->dimHeadSizeQc;
    uint32_t colQr = baseParams_->dimHeadRope;
    uint32_t oriCol = mmQcQrParam_.n;
    uint32_t colOffsetCube = 0;

    // DequantSplitN参数
    uint32_t srcStride = baseParams_->headSizeQc + baseParams_->headSizeQr - baseParams_->dimHeadSizeQc / cvRatio_;
    uint32_t dstStride = baseParams_->headSizeQc - baseParams_->dimHeadSizeQc / cvRatio_;
    uint32_t colQcSingle = colQc / cvRatio_;
    uint32_t colOffsetVec = 0;
    uint32_t inputOffset = subBlockIdx_ * colQcSingle;
    uint32_t outputOffset = inputOffset;

    // RopeQrSplitN参数
    uint32_t ropeDstStride = baseParams_->headSizeQr - colQr;
    uint32_t ropeCntDown = mmQcQrParam_.m / 2;
    uint32_t ropeCnt = cvRatio_ == 1 ? ropeCntDown : (mmQcQrParam_.m + subBlockIdx_) / cvRatio_;
    int64_t ropeStride = static_cast<int64_t>(baseParams_->numHeadSize) * static_cast<int64_t>(colQc + colQr);
    uint32_t inputOffsetRope = ropeCntDown * subBlockIdx_ * ropeStride;
    uint32_t outputOffsetRope = ropeCntDown * subBlockIdx_ * baseParams_->headSizeQr;
    uint32_t deqScaleOffset = 0;
    uint32_t colOffsetRope = 0;
    uint32_t deQuantScaleCqOffset = ropeCntDown * subBlockIdx_ * FP32_BLOCK_ELEMENT_NUM;
    uint32_t dequantLoopCount = 0;
    uint32_t totalDequantLoops = CeilDiv(oriCol, (colQc + colQr));
    bool needSparseSync = totalDequantLoops > MAX_SYNC_FLAG_COUNT;
    bool ubPathAiv = false;
    uint32_t waveElems = 0;
    {
        const bool ubDirectSupportedA = (MLAPT::enableDequantOpt || nqWaveMode) && !MLAPT::enableGroupComputeOpt &&
                                        (!isFp8E8m0 || IsMxWaveCapable());
        const bool ubActiveA = ubDirectSupportedA && MLA_USE_UB_DIRECT;
        ubPathAiv = ubActiveA && (cvRatio_ == 1);
        waveElems = baseParams_->stepBatchSize * mmQcQrParam_.baseN;
    }
    while (colOffsetCube < oriCol) { // 循环CeilDiv(oriCol, colCube)次
        uint32_t prevColOffsetCube = colOffsetCube;
        colOffsetCube += colCube;
        if (colOffsetCube > oriCol) { // 当oriCol不被colCube整除时，mm最后一个base块需要刷新col end
            colOffsetCube = oriCol;
        }
        uint32_t curWaveWidth = colOffsetCube - prevColOffsetCube;
        LocalTensor<mmQcQrOutputType> waveUb;
        if constexpr (std::is_same<mmQcQrInputType, bfloat16_t>::value) {
            if (ubPathAiv && nqWaveTokenPending_) {
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
                nqWaveTokenPending_ = false;
            }
        }
        if constexpr (!std::is_same<mmQcQrInputType, bfloat16_t>::value) {
            if (ubPathAiv) {
                SetFlag<HardEvent::MTE3_V>(EVENT_ID6);
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID6);
            }
        }
        if (ubPathAiv) {
            CrossCoreWaitFlag<SYNC_MODE_CUBE_VEC, PIPE_V>(FINISH_MM_QCQR_SPLIT_N);
            waveUb = cvSharedUb_.Get<mmQcQrOutputType>()[(mm3WaveSeqAiv_ & 1u) * waveElems];
        } else {
            CrossCoreWaitFlag<SYNC_MODE_CUBE_VEC, PIPE_MTE2>(FINISH_MM_QCQR_SPLIT_N); // 原GM路径
        }
        if constexpr (std::is_same<mmQcQrInputType, bfloat16_t>::value) {
            CopyQcWaveSplitNLoop(mmQnPreDequantResOffset, colQc, colQr, colOffsetCube, dstStride, inputOffset,
                                 outputOffset, totalDequantLoops, needSparseSync, colOffsetVec, dequantLoopCount,
                                 waveUb, ubPathAiv ? curWaveWidth : 0);
        } else {
            DequantRopeSplitNDequantLoop(mmQnPreDequantOffset, mmQnPreDequantResOffset, colQc, colQr, colOffsetCube,
                                         srcStride, dstStride, inputOffset, outputOffset, totalDequantLoops,
                                         needSparseSync, colOffsetVec, dequantLoopCount, waveUb,
                                         ubPathAiv ? curWaveWidth : 0);
        }
        DequantRopeSplitNRopeLoop(ropeQrOffset, ropeQrResOffset, colQc, colQr, colOffsetCube, ropeDstStride, ropeCnt,
                                  ropeStride, inputOffsetRope, outputOffsetRope, deqScaleOffset, deQuantScaleCqOffset,
                                  colOffsetRope, waveUb, ubPathAiv ? curWaveWidth : 0);
        if (ubPathAiv) {
            if constexpr (std::is_same<mmQcQrInputType, bfloat16_t>::value) {
                SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
                nqWaveTokenPending_ = true;
            }
            CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_V>(FINISH_VEC_QCQR_CONSUMED);
            mm3WaveSeqAiv_++;
        }
    }
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::CopyQcWaveSplitNLoop(
    int64_t mmQnPreDequantResOffset, uint32_t colQc, uint32_t colQr, uint32_t colOffsetCube, uint32_t dstStride,
    uint32_t& inputOffset, uint32_t& outputOffset, uint32_t totalDequantLoops, bool needSparseSync,
    uint32_t& colOffsetVec, uint32_t& dequantLoopCount, const LocalTensor<mmQcQrOutputType>& waveUb,
    const uint32_t wavePitch)
{
    const uint32_t colQcSingle = colQc / cvRatio_;
    GlobalTensor<mmQnInputType> outputGm = mmQcQrResDequantGm_[mmQnPreDequantResOffset];
    while (colOffsetVec + colQc <= colOffsetCube) { // 循环singleNumHeadSize次(与 Dequant 版同构)
        LocalTensor<uint8_t> shareTmpUb = shareBuffer_.Get<uint8_t>()[PostCkvkrShareBase()];
        LocalTensor<mmQcQrOutputType> qcCompactLocal = shareTmpUb.ReinterpretCast<mmQcQrOutputType>();
        CopyStridedVf<mmQcQrOutputType>(qcCompactLocal, waveUb, mmQcQrParam_.m, colQcSingle, wavePitch, colQcSingle);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
        DataCopyParams outputCopyParams{static_cast<uint16_t>(mmQcQrParam_.m),
                                        static_cast<uint16_t>(colQcSingle * sizeof(mmQnInputType) / ALIGN_BLOCK_SIZE),
                                        0, static_cast<uint16_t>(dstStride * sizeof(mmQnInputType) / ALIGN_BLOCK_SIZE)};
        DataCopy(outputGm[outputOffset], qcCompactLocal, outputCopyParams);
        if (!needSparseSync || dequantLoopCount % 2 == 0 || dequantLoopCount == totalDequantLoops - 1) {
            CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_MTE3>(FINISH_VEC_DEQUANT_QC_SPLIT_N);
        }
        dequantLoopCount++;
        colOffsetVec += (colQc + colQr);
        inputOffset += (colQc + colQr);
        outputOffset += colQc;
    }
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::DequantRopeSplitNDequantLoop(
    int64_t mmQnPreDequantOffset, int64_t mmQnPreDequantResOffset, uint32_t colQc, uint32_t colQr,
    uint32_t colOffsetCube, uint32_t srcStride, uint32_t dstStride, uint32_t& inputOffset, uint32_t& outputOffset,
    uint32_t totalDequantLoops, bool needSparseSync, uint32_t& colOffsetVec, uint32_t& dequantLoopCount,
    const LocalTensor<mmQcQrOutputType>& waveUb, const uint32_t wavePitch)
{
    // DequantSplitN
    while (colOffsetVec + colQc <= colOffsetCube) { // 循环singleNumHeadSize次
        if constexpr (IsFullQuantMode<mmQcQrInputType, dequantScaleType, true>()) {
            DequantQcQrSplitN(DequantQcQrSplitNParams{mmQnPreDequantOffset, mmQnPreDequantResOffset, inputOffset,
                                                      outputOffset, srcStride, dstStride},
                              waveUb, wavePitch);
        } else if constexpr (std::is_same<mmQcQrInputType, FP8E4M3>::value && isFp8E8m0) {
            CastQcQrSplitN(CastQcQrSplitNParams{mmQnPreDequantOffset, mmQnPreDequantResOffset, inputOffset,
                                                outputOffset, srcStride, dstStride},
                           waveUb, wavePitch);
        }
        if (!needSparseSync || dequantLoopCount % 2 == 0 || dequantLoopCount == totalDequantLoops - 1) {
            CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_MTE3>(FINISH_VEC_DEQUANT_QC_SPLIT_N);
        }
        dequantLoopCount++;
        colOffsetVec += (colQc + colQr);
        inputOffset += (colQc + colQr);
        outputOffset += colQc;
    }
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::DequantRopeSplitNRopeLoop(
    int64_t ropeQrOffset, int64_t ropeQrResOffset, uint32_t colQc, uint32_t colQr, uint32_t colOffsetCube,
    uint32_t ropeDstStride, uint32_t ropeCnt, int64_t ropeStride, uint32_t& inputOffsetRope, uint32_t& outputOffsetRope,
    uint32_t& deqScaleOffset, uint32_t deQuantScaleCqOffset, uint32_t& colOffsetRope,
    const LocalTensor<mmQcQrOutputType>& waveUb, const uint32_t wavePitch)
{
    uint32_t ropeCntDown = mmQcQrParam_.m / 2;
    // RopeQrSplitN
    while ((colOffsetRope + colQc + colQr) <= colOffsetCube) {
        RopeQrSplitN(
            RopeQrSplitNParams{ropeQrOffset, ropeQrResOffset, inputOffsetRope, deqScaleOffset, outputOffsetRope,
                               ropeStride, ropeDstStride, deQuantScaleCqOffset, 0, ropeCnt, 0U},
            waveUb[colQc], wavePitch);
        if (cvRatio_ == 1) {
            RopeQrSplitN(
                RopeQrSplitNParams{ropeQrOffset, ropeQrResOffset,
                                   static_cast<uint32_t>(inputOffsetRope + ropeCntDown * ropeStride), deqScaleOffset,
                                   static_cast<uint32_t>(outputOffsetRope + ropeCntDown * baseParams_->headSizeQr),
                                   ropeStride, ropeDstStride, ropeCntDown * FP32_BLOCK_ELEMENT_NUM,
                                   ropeCntDown * baseParams_->dimHeadRope, (mmQcQrParam_.m + 1) / 2, 1U},
                waveUb[colQc + ropeCntDown * wavePitch], wavePitch);
        }
        colOffsetRope += (colQc + colQr);
        inputOffsetRope += (colQc + colQr);
        deqScaleOffset += (colQc + colQr);
        outputOffsetRope += colQr;
    }
}

template <typename MLAPT>
template <bool needQnDynamicQuant>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::DequantQcAndRopeQc(AivOffset& aivOffset, int64_t batchOffset,
                                                                      int64_t curStepBatchSize, int64_t numHeadOffset,
                                                                      int64_t mmQnLoops)
{
    // 根据不同分支条件处理
    if constexpr (MLAPT::enableGroupComputeOpt) {
        CrossCoreWaitFlag<SYNC_MODE_CUBE_VEC, PIPE_MTE2>(FINISH_MM_QC);
        DequantQcSplitNGroupCase(aivOffset.mmQnPreDequantOffset, aivOffset.mmQnPreDequantResOffset,
                                 aivOffset.qcScaleOffsetSplitN);
        CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_MTE3>(FINISH_VEC_DEQUANT_QC);
        CrossCoreWaitFlag<SYNC_MODE_CUBE_VEC, PIPE_MTE2>(FINISH_MM_QR);
        RopeQrSplitNGroupCase(aivOffset.ropeQrSplitNOffset, aivOffset.ropeQrResSplitNOffset);
    } else {
        if constexpr (MLAPT::enableDequantOpt || nqWaveMode) {
            DequantAndRopeSplitNSyncMMQcQr(aivOffset.mmQnPreDequantOffset, aivOffset.mmQnPreDequantResOffset,
                                           aivOffset.ropeQrOffset, aivOffset.ropeQrResOffset);
        } else if constexpr (IsFullQuantMode<mmQcQrInputType, dequantScaleType, true>()) {
            CrossCoreWaitFlag<SYNC_MODE_CUBE_VEC, PIPE_MTE2>(FINISH_MM_QCQR);
            WaitAllCore<SYNC_MODE_ALL_VEC, PIPE_MTE3>(FINISH_VEC_ALL);
            DequantQc(aivOffset.mmQnPreDequantOffset, aivOffset.mmQnPreDequantResOffset, aivOffset.curVecToken,
                      aivOffset.curBlockTokenOffset);
            WaitAllCore<SYNC_MODE_ALL_VEC, PIPE_MTE2>(FINISH_VEC_ALL);
            CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_MTE3>(FINISH_VEC_DEQUANT_QC);
            RopeQr(aivOffset.ropeQrOffset, aivOffset.ropeQrResOffset, aivOffset.curVecToken,
                   aivOffset.curBlockTokenOffset);
        } else if constexpr (std::is_same<mmQcQrInputType, FP8E4M3>::value && isFp8E8m0) {
            CrossCoreWaitFlag<SYNC_MODE_CUBE_VEC, PIPE_MTE2>(FINISH_MM_QCQR);
            WaitAllCore<SYNC_MODE_ALL_VEC, PIPE_MTE2>(FINISH_VEC_ALL);
            CastQc(aivOffset.mmQnPreDequantOffset, aivOffset.mmQnPreDequantResOffset, aivOffset.curVecToken);
            WaitAllCore<SYNC_MODE_ALL_VEC, PIPE_MTE3>(FINISH_VEC_ALL);
            CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_MTE3>(FINISH_VEC_DEQUANT_QC);
            RopeQr(aivOffset.ropeQrOffset, aivOffset.ropeQrResOffset, aivOffset.curVecToken,
                   aivOffset.curBlockTokenOffset);
        } else {
            CrossCoreWaitFlag<SYNC_MODE_CUBE_VEC, PIPE_MTE2>(FINISH_MM_QCQR);
            WaitAllCore<SYNC_MODE_ALL_VEC, PIPE_MTE2>(FINISH_VEC_ALL);
            RopeQr(aivOffset.ropeQrOffset, aivOffset.ropeQrResOffset, aivOffset.curVecToken,
                   aivOffset.curBlockTokenOffset);
        }

        if constexpr (needQnDynamicQuant && !(MLAPT::enableDequantOpt)) {
            // 非切N场景，需要等待全部rope的结果做完并搬到GM
            WaitAllCore<SYNC_MODE_ALL_VEC, PIPE_MTE3>(FINISH_VEC_ALL);
        }
        if constexpr (needQnDynamicQuant) {
            DynamicQuantQnAndMulQrSyncMMQn(batchOffset, curStepBatchSize, numHeadOffset, mmQnLoops);
        }
        aivOffset.ropeQrResOffset +=
            static_cast<int64_t>(baseParams_->stepBatchSize) * static_cast<int64_t>(baseParams_->headSizeQr);
    }
}

// 用于算力切分dequant切N场景
template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::DequantQcSplitNGroupCase(int64_t mmQnPreDequantOffset,
                                                                            int64_t mmQnPreDequantResOffset,
                                                                            int64_t qcQrScaleOffset)
{
    if (blockIdx_ >= QC_CORE_NUM * cvRatio_) {
        return;
    }
    uint32_t subBlockIdx_ = blockIdx_ % cvRatio_;
    uint32_t oriCol = (baseParams_->headSizeQc) / QC_CORE_NUM;
    uint32_t curCol = (baseParams_->headSizeQc) / (QC_CORE_NUM * cvRatio_);
    uint32_t srcStride = baseParams_->headSizeQc - curCol;
    uint32_t dstStride = baseParams_->headSizeQc - curCol;

    LocalTensor<uint8_t> shareTmpUb = shareBuffer_.Get<uint8_t>()[PostCkvkrShareBase()];
    Rectangle rectangleParams{
        static_cast<uint32_t>(mmQcQrParam_.m),         // row
        static_cast<uint32_t>(curCol),                 // col
        static_cast<uint32_t>(baseParams_->headSizeQc) // columnStride
    };
    DequantSplitNQc(mmQcQrResDequantGm_[mmQnPreDequantResOffset], mmQcQrResGm_[mmQnPreDequantOffset],
                    deqScaleQcQrW_[qcQrScaleOffset], dequantTool_.deQuantScaleCqLocal_, shareTmpUb, rectangleParams,
                    oriCol, dstStride, subBlockIdx_);
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::DequantQc(int64_t mmQnPreDequantOffset,
                                                             int64_t mmQnPreDequantResOffset, int64_t curVecToken,
                                                             int64_t curBlockTokenOffset)
{
    if (blockIdx_ >= curVectorBlockNum_) {
        return;
    }

    Rectangle rectangleParams{
        static_cast<uint32_t>(baseParams_->stepNumHeadDequant),                      // row
        static_cast<uint32_t>(baseParams_->dimHeadSizeQc),                           // col
        static_cast<uint32_t>(baseParams_->dimHeadRope + baseParams_->dimHeadSizeQc) // columnStride
    };

    for (int64_t curVecTokenIdx = 0; curVecTokenIdx < curVecToken; curVecTokenIdx++) {
        LocalTensor<uint8_t> shareTmpUb = shareBuffer_.Get<uint8_t>()[PostCkvkrShareBase()];
        DequantPerTokenQc(
            mmQcQrResDequantGm_[mmQnPreDequantResOffset], mmQcQrResGm_[mmQnPreDequantOffset], deqScaleQcQrW_,
            dequantTool_.deQuantScaleCqLocal_[(curBlockTokenOffset + curVecTokenIdx) * FP32_BLOCK_ELEMENT_NUM],
            shareTmpUb, rectangleParams, baseParams_->numHeadSize);
        mmQnPreDequantOffset += baseParams_->headSizeQc + baseParams_->headSizeQr;
        mmQnPreDequantResOffset += baseParams_->headSizeQc;
    }
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::CastQc(int64_t mmQnPreCastOffset, int64_t mmQnPreCastResOffset,
                                                          int64_t curVecToken)
{
    if (blockIdx_ >= curVectorBlockNum_) {
        return;
    }

    Rectangle rectangleCastParams{
        static_cast<uint32_t>(baseParams_->stepNumHeadDequant),                      // row
        static_cast<uint32_t>(baseParams_->dimHeadSizeQc),                           // col
        static_cast<uint32_t>(baseParams_->dimHeadRope + baseParams_->dimHeadSizeQc) // columnStride
    };

    for (int64_t curVecTokenIdx = 0; curVecTokenIdx < curVecToken; curVecTokenIdx++) {
        LocalTensor<uint8_t> shareTmpUb = shareBuffer_.Get<uint8_t>()[PostCkvkrShareBase()];
        CastPerTokenQc(mmQcQrResDequantGm_[mmQnPreCastResOffset], mmQcQrResGm_[mmQnPreCastOffset], shareTmpUb,
                       rectangleCastParams, baseParams_->numHeadSize);

        mmQnPreCastOffset += baseParams_->headSizeQc + baseParams_->headSizeQr;
        mmQnPreCastResOffset += baseParams_->headSizeQc;
    }
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::DynamicQuantQnAndMulQrSyncMMQn(int64_t batchOffset,
                                                                                  int64_t curStepBatchSize,
                                                                                  int64_t numHeadOffset,
                                                                                  int64_t mmQnLoops)
{
    int64_t curStepBatchSizeVec = (curStepBatchSize + (blockIdx_ % cvRatio_)) / cvRatio_;
    if (blockIdx_ >= baseParams_->mm4BlockNum * cvRatio_) {
        return;
    }
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);

    constexpr uint32_t DYN_QUANT_IN_READY = EVENT_ID0;
    constexpr uint32_t MUL_QR_COPY_READY = EVENT_ID3;
    constexpr uint32_t DYN_QUANT_OUT_READY = EVENT_ID3;

    int64_t batchOffInBlk = (blockIdx_ % cvRatio_) * (curStepBatchSize / cvRatio_);
    int64_t ckvTotalSize =
        static_cast<int64_t>(baseParams_->numHeadSize) * static_cast<int64_t>(baseParams_->headSizeCkv);
    int64_t dynamicQuantQueryOffset =
        batchOffInBlk * ckvTotalSize + numHeadOffset * static_cast<int64_t>(baseParams_->headSizeCkv);
    int64_t dynamicQuantQueryResOffset = batchOffset * ckvTotalSize + batchOffInBlk * ckvTotalSize +
                                         numHeadOffset * static_cast<int64_t>(baseParams_->headSizeCkv);
    int64_t scaleQueryNopeOffset = batchOffset * static_cast<int64_t>(baseParams_->numHeadSize) +
                                   batchOffInBlk * static_cast<int64_t>(baseParams_->numHeadSize) + numHeadOffset;
    int64_t queryOutStride = ckvTotalSize;
    int64_t qrOutputStride =
        static_cast<int64_t>(baseParams_->numHeadSize) * static_cast<int64_t>(baseParams_->dimHeadRope);
    int64_t qrPostProcessResOffset = batchOffset * static_cast<int64_t>(baseParams_->headSizeQr) +
                                     numHeadOffset * static_cast<int64_t>(baseParams_->dimHeadRope) +
                                     batchOffInBlk * static_cast<int64_t>(baseParams_->headSizeQr);

    LocalTensor<uint8_t> shareTmpUb = shareBuffer_.Get<uint8_t>()[PostCkvkrShareBase()];

    float quantScaleCkv = quantScaleCkvGm_.GetValue(0);
    const bool mm4UbSupportedA = MLAPT::enableDequantOpt && !MLAPT::enableGroupComputeOpt &&
                                 (!isFp8E8m0 || (MLA_MM4_UB_MXFP8_ENABLE && IsMxWaveCapable())) &&
                                 (!std::is_same<mmInputType, int8_t>::value || MLA_MM4_UB_INT8_ENABLE);
    constexpr bool useMm4UbDirectA = MLA_USE_MM4_UB;
    const bool mm4UbPathAiv =
        mm4UbSupportedA && useMm4UbDirectA && (cvRatio_ == 1) && (mmQnParam_.k <= mmQnParam_.baseK);
    if (mm4UbPathAiv) {
        if constexpr (MLA_MM4_MULQR_FUSED) {
            const uint32_t row = mmQnParam_.m;
            const uint32_t rows0 = (row + 1) / 2;
            const uint32_t rows1 = row - rows0;
            const uint32_t halfCnt = (rows1 > 0) ? 2 : 1;
            const uint32_t col = baseParams_->headSizeCkv;     // 512
            const uint32_t colRope = baseParams_->dimHeadRope; // 64
            const uint32_t stagingBytes = static_cast<uint32_t>(Align(
                static_cast<uint32_t>(rows0 * col * sizeof(queryOutputType)), static_cast<uint32_t>(ALIGN_BLOCK_SIZE)));
            uint32_t fusedNeed = stagingBytes;
            if constexpr (MLA_USE_STAGING2) {
                fusedNeed += stagingBytes; // staging1 真实分配(不再别名 MulQr 死区)
            }
            fusedNeed += static_cast<uint32_t>(Align(static_cast<uint32_t>(rows0 * colRope * sizeof(ropeOutputType)),
                                                     static_cast<uint32_t>(ALIGN_BLOCK_SIZE))); // qrInputSeg
            fusedNeed += static_cast<uint32_t>(Align(static_cast<uint32_t>(rows0 * colRope * sizeof(float)),
                                                     static_cast<uint32_t>(ALIGN_BLOCK_SIZE))); // qrFp32Seg
            fusedNeed += static_cast<uint32_t>(Align(static_cast<uint32_t>(mmQnLoops * row * sizeof(float)),
                                                     static_cast<uint32_t>(ALIGN_BLOCK_SIZE))); // scaleAcc
            fusedNeed += CeilDivT(rows0, FP32_BLOCK_ELEMENT_NUM) * FP32_BLOCK_ELEMENT_NUM * FP32_BLOCK_ELEMENT_NUM *
                         sizeof(float); // scaleBrcbSeg(Brcb: CeilDiv(rows0,8)×64 float)
            const bool mulqrFusedFits = fusedNeed <= (ubShareBytes_ - PostCkvkrShareBase());
            if (mulqrFusedFits) {
                const uint32_t cvPingElems =
                    baseParams_->stepBatchSize * mmQcQrParam_.baseN * sizeof(mmQcQrOutputType) / sizeof(mmQnOutputType);
                constexpr uint32_t MUL_QR = EVENT_ID1;
                constexpr uint32_t MUL_QR_INPUT_READY = EVENT_ID3;
                constexpr uint32_t SCALE_OUT_READY = EVENT_ID0;
                constexpr uint32_t DYNAMIC_QUANT_OUTPUT_READY = EVENT_ID3;
                constexpr uint32_t STAGING_READY1 = EVENT_ID5;
                constexpr uint64_t ropeBlockAlign = ALIGN_BLOCK_SIZE / sizeof(ropeOutputType); // bf16: 16
                constexpr uint64_t computeBlockAlign = ALIGN_BLOCK_SIZE / sizeof(float);       // fp32: 8
                uint32_t ubOff = 0;
                LocalTensor<queryOutputType> staging = shareTmpUb.ReinterpretCast<queryOutputType>();
                ubOff += stagingBytes;
                LocalTensor<queryOutputType> staging1;
                if constexpr (MLA_USE_STAGING2) {
                    staging1 = shareTmpUb[ubOff].ReinterpretCast<queryOutputType>();
                    ubOff += stagingBytes;
                }
                LocalTensor<ropeOutputType> qrInputSeg = shareTmpUb[ubOff].template ReinterpretCast<ropeOutputType>();
                ubOff += static_cast<uint32_t>(Align(static_cast<uint32_t>(rows0 * colRope * sizeof(ropeOutputType)),
                                                     static_cast<uint32_t>(ALIGN_BLOCK_SIZE)));
                LocalTensor<float> qrFp32Seg = shareTmpUb[ubOff].ReinterpretCast<float>();
                ubOff += static_cast<uint32_t>(Align(static_cast<uint32_t>(rows0 * colRope * sizeof(float)),
                                                     static_cast<uint32_t>(ALIGN_BLOCK_SIZE)));
                LocalTensor<ropeOutputType> outputLocalRope = qrFp32Seg.ReinterpretCast<ropeOutputType>();
                LocalTensor<float> scaleAcc = shareTmpUb[ubOff].ReinterpretCast<float>();
                ubOff += static_cast<uint32_t>(Align(static_cast<uint32_t>(mmQnLoops * row * sizeof(float)),
                                                     static_cast<uint32_t>(ALIGN_BLOCK_SIZE)));
                LocalTensor<float> scaleBrcbSeg = shareTmpUb[ubOff].ReinterpretCast<float>();

                SetFlag<HardEvent::MTE3_V>(DYNAMIC_QUANT_OUTPUT_READY);
                if constexpr (MLA_USE_STAGING2) {
                    SetFlag<HardEvent::MTE3_V>(STAGING_READY1);
                }
                SetFlag<HardEvent::V_MTE2>(MUL_QR_INPUT_READY);
                SetFlag<HardEvent::MTE3_V>(MUL_QR);
                SetFlag<HardEvent::MTE3_V>(SCALE_OUT_READY);

                uint32_t halfBlockIdx = 0;
                for (int64_t loopIdx = 0; loopIdx < mmQnLoops; loopIdx++) {
                    for (uint32_t h = 0; h < halfCnt; h++) {
                        const uint32_t rows = (h == 0) ? rows0 : rows1;
                        CrossCoreWaitFlag<SYNC_MODE_CUBE_VEC, PIPE_V>(FINISH_MM_QN_SPLIT_N);
                        LocalTensor<mmQnOutputType> waveHalf =
                            cvSharedUb_.Get<mmQnOutputType>()[(mm3WaveSeqAiv_ & 1u) * cvPingElems];
                        LocalTensor<queryOutputType> curStaging = staging;
                        uint32_t curReady = DYNAMIC_QUANT_OUTPUT_READY;
                        if constexpr (MLA_USE_STAGING2) {
                            const uint32_t curPing = halfBlockIdx & 1u;
                            curStaging = (curPing == 0u) ? staging : staging1;
                            curReady =
                                (curPing == 0u) ? static_cast<uint32_t>(DYNAMIC_QUANT_OUTPUT_READY) : STAGING_READY1;
                        }
                        LocalTensor<float> scaleAccSeg =
                            scaleAcc[loopIdx * static_cast<int64_t>(row) + static_cast<int64_t>(h) * rows0];
                        WaitFlag<HardEvent::MTE3_V>(curReady);
                        if constexpr (MLA_USE_DQ_V2) {
                            DynamicQuantPerTokenVf2(curStaging, scaleAccSeg, waveHalf, rows, col);
                        } else {
                            DynamicQuantPerTokenVf(curStaging, scaleAccSeg, waveHalf, rows, col);
                        }
                        SetFlag<HardEvent::V_MTE3>(curReady);
                        WaitFlag<HardEvent::V_MTE3>(curReady);
                        DataCopyParams outParams{
                            static_cast<uint16_t>(rows),
                            static_cast<uint16_t>(col * sizeof(queryOutputType) / ALIGN_BLOCK_SIZE), 0,
                            static_cast<uint16_t>((queryOutStride - col) * sizeof(queryOutputType) / ALIGN_BLOCK_SIZE)};
                        DataCopy(queryOutGm_[dynamicQuantQueryResOffset +
                                             static_cast<int64_t>(h) * static_cast<int64_t>(rows0) * queryOutStride],
                                 curStaging, outParams);
                        SetFlag<HardEvent::MTE3_V>(curReady);
                        halfBlockIdx++;
                        CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_V>(FINISH_VEC_QCQR_CONSUMED);
                        mm3WaveSeqAiv_++;
                        WaitFlag<HardEvent::MTE3_V>(SCALE_OUT_READY);
                        Brcb(scaleBrcbSeg, scaleAccSeg, CeilDivT(rows, FP32_BLOCK_ELEMENT_NUM),
                             {1, FP32_BLOCK_ELEMENT_NUM});
                        DataCopyParams segScaleParams{
                            static_cast<uint16_t>(rows), static_cast<uint16_t>(sizeof(float)), 0,
                            static_cast<uint16_t>((baseParams_->numHeadSize - 1) * sizeof(float))};
                        SetFlag<HardEvent::V_MTE3>(SCALE_OUT_READY);
                        WaitFlag<HardEvent::V_MTE3>(SCALE_OUT_READY); // MTE3: Brcb 已 retire 可读
                        DataCopyPad(dequantScaleQNopeGm_[scaleQueryNopeOffset +
                                                         static_cast<int64_t>(h) * static_cast<int64_t>(rows0) *
                                                             static_cast<int64_t>(baseParams_->numHeadSize)],
                                    scaleBrcbSeg, segScaleParams);
                        SetFlag<HardEvent::MTE3_V>(SCALE_OUT_READY); // MTE3: retire 后下段 Brcb 才可覆写

                        WaitFlag<HardEvent::V_MTE2>(MUL_QR_INPUT_READY); // MTE2: 上段 Cast 读 qrInputSeg 已 retire
                        DataCopyParams segInParams{
                            static_cast<uint16_t>(rows), static_cast<uint16_t>(colRope / ropeBlockAlign),
                            static_cast<uint16_t>((qrOutputStride - colRope) / ropeBlockAlign), 0};
                        DataCopy(qrInputSeg,
                                 qrOutGm_[qrPostProcessResOffset +
                                          static_cast<int64_t>(h) * static_cast<int64_t>(rows0) * qrOutputStride],
                                 segInParams);
                        SetFlag<HardEvent::MTE2_V>(MUL_QR_INPUT_READY); // MTE2: 装载 retire 后 V 可读

                        WaitFlag<HardEvent::MTE3_V>(MUL_QR);             // V: 上段 qrOut 搬出已 retire
                        WaitFlag<HardEvent::MTE2_V>(MUL_QR_INPUT_READY); // V: 本段装载已完成
                        Cast(qrFp32Seg, qrInputSeg, RoundMode::CAST_NONE, rows * colRope);
                        SetFlag<HardEvent::V_MTE2>(MUL_QR_INPUT_READY); // V: Cast 已读走 qrInputSeg
                        MulQrVF(qrFp32Seg, qrFp32Seg, scaleBrcbSeg, quantScaleCkv, rows * colRope, computeBlockAlign);
                        Cast(outputLocalRope, qrFp32Seg, RoundMode::CAST_RINT, rows * colRope);
                        SetFlag<HardEvent::V_MTE3>(MUL_QR); // V: Cast 已 retire
                        WaitFlag<HardEvent::V_MTE3>(MUL_QR);
                        DataCopyParams segOutParams{static_cast<uint16_t>(rows),
                                                    static_cast<uint16_t>(colRope / ropeBlockAlign), 0,
                                                    static_cast<uint16_t>((qrOutputStride - colRope) / ropeBlockAlign)};
                        DataCopy(qrOutGm_[qrPostProcessResOffset +
                                          static_cast<int64_t>(h) * static_cast<int64_t>(rows0) * qrOutputStride],
                                 outputLocalRope, segOutParams);
                        SetFlag<HardEvent::MTE3_V>(MUL_QR); // MTE3: retire 后下段 Cast 可覆写
                    }
                    dynamicQuantQueryResOffset += static_cast<int64_t>(baseParams_->headSizeCkv);
                    scaleQueryNopeOffset += 1;
                    qrPostProcessResOffset += static_cast<int64_t>(baseParams_->dimHeadRope);
                }
                WaitFlag<HardEvent::V_MTE2>(MUL_QR_INPUT_READY);
                WaitFlag<HardEvent::MTE3_V>(MUL_QR);
                WaitFlag<HardEvent::MTE3_V>(SCALE_OUT_READY);
                WaitFlag<HardEvent::MTE3_V>(DYNAMIC_QUANT_OUTPUT_READY);
                if constexpr (MLA_USE_STAGING2) {
                    WaitFlag<HardEvent::MTE3_V>(STAGING_READY1);
                }
                return;
            }
        }

        const uint32_t row = mmQnParam_.m;
        const uint32_t rows0 = (row + 1) / 2;
        const uint32_t rows1 = row - rows0;
        const uint32_t halfCnt = (rows1 > 0) ? 2 : 1;
        const uint32_t col = baseParams_->headSizeCkv;     // 512
        const uint32_t colRope = baseParams_->dimHeadRope; // 64
        const uint32_t subRowRope = (row < 64) ? row : 64;
        const uint32_t cvPingElems =
            baseParams_->stepBatchSize * mmQcQrParam_.baseN * sizeof(mmQcQrOutputType) / sizeof(mmQnOutputType);
        constexpr uint32_t MUL_QR = EVENT_ID1;
        constexpr uint32_t MUL_QR_INPUT_COPY_READY = EVENT_ID3;
        constexpr uint32_t DYNAMIC_QUANT_OUTPUT_READY = EVENT_ID3;
        constexpr uint32_t STAGING_READY1 = EVENT_ID5;
        uint32_t ubOff = 0;
        const uint32_t stagingSize = static_cast<uint32_t>(Align(
            static_cast<uint32_t>(rows0 * col * sizeof(queryOutputType)), static_cast<uint32_t>(ALIGN_BLOCK_SIZE)));
        LocalTensor<queryOutputType> staging = shareTmpUb.ReinterpretCast<queryOutputType>();
        ubOff += stagingSize;
        LocalTensor<queryOutputType> staging1;
        if constexpr (MLA_USE_STAGING2) {
            staging1 = shareTmpUb[ubOff].ReinterpretCast<queryOutputType>();
        }
        LocalTensor<ropeOutputType> qrInputLocal = shareTmpUb[ubOff].template ReinterpretCast<ropeOutputType>();
        ubOff += static_cast<uint32_t>(Align(static_cast<uint32_t>(row * colRope * sizeof(ropeOutputType)),
                                             static_cast<uint32_t>(ALIGN_BLOCK_SIZE)));
        LocalTensor<ropeOutputType> outputLocalRope = shareTmpUb[ubOff].template ReinterpretCast<ropeOutputType>();
        ubOff += static_cast<uint32_t>(Align(static_cast<uint32_t>(subRowRope * colRope * sizeof(ropeOutputType)),
                                             static_cast<uint32_t>(ALIGN_BLOCK_SIZE)));
        LocalTensor<float> qrFp32Local = shareTmpUb[ubOff].ReinterpretCast<float>();
        ubOff += static_cast<uint32_t>(Align(static_cast<uint32_t>(subRowRope * colRope * sizeof(float)),
                                             static_cast<uint32_t>(ALIGN_BLOCK_SIZE)));
        LocalTensor<float> reciprocalLocal = qrFp32Local;
        LocalTensor<float> scaleAcc = shareTmpUb[ubOff].ReinterpretCast<float>();
        ubOff += static_cast<uint32_t>(
            Align(static_cast<uint32_t>(mmQnLoops * row * sizeof(float)), static_cast<uint32_t>(ALIGN_BLOCK_SIZE)));
        LocalTensor<float> scaleBrcb = shareTmpUb[ubOff].ReinterpretCast<float>();

        // 令牌预置
        SetFlag<HardEvent::MTE3_V>(DYNAMIC_QUANT_OUTPUT_READY);
        if constexpr (MLA_USE_STAGING2) {
            SetFlag<HardEvent::MTE3_V>(STAGING_READY1); // staging1 预置(乒乓)
        }
        SetFlag<HardEvent::V_MTE2>(MUL_QR_INPUT_COPY_READY);
        uint32_t halfBlockIdx = 0;
        for (int64_t loopIdx = 0; loopIdx < mmQnLoops; loopIdx++) {
            for (uint32_t h = 0; h < halfCnt; h++) {
                const uint32_t rows = (h == 0) ? rows0 : rows1;
                CrossCoreWaitFlag<SYNC_MODE_CUBE_VEC, PIPE_V>(FINISH_MM_QN_SPLIT_N);
                LocalTensor<mmQnOutputType> waveHalf =
                    cvSharedUb_.Get<mmQnOutputType>()[(mm3WaveSeqAiv_ & 1u) * cvPingElems];
                LocalTensor<queryOutputType> curStaging = staging;
                uint32_t curReady = DYNAMIC_QUANT_OUTPUT_READY;
                if constexpr (MLA_USE_STAGING2) {
                    const uint32_t curPing = halfBlockIdx & 1u;
                    curStaging = (curPing == 0u) ? staging : staging1;
                    curReady = (curPing == 0u) ? static_cast<uint32_t>(DYNAMIC_QUANT_OUTPUT_READY) : STAGING_READY1;
                }
                WaitFlag<HardEvent::MTE3_V>(curReady);
                if constexpr (MLA_USE_DQ_V2) {
                    DynamicQuantPerTokenVf2(
                        curStaging, scaleAcc[loopIdx * static_cast<int64_t>(row) + static_cast<int64_t>(h) * rows0],
                        waveHalf, rows, col);
                } else {
                    DynamicQuantPerTokenVf(
                        curStaging, scaleAcc[loopIdx * static_cast<int64_t>(row) + static_cast<int64_t>(h) * rows0],
                        waveHalf, rows, col);
                }
                SetFlag<HardEvent::V_MTE3>(curReady);
                WaitFlag<HardEvent::V_MTE3>(curReady);
                DataCopyParams outParams{
                    static_cast<uint16_t>(rows),
                    static_cast<uint16_t>(col * sizeof(queryOutputType) / ALIGN_BLOCK_SIZE), 0,
                    static_cast<uint16_t>((queryOutStride - col) * sizeof(queryOutputType) / ALIGN_BLOCK_SIZE)};
                DataCopy(queryOutGm_[dynamicQuantQueryResOffset + static_cast<int64_t>(h) * rows0 * queryOutStride],
                         curStaging, outParams);
                SetFlag<HardEvent::MTE3_V>(curReady);
                halfBlockIdx++;
                CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_V>(FINISH_VEC_QCQR_CONSUMED);
                mm3WaveSeqAiv_++;
            }
            dynamicQuantQueryResOffset += static_cast<int64_t>(baseParams_->headSizeCkv);
        }
        for (int64_t loopIdx = 0; loopIdx < mmQnLoops; loopIdx++) {
            Brcb(scaleBrcb, scaleAcc[loopIdx * static_cast<int64_t>(row)], CeilDivT(row, FP32_BLOCK_ELEMENT_NUM),
                 {1, FP32_BLOCK_ELEMENT_NUM});
            DataCopyParams scaleOutCopyParams{static_cast<uint16_t>(row), static_cast<uint16_t>(sizeof(float)), 0,
                                              static_cast<uint16_t>((baseParams_->numHeadSize - 1) * sizeof(float))};
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopyPad(dequantScaleQNopeGm_[scaleQueryNopeOffset], scaleBrcb, scaleOutCopyParams);
            if constexpr (MLA_USE_STAGING2) {
                WaitFlag<HardEvent::MTE3_V>(STAGING_READY1);
                SetFlag<HardEvent::MTE3_MTE2>(STAGING_READY1);
                WaitFlag<HardEvent::MTE3_MTE2>(STAGING_READY1);
            }
            MulQr<ropeOutputType, float>(qrOutGm_[qrPostProcessResOffset], qrOutGm_[qrPostProcessResOffset],
                                         outputLocalRope, qrInputLocal, qrFp32Local, reciprocalLocal, scaleBrcb, row,
                                         colRope, subRowRope, static_cast<uint64_t>(qrOutputStride), quantScaleCkv,
                                         MUL_QR_INPUT_COPY_READY, MUL_QR, cvRatio_);
            if constexpr (MLA_USE_STAGING2) {
                SetFlag<HardEvent::MTE3_V>(STAGING_READY1);
            }
            scaleQueryNopeOffset += 1;
            qrPostProcessResOffset += static_cast<int64_t>(baseParams_->dimHeadRope);
        }
        WaitFlag<HardEvent::V_MTE2>(MUL_QR_INPUT_COPY_READY);
        WaitFlag<HardEvent::MTE3_V>(DYNAMIC_QUANT_OUTPUT_READY);
        if constexpr (MLA_USE_STAGING2) {
            WaitFlag<HardEvent::MTE3_V>(STAGING_READY1);
        }
        return;
    }

    // Dynamic Quant
    SetFlag<HardEvent::MTE3_V>(DYN_QUANT_OUT_READY);
    SetFlag<HardEvent::V_MTE2>(DYN_QUANT_IN_READY);

    // Rope Post Process
    SetFlag<HardEvent::V_MTE2>(MUL_QR_COPY_READY);
    // per-head循环
    for (int64_t loopIdx = 0; loopIdx < mmQnLoops; loopIdx++) {
        CrossCoreWaitFlag<SYNC_MODE_CUBE_VEC, PIPE_MTE2>(FINISH_MM_QN_SPLIT_N);
        DynamicQuantQnWithMulQr<ropeOutputType, dequantScaleQNopeType, queryOutputType>(
            dequantScaleQNopeGm_[scaleQueryNopeOffset], queryOutGm_[dynamicQuantQueryResOffset],
            qrOutGm_[qrPostProcessResOffset], mmQnResGm_[dynamicQuantQueryOffset], shareTmpUb, curStepBatchSizeVec,
            baseParams_->headSizeCkv, baseParams_->numHeadSize, queryOutStride,
            // Rope Post Process
            qrOutGm_[qrPostProcessResOffset], quantScaleCkv, baseParams_->dimHeadRope, qrOutputStride, cvRatio_);

        dynamicQuantQueryOffset += static_cast<int64_t>(baseParams_->headSizeCkv);
        scaleQueryNopeOffset += 1;
        dynamicQuantQueryResOffset += static_cast<int64_t>(baseParams_->headSizeCkv);
        qrPostProcessResOffset += static_cast<int64_t>(baseParams_->dimHeadRope);
    }
    // Rope Post Process
    WaitFlag<HardEvent::V_MTE2>(MUL_QR_COPY_READY);
    // Dynamic Quant
    WaitFlag<HardEvent::V_MTE2>(DYN_QUANT_IN_READY);
    WaitFlag<HardEvent::MTE3_V>(DYN_QUANT_OUT_READY);
}

} // namespace MlaProlog

#endif // MLA_PROLOG_VEC_S1_CUB_S2_H
