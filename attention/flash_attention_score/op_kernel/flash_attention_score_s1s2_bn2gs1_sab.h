/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flash_attention_score_s1s2_bn2gs1_sab.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_S1S2_BN2GS1_SAB_H
#define FLASH_ATTENTION_SCORE_S1S2_BN2GS1_SAB_H

#include "util.h"
#include "dropmask.h"
#include "flash_attention_score_common.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "pse.h"

using matmul::MatmulType;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

namespace FaTscm {
// 定义TSCM buffer全局管理结构体
constexpr int64_t TSCM_BUF_NUM = 2;

struct FaTscmArray {
  __aicore__ inline FaTscmArray() {};
  TBuf<TPosition::TSCM> scm;
  AscendC::TBuffAddr srcAddr;
  int32_t cacheSize{0}; // 表示已缓存base块的个数
  int32_t bufferSize{0}; // base块元素个数
  bool needAlloc{true};
};

__BLOCK_LOCAL__ __inline__ FaTscmArray *TscmGlobal;

__aicore__ inline void InitTscmBuffer(TPipe *tPipe, FaTscmArray *TscmGlobal)
{
  if ASCEND_IS_AIC {
    tPipe->InitBuffer(TscmGlobal[Q_VEC1_INDEX].scm, Q_VEC1_BUFFER_SIZE);
    tPipe->InitBuffer(TscmGlobal[K_V_INDEX].scm, K_V_BUFFER_SIZE);
  }
}
} // namespace FaTscm

using namespace FaTscm;

struct SplitSameABExtraInfo {
    int64_t s2StartIdx;
    int64_t s2EndIdx;
    int64_t s2LoopCount;
    int64_t s1oIdx;
    int64_t boIdx;
    int64_t n2oIdx;
    int64_t goIdx;
    int64_t taskId;
    int8_t taskIdMod2;
    int8_t multiCoreInnerIdxMod2;
    int8_t needNz2Nd;
    int32_t s1RealSize;
    int32_t cubeS1RealSize;
    int32_t s2RealSize;
    int32_t s2AlignedSize;

    int32_t s2RealSizeAlign64 = 0;
    uint64_t duplicateMask = 0;
    int32_t s2RealSizeFloorAlign8 = 0;
    bool s2LastLoop = false;

    int32_t vec1S1BaseSize;
    int32_t vec1S1RealSize;
    int32_t vec2S1BaseSize;
    int32_t vec2S1RealSize;
    int32_t realSplitN;
    int32_t s2LoopLimit;
    int64_t multiCoreInnerIdx;
    int64_t qCoreOffset;
    int64_t kCoreOffset;
    int64_t qRopeCoreOffset = 0;
    int64_t pseS2ComputeSize;
    int64_t s1SizeDelta;
    int64_t s1SizeAcc;
    int64_t s2SizeAcc;
    int64_t attenB1SSOffset;
    int64_t attenMaskS2Size;
    int64_t s1Size;
    int64_t s2Size;
    int64_t softmaxMaxOffset;
    int64_t vecCoreOffset;
};

enum class MmPolicyType {
    NORMAL = 0,
    UNSPLITK = 1
};

constexpr int64_t VEC_BUFFER_SIZE = 1024;

// INPUT_T - means data type for input
// T       - means data type when calc
template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T = INPUT_T, CubeFormat bmm1Format = CubeFormat::ND,
          MmPolicyType mmPolicyType = MmPolicyType::NORMAL, bool hasRope = false>
class FlashAttentionScoreS1s2Bn2gs1SameAB {
public:
    __aicore__ inline FlashAttentionScoreS1s2Bn2gs1SameAB(){};

    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t* queryRope, __gm__ uint8_t *key, __gm__ uint8_t *keyRope,
                                __gm__ uint8_t *value, __gm__ uint8_t *pse,
                                __gm__ uint8_t *dropMask, __gm__ uint8_t *paddingMask, __gm__ uint8_t *prefix,
                                __gm__ uint8_t *attenMask, __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
                                __gm__ uint8_t *softmaxOut, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
                                const FlashAttentionScoreGeneralTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void Process();

    static constexpr MatmulConfig CFG_CUSTOM_EXCEED = (mmPolicyType == MmPolicyType::UNSPLITK) ? CFG_DIS_UNIT_FLAG_EXCEED : CFG_EXCEED;

    // define matmul
    using a1Type = typename AscendC::Conditional<
        (mmPolicyType == MmPolicyType::UNSPLITK),
        MatmulType<TPosition::TSCM, CubeFormat::ND, INPUT_T, false, LayoutMode::NONE, true>,
        MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, false, LayoutMode::NONE, true>>::type;
    using b1Type = typename AscendC::Conditional<
        (mmPolicyType == MmPolicyType::UNSPLITK),
        MatmulType<TPosition::TSCM, CubeFormat::ND, INPUT_T, true, LayoutMode::NONE, true>,
        MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, true, LayoutMode::NONE, true>>::type;
    using bias1Type = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using c1Type = MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using mm1Type = matmul::MatmulImpl<a1Type, b1Type, c1Type, bias1Type, CFG_CUSTOM_EXCEED>;
    mm1Type bmm1;

    using c1NzType = MatmulType<TPosition::GM, CubeFormat::NZ, T>;
    using mm1NzType = matmul::MatmulImpl<a1Type, b1Type, c1NzType, bias1Type, CFG_CUSTOM_EXCEED>;
    mm1NzType bmm1Nz;

    // define batchmatmul
    using a2Type = typename AscendC::Conditional<
        (mmPolicyType == MmPolicyType::UNSPLITK),
        MatmulType<TPosition::TSCM, CubeFormat::NZ, INPUT_T, false, LayoutMode::NONE, true>,
        MatmulType<TPosition::GM, CubeFormat::NZ, INPUT_T, false, LayoutMode::NONE, true>>::type;
    using b2Type = typename AscendC::Conditional<
        (mmPolicyType == MmPolicyType::UNSPLITK),
        MatmulType<TPosition::TSCM, CubeFormat::ND, INPUT_T, false, LayoutMode::NONE, true>,
        MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, false, LayoutMode::NONE, true>>::type;
    using bias2Type = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using c2NzType = MatmulType<TPosition::GM, CubeFormat::NZ, T>;
    using mm2Type = matmul::MatmulImpl<a2Type, b2Type, c2NzType, bias2Type, CFG_CUSTOM_EXCEED>;
    mm2Type bmm2;

protected:
    __aicore__ inline void InitInput(__gm__ uint8_t *query, __gm__ uint8_t* queryRope, __gm__ uint8_t *key, __gm__ uint8_t *keyRope,
                                     __gm__ uint8_t *value,
                                     __gm__ uint8_t *pse, __gm__ uint8_t *dropMask, __gm__ uint8_t *paddingMask,
                                     __gm__ uint8_t *prefix, __gm__ uint8_t *attenMask, __gm__ uint8_t *softmaxMax,
                                     __gm__ uint8_t *softmaxSum, __gm__ uint8_t *softmaxOut,
                                     __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
                                     const FlashAttentionScoreGeneralTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void IterateBmm2(SplitSameABExtraInfo &extraInfo,
                                       matmul::MatmulImpl<a2Type, b2Type, c2NzType, bias2Type, CFG_CUSTOM_EXCEED> &bmm2);
    __aicore__ inline void NdToNz(SplitSameABExtraInfo &extraInfo, LocalTensor<INPUT_T> &nzResUb,
                                  LocalTensor<INPUT_T> &vec1ResUb, int64_t loopIdx);
    __aicore__ inline void SetExtraInfo(SplitSameABExtraInfo &extraInfo, int64_t taskId, int64_t s2LoopCount,
                                        int64_t s2LoopLimit, int64_t multiCoreInnerIdx);
    __aicore__ inline void SetTiling(const FlashAttentionScoreGeneralTilingData *__restrict tilingData);
    __aicore__ inline void InitBuffer();
    __aicore__ inline void ComputeConstexpr();
    __aicore__ inline void ComputeAxisIdx(int64_t multiCoreInnerIdx);
    template <typename T2>
    __aicore__ inline void IterateBmm1(SplitSameABExtraInfo &extraInfo,
                                       matmul::MatmulImpl<a1Type, b1Type, T2, bias1Type, CFG_CUSTOM_EXCEED> &bmm1);
    template <typename T2>
    __aicore__ inline void Bmm1SetTensorA(SplitSameABExtraInfo &extraInfo,
                                          matmul::MatmulImpl<a1Type, b1Type, T2, bias1Type, CFG_CUSTOM_EXCEED> &bmm1);
    template <typename T2>
    __aicore__ inline void Bmm1SetTensorARope(SplitSameABExtraInfo &extraInfo,
                                          matmul::MatmulImpl<a1Type, b1Type, T2, bias1Type, CFG_CUSTOM_EXCEED> &bmm1);
    template <typename T2>
    __aicore__ inline void SetBmm1TensorB(SplitSameABExtraInfo &extraInfo,
                                          matmul::MatmulImpl<a1Type, b1Type, T2, bias1Type, CFG_CUSTOM_EXCEED> &bmm1);
    template <typename T2>
    __aicore__ inline void SetBmm1TensorBRope(SplitSameABExtraInfo &extraInfo,
                                          matmul::MatmulImpl<a1Type, b1Type, T2, bias1Type, CFG_CUSTOM_EXCEED> &bmm1);
    __aicore__ inline void ComputeBmm1Tail(SplitSameABExtraInfo &extraInfo);
    __aicore__ inline void ProcessVec1(SplitSameABExtraInfo &extraInfo);
    __aicore__ inline void CopyInAttenMask(SplitSameABExtraInfo &extraInfo, int64_t loopIdx, int64_t maskOffset,
                                           bool secondTime = false);
    __aicore__ inline void GetAttenMaskComputeMode(int64_t deltaCausalOrNext, int64_t deltaPre, int64_t s1Offset,
                                                   SplitSameABExtraInfo &extraInfo);
    __aicore__ inline int64_t ComputeAttenMaskOffset(SplitSameABExtraInfo &extraInfo, int64_t loopIdx);
    __aicore__ inline int64_t ComputeOffsetForNoCompress(SplitSameABExtraInfo &extraInfo, int64_t loopIdx);
    __aicore__ inline void GetBmm1Result(SplitSameABExtraInfo &extraInfo, LocalTensor<T> &bmm1ResUb, int64_t loopIdx);
    __aicore__ inline void ComputeAttenMask(SelectWithBytesMaskShapeInfo &shapeInfo, LocalTensor<T> &bmm1ResUb,
                                            LocalTensor<uint8_t> &maskUb, const uint8_t maskType, event_t vWaitMte2);

    __aicore__ inline void SoftMaxCompute(SplitSameABExtraInfo &extraInfo, LocalTensor<T> &srcTensor, int64_t loopIdx);
    __aicore__ inline void SoftMaxCheckResCompress(SplitSameABExtraInfo &extraInfo, int64_t vec1S1realSplitN);
    __aicore__ inline void InvalidLineSplitS2Process(SplitSameABExtraInfo &extraInfo, LocalTensor<T> &srcTensor,
                                                     LocalTensor<T> &maxUb, int64_t loopIdx);
    __aicore__ inline void ProcessVec2(SplitSameABExtraInfo &extraInfo);
    __aicore__ inline void Bmm2ResultMul(SplitSameABExtraInfo &extraInfo, LocalTensor<T> &bmm2ResUb, int64_t s1oIdx);
    __aicore__ inline void Bmm2ResultDiv(SplitSameABExtraInfo &extraInfo, int64_t s1oIdx);
    __aicore__ inline void Bmm2DataCopyOut(SplitSameABExtraInfo &extraInfo, int64_t s1oIdx, int64_t mm2ResCalcSize);
    __aicore__ inline void SoftmaxDataCopyOut(SplitSameABExtraInfo &extraInfo);

    // sparse 用函数
    __aicore__ inline void GetS1LoopRange(int64_t &multiCoreInnerOffset, int64_t &multiCoreInnerLimit);
    __aicore__ inline void GetS2LoopRange();

    // TSCM mm
    __aicore__ inline void mm1LoadData(LocalTensor<INPUT_T> &tscmTensor,
                                    const GlobalTensor<INPUT_T> &globalTensor,
                                    int32_t tileHeight, int32_t tileWidth, uint8_t tscmIndex,
                                    int32_t posL1, int32_t baseBlockSize, int32_t orgWidth = 192);
    __aicore__ inline void mm2LoadDataA(LocalTensor<INPUT_T> &tscmTensor,
                                    const GlobalTensor<INPUT_T> &globalTensor, const DataCopyParams &copyParams,
                                    uint8_t tscmIndex, int32_t posL1, int32_t baseBlockSize);
    __aicore__ inline void mm2LoadDataB(LocalTensor<INPUT_T> &tscmTensor, const GlobalTensor<INPUT_T> &globalTensor,
                            int32_t tileHeight, int32_t tileWidth, uint8_t tscmIndex,
                                    int32_t posL1, int32_t baseBlockSize, int32_t orgWidth = 192);
    __aicore__ inline void CopyGmToL1(LocalTensor<INPUT_T> &l1Tensor, const GlobalTensor<INPUT_T> &gmSrcTensor, uint32_t srcN, 
                                    uint32_t srcD, uint32_t srcDstride);

    uint32_t cubeS1BaseSize;
    uint32_t vecS1BaseSize;
    uint32_t s2BaseSize;
    uint32_t dSize;
    int64_t dRopeSize = 0;
    uint32_t d2Size;
    int64_t dSizeAlign16;
    int64_t d2SizeAlign16;
    int64_t s1Size;
    int64_t s2Size;
    int64_t s1OuterSize;

    // sparse 用参数
    int64_t s2StartIdx;
    int64_t s2EndIdx;
    int64_t nextS2EndIdx;

    // BNG 外循环
    int64_t bngStartIdx;
    int64_t bngEndIdx;
    int64_t qCoreOffset;
    int64_t qRopeCoreOffset = 0;
    int32_t softmaxReduceSize = 8;

    // TSCM 参数
    int32_t mm1BaseM;
    int32_t mm1BaseN;
    int32_t mm1BaseK;
    int32_t mm1ABaseSize;
    int32_t mm1BBaseSize;

    int32_t mm2BaseM;
    int32_t mm2BaseN;
    int32_t mm2BaseK;
    int32_t mm2ABaseSize;
    int32_t mm2BBaseSize;

    // 资源分配
    TBuf<> maskTBufPing;
    TBuf<> maskTBufPong;
    TBuf<> pseTBuf;
    TBuf<> stage1PingBuf;
    TBuf<> stage1PongBuf;
    TBuf<> stage2TBuf;
    TBuf<> softmaxSumBuf[2];
    TBuf<> softmaxExpBuf[2];
    TBuf<> softmaxMaxBuf;
    TBuf<> commonTBuf; // common的复用空间
    TBuf<> softmaxTempBuf;

    LocalTensor<T> softmaxExpUb;
    GlobalTensor<T> mm1Res[2];
    GlobalTensor<T> mm2Res[2];
    GlobalTensor<T> vec2Res[2];
    GlobalTensor<INPUT_T> stage1Res[2];
    GlobalTensor<half> pseAlibiGm;

    constexpr static int32_t repeatMaxBytes = 256;
    constexpr static int32_t repeatMaxTimes = 255;
    // 轴的乘积
    int64_t gS1o;
    int64_t n2GS1o;
    int64_t s1D;
    int64_t gS1D;
    int64_t n2GS1D;
    int64_t s2D;
    int64_t n2S2D;
    int64_t s1S2;
    int64_t gS1S2;
    int64_t n2GS1S2;
    int64_t gS1;
    int64_t n2GS1;
    int64_t gD;
    int64_t gDRope = 0;
    int64_t n2D;
    int64_t n2DRope = 0;
    int64_t bN2D;
    int64_t bN2DRope = 0;
    int64_t n2G;
    int64_t n2GD;
    int64_t n2GDRope = 0;
    int64_t bN2GD;
    int64_t bN2GDRope = 0;
    int64_t gS2;

    //MLA D不等长，V和OUT为D2
    int64_t gD2;
    int64_t n2S2D2;
    int64_t s2D2;
    int64_t s1BaseD2;
    int64_t s1D2;
    int64_t gS1D2;
    int64_t n2GS1D2;
    int64_t n2D2;
    int64_t bN2D2;
    int64_t n2GD2;
    int64_t bN2GD2;
    int64_t s1BaseN2GD2;
    int64_t s1BaseBN2GD2;

    // s2base*N之后的长度
    int64_t s2BaseNratioSize;
    int64_t s2BaseN2D;
    int64_t s2BaseN2DRope = 0;
    int64_t s2BaseBN2D;
    int64_t s2BaseBN2DRope = 0;
    int64_t s1BaseN2GD;
    int64_t s1BaseN2GDRope = 0;
    int64_t s1BaseBN2GD;
    int64_t s1BaseD;
    int64_t s2BaseNratioD;
    int64_t s2BaseNratioN2D;
    int64_t s2BaseNratioN2DRope = 0;
    int64_t s2BaseNratioBN2D;
    int64_t s2BaseNratioBN2DRope = 0;
    int64_t bN2G;
    int64_t n2GS2;
    int64_t s2SizeSum;

    int64_t mm1Ka1 = 0;
    int64_t mm1Ka2 = 0;
    int64_t mm1Kb;
    int64_t mm1Kb1 = 0;
    int64_t mm1Kb2 = 0;
    int64_t mm2Kb;
    // 当splitN大于16时，需要修改softMaxCheckRes数据类型
    uint16_t softMaxCheckRes = SOFTMAX_CHECK_RES_DEFAULT_VALUE;
    uint32_t negativeIntScalar = NEGATIVE_MIN_VAULE_FP32;
    T negativeFloatScalar;
    T positiveFloatScalar;

    AttenMaskComputeMode attenMaskComputeMode = AttenMaskComputeMode::NORMAL_MODE;

    int32_t vecBlockIdx;
    int32_t cubeBlockIdx;
    int32_t cubeSubIdx;
    const FlashAttentionScoreGeneralTilingData *__restrict tilingData;

    int64_t boIdx;
    int64_t n2oIdx;
    int64_t goIdx;
    int64_t s1oIdx;

    TPipe *pipe;

    GlobalTensor<INPUT_T> queryGm;
    GlobalTensor<INPUT_T> queryRopeGm;
    GlobalTensor<INPUT_T> keyGm;
    GlobalTensor<INPUT_T> keyRopeGm;
    GlobalTensor<INPUT_T> pseGm;
    __gm__ uint8_t *pseSlope;
    GM_ADDR prefixNAddr;
    GlobalTensor<INPUT_T> valueGm;
    GlobalTensor<INPUT_T> attentionOutGm;
    GlobalTensor<float> softmaxMaxGm;
    GlobalTensor<float> softmaxSumGm;
    GlobalTensor<uint8_t> dropMaskGm;
    GlobalTensor<uint8_t> attenMaskGmInt;

    bool dropMaskUnAligned;
    int64_t attenMaskOffsetPre = 0;
    PseInfo pseInfo = {0};
    DropMaskInfo dropMaskInfo = {0};

    uint64_t SYNC_C1_V1_FLAG[3] = {4, 5, 6};
    uint64_t SYNC_V1_C2_FLAG[3] = {7, 8, 9};
    uint64_t SYNC_C2_V2_FLAG[3] = {1, 2, 3};
};

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool hasRope>
__aicore__ inline void
FlashAttentionScoreS1s2Bn2gs1SameAB<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                 bmm1Format, mmPolicyType, hasRope>::Init(
                                                   __gm__ uint8_t *query, __gm__ uint8_t* queryRope,
                                                   __gm__ uint8_t *key, __gm__ uint8_t *keyRope,
                                                   __gm__ uint8_t *value,
                                                   __gm__ uint8_t *pse, __gm__ uint8_t *dropMask,
                                                   __gm__ uint8_t *paddingMask, __gm__ uint8_t *prefix,
                                                   __gm__ uint8_t *attenMask, __gm__ uint8_t *softmaxMax,
                                                   __gm__ uint8_t *softmaxSum, __gm__ uint8_t *softmaxOut,
                                                   __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
                                                   const FlashAttentionScoreGeneralTilingData *__restrict tiling,
                                                   TPipe *tPipe)
{
    this->InitInput(query, queryRope, key, keyRope, value, pse, dropMask, paddingMask, prefix, attenMask, softmaxMax, softmaxSum,
                    softmaxOut, attentionOut, workspace, tiling, tPipe); // gm设置

    this->ComputeConstexpr();
    this->InitBuffer();
    if constexpr (hasDrop == true) {
        LocalTensor<T> apiTmpBuffer = this->commonTBuf.template Get<T>();
        DropOutBitModeInit(apiTmpBuffer);
    }
    if constexpr (hasPse == true) {
        if (this->vecBlockIdx < this->tilingData->multiCoreParams.coreNum) {
            LocalTensor<half> pseHelpBuffer = this->stage1PingBuf.template Get<half>();
            PseInnerAlibiCreate<hasPse>(this->pseAlibiGm, pseHelpBuffer, this->pseInfo);
        }
    }
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool hasRope>
__aicore__ inline void
FlashAttentionScoreS1s2Bn2gs1SameAB<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                 bmm1Format, mmPolicyType, hasRope>::InitInput(
                                                        __gm__ uint8_t *query, __gm__ uint8_t *queryRope,
                                                        __gm__ uint8_t *key, __gm__ uint8_t *keyRope,
                                                        __gm__ uint8_t *value, __gm__ uint8_t *pse,
                                                        __gm__ uint8_t *dropMask, __gm__ uint8_t *paddingMask,
                                                        __gm__ uint8_t *prefix, __gm__ uint8_t *attenMask,
                                                        __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
                                                        __gm__ uint8_t *softmaxOut, __gm__ uint8_t *attentionOut,
                                                        __gm__ uint8_t *workspace,
                                                        const FlashAttentionScoreGeneralTilingData *__restrict tiling,
                                                        TPipe *tPipe)
{
    if ASCEND_IS_AIV {
        this->vecBlockIdx = GetBlockIdx(); // vec:0-47
        this->cubeBlockIdx = this->vecBlockIdx / 2;
        this->cubeSubIdx = this->vecBlockIdx % 2;
    } else {
        this->cubeBlockIdx = GetBlockIdx(); // cube:0-23
        this->vecBlockIdx = this->cubeBlockIdx * 2;
        this->cubeSubIdx = 0;
    }

    if ASCEND_IS_AIC {
        this->bmm1.SetSubBlockIdx(0);
        this->bmm1.Init(&tiling->bmm1TilingData, tPipe);
        this->bmm1Nz.SetSubBlockIdx(0);
        this->bmm1Nz.Init(&tiling->bmm1TilingData, tPipe);
        this->bmm2.SetSubBlockIdx(0);
        this->bmm2.Init(&tiling->bmm2TilingData, tPipe);
    }

    this->pipe = tPipe;
    this->SetTiling(tiling);

    // init global buffer
    this->queryGm.SetGlobalBuffer((__gm__ INPUT_T *)query);
    if constexpr (hasRope == true) {
        this->queryRopeGm.SetGlobalBuffer((__gm__ INPUT_T *)queryRope);
    }
    this->keyGm.SetGlobalBuffer((__gm__ INPUT_T *)key);
    if constexpr (hasRope == true) {
        this->keyRopeGm.SetGlobalBuffer((__gm__ INPUT_T *)keyRope);
    }
    this->valueGm.SetGlobalBuffer((__gm__ INPUT_T *)value);
    this->pseGm.SetGlobalBuffer((__gm__ INPUT_T *)pse);
    this->pseSlope = pse;
    this->prefixNAddr = prefix;
    this->dropMaskUnAligned = this->tilingData->inputParams.needDropMaskOp == 1;
    if (this->dropMaskUnAligned) {
        this->dropMaskGm.SetGlobalBuffer(workspace);
        if constexpr (hasDrop == true) {
            workspace += CeilDiv(this->tilingData->dropmaskParams.shapeTotalSize, 512) * 512;
        }
    } else {
        this->dropMaskGm.SetGlobalBuffer((__gm__ uint8_t *)dropMask);
    }
    this->attenMaskGmInt.SetGlobalBuffer((__gm__ uint8_t *)attenMask);
    this->softmaxMaxGm.SetGlobalBuffer((__gm__ float *)softmaxMax);
    this->softmaxSumGm.SetGlobalBuffer((__gm__ float *)softmaxSum);
    this->attentionOutGm.SetGlobalBuffer((__gm__ INPUT_T *)attentionOut);

    // 补齐到512， 统一按T处理
    int64_t mm1ResultSize = cubeS1BaseSize * s2BaseSize;
    int64_t mmNRatioOffset = CeilDiv(mm1ResultSize * this->tilingData->coreParams.nRatio, 128) * 128 * sizeof(T);
    int64_t mm2ResultSize = cubeS1BaseSize * d2SizeAlign16;
    int64_t mm2Offset = CeilDiv(mm2ResultSize, 128) * 128 * 4;
    int64_t bmm1AndVec1Ratio = GM_DOUBLE_BUFFER;
    int64_t vector1OffsetPing = 0;
    int64_t vector1OffsetPong = mmNRatioOffset;

    // bmm1、bmm2 NZ场景，stage1Result不与bmm1Result共用空间，需要占用2倍mmNRatioOffset空间
    vector1OffsetPing = mmNRatioOffset * GM_DOUBLE_BUFFER;
    vector1OffsetPong = vector1OffsetPing + mmNRatioOffset / 2;
    bmm1AndVec1Ratio = GM_DOUBLE_BUFFER + 1;

    int64_t totalOffset = mmNRatioOffset * bmm1AndVec1Ratio + mm2Offset * 2 * GM_DOUBLE_BUFFER;

    int64_t pseInnerAlibiSize = this->tilingData->coreParams.pseAlibiBaseS1 *
                                this->tilingData->coreParams.pseAlibiBaseS2 * sizeof(half);
    int64_t pseAlibiOffset =  CeilDiv(pseInnerAlibiSize, 512) * 512;
    totalOffset += pseAlibiOffset;

    // bmm1Result，占用2倍mmNRatioOffset空间
    this->mm1Res[0].SetGlobalBuffer((__gm__ T *)(workspace + this->cubeBlockIdx * totalOffset));
    this->mm1Res[1].SetGlobalBuffer((__gm__ T *)(workspace + this->cubeBlockIdx * totalOffset + mmNRatioOffset));

    // stage1Result，不占用/占用1倍/占用2倍mmNRatioOffset空间
    this->stage1Res[0].SetGlobalBuffer(
        (__gm__ INPUT_T *)(workspace + this->cubeBlockIdx * totalOffset + vector1OffsetPing));
    this->stage1Res[1].SetGlobalBuffer(
        (__gm__ INPUT_T *)(workspace + this->cubeBlockIdx * totalOffset + vector1OffsetPong));

    // bmm2Result，占用2倍mmOffset空间
    this->mm2Res[0].SetGlobalBuffer(
        (__gm__ T *)(workspace + this->cubeBlockIdx * totalOffset + mmNRatioOffset * bmm1AndVec1Ratio));
    this->mm2Res[1].SetGlobalBuffer(
        (__gm__ T *)(workspace + this->cubeBlockIdx * totalOffset + mmNRatioOffset * bmm1AndVec1Ratio + mm2Offset));

    // vec2阶段，占用2倍mmOffset空间，仅在D轴大于64的情况下出现
    this->vec2Res[0].SetGlobalBuffer((__gm__ T *)(workspace + this->cubeBlockIdx * totalOffset + mmNRatioOffset *
                                                  bmm1AndVec1Ratio + mm2Offset * 2));
    this->vec2Res[1].SetGlobalBuffer((__gm__ T *)(workspace + this->cubeBlockIdx * totalOffset + mmNRatioOffset *
                                                  bmm1AndVec1Ratio + mm2Offset * 3));
    uint64_t pseAlibiAddr = this->cubeBlockIdx * totalOffset + mmNRatioOffset * bmm1AndVec1Ratio + 4 * mm2Offset;

    this->pseAlibiGm.SetGlobalBuffer((__gm__ half*)(workspace + pseAlibiAddr));
    if constexpr (IsSameType<T, half>::value) {
        this->negativeIntScalar = NEGATIVE_MIN_VAULE_FP16;
    }
    GetExtremeValue(this->negativeFloatScalar, this->positiveFloatScalar);
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool hasRope>
__aicore__ inline void FlashAttentionScoreS1s2Bn2gs1SameAB<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                            bmm1Format, mmPolicyType, hasRope>::SetTiling(const FlashAttentionScoreGeneralTilingData *__restrict tilingData)
{
    // copy base params
    this->tilingData = tilingData;
    this->cubeS1BaseSize = this->tilingData->coreParams.s1BaseSize;
    this->vecS1BaseSize = this->cubeS1BaseSize / 2;
    this->s2BaseSize = this->tilingData->coreParams.s2BaseSize;
    this->dSize = this->tilingData->inputParams.dSize;
    if constexpr (hasRope == true) {
        this->dRopeSize = this->tilingData->coreParams.dRopeSize;
    }
    this->d2Size = this->tilingData->coreParams.d2Size;
    this->dSizeAlign16 = CeilDiv(this->dSize + this->dRopeSize, 16) * 16;
    this->d2SizeAlign16 = CeilDiv(this->d2Size, 16) * 16;
};

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool hasRope>
__aicore__ inline void FlashAttentionScoreS1s2Bn2gs1SameAB<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                            bmm1Format, mmPolicyType, hasRope>::InitBuffer()
{
    uint64_t stage1Size = 8 * 1024;
    uint64_t stage1AttenSize = 9 * 1024;
    uint64_t stage1PongSize = 34 * 1024;
    uint64_t stage2Size = 64 * 128;
    uint64_t maskTBufPongSize = 16 * 1024;

    // 可选输入的buffer空间，保持和stage1处理的size一致
    this->pipe->InitBuffer(this->maskTBufPing, stage1AttenSize); // 可以给attenmask 9k
    this->pipe->InitBuffer(this->maskTBufPong, maskTBufPongSize); // 可以给dropoutmask 16k
    this->pipe->InitBuffer(this->pseTBuf, 16384); // pse 16k

    this->pipe->InitBuffer(this->stage1PingBuf, stage2Size * sizeof(T)); // t.a 32k
    this->pipe->InitBuffer(this->stage2TBuf, stage2Size * sizeof(T));    // t.c 32k
    this->pipe->InitBuffer(this->commonTBuf, stage2Size * sizeof(T));    // t.b 32k

    this->pipe->InitBuffer(this->softmaxSumBuf[0], vecS1BaseSize * 4 * this->softmaxReduceSize);  // 2k
    this->pipe->InitBuffer(this->softmaxSumBuf[1], vecS1BaseSize * 4 * this->softmaxReduceSize);  // 2k
    this->pipe->InitBuffer(this->softmaxMaxBuf, vecS1BaseSize * 4 * this->softmaxReduceSize);     // 2k
    this->pipe->InitBuffer(this->softmaxExpBuf[0], vecS1BaseSize * 4 * this->softmaxReduceSize);  // 2k
    this->pipe->InitBuffer(this->softmaxExpBuf[1], vecS1BaseSize * 4 * this->softmaxReduceSize);  // 2k
    if (this->softmaxReduceSize == 1) {
        this->pipe->InitBuffer(this->softmaxTempBuf, vecS1BaseSize * blockBytes);    // 16k
    }
    this->pipe->InitBuffer(this->stage1PongBuf, stage1PongSize); // i.a 34k
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool hasRope>
__aicore__ inline void FlashAttentionScoreS1s2Bn2gs1SameAB<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                        bmm1Format, mmPolicyType, hasRope>::ComputeConstexpr()
{
    // 计算轴的乘积
    this->s1OuterSize = this->tilingData->coreParams.s1OuterSize;
    this->s1D = this->tilingData->inputParams.s1Size * dSize;
    this->s2D = this->tilingData->inputParams.s2Size * dSize;
    this->gD = this->tilingData->inputParams.gSize * dSize;
    this->n2D = this->tilingData->inputParams.n2Size * dSize;
    this->s1S2 = this->tilingData->inputParams.s1Size * this->tilingData->inputParams.s2Size;
    this->gS1 = this->tilingData->inputParams.gSize * this->tilingData->inputParams.s1Size;
    this->n2G = this->tilingData->inputParams.n2Size * this->tilingData->inputParams.gSize;
    this->gS1o = this->tilingData->inputParams.gSize * this->s1OuterSize;

    this->bN2D = this->tilingData->inputParams.bSize * n2D;
    this->n2GS1o = this->tilingData->inputParams.n2Size * this->gS1o;
    this->gS1D = this->tilingData->inputParams.gSize * this->s1D;
    this->n2S2D = this->tilingData->inputParams.n2Size * this->s2D;
    this->n2GD = this->tilingData->inputParams.n2Size * this->gD;
    this->bN2GD = this->tilingData->inputParams.bSize * n2GD;
    this->gS1S2 = this->tilingData->inputParams.gSize * this->s1S2;
    this->n2GS1 = this->tilingData->inputParams.n2Size * this->gS1;

    this->n2GS1D = this->tilingData->inputParams.n2Size * this->gS1D;
    this->n2GS1S2 = this->tilingData->inputParams.n2Size * this->gS1S2;

    this->gD2 = this->tilingData->inputParams.gSize * d2Size;
    this->n2D2 = this->tilingData->inputParams.n2Size * d2Size;
    this->s1D2 = this->tilingData->inputParams.s1Size * d2Size;
    this->gS1D2 = this->tilingData->inputParams.gSize * this->s1D2;
    this->n2GS1D2 = this->tilingData->inputParams.n2Size * this->gS1D2;
    this->s2D2 = this->tilingData->inputParams.s2Size * d2Size;
    this->n2S2D2 = this->tilingData->inputParams.n2Size * this->s2D2;
    this->bN2D2 = this->tilingData->inputParams.bSize * n2D2;
    this->n2GD2 = this->tilingData->inputParams.n2Size * this->gD2;
    this->bN2GD2 = this->tilingData->inputParams.bSize * n2GD2;

    // 计算切分轴的乘积
    this->s2BaseN2D = this->s2BaseSize * this->n2D;
    this->s2BaseNratioSize = this->s2BaseSize * this->tilingData->coreParams.nRatio;
    
    if constexpr (hasRope == true) {
        this->gDRope = this->tilingData->inputParams.gSize * dRopeSize;
        this->n2DRope = this->tilingData->inputParams.n2Size * dRopeSize;
        this->n2GDRope = this->tilingData->inputParams.n2Size * this->gDRope;
        this->s2BaseN2DRope = this->s2BaseSize * this->n2DRope;
        this->bN2DRope = this->tilingData->inputParams.bSize * n2DRope;
        this->bN2GDRope = this->tilingData->inputParams.bSize * n2GDRope;
    }

    if constexpr (layOutType == LayOutTypeEnum::LAYOUT_BSH) {
        // BSH/BSNGD
        this->s1BaseN2GD = this->cubeS1BaseSize * this->n2GD;
        this->s1BaseN2GD2 = this->cubeS1BaseSize * this->n2GD2;
        this->s2BaseNratioN2D = this->s2BaseN2D * this->tilingData->coreParams.nRatio;
        this->mm1Ka1 = this->n2GD;
        this->mm1Kb1 = this->n2D;
        this->mm1Kb = this->mm1Kb1;
        this->mm2Kb = this->n2D2;
    } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_SBH) {
        // SBH/SBNGD
        this->bN2G = this->tilingData->inputParams.bSize * this->n2G;
        this->s1BaseBN2GD = this->cubeS1BaseSize * this->tilingData->inputParams.bSize * this->n2GD;
        this->s1BaseBN2GD2 = this->cubeS1BaseSize * this->tilingData->inputParams.bSize * this->n2GD2;
        this->s2BaseBN2D = this->tilingData->inputParams.bSize * this->s2BaseN2D;
        this->s2BaseBN2DRope = this->tilingData->inputParams.bSize * this->s2BaseN2DRope;
        this->s2BaseNratioBN2D = this->s2BaseBN2D * this->tilingData->coreParams.nRatio;
        this->s2BaseNratioBN2DRope = this->s2BaseBN2DRope * this->tilingData->coreParams.nRatio;
        this->mm1Ka1 = this->bN2GD;
        this->mm1Ka2 = this->bN2GDRope;
        this->mm1Kb1 = this->bN2D;
        this->mm1Kb2 = this->bN2DRope;
        this->mm1Kb = this->mm1Kb1 + this->mm1Kb2;
        this->mm2Kb = this->bN2D2;
    } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_BNSD) {
        // BNSD
        this->s1BaseD = this->cubeS1BaseSize * this->dSize;
        this->s1BaseD2 = this->cubeS1BaseSize * this->d2Size;
        this->s2BaseNratioD = this->s2BaseNratioSize * this->dSize;
        this->mm1Ka1 = this->dSize;
        this->mm1Kb1 = this->dSize;
        this->mm1Kb = this->mm1Kb1; 
        this->mm2Kb = this->d2Size;
    }

    if (this->tilingData->inputParams.pseShapeType == pse1S2) {
        this->gS2 = this->tilingData->inputParams.gSize * this->tilingData->inputParams.s2Size;
        this->n2GS2 = this->tilingData->inputParams.n2Size * this->gS2;
    }
    if constexpr (hasPse == true) {
        this->pseInfo.gSize = this->tilingData->inputParams.gSize;
        this->pseInfo.pseShapeType = this->tilingData->inputParams.pseShapeType;
        this->pseInfo.pseType = this->tilingData->inputParams.pseType;
        this->pseInfo.n2G = this->n2G;
        this->pseInfo.pseBSize = this->tilingData->inputParams.pseBSize;
        this->pseInfo.s1BaseSize = this->cubeS1BaseSize;
        this->pseInfo.pseS1Size = this->tilingData->inputParams.pseS1Size;
        this->pseInfo.pseS2Size = this->tilingData->inputParams.pseS2Size;
        this->pseInfo.s2BaseNratioSize = this->s2BaseNratioSize;
        this->pseInfo.pseEncodeType = (uint32_t)this->tilingData->inputParams.pseEncodeType;
        this->pseInfo.pseAlibiBaseS1 = this->tilingData->coreParams.pseAlibiBaseS1;
        this->pseInfo.pseAlibiBaseS2 = this->tilingData->coreParams.pseAlibiBaseS2;
        this->pseInfo.qStartIdx = this->tilingData->inputParams.qStartIdx;
        this->pseInfo.kvStartIdx = this->tilingData->inputParams.kvStartIdx;
    }
    if constexpr (hasDrop == true) {
        this->dropMaskInfo.gSize = this->tilingData->inputParams.gSize;
        this->dropMaskInfo.n2G = this->n2G;
        this->dropMaskInfo.s1BaseSize = this->cubeS1BaseSize;
        this->dropMaskInfo.s2BaseNratioSize = this->s2BaseNratioSize;
    }
    if (this->cubeS1BaseSize > 256) {
        this->softmaxReduceSize = 1;
    }
    if constexpr (mmPolicyType == MmPolicyType::UNSPLITK) {
        this->mm1BaseM = this->tilingData->bmm1TilingData.baseM;
        this->mm1BaseN = this->tilingData->bmm1TilingData.baseN;
        this->mm1BaseK = this->tilingData->bmm1TilingData.baseK;
        this->mm1ABaseSize = this->mm1BaseM * this->mm1BaseK;
        this->mm1BBaseSize = this->mm1BaseK * this->mm1BaseN;

        this->mm2BaseM = this->tilingData->bmm2TilingData.baseM;
        this->mm2BaseN = this->tilingData->bmm2TilingData.baseN;
        this->mm2BaseK = this->tilingData->bmm2TilingData.baseK;
        this->mm2ABaseSize = this->mm2BaseM * this->mm2BaseK;
        this->mm2BBaseSize = this->mm2BaseK * this->mm2BaseN;
    }
}

// sparse functions
template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool hasRope>
__aicore__ inline void
FlashAttentionScoreS1s2Bn2gs1SameAB<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                 bmm1Format, mmPolicyType, hasRope>::GetS1LoopRange(int64_t &multiCoreInnerOffset,
                                                             int64_t &multiCoreInnerLimit)
{
    // 计算sparse场景下s1的循环范围
    if constexpr (layOutType == LayOutTypeEnum::LAYOUT_TND) {
        // sparse场景下负载均衡后每个核获取的结果
        multiCoreInnerOffset = this->tilingData->multiCoreParams.sparseStartIdx[this->cubeBlockIdx];
        if (likely((this->tilingData->multiCoreParams.coreNum - 2) > this->vecBlockIdx)) {
            multiCoreInnerLimit = this->tilingData->multiCoreParams.sparseStartIdx[this->cubeBlockIdx + 1];
        } else {
            multiCoreInnerLimit = this->tilingData->multiCoreParams.totalSize;
        }
    } else {
        if (this->tilingData->inputParams.sparseType > 0) {
            // sparse场景下负载均衡后每个核获取的结果
            multiCoreInnerOffset = this->tilingData->multiCoreParams.sparseStartIdx[this->cubeBlockIdx];
            if (likely((this->tilingData->multiCoreParams.coreNum - 2) > this->vecBlockIdx)) {
                multiCoreInnerLimit = this->tilingData->multiCoreParams.sparseStartIdx[this->cubeBlockIdx + 1];
            } else {
                multiCoreInnerLimit = this->tilingData->multiCoreParams.totalSize;
            }
        }
    }
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool hasRope>
__aicore__ inline void
FlashAttentionScoreS1s2Bn2gs1SameAB<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                bmm1Format, mmPolicyType, hasRope>::GetS2LoopRange()
{
    // 计算S2的循环范围相关参数: 后续可 使用static_cast<uint32_t>优化scale性能
    if (this->tilingData->inputParams.sparseType == static_cast<uint8_t>(SparseModeEnum::CAUSAL)) { // 下三角
        this->s2StartIdx = 0;
        this->s2EndIdx = Min((this->s1oIdx + 1) * this->cubeS1BaseSize, this->s2Size);
    } else if (this->tilingData->inputParams.sparseType ==
               static_cast<uint8_t>(SparseModeEnum::BAND)) { // 对角线往外扩散场景, s1和s2可能不同
        this->s2StartIdx = Max(this->s1oIdx * this->cubeS1BaseSize -
                               this->tilingData->coreParams.s1SparseValidSize, 0);
        this->s2EndIdx = Min((this->s1oIdx + 1) * this->cubeS1BaseSize + this->tilingData->coreParams.s2SparseValidSize,
                             this->s2Size);
    } else if (this->tilingData->inputParams.sparseType == static_cast<uint8_t>(SparseModeEnum::PREFIX)) {
        this->s2StartIdx = 0;
        this->s2EndIdx = Max(this->cubeS1BaseSize * (this->s1oIdx + 1) - this->s1Size + this->s2Size,
                             ((__gm__ int64_t *)this->prefixNAddr)[this->boIdx]);
    } else { // 其它场景, 如无attention mask
        this->s2StartIdx = 0;
        this->s2EndIdx = this->s2Size;
        return;
    }

    if (this->tilingData->inputParams.sparseType == static_cast<uint8_t>(SparseModeEnum::BAND)) {
        // s1baseSize行都无效时, 将startIdx设置为0, endIdx设置为S2realSize
        if (this->s2EndIdx - this->s2StartIdx <= 0) {
            this->s2StartIdx = 0;
            this->s2EndIdx = Min(this->s2Size, 128L);
        }
    } else if (this->tilingData->inputParams.sparseType == static_cast<uint8_t>(SparseModeEnum::PREFIX)) {
        if (this->s2EndIdx <= this->s2StartIdx) {
            // 无效行场景至少要算一个基本块
            this->s2EndIdx = 128L;
        } else {
            this->s2EndIdx = CeilDiv(this->s2EndIdx, s2BaseSize) * s2BaseSize;
        }
        this->s2EndIdx = Min(this->s2EndIdx, this->s2Size);
    }
    return;
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool hasRope>
__aicore__ inline void FlashAttentionScoreS1s2Bn2gs1SameAB<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                        bmm1Format, mmPolicyType, hasRope>::Process()
{
    // 确定核内切分起点
    int64_t multiCoreInnerOffset = this->cubeBlockIdx * this->tilingData->multiCoreParams.splitFactorSize;
    int64_t multiCoreInnerLimit = multiCoreInnerOffset + this->tilingData->multiCoreParams.splitFactorSize;
    if (this->tilingData->multiCoreParams.totalSize < multiCoreInnerLimit) {
        multiCoreInnerLimit = this->tilingData->multiCoreParams.totalSize;
    }
    // 计算sparse场景下s1的循环范围
    this->GetS1LoopRange(multiCoreInnerOffset, multiCoreInnerLimit);

    SplitSameABExtraInfo extraInfo[3];
    int64_t taskId = 0;
    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));

    bool notSecondLast = true;
    bool notLast = true;
    multiCoreInnerLimit += 2;
    for (int64_t multiCoreInnerIdx = multiCoreInnerOffset; multiCoreInnerIdx < multiCoreInnerLimit;
            ++multiCoreInnerIdx) {
        if (multiCoreInnerIdx == multiCoreInnerLimit - 2) {
            notSecondLast = false;
        } else if (multiCoreInnerIdx == multiCoreInnerLimit - 1) {
            notLast = false;
        }

        int64_t s2LoopLimit;
        this->softMaxCheckRes = SOFTMAX_CHECK_RES_DEFAULT_VALUE;
        bool notLastTwoLoop = notSecondLast && notLast;
        if (notLastTwoLoop) {
            this->ComputeAxisIdx(multiCoreInnerIdx);

            // s2轴循环计数, 支持sparse和非sparse场景
            this->GetS2LoopRange();
            s2LoopLimit = CeilDiv(this->s2EndIdx - this->s2StartIdx, s2BaseNratioSize) - 1;
        } else {
            s2LoopLimit = 0;
        }
        for (int64_t s2LoopCount = 0; s2LoopCount <= s2LoopLimit; ++s2LoopCount) {
            if (taskId >= 1 && notLast) {
                // 对应extraInfo[(i+2)%3]
                if ASCEND_IS_AIV {
                    CrossCoreWaitFlag(SYNC_C1_V1_FLAG[(taskId + 2) % 3]);
                }
            }

            if (notLastTwoLoop) {
                this->SetExtraInfo(extraInfo[taskId % 3], taskId, s2LoopCount, s2LoopLimit, multiCoreInnerIdx);
                if ASCEND_IS_AIC {
                    if constexpr (bmm1Format == CubeFormat::NZ) {
                        if (extraInfo[taskId % 3].needNz2Nd == 1) {
                            this->IterateBmm1(extraInfo[taskId % 3], this->bmm1Nz);
                        } else {
                            this->IterateBmm1(extraInfo[taskId % 3], this->bmm1);
                        }
                    } else {
                        this->IterateBmm1(extraInfo[taskId % 3], this->bmm1);
                    }
                    CrossCoreSetFlag<2, PIPE_FIX>(SYNC_C1_V1_FLAG[taskId % 3]);
                }
            }

            if (taskId > 0 && notLast) {
                if ASCEND_IS_AIV {
                    this->ProcessVec1(extraInfo[(taskId + 2) % 3]);
                    AscendC::SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);                
                    CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_V1_C2_FLAG[(taskId + 2) % 3]);
                }
            }

            if (taskId > 1) {
                // 对应extraInfo[(i+1)%3]
                if ASCEND_IS_AIV {
                    CrossCoreWaitFlag(SYNC_C2_V2_FLAG[(taskId + 1) % 3]);
                }
            }

            if (taskId > 0 && notLast) {
                if ASCEND_IS_AIV {
                    AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
                }
                if ASCEND_IS_AIC {
                    CrossCoreWaitFlag(SYNC_V1_C2_FLAG[(taskId + 2) % 3]);
                    this->IterateBmm2(extraInfo[(taskId + 2) % 3], this->bmm2);
                    CrossCoreSetFlag<2, PIPE_FIX>(SYNC_C2_V2_FLAG[(taskId + 2) % 3]);
                }     
            }

            if (taskId > 1) {
                if ASCEND_IS_AIV {
                    this->ProcessVec2(extraInfo[(taskId + 1) % 3]);
                }
            }
            taskId++;
        }
    }
};

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool hasRope>
__aicore__ inline void
FlashAttentionScoreS1s2Bn2gs1SameAB<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                bmm1Format, mmPolicyType, hasRope>::ComputeAxisIdx(int64_t multiCoreInnerIdx)
{
    // 计算轴的idx
    this->boIdx = multiCoreInnerIdx / this->n2GS1o;
    this->n2oIdx = multiCoreInnerIdx % this->n2GS1o / this->gS1o;
    this->goIdx = multiCoreInnerIdx % this->gS1o / this->s1OuterSize;
    this->s1oIdx = multiCoreInnerIdx % this->s1OuterSize;
    this->s1Size = this->tilingData->inputParams.s1Size;
    this->s2Size = this->tilingData->inputParams.s2Size;
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool hasRope>
__aicore__ inline void
FlashAttentionScoreS1s2Bn2gs1SameAB<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                 bmm1Format, mmPolicyType, hasRope>::SetExtraInfo(SplitSameABExtraInfo &extraInfo, int64_t taskId,
                                                           int64_t s2LoopCount, int64_t s2LoopLimit,
                                                           int64_t multiCoreInnerIdx)
{
    extraInfo.s2StartIdx = this->s2StartIdx;
    extraInfo.s2EndIdx = this->s2EndIdx;
    extraInfo.s2LoopCount = s2LoopCount;
    extraInfo.s1oIdx = this->s1oIdx;
    extraInfo.boIdx = this->boIdx;
    extraInfo.n2oIdx = this->n2oIdx;
    extraInfo.goIdx = this->goIdx;
    extraInfo.taskId = taskId;
    extraInfo.taskIdMod2 = taskId % 2;
    extraInfo.s2LoopLimit = s2LoopLimit;
    extraInfo.multiCoreInnerIdx = multiCoreInnerIdx;
    extraInfo.multiCoreInnerIdxMod2 = multiCoreInnerIdx % 2;
    extraInfo.s1Size = this->tilingData->inputParams.s1Size;
    extraInfo.s2Size = this->tilingData->inputParams.s2Size;
    extraInfo.attenB1SSOffset = extraInfo.boIdx * this->s1S2;
    extraInfo.attenMaskS2Size = this->tilingData->inputParams.attenMaskS2Size;
    extraInfo.s2SizeAcc = extraInfo.boIdx * extraInfo.s2Size;
    extraInfo.s1SizeAcc = extraInfo.boIdx * extraInfo.s1Size;
    extraInfo.cubeS1RealSize = Min(this->cubeS1BaseSize,
                                   this->tilingData->inputParams.s1Size - extraInfo.s1oIdx * this->cubeS1BaseSize);
    extraInfo.s1RealSize = (extraInfo.cubeS1RealSize + 1) / 2;
    extraInfo.vecCoreOffset = this->cubeSubIdx * extraInfo.s1RealSize;
    if (this->cubeSubIdx == 1) {
        extraInfo.s1RealSize = extraInfo.cubeS1RealSize - extraInfo.s1RealSize;
    }

    this->ComputeBmm1Tail(extraInfo);
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool hasRope>
__aicore__ inline void
FlashAttentionScoreS1s2Bn2gs1SameAB<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                bmm1Format, mmPolicyType, hasRope>::ComputeBmm1Tail(SplitSameABExtraInfo &extraInfo)
{
    extraInfo.s2RealSize = this->s2BaseNratioSize;
    extraInfo.s2AlignedSize = extraInfo.s2RealSize;
    extraInfo.s2RealSizeAlign64 = extraInfo.s2AlignedSize;
    extraInfo.s2RealSizeFloorAlign8 = extraInfo.s2RealSize / 8 * 8;
    extraInfo.s2LastLoop = false;
    if (extraInfo.s2StartIdx + (extraInfo.s2LoopCount + 1) * extraInfo.s2RealSize >= extraInfo.s2EndIdx) {
        extraInfo.s2RealSize = extraInfo.s2EndIdx - extraInfo.s2LoopCount * extraInfo.s2RealSize - extraInfo.s2StartIdx;
        extraInfo.s2AlignedSize = Align(extraInfo.s2RealSize);
        extraInfo.s2RealSizeAlign64 = CeilDiv(extraInfo.s2RealSize, 64) * 64;
        extraInfo.s2RealSizeFloorAlign8 = extraInfo.s2RealSize / 8 * 8;
        extraInfo.s2LastLoop = true;
        extraInfo.duplicateMask = (((uint64_t)1 << (extraInfo.s2RealSizeAlign64 - extraInfo.s2RealSizeFloorAlign8)) - 1) &
            (~(((uint64_t)1 << (extraInfo.s2RealSize - extraInfo.s2RealSizeFloorAlign8)) - 1));
        if (extraInfo.s2RealSizeAlign64 - extraInfo.s2RealSizeFloorAlign8 == 64) {
            extraInfo.duplicateMask = 0xffffffffffffffff &
                (~(((uint64_t)1 << (extraInfo.s2RealSize - extraInfo.s2RealSizeFloorAlign8)) - 1));
        }
    }

    extraInfo.vec1S1BaseSize = Min(1024 / extraInfo.s2RealSizeAlign64 * 8, extraInfo.s1RealSize); // 基本块为8 * 1024大小
    extraInfo.realSplitN = CeilDiv(extraInfo.s1RealSize, extraInfo.vec1S1BaseSize);

    if (dSizeAlign16 > 64) {
        extraInfo.vec2S1BaseSize = 1024 / dSizeAlign16 * 8;
    } else {
        extraInfo.vec2S1BaseSize = Min(128, extraInfo.s1RealSize);
    }
    if constexpr (bmm1Format == CubeFormat::NZ) {
        extraInfo.needNz2Nd = (extraInfo.s2RealSize % 64 == 0) ? 0 : 1;
    } else {
        extraInfo.needNz2Nd = 0;
    }
    return;
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool hasRope>
__aicore__ inline void
FlashAttentionScoreS1s2Bn2gs1SameAB<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                bmm1Format, mmPolicyType, hasRope>::CopyGmToL1(LocalTensor<INPUT_T> &l1Tensor, const GlobalTensor<INPUT_T> &gmSrcTensor,
                                                     uint32_t srcN, uint32_t srcD, uint32_t srcDstride)
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = srcN; // 行数
    nd2nzPara.dValue = srcD;
    nd2nzPara.srcDValue = srcDstride;
    nd2nzPara.dstNzC0Stride = CeilDiv(srcN, 16) * 16; // 对齐到16 单位block
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    AscendC::DataCopy(l1Tensor, gmSrcTensor, nd2nzPara);
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool hasRope>
__aicore__ inline void
FlashAttentionScoreS1s2Bn2gs1SameAB<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                bmm1Format, mmPolicyType, hasRope>::mm1LoadData(LocalTensor<INPUT_T> &tscmTensor,
                                    const GlobalTensor<INPUT_T> &globalTensor,
                                    int32_t tileHeight, int32_t tileWidth, uint8_t tscmIndex,
                                    int32_t posL1, int32_t baseBlockSize, int32_t orgWidth)
{
    if (posL1 < TscmGlobal[tscmIndex].cacheSize) {
        LocalTensor<INPUT_T> aL1Tensor;
        aL1Tensor.SetAddr(TscmGlobal[tscmIndex].srcAddr);
        tscmTensor = aL1Tensor[posL1 * baseBlockSize];
        return;
    }
    if (TscmGlobal[tscmIndex].needAlloc) {
        tscmTensor = TscmGlobal[tscmIndex].scm.AllocTensor<INPUT_T>();
        TscmGlobal[tscmIndex].needAlloc = false;
        TscmGlobal[tscmIndex].srcAddr = tscmTensor.address_;
        TscmGlobal[tscmIndex].bufferSize = tscmTensor.GetSize();
    } else {
        LocalTensor<INPUT_T> aL1Tensor;
        aL1Tensor.SetAddr(TscmGlobal[tscmIndex].srcAddr);
        tscmTensor = aL1Tensor[posL1 * baseBlockSize];
    }

    CopyGmToL1(tscmTensor, globalTensor[posL1 * baseBlockSize], tileHeight, tileWidth, orgWidth);

    TscmGlobal[tscmIndex].cacheSize += 2;
    return;
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool hasRope>
__aicore__ inline void
FlashAttentionScoreS1s2Bn2gs1SameAB<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                bmm1Format, mmPolicyType, hasRope>::mm2LoadDataA(LocalTensor<INPUT_T> &tscmTensor,
                                    const GlobalTensor<INPUT_T> &globalTensor, const DataCopyParams &copyParams,
                                    uint8_t tscmIndex, int32_t posL1, int32_t baseBlockSize)
{
    int32_t baseBlockNum = TscmGlobal[tscmIndex].bufferSize / baseBlockSize;
    posL1 = posL1 % baseBlockNum;
    if (posL1 == 0) {
      TscmGlobal[tscmIndex].cacheSize = 0;
    }
    if (posL1 < TscmGlobal[tscmIndex].cacheSize) {
        LocalTensor<INPUT_T> aL1Tensor;
        aL1Tensor.SetAddr(TscmGlobal[tscmIndex].srcAddr);
        tscmTensor = aL1Tensor[posL1 * baseBlockSize];
        return;
    }
    if (TscmGlobal[tscmIndex].needAlloc) {
        tscmTensor = TscmGlobal[tscmIndex].scm.AllocTensor<INPUT_T>();
        TscmGlobal[tscmIndex].needAlloc = false;
    } else {
        LocalTensor<INPUT_T> aL1Tensor;
        aL1Tensor.SetAddr(TscmGlobal[tscmIndex].srcAddr);
        tscmTensor = aL1Tensor[posL1 * baseBlockSize];
    }

    DataCopy(tscmTensor, globalTensor, copyParams);

    if (posL1 == 0) {
        TscmGlobal[tscmIndex].srcAddr = tscmTensor.address_;
        TscmGlobal[tscmIndex].bufferSize = tscmTensor.GetSize();
    }

    TscmGlobal[tscmIndex].cacheSize += 2;
    return;
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool hasRope>
__aicore__ inline void
FlashAttentionScoreS1s2Bn2gs1SameAB<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                bmm1Format, mmPolicyType, hasRope>::mm2LoadDataB(LocalTensor<INPUT_T> &tscmTensor,
                                    const GlobalTensor<INPUT_T> &globalTensor,
                                    int32_t tileHeight, int32_t tileWidth, uint8_t tscmIndex,
                                    int32_t posL1, int32_t baseBlockSize, int32_t orgWidth)
{
    // 是否需要缓存,当前一次只搬一个base块
    if (posL1 < TscmGlobal[tscmIndex].cacheSize) {
        LocalTensor<INPUT_T> aL1Tensor;
        aL1Tensor.SetAddr(TscmGlobal[tscmIndex].srcAddr);
        tscmTensor = aL1Tensor[posL1 * baseBlockSize];
        return;
    }
    if (TscmGlobal[tscmIndex].needAlloc) {
        tscmTensor = TscmGlobal[tscmIndex].scm.AllocTensor<INPUT_T>();
        TscmGlobal[tscmIndex].needAlloc = false;
        TscmGlobal[tscmIndex].srcAddr = tscmTensor.address_;
    } else {
        LocalTensor<INPUT_T> aL1Tensor;
        aL1Tensor.SetAddr(TscmGlobal[tscmIndex].srcAddr);
        tscmTensor = aL1Tensor[posL1 * baseBlockSize];
    }

    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 2;
    nd2nzPara.nValue = tileHeight; // 行数
    nd2nzPara.dValue = tileWidth;
    nd2nzPara.srcDValue = orgWidth;
    nd2nzPara.dstNzC0Stride = CeilDiv(tileHeight, 16) * 16; // 对齐到16 单位block
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = baseBlockSize;
    nd2nzPara.dstNzMatrixStride = baseBlockSize;
    AscendC::DataCopy(tscmTensor, globalTensor[posL1 * baseBlockSize], nd2nzPara);

    TscmGlobal[tscmIndex].cacheSize += 2;
    return;
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool hasRope>
template <typename T2>
__aicore__ inline void
FlashAttentionScoreS1s2Bn2gs1SameAB<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                 bmm1Format, mmPolicyType, hasRope>::IterateBmm1(SplitSameABExtraInfo &extraInfo, 
                                                          matmul::MatmulImpl<a1Type, b1Type, T2, bias1Type, CFG_CUSTOM_EXCEED> &bmm1)
{
    if constexpr (hasRope)
    {
        bmm1.SetOrgShape(extraInfo.cubeS1RealSize, this->mm1Kb, this->mm1Ka2, this->mm1Kb2, extraInfo.s2RealSize);
        this->Bmm1SetTensorARope(extraInfo, bmm1);
        this->SetBmm1TensorBRope(extraInfo, bmm1);
        bmm1.template IterateAll<false>(this->mm1Res[extraInfo.taskIdMod2], 0, false, false);
        bmm1.End();
    }

    bmm1.SetOrgShape(extraInfo.cubeS1RealSize, this->mm1Kb, this->mm1Ka1, this->mm1Kb1, extraInfo.s2RealSize);
    this->Bmm1SetTensorA(extraInfo, bmm1);
    this->SetBmm1TensorB(extraInfo, bmm1);
    if constexpr (mmPolicyType != MmPolicyType::UNSPLITK) {
        bmm1.template IterateAll<false>(this->mm1Res[extraInfo.taskIdMod2], hasRope, false, true);
        bmm1.End();
        return;
    }

    int32_t aRowNum = extraInfo.cubeS1RealSize / mm1BaseM;
    int32_t aColNum = this->mm1Ka1 / mm1BaseK;
    int32_t bRowNum = aColNum;
    int32_t bColNum = extraInfo.s2RealSize / mm1BaseN;
    int32_t rowNum = aRowNum;
    int32_t colNum = bColNum;

    LocalTensor<INPUT_T> scmATensor;
    LocalTensor<INPUT_T> scmBTensor;

    TscmGlobal[Q_VEC1_INDEX].needAlloc = true;
    TscmGlobal[K_V_INDEX].needAlloc = true;
    TscmGlobal[Q_VEC1_INDEX].cacheSize = 0;
    TscmGlobal[K_V_INDEX].cacheSize = 0;

    int32_t posA = 0;
    int32_t posB = 0;

    int32_t cOffset = 0;
    event_t eventIdMte1ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE1_MTE2));
    event_t eventIdMte2ToMte1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE1));
    AscendC::SetFlag<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2);
    AscendC::WaitFlag<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2);
    for (int32_t curCol = 0;curCol < colNum;++curCol) {
        for (int32_t curRow = 0;curRow < rowNum;++curRow) {
            for (int32_t curK = 0;curK < aColNum;++curK) {
                posA = curRow * aColNum + curK;
                posB = curCol * bRowNum + curK;
                this->mm1LoadData(scmATensor, this->queryGm[extraInfo.qCoreOffset],
                            mm1BaseM, mm1BaseK * 2, Q_VEC1_INDEX, posA, mm1ABaseSize);
                this->mm1LoadData(scmBTensor, this->keyGm[extraInfo.kCoreOffset],
                            mm1BaseN, mm1BaseK * 2, K_V_INDEX, posB, mm1BBaseSize);
                AscendC::SetFlag<HardEvent::MTE2_MTE1>(eventIdMte2ToMte1);
                AscendC::WaitFlag<HardEvent::MTE2_MTE1>(eventIdMte2ToMte1);
                bmm1.SetTensorA(scmATensor);
                bmm1.SetTensorB(scmBTensor, true);
                bmm1.SetTail(mm1BaseM, mm1BaseN, mm1BaseK);
                bmm1.template Iterate<false>(curK != 0);
                cOffset = curCol * mm1BaseN + curRow * extraInfo.s2RealSize * mm1BaseM;
            }
            bmm1.GetTensorC(this->mm1Res[extraInfo.taskIdMod2][cOffset]);
        }
    }
    AscendC::SetFlag<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2);
    AscendC::WaitFlag<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2);

    bmm1.End();
    scmATensor.SetAddr(TscmGlobal[Q_VEC1_INDEX].srcAddr);
    scmBTensor.SetAddr(TscmGlobal[K_V_INDEX].srcAddr);
    TscmGlobal[Q_VEC1_INDEX].scm.FreeTensor(scmATensor);
    TscmGlobal[K_V_INDEX].scm.FreeTensor(scmBTensor);
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool hasRope>
template <typename T2>
__aicore__ inline void
FlashAttentionScoreS1s2Bn2gs1SameAB<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                 bmm1Format, mmPolicyType, hasRope>::Bmm1SetTensorA(SplitSameABExtraInfo &extraInfo,
                                                             matmul::MatmulImpl<a1Type, b1Type, T2, bias1Type, CFG_CUSTOM_EXCEED> &bmm1)
{
    // 计算gm上的offset
    int64_t bOffset = 0;

    // s1需要考虑inner轴的影响
    int64_t s1Offset = 0;
    int64_t n2Offset = 0;
    int64_t gOffset = 0;
    if constexpr (layOutType == LayOutTypeEnum::LAYOUT_BSH) {
        // BSH/BSNGD
        bOffset = extraInfo.boIdx * this->n2GS1D;
        s1Offset = extraInfo.s1oIdx * this->s1BaseN2GD;
        n2Offset = extraInfo.n2oIdx * this->gD;
        gOffset = extraInfo.goIdx * dSize;
    } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_SBH) {
        // SBH/SBNGD
        s1Offset = extraInfo.s1oIdx * this->s1BaseBN2GD;
        bOffset = extraInfo.boIdx * this->n2GD;
        n2Offset = extraInfo.n2oIdx * this->gD;
        gOffset = extraInfo.goIdx * dSize;
    } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_BNSD) {
        // bnsd
        bOffset = extraInfo.boIdx * this->n2GS1D;
        n2Offset = extraInfo.n2oIdx * this->gS1D;
        gOffset = extraInfo.goIdx * this->s1D;
        s1Offset = extraInfo.s1oIdx * this->s1BaseD;
    } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_TND) {
        bOffset = extraInfo.s1SizeAcc * this->n2GD;
        s1Offset = extraInfo.s1oIdx * this->s1BaseN2GD;
        n2Offset = extraInfo.n2oIdx * this->gD;
        gOffset = extraInfo.goIdx * this->dSize;
    }
    this->qCoreOffset = bOffset + n2Offset + gOffset + s1Offset;
    extraInfo.qCoreOffset = this->qCoreOffset;
    if constexpr (mmPolicyType != MmPolicyType::UNSPLITK) {
        bmm1.SetTensorA(this->queryGm[extraInfo.qCoreOffset]);
    }
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool hasRope>
template <typename T2>
__aicore__ inline void
FlashAttentionScoreS1s2Bn2gs1SameAB<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                 bmm1Format, mmPolicyType, hasRope>::Bmm1SetTensorARope(SplitSameABExtraInfo &extraInfo,
                                                    matmul::MatmulImpl<a1Type, b1Type, T2, bias1Type, CFG_CUSTOM_EXCEED> &bmm1)
{
    int64_t bOffset = 0;
    int64_t s1Offset = 0;
    int64_t n2Offset = 0;
    int64_t gOffset = 0;
    if constexpr (layOutType == LayOutTypeEnum::LAYOUT_SBH) {
        // SBH/SBNGD
        s1Offset = extraInfo.s1oIdx * this->s1BaseBN2GDRope;
        bOffset = extraInfo.boIdx * this->n2GDRope;
        n2Offset = extraInfo.n2oIdx * this->gDRope;
        gOffset = extraInfo.goIdx * dRopeSize;
    } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_TND) {
        bOffset = extraInfo.s1SizeAcc * this->n2GDRope;
        s1Offset = extraInfo.s1oIdx * this->s1BaseN2GDRope;
        n2Offset = extraInfo.n2oIdx * this->gDRope;
        gOffset = extraInfo.goIdx * this->dRopeSize;
    }
    this->qRopeCoreOffset = bOffset + n2Offset + gOffset + s1Offset;
    extraInfo.qRopeCoreOffset = this->qRopeCoreOffset;
    bmm1.SetTensorA(this->queryRopeGm[extraInfo.qRopeCoreOffset]);
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool hasRope>
template <typename T2>
__aicore__ inline void
FlashAttentionScoreS1s2Bn2gs1SameAB<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                 bmm1Format, mmPolicyType, hasRope>::SetBmm1TensorB(SplitSameABExtraInfo &extraInfo,
                                                    matmul::MatmulImpl<a1Type, b1Type, T2, bias1Type, CFG_CUSTOM_EXCEED> &bmm1)
{
    // 计算gm上的offset
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t s2Offset = 0;
    if constexpr (layOutType == LayOutTypeEnum::LAYOUT_BSH) {
        // BSH/BSND
        bOffset = extraInfo.boIdx * this->n2S2D;
        s2Offset = extraInfo.s2StartIdx * this->n2D + extraInfo.s2LoopCount * this->s2BaseNratioN2D;
        n2Offset = extraInfo.n2oIdx * dSize;
    } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_SBH) {
        // SBH/SBND
        s2Offset = extraInfo.s2StartIdx * this->bN2D + extraInfo.s2LoopCount * this->s2BaseNratioBN2D;
        bOffset = extraInfo.boIdx * this->n2D;
        n2Offset = extraInfo.n2oIdx * dSize;
    } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_BNSD) {
        // BNSD
        bOffset = extraInfo.boIdx * this->n2S2D;
        n2Offset = extraInfo.n2oIdx * this->s2D;
        s2Offset = extraInfo.s2StartIdx * dSize + extraInfo.s2LoopCount * this->s2BaseNratioD;
    } else
    if constexpr (layOutType == LayOutTypeEnum::LAYOUT_TND) {
        bOffset = extraInfo.s2SizeAcc * this->n2D;
        s2Offset = extraInfo.s2StartIdx * this->n2D + extraInfo.s2LoopCount * this->s2BaseNratioN2D;
        n2Offset = extraInfo.n2oIdx * this->dSize;
    }
    extraInfo.kCoreOffset = bOffset + n2Offset + s2Offset;
    if constexpr (mmPolicyType != MmPolicyType::UNSPLITK) {
        bmm1.SetTensorB(this->keyGm[extraInfo.kCoreOffset], true);
        bmm1.SetTail(extraInfo.cubeS1RealSize, extraInfo.s2RealSize, dSize);
    }
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool hasRope>
template <typename T2>
__aicore__ inline void
FlashAttentionScoreS1s2Bn2gs1SameAB<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                 bmm1Format, mmPolicyType, hasRope>::SetBmm1TensorBRope(SplitSameABExtraInfo &extraInfo,
                                                    matmul::MatmulImpl<a1Type, b1Type, T2, bias1Type, CFG_CUSTOM_EXCEED> &bmm1)
{
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t s2Offset = 0;
    if constexpr (layOutType == LayOutTypeEnum::LAYOUT_SBH) {
        s2Offset = extraInfo.s2StartIdx * this->bN2DRope + extraInfo.s2LoopCount * this->s2BaseNratioBN2DRope;
        bOffset = extraInfo.boIdx * this->n2DRope;
        n2Offset = extraInfo.n2oIdx * dRopeSize;
    } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_TND) {
        bOffset = extraInfo.s2SizeAcc * this->n2DRope;
        s2Offset = extraInfo.s2StartIdx * this->n2DRope + extraInfo.s2LoopCount * this->s2BaseNratioN2DRope;
        n2Offset = extraInfo.n2oIdx * this->dRopeSize;
    }
    int64_t kRopeCoreOffset = bOffset + n2Offset + s2Offset;
    bmm1.SetTensorB(this->keyRopeGm[kRopeCoreOffset], true);
    bmm1.SetTail(extraInfo.cubeS1RealSize, extraInfo.s2RealSize, dRopeSize);
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool hasRope>
__aicore__ inline void
FlashAttentionScoreS1s2Bn2gs1SameAB<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                bmm1Format, mmPolicyType, hasRope>::NdToNz(SplitSameABExtraInfo &extraInfo, LocalTensor<INPUT_T> &nzResUb,
                                                    LocalTensor<INPUT_T> &vec1ResUb, int64_t loopIdx)
{
    auto nzResUbTmp = nzResUb.template ReinterpretCast<half>();
    auto vec1ResUbTmp = vec1ResUb.template ReinterpretCast<half>();
    int64_t s21 = CeilDiv(extraInfo.s2AlignedSize, 16L);
    int64_t s2InnerLoop = s21 / 8L;
    int64_t s2InnerRemain = s21 % 8L;
    int64_t repeatMaxSize16 = this->repeatMaxBytes / sizeof(INPUT_T);
    CopyRepeatParams repeatParams;
    repeatParams.srcStride = 1;
    repeatParams.dstStride = extraInfo.vec1S1RealSize + 1; // 防止bank冲突，所以+1
    repeatParams.srcRepeatSize = extraInfo.s2RealSizeAlign64 / 16;
    repeatParams.dstRepeatSize = 1;
    int32_t s1OuterLoop = extraInfo.vec1S1RealSize / this->repeatMaxTimes;
    int32_t s1OuterRemain = extraInfo.vec1S1RealSize % this->repeatMaxTimes;
    int32_t s1OuterBmm1Offset = this->repeatMaxTimes * extraInfo.s2RealSizeAlign64;
    int32_t s1OuterTempOffset = this->repeatMaxTimes * 16;
    int64_t vec1Offset = extraInfo.vecCoreOffset * 16 + loopIdx * extraInfo.vec1S1BaseSize * 16;
    int64_t offsetJ = 8 * extraInfo.vec1S1RealSize * 16 + 128;
    AscendC::PipeBarrier<PIPE_V>();
    for (int64_t outerIndex = 0; outerIndex < s1OuterLoop; ++outerIndex) {
        for (int64_t j = 0; j < s2InnerLoop; ++j) {
            Copy(nzResUbTmp[outerIndex * s1OuterTempOffset + j * offsetJ],
                    vec1ResUbTmp[outerIndex * s1OuterBmm1Offset + j * 128],
                    repeatMaxSize16, this->repeatMaxTimes, repeatParams);
        }
        if (s2InnerRemain) {
            Copy(nzResUbTmp[outerIndex * s1OuterTempOffset + s2InnerLoop * offsetJ],
                    vec1ResUbTmp[outerIndex * s1OuterBmm1Offset + s2InnerLoop * 128],
                    s2InnerRemain * 16, this->repeatMaxTimes, repeatParams);
        }
    }

    if (s1OuterRemain) {
        for (int64_t j = 0; j < s2InnerLoop; ++j) {
            Copy(nzResUbTmp[s1OuterLoop * s1OuterTempOffset + j * offsetJ],
                    vec1ResUbTmp[s1OuterLoop * s1OuterBmm1Offset + j * 128],
                    repeatMaxSize16, s1OuterRemain, repeatParams);
        }
        if (s2InnerRemain) {
            Copy(nzResUbTmp[s1OuterLoop * s1OuterTempOffset + s2InnerLoop * offsetJ],
                    vec1ResUbTmp[s1OuterLoop * s1OuterBmm1Offset + s2InnerLoop * 128],
                    s2InnerRemain * 16, s1OuterRemain, repeatParams);
        }
    }

    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    AscendC::SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    AscendC::WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = s21;
    dataCopyParams.blockLen = extraInfo.vec1S1RealSize;
    dataCopyParams.srcStride = 1;
    dataCopyParams.dstStride = CeilDiv(extraInfo.cubeS1RealSize, 16) * 16 - extraInfo.vec1S1RealSize;
    DataCopy(this->stage1Res[extraInfo.taskIdMod2][vec1Offset], nzResUb, dataCopyParams);
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool hasRope>
__aicore__ inline void
FlashAttentionScoreS1s2Bn2gs1SameAB<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                bmm1Format, mmPolicyType, hasRope>::ProcessVec1(SplitSameABExtraInfo &extraInfo)
{
    if (extraInfo.s1RealSize == 0) {
        return;
    }
    LocalTensor<T> stage1PingTensor = this->stage1PingBuf.template Get<T>(); // t.a 32k
    LocalTensor<T> stage1PongTensor = this->stage1PongBuf.template Get<T>(); // i.a 32k
    LocalTensor<T> &actualUseTensor = (extraInfo.needNz2Nd == 0) ? stage1PongTensor : stage1PingTensor;
    LocalTensor<T> commonTBuf = this->commonTBuf.template Get<T>(); // t.b 32k

    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
    event_t eventIdVToMte2A = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    event_t eventIdVToMte2B = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    event_t eventIdVToMte2C = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    event_t eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    event_t eventIdDropMte3ToMte2;
    if constexpr (hasPse == true && hasDrop == true && !IsSameType<T, INPUT_T>::value) {
        eventIdDropMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
    }
    extraInfo.vec1S1RealSize = extraInfo.vec1S1BaseSize;
    for (int32_t loopIdx = 0; loopIdx < extraInfo.realSplitN; ++loopIdx) {
        if (loopIdx == extraInfo.realSplitN - 1) {
            extraInfo.vec1S1RealSize = extraInfo.s1RealSize - loopIdx * extraInfo.vec1S1BaseSize;
        }
        if (loopIdx > 0) {
            AscendC::WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2B);
        } else {
            event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
            AscendC::SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
            AscendC::WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        }

        // FP32场景，需要等待vec1上一轮输出搬完
        if constexpr (IsSameType<INPUT_T, float>::value) {
            event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            AscendC::SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        }
        this->GetBmm1Result(extraInfo, actualUseTensor, loopIdx);

        // mul需要等bmm结果搬完
        AscendC::SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        AscendC::WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

        this->CopyInAttenMask(extraInfo, loopIdx, -1);
        if (this->tilingData->inputParams.pseType != (uint32_t)PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
            AscendC::PipeBarrier<PIPE_V>();
            Muls(stage1PingTensor, actualUseTensor, static_cast<T>(this->tilingData->inputParams.scaleValue),
                 extraInfo.vec1S1RealSize * extraInfo.s2RealSizeAlign64);
        }
        if constexpr (hasPse == true) {
            this->pseInfo.loopIdx = loopIdx;
            this->pseInfo.s2StartIdx = extraInfo.s2StartIdx;
            this->pseInfo.s2LoopCount = extraInfo.s2LoopCount;
            this->pseInfo.bSSOffset = extraInfo.attenB1SSOffset;
            this->pseInfo.n2oIdx = extraInfo.n2oIdx;
            this->pseInfo.s1Size = extraInfo.s1Size;
            this->pseInfo.s2Size = extraInfo.s2Size;
            this->pseInfo.goIdx = extraInfo.goIdx;
            this->pseInfo.s1oIdx = extraInfo.s1oIdx;
            this->pseInfo.vec1S1BaseSize = extraInfo.vec1S1BaseSize;
            this->pseInfo.s2SizeAcc = extraInfo.s2SizeAcc;
            this->pseInfo.boIdx = extraInfo.boIdx;
            this->pseInfo.s2AlignedSize = extraInfo.s2RealSizeAlign64;
            this->pseInfo.vec1S1RealSize = extraInfo.vec1S1RealSize;
            this->pseInfo.s2RealSize = extraInfo.s2RealSize;
            this->pseInfo.needCast = true;
            this->pseInfo.vecCoreOffset = extraInfo.vecCoreOffset;
            bool innerAlibiFlag = false; // alibi核内生成相关配置，仅在LAYOUT=TND，SparseMode=8时生效
            if constexpr (layOutType == LayOutTypeEnum::LAYOUT_TND) {
                if (this->tilingData->inputParams.sparseType ==
                    static_cast<uint8_t>(SparseModeEnum::BAND_LEFT_UP_CAUSAL) && this->pseInfo.boIdx != 0) {
                    innerAlibiFlag = true;
                }
            }

            if (this->pseInfo.pseType == (uint32_t)PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
                this->pseInfo.pseType == (uint32_t)PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
                LocalTensor<half> pseUb = this->pseTBuf.template Get<half>();
                if (innerAlibiFlag) {
                    this->pseInfo.kvStartIdx = 0;
                    this->pseInfo.qStartIdx = 0;
                }
                PseSlopeCopyIn<T, hasPse>(commonTBuf, pseUb, this->pseSlope, this->pseAlibiGm, this->pseInfo, 64);
            } else {
                LocalTensor<INPUT_T> pseUb = this->pseTBuf.template Get<INPUT_T>();
                PseCopyIn<INPUT_T, T, layOutType, hasPse>(commonTBuf, pseUb, this->pseGm, this->pseInfo, 64);
                // FP32场景，需要等PSE输入搬完再启动计算
                if constexpr (IsSameType<INPUT_T, float>::value) {
                    AscendC::SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                    AscendC::WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                }
            }
            AscendC::PipeBarrier<PIPE_V>();
            PseCompute<T, hasPse>(this->tilingData->inputParams.pseType !=
                                  (uint32_t)PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE ? stage1PingTensor : actualUseTensor,
                                  commonTBuf, this->pseInfo);
        }
        if (this->tilingData->inputParams.pseType == (uint32_t)PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
            AscendC::PipeBarrier<PIPE_V>();
            Muls(stage1PingTensor, actualUseTensor, static_cast<T>(this->tilingData->inputParams.scaleValue),
            extraInfo.vec1S1RealSize * extraInfo.s2RealSizeAlign64);
        }
        if constexpr (hasAtten) {
            SelectWithBytesMaskShapeInfo shapeInfo;
            shapeInfo.firstAxis = extraInfo.vec1S1RealSize;
            shapeInfo.srcLastAxis = extraInfo.s2RealSizeAlign64;
            shapeInfo.maskLastAxis = extraInfo.s2RealSizeAlign64;
            stage1PingTensor.SetSize(extraInfo.vec1S1RealSize * extraInfo.s2RealSizeAlign64);
            if (this->attenMaskComputeMode != AttenMaskComputeMode::NO_NEED_COMPUTE_MODE &&
                this->attenMaskComputeMode != AttenMaskComputeMode::PREFIX_COMPUTE_MODE) {
                uint8_t maskType = (this->attenMaskComputeMode == AttenMaskComputeMode::PRE_ONLY_MODE) ? 1 : 0;
                LocalTensor<uint8_t> attenMaskUb = this->maskTBufPing.template Get<uint8_t>();
                this->ComputeAttenMask(shapeInfo, stage1PingTensor, attenMaskUb, maskType, eventIdMte2ToV);
            }

            if (this->attenMaskComputeMode == AttenMaskComputeMode::PRE_AND_NEXT_MODE ||
                this->attenMaskComputeMode == AttenMaskComputeMode::PREFIX_COMPUTE_MODE) {
                event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
                AscendC::SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
                AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
                AscendC::SetFlag<HardEvent::V_MTE2>(eventIdVToMte2C);
                AscendC::WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2C);
                this->CopyInAttenMask(extraInfo, loopIdx, this->attenMaskOffsetPre, true);
                LocalTensor<uint8_t> secondTimeMaskUb;
                uint8_t maskType;
                if (this->attenMaskComputeMode == AttenMaskComputeMode::PREFIX_COMPUTE_MODE) {
                    int32_t maskNum = extraInfo.vec1S1RealSize * extraInfo.s2RealSizeAlign64 / 2; // 除2数据量按照uint16类型折半
                    secondTimeMaskUb = this->maskTBufPing.template Get<uint8_t>();
                    LocalTensor<uint8_t> attenMaskPrefixUb = this->pseTBuf.template Get<uint8_t>();
                    auto attenMaskCasualTmp = secondTimeMaskUb.ReinterpretCast<uint16_t>();
                    auto attenMaskPrefixUbTmp = attenMaskPrefixUb.ReinterpretCast<uint16_t>();
                    AscendC::SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                    AscendC::WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                    And(attenMaskCasualTmp, attenMaskCasualTmp, attenMaskPrefixUbTmp, maskNum);
                    maskType = 0;
                    AscendC::PipeBarrier<PIPE_V>();
                } else {
                    secondTimeMaskUb = this->pseTBuf.template Get<uint8_t>();
                    maskType = 1;
                }
                this->ComputeAttenMask(shapeInfo, stage1PingTensor, secondTimeMaskUb, maskType, eventIdMte2ToV);
            }
        }
        if (loopIdx < extraInfo.realSplitN - 1) {
            AscendC::SetFlag<HardEvent::V_MTE2>(eventIdVToMte2B);
        }

        if (loopIdx > 0) {
            AscendC::WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2A);
        }

        if constexpr (hasDrop == true) {
            LocalTensor<uint8_t> dropMaskUb = this->maskTBufPong.template Get<uint8_t>();
            this->dropMaskInfo.s1InnerIdx = loopIdx;
            this->dropMaskInfo.s2StartIdx = extraInfo.s2StartIdx;
            this->dropMaskInfo.s2Idx = extraInfo.s2LoopCount;
            this->dropMaskInfo.bSSOffset = extraInfo.attenB1SSOffset;
            this->dropMaskInfo.n2OutIdx = extraInfo.n2oIdx;
            this->dropMaskInfo.s1Size = extraInfo.s1Size;
            this->dropMaskInfo.s2Size = extraInfo.s2Size;
            this->dropMaskInfo.gOutIdx = extraInfo.goIdx;
            this->dropMaskInfo.s1OutIdx = extraInfo.s1oIdx;
            this->dropMaskInfo.splitS1BaseSize = extraInfo.vec1S1BaseSize;
            this->dropMaskInfo.s1CopySize = static_cast<uint32_t>(extraInfo.vec1S1RealSize);
            this->dropMaskInfo.s2CopySize = static_cast<uint32_t>(extraInfo.s2RealSize);
            this->dropMaskInfo.s2TotalSize = extraInfo.s2Size;
            this->dropMaskInfo.boolMode = this->dropMaskUnAligned;
            this->dropMaskInfo.vecCoreOffset = extraInfo.vecCoreOffset;
            if constexpr (hasPse == true && !IsSameType<T, INPUT_T>::value) {
                if (loopIdx > 0) {
                    AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventIdDropMte3ToMte2);
                }
            }
            CopyInDropMask<hasDrop>(dropMaskUb, this->dropMaskGm, this->dropMaskGm, this->dropMaskInfo, 64);
        }

        this->SoftMaxCompute(extraInfo, stage1PingTensor, loopIdx);

        if constexpr (hasDrop == true) {
            LocalTensor<uint8_t> apiTmpBuffer = this->commonTBuf.template Get<uint8_t>();
            LocalTensor<uint8_t> dropMaskUb = this->maskTBufPong.template Get<uint8_t>();
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            AscendC::WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            this->dropMaskInfo.firstAxis = static_cast<uint32_t>(extraInfo.vec1S1RealSize);
            this->dropMaskInfo.lstAxis = static_cast<uint32_t>(extraInfo.s2RealSizeAlign64);
            this->dropMaskInfo.maskLstAxis = this->dropMaskInfo.lstAxis;
            this->dropMaskInfo.keepProb = this->tilingData->inputParams.keepProb;
            ComputeDropMask<T, hasDrop>(stage1PingTensor, stage1PingTensor, dropMaskUb, apiTmpBuffer,
                                        this->dropMaskInfo);
        }

        if (loopIdx < extraInfo.realSplitN - 1) {
            AscendC::SetFlag<HardEvent::V_MTE2>(eventIdVToMte2A);
        }

        if (loopIdx > 0) {
            AscendC::WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        }
        AscendC::PipeBarrier<PIPE_V>();
        if constexpr (!IsSameType<T, INPUT_T>::value) {
            LocalTensor<INPUT_T> stage1CastTensor;
            if constexpr (hasPse == true) {
                stage1CastTensor = this->maskTBufPong.template Get<INPUT_T>();
            } else {
                stage1CastTensor = this->pseTBuf.template Get<INPUT_T>();
            }
            Cast(stage1CastTensor, stage1PingTensor, RoundMode::CAST_ROUND,
                 extraInfo.vec1S1RealSize * extraInfo.s2RealSizeAlign64);
            AscendC::SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            AscendC::WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            LocalTensor<INPUT_T> stage1NzTensor = this->stage2TBuf.template Get<INPUT_T>();
            NdToNz(extraInfo, stage1NzTensor, stage1CastTensor, loopIdx);

            if constexpr (hasPse == true && hasDrop == true && !IsSameType<T, INPUT_T>::value) {
                if (loopIdx < extraInfo.realSplitN - 1) {
                    AscendC::SetFlag<HardEvent::MTE3_MTE2>(eventIdDropMte3ToMte2);
                }
            }
        } else {
            AscendC::SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            AscendC::WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            DataCopyParams dataCopyParams;
            dataCopyParams.dstStride = 0;
            if (extraInfo.s2LastLoop) {
                dataCopyParams.blockCount = extraInfo.vec1S1RealSize;
                dataCopyParams.blockLen = extraInfo.s2AlignedSize / 8;
                dataCopyParams.srcStride = (extraInfo.s2RealSizeAlign64 - extraInfo.s2AlignedSize) / 8;
            } else {
                dataCopyParams.blockCount = 1;
                dataCopyParams.blockLen = extraInfo.vec1S1RealSize * extraInfo.s2AlignedSize / 8;
                dataCopyParams.srcStride = 0;
            }
            DataCopy(
                this->stage1Res[extraInfo.taskIdMod2][(extraInfo.vecCoreOffset +
                                                       loopIdx * extraInfo.vec1S1BaseSize) * extraInfo.s2AlignedSize],
                stage1PingTensor, dataCopyParams);
        }

        if (loopIdx < extraInfo.realSplitN - 1) {
            AscendC::SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        }
    }
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIdMte2ToV);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte2A);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte2B);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte2C);
    if constexpr (hasPse == true && hasDrop == true && !IsSameType<T, INPUT_T>::value) {
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(eventIdDropMte3ToMte2);
    }
    return;
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool hasRope>
__aicore__ inline void
FlashAttentionScoreS1s2Bn2gs1SameAB<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                bmm1Format, mmPolicyType, hasRope>::CopyInAttenMask(SplitSameABExtraInfo &extraInfo, int64_t loopIdx,
                                                             int64_t maskOffset, bool secondTime)
{
    if constexpr (hasAtten == true) {
        LocalTensor<uint8_t> attenMaskUb;
        if (secondTime) {
            attenMaskUb = this->pseTBuf.template Get<uint8_t>();
        } else {
            attenMaskUb = this->maskTBufPing.template Get<uint8_t>();
        }
        if (maskOffset == -1) {
            maskOffset = this->ComputeAttenMaskOffset(extraInfo, loopIdx);
        }
        if (this->attenMaskComputeMode == AttenMaskComputeMode::NO_NEED_COMPUTE_MODE) {
            return;
        }
        if (this->attenMaskComputeMode == AttenMaskComputeMode::PRE_ONLY_MODE ||
            this->attenMaskComputeMode == AttenMaskComputeMode::PREFIX_N_COMPUTE_MODE) {
            maskOffset = this->attenMaskOffsetPre;
        }

        int64_t s2StrideSize = this->tilingData->inputParams.attenMaskS2Size;
        if constexpr (layOutType == LayOutTypeEnum::LAYOUT_TND) {
            if (this->tilingData->inputParams.attenMaskShapeType == attenMaskS1S2) {
                s2StrideSize = this->tilingData->inputParams.s2Size;
            } else if (this->tilingData->inputParams.attenMaskShapeType == attenMaskTT) {
                s2StrideSize = this->s2SizeSum;
            }
            // band compress mode
            if (this->tilingData->inputParams.attenMaskCompressMode !=
                static_cast<uint8_t>(AttenMaskCompressMode::NO_COMPRESS_MODE)) {
                s2StrideSize = this->tilingData->inputParams.attenMaskS2Size;
            }
        }
        BoolCopyIn(attenMaskUb, this->attenMaskGmInt, maskOffset, extraInfo.vec1S1RealSize, extraInfo.s2RealSize,
                   s2StrideSize, 64);
        return;
    }
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool hasRope>
__aicore__ inline int64_t
FlashAttentionScoreS1s2Bn2gs1SameAB<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                bmm1Format, mmPolicyType, hasRope>::ComputeAttenMaskOffset(SplitSameABExtraInfo &extraInfo, int64_t loopIdx)
{
    if constexpr (hasAtten == true) {
        if (this->tilingData->inputParams.attenMaskCompressMode ==
            static_cast<uint8_t>(AttenMaskCompressMode::NO_COMPRESS_MODE)) {
            return this->ComputeOffsetForNoCompress(extraInfo, loopIdx);
        }
        if constexpr (layOutType == LayOutTypeEnum::LAYOUT_TND) {
            // compress mode
            int64_t delta = 0;
            int64_t deltaPre = 0;
            int64_t deltaN = static_cast<int64_t>((extraInfo.s1Size)) - static_cast<int64_t>((extraInfo.s2Size));
            int64_t s1Offset = extraInfo.s1oIdx * this->cubeS1BaseSize + extraInfo.vecCoreOffset +
                               loopIdx * extraInfo.vec1S1BaseSize;
            int64_t s2Offset = extraInfo.s2StartIdx + extraInfo.s2LoopCount * this->s2BaseNratioSize;
            if (this->tilingData->inputParams.attenMaskCompressMode ==
                static_cast<uint8_t>(AttenMaskCompressMode::LEFT_UP_CAUSAL_MODE)) {
                delta = s1Offset - s2Offset;
            } else if (this->tilingData->inputParams.attenMaskCompressMode ==
                       static_cast<uint8_t>(AttenMaskCompressMode::RIGHT_DOWN_CAUSAL_MODE)) {
                delta = s1Offset - s2Offset - deltaN;
            } else if (this->tilingData->inputParams.attenMaskCompressMode ==
                       static_cast<uint8_t>(AttenMaskCompressMode::BAND_MODE)) {
                int64_t tmpPre = this->tilingData->inputParams.preTokens;
                int64_t tmpNext = this->tilingData->inputParams.nextTokens;
                int64_t transPreTokens = extraInfo.s1Size - Max(extraInfo.s2Size - tmpPre, 0);
                int64_t transNextTokens = extraInfo.s2Size - Max(extraInfo.s1Size - tmpNext, 0);
                deltaPre = s1Offset - s2Offset - transPreTokens - 1;
                int64_t maskOffsetPre =
                    ComputeOffsetForCausal(deltaPre, extraInfo.vec1S1BaseSize, this->s2BaseNratioSize,
                                           this->tilingData->inputParams.attenMaskS2Size);
                this->attenMaskOffsetPre = maskOffsetPre; // save offset value for the 2nd mask operation.
                delta = s1Offset - s2Offset + transNextTokens;
            } else if (this->tilingData->inputParams.attenMaskCompressMode ==
                       static_cast<uint8_t>(AttenMaskCompressMode::RIGHT_DOWN_CAUSAL_BAND_MODE)) {
                if (extraInfo.boIdx == this->tilingData->inputParams.bandIndex) {
                    delta = s1Offset - s2Offset - deltaN + this->tilingData->inputParams.nextTokens;
                } else {
                    delta = s1Offset - s2Offset - deltaN;
                }
            } else if (this->tilingData->inputParams.attenMaskCompressMode ==
                       static_cast<uint8_t>(AttenMaskCompressMode::BAND_LEFT_UP_CAUSAL_MODE)) {
                if (extraInfo.boIdx == this->tilingData->inputParams.bandIndex) {
                    delta = s1Offset - s2Offset + extraInfo.s2Size -
                            Max(extraInfo.s1Size - this->tilingData->inputParams.nextTokens, 0);
                } else {
                    delta = s1Offset - s2Offset;
                }
            } else if (this->tilingData->inputParams.attenMaskCompressMode ==
                       static_cast<uint8_t>(AttenMaskCompressMode::PREFIX_MODE)) {
                delta = s1Offset - s2Offset - deltaN;
                if ((extraInfo.s1Size + ((__gm__ int64_t *)this->prefixNAddr)[extraInfo.boIdx]) > extraInfo.s2Size) {
                    // prefix reuse attenMaskOffsetPre
                    deltaPre = ((__gm__ int64_t *)this->prefixNAddr)[extraInfo.boIdx] - extraInfo.s2StartIdx -
                               extraInfo.s2LoopCount * this->s2BaseNratioSize;
                    this->attenMaskOffsetPre = ComputeOffsetForPrefixRectangle(
                        deltaPre, this->s2BaseNratioSize, this->tilingData->inputParams.attenMaskS2Size);
                    if (this->vecBlockIdx + extraInfo.vec1S1RealSize < prefixAttenMaskDownHeight) { // in case of out of bound
                        this->attenMaskOffsetPre += this->tilingData->inputParams.attenMaskS2Size * this->vecBlockIdx;
                    }
                }
            } else {
                return 0;
            }
            this->GetAttenMaskComputeMode(delta, deltaPre, s1Offset, extraInfo);
            return ComputeOffsetForCausal(delta, extraInfo.vec1S1BaseSize, this->s2BaseNratioSize,
                                          this->tilingData->inputParams.attenMaskS2Size);
        }
        // compress mode
        int64_t deltaCausalOrNext = 0;
        int64_t deltaPre = 0;
        int64_t deltaN = extraInfo.s1Size - extraInfo.s2Size;
        int64_t s1Offset = extraInfo.s1oIdx * this->cubeS1BaseSize + extraInfo.vecCoreOffset +
                           loopIdx * extraInfo.vec1S1BaseSize;
        int64_t s2Offset = extraInfo.s2StartIdx + extraInfo.s2LoopCount * this->s2BaseNratioSize;
        if (this->tilingData->inputParams.attenMaskCompressMode ==
            static_cast<uint8_t>(AttenMaskCompressMode::LEFT_UP_CAUSAL_MODE)) {
            deltaCausalOrNext = s1Offset - s2Offset;
        } else if (this->tilingData->inputParams.attenMaskCompressMode ==
                   static_cast<uint8_t>(AttenMaskCompressMode::RIGHT_DOWN_CAUSAL_MODE)) {
            deltaCausalOrNext = s1Offset - s2Offset - deltaN;
        } else if (this->tilingData->inputParams.attenMaskCompressMode ==
                   static_cast<uint8_t>(AttenMaskCompressMode::BAND_MODE)) {
            deltaPre = s1Offset - s2Offset - this->tilingData->inputParams.preTokens - 1;
            this->attenMaskOffsetPre =
                ComputeOffsetForCausal(deltaPre, extraInfo.vec1S1BaseSize, this->s2BaseNratioSize,
                                       this->tilingData->inputParams.attenMaskS2Size);
            deltaCausalOrNext = s1Offset - s2Offset + this->tilingData->inputParams.nextTokens;
        } else if (this->tilingData->inputParams.attenMaskCompressMode ==
                   static_cast<uint8_t>(AttenMaskCompressMode::PREFIX_MODE)) {
            deltaCausalOrNext = s1Offset - s2Offset - deltaN;
            if ((extraInfo.s1Size + ((__gm__ int64_t *)this->prefixNAddr)[extraInfo.boIdx]) > extraInfo.s2Size) {
                // prefix reuse attenMaskOffsetPre
                deltaPre = ((__gm__ int64_t *)this->prefixNAddr)[extraInfo.boIdx] - extraInfo.s2StartIdx -
                           extraInfo.s2LoopCount * this->s2BaseNratioSize;
                this->attenMaskOffsetPre = ComputeOffsetForPrefixRectangle(
                    deltaPre, this->s2BaseNratioSize, this->tilingData->inputParams.attenMaskS2Size);
                if (this->vecBlockIdx + extraInfo.vec1S1RealSize < prefixAttenMaskDownHeight) { // in case of out of bound
                    this->attenMaskOffsetPre += this->tilingData->inputParams.attenMaskS2Size * this->vecBlockIdx;
                }
            }
        } else {
            return 0;
        }
        this->GetAttenMaskComputeMode(deltaCausalOrNext, deltaPre, s1Offset, extraInfo);
        return ComputeOffsetForCausal(deltaCausalOrNext, extraInfo.vec1S1BaseSize, s2BaseNratioSize,
                                      this->tilingData->inputParams.attenMaskS2Size);
    }
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool hasRope>
__aicore__ inline void
FlashAttentionScoreS1s2Bn2gs1SameAB<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                bmm1Format, mmPolicyType, hasRope>::GetAttenMaskComputeMode(int64_t deltaCausalOrNext, int64_t deltaPre,
                                                                     int64_t s1Offset, SplitSameABExtraInfo &extraInfo)
{
    if constexpr (hasAtten == true) {
        int64_t causalOrNextFactor = deltaCausalOrNext - extraInfo.s2AlignedSize;
        if (this->tilingData->inputParams.attenMaskCompressMode ==
                static_cast<uint8_t>(AttenMaskCompressMode::LEFT_UP_CAUSAL_MODE) ||
            this->tilingData->inputParams.attenMaskCompressMode ==
                static_cast<uint8_t>(AttenMaskCompressMode::RIGHT_DOWN_CAUSAL_MODE)) {
            if (causalOrNextFactor >= 0) {
                this->attenMaskComputeMode = AttenMaskComputeMode::NO_NEED_COMPUTE_MODE;
            } else {
                this->attenMaskComputeMode = AttenMaskComputeMode::CAUSAL_OR_NEXT_ONLY_MODE;
            }
            return;
        }
        if (this->tilingData->inputParams.attenMaskCompressMode ==
            static_cast<uint8_t>(AttenMaskCompressMode::BAND_MODE)) {
            int64_t preFactor = deltaPre + 1 + extraInfo.vec1S1BaseSize;
            if (causalOrNextFactor >= 0 && preFactor <= 0) {
                this->attenMaskComputeMode = AttenMaskComputeMode::NO_NEED_COMPUTE_MODE;
            } else if (causalOrNextFactor < 0 && preFactor <= 0) {
                this->attenMaskComputeMode = AttenMaskComputeMode::CAUSAL_OR_NEXT_ONLY_MODE;
            } else if (causalOrNextFactor >= 0 && preFactor > 0) {
                this->attenMaskComputeMode = AttenMaskComputeMode::PRE_ONLY_MODE;
            } else {
                this->attenMaskComputeMode = AttenMaskComputeMode::PRE_AND_NEXT_MODE;
            }
        }
        if (this->tilingData->inputParams.attenMaskCompressMode ==
            static_cast<uint8_t>(AttenMaskCompressMode::PREFIX_MODE)) {
            int64_t preFactor = deltaPre - extraInfo.s2AlignedSize;
            // Triangular part and rectangular part have one is not counted, then the whole is not counted,
            // otherwise it needs to be calculated
            if (causalOrNextFactor >= 0 || preFactor >= 0 || deltaPre > extraInfo.s2Size) {
                // attenmask value is all 0, no need to compute
                this->attenMaskComputeMode = AttenMaskComputeMode::NO_NEED_COMPUTE_MODE;
            } else {
                int64_t intersectionX = extraInfo.s1Size - extraInfo.s2Size +
                    ((__gm__ int64_t *)this->prefixNAddr)[extraInfo.boIdx];
                if (s1Offset >= intersectionX) {
                    this->attenMaskComputeMode = AttenMaskComputeMode::CAUSAL_OR_NEXT_ONLY_MODE;
                } else if (s1Offset + extraInfo.vec1S1BaseSize <= intersectionX) {
                    this->attenMaskComputeMode = AttenMaskComputeMode::PREFIX_N_COMPUTE_MODE;
                } else {
                    this->attenMaskComputeMode = AttenMaskComputeMode::PREFIX_COMPUTE_MODE;
                }
            }
        }
        return;
    }
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool hasRope>
__aicore__ inline int64_t
FlashAttentionScoreS1s2Bn2gs1SameAB<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                            bmm1Format, mmPolicyType, hasRope>::ComputeOffsetForNoCompress(SplitSameABExtraInfo &extraInfo, int64_t loopIdx)
{
    if constexpr (hasAtten == true) {
        int64_t bOffset = 0;
        int64_t n2Offset = 0;
        int64_t gOffset = 0;
        int64_t s1Offset = (extraInfo.s1oIdx * this->cubeS1BaseSize + extraInfo.vecCoreOffset +
                            loopIdx * extraInfo.vec1S1BaseSize) * extraInfo.s2Size;
        int64_t s2Offset = extraInfo.s2StartIdx + extraInfo.s2LoopCount * s2BaseNratioSize;
        if (this->tilingData->inputParams.attenMaskShapeType == attenMaskBN2GS1S2) {
            bOffset = extraInfo.attenB1SSOffset * this->n2G;
            n2Offset = extraInfo.n2oIdx * this->tilingData->inputParams.gSize * extraInfo.s1Size * extraInfo.s2Size;
            gOffset = extraInfo.goIdx * extraInfo.s1Size * extraInfo.s2Size;
        } else if (this->tilingData->inputParams.attenMaskShapeType == attenMaskBS1S2) {
            bOffset = extraInfo.attenB1SSOffset;
        } else if (this->tilingData->inputParams.attenMaskShapeType == attenMaskS1S2) {
            s1Offset = (extraInfo.s1oIdx * this->cubeS1BaseSize + extraInfo.vecCoreOffset +
                       loopIdx * extraInfo.vec1S1BaseSize) * this->tilingData->inputParams.s2Size;
        } else if (this->tilingData->inputParams.attenMaskShapeType == attenMaskTT) {
            s1Offset = extraInfo.s1SizeAcc + extraInfo.s1oIdx * this->cubeS1BaseSize + extraInfo.vecCoreOffset +
                       loopIdx * extraInfo.vec1S1BaseSize;
            s1Offset = s1Offset * this->s2SizeSum;
            s2Offset = s2Offset + extraInfo.s2SizeAcc;
        }
        return bOffset + n2Offset + gOffset + s1Offset + s2Offset;
    }
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool hasRope>
__aicore__ inline void
FlashAttentionScoreS1s2Bn2gs1SameAB<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                    bmm1Format, mmPolicyType, hasRope>::GetBmm1Result(SplitSameABExtraInfo &extraInfo,
                                               LocalTensor<T> &bmm1ResUb, int64_t loopIdx)
{
    if constexpr (bmm1Format == CubeFormat::NZ) {
        if (extraInfo.needNz2Nd == 1) {
            Nz2NdInfo nz2NdInfo;
            nz2NdInfo.ndFirstAxisRealSize = extraInfo.cubeS1RealSize;
            nz2NdInfo.ndFirstAxisBaseSize = extraInfo.vec1S1BaseSize;
            nz2NdInfo.ndFirstAxisLoopSize = extraInfo.vec1S1RealSize;
            nz2NdInfo.ndLastAxis = extraInfo.s2RealSizeAlign64;
            nz2NdInfo.loopIdx = loopIdx;
            nz2NdInfo.vecCoreOffset = extraInfo.vecCoreOffset;
            LocalTensor<T> tempUb = this->stage1PongBuf.template Get<T>();
            NzToNd(nz2NdInfo, this->mm1Res[extraInfo.taskIdMod2], tempUb, bmm1ResUb);
            return;
        }
    }
    if (likely(extraInfo.s2RealSizeAlign64 == extraInfo.s2RealSize)) {
        DataCopy2D(bmm1ResUb,
                   this->mm1Res[extraInfo.taskIdMod2][(extraInfo.vecCoreOffset + loopIdx * extraInfo.vec1S1BaseSize) *
                                                      extraInfo.s2RealSize],
                   extraInfo.vec1S1RealSize, extraInfo.s2RealSize, extraInfo.s2RealSize);
    } else {
        DataCopyParams dataCopyParams;
        DataCopyPadParams dataCopyPadParams;
        dataCopyParams.blockCount = extraInfo.vec1S1RealSize;
        dataCopyParams.blockLen = extraInfo.s2RealSize * sizeof(T);
        dataCopyParams.srcStride = 0;
        if (extraInfo.s2LastLoop) {
            dataCopyParams.dstStride = (extraInfo.s2RealSizeAlign64 - extraInfo.s2RealSize) / 8;
            dataCopyPadParams.isPad = false;
        } else {
            dataCopyParams.dstStride = 0;
            dataCopyPadParams.isPad = true;
            dataCopyPadParams.rightPadding = extraInfo.s2AlignedSize - extraInfo.s2RealSize;
            if (dataCopyPadParams.rightPadding > blockSize) {
                dataCopyPadParams.rightPadding -= blockSize;
                dataCopyParams.dstStride = 1;
                int32_t s2BlockAlignedSize = CeilDiv(extraInfo.s2RealSize, blockSize) * blockSize;
                Duplicate<T>(bmm1ResUb[s2BlockAlignedSize], 0, blockSize, extraInfo.vec1S1RealSize, 0,
                            extraInfo.s2AlignedSize * sizeof(T) / blockBytes);
            }
            dataCopyPadParams.paddingValue = 0;
        }
        DataCopyPad(bmm1ResUb,
                    this->mm1Res[extraInfo.taskIdMod2][(extraInfo.vecCoreOffset + loopIdx * extraInfo.vec1S1BaseSize) *
                                                       extraInfo.s2RealSize], dataCopyParams, dataCopyPadParams);
    }
    uint32_t bmm1ResUbShape[] = {static_cast<uint32_t>(extraInfo.vec1S1RealSize),
                                 static_cast<uint32_t>(extraInfo.s2RealSizeAlign64)};
    bmm1ResUb.SetShapeInfo(ShapeInfo(2, bmm1ResUbShape, DataFormat::ND));
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool hasRope>
__aicore__ inline void
FlashAttentionScoreS1s2Bn2gs1SameAB<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                bmm1Format, mmPolicyType, hasRope>::ComputeAttenMask(SelectWithBytesMaskShapeInfo &shapeInfo,
                                                              LocalTensor<T> &bmm1ResUb,
                                                              LocalTensor<uint8_t> &attenMaskUb,
                                                              const uint8_t maskType, event_t vWaitMte2)
{
    if constexpr (hasAtten == true) {
        LocalTensor<uint8_t> apiTmpBuffer = commonTBuf.template Get<uint8_t>();
        attenMaskUb.SetSize(shapeInfo.firstAxis * shapeInfo.maskLastAxis);
        AscendC::SetFlag<HardEvent::MTE2_V>(vWaitMte2);
        AscendC::WaitFlag<HardEvent::MTE2_V>(vWaitMte2);
        event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        AscendC::SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        AscendC::WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        if (maskType == 0) {
            SelectWithBytesMask(bmm1ResUb, bmm1ResUb, this->negativeFloatScalar, attenMaskUb, apiTmpBuffer, shapeInfo);
        } else {
            SelectWithBytesMask(bmm1ResUb, this->negativeFloatScalar, bmm1ResUb, attenMaskUb, apiTmpBuffer, shapeInfo);
        }
        return;
    }
}


template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool hasRope>
__aicore__ inline void
FlashAttentionScoreS1s2Bn2gs1SameAB<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                            bmm1Format, mmPolicyType, hasRope>::SoftMaxCheckResCompress(SplitSameABExtraInfo &extraInfo,
                                                                 int64_t vec1S1realSplitN)
{
    if constexpr (implMode == ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION) {
        if (unlikely(extraInfo.realSplitN == 1)) {
            bool res = IsIncludeInvalidLine(this->softMaxCheckRes, vec1S1realSplitN);
            UpdateSoftMaxCheckRes(this->softMaxCheckRes, 0, res);
        } else {
            int64_t avg = CeilDiv(vec1S1realSplitN, extraInfo.realSplitN);
            for (int64_t i = 0; i < extraInfo.realSplitN; ++i) {
                int64_t endIdx = Min((i + 1) * avg, vec1S1realSplitN) - 1;
                if (IsIncludeInvalidLine(this->softMaxCheckRes, endIdx, i * avg)) {
                    UpdateSoftMaxCheckRes(this->softMaxCheckRes, i, true);
                } else {
                    UpdateSoftMaxCheckRes(this->softMaxCheckRes, i, false);
                }
            }
        }
    }
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool hasRope>
__aicore__ inline void
FlashAttentionScoreS1s2Bn2gs1SameAB<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                bmm1Format, mmPolicyType, hasRope>::InvalidLineSplitS2Process(SplitSameABExtraInfo &extraInfo,
                                                                       LocalTensor<T> &srcTensor,
                                                                       LocalTensor<T> &maxUb, int64_t loopIdx)
{
    if constexpr (implMode == ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION) {
        if (loopIdx == 0 && extraInfo.s2LoopCount == extraInfo.s2LoopLimit) {
            int64_t vec1S1BaseSize = Min(8, extraInfo.s1RealSize);
            int64_t vec1S1realSplitN = CeilDiv(extraInfo.s1RealSize, vec1S1BaseSize);
            if (extraInfo.realSplitN != vec1S1realSplitN) {
                this->SoftMaxCheckResCompress(extraInfo, vec1S1realSplitN);
            }
        }

        if (hasInvalidLine(this->softMaxCheckRes, loopIdx)) {
            SoftMaxShapeInfo softmaxShapeInfo{
                static_cast<uint32_t>(extraInfo.vec1S1RealSize), static_cast<uint32_t>(extraInfo.s2RealSizeAlign64),
                static_cast<uint32_t>(extraInfo.vec1S1RealSize), static_cast<uint32_t>(extraInfo.s2RealSize)};
            bool res = false;
            if (this->softmaxReduceSize == 1) {
                res = AdjustSoftMaxRes<T, T, false, 1>(srcTensor, maxUb, this->negativeIntScalar,
                                                       0.0, softmaxShapeInfo);
            } else {
                res = AdjustSoftMaxRes<T, T>(srcTensor, maxUb, this->negativeIntScalar, 0.0, softmaxShapeInfo);
            }
            UpdateSoftMaxCheckRes(this->softMaxCheckRes, loopIdx, res);
        }

        if (loopIdx == extraInfo.realSplitN - 1 && extraInfo.s2LoopCount == extraInfo.s2LoopLimit &&
            IsIncludeInvalidLine(this->softMaxCheckRes, extraInfo.realSplitN)) {
            LocalTensor<T> maxTensor = this->softmaxMaxBuf.template Get<T>();
            LocalTensor<T> sumTensor;
            sumTensor = this->softmaxSumBuf[extraInfo.multiCoreInnerIdxMod2].template Get<T>();

            SoftMaxShapeInfo softmaxShapeInfo{
                static_cast<uint32_t>(extraInfo.s1RealSize), static_cast<uint32_t>(fp32BaseSize),
                static_cast<uint32_t>(extraInfo.s1RealSize), static_cast<uint32_t>(fp32BaseSize)};
            if (this->softmaxReduceSize == 1) {
                int32_t calcCnt = (extraInfo.s1RealSize * 4 - 1 + 256) / 256 * 256 / 4;
                LocalTensor<uint8_t> maskTensor = this->softmaxTempBuf.template Get<uint8_t>();
                CompareScalar(maskTensor, maxTensor, static_cast<T>(this->negativeFloatScalar),
                              AscendC::CMPMODE::NE, calcCnt);
                Select(sumTensor, maskTensor, sumTensor, static_cast<T>(this->positiveFloatScalar),
                       AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, calcCnt);
            } else {
                AdjustSoftMaxRes<T, T>(sumTensor, maxTensor, this->negativeIntScalar,
                                       this->positiveFloatScalar, softmaxShapeInfo);
            }
        }
    }
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool hasRope>
__aicore__ inline void
FlashAttentionScoreS1s2Bn2gs1SameAB<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                            bmm1Format, mmPolicyType, hasRope>::SoftMaxCompute(SplitSameABExtraInfo &extraInfo,
                                                        LocalTensor<T> &srcTensor, int64_t loopIdx)
{
    int64_t vec1S1RealSize = CeilDiv(extraInfo.vec1S1RealSize, 8) * 8;
    uint32_t bmm1ResUbShape[] = {static_cast<uint32_t>(vec1S1RealSize),
                                 static_cast<uint32_t>(extraInfo.s2RealSizeAlign64)};
    uint32_t bmm1ResUbOrgShape[] = {static_cast<uint32_t>(vec1S1RealSize),
                                    static_cast<uint32_t>(extraInfo.s2RealSizeAlign64)};
    srcTensor.SetShapeInfo(ShapeInfo(2, bmm1ResUbShape, 2, bmm1ResUbOrgShape, DataFormat::ND));
    srcTensor.SetSize(vec1S1RealSize * extraInfo.s2RealSizeAlign64);

    uint32_t maxSumShape[] = {static_cast<uint32_t>(extraInfo.vec1S1RealSize),
                              static_cast<uint32_t>(this->softmaxReduceSize)};
    LocalTensor<T> sumUb = this->softmaxSumBuf[extraInfo.multiCoreInnerIdxMod2]
                           .template Get<T>()[loopIdx * extraInfo.vec1S1BaseSize * this->softmaxReduceSize];
    LocalTensor<T> maxUb = this->softmaxMaxBuf
                           .template Get<T>()[loopIdx * extraInfo.vec1S1BaseSize * this->softmaxReduceSize];

    sumUb.SetShapeInfo(ShapeInfo(2, maxSumShape, DataFormat::ND));
    maxUb.SetShapeInfo(ShapeInfo(2, maxSumShape, DataFormat::ND));
    if (extraInfo.s2LastLoop && extraInfo.s2RealSize % 64 != 0) {
        uint64_t mask[2] = {extraInfo.duplicateMask, 0};
        Duplicate<T>(srcTensor[extraInfo.s2RealSizeFloorAlign8], this->negativeFloatScalar, mask, vec1S1RealSize,
                    1, extraInfo.s2RealSizeAlign64 * sizeof(T) / blockBytes);
    }

    LocalTensor<T> expUb = this->softmaxExpBuf[extraInfo.taskIdMod2]
                           .template Get<T>()[loopIdx * extraInfo.vec1S1BaseSize * this->softmaxReduceSize];

    expUb.SetShapeInfo(ShapeInfo(2, maxSumShape, DataFormat::ND));
    LocalTensor<uint8_t> apiTmpBuffer = this->commonTBuf.template Get<uint8_t>();
    AscendC::PipeBarrier<PIPE_V>();
    if (unlikely(extraInfo.s2LoopCount == 0)) {
        if (this->softmaxReduceSize == 1) {
            SoftMaxTiling newTiling = AscendC::SoftMaxFlashV2TilingFuncImpl(vec1S1RealSize,
                                                                            extraInfo.s2RealSizeAlign64, sizeof(T),
                                                                            sizeof(T),
                                                                            apiTmpBuffer.GetSize() / sizeof(T),
                                                                            false, true, false, true);
            SoftmaxFlashV2<T, false, true, true, false, SOFTMAX_REDUCE_CFG>(srcTensor, sumUb, maxUb, srcTensor, expUb,
                                                                            sumUb, maxUb, apiTmpBuffer, newTiling);
        } else {
            SoftMaxTiling newTiling = AscendC::SoftMaxFlashV2TilingFuncImpl(vec1S1RealSize,
                                                                            extraInfo.s2RealSizeAlign64, sizeof(T),
                                                                            sizeof(T),
                                                                            apiTmpBuffer.GetSize() / sizeof(T),
                                                                            false, true);
            SoftmaxFlashV2<T, false, true, true, false, SOFTMAX_DEFAULT_CFG>(srcTensor, sumUb, maxUb, srcTensor, expUb,
                                                                             sumUb, maxUb, apiTmpBuffer, newTiling);
        }
    } else {
        if (this->softmaxReduceSize == 1) {
                SoftMaxTiling newTiling = AscendC::SoftMaxFlashV2TilingFuncImpl(vec1S1RealSize,
                                                                                extraInfo.s2RealSizeAlign64, sizeof(T),
                                                                                sizeof(T),
                                                                                apiTmpBuffer.GetSize() / sizeof(T),
                                                                                true, true, false, true);
                SoftmaxFlashV2<T, true, true, true, false, SOFTMAX_REDUCE_CFG>(srcTensor, sumUb, maxUb, srcTensor, expUb,
                                                                               sumUb, maxUb, apiTmpBuffer, newTiling);
        } else {
            SoftMaxTiling newTiling = AscendC::SoftMaxFlashV2TilingFuncImpl(vec1S1RealSize,
                                                                            extraInfo.s2RealSizeAlign64, sizeof(T),
                                                                            sizeof(T),
                                                                            apiTmpBuffer.GetSize() / sizeof(T),
                                                                            true, true);
            SoftmaxFlashV2<T, true, true, true, false, SOFTMAX_DEFAULT_CFG>(srcTensor, sumUb, maxUb, srcTensor, expUb,
                                                                            sumUb, maxUb, apiTmpBuffer, newTiling);
        }
    }

    if constexpr (implMode == ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION) {
        if (extraInfo.s2Size > SPLIT_S2_SIZE_LIMIT) {
            this->InvalidLineSplitS2Process(extraInfo, srcTensor, maxUb, loopIdx);
        } else {
            SoftMaxShapeInfo softmaxShapeInfo{
                static_cast<uint32_t>(vec1S1RealSize), static_cast<uint32_t>(extraInfo.s2RealSizeAlign64),
                static_cast<uint32_t>(vec1S1RealSize), static_cast<uint32_t>(extraInfo.s2RealSizeAlign64)};
            if (this->softmaxReduceSize == 1) {
                AdjustSoftMaxRes<T, T, false, 1>(srcTensor, maxUb, this->negativeIntScalar, 0.0, softmaxShapeInfo);
            } else {
                AdjustSoftMaxRes<T, T>(srcTensor, maxUb, this->negativeIntScalar, 0.0, softmaxShapeInfo);
            }
            if (loopIdx == extraInfo.realSplitN - 1 && extraInfo.s2LoopCount == extraInfo.s2LoopLimit) {
                LocalTensor<T> maxTensor = this->softmaxMaxBuf.template Get<T>();
                LocalTensor<T> sumTensor;
                sumTensor = this->softmaxSumBuf[extraInfo.multiCoreInnerIdxMod2].template Get<T>();

                SoftMaxShapeInfo softmaxFullShapeInfo{
                    static_cast<uint32_t>(extraInfo.s1RealSize), static_cast<uint32_t>(fp32BaseSize),
                    static_cast<uint32_t>(extraInfo.s1RealSize), static_cast<uint32_t>(fp32BaseSize)};
                if (this->softmaxReduceSize == 1) {
                    int32_t calcCnt = (extraInfo.s1RealSize * 4 - 1 + 256) / 256 * 256 / 4;
                    LocalTensor<uint8_t> maskTensor = this->softmaxTempBuf.template Get<uint8_t>();
                    CompareScalar(maskTensor, maxTensor, static_cast<T>(this->negativeFloatScalar),
                                  AscendC::CMPMODE::NE, calcCnt);
                    Select(sumTensor, maskTensor, sumTensor, static_cast<T>(this->positiveFloatScalar),
                           AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, calcCnt);
                } else {
                    AdjustSoftMaxRes<T, T>(sumTensor, maxTensor, this->negativeIntScalar,
                                           this->positiveFloatScalar, softmaxFullShapeInfo);
                }
            }
        }
    }

    if (loopIdx == extraInfo.realSplitN - 1 && extraInfo.s2LoopCount == extraInfo.s2LoopLimit) {
        if (this->tilingData->inputParams.tndSoftmaxOut == 1) {
            // layout of output softmax_max and softmax_out should be TND
            extraInfo.softmaxMaxOffset =
                (extraInfo.s1SizeAcc * this->n2G +
                extraInfo.s1oIdx * this->cubeS1BaseSize * this->n2G + extraInfo.vecCoreOffset * this->n2G +
                extraInfo.n2oIdx * this->tilingData->inputParams.gSize +
                extraInfo.goIdx) * static_cast<int64_t>(fp32BaseSize);
        } else {
            extraInfo.softmaxMaxOffset =
                (extraInfo.s1SizeAcc * this->n2G +
                extraInfo.n2oIdx * this->tilingData->inputParams.gSize * extraInfo.s1Size +
                extraInfo.goIdx * extraInfo.s1Size + extraInfo.s1oIdx * this->cubeS1BaseSize + extraInfo.vecCoreOffset) *
                static_cast<int64_t>(fp32BaseSize);
        }
        int64_t calculateSize = extraInfo.s1RealSize * fp32BaseSize;
        LocalTensor<float> maxTensor = this->softmaxMaxBuf.template Get<float>();
        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        if (this->softmaxReduceSize == 1) {
            LocalTensor<T> softmaxTemp = this->softmaxTempBuf.template Get<T>();
            AscendC::PipeBarrier<PIPE_V>();
            Brcb(softmaxTemp, maxTensor, (extraInfo.s1RealSize + 7) / 8, {1, 8});
            AscendC::SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            AscendC::WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            if (this->tilingData->inputParams.tndSoftmaxOut == 1) {
                DataCopy(this->softmaxMaxGm[extraInfo.softmaxMaxOffset], softmaxTemp,
                    {static_cast<uint16_t>(extraInfo.s1RealSize), 1, 0, static_cast<uint16_t>(this->n2G - 1)});
            } else {
                DataCopy(this->softmaxMaxGm[extraInfo.softmaxMaxOffset], softmaxTemp, calculateSize);
            }
        } else {
            AscendC::SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            AscendC::WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            if (this->tilingData->inputParams.tndSoftmaxOut == 1) {
                DataCopy(this->softmaxMaxGm[extraInfo.softmaxMaxOffset], maxTensor,
                    {static_cast<uint16_t>(extraInfo.s1RealSize), 1, 0, static_cast<uint16_t>(this->n2G - 1)});
            } else {
                DataCopy(this->softmaxMaxGm[extraInfo.softmaxMaxOffset], maxTensor, calculateSize);
            }
        }
    }
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool hasRope>
__aicore__ inline void FlashAttentionScoreS1s2Bn2gs1SameAB<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                        bmm1Format, mmPolicyType, hasRope>::IterateBmm2(SplitSameABExtraInfo &extraInfo,
                                            matmul::MatmulImpl<a2Type, b2Type, c2NzType, bias2Type, CFG_CUSTOM_EXCEED> &bmm2)
{
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t s2Offset = 0;

    if constexpr (layOutType == LayOutTypeEnum::LAYOUT_BSH) {
        // BSH/BSND
        bOffset = extraInfo.boIdx * this->n2S2D2;
        s2Offset = extraInfo.s2StartIdx * this->n2D2 + extraInfo.s2LoopCount * s2BaseNratioSize * this->n2D2;
        n2Offset = extraInfo.n2oIdx * d2Size;
    } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_SBH) {
        // SBH/SBND
        s2Offset = extraInfo.s2StartIdx * this->bN2D2 + extraInfo.s2LoopCount * s2BaseNratioSize * this->bN2D2;
        bOffset = extraInfo.boIdx * this->n2D2;
        n2Offset = extraInfo.n2oIdx * d2Size;
    } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_BNSD) {
        // BNSD
        bOffset = extraInfo.boIdx * this->n2S2D2;
        n2Offset = extraInfo.n2oIdx * this->s2D2;
        s2Offset = extraInfo.s2StartIdx * d2Size + extraInfo.s2LoopCount * s2BaseNratioSize * d2Size;
    } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_TND) {
        // TND
        bOffset = extraInfo.s2SizeAcc * this->n2D2;
        s2Offset = extraInfo.s2StartIdx * this->n2D2 + extraInfo.s2LoopCount * this->s2BaseNratioSize * this->n2D2;
        n2Offset = extraInfo.n2oIdx * this->d2Size;
    }

    int64_t vCoreOffset = bOffset + n2Offset + s2Offset;
    bmm2.SetOrgShape(extraInfo.cubeS1RealSize, this->mm2Kb, extraInfo.s2AlignedSize, this->mm2Kb, this->d2Size);

    if constexpr (mmPolicyType != MmPolicyType::UNSPLITK) {
        bmm2.SetTensorA(this->stage1Res[extraInfo.taskIdMod2]);
        bmm2.SetTensorB(this->valueGm[vCoreOffset]);
        bmm2.SetTail(extraInfo.cubeS1RealSize, this->d2Size, extraInfo.s2RealSize);

        bmm2.template IterateAll<false>(this->mm2Res[extraInfo.taskIdMod2], false, false, true);
        bmm2.End();
        return;
    }

    LocalTensor<INPUT_T> scmATensor;
    LocalTensor<INPUT_T> scmBTensor;
    
    int32_t aRowNum = extraInfo.cubeS1RealSize / mm2BaseM;
    int32_t aColNum = extraInfo.s2AlignedSize / mm2BaseK;
    int32_t bRowNum = aColNum;
    int32_t bColNum = this->d2Size / mm2BaseN;
    int32_t posA = 0;
    int32_t posB = 0;

    TscmGlobal[Q_VEC1_INDEX].needAlloc = true;
    TscmGlobal[K_V_INDEX].needAlloc = true;
    TscmGlobal[Q_VEC1_INDEX].cacheSize = 0;
    TscmGlobal[K_V_INDEX].cacheSize = 0;

    auto aSrc = this->stage1Res[extraInfo.taskIdMod2];
    auto bSrc = this->valueGm[vCoreOffset];

    int32_t cOffset = 0;
    int32_t aSrcOffset = 0;

    const uint16_t blockCount = mm2BaseK * 2 / BLOCK_CUBE;
    const uint16_t blockLen = BLOCK_CUBE * mm2BaseM * sizeof(INPUT_T) / BLOCK_BYTE;
    const uint16_t srcStride = aRowNum == 1 ? 0 : blockLen; // singleCoreM == BaseM or 2 * BaseM
    const uint16_t dstStride = 0;
    AscendC::DataCopyParams copyParams = {blockCount, blockLen, srcStride, dstStride};
    event_t eventIdMte1ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE1_MTE2));
    event_t eventIdMte2ToMte1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE1));
    event_t Mte1ToMte2Flag[2];
    Mte1ToMte2Flag[0] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>());
    Mte1ToMte2Flag[1] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>());
    AscendC::SetFlag<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2);
    AscendC::WaitFlag<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2);
    for (int32_t curRow = 0;curRow < aRowNum;curRow++) {
        for (int32_t curCol = 0;curCol < bColNum;curCol++) {
            for (int32_t curK = 0;curK < aColNum;curK++) {
                posA = curRow * aColNum + curK;
                posB = curK * bColNum + curCol;
                aSrcOffset = mm2BaseK * extraInfo.cubeS1RealSize * curK + BLOCK_CUBE * mm2BaseM * curRow;
                this->mm2LoadDataB(scmBTensor, bSrc, mm2BaseK, mm2BaseN, K_V_INDEX, posB, mm2BBaseSize, this->d2Size);
                if (posA % 2 == 0 && posA > 0) {
                    AscendC::WaitFlag<HardEvent::MTE1_MTE2>(Mte1ToMte2Flag[(posA / 2) % 2]);
                }
                this->mm2LoadDataA(scmATensor, aSrc[aSrcOffset], copyParams, Q_VEC1_INDEX, posA, mm2ABaseSize);
                
                AscendC::SetFlag<HardEvent::MTE2_MTE1>(eventIdMte2ToMte1);
                AscendC::WaitFlag<HardEvent::MTE2_MTE1>(eventIdMte2ToMte1);
                bmm2.SetTensorA(scmATensor);
                bmm2.SetTensorB(scmBTensor);
                bmm2.SetTail(mm2BaseM, mm2BaseN, mm2BaseK);
                bmm2.template Iterate<false>(curK != 0);
                if (posA % 2 == 0) {
                    AscendC::SetFlag<HardEvent::MTE1_MTE2>(Mte1ToMte2Flag[(posA / 2 + 1) % 2]);
                }
                cOffset = curRow * mm2BaseM * BLOCK_CUBE;
            }
            bmm2.GetTensorC(this->mm2Res[extraInfo.taskIdMod2][cOffset]);
        }
    }
    AscendC::WaitFlag<HardEvent::MTE1_MTE2>(Mte1ToMte2Flag[(posA / 2 + 1) % 2]);

    GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_MTE2>(Mte1ToMte2Flag[0]);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_MTE2>(Mte1ToMte2Flag[1]);

    bmm2.End();
    scmATensor.SetAddr(TscmGlobal[Q_VEC1_INDEX].srcAddr);
    scmBTensor.SetAddr(TscmGlobal[K_V_INDEX].srcAddr);
    TscmGlobal[Q_VEC1_INDEX].scm.FreeTensor(scmATensor);
    TscmGlobal[K_V_INDEX].scm.FreeTensor(scmBTensor);
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool hasRope>
__aicore__ inline void
FlashAttentionScoreS1s2Bn2gs1SameAB<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                bmm1Format, mmPolicyType, hasRope>::ProcessVec2(SplitSameABExtraInfo &extraInfo)
{
    if (extraInfo.s1RealSize == 0) {
        return;
    }
    // 获取缓存bmm2的计算结果
    LocalTensor<T> bmm2ResUb = this->stage2TBuf.template Get<T>();
    LocalTensor<T> stage2BufTensor = this->commonTBuf.template Get<T>();
    int64_t vec2LoopLimit = CeilDiv(extraInfo.s1RealSize, extraInfo.vec2S1BaseSize);

    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    event_t eventIdMte2ToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    extraInfo.vec2S1RealSize = extraInfo.vec2S1BaseSize;
    for (int64_t s1oIdx = 0; s1oIdx < vec2LoopLimit; ++s1oIdx) {
        if (s1oIdx == vec2LoopLimit - 1) {
            extraInfo.vec2S1RealSize = extraInfo.s1RealSize - s1oIdx * extraInfo.vec2S1BaseSize;
        }
        int64_t mm2ResCalcSize = extraInfo.vec2S1RealSize * this->d2SizeAlign16;
        int64_t mm2ResOffset = (extraInfo.vecCoreOffset + s1oIdx * extraInfo.vec2S1BaseSize) * this->d2SizeAlign16;
        AscendC::SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        Nz2NdInfo nz2NdInfo;
        nz2NdInfo.ndFirstAxisRealSize = extraInfo.cubeS1RealSize;
        nz2NdInfo.ndFirstAxisBaseSize = extraInfo.vec2S1BaseSize;
        nz2NdInfo.ndFirstAxisLoopSize = extraInfo.vec2S1RealSize;
        nz2NdInfo.ndLastAxis = this->d2SizeAlign16;
        nz2NdInfo.loopIdx = s1oIdx;
        nz2NdInfo.vecCoreOffset = extraInfo.vecCoreOffset;
        event_t eventIdVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        AscendC::SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2);
        AscendC::WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2);
        LocalTensor<T> tempUb = this->stage1PongBuf.template Get<T>();
        NzToNd(nz2NdInfo, this->mm2Res[extraInfo.taskIdMod2], tempUb, stage2BufTensor);
        AscendC::PipeBarrier<PIPE_V>();

        if (likely(extraInfo.s2LoopCount != 0)) {
            AscendC::SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            DataCopy(bmm2ResUb, this->vec2Res[extraInfo.multiCoreInnerIdxMod2][mm2ResOffset], mm2ResCalcSize);
        }

        AscendC::SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        AscendC::WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        if (unlikely(extraInfo.s2LoopCount == 0)) {
            DataCopy(bmm2ResUb, stage2BufTensor, mm2ResCalcSize);
        } else {
            this->Bmm2ResultMul(extraInfo, bmm2ResUb, s1oIdx);
            AscendC::PipeBarrier<PIPE_V>();
            Add(bmm2ResUb, bmm2ResUb, stage2BufTensor, mm2ResCalcSize);
        }

        if (extraInfo.s2LoopCount == extraInfo.s2LoopLimit) {
            Bmm2ResultDiv(extraInfo, s1oIdx);
            Bmm2DataCopyOut(extraInfo, s1oIdx, mm2ResCalcSize);
            if (s1oIdx == vec2LoopLimit - 1) {
                SoftmaxDataCopyOut(extraInfo);
            }
            event_t eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
            AscendC::SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
            AscendC::WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        } else {
            AscendC::SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            AscendC::WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            DataCopy(this->vec2Res[extraInfo.multiCoreInnerIdxMod2][mm2ResOffset], bmm2ResUb, mm2ResCalcSize);
            AscendC::SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        }
    }
    return;
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool hasRope>
__aicore__ inline void
FlashAttentionScoreS1s2Bn2gs1SameAB<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                bmm1Format, mmPolicyType, hasRope>::Bmm2ResultMul(SplitSameABExtraInfo &extraInfo,
                                                           LocalTensor<T> &bmm2ResUb, int64_t s1oIdx)
{
    AscendC::PipeBarrier<PIPE_V>();
    LocalTensor<T> expUb = softmaxExpBuf[extraInfo.taskIdMod2].template Get<T>();

    BinaryRepeatParams repeatParams;
    repeatParams.src0BlkStride = 0;
    repeatParams.src0RepStride = 1;
    repeatParams.src1RepStride = d2SizeAlign16 / blockSize;
    repeatParams.dstRepStride = d2SizeAlign16 / blockSize;

    // s1长度可能会超过255限制，修改成双重循环
    // 根据一次最多计算的byte数量，对bmm2Res分组mul
    int32_t loop = d2SizeAlign16 / repeatMaxSize;
    int32_t remain = d2SizeAlign16 % repeatMaxSize;
    int64_t softmaxTempOffset = s1oIdx * extraInfo.vec2S1BaseSize * 8;
    if (this->softmaxReduceSize == 1) {
        LocalTensor<T> softmaxTemp = this->softmaxTempBuf.template Get<T>();
        Brcb(softmaxTemp, expUb, (extraInfo.s1RealSize + 7) / 8, {1, 8});
        AscendC::PipeBarrier<PIPE_V>();
        for (int i = 0; i < loop; ++i) {
            Mul(bmm2ResUb[i * repeatMaxSize], softmaxTemp[softmaxTempOffset], bmm2ResUb[i * repeatMaxSize],
                repeatMaxSize, extraInfo.vec2S1RealSize, repeatParams);
        }
        if (remain) {
            Mul(bmm2ResUb[loop * repeatMaxSize], softmaxTemp[softmaxTempOffset],
                bmm2ResUb[loop * repeatMaxSize], remain, extraInfo.vec2S1RealSize, repeatParams);
        }
    } else {
        for (int i = 0; i < loop; ++i) {
            Mul(bmm2ResUb[i * repeatMaxSize], expUb[softmaxTempOffset], bmm2ResUb[i * repeatMaxSize],
                repeatMaxSize, extraInfo.vec2S1RealSize, repeatParams);
        }
        if (remain) {
            Mul(bmm2ResUb[loop * repeatMaxSize], expUb[softmaxTempOffset],
                bmm2ResUb[loop * repeatMaxSize], remain, extraInfo.vec2S1RealSize, repeatParams);
        }
    }
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool hasRope>
__aicore__ inline void
FlashAttentionScoreS1s2Bn2gs1SameAB<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                bmm1Format, mmPolicyType, hasRope>::Bmm2ResultDiv(SplitSameABExtraInfo &extraInfo, int64_t s1oIdx)
{
    LocalTensor<T> bmm2ResUb = this->stage2TBuf.template Get<T>();

    BinaryRepeatParams repeatParams;
    repeatParams.src0BlkStride = 1;
    repeatParams.src0RepStride = d2SizeAlign16 / blockSize;
    repeatParams.src1BlkStride = 0;
    repeatParams.src1RepStride = 1;
    repeatParams.dstRepStride = d2SizeAlign16 / blockSize;
    int32_t loop = d2SizeAlign16 / repeatMaxSize;
    int32_t remain = d2SizeAlign16 % repeatMaxSize;

    LocalTensor<float> sumUb = softmaxSumBuf[extraInfo.multiCoreInnerIdxMod2].template Get<float>();
    int32_t calcSize = sumUb.GetSize();
    // 用optionalInputQueue的queue
    AscendC::PipeBarrier<PIPE_V>();
    if (this->softmaxReduceSize == 1) {
        LocalTensor<T> softmaxTemp = this->softmaxTempBuf.template Get<T>();
        Brcb(softmaxTemp, sumUb, (extraInfo.s1RealSize + 7) / 8, {1, 8});
        AscendC::PipeBarrier<PIPE_V>();
        for (int i = 0; i < loop; ++i) {
            Div(bmm2ResUb[i * repeatMaxSize], bmm2ResUb[i * repeatMaxSize],
                softmaxTemp[s1oIdx * extraInfo.vec2S1BaseSize * 8], repeatMaxSize, extraInfo.vec2S1RealSize, repeatParams);
        }
        if (likely(remain)) {
            Div(bmm2ResUb[loop * repeatMaxSize], bmm2ResUb[loop * repeatMaxSize],
                softmaxTemp[s1oIdx * extraInfo.vec2S1BaseSize * 8], remain, extraInfo.vec2S1RealSize, repeatParams);
        }
    } else {
        if constexpr (IsSameType<T, half>::value) {
            LocalTensor<float> commonBufTensor = this->commonTBuf.template Get<float>();
            Copy(commonBufTensor, sumUb, 64, calcSize / 64, {2, 1, 16, 8});
            Copy(commonBufTensor[8], sumUb, 64, calcSize / 64, {2, 1, 16, 8});
            LocalTensor<T> sumCastTensor =
                softmaxExpBuf[0].template Get<T>(this->tilingData->tensorSizeParams.softmaxExpUbSize);

            Cast(sumCastTensor, commonBufTensor, RoundMode::CAST_ROUND, 2 * sumUb.GetSize());
            for (int i = 0; i < loop; i++) {
                Div(bmm2ResUb[i * repeatMaxSize], bmm2ResUb[i * repeatMaxSize], sumCastTensor, repeatMaxSize,
                    extraInfo.vec2S1RealSize, repeatParams);
            }
            if (likely(remain)) {
                Div(bmm2ResUb[loop * repeatMaxSize], bmm2ResUb[loop * repeatMaxSize], sumCastTensor, remain,
                    extraInfo.vec2S1RealSize, repeatParams);
            }
        } else {
            for (int i = 0; i < loop; i++) {
                Div(bmm2ResUb[i * repeatMaxSize], bmm2ResUb[i * repeatMaxSize],
                    sumUb[s1oIdx * extraInfo.vec2S1BaseSize * 8], repeatMaxSize, extraInfo.vec2S1RealSize, repeatParams);
            }
            if (likely(remain)) {
                Div(bmm2ResUb[loop * repeatMaxSize], bmm2ResUb[loop * repeatMaxSize],
                    sumUb[s1oIdx * extraInfo.vec2S1BaseSize * 8], remain, extraInfo.vec2S1RealSize, repeatParams);
            }
        }
    }
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool hasRope>
__aicore__ inline void
FlashAttentionScoreS1s2Bn2gs1SameAB<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                bmm1Format, mmPolicyType, hasRope>::Bmm2DataCopyOut(SplitSameABExtraInfo &extraInfo, int64_t s1oIdx,
                                                             int64_t mm2ResCalcSize)
{
    LocalTensor<T> bmm2ResUb = this->stage2TBuf.template Get<T>();
    LocalTensor<INPUT_T> attenOut = this->stage2TBuf.template Get<INPUT_T>();
    bmm2ResUb.SetSize(mm2ResCalcSize);
    AscendC::PipeBarrier<PIPE_V>();

    if constexpr (!IsSameType<INPUT_T, T>::value) {
        Cast(attenOut, bmm2ResUb, RoundMode::CAST_ROUND, mm2ResCalcSize);
    } else {
        DataCopy(attenOut, bmm2ResUb, mm2ResCalcSize);
    }

    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    AscendC::SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    AscendC::WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

    DataCopyParams dataCopyParams;
    dataCopyParams.blockLen = this->d2Size * sizeof(INPUT_T);
    dataCopyParams.srcStride = 0;
    int64_t dstStride = 0;
    int64_t attenOutOffset = this->d2Size;
    int64_t datacopyOffset = this->d2Size;
    int64_t bOffset = extraInfo.boIdx * this->n2GS1D2;
    int64_t n2Offset = extraInfo.n2oIdx * this->gS1D2;
    int64_t gOffset = extraInfo.goIdx * this->s1D2;
    int64_t s1Offset = extraInfo.s1oIdx * this->s1BaseD2;

    if constexpr (layOutType == LayOutTypeEnum::LAYOUT_BSH) {
        datacopyOffset = this->n2GD2;
        attenOutOffset = this->n2GD2;
        dstStride = (this->tilingData->inputParams.n2Size * this->tilingData->inputParams.gSize - 1) * this->d2Size *
                    sizeof(INPUT_T);
        bOffset = extraInfo.boIdx * this->n2GS1D2;
        s1Offset = extraInfo.s1oIdx * this->s1BaseN2GD2;
        n2Offset = extraInfo.n2oIdx * this->gD2;
        gOffset = extraInfo.goIdx * d2Size;
    } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_SBH) {
        datacopyOffset = this->bN2G * this->d2Size;
        attenOutOffset = this->bN2GD2;
        dstStride = (this->tilingData->inputParams.bSize * this->tilingData->inputParams.n2Size *
                         this->tilingData->inputParams.gSize - 1) * this->d2Size * sizeof(INPUT_T);       
        s1Offset = extraInfo.s1oIdx * this->s1BaseBN2GD2;
        bOffset = extraInfo.boIdx * this->n2GD2;
        n2Offset = extraInfo.n2oIdx * this->gD2;
        gOffset = extraInfo.goIdx * d2Size;
    } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_TND) {
        datacopyOffset = this->n2GD2;
        attenOutOffset = this->n2GD2;
        dstStride = (this->tilingData->inputParams.n2Size * this->tilingData->inputParams.gSize - 1) * this->d2Size *
                    sizeof(INPUT_T);  
        bOffset = extraInfo.s1SizeAcc * this->n2GD2;
        s1Offset = extraInfo.s1oIdx * this->s1BaseN2GD2;
        n2Offset = extraInfo.n2oIdx * this->gD2;
        gOffset = extraInfo.goIdx * this->d2Size;
    }
    int64_t outCoreOffset = bOffset + n2Offset + gOffset + s1Offset;

    // dataCopyParams.dstStride类型定义uint16_t，65535是其最大值
    if (likely(dstStride <= 65535)) {
        dataCopyParams.blockCount = extraInfo.vec2S1RealSize;
        dataCopyParams.dstStride = static_cast<uint16_t>(dstStride);
        DataCopyPad(this->attentionOutGm[outCoreOffset +
                    (s1oIdx * extraInfo.vec2S1BaseSize + extraInfo.vecCoreOffset) * attenOutOffset],
                    attenOut, dataCopyParams);
    } else {
        dataCopyParams.blockCount = 1;
        dataCopyParams.dstStride = 0;

        for (int32_t i = 0; i < extraInfo.vec2S1RealSize; i++) {
            DataCopyPad(this->attentionOutGm[outCoreOffset +
                        (s1oIdx * extraInfo.vec2S1BaseSize + extraInfo.vecCoreOffset) * attenOutOffset +
                        i * datacopyOffset], attenOut[i * this->d2SizeAlign16], dataCopyParams);
        }
    }
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool hasRope>
__aicore__ inline void
FlashAttentionScoreS1s2Bn2gs1SameAB<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                bmm1Format, mmPolicyType, hasRope>::SoftmaxDataCopyOut(SplitSameABExtraInfo &extraInfo)
{
    LocalTensor<float> sumTensor = this->softmaxSumBuf[extraInfo.multiCoreInnerIdxMod2].template Get<float>();
    if (this->softmaxReduceSize == 1) {
        LocalTensor<T> softmaxTemp = this->softmaxTempBuf.template Get<T>();
        Brcb(softmaxTemp, sumTensor, (extraInfo.s1RealSize + 7) / 8, {1, 8});
        event_t eventIdVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        AscendC::SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
        AscendC::WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
        if (this->tilingData->inputParams.tndSoftmaxOut == 1) {
            DataCopy(this->softmaxSumGm[extraInfo.softmaxMaxOffset], softmaxTemp,
                {static_cast<uint16_t>(extraInfo.s1RealSize), 1, 0, static_cast<uint16_t>(this->n2G - 1)});
        } else {
            DataCopy(this->softmaxSumGm[extraInfo.softmaxMaxOffset], softmaxTemp, extraInfo.s1RealSize * fp32BaseSize);
        }
    } else {
        if (this->tilingData->inputParams.tndSoftmaxOut == 1) {
            DataCopy(this->softmaxSumGm[extraInfo.softmaxMaxOffset], sumTensor,
                {static_cast<uint16_t>(extraInfo.s1RealSize), 1, 0, static_cast<uint16_t>(this->n2G - 1)});
        } else {
            DataCopy(this->softmaxSumGm[extraInfo.softmaxMaxOffset], sumTensor, extraInfo.s1RealSize * fp32BaseSize);
        }
    }
}

#endif // FLASH_ATTENTION_SCORE_S1S2_BN2GS1_SAB_H
