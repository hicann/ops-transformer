/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _fused_floyd_attention_grad_S1S2_BN2GS1S2_H_
#define _fused_floyd_attention_grad_S1S2_BN2GS1S2_H_

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "kernel_operator_list_tensor_intf.h"

using namespace matmul;
using namespace AscendC;

struct EvenvIdList
{
    event_t structVWaitMte2Ping;
    event_t structVWaitMte2Pong;
    event_t structMte2WaitMte3Ping;
    event_t structMte2WaitMte3Pong;
    event_t structMte3WaitVPing;
    event_t structMte3WaitVPong;
    event_t structMte2WaitVPing;
    event_t structMte2WaitVPong;
};

constexpr static MatmulConfig NORM_DISABLE_INIT = {true,  false, false, 0,     0,     0,     false, false,
                                                   false, false, 0,     0,     0,     0,     0,     0,
                                                   0,     0,     true,  false, false, false, false, false};

__aicore__ inline void AttenMaskBoolCopyIn(LocalTensor<uint8_t> &dstTensor, GlobalTensor<uint8_t> &srcTensor,
    int64_t srcOffset, uint32_t s1Size, uint32_t s2Size, int64_t totalS2Size, int64_t attenMaskShapeType=0)
{
    constexpr int32_t blockBytes = 32;
    uint32_t alignedS2Size = CeilDiv(s2Size, blockBytes) * blockBytes;
    uint32_t shapeArray[] = {s1Size, alignedS2Size};
    dstTensor.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
    dstTensor.SetSize(s1Size * alignedS2Size);
    DataCopyParams dataCopyParams;
    if (attenMaskShapeType == 0) {
        dataCopyParams.blockCount = s1Size;
    } else if (attenMaskShapeType == 3) {
        dataCopyParams.blockCount = 1;
    }
    dataCopyParams.dstStride = 0;
    if (totalS2Size % blockBytes == 0) {
        dataCopyParams.blockLen = alignedS2Size / blockBytes;
        dataCopyParams.srcStride = (totalS2Size - alignedS2Size) / blockBytes;
        DataCopy(dstTensor, srcTensor[srcOffset], dataCopyParams);
        if (attenMaskShapeType == 3) {
            event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            uint32_t logStep = 0;
            for (uint32_t tmp = 1; tmp < s1Size; tmp <<= 1) ++logStep;
            /* 3. 逐级“自身翻倍”广播 */
            uint32_t curRows = 1;                      // 当前已经准备好的行数
            uint32_t rowBytes = alignedS2Size;         // 每行字节数
            for (uint32_t step = 0; step < logStep; ++step) {
                /* 把 [0, curRows) 行拷贝到 [curRows, 2*curRows) 行 */
                DataCopyParams bc;
                bc.blockCount = curRows;               // 拷贝 curRows 行
                bc.blockLen   = rowBytes / blockBytes; // 每行 block 数
                bc.dstStride  = 0;                     // 目的行之间连续
                bc.srcStride  = 0;                     // 源行之间也连续
                AscendC::PipeBarrier<PIPE_V>();
                DataCopy(dstTensor[curRows * rowBytes], // 目的首地址
                        dstTensor,                     // 源首地址
                        bc);
                curRows <<= 1;                          // 行数翻倍
            }
        }
    } else {
        dataCopyParams.blockLen = s2Size;
        dataCopyParams.srcStride = totalS2Size - s2Size;
        DataCopyPadParams dataCopyPadParams;
        dataCopyPadParams.isPad = true;
        dataCopyPadParams.rightPadding = alignedS2Size - s2Size;
        dataCopyPadParams.paddingValue = 0;
        DataCopyPad(dstTensor, srcTensor[srcOffset], dataCopyParams, dataCopyPadParams);
    }
}

__aicore__ inline void DataCopyOut(const __gm__ void *gm, const LocalTensor<int8_t> &co1Local,
                                   const void *dataCopyOutParams, const uint64_t tilingPtr, const uint64_t dataPtr)
{
    const DataCopyOutParams *param = reinterpret_cast<const DataCopyOutParams *>(dataCopyOutParams);
    uint64_t dstStride = dataPtr * 16 / 8 - param->burstLen;
    FixpipeParams<float> fixpipeParams(param->cBurstNum, param->burstLen, param->srcStride,
                                       static_cast<uint32_t>(dstStride));

    if (param->enUnitFlag) {
        fixpipeParams.unitFlag = 3;
    }
    LocalTensor<float> tmpLocal = co1Local.template ReinterpretCast<float>();
    GlobalTensor<float> tmpGm;
    tmpGm.SetGlobalBuffer((__gm__ float *)(gm));
    Fixpipe(tmpGm, tmpLocal, fixpipeParams);
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK = 0, const uint32_t IS_PSE = 1,
          const uint32_t IS_DROP = 1, const CubeFormat MM_OUT_FORMAT = CubeFormat::ND, const uint32_t INPUT_LAYOUT = 0,
          const CubeFormat MM2_OUT_FORMAT = CubeFormat::NZ, const uint32_t TND_S1_PP = 0>
class FusedFloydAttentionGradS1s2Bn2gs1s2 {
public:
    __aicore__ inline FusedFloydAttentionGradS1s2Bn2gs1s2(){};

    __aicore__ inline void Init(__gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *key_1, __gm__ uint8_t *value_1, 
                                __gm__ uint8_t *dx, __gm__ uint8_t *query, __gm__ uint8_t *atten_mask,
                                __gm__ uint8_t *forward_res, __gm__ uint8_t *softmax_max, __gm__ uint8_t *softmax_sum,
                                __gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv, __gm__ uint8_t *dk_1, 
                                __gm__ uint8_t *dv_1, __gm__ uint8_t *workspace,
                                const FusedFloydAttentionGradTilingDataS1s2Bn2gs1s2 *__restrict ordTilingData,
                                TPipe *pipe_in);
    __aicore__ inline void CopyInSoftMaxGrad(LocalTensor<T1> &dstTensor, LocalTensor<T1> &dstTensor2,
                                             int64_t softmaxGradOffset, uint32_t s1Extend, uint32_t dExtend,
                                             uint32_t dExtendAlign);
    __aicore__ inline void CalcSoftMaxGrad(LocalTensor<float> &sfmgClc3, int64_t aTensorOffset, uint32_t s1Extend);
    __aicore__ inline void CopyInSoftMax(LocalTensor<float> &dstTensor, uint32_t s1Extend, uint32_t softMaxOffset);
    __aicore__ inline void CalcSoftMax(LocalTensor<T2> &dstTensor, LocalTensor<float> &srcTensor, uint32_t s1Extend,
                                       uint32_t s2Extend, uint32_t s2ExtendAlign, const SoftMaxTiling &tiling);
    __aicore__ inline void CopyInAttenMaskBool(const  LocalTensor<uint8_t> &dstTensor, int64_t attenMaskOffset,
                                               uint32_t s1Extend, uint32_t s2Extend);
    __aicore__ inline void CalcAttenMaskBool(LocalTensor<T2> &dstTensor, LocalTensor<uint8_t> srcTensor,
                                             uint32_t s1Extend, uint32_t s2Extend, uint8_t maskType = 0);

    __aicore__ inline void ComputeBMMK1(uint32_t aTensorOffsetCv, uint32_t bTensorOffsetCv, uint32_t s1StrideSize, 
                                        uint32_t s2StrideSize, uint32_t gExtend, uint32_t s1Extend, uint32_t s2Extend, uint32_t s2ExtendAlign);
    __aicore__ inline void ComputeBMM2K1(uint32_t aTensorOffsetCv, uint32_t bTensorOffsetCv, uint32_t cTensorOffsetCv, 
                                                uint32_t gExtend, uint32_t s1Extend, uint32_t s2Extend, uint32_t s2ExtendAlign);
    __aicore__ inline void ComputeBMM3K1V1(uint32_t aTensorOffsetCv, uint32_t bTensorOffsetCv, uint32_t cTensorOffsetCv, 
                                            uint32_t gExtend, uint32_t s1Extend, uint32_t s2Extend, uint32_t s2ExtendAlign);
    __aicore__ inline void Process();
    __aicore__ inline bool IsCubeBlockNeedCompute(int64_t baseIndex);
    __aicore__ inline void InitIndex(int64_t index);
    __aicore__ inline void SubGrapA(int64_t curIdx, int64_t curS1Idx, int64_t curS2Idx, int64_t LoopGIdx, EvenvIdList &eventIdList);
    __aicore__ inline void SubGrapB(int64_t curIdx, int64_t curS1Idx, int64_t curS2Idx, int64_t LoopGIdx, EvenvIdList &eventIdList);
    __aicore__ inline void Compute(int64_t preIndex, int64_t nextIndex);
    __aicore__ inline void SyncALLCores();
    __aicore__ inline void LocalAllocEventID(EvenvIdList &eventIdList);
    __aicore__ inline void LocalReleaseEventID(EvenvIdList &eventIdList);

    using aType1 = MatmulType<TPosition::GM, CubeFormat::ND, T1>;
    using bType1 = MatmulType<TPosition::GM, CubeFormat::ND, T1, true>;
    using cType1 = MatmulType<TPosition::GM, MM_OUT_FORMAT, T2>;
    using biasType1 = MatmulType<TPosition::GM, CubeFormat::ND, float>;

    using aType2 = MatmulType<TPosition::GM, MM_OUT_FORMAT, T1, true>;
    using bType2 = MatmulType<TPosition::GM, CubeFormat::ND, T1>;
    using cType2 = MatmulType<TPosition::GM, MM2_OUT_FORMAT, float>;
    using biasType2 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    
    Matmul<aType1, bType1, cType1, biasType1, NORM_DISABLE_INIT> mm1;
    using modeTypeMm = typename AscendC::Conditional<
        (MM2_OUT_FORMAT == CubeFormat::NZ),
        Matmul<aType2, bType2, cType2, biasType2, NORM_DISABLE_INIT, MatmulCallBackFunc<DataCopyOut>>,
        Matmul<aType2, bType2, cType2, biasType2, NORM_DISABLE_INIT>>::type;
    modeTypeMm mm3;

    constexpr static MatmulConfig LIMIT_CFG = GetNormalConfig(true);

    using aTypeBmm1 = MatmulType<TPosition::GM, CubeFormat::ND, T1, false, LayoutMode::BSNGD>;
    using bTypeBmm1 = MatmulType<TPosition::GM, CubeFormat::ND, T1, true, LayoutMode::BSNGD>;
    using cTypeBmm1 = MatmulType<TPosition::GM, CubeFormat::ND, T2,  false, LayoutMode::BSNGD>;
    using biasType = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    matmul::Matmul<aTypeBmm1, bTypeBmm1, cTypeBmm1, biasType, LIMIT_CFG> bmm1k1;

    using aTypeBmm2 = MatmulType<TPosition::GM, CubeFormat::ND, T1, false, LayoutMode::BSNGD>;
    using bTypeBmm2 = MatmulType<TPosition::GM, CubeFormat::ND, T1, false, LayoutMode::BSNGD>;
    using cTypeBmm2 = MatmulType<TPosition::GM, CubeFormat::ND, T2,  false, LayoutMode::BSNGD>;
    matmul::Matmul<aTypeBmm2, bTypeBmm2, cTypeBmm2, biasType, LIMIT_CFG> bmm2k1;

    using aTypeBmm3 = MatmulType<TPosition::GM, CubeFormat::ND, T1, true, LayoutMode::BSNGD>;
    using bTypeBmm3 = MatmulType<TPosition::GM, CubeFormat::ND, T1, false, LayoutMode::BSNGD>;
    using cTypeBmm3 = MatmulType<TPosition::GM, CubeFormat::ND, T2,  false, LayoutMode::BSNGD>;
    matmul::Matmul<aTypeBmm3, bTypeBmm3, cTypeBmm3, biasType, LIMIT_CFG> bmm3k1v1;

    using aTypeBmmqkv = MatmulType<TPosition::GM, CubeFormat::ND, T1, false, LayoutMode::BNGS1S2>;
    using bTypeBmmqkv = MatmulType<TPosition::GM, CubeFormat::ND, T1, false, LayoutMode::BNGS1S2>;
    using cTypeBmmqkv = MatmulType<TPosition::GM, CubeFormat::ND, T2,  false, LayoutMode::BNGS1S2>;
    matmul::Matmul<aTypeBmm3, bTypeBmm3, cTypeBmm3, biasType, LIMIT_CFG> bmmqkv;

    __aicore__ inline void NZCopyIn(int64_t mmAddr, GlobalTensor<T2> &mmWspGm, LocalTensor<T2> &mmTensorCurr,
                                    uint32_t s1VecSize, uint32_t s2VecSize);
    __aicore__ inline void NZ2ND(LocalTensor<T2> &mmTensorCurr, LocalTensor<T2> &tmpTensor, uint32_t s1VecSize,
                                 uint32_t s2VecSize);
    __aicore__ inline void ND2NZ(LocalTensor<T1> &mmTensorCurr, LocalTensor<T1> &tmpTensor, uint32_t s1VecSize,
                                 uint32_t s2VecSize);

protected:
    TPipe *pipe;
    TBuf<> ubBuffer;
    TBuf<> tmpBuffer;
    TBuf<> vecClc3;

    uint32_t coreNum;
    int64_t cBlockIdx;
    const FusedFloydAttentionGradTilingDataS1s2Bn2gs1s2 *__restrict TilingData;

    // input
    GlobalTensor<T1> keyGm, valueGm, dxGm, queryGm, forwardResGm, keyGm1, valueGm1;
    GlobalTensor<uint8_t> maskWorkSpaceGm, attenMaskU8Gm;
    GlobalTensor<float> softmaxMaxGm, softmaxSumGm;

    // output
    GlobalTensor<float> dqWorkSpaceGm, dkWorkSpaceGm, dvWorkSpaceGm, dk1WorkSpaceGm, dv1WorkSpaceGm;
    GlobalTensor<T1> dropWorkSpaceGm, mulWorkSpaceGm;

    // workspace
    GlobalTensor<T2> mm1WorkspaceGm;
    GlobalTensor<T2> mm2WorkspaceGm;

    __gm__ uint8_t *actual_seq_qlen_addr;
    __gm__ uint8_t *actual_seq_kvlen_addr;

    // AscendC
    GlobalTensor<int32_t> syncGlobal;

    // matmal1/matmal2 result buffer
    GlobalTensor<float> matmalResultBuffer1;
    GlobalTensor<float> matmalResultBuffer2;

    GlobalTensor<half> pseAlibiGm;
    GlobalTensor<float> dvGm;


    constexpr static uint32_t BNGSD = 0;
    constexpr static uint32_t SBNGD = 1;
    constexpr static uint32_t BSNGD = 2;
    constexpr static uint32_t TND = 3;
    constexpr static uint32_t ENABLE = 1;

    T2 mulsValue = -10000.0;

    // optional control
    float keepProb;
    int64_t s1Token;
    int64_t s2Token;
    int64_t actualCalcS1Token;
    int64_t actualCalcS2Token;
    uint32_t sparseMode;

    // org shape info
    int64_t b;
    int64_t n2;
    int64_t g;
    int64_t s1;
    int64_t s2;
    int64_t d;
    int64_t attenMaskDimS2;

    uint32_t baseMN;
    uint32_t cubeBaseMN;
    uint32_t singleBaseMN;

    // split info
    int64_t s1Outer;
    int64_t bmmS1base;
    uint32_t s1CvRatio;
    uint32_t s1CvInner;
    uint32_t s1CvTail;

    int64_t s2Outer;
    uint32_t s2CvRatio;
    uint32_t s2Inner;
    uint32_t s2CvInner;
    uint32_t sfmgdOuter;
    uint32_t sfmgdInner;
    uint32_t sfmgdTail;
    uint32_t sfmgdTailAlign;
    int64_t gOuter;
    uint32_t gInner;
    uint32_t gTail;
    bool dropBitMode;

    int64_t blockOuter;

    // sparse block info
    const int64_t *blockStarts;
    const int64_t *blockEnds;

    // buferinfo
    uint32_t matmalWorkspaceSize;

    // base info
    int64_t n2gs1os2o;
    int64_t gs1os2o;
    int64_t s1os2o;

    int64_t baseIdx{0};
    int64_t bDimIdx{0};
    int64_t n2DimIdx{0};
    int64_t gDimIdx{0};
    int64_t s1oDimIdx{0};
    int64_t s2oCvDimIdx{0};


    uint32_t preS2CvBegin{0};
    uint32_t preS2CvEnd{0};
    uint32_t nextS2CvBegin{0};
    uint32_t nextS2CvEnd{0};

    int32_t isStart = 1;
    uint32_t pingpongIdx = 1;

    // db
    int64_t s2CvExtend = 0;
    uint32_t s1CvExtend{0};
    int64_t gExtend{0};
    int64_t s2CvExtendAlign = 0;
    int64_t s1CvExtendAlign = 0;
    uint32_t s1VecLoop = 0;
    uint32_t s1VecSize = 0;
    uint32_t s1ExtendSubGraph = 0;
    uint32_t s2Extend = 0;
    uint32_t s2ExtendAlign = 0;
    uint32_t s2VecLoop = 0;
    uint32_t s2VecSize = 0;
    uint32_t s2VecSizeAlign = 0;

    // db buffer
    constexpr static uint32_t T2Begin = 0;
    constexpr static uint32_t T1Begin = 33 * 1024;
    constexpr static uint32_t BoolBegin = 51 * 1024;
    constexpr static uint32_t T2BlockBegin = 59 * 1024;
    constexpr static uint32_t U8Begin = 67 * 1024;
    constexpr static uint32_t DbBegin = 75 * 1024;
    constexpr static uint32_t hufTmpBuffBegin = 16 * 1024;

    // calDtype
    constexpr static uint32_t calDtypeBytes = 4;

    // other const
    constexpr static uint32_t cal_block_num = 32 / sizeof(T2);
    constexpr static uint32_t cal_repeat_num = 256 / sizeof(T2);
    constexpr static uint32_t input_block_num = 32 / sizeof(T1);
    constexpr static uint32_t SYNC_GLOBAL_WORKSPACE_SIZE = 16 * 1024;
    constexpr static uint32_t ADDR_ALIGN_SIZE = 512;
    constexpr static uint32_t INPUT_NUMS = 2;
    constexpr static uint32_t BLOCK_SIZE = 32;
    constexpr static int64_t C0_SIZE = 16;
    constexpr static int64_t VEC_REPEAT = 8;
    constexpr static uint32_t MAX_BASIC_BLOCK_SIZE = 1024;
    constexpr static uint32_t PSE_PERFORMANCE_MODE = 0x12;
    constexpr static uint32_t PREFIX_COMPRESS_CAUSAL_S_SIZE = 2048;
    constexpr static uint32_t PREFIX_COMPRESS_ALL_MASK_S1_SIZE = 1024;
    constexpr static uint32_t SFMG_RESULT_DB_SIZE = 4 * 1024;
    // SFMG_DB_SIZE = 18 + 18 + 9+ 9 + 18 = 72K（GraphAB的DB是74K）, CLC1占用18K
    constexpr static uint32_t SFMG_DB_INPUT_UBSIZE = 18 * 1024;
    constexpr static uint32_t SFMG_S1_ALIGN_NUM = 8;

    constexpr static uint32_t VEC_S2_LEN = 256;
    enum class AttenMaskCompress {
        Empty = 0,
        PreOnly = 1,
        NextOnly = 2,
        All = 3
    };
    AttenMaskCompress AttenBandMode = AttenMaskCompress::All;
};

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT, const uint32_t TND_S1_PP>
__aicore__ inline void
FusedFloydAttentionGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
                                    MM2_OUT_FORMAT, TND_S1_PP>::CopyInAttenMaskBool(const LocalTensor<uint8_t> &dstTensor,
                                                                         int64_t attenMaskOffset, uint32_t s1Extend,
                                                                         uint32_t s2Extend)
{
    AscendC::DataCopyExtParams intriParams;
    intriParams.blockCount = s1Extend;
    intriParams.blockLen = s2Extend * sizeof(uint8_t);
    intriParams.srcStride = (attenMaskDimS2 - s2Extend) * sizeof(uint8_t);
    intriParams.dstStride = 0;
    intriParams.rsv = 0;
    DataCopyPad(dstTensor, attenMaskU8Gm[attenMaskOffset], intriParams, {false, 0, 0, 0});
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT, const uint32_t TND_S1_PP>
__aicore__ inline void
FusedFloydAttentionGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
                                    MM2_OUT_FORMAT, TND_S1_PP>::CalcAttenMaskBool(LocalTensor<T2> &dstTensor,
                                                                       LocalTensor<uint8_t> srcTensor,
                                                                       uint32_t s1Extend, uint32_t s2Extend,
                                                                       uint8_t maskType)
{
    LocalTensor<uint8_t> tmpUbBuffer = tmpBuffer.Get<uint8_t>();

    T2 scalar;
    if constexpr (IsSameType<T2, float>::value) {
        uint32_t tmp = 0xFF7FFFFF;
        scalar = *((float *)&tmp);
    } else {
        uint16_t tmp = 0xFBFF;
        scalar = *((half *)&tmp);
    }

    SelectWithBytesMaskShapeInfo info;
    info.firstAxis = s1Extend;
    info.srcLastAxis = s2Extend;
    info.maskLastAxis = (s2Extend * sizeof(uint8_t) + 31) / 32 * 32 / sizeof(uint8_t);
    dstTensor.SetSize(info.firstAxis * info.srcLastAxis);
    srcTensor.SetSize(info.firstAxis * info.maskLastAxis);
    if (maskType == 0) {
        SelectWithBytesMask(dstTensor, dstTensor, scalar, srcTensor, tmpUbBuffer, info);
    } else {
        SelectWithBytesMask(dstTensor, scalar, dstTensor, srcTensor, tmpUbBuffer, info);
    }
}


template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT, const uint32_t TND_S1_PP>
__aicore__ inline void FusedFloydAttentionGradS1s2Bn2gs1s2<
    T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
    MM2_OUT_FORMAT, TND_S1_PP>::Init(__gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *key_1, __gm__ uint8_t *value_1,
                                     __gm__ uint8_t *dx, __gm__ uint8_t *query, __gm__ uint8_t *atten_mask,
                                     __gm__ uint8_t *forward_res, __gm__ uint8_t *softmax_max, __gm__ uint8_t *softmax_sum,
                                     __gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv, __gm__ uint8_t *dk_1, 
                                     __gm__ uint8_t *dv_1, __gm__ uint8_t *workspace,
                                     const FusedFloydAttentionGradTilingDataS1s2Bn2gs1s2 *__restrict ordTilingData,
                                     TPipe *pipe_in)
{
    keyGm.SetGlobalBuffer((__gm__ T1 *)key);
    valueGm.SetGlobalBuffer((__gm__ T1 *)value);
    keyGm1.SetGlobalBuffer((__gm__ T1 *)key_1);
    valueGm1.SetGlobalBuffer((__gm__ T1 *)value_1);
    dxGm.SetGlobalBuffer((__gm__ T1 *)dx);
    queryGm.SetGlobalBuffer((__gm__ T1 *)query);
    forwardResGm.SetGlobalBuffer((__gm__ T1 *)forward_res);

    attenMaskU8Gm.SetGlobalBuffer((__gm__ uint8_t *)atten_mask);
    softmaxMaxGm.SetGlobalBuffer((__gm__ float *)softmax_max);
    softmaxSumGm.SetGlobalBuffer((__gm__ float *)softmax_sum);
    dvGm.SetGlobalBuffer((__gm__ float *)dv);

    // init current core tilingInfo
    cBlockIdx = GetBlockIdx();
    TilingData = ordTilingData;
    pipe = pipe_in;
    coreNum = TilingData->s1s2BNGS1S2BaseParams.coreNum;

    // shape info
    b = TilingData->s1s2BNGS1S2BaseParams.b;
    n2 = TilingData->s1s2BNGS1S2BaseParams.n2;
    g = TilingData->s1s2BNGS1S2BaseParams.g;

    s1 = TilingData->s1s2BNGS1S2BaseParams.s1;
    s2 = TilingData->s1s2BNGS1S2BaseParams.s2;
    d = TilingData->s1s2BNGS1S2BaseParams.d;
    attenMaskDimS2 = TilingData->s1s2BNGS1S2BaseParams.s2;

    s1Token = TilingData->s1s2BNGS1S2BaseParams.s1Token;
    s2Token = TilingData->s1s2BNGS1S2BaseParams.s2Token;
    actualCalcS1Token = s1Token;
    actualCalcS2Token = s2Token;
    sparseMode = TilingData->s1s2BNGS1S2BaseParams.sparseMode;

    // split info
    s1Outer = TilingData->s1s2BNGS1S2SplitCoreParams.s1Outer;
    bmmS1base = TilingData->s1s2BNGS1S2SplitCoreParams.bmmS1base;
    s1CvRatio = TilingData->s1s2BNGS1S2SplitCoreParams.s1CvRatio;
    s1CvInner = TilingData->s1s2BNGS1S2SplitCoreParams.s1CvInner;
    s1CvTail = TilingData->s1s2BNGS1S2SplitCoreParams.s1CvTail;
    s2Outer = TilingData->s1s2BNGS1S2SplitCoreParams.s2Outer;
    s2CvRatio = TilingData->s1s2BNGS1S2SplitCoreParams.s2CvRatio;
    s2Inner = TilingData->s1s2BNGS1S2SplitCoreParams.s2Inner;
    s2CvInner = s2Inner * s2CvRatio;

    sfmgdOuter = TilingData->s1s2BNGS1S2SplitCoreParams.sfmgdOuter;
    sfmgdInner = TilingData->s1s2BNGS1S2SplitCoreParams.sfmgdFactor;
    sfmgdTail = TilingData->s1s2BNGS1S2SplitCoreParams.sfmgdTail;
    sfmgdTailAlign = (sfmgdTail + input_block_num - 1) / input_block_num * input_block_num;

    gOuter = TilingData->s1s2BNGS1S2SplitCoreParams.gOuter;
    gInner = TilingData->s1s2BNGS1S2SplitCoreParams.gInner;
    gTail = TilingData->s1s2BNGS1S2SplitCoreParams.gTail;
    // no sparse blockouter ceil to even
    blockOuter = (TilingData->s1s2BNGS1S2SplitCoreParams.blockOuter + 1) / 2 * 2;

    baseMN = TilingData->s1s2BNGS1S2SplitCoreParams.baseMN;
    cubeBaseMN = gInner * s1CvInner * s2CvInner;
    singleBaseMN = s1CvInner * s2CvInner;

    blockStarts = TilingData->s1s2BNGS1S2BlockNumList.blockStarts;
    blockEnds = TilingData->s1s2BNGS1S2BlockNumList.blockEnds;

    actual_seq_qlen_addr = nullptr;
    actual_seq_kvlen_addr = nullptr;
    dropBitMode = s2 % 8 == 0;
    keepProb = TilingData->s1s2BNGS1S2BaseParams.keepProb;

    // idx info
    n2gs1os2o = n2 * gOuter * s1Outer * s2Outer;
    gs1os2o = gOuter * s1Outer * s2Outer;
    s1os2o = s1Outer * s2Outer;

    int64_t maskPreBlockTotal = TilingData->preTilingData.maskPreBlockTotal;
    int64_t qPostBlockTotal = TilingData->postTilingData.qSizeAlign;
    int64_t kvPostBlockTotal = TilingData->postTilingData.kvSizeAlign;
    int64_t k1v1PostBlockTotal = TilingData->postTilingData.k1v1SizeAlign;

    // init workspace address
    syncGlobal.SetGlobalBuffer((__gm__ int32_t *)workspace);
    int64_t workspaceOffsets = SYNC_GLOBAL_WORKSPACE_SIZE;
    dqWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + workspaceOffsets / sizeof(T2));
    workspaceOffsets =
        (workspaceOffsets + qPostBlockTotal * sizeof(float) + ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
    dkWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + workspaceOffsets / sizeof(T2));
    workspaceOffsets =
        (workspaceOffsets + kvPostBlockTotal * sizeof(float) + ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
    dvWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + workspaceOffsets / sizeof(T2));
    workspaceOffsets =
        (workspaceOffsets + kvPostBlockTotal * sizeof(float) + ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
    dk1WorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + workspaceOffsets / sizeof(T2));
    workspaceOffsets =
        (workspaceOffsets + k1v1PostBlockTotal * sizeof(float) + ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
    dv1WorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + workspaceOffsets / sizeof(T2));
    workspaceOffsets =
        (workspaceOffsets + k1v1PostBlockTotal * sizeof(float) + ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;

    int64_t pseInnerAlibiSize = TilingData->s1s2BNGS1S2BaseParams.pseAlibiBaseS1 *
                                this->TilingData->s1s2BNGS1S2BaseParams.pseAlibiBaseS2 * sizeof(half);
    int64_t pseAlibiOffset =  CeilDiv(pseInnerAlibiSize, 512) * 512;

    // matmal1 and matmal2 workspace size
    matmalWorkspaceSize = cubeBaseMN * sizeof(float);
    mm1WorkspaceGm.SetGlobalBuffer((__gm__ T2 *)(workspace + workspaceOffsets + cBlockIdx * matmalWorkspaceSize));
    mm2WorkspaceGm.SetGlobalBuffer(
        (__gm__ T2 *)(workspace + workspaceOffsets + coreNum * matmalWorkspaceSize + cBlockIdx * matmalWorkspaceSize));

    // drop workspace offset
    workspaceOffsets = (workspaceOffsets + coreNum * cubeBaseMN * sizeof(float) * INPUT_NUMS + ADDR_ALIGN_SIZE) /
                       ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
    dropWorkSpaceGm.SetGlobalBuffer((__gm__ T1 *)workspace + workspaceOffsets / sizeof(T1));

    // mul workspace offset
    workspaceOffsets = (workspaceOffsets + coreNum * cubeBaseMN * sizeof(T1) * 2 + ADDR_ALIGN_SIZE) /
                       ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
    mulWorkSpaceGm.SetGlobalBuffer((__gm__ T1 *)workspace + workspaceOffsets / sizeof(T1));

    uint64_t pseAlibiAddr = (workspaceOffsets + coreNum * cubeBaseMN * sizeof(T1) * 2 + ADDR_ALIGN_SIZE) /
                       ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
    this->pseAlibiGm.SetGlobalBuffer((__gm__ half*)(workspace + pseAlibiAddr + cBlockIdx * pseAlibiOffset));

    InitOutput<int32_t>(syncGlobal[GetBlockIdx() * 8], 8, 0);

    pipe->InitBuffer(ubBuffer, 150 * 1024);
    pipe->InitBuffer(tmpBuffer, 33 * 1024);
    pipe->InitBuffer(vecClc3, 8 * 1024);
}


template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT, const uint32_t TND_S1_PP>
__aicore__ inline void
FusedFloydAttentionGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
                                    MM2_OUT_FORMAT, TND_S1_PP>::ComputeBMMK1(uint32_t aTensorOffsetCv, uint32_t bTensorOffsetCv,
                                                                       uint32_t s1StrideSize, uint32_t s2StrideSize, uint32_t gExtend, uint32_t s1Extend,
                                                                       uint32_t s2Extend, uint32_t s2ExtendAlign)
{
    if ASCEND_IS_AIV {
        for (uint32_t i = 0; i < s1Extend / bmmS1base; i++) { //s1 batch 轴切分base
            bmm1k1.SetOrgShape(g, s2, s2StrideSize, s2StrideSize, s2ExtendAlign);//M = g, N = s2, Ka = d, Kb=d, Kc = s2CvExtendAlign
            bmm1k1.SetTail(gExtend, s2Extend, d);  // M N K
            bmm1k1.SetTensorA(queryGm[aTensorOffsetCv + i * bmmS1base * d]);
            bmm1k1.SetTensorB(keyGm1[bTensorOffsetCv + i * bmmS1base * d], true);
            bmm1k1.template IterateBatch(mm2WorkspaceGm[i * bmmS1base * s2ExtendAlign],
                                        static_cast<uint32_t>(bmmS1base), 
                                        static_cast<uint32_t>(bmmS1base),
                                        false,
                                        static_cast<uint32_t>(0), 
                                        static_cast<uint32_t>(0), 
                                        static_cast<uint32_t>(0), 
                                        false,
                                        static_cast<uint8_t>(1)
                                        );
            bmm1k1.End();

            bmm1k1.SetTensorA(dxGm[aTensorOffsetCv + i * bmmS1base * d]);
            bmm1k1.SetTensorB(valueGm1[bTensorOffsetCv + i * bmmS1base * d], true);
            bmm1k1.template IterateBatch(mm1WorkspaceGm[i * bmmS1base * s2ExtendAlign],
                                        static_cast<uint32_t>(bmmS1base), 
                                        static_cast<uint32_t>(bmmS1base),
                                        false,
                                        static_cast<uint32_t>(0), 
                                        static_cast<uint32_t>(0), 
                                        static_cast<uint32_t>(0), 
                                        false,
                                        static_cast<uint8_t>(1)
                                        );
            bmm1k1.End();
        }
    }
}


template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT, const uint32_t TND_S1_PP>
__aicore__ inline void
FusedFloydAttentionGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
                                    MM2_OUT_FORMAT, TND_S1_PP>::ComputeBMM2K1(uint32_t aTensorOffsetCv, uint32_t bTensorOffsetCv, uint32_t cTensorOffsetCv, 
                                                                        uint32_t gExtend, uint32_t s1Extend, 
                                                                       uint32_t s2Extend, uint32_t s2ExtendAlign)
{
    if ASCEND_IS_AIV {
        for (uint32_t i = 0; i < s1Extend / bmmS1base; i++) {
            bmm2k1.SetOrgShape(g, d, s2ExtendAlign);//M = g, N = d, Ka = s2, Kb=s2, Kc = d
            bmm2k1.SetTail(gExtend, -1, s2Extend);  // M N K
            bmm2k1.SetTensorA(mulWorkSpaceGm[aTensorOffsetCv + i * bmmS1base * s2ExtendAlign]);
            bmm2k1.SetTensorB(keyGm1[bTensorOffsetCv + i * bmmS1base * d]);
            bmm2k1.template IterateBatch<true, false>(dqWorkSpaceGm[cTensorOffsetCv + i * bmmS1base * d],
                                        static_cast<uint32_t>(bmmS1base), 
                                        static_cast<uint32_t>(bmmS1base),
                                        false,
                                        static_cast<uint32_t>(0), 
                                        static_cast<uint32_t>(0), 
                                        static_cast<uint32_t>(0), 
                                        false,
                                        static_cast<uint8_t>(1));
        }
    }
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT, const uint32_t TND_S1_PP>
__aicore__ inline void
FusedFloydAttentionGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
                                    MM2_OUT_FORMAT, TND_S1_PP>::ComputeBMM3K1V1(uint32_t aTensorOffsetCv, uint32_t bTensorOffsetCv, uint32_t cTensorOffsetCv, 
                                                                        uint32_t gExtend, uint32_t s1Extend, 
                                                                       uint32_t s2Extend, uint32_t s2ExtendAlign)
{
    if ASCEND_IS_AIV {
        for (uint32_t i = 0; i < s1Extend / bmmS1base; i++) {
            bmm3k1v1.SetOrgShape(s2ExtendAlign, d, g);//M = s2, N = d, Ka = g
            bmm3k1v1.SetTail(s2Extend, -1, gExtend);  // M N K
            bmm3k1v1.SetTensorA(dropWorkSpaceGm[aTensorOffsetCv + i * bmmS1base * s2ExtendAlign], true);
            bmm3k1v1.SetTensorB(dxGm[bTensorOffsetCv + i * bmmS1base * d]);
            bmm3k1v1.template IterateBatch<true, false>(dv1WorkSpaceGm[cTensorOffsetCv + i * bmmS1base * d],
                                        static_cast<uint32_t>(bmmS1base), 
                                        static_cast<uint32_t>(bmmS1base),
                                        false,
                                        static_cast<uint32_t>(0), 
                                        static_cast<uint32_t>(0), 
                                        static_cast<uint32_t>(0), 
                                        false,
                                        static_cast<uint8_t>(1)
                                        );

            bmm3k1v1.SetOrgShape(s2ExtendAlign, d, g);//M = s2, N = d, Ka = g
            bmm3k1v1.SetTail(s2Extend, -1, gExtend);  // M N K
            bmm3k1v1.SetTensorA(mulWorkSpaceGm[aTensorOffsetCv + i * bmmS1base * s2ExtendAlign], true);
            bmm3k1v1.SetTensorB(queryGm[bTensorOffsetCv + i * bmmS1base * d]);
            bmm3k1v1.template IterateBatch<true, false>(dk1WorkSpaceGm[cTensorOffsetCv + i * bmmS1base * d],
                                        static_cast<uint32_t>(bmmS1base), 
                                        static_cast<uint32_t>(bmmS1base),
                                        false,
                                        static_cast<uint32_t>(0), 
                                        static_cast<uint32_t>(0), 
                                        static_cast<uint32_t>(0), 
                                        false,
                                        static_cast<uint8_t>(1)
                                        );
        }
    }
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT, const uint32_t TND_S1_PP>
__aicore__ inline void
FusedFloydAttentionGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
                                    MM2_OUT_FORMAT, TND_S1_PP>::CopyInSoftMaxGrad(LocalTensor<T1> &dstTensor,
                                                                       LocalTensor<T1> &dstTensor2,
                                                                       int64_t softmaxGradFrontOffset,
                                                                       uint32_t s1Extend, uint32_t dExtend,
                                                                       uint32_t dExtendAlign)
{
    int64_t transpse_stride = 0;
    transpse_stride = (d - dExtend) * sizeof(T1);

    // 如果n2*g*d = dExtend,但是不对齐，理应走DataCopyPad给后面补0
    if (transpse_stride == 0 && (dExtend % 16 == 0)) {
        DataCopy(dstTensor, forwardResGm[softmaxGradFrontOffset], s1Extend * dExtend);
        DataCopy(dstTensor2, dxGm[softmaxGradFrontOffset], s1Extend * dExtend);
    } else {
        DataCopyPad(dstTensor, forwardResGm[softmaxGradFrontOffset],
                    {static_cast<uint16_t>(s1Extend), static_cast<uint32_t>(dExtend * sizeof(T1)),
                     static_cast<uint32_t>(transpse_stride), 0, 0},
                    {true, 0, static_cast<uint8_t>((dExtendAlign - dExtend)), 0});
        DataCopyPad(dstTensor2, dxGm[softmaxGradFrontOffset],
                    {static_cast<uint16_t>(s1Extend), static_cast<uint32_t>(dExtend * sizeof(T1)),
                     static_cast<uint32_t>(transpse_stride), 0, 0},
                    {true, 0, static_cast<uint8_t>((dExtendAlign - dExtend)), 0});
    }
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT, const uint32_t TND_S1_PP>
__aicore__ inline void
FusedFloydAttentionGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
                                    MM2_OUT_FORMAT, TND_S1_PP>::CalcSoftMaxGrad(LocalTensor<float> &sfmgClc3,
                                                                     int64_t aTensorOffset, uint32_t s1Extend)
{
    LocalTensor<float> sfmgClc1 = ubBuffer.GetWithOffset<float>(32 * 1024 / sizeof(T2), 0);
    LocalTensor<float> sfmgClc2 = ubBuffer.GetWithOffset<float>(32 * 1024 / sizeof(T2), 32 * 1024);
    Duplicate<float>(sfmgClc3, 0.0, s1Extend * 32 / sizeof(float));

    event_t mte2WaitV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    for (uint32_t sfmgdIdx = 0; sfmgdIdx < sfmgdOuter; sfmgdIdx++) {
        LocalTensor<T1> vecInBuffer;
        LocalTensor<T1> vecInBuffer2;
        if constexpr (!IsSameType<T1, float>::value) {
            vecInBuffer = ubBuffer.GetWithOffset<T1>(16 * 1024 / sizeof(T1), 64 * 1024);
            vecInBuffer2 = ubBuffer.GetWithOffset<T1>(16 * 1024 / sizeof(T1), 80 * 1024);
        }
        LocalTensor<uint8_t> vecOutBuffer = ubBuffer.GetWithOffset<uint8_t>(32 * 1024 / sizeof(uint8_t), 96 * 1024);
        LocalTensor<T2> softmaxGradTmp = ubBuffer.GetWithOffset<T2>(8 * 1024 / sizeof(T2), 128 * 1024);
        int64_t softmaxGradFrontOffset = aTensorOffset + sfmgdIdx * sfmgdInner;
        uint32_t dExtend = (sfmgdIdx == sfmgdOuter - 1) ? sfmgdTail : sfmgdInner;
        uint32_t dExtendAlign = (sfmgdIdx == sfmgdOuter - 1) ? sfmgdTailAlign : sfmgdInner;
        bool isBasicBlock = (s1Extend % 8 == 0) && (dExtend % 64 == 0);

        if (sfmgdIdx > 0) {
            WaitFlag<HardEvent::V_MTE2>(mte2WaitV);
        }
        if constexpr (!IsSameType<T1, float>::value) {
            CopyInSoftMaxGrad(vecInBuffer, vecInBuffer2, softmaxGradFrontOffset, s1Extend, dExtend, dExtendAlign);
        } else {
            CopyInSoftMaxGrad(sfmgClc1, sfmgClc2, softmaxGradFrontOffset, s1Extend, dExtend, dExtendAlign);
        }
        event_t vWaitMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(vWaitMte2);
        WaitFlag<HardEvent::MTE2_V>(vWaitMte2);
        if constexpr (!IsSameType<T1, float>::value) {
            Cast(sfmgClc1, vecInBuffer, RoundMode::CAST_NONE, s1Extend * dExtendAlign);
            Cast(sfmgClc2, vecInBuffer2, RoundMode::CAST_NONE, s1Extend * dExtendAlign);
            PipeBarrier<PIPE_V>();
        }
        uint32_t shapeArray[2];
        shapeArray[0] = s1Extend;
        shapeArray[1] = dExtendAlign;
        sfmgClc1.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
        sfmgClc2.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
        uint32_t shapeArray1[2];
        shapeArray1[0] = s1Extend;
        shapeArray1[1] = 32 / sizeof(float);
        softmaxGradTmp.SetShapeInfo(ShapeInfo(2, shapeArray1, DataFormat::ND));

        if (isBasicBlock) {
            SoftmaxGradFront<float, true>(softmaxGradTmp, sfmgClc1, sfmgClc2, vecOutBuffer,
                                          TilingData->softmaxGradTilingData);
        } else {
            SoftmaxGradFront<float, false>(softmaxGradTmp, sfmgClc1, sfmgClc2, vecOutBuffer,
                                           TilingData->softmaxGradTilingData);
        }
        PipeBarrier<PIPE_V>();
        Add(sfmgClc3, softmaxGradTmp, sfmgClc3, s1Extend * 32 / sizeof(float));
        if (sfmgdIdx < (sfmgdOuter - 1)) {
            SetFlag<HardEvent::V_MTE2>(mte2WaitV);
        }
    }
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(mte2WaitV);
    PipeBarrier<PIPE_ALL>();
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT, const uint32_t TND_S1_PP>
__aicore__ inline void
FusedFloydAttentionGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
                                    MM2_OUT_FORMAT, TND_S1_PP>::CopyInSoftMax(LocalTensor<float> &dstTensor, uint32_t s1Extend,
                                                                   uint32_t softMaxOffset)
{
    DataCopyPad(dstTensor, softmaxSumGm[softMaxOffset], {1, static_cast<uint16_t>(s1Extend * 32), 0, 0},
                {false, 0, 0, 0});
    DataCopyPad(dstTensor[s1Extend * 32 / sizeof(float)], softmaxMaxGm[softMaxOffset],
                {1, static_cast<uint16_t>(s1Extend * 32), 0, 0}, {false, 0, 0, 0});

    // AscendC::DumpTensor(dstTensor, 1111111, 32);
    // AscendC::DumpTensor(dstTensor[s1Extend * 32 / sizeof(float)], 222222, 32);



}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT, const uint32_t TND_S1_PP>
__aicore__ inline void
FusedFloydAttentionGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
                                    MM2_OUT_FORMAT, TND_S1_PP>::CalcSoftMax(LocalTensor<T2> &dstTensor,
                                                                 LocalTensor<float> &srcTensor, uint32_t s1Extend,
                                                                 uint32_t s2Extend, uint32_t s2ExtendAlign,
                                                                 const SoftMaxTiling &tiling)
{
    bool isBasicBlock = (s1Extend % 8 == 0) && (s2Extend % 64 == 0);

    if (isBasicBlock) {
        LocalTensor<uint8_t> vecOutBuffer = tmpBuffer.Get<uint8_t>();
        uint32_t shapeArray1[2];
        shapeArray1[0] = s1Extend;
        shapeArray1[1] = s2Extend;
        dstTensor.SetShapeInfo(ShapeInfo(2, shapeArray1, DataFormat::ND));
        SimpleSoftMax<T2, true, true>(dstTensor, srcTensor, srcTensor[s1Extend * 32 / sizeof(float)], dstTensor,
                                      vecOutBuffer, tiling);
    } else {
        LocalTensor<T2> vecOutBuffer = tmpBuffer.Get<T2>();
        uint32_t sub_block_count = (s2Extend + cal_repeat_num - 1) / cal_repeat_num;

        for (uint32_t subIdx = 0; subIdx < sub_block_count; subIdx++) {
            uint32_t subMaskCount =
                (subIdx == sub_block_count - 1) ? (s2Extend - subIdx * cal_repeat_num) : cal_repeat_num;
            Sub(dstTensor[subIdx * cal_repeat_num], dstTensor[subIdx * cal_repeat_num], srcTensor[s1Extend * 8],
                subMaskCount, s1Extend,
                {static_cast<uint8_t>(1), static_cast<uint8_t>(1), 0, static_cast<uint8_t>(s2ExtendAlign / 8),
                 static_cast<uint8_t>(s2ExtendAlign / 8), 1});

            PipeBarrier<PIPE_V>();
            Exp(vecOutBuffer[subIdx * cal_repeat_num], dstTensor[subIdx * cal_repeat_num], subMaskCount, s1Extend,
                {static_cast<uint8_t>(1), static_cast<uint8_t>(1), static_cast<uint8_t>(s2ExtendAlign / 8),
                 static_cast<uint8_t>(s2ExtendAlign / 8)});
            
            PipeBarrier<PIPE_V>();
            Div(dstTensor[subIdx * cal_repeat_num], vecOutBuffer[subIdx * cal_repeat_num], srcTensor, subMaskCount,
                s1Extend,
                {static_cast<uint8_t>(1), static_cast<uint8_t>(1), 0, static_cast<uint8_t>(s2ExtendAlign / 8),
                 static_cast<uint8_t>(s2ExtendAlign / 8), 1});

            PipeBarrier<PIPE_V>();
        }
    }
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT, const uint32_t TND_S1_PP>
__aicore__ inline void FusedFloydAttentionGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT,
                                                           INPUT_LAYOUT, MM2_OUT_FORMAT, TND_S1_PP>::Process()
{
    int64_t preIndex = cBlockIdx;
    int64_t total = static_cast<int64_t>(b) * n2 * gOuter * s1Outer * s2Outer;
    if (cBlockIdx >= total) {
        return;
    }

    IsCubeBlockNeedCompute(preIndex);
    preS2CvBegin = nextS2CvBegin;
    preS2CvEnd = nextS2CvEnd;

    int64_t totalTemp = total + blockOuter;
    for (int64_t blockInnerIdx = cBlockIdx + blockOuter; blockInnerIdx < totalTemp; blockInnerIdx += blockOuter) {
        if (IsCubeBlockNeedCompute(blockInnerIdx) || (blockInnerIdx >= total)) {
            int64_t nextIndex = blockInnerIdx;
            if (blockInnerIdx >= total) {
                nextIndex = 0u;
            }
            Compute(preIndex, nextIndex);
            preIndex = nextIndex;
            preS2CvBegin = nextS2CvBegin;
            preS2CvEnd = nextS2CvEnd;
        }
    }
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT, const uint32_t TND_S1_PP>
__aicore__ inline bool
FusedFloydAttentionGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
                                    MM2_OUT_FORMAT, TND_S1_PP>::IsCubeBlockNeedCompute(int64_t baseIdx)
{
    uint32_t gDimTail = baseIdx % s1os2o;   // s2idx * s1Outer + s1idx

    uint32_t s2oCvDimIdx = gDimTail / s1Outer; 
    uint32_t s1oDimIdx = gDimTail % s1Outer;

    uint32_t s2IdxLeft = s2oCvDimIdx * s2CvInner;
    uint32_t s2IdxRight = (s2oCvDimIdx + 1) * s2CvInner < s2 ? (s2oCvDimIdx + 1) * s2CvInner : s2;
    
    nextS2CvBegin = s2IdxLeft; //换batch的时候可能出现尾块
    nextS2CvEnd = s2IdxRight;
    return true;
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT, const uint32_t TND_S1_PP>
__aicore__ inline void
FusedFloydAttentionGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
                                    MM2_OUT_FORMAT, TND_S1_PP>::NZCopyIn(int64_t mmAddr, GlobalTensor<T2> &mmWspGm,
                                                              LocalTensor<T2> &mmTensorCurr, uint32_t s1VecSize,
                                                              uint32_t s2VecSize)
{
    /*
    Func:
    MM输出NZ数据，数据搬运进UB
    */
    DataCopyParams intriParams;
    intriParams.blockCount = s2VecSize / C0_SIZE;
    intriParams.blockLen = s1VecSize * C0_SIZE / cal_block_num;
    intriParams.srcStride = s1CvExtend * C0_SIZE / cal_block_num - intriParams.blockLen;
    intriParams.dstStride = 1;
    DataCopy(mmTensorCurr, mmWspGm[mmAddr], intriParams);
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT, const uint32_t TND_S1_PP>
__aicore__ inline void
FusedFloydAttentionGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
                                    MM2_OUT_FORMAT, TND_S1_PP>::NZ2ND(LocalTensor<T2> &mmTensorCurr, LocalTensor<T2> &tmpTensor,
                                                           uint32_t s1VecSize, uint32_t s2VecSize)
{
    /*
    Func:
    将NZ转为ND
    */
    CopyRepeatParams nz2ndParams;
    nz2ndParams.srcStride = s1VecSize * C0_SIZE / cal_block_num + 1;
    nz2ndParams.dstStride = C0_SIZE / cal_block_num;
    nz2ndParams.srcRepeatSize = C0_SIZE / cal_block_num;
    nz2ndParams.dstRepeatSize = s2VecSize / cal_block_num;

    uint16_t c0_repeat = C0_SIZE / cal_block_num;
    uint16_t c1_repeat = s2VecSize / C0_SIZE / VEC_REPEAT;
    uint16_t c1_remain = s2VecSize / C0_SIZE % VEC_REPEAT;
    uint16_t n_repeat = s1VecSize;
    for (uint16_t i = 0; i < c0_repeat; ++i) {
        for (uint16_t j = 0; j < c1_repeat; ++j) {
            Copy(mmTensorCurr[i * cal_block_num + j * VEC_REPEAT * C0_SIZE],
                 tmpTensor[i * cal_block_num + j * VEC_REPEAT * (s1VecSize * C0_SIZE + cal_block_num)],
                 VEC_REPEAT * cal_block_num, n_repeat, nz2ndParams);
        }
        if (c1_remain > 0) {
            Copy(mmTensorCurr[i * cal_block_num + c1_repeat * VEC_REPEAT * C0_SIZE],
                 tmpTensor[i * cal_block_num + c1_repeat * VEC_REPEAT * (s1VecSize * C0_SIZE + cal_block_num)],
                 VEC_REPEAT * c1_remain, n_repeat, nz2ndParams);
        }
    }
    PipeBarrier<PIPE_V>();
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT, const uint32_t TND_S1_PP>
__aicore__ inline void
FusedFloydAttentionGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
                                    MM2_OUT_FORMAT, TND_S1_PP>::ND2NZ(LocalTensor<T1> &mmTensorCurr, LocalTensor<T1> &tmpTensor,
                                                           uint32_t s1VecSize, uint32_t s2VecSize)
{
    /*
    Func:
    将ND转为NZ
    */

    CopyRepeatParams nd2nzParams;
    nd2nzParams.dstStride = s1VecSize * C0_SIZE / input_block_num + 1;
    nd2nzParams.srcStride = C0_SIZE / input_block_num;
    nd2nzParams.dstRepeatSize = C0_SIZE / input_block_num;
    nd2nzParams.srcRepeatSize = s2VecSize / input_block_num;

    uint16_t c1_repeat = s2VecSize / C0_SIZE / VEC_REPEAT;
    uint16_t c1_remain = s2VecSize / C0_SIZE % VEC_REPEAT;

    auto mmTensorCurrTmp = mmTensorCurr.template ReinterpretCast<half>();
    auto tmpTensorTmp = tmpTensor.template ReinterpretCast<half>();

    for (uint16_t j = 0; j < c1_repeat; ++j) {
        Copy(mmTensorCurrTmp[j * 8 * (s1VecSize + 1) * C0_SIZE], tmpTensorTmp[j * 128], VEC_REPEAT * input_block_num,
             s1VecSize, nd2nzParams);
    }

    if (c1_remain > 0) {
        Copy(mmTensorCurrTmp[c1_repeat * 8 * (s1VecSize + 1) * C0_SIZE], tmpTensorTmp[c1_repeat * 128],
             input_block_num * c1_remain, s1VecSize, nd2nzParams);
    }
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT, const uint32_t TND_S1_PP>
__aicore__ inline void FusedFloydAttentionGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT,
                                                           INPUT_LAYOUT, MM2_OUT_FORMAT, TND_S1_PP>::InitIndex(int64_t index)
{
    baseIdx = index;
    bDimIdx = baseIdx / n2gs1os2o;
    uint32_t bDimTail = baseIdx % n2gs1os2o;
    n2DimIdx = bDimTail / gs1os2o;
    uint32_t n2DimTail = bDimTail % gs1os2o;
    gDimIdx = n2DimTail / s1os2o;
    uint32_t gDimTail = n2DimTail % s1os2o;
    s2oCvDimIdx = gDimTail / s1Outer;
    s1oDimIdx = gDimTail % s1Outer;
    s1CvExtend = (s1oDimIdx == s1Outer - 1) ? s1CvTail : s1CvInner;
    gExtend = gDimIdx == gOuter -1 ? gTail : gInner;
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT, const uint32_t TND_S1_PP>
__aicore__ inline void
FusedFloydAttentionGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
                                    MM2_OUT_FORMAT, TND_S1_PP>::SubGrapA(int64_t curIdx, int64_t curS1Idx, int64_t curS2Idx, int64_t LoopGIdx,
                                    EvenvIdList &eventIdList)
{
    int64_t actualS1Len;
    int64_t actualS2Len;
    bool isPing = ((curIdx % 2) == 0);
    s2Extend = (curS2Idx == s2VecLoop - 1) ? (s2CvExtend - (s2VecLoop - 1) * s2VecSize) : s2VecSize;
    s2ExtendAlign = (s2Extend + 15) / 16 * 16;
    uint32_t s2VBegin = preS2CvBegin + curS2Idx * s2VecSize;

    event_t curEventIdMte2WaitMte3 = isPing ?
                                     eventIdList.structMte2WaitMte3Ping : eventIdList.structMte2WaitMte3Pong;
    uint32_t ubBufferOffset = isPing ? 0 : DbBegin;

    if (curIdx > 1) {
        WaitFlag<HardEvent::MTE3_MTE2>(curEventIdMte2WaitMte3);
    }

    LocalTensor<float> vecInBuffer3 =
        ubBuffer.GetWithOffset<float>(8 * 1024 / sizeof(float), ubBufferOffset + T2BlockBegin);
    int64_t softMaxOffset = 0;

    softMaxOffset = (((static_cast<int64_t>(bDimIdx) * n2 + n2DimIdx) * g + gDimIdx * gInner + LoopGIdx) * s1 + s1oDimIdx * s1CvInner +
                        curS1Idx * s1VecSize) * 32 / sizeof(float);
    
    CopyInSoftMax(vecInBuffer3, s1ExtendSubGraph, softMaxOffset);

    LocalTensor<T1> pseUbT1 = ubBuffer.GetWithOffset<T1>(16 * 1024 / sizeof(T1), ubBufferOffset + T1Begin);
    LocalTensor<half> pseUb = pseUbT1.template ReinterpretCast<half>();

    LocalTensor<uint8_t> attenMaskUbuint8 =
        ubBuffer.GetWithOffset<uint8_t>(8 * 1024 / sizeof(uint8_t), ubBufferOffset + BoolBegin);
    int64_t attenMaskOffsetPre = 0;
    bool prefixCompressCanSimplify = false;

    if constexpr (IS_ATTEN_MASK == ENABLE) {
        int64_t attenMaskOffset = (((static_cast<int64_t>(bDimIdx)) * g + gDimIdx * gInner + LoopGIdx) * s2 + s2VBegin);
        AttenMaskBoolCopyIn(attenMaskUbuint8, attenMaskU8Gm, attenMaskOffset, s1ExtendSubGraph, s2Extend, s2, 3);
    }

    LocalTensor<uint8_t> vecInDropBuffer =
        ubBuffer.GetWithOffset<uint8_t>(8 * 1024 / sizeof(uint8_t), ubBufferOffset + U8Begin);

    LocalTensor<float> vecClc2Buffer =
        ubBuffer.GetWithOffset<float>(32 * 1024 / sizeof(float), ubBufferOffset + T2Begin);
    if constexpr (MM_OUT_FORMAT == CubeFormat::ND) {
        if (s2VecLoop == 1) {
            DataCopy(vecClc2Buffer, mm2WorkspaceGm[curS1Idx * s1VecSize * s2ExtendAlign + LoopGIdx * singleBaseMN],
                     s1ExtendSubGraph * s2ExtendAlign);
        } else {
            DataCopyPad(vecClc2Buffer, mm2WorkspaceGm[curS1Idx * s1VecSize * s2CvExtendAlign + curS2Idx * s2VecSize + LoopGIdx * singleBaseMN],
                        {static_cast<uint16_t>(s1ExtendSubGraph), static_cast<uint16_t>(s2ExtendAlign * sizeof(float)),
                         static_cast<uint16_t>((s2CvExtendAlign - s2ExtendAlign) * sizeof(float)), 0},
                        {false, 0, 0, 0});
        }
        event_t vWaitMte2 = isPing ? eventIdList.structVWaitMte2Ping : eventIdList.structVWaitMte2Pong;
        SetFlag<HardEvent::MTE2_V>(vWaitMte2);
        WaitFlag<HardEvent::MTE2_V>(vWaitMte2);
    } else {
        int64_t mmAddr = curS1Idx * s1VecSize * C0_SIZE + curS2Idx * s1CvExtend * s2VecSizeAlign + LoopGIdx * s2CvExtendAlign * s1CvExtend;
        NZCopyIn(mmAddr, mm2WorkspaceGm, vecClc2Buffer, s1VecSize, s2ExtendAlign);
        event_t vWaitMte2 = isPing ? eventIdList.structVWaitMte2Ping : eventIdList.structVWaitMte2Pong;
        SetFlag<HardEvent::MTE2_V>(vWaitMte2);
        SetFlag<HardEvent::MTE2_V>(vWaitMte2);
        auto tmpTensor = tmpBuffer.Get<T2>();
        DataCopy(tmpTensor, vecClc2Buffer, s1VecSize * s2ExtendAlign + s2ExtendAlign / C0_SIZE * VEC_REPEAT);
        PipeBarrier<PIPE_V>();
        NZ2ND(vecClc2Buffer, tmpTensor, s1VecSize, s2ExtendAlign);
    }

    ///////////////////////////////////////////////////////////////
    // pse + muls
    PipeBarrier<PIPE_V>();
    Muls(vecClc2Buffer, vecClc2Buffer, (T2)(TilingData->s1s2BNGS1S2BaseParams.scaleValue),
        s1ExtendSubGraph * s2ExtendAlign);

    ///////////////////////////////////////////////////////////////
    // pse shape  0--BN2G1S2    1--BN2GS1S2
    ///////////////////////////////////////////////////////////////
    // attenMask
    ///////////////////////////////////////////////////////////////
    // attenMaskOffset     attenMaskShapeType  0--111S1S2        1--B11S1S2         2--BN2GS1S2
    if constexpr (IS_ATTEN_MASK == ENABLE) {
        PipeBarrier<PIPE_V>();
        CalcAttenMaskBool(vecClc2Buffer, attenMaskUbuint8, s1ExtendSubGraph, s2ExtendAlign);
    }

    ///////////////////////////////////////////////////////////////
    // simpleSoftMax
    ///////////////////////////////////////////////////////////////
    PipeBarrier<PIPE_V>();
    CalcSoftMax(vecClc2Buffer, vecInBuffer3, s1ExtendSubGraph, s2Extend, s2ExtendAlign, TilingData->softmaxTilingData);

    ///////////////////////////////////////////////////////////////
    // dropout
    ///////////////////////////////////////////////////////////////
    LocalTensor<T2> vecDropBuffer = vecClc2Buffer;

    ///////////////////////////////////////////////////////////////
    // cast fp322bf16
    ///////////////////////////////////////////////////////////////
    LocalTensor<T1> vecOut1Buffer1;
    if constexpr (!IsSameType<T1, float>::value) {
        vecOut1Buffer1 = ubBuffer.GetWithOffset<T1>(17 * 1024 / sizeof(T1), ubBufferOffset + T1Begin);
        PipeBarrier<PIPE_V>();
        Cast(vecOut1Buffer1, vecDropBuffer, RoundMode::CAST_ROUND, s1ExtendSubGraph * s2ExtendAlign);
    }
    if constexpr (MM_OUT_FORMAT == CubeFormat::NZ) {
        PipeBarrier<PIPE_V>();
        LocalTensor<T1> tmpTensor = tmpBuffer.Get<T1>();
        if constexpr (!IsSameType<T1, float>::value) {
            DataCopy(tmpTensor, vecOut1Buffer1, s1ExtendSubGraph * s2ExtendAlign);
            PipeBarrier<PIPE_V>();
            ND2NZ(vecOut1Buffer1, tmpTensor, s1ExtendSubGraph, s2ExtendAlign);

            event_t mte3WaitV = isPing ?
                                     eventIdList.structMte3WaitVPing : eventIdList.structMte3WaitVPong;
            SetFlag<HardEvent::V_MTE3>(mte3WaitV);
            WaitFlag<HardEvent::V_MTE3>(mte3WaitV);
            DataCopyPad(dropWorkSpaceGm[pingpongIdx * coreNum * cubeBaseMN + cBlockIdx * cubeBaseMN +
                                        curS1Idx * s1VecSize * C0_SIZE + curS2Idx * s1CvExtendAlign * s2VecSize],
                        vecOut1Buffer1,
                        {static_cast<uint16_t>(s2ExtendAlign / C0_SIZE),
                        static_cast<uint16_t>(s1ExtendSubGraph * C0_SIZE * sizeof(T1)), 1,
                        static_cast<uint16_t>((s1CvExtendAlign - s1ExtendSubGraph) * C0_SIZE * sizeof(T1))});
        } else {
            DataCopy(tmpTensor, vecDropBuffer, s1ExtendSubGraph * s2ExtendAlign);
            PipeBarrier<PIPE_V>();
            ND2NZ(vecDropBuffer, tmpTensor, s1ExtendSubGraph, s2ExtendAlign);

            event_t mte3WaitV = isPing ?
                                     eventIdList.structMte3WaitVPing : eventIdList.structMte3WaitVPong;
            SetFlag<HardEvent::V_MTE3>(mte3WaitV);
            WaitFlag<HardEvent::V_MTE3>(mte3WaitV);
            DataCopyPad(dropWorkSpaceGm[pingpongIdx * coreNum * cubeBaseMN + cBlockIdx * cubeBaseMN +
                                        curS1Idx * s1VecSize * C0_SIZE + curS2Idx * s1CvExtendAlign * s2VecSize],
                        vecDropBuffer,
                        {static_cast<uint16_t>(s2ExtendAlign / C0_SIZE),
                        static_cast<uint16_t>(s1ExtendSubGraph * C0_SIZE * sizeof(T1)), 1,
                        static_cast<uint16_t>((s1CvExtendAlign - s1ExtendSubGraph) * C0_SIZE * sizeof(T1))});
        }
    } else {
        event_t mte3WaitV = isPing ?
                                     eventIdList.structMte3WaitVPing : eventIdList.structMte3WaitVPong;
        SetFlag<HardEvent::V_MTE3>(mte3WaitV);
        WaitFlag<HardEvent::V_MTE3>(mte3WaitV);
        if constexpr (!IsSameType<T1, float>::value) {
                DataCopyPad(dropWorkSpaceGm[pingpongIdx * coreNum * cubeBaseMN + cBlockIdx * cubeBaseMN + LoopGIdx * singleBaseMN +
                                    curS1Idx * s1VecSize * s2CvExtendAlign + curS2Idx * s2VecSize],
                    vecOut1Buffer1,
                    {static_cast<uint16_t>(s1ExtendSubGraph), static_cast<uint16_t>(s2ExtendAlign * sizeof(T1)), 0,
                     static_cast<uint16_t>((s2CvExtendAlign - s2ExtendAlign) * sizeof(T1))});
        } else {
                DataCopyPad(dropWorkSpaceGm[pingpongIdx * coreNum * cubeBaseMN + cBlockIdx * cubeBaseMN + LoopGIdx * singleBaseMN +
                                    curS1Idx * s1VecSize * s2CvExtendAlign + curS2Idx * s2VecSize],
                    vecDropBuffer,
                    {static_cast<uint16_t>(s1ExtendSubGraph), static_cast<uint16_t>(s2ExtendAlign * sizeof(T1)), 0,
                     static_cast<uint16_t>((s2CvExtendAlign - s2ExtendAlign) * sizeof(T1))});
        }
    }
    SetFlag<HardEvent::MTE3_MTE2>(curEventIdMte2WaitMte3);
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT, const uint32_t TND_S1_PP>
__aicore__ inline void
FusedFloydAttentionGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
                                    MM2_OUT_FORMAT, TND_S1_PP>::SubGrapB(int64_t curIdx, int64_t curS1Idx, int64_t curS2Idx, int64_t LoopGIdx,
                                    EvenvIdList &eventIdList)
{
    bool isPing = ((curIdx % 2) == 0);
    event_t curEventId = isPing ?
                                     eventIdList.structMte2WaitMte3Ping : eventIdList.structMte2WaitMte3Pong;
    uint32_t ubBufferOffset = isPing ? 0 : DbBegin;
    s2Extend = (curS2Idx == s2VecLoop - 1) ? (s2CvExtend - (s2VecLoop - 1) * s2VecSize) : s2VecSize;
    s2ExtendAlign = (s2Extend + 15) / 16 * 16;

    WaitFlag<HardEvent::MTE3_MTE2>(curEventId);

    LocalTensor<uint8_t> vecInDropBuffer =
        ubBuffer.GetWithOffset<uint8_t>(8 * 1024 / sizeof(uint8_t), ubBufferOffset + U8Begin);

    LocalTensor<T2> vecClc1Buffer = ubBuffer.GetWithOffset<T2>(32 * 1024 / sizeof(T2), ubBufferOffset + T1Begin);
    if constexpr (MM_OUT_FORMAT == CubeFormat::ND) {
        if (s2VecLoop == 1) {
            DataCopy(vecClc1Buffer, mm1WorkspaceGm[curS1Idx * s1VecSize * s2ExtendAlign + LoopGIdx * singleBaseMN ],
                     s1ExtendSubGraph * s2ExtendAlign);
        } else {
            DataCopyPad(vecClc1Buffer, mm1WorkspaceGm[curS1Idx * s1VecSize * s2CvExtendAlign + curS2Idx * s2VecSize + LoopGIdx * singleBaseMN],
                        {static_cast<uint16_t>(s1ExtendSubGraph), static_cast<uint16_t>(s2ExtendAlign * sizeof(float)),
                         static_cast<uint16_t>((s2CvExtendAlign - s2ExtendAlign) * sizeof(float)), 0},
                        {false, 0, 0, 0});
        }
        event_t vWaitMte2 = isPing ?
                                     eventIdList.structVWaitMte2Ping : eventIdList.structVWaitMte2Pong;
        SetFlag<HardEvent::MTE2_V>(vWaitMte2);
        WaitFlag<HardEvent::MTE2_V>(vWaitMte2);
    } else {
        int64_t mmAddr = curS1Idx * s1VecSize * C0_SIZE + curS2Idx * s1CvExtend * s2VecSizeAlign;
        NZCopyIn(mmAddr, mm1WorkspaceGm, vecClc1Buffer, s1VecSize, s2ExtendAlign);
        event_t vWaitMte2 = isPing ?
                                     eventIdList.structVWaitMte2Ping : eventIdList.structVWaitMte2Pong;
        SetFlag<HardEvent::MTE2_V>(vWaitMte2);
        WaitFlag<HardEvent::MTE2_V>(vWaitMte2);
        auto tmpTensor = tmpBuffer.Get<T2>();
        DataCopy(tmpTensor, vecClc1Buffer, s1VecSize * s2ExtendAlign + s2ExtendAlign / C0_SIZE * VEC_REPEAT);
        PipeBarrier<PIPE_V>();
        NZ2ND(vecClc1Buffer, tmpTensor, s1VecSize, s2ExtendAlign);
    }

    ///////////////////////////////////////////////////////////////
    // ss
    ///////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////
    // sub to improve
    ///////////////////////////////////////////////////////////////
    uint32_t sub_block_cout = (s2ExtendAlign + cal_repeat_num - 1) / cal_repeat_num;
    uint32_t sfmgStartIndex = s1CvRatio > 1 ? 0 : curS1Idx * s1VecSize * 8;

    LocalTensor<float> sfmgClc3;
    sfmgClc3 = vecClc3.Get<float>();

    PipeBarrier<PIPE_V>();
    for (uint32_t subIdx = 0; subIdx < sub_block_cout; subIdx++) {
        uint32_t subMaskCout =
            (subIdx == sub_block_cout - 1) ? (s2ExtendAlign - subIdx * cal_repeat_num) : cal_repeat_num;
        Sub(vecClc1Buffer[subIdx * cal_repeat_num], vecClc1Buffer[subIdx * cal_repeat_num], sfmgClc3[sfmgStartIndex],
            subMaskCout, s1ExtendSubGraph,
            {static_cast<uint8_t>(1), static_cast<uint8_t>(1), 0, static_cast<uint8_t>(s2ExtendAlign / 8),
             static_cast<uint8_t>(s2ExtendAlign / 8), 1});
    }

    ///////////////////////////////////////////////////////////////
    // mul
    ///////////////////////////////////////////////////////////////
    PipeBarrier<PIPE_V>();
    LocalTensor<float> vecClc2Buffer =
        ubBuffer.GetWithOffset<float>(32 * 1024 / sizeof(float), ubBufferOffset + T2Begin);
    Mul(vecClc1Buffer, vecClc1Buffer, vecClc2Buffer, s1ExtendSubGraph * s2ExtendAlign);
    LocalTensor<T1> vecOutBuffer;
    if constexpr (!IsSameType<T1, float>::value) {
        vecOutBuffer = ubBuffer.GetWithOffset<T1>(17 * 1024 / sizeof(T1), ubBufferOffset + T1Begin);
        PipeBarrier<PIPE_V>();
        Cast(vecOutBuffer, vecClc1Buffer, RoundMode::CAST_ROUND, s1ExtendSubGraph * s2ExtendAlign);
    }
    if constexpr (MM_OUT_FORMAT == CubeFormat::NZ) {
        PipeBarrier<PIPE_V>();
        auto tmpTensor1 = tmpBuffer.Get<T1>();
        if constexpr (IsSameType<T1, float>::value) {
            DataCopy(tmpTensor1, vecClc1Buffer, s1ExtendSubGraph * s2ExtendAlign);
            PipeBarrier<PIPE_V>();
            ND2NZ(vecClc1Buffer, tmpTensor1, s1ExtendSubGraph, s2ExtendAlign);
        } else {
            DataCopy(tmpTensor1, vecOutBuffer, s1ExtendSubGraph * s2ExtendAlign);
            PipeBarrier<PIPE_V>();
            ND2NZ(vecOutBuffer, tmpTensor1, s1ExtendSubGraph, s2ExtendAlign);
        }

        event_t mte3WaitV = isPing ?
                                     eventIdList.structMte3WaitVPing : eventIdList.structMte3WaitVPong;
        SetFlag<HardEvent::V_MTE3>(mte3WaitV);
        WaitFlag<HardEvent::V_MTE3>(mte3WaitV);

        if constexpr(IsSameType<T1, float>::value){
            DataCopyPad(mulWorkSpaceGm[pingpongIdx * coreNum * cubeBaseMN + cBlockIdx * cubeBaseMN +
                                   curS1Idx * s1VecSize * C0_SIZE + curS2Idx * s1CvExtendAlign * s2VecSize],
            vecClc1Buffer,
            {static_cast<uint16_t>(s2ExtendAlign / C0_SIZE),
                static_cast<uint16_t>(s1ExtendSubGraph * C0_SIZE * sizeof(T1)), 1,
            static_cast<uint16_t>((s1CvExtendAlign - s1ExtendSubGraph) * C0_SIZE * sizeof(T1))});
        } else {
            DataCopyPad(mulWorkSpaceGm[pingpongIdx * coreNum * cubeBaseMN + cBlockIdx * cubeBaseMN +
                                   curS1Idx * s1VecSize * C0_SIZE + curS2Idx * s1CvExtendAlign * s2VecSize],
            vecOutBuffer,
            {static_cast<uint16_t>(s2ExtendAlign / C0_SIZE),
                static_cast<uint16_t>(s1ExtendSubGraph * C0_SIZE * sizeof(T1)), 1,
            static_cast<uint16_t>((s1CvExtendAlign - s1ExtendSubGraph) * C0_SIZE * sizeof(T1))});
        }
    } else {
        event_t mte3WaitV = isPing ?
                                     eventIdList.structMte3WaitVPing : eventIdList.structMte3WaitVPong;
        SetFlag<HardEvent::V_MTE3>(mte3WaitV);
        WaitFlag<HardEvent::V_MTE3>(mte3WaitV);

        if constexpr(IsSameType<T1, float>::value) {
               DataCopyPad(mulWorkSpaceGm[pingpongIdx * coreNum * cubeBaseMN + cBlockIdx * cubeBaseMN + LoopGIdx * singleBaseMN + 
                                   curS1Idx * s1VecSize * s2CvExtendAlign + curS2Idx * s2VecSize],
                    vecClc1Buffer,
                    {static_cast<uint16_t>(s1ExtendSubGraph), static_cast<uint16_t>(s2ExtendAlign * sizeof(T1)), 0,
                     static_cast<uint16_t>((s2CvExtendAlign - s2ExtendAlign) * sizeof(T1))});
        } else {
                  DataCopyPad(mulWorkSpaceGm[pingpongIdx * coreNum * cubeBaseMN + cBlockIdx * cubeBaseMN + LoopGIdx * singleBaseMN + 
                                   curS1Idx * s1VecSize * s2CvExtendAlign + curS2Idx * s2VecSize],
                    vecOutBuffer,
                    {static_cast<uint16_t>(s1ExtendSubGraph), static_cast<uint16_t>(s2ExtendAlign * sizeof(T1)), 0,
                     static_cast<uint16_t>((s2CvExtendAlign - s2ExtendAlign) * sizeof(T1))});
        }
    }
    if ((s1VecLoop * s2VecLoop > 2) && (curIdx < (s1VecLoop * s2VecLoop - 2))) {
        SetFlag<HardEvent::MTE3_MTE2>(curEventId);
    }
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT, const uint32_t TND_S1_PP>
__aicore__ inline void FusedFloydAttentionGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT,
                                                           INPUT_LAYOUT, MM2_OUT_FORMAT, TND_S1_PP>::Compute(int64_t preIndex,
                                                                                                  int64_t nextIndex)
{
    pingpongIdx = 1 - pingpongIdx;
    if (isStart == 1) {
        InitIndex(preIndex);
    }
    int64_t mm1aTensorOffsetCv = 0;
    int64_t mm2aTensorOffsetCv = 0;
    int64_t bTensorOffsetCv = 0;
    int64_t bmm1k1aTensorOffsetCv = 0;
    int64_t bmm1k1bTensorOffsetCv = 0;
    s2CvExtend = preS2CvEnd - preS2CvBegin;
    s2CvExtendAlign = (s2CvExtend + 15) / 16 * 16;
    s1CvExtendAlign = (s1CvExtend + 15) / 16 * 16;
    int64_t dqOffset = 0;
    int64_t dkvOffset = 0;
    int64_t s1StrideSize = 0;
    int64_t s2StrideSize = 0;
    int64_t dAlign = (d + 15) / 16 * 16;
    int64_t actualS1Len = 0;
    int64_t actualS2Len;

    int64_t gPreExtend = gExtend;

    singleBaseMN = s1CvInner * s2CvExtendAlign;

    // BNGSD
    mm1aTensorOffsetCv =(((static_cast<int64_t>(bDimIdx) * n2 + n2DimIdx) * g + gDimIdx * gInner) * s1 + s1oDimIdx * s1CvInner) * d;
    mm2aTensorOffsetCv = mm1aTensorOffsetCv;
    bTensorOffsetCv = (((static_cast<int64_t>(bDimIdx) * n2 + n2DimIdx) * g + gDimIdx * gInner) * s2 + preS2CvBegin) * d;
    bmm1k1aTensorOffsetCv = mm1aTensorOffsetCv;
    bmm1k1bTensorOffsetCv = (((static_cast<int64_t>(bDimIdx) * n2 + n2DimIdx) * s2 + preS2CvBegin) * s1 + s1oDimIdx * s1CvInner) * d;
    s1StrideSize = d;
    s2StrideSize = d;

    
    if constexpr (MM2_OUT_FORMAT == CubeFormat::ND) {
        dqOffset = mm1aTensorOffsetCv;
        dkvOffset = bTensorOffsetCv;
    }

    if (isStart == 1) {
        if constexpr (MM_OUT_FORMAT == CubeFormat::NZ) {
            mm1.SetOrgShape(s1CvExtend, s2, s1StrideSize, s2StrideSize, s2CvExtendAlign);
        } else {
            mm1.SetOrgShape(s1, s2, s1StrideSize, s2StrideSize, s2CvExtendAlign);
        }

        for (uint32_t i = 0; i < gPreExtend; i++) {
            mm1.SetTail(s1CvExtend, s2CvExtend, d); // M N K
            mm1.SetTensorA(dxGm[mm1aTensorOffsetCv + i * s1 * d]);
            mm1.SetTensorB(valueGm[bTensorOffsetCv + i * s2 * d], true);
            mm1.template IterateAll<false>(mm1WorkspaceGm [i * singleBaseMN], false, false, true);
            mm1.WaitIterateAll();
            mm1.End();

            mm1.SetTail(s1CvExtend, s2CvExtend, d); // M N K
            mm1.SetTensorA(queryGm[mm2aTensorOffsetCv + i * s1 * d]);
            mm1.SetTensorB(keyGm[bTensorOffsetCv + i * s2 * d], true);
            mm1.template IterateAll<false>(mm2WorkspaceGm[i * singleBaseMN], false, false, true);
            mm1.WaitIterateAll();
            mm1.End();
        }

        ComputeBMMK1(bmm1k1aTensorOffsetCv, bmm1k1bTensorOffsetCv ,s1StrideSize, s2StrideSize, gPreExtend, s1CvExtend, s2CvExtend, s2CvExtendAlign);
        isStart = 0;
    }

    s2VecSize = s2CvExtend > VEC_S2_LEN ? VEC_S2_LEN : s2CvExtend;
    s2VecSize = s2VecSize == 0 ? cal_block_num : s2VecSize;
    s2VecLoop = (s2CvExtend + s2VecSize - 1) / s2VecSize;

    if constexpr (IS_ATTEN_MASK == ENABLE) {
        // attenmask last dim 32B align
        s2VecSizeAlign = (s2VecSize + 31) / 32 * 32;
    } else {
        s2VecSizeAlign = (s2VecSize + 15) / 16 * 16;
    }
    
    s1VecSize = baseMN / s2VecSizeAlign;
    s1VecSize = s1VecSize < s1CvExtend ? s1VecSize : s1CvExtend;
    s1VecSize = s1VecSize > 128 ? 128 : s1VecSize;
    s1VecLoop = (s1CvExtend + s1VecSize - 1) / s1VecSize;

    ///////////////////////////////////////////////////////////////
    // SoftmaxGradFront
    ///////////////////////////////////////////////////////////////
    for (uint32_t LoopG = 0; LoopG < gPreExtend; LoopG++) {
        PipeBarrier<PIPE_ALL>();
        LocalTensor<float> sfmgClc3 = vecClc3.Get<float>();
        if (s1CvRatio <= 1) {
            CalcSoftMaxGrad(sfmgClc3, mm1aTensorOffsetCv + LoopG * s1 * d, s1CvExtend);
        }

        uint32_t curIdxPing = 0;
        uint32_t curIdxPong = 0;
        uint32_t curS2IdxPing = 0;
        uint32_t curS2IdxPong = 0;
        for (uint32_t curS1Idx = 0; curS1Idx < s1VecLoop; curS1Idx++) {
            s1ExtendSubGraph = (curS1Idx == s1VecLoop - 1) ? (s1CvExtend - (s1VecLoop - 1) * s1VecSize) : s1VecSize;
            if (s1CvRatio > 1) {
                PipeBarrier<PIPE_ALL>();
                int64_t sfmgOffset = (((static_cast<int64_t>(bDimIdx) * n2 + n2DimIdx) * g + gDimIdx * gInner + LoopG) * s1 +
                            s1oDimIdx * s1CvInner + curS1Idx * s1VecSize) * d;
                LocalTensor<float> sfmgClc3 = vecClc3.Get<float>();
                CalcSoftMaxGrad(sfmgClc3, sfmgOffset, s1ExtendSubGraph);
            }

            EvenvIdList eventIdList;
            LocalAllocEventID(eventIdList);

            for (uint32_t curS2Idx = 0; curS2Idx < s2VecLoop; curS2Idx = curS2Idx + 2) {
                curS2IdxPing = curS2Idx;
                curS2IdxPong = curS2Idx + 1;
                curIdxPing = curS1Idx * s2VecLoop + curS2IdxPing;
                curIdxPong = curS1Idx * s2VecLoop + curS2IdxPong;
                SubGrapA(curIdxPing, curS1Idx, curS2IdxPing, LoopG, eventIdList);
                if (curS2IdxPong < s2VecLoop) {
                    SubGrapA(curIdxPong, curS1Idx, curS2IdxPong, LoopG, eventIdList);
                }

                SubGrapB(curIdxPing, curS1Idx, curS2IdxPing, LoopG, eventIdList);
                if (curS2IdxPong < s2VecLoop) {
                    SubGrapB(curIdxPong, curS1Idx, curS2IdxPong, LoopG, eventIdList);
                }
            }
            LocalReleaseEventID(eventIdList);
        }
        PipeBarrier<PIPE_ALL>();
    }

    uint32_t preS1Extend = s1CvExtend;
    if (nextIndex != 0) {
        InitIndex(nextIndex);
        int64_t nextS2CvExtend = nextS2CvEnd - nextS2CvBegin;
        int64_t nextS2CvExtendAlign = (nextS2CvExtend + 15) / 16 * 16;
        int64_t mm1aTensorOffsetCv1 = 0;
        int64_t mm2aTensorOffsetCv1 = 0;
        int64_t bTensorOffsetCv1 = 0;

        int64_t bmm1k1aTensorOffsetCv1 = 0;
        int64_t bmm1k1bTensorOffsetCv1 = 0;

        int64_t nextSingleBaseMN = s1CvInner * nextS2CvExtendAlign;

        mm1aTensorOffsetCv1 =
            (((static_cast<int64_t>(bDimIdx) * n2 + n2DimIdx) * g + gDimIdx * gInner) * s1 + s1oDimIdx * s1CvInner) * d;
        mm2aTensorOffsetCv1 = mm1aTensorOffsetCv1;
        bTensorOffsetCv1 = (((static_cast<int64_t>(bDimIdx) * n2 + n2DimIdx) * g + gDimIdx * gInner)* s2 + nextS2CvBegin) * d;

        bmm1k1aTensorOffsetCv1 = mm1aTensorOffsetCv1;
        bmm1k1bTensorOffsetCv1 = (((static_cast<int64_t>(bDimIdx) * n2 + n2DimIdx) * s2 + nextS2CvBegin) * s1 + s1oDimIdx * s1CvInner) * d;
        
        if constexpr (MM_OUT_FORMAT == CubeFormat::NZ) {
            mm1.SetOrgShape(s1CvExtend, s2, s1StrideSize, s2StrideSize, nextS2CvExtendAlign);
        } else {
            mm1.SetOrgShape(s1, s2, s1StrideSize, s2StrideSize, nextS2CvExtendAlign);
        }

        for (uint32_t i = 0; i < gExtend; i++) {
            mm1.SetTail(s1CvExtend, nextS2CvExtend, d);
            mm1.SetTensorA(dxGm[mm1aTensorOffsetCv1 + i * s1 * d]);
            mm1.SetTensorB(valueGm[bTensorOffsetCv1 + i * s2 * d], true);
            mm1.template IterateAll<false>(mm1WorkspaceGm[i * nextSingleBaseMN], false, false, true);
            mm1.WaitIterateAll();
            mm1.End();

            mm1.SetTail(s1CvExtend, nextS2CvExtend, d);
            mm1.SetTensorA(queryGm[mm2aTensorOffsetCv1 + i * s1 * d]);
            mm1.SetTensorB(keyGm[bTensorOffsetCv1 + i * s2 * d], true);
            mm1.template IterateAll<false>(mm2WorkspaceGm[i * nextSingleBaseMN], false, false, true);
            mm1.WaitIterateAll();
            mm1.End();
        }
        ComputeBMMK1(bmm1k1aTensorOffsetCv1, bmm1k1bTensorOffsetCv1, s1StrideSize, s2StrideSize, gExtend, s1CvExtend, nextS2CvExtend, nextS2CvExtendAlign);
    }

    int64_t s1_size = s1;
    int64_t s1TndSize = actualS1Len;
    if constexpr (MM_OUT_FORMAT == CubeFormat::NZ) {
        s1_size = s1CvExtendAlign;
        s1TndSize = s1CvExtendAlign;
    }

    ///////////////////////////////////////////////////////////////
    // Matmal4 dq
    ///////////////////////////////////////////////////////////////
    // left [B, N2, G, S1, s2] right [B, N2, 1, S2, D] output [B, N2, G, S1, D]
    ComputeBMM2K1(pingpongIdx * coreNum * cubeBaseMN + cBlockIdx * cubeBaseMN, bmm1k1bTensorOffsetCv, dqOffset, gPreExtend, preS1Extend, s2CvExtend, s2CvExtendAlign);
    ComputeBMM3K1V1(pingpongIdx * coreNum * cubeBaseMN + cBlockIdx * cubeBaseMN, mm2aTensorOffsetCv, bmm1k1bTensorOffsetCv, gPreExtend, preS1Extend, s2CvExtend, s2CvExtendAlign);

    for (uint32_t i = 0; i < gPreExtend; i++) {
        if (MM2_OUT_FORMAT == CubeFormat::NZ) {
            mm3.SetOrgShape(s1_size, d, s2CvExtendAlign);
            mm3.SetSelfDefineData(s1);
        } else {
            mm3.SetOrgShape(s1_size, d, s2CvExtendAlign);
        }
        
        mm3.SetTail(preS1Extend, -1, s2CvExtend); // M N K
        mm3.SetTensorA(mulWorkSpaceGm[pingpongIdx * coreNum * cubeBaseMN + cBlockIdx * cubeBaseMN + i * singleBaseMN]);
        mm3.SetTensorB(keyGm[bTensorOffsetCv + i * s2 * d]);
        mm3.template IterateAll<false>(dqWorkSpaceGm[dqOffset + i * s1 * d], true);
        mm3.End();

        ///////////////////////////////////////////////////////////////
        // Matmal4 dk
        ///////////////////////////////////////////////////////////////
        // left [B, N2, G, S1, S2] right [B, N2, 1, S1, D] output [B, N2, G, S2, D]
        if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ) {
            mm3.SetOrgShape(s2CvExtendAlign, d, s1_size);
            mm3.SetSelfDefineData(s2);
        } else {
            mm3.SetOrgShape(s2CvExtendAlign, d, s1_size);
        }

        mm3.SetTail(s2CvExtend, -1, preS1Extend);
        mm3.SetTensorA(mulWorkSpaceGm[pingpongIdx * coreNum * cubeBaseMN + cBlockIdx * cubeBaseMN + i * singleBaseMN], true);
        mm3.SetTensorB(queryGm[mm2aTensorOffsetCv + i * s1 * d]);
        mm3.template IterateAll<false>(dkWorkSpaceGm[dkvOffset + i * s2 * d], true);
        mm3.End();

        ///////////////////////////////////////////////////////////////
        // Matmal5 dv
        ///////////////////////////////////////////////////////////////
        // left [B, N2, G, S1, S2] right [B, N2, G, S1, D] output [B, N2, 1, S2, D]
        mm3.SetTail(s2CvExtend, -1, preS1Extend);
        mm3.SetTensorA(dropWorkSpaceGm[pingpongIdx * coreNum * cubeBaseMN + cBlockIdx * cubeBaseMN + i * singleBaseMN], true);
        mm3.SetTensorB(dxGm[mm1aTensorOffsetCv + i * s1 * d]);

        if constexpr (IsSameType<T1, float>::value) {
            if (nextIndex == 0) {
                mm3.template IterateAll<true>(dvGm[dkvOffset + i * s2 * d], true);
            } else {
                mm3.template IterateAll<false>(dvGm[dkvOffset + i * s2 * d], true);
            }
        } else {
            if (nextIndex == 0) {
                mm3.template IterateAll<true>(dvWorkSpaceGm[dkvOffset + i * s2 * d], true);
            } else {
                mm3.template IterateAll<false>(dvWorkSpaceGm[dkvOffset + i * s2 * d], true);
            }
        }
        mm3.End();
    }
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT, const uint32_t TND_S1_PP>
__aicore__ inline void FusedFloydAttentionGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT,
                                                           INPUT_LAYOUT, MM2_OUT_FORMAT, TND_S1_PP>::LocalAllocEventID(EvenvIdList &eventIdList)
{
    eventIdList.structVWaitMte2Ping = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
    eventIdList.structVWaitMte2Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
    eventIdList.structMte2WaitMte3Ping = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
    eventIdList.structMte2WaitMte3Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
    eventIdList.structMte3WaitVPing = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
    eventIdList.structMte3WaitVPong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
    eventIdList.structMte2WaitVPing = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    eventIdList.structMte2WaitVPong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT, const uint32_t TND_S1_PP>
__aicore__ inline void FusedFloydAttentionGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT,
                                                           INPUT_LAYOUT, MM2_OUT_FORMAT, TND_S1_PP>::LocalReleaseEventID(EvenvIdList &eventIdList)
{
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(eventIdList.structVWaitMte2Ping);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(eventIdList.structVWaitMte2Pong);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(eventIdList.structMte2WaitMte3Ping);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(eventIdList.structMte2WaitMte3Pong);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(eventIdList.structMte3WaitVPing);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(eventIdList.structMte3WaitVPong);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(eventIdList.structMte2WaitVPing);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(eventIdList.structMte2WaitVPong);
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT, const uint32_t TND_S1_PP>
__aicore__ inline void FusedFloydAttentionGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT,
                                                           INPUT_LAYOUT, MM2_OUT_FORMAT, TND_S1_PP>::SyncALLCores()
{
    SyncAll();
}

#endif // _fused_floyd_attention_grad_S1S2_BN2GS1S2_H_
