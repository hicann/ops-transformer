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
 * \file nsa_selected_attention_infer.h
 * \brief
 */
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "nsa_public_define.h"

using namespace matmul;
using AscendC::CacheMode;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;
using AscendC::PipeBarrier;

#define PRE_LOAD_NUM 4

struct ExtraInfo {
    uint32_t loop = 0;
    uint32_t bIdx = 0;
    uint32_t n2Idx = 0;
    uint32_t s1Idx = 0;
    uint32_t s2Idx = 0;
    uint32_t bn2IdxInCurCore = 0;
    uint32_t bn2IdxInPreCore = 0;
    uint32_t curSInnerLoopTimes = 0;
    uint64_t tensorAOffset = 0;
    uint64_t tensorBOffset = 0;
    uint64_t tensorARopeOffset = 0;
    uint64_t tensorBRopeOffset = 0;
    uint64_t attenOutOffset = 0;

    uint32_t actualSingleProcessSInnerSize = 0;
    uint32_t actualSingleProcessSInnerSizeAlign = 0;
    bool isFirstSInnerLoop = false;
    bool needSetOrgShape = false;
    uint32_t s2BatchOffset = 0;
    ExtraInfo() {
        loop = 0;
        bIdx = 0;
        n2Idx = 0;
        s1Idx = 0;
        s2Idx = 0;
        bn2IdxInCurCore = 0;
        bn2IdxInPreCore = 0;
        curSInnerLoopTimes = 0;
        tensorAOffset = 0;
        tensorBOffset = 0;
        attenOutOffset = 0;

        actualSingleProcessSInnerSize = 0;
        actualSingleProcessSInnerSizeAlign = 0;
        isFirstSInnerLoop = false;
        needSetOrgShape = false;
        s2BatchOffset = 0;
    }
};

template <typename NSAT> class NsaSelectAttentionInfer {
public:
    __aicore__ inline NsaSelectAttentionInfer(){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                __gm__ uint8_t *topkIndices, __gm__ uint8_t *attenMask, __gm__ uint8_t *blockTable,
                                __gm__ uint8_t *actualQSeqLengths, __gm__ uint8_t *actualSeqLengths,
                                __gm__ uint8_t *attentionOut,
                                __gm__ uint8_t *workspace,
                                const NsaSelectAttentionInferTilingData *__restrict tiling, __gm__ uint8_t *gmTiling,
                                TPipe *tPipe);
    
    __aicore__ inline void Process();
    __aicore__ inline bool IsFinish(uint32_t loop);

    // 中间计算数据类型为float，高精度模式
    using T = float;

    using Q_T = typename NSAT::queryType;
    using KV_T = typename NSAT::kvType;
    using OUT_T = typename NSAT::outputType;
    using ORIGIN_T = typename NSAT::orginalType;
    static constexpr LAYOUT LAYOUT_T = NSAT::layout; //change back by wy

    using MM_OUT_T = float;

protected:
    const NsaSelectAttentionInferTilingData *__restrict tilingData = nullptr;
    TPipe *pipe = nullptr;

    GlobalTensor<Q_T> queryGm;
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<KV_T> valueGm;
    GlobalTensor<OUT_T> attentionOutGm;
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<int32_t> topKGm;
    GlobalTensor<uint64_t> actualSeqLengthsGm;
    GlobalTensor<uint64_t> actualQSeqLengthsGm;
    // workspace
    GlobalTensor<MM_OUT_T> mm1ResGm;
    GlobalTensor<KV_T> vec1ResGm;
    GlobalTensor<MM_OUT_T> mm2ResGm;
    GlobalTensor<T> vec2ResGm;
    GlobalTensor<T> accumOutGm; // no
    GlobalTensor<T> lseSumFdGm; // no
    GlobalTensor<T> lseMaxFdGm; // no

    // L1
    // KP复用kpL1Tensor
    TBuf<TPosition::A1> kpBufL1;
    LocalTensor<KV_T> kpL1Tensor;

    TBuf<TPosition::A1> valueBufL1;
    LocalTensor<KV_T> vL1Tensor;

    TBuf<TPosition::A1> queryBufL1;
    LocalTensor<Q_T> qL1Tensor;

    // L1 buffer size
    static constexpr uint32_t L1_KP_SIZE = (128 * (128 + 64) * sizeof(KV_T));
    static constexpr uint32_t L1_V_SIZE = (2 * 128 * 128 * sizeof(KV_T));
    static constexpr uint32_t L1_Q_SIZE = (128 * (128 + 64) * sizeof(Q_T));

    // L0 buffer size
    static constexpr uint32_t L0A_PP_SIZE = (32 * 1024); // 128*128*2
    static constexpr uint32_t L0B_PP_SIZE = (32 * 1024);
    static constexpr uint32_t L0C_PP_SIZE = (64 * 1024); // 128 * 128 *4 

    // mte2 <> mte1 EventID
    static constexpr uint32_t KP_EVENT0 = EVENT_ID4;
    static constexpr uint32_t KP_EVENT1 = EVENT_ID5;
    static constexpr uint32_t V_EVENT0 = EVENT_ID2;
    static constexpr uint32_t Q_EVENT1 = EVENT_ID3;

    // m <> mte1 EventID
    static constexpr uint32_t L0A_EVENT0 = EVENT_ID3;
    static constexpr uint32_t L0A_EVENT1 = EVENT_ID4;
    static constexpr uint32_t L0B_EVENT0 = EVENT_ID5;
    static constexpr uint32_t L0B_EVENT1 = EVENT_ID6;

    // fix <> m EventID
    static constexpr uint32_t L0C_EVENT0 = EVENT_ID3;
    static constexpr uint32_t L0C_EVENT1 = EVENT_ID4;

    uint32_t kpL1BufIter = 0; // double buffer
    uint32_t vL1BufIter = 0;
    uint32_t aL0BufIter = 0;
    uint32_t bL0BufIter = 0;
    uint32_t cL0BufIter = 0;

    // queue
    TQue<QuePosition::VECIN, 1> inputQue1;   // 32K, inque
    TQue<QuePosition::VECIN, 1> inputQue2;   // 16K, inque
    TQue<QuePosition::VECOUT, 1> outputQue1; // 32K, outque
    TQue<QuePosition::A1, 1> inputQueL1A;
    TQue<QuePosition::B1, 1> inputQueL1B;
    // L0_A
    TBuf<TPosition::A2> tmpBufL0A;
    LocalTensor<KV_T> aL0TensorPingPong;
    // L0_B
    TBuf<TPosition::B2> tmpBufL0B;
    LocalTensor<KV_T> bL0TensorPingPong;
    // L0_C
    TBuf<TPosition::CO1> tmpBufL0C;
    LocalTensor<MM_OUT_T> cL0TensorPingPong;

    // 临时tbuf
    TBuf<> tmpBuff1; // 32K
    TBuf<> tmpBuff2; // 32K

    // 常驻tbuf
    TBuf<> softmaxMaxBuff[PRE_LOAD_NUM]; // PRE_LOAD_NUM * 2K
    TBuf<> softmaxExpBuff[PRE_LOAD_NUM]; // PRE_LOAD_NUM * 2K
    TBuf<> softmaxSumBuff[PRE_LOAD_NUM]; // PRE_LOAD_NUM * 2K
    TBuf<> softmaxMaxDefaultBuff;     // 2K
    TBuf<> softmaxSumDefaultBuff;     // 2K

    LocalTensor<T> softmaxMaxUb[PRE_LOAD_NUM];
    LocalTensor<T> softmaxSumUb[PRE_LOAD_NUM];
    LocalTensor<T> softmaxExpUb[PRE_LOAD_NUM];
    LocalTensor<T> softmaxMaxDefaultUb;
    LocalTensor<T> softmaxSumDefaultUb;

    static constexpr uint64_t SYNC_MODE2 = 2;
    static constexpr uint64_t SYNC_C1_V1_FLAG = 7;
    static constexpr uint64_t SYNC_V1_C2_FLAG = 8;
    static constexpr uint64_t SYNC_C2_V2_FLAG = 9;

    static constexpr uint32_t HALF_BLOCK_NUM = 16;
    static constexpr uint32_t BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(T);
    static constexpr uint32_t BASE_BLOCK_MAX_ELEMENT_NUM = BUFFER_SIZE_BYTE_32K / sizeof(T);
    static constexpr T SOFTMAX_MIN_NUM = -2e38;
    uint32_t msdIterNum = 0U;
    uint64_t sUnitSize = 0;

    // for workspace pingpong
    const uint32_t dbWorkspaceRatio = PRE_LOAD_NUM;

    __gm__ uint8_t *keyPtr = nullptr;
    __gm__ uint8_t *valuePtr = nullptr;

    uint32_t tmpBlockIdx = 0U;
    uint32_t aiCoreIdx = 0U;
    __gm__ uint8_t *blocktablePtr = nullptr;

    // tilingdata
    uint64_t singleProcessSInnerSize = 0U;
    uint32_t sInnerLoopTimes = 0U;
    uint64_t singleProcessSInnerSizeTail = 0U;
    uint32_t formerCoreNum = 0U;
    uint32_t usedCoreNum = 0U;
    uint32_t bIdx = 0U;
    uint32_t n2Idx = 0U;
    // split Q
    uint32_t s1Idx = 0U;
    uint32_t mmResUbSize = 0U;
    uint32_t bmm2ResUbSize = 0U;

    uint64_t batchSize = 0ULL;
    uint64_t qHeadNum = 0ULL;
    uint64_t kvHeadNum = 0ULL;
    uint64_t gSize = 0ULL;
    uint64_t gSizeVector = 0ULL;
    uint64_t gSizeCube = 0ULL;
    uint64_t gSizeStart = 0ULL;
    uint64_t kvSeqSize = 0ULL;
    uint64_t headDim = 0ULL;
    uint64_t headDimV = 0ULL;
    uint64_t headDimAlign = 0ULL;
    uint64_t headDimVAlign = 0ULL;
    uint64_t selectedBlockSize = 0;
    uint64_t selectedBlockCount = 0;

    // pageAttention
    uint32_t kvCacheBlockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;

    // offset
    uint64_t tensorACoreOffset = 0ULL;
    uint64_t tensorAttenOutCoreOffset = 0ULL;
    uint64_t tensorBOffset = 0ULL;
    uint64_t valueOffset = 0ULL;
    uint64_t attenOutOffset = 0ULL;

    // splitKV
    uint32_t splitKVNum = 0U;
    uint32_t s2Idx = 0U;
    uint32_t s2IdxFD = 0U;
    uint64_t sInnerLoopSize = 0ULL;

    uint64_t curActualSeqLen = 0ULL;
    uint64_t curActualSeqLenOri = 0ULL;
    uint64_t curSingleProcessSInnerSizeAlign = 0ULL;
    uint64_t actualSingleProcessSInnerSize = 0ULL;
    uint64_t actualSingleProcessSInnerSizeAlign = 0ULL;
    uint32_t beforeBlockSplitBn2Nums = 0U;
    uint32_t bn2LoopTimes = 0U;

    uint32_t actualLenDims = 0U;
    uint32_t actualLenQDims = 0U;
    // 记录当前轮的bIdx nIdx s2Idx s1Idx actualLen
    uint32_t curBIdx = 0;
    uint32_t curN2Idx = 0;
    uint32_t curS2Idx = 0;
    uint32_t curS1Idx = 0;
    uint32_t curSInnerLoopTimes = 0;

    uint32_t bn2IdxInCurCore = 0;
    uint32_t bn2s2LoopTimes = 0;
    uint32_t numDouble = 2;
    uint32_t numEight = 8;
    ExtraInfo extraInfo[PRE_LOAD_NUM]{};

    // topk索引取数参数
    // mm1
    int64_t x = 0; // 上一轮topk索引已处理数据量
    int64_t currentTopDeal = -1; // 上一轮topk索引最大处理数据量
    int64_t topKIdOffset = 0; // 当前topk索引偏移
    int64_t idIntopKRecord = -1; // 当前topk索引偏移
    // mm2
    int64_t xMM2 = 0;
    int64_t currentTopDealMM2 = -1;
    int64_t topKIdOffsetMM2 = 0;
    int64_t idIntopKRecordMM2 = -1;
    bool isMtpFlag = false;
    uint32_t qSeqSize = 1;
    uint32_t curBatchQseqlen = 1;
    __aicore__ inline void CalcParams(uint32_t loop);
    __aicore__ inline void ComputeMm1(uint32_t loop);
    __aicore__ inline void ProcessVec1L(uint32_t loop);
    __aicore__ inline void ComputeMm2(uint32_t loop);
    __aicore__ inline void ProcessVec2L(uint32_t loop);
    __aicore__ inline void SetLoopTimes();
    __aicore__ inline void CopyInMm1AToL1(LocalTensor<KV_T>& aL1Tensor, ExtraInfo& info);
    __aicore__ inline void CopyInMm1BToL1ForPA(LocalTensor<KV_T>& bL1Tensor, uint64_t keyGmBaseOffset,
                                uint32_t copyTotalRowCnt, uint32_t copyStartRowCnt, uint32_t nActCopyRowCount);
    __aicore__ inline void LoadDataMm1A(LocalTensor<KV_T>& aL0Tensor, LocalTensor<KV_T>& aL1Tensor, uint32_t idx,
        uint32_t copyColCnt);
    __aicore__ inline void LoadDataMm1B(LocalTensor<KV_T>& bl0Tensor, LocalTensor<KV_T>& bl1Tensor, uint32_t k,
        uint32_t copyColCnt, uint32_t i, uint32_t copyRowCnt);
    __aicore__ inline void CopyInMm2AToL1(LocalTensor<KV_T>& aL1Tensor, ExtraInfo& info, uint32_t kCopyIdx,
        uint32_t kCopyRowCount, uint32_t kActCopyRowCountAlign, uint32_t kActCopyRowCount);
    __aicore__ inline void CopyInMm2BToL1ForPA(LocalTensor<KV_T>& bL1Tensor, uint64_t valueGmBaseOffset,
        uint32_t copyTotalRowCnt, uint32_t copyStartRowCnt, uint32_t kActCopyRowCount);
    __aicore__ inline void LoadDataMm2A(LocalTensor<KV_T> aL0Tensor, LocalTensor<KV_T> aL1Tensor, uint32_t kSize);

    bool curActSeqLenIsZero = false;

    template <typename T> __aicore__ inline T Align(T num, T rnd)
    {
        return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
    }
    __aicore__ inline void InitTilingData();
    __aicore__ inline void InitCalcParamsEach();
    __aicore__ inline void InitBuffers();
    __aicore__ inline void InitActualSeqLen(__gm__ uint8_t *actualSeqLengths);
    __aicore__ inline void InitActualQSeqLen(__gm__ uint8_t *actualQSeqLengths);
    __aicore__ inline void GetActualSeqLen();
    __aicore__ inline void UpdateInnerLoopCond();
    __aicore__ inline void DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx);

    __aicore__ inline void GetBN2id(const uint32_t bn2Idx);

    __aicore__ inline void DealBmm1ResBaseBlock(const uint32_t loop, uint32_t startRow, uint32_t dealRowCount,
                                                uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void ProcessVec1Inner(uint32_t loop);

    __aicore__ inline void DealBmm2ResBaseBlock(const uint32_t loop, uint32_t startRow, uint32_t dealRowCount,
                                                uint32_t columnCount, uint32_t actualColumnCount);
    
    __aicore__ inline void ProcessVec2Inner(uint32_t loop);

    __aicore__ inline void SoftmaxFlashV2Compute(uint32_t loop, LocalTensor<T> &mmResUb, LocalTensor<uint8_t> &softmaxTmpUb,
                                                uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
                                                uint32_t actualColumnCount);

    __aicore__ inline void Bmm2DataCopyOut(uint64_t attenOutOffset, LocalTensor<OUT_T> &attenOutUb, uint32_t startRow, uint32_t dealRowCount,
                                        uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void Bmm2CastAndCopyOut(ExtraInfo& info, LocalTensor<T> &bmm2ResUb, uint32_t startRow, uint32_t dealRowCount,
                                            uint32_t columnCount, uint32_t actualColumnCount);

    __aicore__ inline void InitAllZeroOutput(uint32_t bIdx, uint32_t n2Idx);

    __aicore__ inline void ElewiseCompute(uint32_t loop, LocalTensor<T> &mmResUb, uint32_t dealRowCount, uint32_t columnCount);
};

template <typename NSAT> __aicore__ inline void NsaSelectAttentionInfer<NSAT>::InitTilingData()
{
    singleProcessSInnerSize = tilingData->nsaSelectAttentionInferSingleCoreParams.singleProcessSInnerSize;
    sInnerLoopTimes = tilingData->nsaSelectAttentionInferSingleCoreParams.sInnerLoopTimes;
    singleProcessSInnerSizeTail = tilingData->nsaSelectAttentionInferSingleCoreParams.singleProcessSInnerSizeTail;
    usedCoreNum = tilingData->nsaSelectAttentionInferSingleCoreParams.usedCoreNum;
    formerCoreNum = tilingData->nsaSelectAttentionInferSingleCoreParams.formerCoreNum;
    splitKVNum = tilingData->splitKVParams.s2;
    sInnerLoopSize = tilingData->splitKVParams.sInnerLoopSize;

    mmResUbSize = tilingData->nsaSelectAttentionInferSingleCoreTensorSize.mmResUbSize;
    bmm2ResUbSize = tilingData->nsaSelectAttentionInferSingleCoreTensorSize.bmm2ResUbSize;

    batchSize = tilingData->baseParams.batchSize;
    kvHeadNum = tilingData->baseParams.kvHeadNum;
    qHeadNum = tilingData->baseParams.qHeadNum;
    gSize = tilingData->baseParams.nNumOfQInOneGroup;
    gSizeCube = gSize;
    gSizeVector = (gSize + 1) / numDouble; // vector0 : half gsize, align up
    gSizeStart = 0;
    if (tmpBlockIdx % numDouble == 1) { // vector1
    gSizeStart = gSizeVector;
    gSizeVector = gSize - gSizeVector;
    }
    kvSeqSize = tilingData->baseParams.seqSize;
    headDim = tilingData->baseParams.headSize;
    headDimV = tilingData->baseParams.headSizeV;
    selectedBlockSize = tilingData->baseParams.selectedBlockSize;
    selectedBlockCount = tilingData->baseParams.selectedBlockCount;
    msdIterNum = tilingData->baseParams.msdIterNum;
    headDimAlign = Align(headDim, BYTE_BLOCK);
    headDimVAlign = Align(headDimV, BYTE_BLOCK);
    maxBlockNumPerBatch = tilingData->baseParams.maxBlockNumPerBatch;
    kvCacheBlockSize = tilingData->baseParams.blockSize;
    isMtpFlag = tilingData->baseParams.isMtpFlag;
    qSeqSize = tilingData->baseParams.qSeqSize;
}

template <typename NSAT> __aicore__ inline void NsaSelectAttentionInfer<NSAT>::InitBuffers()
{
    if ASCEND_IS_AIV {
        // queue
        pipe->InitBuffer(inputQue1, 1, BUFFER_SIZE_BYTE_32K);
        pipe->InitBuffer(inputQue2, 1, BUFFER_SIZE_BYTE_16K);
        pipe->InitBuffer(outputQue1, 1, BUFFER_SIZE_BYTE_32K);

        // tmpBuff
        pipe->InitBuffer(tmpBuff1, BUFFER_SIZE_BYTE_32K);
        pipe->InitBuffer(tmpBuff2, BUFFER_SIZE_BYTE_32K);

        // 常驻buffer
        // 预留空间2K = 64 * 32，支持 gSize = 64
        // brcb 操作每次操作8*32字节输出，startRow接近64时，
        // 输出最多可能超出2k空间7*32字节， 这里预留256B防止越界
        for (int i = 0; i < PRE_LOAD_NUM; i++) {
            pipe->InitBuffer(softmaxMaxBuff[i], BUFFER_SIZE_BYTE_256B * numEight);
            pipe->InitBuffer(softmaxExpBuff[i], BUFFER_SIZE_BYTE_256B * numEight);
            pipe->InitBuffer(softmaxSumBuff[i], BUFFER_SIZE_BYTE_256B * numEight);
        }
        pipe->InitBuffer(softmaxMaxDefaultBuff, BUFFER_SIZE_BYTE_256B * numEight);
        pipe->InitBuffer(softmaxSumDefaultBuff, BUFFER_SIZE_BYTE_256B * numEight);
    } else {  
    // L1
    pipe->InitBuffer(queryBufL1, L1_Q_SIZE);
    qL1Tensor = queryBufL1.Get<Q_T>();

    pipe->InitBuffer(kpBufL1, L1_KP_SIZE * 2);
    kpL1Tensor = kpBufL1.Get<KV_T>();

    pipe->InitBuffer(valueBufL1, L1_V_SIZE * 2);
    vL1Tensor = valueBufL1.Get<KV_T>();

    // L0A
    pipe->InitBuffer(tmpBufL0A, L0A_PP_SIZE * 2);
    aL0TensorPingPong = tmpBufL0A.Get<KV_T>();

    // L0B
    pipe->InitBuffer(tmpBufL0B, L0B_PP_SIZE * 2);
    bL0TensorPingPong = tmpBufL0B.Get<KV_T>();

    // L0C
    pipe->InitBuffer(tmpBufL0C, L0C_PP_SIZE * 2);
    cL0TensorPingPong = tmpBufL0C.Get<MM_OUT_T>();
    }
}

template <typename NSAT>
__aicore__ inline void NsaSelectAttentionInfer<NSAT>::InitActualSeqLen(__gm__ uint8_t *actualSeqLengths)
{
    actualLenDims = tilingData->baseParams.actualLenDims;
    if (actualLenDims != 0) {
        actualSeqLengthsGm.SetGlobalBuffer((__gm__ uint64_t *)actualSeqLengths, actualLenDims);
    }
}

template <typename NSAT>
__aicore__ inline void NsaSelectAttentionInfer<NSAT>::InitActualQSeqLen(__gm__ uint8_t *actualQSeqLengths)
{
    actualLenQDims = tilingData->baseParams.actualLenQDims;
    if (actualLenQDims != 0) {
        actualQSeqLengthsGm.SetGlobalBuffer((__gm__ uint64_t *)actualQSeqLengths, actualLenQDims);
    }
}

template <typename NSAT>
__aicore__ inline void NsaSelectAttentionInfer<NSAT>::InitAllZeroOutput(uint32_t bIdx, uint32_t n2Idx)
{
    uint32_t copySize = gSize * headDimV;
    matmul::InitOutput<OUT_T>(attentionOutGm[(bIdx * kvHeadNum + n2Idx) * copySize], copySize, 0);
}

template <typename NSAT> __aicore__ inline void NsaSelectAttentionInfer<NSAT>::GetActualSeqLen()
{
    curActualSeqLen = actualSeqLengthsGm.GetValue(bIdx);
    uint32_t curActualSelNum = Ceil(curActualSeqLen, selectedBlockSize);
    uint32_t validTopkNum = 0;
    bool lastIndexFlag = false;
    bool lastIndexFlag2 = false;
    curBatchQseqlen = actualQSeqLengthsGm.GetValue(bIdx);
    uint32_t curTotalQSeqLenOffset = 0;
    for (uint64_t i = 0; i < bIdx; ++i) {
        curTotalQSeqLenOffset += actualQSeqLengthsGm.GetValue(i);
    }
    for (uint64_t topkIdx = 0; topkIdx < selectedBlockCount; topkIdx++) {
        uint64_t baseOffset = 0;
        if constexpr (LAYOUT_T == LAYOUT::TND) {
            baseOffset = curTotalQSeqLenOffset * kvHeadNum * selectedBlockCount + s1Idx * kvHeadNum * selectedBlockCount;
        } else {
            baseOffset = bIdx * qSeqSize * kvHeadNum * selectedBlockCount + s1Idx * kvHeadNum * selectedBlockCount;
        }
        int32_t topKIndices = topKGm.GetValue(baseOffset + topkIdx);
        if (topKIndices > -1) {
            validTopkNum++;
        }
        if (topKIndices == (curActualSelNum - 1)) {
            lastIndexFlag = true;
        }
        if (topKIndices == (curActualSelNum - 2)) {
            lastIndexFlag2 = true;
        }
    }
    uint32_t maskedLen = curBatchQseqlen - 1 - s1Idx;
    uint32_t tailLen = curActualSeqLen - (curActualSelNum - 1) * selectedBlockSize;
    uint32_t tailMasked = maskedLen > tailLen ? tailLen : maskedLen;
    uint32_t penultimateMasked = tailMasked < maskedLen ? maskedLen - tailLen : 0;
    curActualSeqLenOri = curActualSeqLen - maskedLen;
    if (lastIndexFlag && !lastIndexFlag2) {
        curActualSeqLen = (curActualSeqLen - (curActualSelNum - validTopkNum) * selectedBlockSize) - tailMasked;
    } else if (lastIndexFlag && lastIndexFlag2) {
        curActualSeqLen = (curActualSeqLen - (curActualSelNum - validTopkNum) * selectedBlockSize) - maskedLen;
    } else if (!lastIndexFlag && lastIndexFlag2) {
        curActualSeqLen = validTopkNum * selectedBlockSize - penultimateMasked;
    } else if(!lastIndexFlag && !lastIndexFlag2) {
        curActualSeqLen = validTopkNum * selectedBlockSize;
    }
}

template <typename NSAT>
__aicore__ inline void NsaSelectAttentionInfer<NSAT>::GetBN2id(const uint32_t bn2Idx)
{
    uint32_t globalTaskIdx = beforeBlockSplitBn2Nums + bn2Idx;
    uint32_t batchId = 0;
    uint32_t prevBatchTasks = 0;
    // 1. 计算 batchId
    while (batchId < batchSize && globalTaskIdx >= prevBatchTasks + actualQSeqLengthsGm.GetValue(batchId) * kvHeadNum) {
        prevBatchTasks += actualQSeqLengthsGm.GetValue(batchId) * kvHeadNum;
        batchId++;
    }
    bIdx = batchId;
    // 2. 计算 kvheadId 和 S1Id
    uint32_t offset = globalTaskIdx - prevBatchTasks;
    n2Idx = offset / actualQSeqLengthsGm.GetValue(batchId);
    s1Idx = offset % actualQSeqLengthsGm.GetValue(batchId);
}

template <typename NSAT>
__aicore__ inline void NsaSelectAttentionInfer<NSAT>::DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx)
{
    if ASCEND_IS_AIV {
        InitAllZeroOutput(bIdx, n2Idx);
    }
}

template <typename NSAT> __aicore__ inline void NsaSelectAttentionInfer<NSAT>::UpdateInnerLoopCond()
{
    if (curActualSeqLen == 0) {
        curActSeqLenIsZero = true;
        return;
    }
    curActSeqLenIsZero = false;

    int32_t remainSinnerSize = (int32_t)curActualSeqLen;
    int32_t computeSinnerSize = (int32_t)curActualSeqLen;
    if (remainSinnerSize > 0) {
        if (computeSinnerSize <= singleProcessSInnerSize) {
            singleProcessSInnerSizeTail = computeSinnerSize;
            sInnerLoopTimes = 1;
        } else {
            sInnerLoopTimes = (computeSinnerSize + singleProcessSInnerSize - 1) / singleProcessSInnerSize;
            singleProcessSInnerSizeTail = computeSinnerSize - (sInnerLoopTimes - 1) * singleProcessSInnerSize;
        }
    } else {
        sInnerLoopTimes = 0;
    }
}

template <typename NSAT>
__aicore__ inline void NsaSelectAttentionInfer<NSAT>::Init(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
    __gm__ uint8_t *topkIndices,  __gm__ uint8_t *attenMask, __gm__ uint8_t *blockTable,
    __gm__ uint8_t *actualQSeqLengths,  __gm__ uint8_t *actualSeqLengths,  __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
    const NsaSelectAttentionInferTilingData *__restrict tiling, __gm__ uint8_t *gmTiling, TPipe *tPipe)
{
    if ASCEND_IS_AIV {
        tmpBlockIdx = GetBlockIdx(); // vec:0-47
        aiCoreIdx = tmpBlockIdx / numDouble;
    } else {
        tmpBlockIdx = GetBlockIdx(); // cube:0-23
        aiCoreIdx = tmpBlockIdx;
    }
    tilingData = tiling;
    InitTilingData();
    InitCalcParamsEach();
    pipe = tPipe;
    keyPtr = key;
    valuePtr = value;
    blocktablePtr = blockTable;

    curSingleProcessSInnerSizeAlign = 0ULL; // prefix场景计算user prompt前必须重新初始化
    actualSingleProcessSInnerSize = 0ULL;
    actualSingleProcessSInnerSizeAlign = 0ULL;

    // init global buffer
    queryGm.SetGlobalBuffer((__gm__ Q_T *)query);
    attentionOutGm.SetGlobalBuffer((__gm__ OUT_T *)attentionOut);
    
    // batch连续时,只需要初始化一次;不连续时,需要在使用时根据batchIdx初始化
    keyGm.SetGlobalBuffer((__gm__ KV_T *)key);
    valueGm.SetGlobalBuffer((__gm__ KV_T *)value);

    if (pipe != nullptr) {
        InitBuffers();
    }
    InitActualSeqLen(actualSeqLengths);
    InitActualQSeqLen(actualQSeqLengths);
    blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
    topKGm.SetGlobalBuffer((__gm__ int32_t *)topkIndices);

    if ASCEND_IS_AIV {
        for (int i = 0; i < PRE_LOAD_NUM; i++) {
            softmaxMaxUb[i] = softmaxMaxBuff[i].Get<T>();
            softmaxSumUb[i] = softmaxSumBuff[i].Get<T>();
            softmaxExpUb[i] = softmaxExpBuff[i].Get<T>();
        }
        softmaxMaxDefaultUb = softmaxMaxDefaultBuff.Get<T>();
        softmaxSumDefaultUb = softmaxSumDefaultBuff.Get<T>();
    }

    uint64_t offset = 0;
    mm1ResGm.SetGlobalBuffer(
            (__gm__ MM_OUT_T *)(workspace + offset + aiCoreIdx * dbWorkspaceRatio * mmResUbSize * sizeof(MM_OUT_T)));
    offset += GetBlockNum() * dbWorkspaceRatio * mmResUbSize * sizeof(MM_OUT_T);

    vec1ResGm.SetGlobalBuffer(
        (__gm__ KV_T *)(workspace + offset + aiCoreIdx * dbWorkspaceRatio * mmResUbSize * sizeof(KV_T)));
    offset += GetBlockNum() * dbWorkspaceRatio * mmResUbSize * sizeof(KV_T);


    mm2ResGm.SetGlobalBuffer(
            (__gm__ MM_OUT_T *)(workspace + offset + aiCoreIdx * dbWorkspaceRatio * bmm2ResUbSize * sizeof(MM_OUT_T)));
    offset += GetBlockNum() * dbWorkspaceRatio * bmm2ResUbSize * sizeof(MM_OUT_T);

    vec2ResGm.SetGlobalBuffer(
            (__gm__ T *)(workspace + offset + aiCoreIdx * dbWorkspaceRatio* bmm2ResUbSize * sizeof(T)));
    offset += GetBlockNum() * dbWorkspaceRatio * bmm2ResUbSize * sizeof(T);
}

template <typename NSAT> __aicore__ inline void NsaSelectAttentionInfer<NSAT>::InitCalcParamsEach()
{
    // 这里是编译器优化写法，定义一个局部数组变量coreSidxEnd(存在栈上)，使用copy_data_align64接口
    // 可以只从ub中拷贝tiling中coreSidxEnd的内容到栈上，而非将整个nsaSelectAttentionInferCoreParams
    // 内容拷贝到栈，减少拷贝时间。
#ifdef ASCENDC_CPU_DEBUG
    const uint32_t *coreSidxEnd = tilingData->nsaSelectAttentionInferCoreParams.coreSidxEnd;
#else
    uint32_t coreSidxEnd[50];
    copy_data_align64((uint8_t *)coreSidxEnd, (uint8_t *)(tilingData->nsaSelectAttentionInferCoreParams.coreSidxEnd),
                    sizeof(coreSidxEnd));
#endif
    bn2LoopTimes = coreSidxEnd[aiCoreIdx + 1] - coreSidxEnd[aiCoreIdx];
    beforeBlockSplitBn2Nums = coreSidxEnd[aiCoreIdx];
}

template <typename NSAT>
__aicore__ inline void NsaSelectAttentionInfer<NSAT>::SoftmaxFlashV2Compute(uint32_t loop,
    LocalTensor<T> &mmResUb, LocalTensor<uint8_t> &softmaxTmpUb, uint32_t startRow, uint32_t dealRowCount,
    uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t baseOffset = startRow * BLOCK_ELEMENT_NUM;
    SoftMaxShapeInfo srcShape{dealRowCount, columnCount, dealRowCount, actualColumnCount};
    SoftMaxTiling newTiling =
        SoftMaxFlashV2TilingFunc(srcShape, sizeof(T), sizeof(T), softmaxTmpUb.GetSize(), true, false);

    LocalTensor<T> inSumTensor;
    LocalTensor<T> inMaxTensor;
    uint32_t outIdx = loop % (PRE_LOAD_NUM);
    if (extraInfo[loop % PRE_LOAD_NUM].isFirstSInnerLoop) {
        inMaxTensor = softmaxMaxDefaultUb;
        inSumTensor = softmaxSumDefaultUb;
    } else {
        uint32_t inIdx = (loop -1) % (PRE_LOAD_NUM);
        inMaxTensor = softmaxMaxUb[inIdx][baseOffset];
        inSumTensor = softmaxSumUb[inIdx][baseOffset];
    }

    AscendC::SoftmaxFlashV2<T, true, true, false, false, IFA_SOFTMAX_FLASHV2_CFG>(
        mmResUb,  // dstTensor
        softmaxSumUb[outIdx][baseOffset],  // expSumTensor 输出tensor
        softmaxMaxUb[outIdx][baseOffset],   // maxTensor rowmax 
        mmResUb,  // srcTensor
        softmaxExpUb[outIdx][baseOffset],  // expMaxTensor maxi
        inSumTensor,  // inExpSumTensor
        inMaxTensor, 
        softmaxTmpUb,  // 临时空间
        newTiling, 
        srcShape);
    PipeBarrier<PIPE_V>();
}

template <typename NSAT>
__aicore__ inline void
NsaSelectAttentionInfer<NSAT>::Bmm2DataCopyOut(uint64_t attenOutOffset, LocalTensor<OUT_T> &attenOutUb,
    uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = dealRowCount;
    dataCopyParams.blockLen = actualColumnCount * sizeof(OUT_T);
    dataCopyParams.srcStride = (columnCount - actualColumnCount) / (BYTE_BLOCK / sizeof(OUT_T));
    dataCopyParams.dstStride = 0;
    DataCopyPad(attentionOutGm[attenOutOffset + (gSizeStart + startRow) * actualColumnCount], attenOutUb, dataCopyParams);
}

template <typename NSAT>
__aicore__ inline void
NsaSelectAttentionInfer<NSAT>::Bmm2CastAndCopyOut(ExtraInfo& info, LocalTensor<T> &bmm2ResUb, uint32_t startRow,
                                                                uint32_t dealRowCount, uint32_t columnCount,
                                                                uint32_t actualColumnCount)
{
    LocalTensor<OUT_T> tmpBmm2ResCastTensor = outputQue1.AllocTensor<OUT_T>();
    if constexpr (IsSameType<OUT_T, bfloat16_t>::value) { // bf16 采取四舍六入五成双模式
        Cast(tmpBmm2ResCastTensor, bmm2ResUb, AscendC::RoundMode::CAST_RINT, dealRowCount * columnCount);
    } else {
        Cast(tmpBmm2ResCastTensor, bmm2ResUb, AscendC::RoundMode::CAST_ROUND, dealRowCount * columnCount);
    }
    outputQue1.EnQue(tmpBmm2ResCastTensor);
    outputQue1.DeQue<OUT_T>();
    Bmm2DataCopyOut(info.attenOutOffset, tmpBmm2ResCastTensor, startRow, dealRowCount, columnCount, actualColumnCount);
    outputQue1.FreeTensor(tmpBmm2ResCastTensor);
}

template <typename NSAT>
__aicore__ inline void
NsaSelectAttentionInfer<NSAT>::ElewiseCompute(uint32_t loop, LocalTensor<T> &mmResUb, uint32_t dealRowCount,
                                              uint32_t columnCount)
{
    Muls(mmResUb, mmResUb, static_cast<T>(tilingData->baseParams.scaleValue), dealRowCount * columnCount);
    PipeBarrier<PIPE_V>();
}

template <typename NSAT>
__aicore__ inline void
NsaSelectAttentionInfer<NSAT>::DealBmm1ResBaseBlock(const uint32_t loop, uint32_t startRow,
                                                    uint32_t dealRowCount, uint32_t columnCount,
                                                    uint32_t actualColumnCount)
{
    ExtraInfo& info = extraInfo[loop % (PRE_LOAD_NUM)];
    uint32_t computeSize = dealRowCount * columnCount;
    LocalTensor<T> mmResUb = tmpBuff1.Get<T>();

    uint64_t inOutGmOffset = (loop % PRE_LOAD_NUM) * mmResUbSize + (gSizeStart + startRow) * columnCount;
    LocalTensor<MM_OUT_T> tmpMmResUb = inputQue1.AllocTensor<MM_OUT_T>();
    DataCopy(tmpMmResUb, mm1ResGm[inOutGmOffset], computeSize);
    inputQue1.EnQue(tmpMmResUb);
    inputQue1.DeQue<MM_OUT_T>();
    DataCopy(mmResUb, tmpMmResUb, computeSize);
    inputQue1.FreeTensor(tmpMmResUb);
    PipeBarrier<PIPE_V>();
    ElewiseCompute(loop, mmResUb, dealRowCount, columnCount);
    LocalTensor<T> tmpAFloorUb = tmpBuff2.Get<T>();
    LocalTensor<uint8_t> softmaxTmpUb = tmpAFloorUb.template ReinterpretCast<uint8_t>();
    SoftmaxFlashV2Compute(loop, mmResUb, softmaxTmpUb, startRow, dealRowCount, columnCount, actualColumnCount);

    LocalTensor<KV_T> tmpMMResCastTensor = outputQue1.AllocTensor<KV_T>();
    Cast(tmpMMResCastTensor, mmResUb, AscendC::RoundMode::CAST_ROUND, computeSize);

    outputQue1.EnQue(tmpMMResCastTensor);
    outputQue1.DeQue<KV_T>();
    DataCopy(vec1ResGm[inOutGmOffset], tmpMMResCastTensor, computeSize);
    
    outputQue1.FreeTensor(tmpMMResCastTensor);
}

template <typename NSAT>
__aicore__ inline void NsaSelectAttentionInfer<NSAT>::ProcessVec1Inner(uint32_t loop)
{
    ExtraInfo& info = extraInfo[loop % (PRE_LOAD_NUM)];
    uint32_t gSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / info.actualSingleProcessSInnerSizeAlign;
    if (gSplitSize > gSizeVector) {
        gSplitSize = gSizeVector;
    }
    uint32_t loopCount = (gSizeVector + gSplitSize - 1) / gSplitSize;
    uint32_t tailSplitSize = gSizeVector - (loopCount - 1) * gSplitSize;
    for (uint32_t i = 0, dealSize = gSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        DealBmm1ResBaseBlock(loop, i * gSplitSize, dealSize, info.actualSingleProcessSInnerSizeAlign,
                                info.actualSingleProcessSInnerSize);
    }
}

template <typename NSAT>
__aicore__ inline void
NsaSelectAttentionInfer<NSAT>::DealBmm2ResBaseBlock(const uint32_t loop, uint32_t startRow,
                                                    uint32_t dealRowCount, uint32_t columnCount,
                                                    uint32_t actualColumnCount)
{
    ExtraInfo& info = extraInfo[loop % (PRE_LOAD_NUM)];
    uint32_t vec2ComputeSize = dealRowCount * columnCount;
    uint32_t baseOffset = startRow * BLOCK_ELEMENT_NUM;
    LocalTensor<T> bmm2ResUb = tmpBuff1.Get<T>();
    bmm2ResUb.SetSize(vec2ComputeSize);

    uint64_t inOutBaseOffset = (gSizeStart + startRow) * columnCount;
    uint64_t srcGmOffset = (loop % PRE_LOAD_NUM) * bmm2ResUbSize + inOutBaseOffset;
    LocalTensor<MM_OUT_T> tmpBmm2ResUb = inputQue1.AllocTensor<MM_OUT_T>();
    DataCopy(tmpBmm2ResUb, mm2ResGm[srcGmOffset], vec2ComputeSize);
    inputQue1.EnQue(tmpBmm2ResUb);
    inputQue1.DeQue<MM_OUT_T>();
    DataCopy(bmm2ResUb, tmpBmm2ResUb, vec2ComputeSize);
    inputQue1.FreeTensor(tmpBmm2ResUb);

    // 除第一个循环外，均需要更新中间计算结果
    if (info.s2Idx > 0) {
        event_t eventIdMte2WaitMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte2WaitMte3);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte2WaitMte3);
        LocalTensor<T> bmm2ResPreUb = inputQue2.AllocTensor<T>();
        uint64_t vec2ResGmOffset = ((loop - 1) % PRE_LOAD_NUM) * bmm2ResUbSize + inOutBaseOffset;
        DataCopy(bmm2ResPreUb, vec2ResGm[vec2ResGmOffset], vec2ComputeSize);
        inputQue2.EnQue(bmm2ResPreUb);

        inputQue2.DeQue<T>();
        PipeBarrier<PIPE_V>();
        RowMuls(bmm2ResPreUb, bmm2ResPreUb, softmaxExpUb[loop % (PRE_LOAD_NUM)][baseOffset], dealRowCount, columnCount, actualColumnCount);
        PipeBarrier<PIPE_V>();
        Add(bmm2ResUb, bmm2ResUb, bmm2ResPreUb, vec2ComputeSize);
        inputQue2.FreeTensor(bmm2ResPreUb);
    }

    // 最后一次输出计算结果，否则将中间结果暂存至workspace
    if (info.s2Idx + 1 == info.curSInnerLoopTimes) {
        PipeBarrier<PIPE_V>();
        RowDivs(bmm2ResUb, bmm2ResUb, softmaxSumUb[loop % (PRE_LOAD_NUM)][baseOffset], dealRowCount, columnCount, actualColumnCount);

        PipeBarrier<PIPE_V>();
        Bmm2CastAndCopyOut(info, bmm2ResUb, startRow, dealRowCount, columnCount, actualColumnCount);
    } else {
        PipeBarrier<PIPE_V>();
        LocalTensor<T> tmpBmm2Res = outputQue1.AllocTensor<T>();
        DataCopy(tmpBmm2Res, bmm2ResUb, dealRowCount * columnCount);
        outputQue1.EnQue(tmpBmm2Res);
        outputQue1.DeQue<T>();
        uint64_t vec2ResGmOffset = (loop % PRE_LOAD_NUM) * bmm2ResUbSize + inOutBaseOffset;
        DataCopy(vec2ResGm[vec2ResGmOffset], tmpBmm2Res, vec2ComputeSize);

        outputQue1.FreeTensor(tmpBmm2Res);
    }
}

template <typename NSAT>
__aicore__ inline void NsaSelectAttentionInfer<NSAT>::ProcessVec2Inner(uint32_t loop)
{
    uint32_t gSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / headDimVAlign;
    if (gSplitSize > gSizeVector) {
        gSplitSize = gSizeVector;
    }
    uint32_t loopCount = (gSizeVector + gSplitSize - 1) / gSplitSize;
    uint32_t tailSplitSize = gSizeVector - (loopCount - 1) * gSplitSize;

    for (uint32_t i = 0, dealSize = gSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        DealBmm2ResBaseBlock(loop, i * gSplitSize, dealSize, headDimVAlign, headDimV);
    }
}

template <typename NSAT>
__aicore__ inline void NsaSelectAttentionInfer<NSAT>::CalcParams(uint32_t loop) {
    ExtraInfo& info = extraInfo[loop % (PRE_LOAD_NUM)];
    info.loop = loop;
    if (loop != 0) {
        {
            if (curS2Idx + 1 >= curSInnerLoopTimes) {
                curS2Idx = 0;
                curS1Idx = (curS1Idx  + 1) % actualQSeqLengthsGm.GetValue(curBIdx);
                curN2Idx = (curN2Idx  + (curS1Idx == 0 ? 1 : 0)) % kvHeadNum;
                curBIdx = curBIdx + (curN2Idx == 0 && curS1Idx == 0 ? 1 : 0);
                curActSeqLenIsZero = true;
                while (curActSeqLenIsZero) {
                    bIdx = curBIdx;
                    n2Idx = curN2Idx;
                    s1Idx = curS1Idx;
                    GetActualSeqLen();
                    // 根据当前块实际长度, 重配flashattention循环条件
                    UpdateInnerLoopCond();
                    if (curActSeqLenIsZero) {
                        curBIdx++;
                        curN2Idx = 0;
                        continue;
                    }
                }
                curS2Idx = s2Idx;
                curSInnerLoopTimes = sInnerLoopTimes;
            } else {
                curS2Idx++;
            }
        }
    } else {
        {
            // b * S
            curActSeqLenIsZero = true;
            while (curActSeqLenIsZero) {
                bIdx = curBIdx;
                n2Idx = curN2Idx;
                s1Idx = curS1Idx;
                GetActualSeqLen();
                // 根据当前块实际长度, 重配flashattention循环条件
                UpdateInnerLoopCond();
                if (curActSeqLenIsZero) {
                    curBIdx++;
                    curN2Idx = 0;
                    continue;
                }
            }
            curS2Idx = s2Idx;
            curSInnerLoopTimes = sInnerLoopTimes;
        }
    }
    info.bIdx = curBIdx;
    info.n2Idx = curN2Idx;
    info.s2Idx = curS2Idx;
    info.s1Idx = curS1Idx;
    info.curSInnerLoopTimes = curSInnerLoopTimes;

    info.isFirstSInnerLoop = (info.s2Idx == 0);
    info.bn2IdxInPreCore = info.bn2IdxInCurCore;
    if (info.isFirstSInnerLoop) {
        bn2IdxInCurCore++;
    }
    info.bn2IdxInCurCore = bn2IdxInCurCore - 1;
    if (info.isFirstSInnerLoop) {
        if constexpr (LAYOUT_T == LAYOUT::TND) {
            uint32_t curTotalQSeqLenOffset = 0;
            for (uint64_t i = 0; i < info.bIdx; ++i) {
                curTotalQSeqLenOffset += actualQSeqLengthsGm.GetValue(i);
            }
            tensorACoreOffset = curTotalQSeqLenOffset * qHeadNum * headDim + info.s1Idx * qHeadNum * headDim + info.n2Idx * gSize * headDim;
            tensorAttenOutCoreOffset = curTotalQSeqLenOffset * qHeadNum * headDimV + info.s1Idx * qHeadNum * headDimV + info.n2Idx * gSize * headDimV; //bsnd
        } else {
            // B,S1,N2,G,D
            tensorACoreOffset = info.bIdx * qSeqSize * qHeadNum * headDim + info.s1Idx * qHeadNum * headDim + info.n2Idx * gSize * headDim; //bsnd
            tensorAttenOutCoreOffset = info.bIdx * qSeqSize * qHeadNum * headDimV + info.s1Idx * qHeadNum * headDimV + info.n2Idx * gSize * headDimV; //bsnd
        }
    }
    info.tensorAOffset = tensorACoreOffset;
    info.attenOutOffset = tensorAttenOutCoreOffset;
    info.actualSingleProcessSInnerSize = singleProcessSInnerSize;
    if (info.s2Idx == info.curSInnerLoopTimes - 1) {
        info.actualSingleProcessSInnerSize = singleProcessSInnerSizeTail;
    }
    info.actualSingleProcessSInnerSizeAlign = Align((uint32_t)info.actualSingleProcessSInnerSize, HALF_BLOCK_NUM);

    info.needSetOrgShape = (curSingleProcessSInnerSizeAlign != info.actualSingleProcessSInnerSizeAlign);
    if (info.needSetOrgShape) {
        curSingleProcessSInnerSizeAlign = info.actualSingleProcessSInnerSizeAlign;
    }

    uint64_t sInnerOffsetDataSize = info.s2Idx * singleProcessSInnerSize;

    info.s2BatchOffset = sInnerOffsetDataSize;
}

template <typename NSAT>
__aicore__ inline void NsaSelectAttentionInfer<NSAT>::ProcessVec2L(uint32_t loop) {
    if (gSizeVector == 0) {
        return;
    }
    ProcessVec2Inner(loop);
}

template <typename NSAT>
__aicore__ inline void NsaSelectAttentionInfer<NSAT>::ProcessVec1L(uint32_t loop)
{
    if (gSizeVector == 0) {
        return;
    }
    ProcessVec1Inner(loop);
}

template <typename NSAT> __aicore__ inline void NsaSelectAttentionInfer<NSAT>::SetLoopTimes() {
    // 获取总的循环次数，此处假设所有head的S=Smax
    bn2s2LoopTimes = 0;
    bool hasInit = false;
    for (uint32_t bn2Idx = 0; bn2Idx < bn2LoopTimes; bn2Idx++) {
        GetBN2id(bn2Idx);
        GetActualSeqLen();
        // 根据当前块实际长度, 重配flashattention循环条件
        UpdateInnerLoopCond();
        if (curActSeqLenIsZero) {
            DealActSeqLenIsZero(bIdx, n2Idx);
            continue;
        }
        if (!hasInit) {
            curBIdx = bIdx;
            curN2Idx = n2Idx;
            curS2Idx = s2Idx;
            curS1Idx = s1Idx;
            curSInnerLoopTimes = sInnerLoopTimes;
            hasInit = true;
        }
        bn2s2LoopTimes += sInnerLoopTimes;
    }
}

template <typename NSAT>
__aicore__ inline void NsaSelectAttentionInfer<NSAT>::CopyInMm1AToL1(LocalTensor<KV_T> &l1Tensor, ExtraInfo& info)
{
    uint32_t mmRowCount = msdIterNum * gSize;
    uint32_t copyIterNum = (mmRowCount + 15) / 16;
    uint32_t copyStride = 16 * headDimAlign;
    for(int i = 0; i < copyIterNum; i++){
        Nd2NzParams mm1Nd2NzParamsForA;
        mm1Nd2NzParamsForA.ndNum = 1; // ND矩阵的个数
        if(i == copyIterNum - 1) {
            mm1Nd2NzParamsForA.nValue = msdIterNum * gSize - i * 16;
        }
        else {
            mm1Nd2NzParamsForA.nValue = 16; // 单个ND矩阵的行数, 单位为元素个数  16
        }
        mm1Nd2NzParamsForA.dValue = headDimAlign; // 单个ND矩阵的列数, 单位为元素个数   128
        mm1Nd2NzParamsForA.srcDValue = headDimAlign; // 同一个ND矩阵中相邻行起始地址之间的偏移, 单位为元素个数  128
        mm1Nd2NzParamsForA.srcNdMatrixStride = 0; // 相邻ND矩阵起始地址之间的偏移, 单位为元素个数
        mm1Nd2NzParamsForA.dstNzC0Stride = 16; // 转换为NZ矩阵后，相邻Block起始地址之间的偏移, 单位为Block个数  16
        mm1Nd2NzParamsForA.dstNzNStride = 1; // 转换为NZ矩阵后，ND中之前相邻两行在NZ矩阵中起始地址之间的偏移
        mm1Nd2NzParamsForA.dstNzMatrixStride = 0; // 两个NZ矩阵，起始地址之间的偏移   
        DataCopy(l1Tensor[i * copyStride], queryGm[info.tensorAOffset + i * copyStride], mm1Nd2NzParamsForA);
    }
}

template <typename NSAT>
__aicore__ inline void NsaSelectAttentionInfer<NSAT>::CopyInMm1BToL1ForPA(
    LocalTensor<KV_T>& bL1Tensor, uint64_t keyGmBaseOffset, uint32_t copyTotalRowCnt, uint32_t copyStartRowCnt, uint32_t nActCopyRowCount)
{
    uint32_t baseN = 256 / sizeof(KV_T);
    uint64_t step = headDim;
    step = headDim * kvHeadNum;
    uint32_t nHead = 0;
    uint32_t nTail = 0;
    uint32_t midNdNum = 0;
    uint32_t copyStartRowCntInBaseN = copyStartRowCnt % baseN;
    uint32_t copyStartRowCntInBaseNTail = 0;
    if (copyStartRowCntInBaseN + nActCopyRowCount <= baseN) {
        nHead = 0;
        midNdNum = 0;
        nTail = nActCopyRowCount;
        copyStartRowCntInBaseNTail = copyStartRowCntInBaseN;
    } else {
        if (copyStartRowCntInBaseN == 0) {
            nHead = 0;
        } else {
            nHead = baseN - copyStartRowCntInBaseN;
        }
        midNdNum = (nActCopyRowCount - nHead) / baseN;
        nTail = (nActCopyRowCount - nHead) % baseN;
        copyStartRowCntInBaseNTail = 0;
    }

    uint32_t dstNzC0StrideTail = baseN;
    if (copyTotalRowCnt % baseN != 0) {
        if ((copyStartRowCnt + nActCopyRowCount) > (copyTotalRowCnt / baseN * baseN)) {
            dstNzC0StrideTail = copyTotalRowCnt % baseN;
        }
    }
    Nd2NzParams mm1Nd2NzParamsForB;
    mm1Nd2NzParamsForB.dValue = headDim;
    mm1Nd2NzParamsForB.srcDValue = step; // 同一个ND矩阵中相邻行起始地址之间的偏移, 单位为元素个数
    mm1Nd2NzParamsForB.dstNzC0Stride = baseN; // 转换为NZ矩阵后，相邻Block起始地址之间的偏移, 单位为Block个数 
    mm1Nd2NzParamsForB.dstNzNStride = 1; // 转换为NZ矩阵后，ND中之前相邻两行在NZ矩阵中起始地址之间的偏移, 单位为Block个数
    if (nHead != 0) {
        mm1Nd2NzParamsForB.ndNum = 1;
        mm1Nd2NzParamsForB.nValue = nHead; // 单个ND矩阵的行数, 单位为元素个数
        mm1Nd2NzParamsForB.srcNdMatrixStride = 0; // 相邻ND矩阵起始地址之间的偏移, 单位为元素个数
        mm1Nd2NzParamsForB.dstNzMatrixStride = 0; // 两个NZ矩阵，起始地址之间的偏移, 单位为元素个数

        uint32_t ndNumFinish = copyStartRowCnt / baseN;
        DataCopy(bL1Tensor[ndNumFinish * baseN * headDimAlign + copyStartRowCntInBaseN * (32 / sizeof(KV_T))], keyGm[keyGmBaseOffset],
                mm1Nd2NzParamsForB);
    }

    if (midNdNum != 0) {
        mm1Nd2NzParamsForB.nValue = baseN; // 单个ND矩阵的行数, 单位为元素个数   256
        mm1Nd2NzParamsForB.dValue = headDim; // 单个ND矩阵的列数, 单位为元素个数   128
        mm1Nd2NzParamsForB.srcDValue = step; // 同一个ND矩阵中相邻行起始地址之间的偏移, 单位为元素个数  // 128
        mm1Nd2NzParamsForB.dstNzC0Stride = baseN;// 转换为NZ矩阵后，相邻Block起始地址之间的偏移, 单位为Block个数   256
        mm1Nd2NzParamsForB.dstNzNStride = 1; // 转换为NZ矩阵后，ND中之前相邻两行在NZ矩阵中起始地址之间的偏移, 单位为Block个数
        int32_t ndNumFinish = (copyStartRowCnt + nHead) / baseN;
        if ((LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND || LAYOUT_T == LAYOUT::TND) && (baseN * step > 65535)) {
            mm1Nd2NzParamsForB.ndNum = 1;
            mm1Nd2NzParamsForB.srcNdMatrixStride = 0; // 相邻ND矩阵起始地址之间的偏移, 单位为元素个数 256*128
            mm1Nd2NzParamsForB.dstNzMatrixStride = 0; // 两个NZ矩阵，起始地址之间的偏移, 单位为元素个数 256*128
            for (uint32_t i = 0; i < midNdNum; i++) {
                DataCopy(bL1Tensor[(ndNumFinish + i) * baseN * headDimAlign], keyGm[keyGmBaseOffset + nHead * step + i * baseN * step], mm1Nd2NzParamsForB);
            }
        } else {
            mm1Nd2NzParamsForB.ndNum = midNdNum;
            mm1Nd2NzParamsForB.srcNdMatrixStride = baseN * step; // 相邻ND矩阵起始地址之间的偏移, 单位为元素个数 256*128
            mm1Nd2NzParamsForB.dstNzMatrixStride = baseN * headDimAlign; // 两个NZ矩阵，起始地址之间的偏移, 单位为元素个数 256*128
            DataCopy(bL1Tensor[ndNumFinish * baseN * headDimAlign], keyGm[keyGmBaseOffset + nHead * step], mm1Nd2NzParamsForB);
        }
    }
    if (nTail != 0) {
        mm1Nd2NzParamsForB.ndNum = 1;
        mm1Nd2NzParamsForB.nValue = nTail; // 单个ND矩阵的行数, 单位为元素个数   actualSingleProcessSInnerSizeAlign为32对齐，nTail已经是32对齐
        mm1Nd2NzParamsForB.dValue = headDim; // 单个ND矩阵的列数, 单位为元素个数   128
        mm1Nd2NzParamsForB.srcNdMatrixStride = 0; // 相邻ND矩阵起始地址之间的偏移, 单位为元素个数 0
        mm1Nd2NzParamsForB.srcDValue = step; // 同一个ND矩阵中相邻行起始地址之间的偏移, 单位为元素个数  // 128
        mm1Nd2NzParamsForB.dstNzC0Stride = dstNzC0StrideTail;// 转换为NZ矩阵后，相邻Block起始地址之间的偏移, 单位为Block个数 
        mm1Nd2NzParamsForB.dstNzNStride = 1; // 转换为NZ矩阵后，ND中之前相邻两行在NZ矩阵中起始地址之间的偏移, 单位为Block个数
        mm1Nd2NzParamsForB.dstNzMatrixStride = 0; // 两个NZ矩阵，起始地址之间的偏移, 单位为元素个数  0

        int32_t ndNumFinish = (copyStartRowCnt + nHead) / baseN + midNdNum;
        DataCopy(bL1Tensor[ndNumFinish * baseN * headDimAlign + copyStartRowCntInBaseNTail * (32 / sizeof(KV_T))], keyGm[keyGmBaseOffset + (nHead + midNdNum * baseN) * step],
                mm1Nd2NzParamsForB);  //需要调整偏移地址，bL1Tensor的偏移， keyGm的偏移
    }
}

template <typename NSAT>
__aicore__ inline void NsaSelectAttentionInfer<NSAT>::LoadDataMm1A(LocalTensor<KV_T>& aL0Tensor,
                        LocalTensor<KV_T>& aL1Tensor, uint32_t k, uint32_t copyColCnt)
{
    uint32_t mmRowCount = msdIterNum * gSize;
    uint32_t copyIterNum = (mmRowCount + 15) / 16;
    for(int i = 0; i < copyIterNum; i++) {
        LoadData2DParams loadData2DParams;
        loadData2DParams.startIndex = 0;
        loadData2DParams.repeatTimes = 16 / 16 * copyColCnt / (32 / sizeof(KV_T)); // copyColCnt:128/64
        loadData2DParams.srcStride = 1;
        loadData2DParams.dstGap = 0;
        loadData2DParams.ifTranspose = false;
        LoadData(aL0Tensor[i * 16 * copyColCnt], aL1Tensor[16 * i * 192 + 16 * 128 * k], loadData2DParams);
    }
}

template <typename NSAT>
__aicore__ inline void NsaSelectAttentionInfer<NSAT>::LoadDataMm1B(LocalTensor<KV_T> &bl0Tensor,
    LocalTensor<KV_T> &bl1Tensor,  uint32_t k, uint32_t copyColCnt, uint32_t i, uint32_t copyRowCnt)
{
    uint32_t blockElementCnt = 32 / sizeof(KV_T);
    LoadData2DParams loadData2DParamsForB;
    loadData2DParamsForB.startIndex = 0;
    loadData2DParamsForB.srcStride = 1;
    loadData2DParamsForB.dstGap = 0;
    loadData2DParamsForB.ifTranspose = false;
    loadData2DParamsForB.repeatTimes = (copyRowCnt / 16) * (copyColCnt / blockElementCnt);
    LoadData(bl0Tensor, bl1Tensor[copyRowCnt * 128 * k], loadData2DParamsForB);
}

template <typename NSAT>
__aicore__ inline void NsaSelectAttentionInfer<NSAT>::ComputeMm1(uint32_t loop) {
    ExtraInfo& info = extraInfo[loop % (PRE_LOAD_NUM)];
    gSize = gSizeCube;
    LocalTensor<KV_T> aL1Tensor = qL1Tensor;
    WaitFlag<HardEvent::MTE1_MTE2>(Q_EVENT1);
    CopyInMm1AToL1(aL1Tensor, info);
    SetFlag<HardEvent::MTE2_MTE1>(Q_EVENT1);
    WaitFlag<HardEvent::MTE2_MTE1>(Q_EVENT1);
    if (info.bn2IdxInCurCore != info.bn2IdxInPreCore && info.isFirstSInnerLoop) {
        x = 0;
        currentTopDeal = -1;
        topKIdOffset = 0;
        idIntopKRecord = -1;
    }
    constexpr int64_t nCopyRowCount = 128; // n方向切分
    int64_t nCopyTimes = (info.actualSingleProcessSInnerSize + nCopyRowCount - 1) / nCopyRowCount;
    int64_t nTailCopyRowCount = info.actualSingleProcessSInnerSize - (nCopyTimes - 1) * nCopyRowCount;
    int64_t nTailCopyRowCountAlign = info.actualSingleProcessSInnerSizeAlign - (nCopyTimes - 1) * nCopyRowCount;
    for (int64_t nCopyIdx = 0, nActCopyRowCount = nCopyRowCount, nActCopyRowCountAlign = nCopyRowCount; nCopyIdx < nCopyTimes; nCopyIdx++) {
        if (nCopyIdx + 1 == nCopyTimes) {
            nActCopyRowCount = nTailCopyRowCount;
            nActCopyRowCountAlign = nTailCopyRowCountAlign;
        }
        WaitFlag<HardEvent::MTE1_MTE2>(KP_EVENT0 + (kpL1BufIter % 2));
        LocalTensor<KV_T> bL1Tensor = kpL1Tensor[(kpL1BufIter % 2) * (L1_KP_SIZE / sizeof(KV_T))];
        int64_t topKBaseOffsetmm1 = 0;
        if constexpr (LAYOUT_T == LAYOUT::TND) {
            uint32_t curTotalQSeqLenOffset = 0;
            for (uint64_t i = 0; i < info.bIdx; ++i) {
                curTotalQSeqLenOffset += actualQSeqLengthsGm.GetValue(i);
            }
            topKBaseOffsetmm1 = curTotalQSeqLenOffset * kvHeadNum * selectedBlockCount + info.s1Idx * kvHeadNum * selectedBlockCount;
        } else {
            topKBaseOffsetmm1 = info.bIdx * qSeqSize * kvHeadNum * selectedBlockCount + info.s1Idx * kvHeadNum * selectedBlockCount;
        }
        int64_t blockTableBaseOffset = info.bIdx * maxBlockNumPerBatch;
        int64_t curSeqIdx = info.s2BatchOffset + nCopyIdx * nCopyRowCount;
        int64_t copyStartRowCnt = 0;
        while (copyStartRowCnt < nActCopyRowCount) {
            if (x == currentTopDeal) {
                topKIdOffset += 1;
            }
            int64_t idIntopK = topKGm.GetValue(topKBaseOffsetmm1 + topKIdOffset);
            if (idIntopK != -1) {
                if(idIntopKRecord != idIntopK) {
                    x = 0;
                    idIntopKRecord = idIntopK;
                    int64_t globalStart = idIntopK * selectedBlockSize;
                    int64_t globalEnd = (globalStart + selectedBlockSize > curActualSeqLenOri ) ?
                                        curActualSeqLenOri : (globalStart + selectedBlockSize);
                    currentTopDeal = globalEnd - globalStart;
                }
                if (currentTopDeal > 0) {
                    int64_t globalStart = idIntopK * selectedBlockSize;
                    int64_t globalEnd = (globalStart + selectedBlockSize > curActualSeqLenOri ) ?
                                        curActualSeqLenOri : (globalStart + selectedBlockSize);
                    globalStart += x;
                    int64_t start_offset = globalStart % kvCacheBlockSize;
                    int64_t start_block_idx = globalStart / kvCacheBlockSize;
                    int64_t end_block_idx = (globalEnd - 1) / kvCacheBlockSize;
                    int64_t reaminRowCnt = start_offset;
                    int64_t copyRowCnt = start_block_idx == end_block_idx ?
                                    globalEnd - globalStart : kvCacheBlockSize - reaminRowCnt;
                    if (copyStartRowCnt + copyRowCnt > nActCopyRowCount) {
                        copyRowCnt = nActCopyRowCount - copyStartRowCnt;
                    }
                    int64_t idInBlockTable =
                        blockTableGm.GetValue(blockTableBaseOffset + start_block_idx);
                    int64_t keyOffset = idInBlockTable * kvCacheBlockSize * headDim * kvHeadNum;
                    keyOffset += (int64_t)(info.n2Idx * headDim) + reaminRowCnt * headDim * kvHeadNum;
                    CopyInMm1BToL1ForPA(bL1Tensor, keyOffset, nActCopyRowCountAlign, copyStartRowCnt, copyRowCnt);
                    // 更新循环变量
                    copyStartRowCnt += copyRowCnt;
                    curSeqIdx += copyRowCnt;
                    x += copyRowCnt;
                } else {
                    x = currentTopDeal;
                }
            }
        }
        SetFlag<HardEvent::MTE2_MTE1>(KP_EVENT0 + (kpL1BufIter % 2));
        WaitFlag<HardEvent::MTE2_MTE1>(KP_EVENT0 + (kpL1BufIter % 2));

        WaitFlag<HardEvent::FIX_M>(L0C_EVENT0 + cL0BufIter % 2);
        LocalTensor<MM_OUT_T> cL0Tensor = cL0TensorPingPong[(cL0BufIter % 2) * (L0C_PP_SIZE / sizeof(MM_OUT_T))];
        uint32_t kSplitSize = 128;
        uint32_t i = 0;
        uint32_t kSize = headDim;
        uint32_t kLoops = (kSize + kSplitSize - 1) / kSplitSize;
        uint32_t subKSize = kSplitSize;
        for (uint32_t k = 0; k < kLoops; k++) {
            if (k + 1 == kLoops) {
                subKSize = kSize - (kLoops - 1) * kSplitSize;
            }
            WaitFlag<HardEvent::M_MTE1>(L0A_EVENT0 + aL0BufIter % 2);
            LocalTensor<KV_T> aL0Tensor = aL0TensorPingPong[(aL0BufIter % 2) * (L0A_PP_SIZE / sizeof(KV_T))];
            LoadDataMm1A(aL0Tensor, aL1Tensor, k, subKSize);
            SetFlag<HardEvent::MTE1_M>(L0A_EVENT0 + aL0BufIter % 2);
            WaitFlag<HardEvent::MTE1_M>(L0A_EVENT0 + aL0BufIter % 2);

            WaitFlag<HardEvent::M_MTE1>(L0B_EVENT0 + bL0BufIter % 2);
            LocalTensor<KV_T> bL0Tensor = bL0TensorPingPong[(bL0BufIter % 2) * (L0B_PP_SIZE / sizeof(KV_T))];
            LoadDataMm1B(bL0Tensor, bL1Tensor, k, subKSize, i, nActCopyRowCountAlign);
            SetFlag<HardEvent::MTE1_M>(L0B_EVENT0 + bL0BufIter % 2);
            WaitFlag<HardEvent::MTE1_M>(L0B_EVENT0 + bL0BufIter % 2);
            MmadParams mmadParams;
            mmadParams.m = msdIterNum * gSize;
            if (mmadParams.m == 1) { //m等于1会默认开GEMV模式，且不可关闭GEMV，所以规避当作矩阵计算
                mmadParams.m = 16;
            }
            mmadParams.n = nActCopyRowCountAlign;
            mmadParams.k = subKSize;
            mmadParams.cmatrixInitVal = (k == 0); // true：C矩阵初始值为0；
            mmadParams.cmatrixSource = false;
            Mmad(cL0Tensor, aL0Tensor, bL0Tensor, mmadParams);
            PipeBarrier<PIPE_M>();
            SetFlag<HardEvent::M_MTE1>(L0A_EVENT0 + aL0BufIter % 2);
            SetFlag<HardEvent::M_MTE1>(L0B_EVENT0 + bL0BufIter % 2);
            aL0BufIter++;
            bL0BufIter++;
        }
        SetFlag<HardEvent::MTE1_MTE2>(KP_EVENT0 + (kpL1BufIter % 2));
        kpL1BufIter++;
        SetFlag<HardEvent::M_FIX>(L0C_EVENT0 + cL0BufIter % 2);
        WaitFlag<HardEvent::M_FIX>(L0C_EVENT0 + cL0BufIter % 2);
        FixpipeParamsV220 fixParams;
        fixParams.nSize = nActCopyRowCountAlign;
        fixParams.mSize = msdIterNum * gSize; // 有效数据不足16行，只需要输出部分行即可
        fixParams.srcStride = ((fixParams.mSize + 15) / 16) * 16;
        fixParams.dstStride = info.actualSingleProcessSInnerSizeAlign; // mm1ResGm两行之间的间隔
        fixParams.ndNum = 1;
        Fixpipe(mm1ResGm[(loop % (PRE_LOAD_NUM)) * mmResUbSize + nCopyIdx * nCopyRowCount], cL0Tensor, fixParams);
        SetFlag<HardEvent::FIX_M>(L0C_EVENT0 + cL0BufIter % 2);
        cL0BufIter++;
    }
    SetFlag<HardEvent::MTE1_MTE2>(Q_EVENT1);
}

template <typename NSAT>
__aicore__ inline void NsaSelectAttentionInfer<NSAT>::CopyInMm2AToL1(LocalTensor<KV_T>& aL1Tensor, ExtraInfo& info, uint32_t kCopyIdx,
        uint32_t kCopyRowCount, uint32_t kActCopyRowCountAlign, uint32_t kActCopyRowCount) {
    uint32_t mmRowCount = msdIterNum * gSize;
    uint32_t copyStrideL1 = 16 * kActCopyRowCountAlign;
    uint32_t copyStrideGm = 16 * info.actualSingleProcessSInnerSizeAlign;
    uint32_t copyIterNum = (mmRowCount + 15) / 16;
    for(int i = 0; i < copyIterNum; i++){
        Nd2NzParams mm1Nd2NzParamsForA;
        mm1Nd2NzParamsForA.ndNum = 1; // ND矩阵的个数
        if(i == copyIterNum - 1) {
            mm1Nd2NzParamsForA.nValue = msdIterNum * gSize - i * 16;
        }
        else {
            mm1Nd2NzParamsForA.nValue = 16; // 单个ND矩阵的行数, 单位为元素个数  16
        }
        mm1Nd2NzParamsForA.dValue = kActCopyRowCount; // 单个ND矩阵的列数, 单位为元素个数
        mm1Nd2NzParamsForA.srcNdMatrixStride = 0; // 相邻ND矩阵起始地址之间的偏移, 单位为元素个数
        mm1Nd2NzParamsForA.srcDValue = info.actualSingleProcessSInnerSizeAlign; // 同一个ND矩阵中相邻行起始地址之间的偏移, 单位为元素个数
        mm1Nd2NzParamsForA.dstNzC0Stride = 16; // 转换为NZ矩阵后，相邻Block起始地址之间的偏移, 单位为Block个数
        mm1Nd2NzParamsForA.dstNzNStride = 1; // 转换为NZ矩阵后，ND中之前相邻两行在NZ矩阵中起始地址之间的偏移
        mm1Nd2NzParamsForA.dstNzMatrixStride = 0; // 两个NZ矩阵，起始地址之间的偏移
        DataCopy(aL1Tensor[i * copyStrideL1], vec1ResGm[(info.loop % (PRE_LOAD_NUM)) * mmResUbSize + kCopyIdx * kCopyRowCount + i * copyStrideGm], mm1Nd2NzParamsForA);
    }
}


template <typename NSAT>
__aicore__ inline void NsaSelectAttentionInfer<NSAT>::CopyInMm2BToL1ForPA(
    LocalTensor<KV_T>& bL1Tensor, uint64_t valueGmBaseOffset, uint32_t copyTotalRowCnt, uint32_t copyStartRowCnt, uint32_t kActCopyRowCount)
{
    uint32_t blockK = 32 / sizeof(KV_T); // 单个ND矩阵的行数
    uint64_t step = headDimV;
    if constexpr (LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND || LAYOUT_T == LAYOUT::TND) {
        step = headDimV * kvHeadNum;
    }

    uint32_t nHead = 0;
    uint32_t nTail = 0;
    uint32_t midNdNum = 0;
    uint32_t copyStartRowCntInBaseN = copyStartRowCnt % blockK;
    uint32_t copyStartRowCntInBaseNTail = 0;
    if (copyStartRowCntInBaseN + kActCopyRowCount <= blockK) {
        nHead = 0;
        midNdNum = 0;
        nTail = kActCopyRowCount;
        copyStartRowCntInBaseNTail = copyStartRowCntInBaseN;
    } else {
        if (copyStartRowCntInBaseN == 0) {
            nHead = 0;
        } else {
            nHead = blockK - copyStartRowCntInBaseN;
        }
        midNdNum = (kActCopyRowCount - nHead) / blockK;
        nTail = (kActCopyRowCount - nHead) % blockK;
        copyStartRowCntInBaseNTail = 0;
    }

    Nd2NzParams mm1Nd2NzParamsForB;
    mm1Nd2NzParamsForB.dValue = headDimV;
    mm1Nd2NzParamsForB.srcDValue = step; // 同一个ND矩阵中相邻行起始地址之间的偏移, 单位为元素个数
    mm1Nd2NzParamsForB.dstNzC0Stride = blockK;
    mm1Nd2NzParamsForB.dstNzNStride = 1;
    if (nHead != 0) {
        mm1Nd2NzParamsForB.ndNum = 1;
        mm1Nd2NzParamsForB.nValue = nHead; // 单个ND矩阵的行数, 单位为元素个数
        mm1Nd2NzParamsForB.srcNdMatrixStride = 0; // 相邻ND矩阵起始地址之间的偏移, 单位为元素个数
        mm1Nd2NzParamsForB.dstNzMatrixStride = 0; // 两个NZ矩阵，起始地址之间的偏移, 单位为元素个数

        uint32_t ndNumFinish = copyStartRowCnt / blockK;
        DataCopy(bL1Tensor[ndNumFinish * blockK * headDimVAlign + copyStartRowCntInBaseN * (32 / sizeof(KV_T))], valueGm[valueGmBaseOffset],
                mm1Nd2NzParamsForB);
    }

    if (midNdNum != 0) {
        int32_t ndNumFinish = (copyStartRowCnt + nHead) / blockK;
        if ((LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND || LAYOUT_T == LAYOUT::TND) && (blockK * step > 65535)) {
            mm1Nd2NzParamsForB.ndNum = 1;
            mm1Nd2NzParamsForB.nValue = blockK; // 单个ND矩阵的行数, 单位为元素个数
            mm1Nd2NzParamsForB.srcNdMatrixStride = 0; // 相邻ND矩阵起始地址之间的偏移, 单位为元素个数
            mm1Nd2NzParamsForB.dstNzMatrixStride = 0; // 两个NZ矩阵，起始地址之间的偏移, 单位为元素个数

            for (uint32_t i = 0; i < midNdNum; i++) {
                DataCopy(bL1Tensor[(ndNumFinish + i) * blockK * headDimVAlign], valueGm[valueGmBaseOffset + nHead * step + i * blockK * step], mm1Nd2NzParamsForB);
            }
        } else {
            mm1Nd2NzParamsForB.ndNum = midNdNum;
            mm1Nd2NzParamsForB.nValue = blockK; // 单个ND矩阵的行数, 单位为元素个数
            mm1Nd2NzParamsForB.srcNdMatrixStride = blockK * step; // 相邻ND矩阵起始地址之间的偏移, 单位为元素个数
            mm1Nd2NzParamsForB.dstNzMatrixStride = blockK * headDimVAlign; // 两个NZ矩阵，起始地址之间的偏移, 单位为元素个数

            DataCopy(bL1Tensor[ndNumFinish * blockK * headDimVAlign], valueGm[valueGmBaseOffset + nHead * step], mm1Nd2NzParamsForB);
        }
    }

    if (nTail != 0) {
        mm1Nd2NzParamsForB.ndNum = 1;
        mm1Nd2NzParamsForB.nValue = nTail; // 单个ND矩阵的行数, 单位为元素个数
        mm1Nd2NzParamsForB.srcNdMatrixStride = 0; // 相邻ND矩阵起始地址之间的偏移, 单位为元素个数
        mm1Nd2NzParamsForB.dstNzMatrixStride = 0; // 两个NZ矩阵，起始地址之间的偏移, 单位为元素个数

        int32_t ndNumFinish = (copyStartRowCnt + nHead) / blockK + midNdNum;
        DataCopy(bL1Tensor[ndNumFinish * blockK * headDimVAlign + copyStartRowCntInBaseNTail * (32 / sizeof(KV_T))], valueGm[valueGmBaseOffset + (nHead + midNdNum * blockK) * step],
                 mm1Nd2NzParamsForB);
    }
}

template <typename NSAT>
__aicore__ inline void NsaSelectAttentionInfer<NSAT>::LoadDataMm2A(LocalTensor<KV_T> aL0Tensor, LocalTensor<KV_T> aL1Tensor, uint32_t kSize) {
    LoadData2DParams loadData2DParams;
    loadData2DParams.startIndex = 0;
    loadData2DParams.repeatTimes = kSize / (32 / sizeof(KV_T));
    loadData2DParams.srcStride = 1;
    loadData2DParams.dstGap = 0;
    loadData2DParams.ifTranspose = false;
    LoadData(aL0Tensor, aL1Tensor, loadData2DParams);
}

template <typename NSAT>
__aicore__ inline void NsaSelectAttentionInfer<NSAT>::ComputeMm2(uint32_t loop) {
    ExtraInfo& info = extraInfo[loop % (PRE_LOAD_NUM)];
    gSize = gSizeCube;
    constexpr uint32_t kCopyRowCount = 256;
    uint32_t kCopyTimes = (info.actualSingleProcessSInnerSize + kCopyRowCount - 1) / kCopyRowCount;
    uint32_t kTailCopyRowCount = info.actualSingleProcessSInnerSize - (kCopyTimes - 1) * kCopyRowCount;
    uint32_t kTailCopyRowCountAlign = info.actualSingleProcessSInnerSizeAlign - (kCopyTimes - 1) * kCopyRowCount;
    LocalTensor<MM_OUT_T> cL0Tensor = cL0TensorPingPong[(cL0BufIter % 2) * (L0C_PP_SIZE / sizeof(MM_OUT_T))];
    WaitFlag<HardEvent::FIX_M>(L0C_EVENT0 + cL0BufIter % 2);
    if (info.bn2IdxInCurCore != info.bn2IdxInPreCore && info.isFirstSInnerLoop) {
        xMM2 = 0;
        currentTopDealMM2 = -1;
        topKIdOffsetMM2 = 0;
        idIntopKRecordMM2 = -1;
    }
    for (uint32_t kCopyIdx = 0, kActCopyRowCount = kCopyRowCount, kActCopyRowCountAlign = kCopyRowCount; kCopyIdx < kCopyTimes; kCopyIdx++) {
        if (kCopyIdx + 1 == kCopyTimes) {
            kActCopyRowCount = kTailCopyRowCount;
            kActCopyRowCountAlign = kTailCopyRowCountAlign;
        }
        LocalTensor<KV_T> aL1Tensor = kpL1Tensor[(kpL1BufIter % 2) * L1_KP_SIZE / sizeof(KV_T)];
        WaitFlag<HardEvent::MTE1_MTE2>(KP_EVENT0 + (kpL1BufIter % 2));
        CopyInMm2AToL1(aL1Tensor, info, kCopyIdx, kCopyRowCount, kActCopyRowCountAlign, kActCopyRowCount);
        SetFlag<HardEvent::MTE2_MTE1>(KP_EVENT0 + kpL1BufIter % 2);
        LocalTensor<KV_T> bL1Tensor = vL1Tensor[(vL1BufIter % 2) * (L1_V_SIZE / sizeof(KV_T))];
        WaitFlag<HardEvent::MTE1_MTE2>(V_EVENT0 + (vL1BufIter % 2));
        int64_t topKBaseOffsetmm2 = 0;
        if constexpr (LAYOUT_T == LAYOUT::TND) {
            uint32_t curTotalQSeqLenOffset = 0;
            for (uint64_t i = 0; i < info.bIdx; ++i) {
                curTotalQSeqLenOffset += actualQSeqLengthsGm.GetValue(i);
            }
            topKBaseOffsetmm2 = curTotalQSeqLenOffset * kvHeadNum * selectedBlockCount + info.s1Idx * kvHeadNum * selectedBlockCount;
        } else {
            topKBaseOffsetmm2 = info.bIdx * qSeqSize * kvHeadNum * selectedBlockCount + info.s1Idx * kvHeadNum * selectedBlockCount;
        }
        int64_t blockTableBaseOffset = info.bIdx * maxBlockNumPerBatch;
        int64_t curSeqIdx = info.s2BatchOffset + kCopyIdx * kCopyRowCount;
        int64_t copyStartRowCnt = 0;
        uint64_t curActualSeqLenCurBatch = actualSeqLengthsGm.GetValue(info.bIdx);
        uint32_t curBatchQseqlen = actualQSeqLengthsGm.GetValue(info.bIdx);
        if (info.s1Idx +1 != curBatchQseqlen) {
            curActualSeqLenCurBatch -= curBatchQseqlen - 1 - info.s1Idx;
        }
        while (copyStartRowCnt < kActCopyRowCount) {
            if (xMM2 == currentTopDealMM2) {
                topKIdOffsetMM2 += 1;
            }
            int64_t idIntopK = topKGm.GetValue(topKBaseOffsetmm2 + topKIdOffsetMM2);
            if (idIntopK != -1) {
                if(idIntopKRecordMM2 != idIntopK) {
                     xMM2 = 0;
                    idIntopKRecordMM2 = idIntopK;
                    int64_t globalStart = idIntopK * selectedBlockSize;
                    int64_t globalEnd = (globalStart + selectedBlockSize > curActualSeqLenCurBatch ) ?
                                         curActualSeqLenCurBatch : (globalStart + selectedBlockSize);
                    currentTopDealMM2 = globalEnd - globalStart;
                }
                if (currentTopDealMM2 > 0) {
                    int64_t globalStart = idIntopK * selectedBlockSize;
                    int64_t globalEnd = (globalStart + selectedBlockSize > curActualSeqLenCurBatch ) ?
                                            curActualSeqLenCurBatch : (globalStart + selectedBlockSize);
                    globalStart += xMM2;
                    int64_t start_offset = globalStart % kvCacheBlockSize;
                    int64_t start_block_idx = globalStart / kvCacheBlockSize;
                    int64_t end_block_idx = (globalEnd - 1) / kvCacheBlockSize;
                    int64_t idInBlockTable =
                        blockTableGm.GetValue(blockTableBaseOffset + start_block_idx);

                    int64_t reaminRowCnt = start_offset;
                    int64_t copyRowCnt = start_block_idx == end_block_idx ?
                                        globalEnd - globalStart : kvCacheBlockSize - reaminRowCnt;
                    if (copyStartRowCnt + copyRowCnt > kActCopyRowCount) {
                        copyRowCnt = kActCopyRowCount - copyStartRowCnt;
                    }
                    int64_t valueOffset = idInBlockTable * kvCacheBlockSize * headDimV * kvHeadNum;
                    valueOffset += (int64_t)(info.n2Idx * headDimV) + reaminRowCnt * headDimV * kvHeadNum;
                    CopyInMm2BToL1ForPA(bL1Tensor, valueOffset, kActCopyRowCount, copyStartRowCnt, copyRowCnt);
                    copyStartRowCnt += copyRowCnt;
                    curSeqIdx += copyRowCnt;
                    xMM2 += copyRowCnt;
                } else {
                    xMM2 = currentTopDealMM2;
                }
            }
        }

        SetFlag<HardEvent::MTE2_MTE1>(V_EVENT0 + (vL1BufIter % 2));
        WaitFlag<HardEvent::MTE2_MTE1>(V_EVENT0 + (vL1BufIter % 2));

        WaitFlag<HardEvent::MTE2_MTE1>(KP_EVENT0 + kpL1BufIter % 2);
        {
            constexpr uint32_t baseK = 256 / sizeof(KV_T);
            uint32_t kLoopTimes = (kActCopyRowCountAlign + baseK - 1) / baseK;
            uint32_t kTailAlign = kActCopyRowCountAlign - (kLoopTimes - 1) * baseK;
            uint32_t kTail = kActCopyRowCount - (kLoopTimes - 1) * baseK;
            for (uint32_t i = 0, actualBaseKAlign = baseK, actualBaseK = baseK; i < kLoopTimes; i++) {
                if (i + 1 == kLoopTimes) {
                    actualBaseKAlign = kTailAlign;
                    actualBaseK = kTail;
                }
                WaitFlag<HardEvent::M_MTE1>(L0A_EVENT0 + aL0BufIter % 2);
                LocalTensor<KV_T> aL0Tensor = aL0TensorPingPong[(aL0BufIter % 2) * (L0A_PP_SIZE / sizeof(KV_T))];
                LocalTensor<KV_T> curAL1Tensor = aL1Tensor[16 * baseK * i];
                uint32_t mmRowCount = msdIterNum * gSize;
                uint32_t copyStrideL0 = 16 * actualBaseKAlign;
                uint32_t copyStrideL1 = 16 * kActCopyRowCountAlign;
                uint32_t copyIterNum = (mmRowCount + 15) / 16;
                for(int k = 0; k < copyIterNum; k++){
                    LoadDataMm2A(aL0Tensor[k * copyStrideL0], curAL1Tensor[k * copyStrideL1], actualBaseKAlign);
                }
                SetFlag<HardEvent::MTE1_M>(L0A_EVENT0 + aL0BufIter % 2);
                WaitFlag<HardEvent::MTE1_M>(L0A_EVENT0 + aL0BufIter % 2);

                WaitFlag<HardEvent::M_MTE1>(L0B_EVENT0 + bL0BufIter % 2);
                LocalTensor<KV_T> bL0Tensor = bL0TensorPingPong[bL0BufIter % 2 * L0B_PP_SIZE / sizeof(KV_T)];
                uint32_t blockElementCnt = 32 / sizeof(KV_T);
                LoadData2dTransposeParams loadData2DTransposeParamsForB;
                loadData2DTransposeParamsForB.startIndex = 0;
                loadData2DTransposeParamsForB.srcStride = 1;
                loadData2DTransposeParamsForB.dstFracGap = 0;
                loadData2DTransposeParamsForB.repeatTimes = (actualBaseKAlign / blockElementCnt) * (headDimVAlign / blockElementCnt);
                loadData2DTransposeParamsForB.dstGap = blockElementCnt / 16 - 1;
                uint32_t l1BaseOffset = baseK * headDimVAlign * i;
                LoadDataWithTranspose(bL0Tensor, bL1Tensor[l1BaseOffset], loadData2DTransposeParamsForB);
                SetFlag<HardEvent::MTE1_M>(L0B_EVENT0 + bL0BufIter % 2);
                WaitFlag<HardEvent::MTE1_M>(L0B_EVENT0 + bL0BufIter % 2);
                MmadParams mmadParams;
                mmadParams.m = msdIterNum * gSize;
                if (mmadParams.m == 1) { //m等于1会默认开GEMV模式，且不可关闭GEMV，所以规避当作矩阵计算
                    mmadParams.m = 16;
                }
                mmadParams.n = 128;
                mmadParams.k = actualBaseK; // 无效数据不参与计算
                mmadParams.cmatrixInitVal = (kCopyIdx == 0) && (i == 0);
                mmadParams.cmatrixSource = false;
                Mmad(cL0Tensor, aL0Tensor, bL0Tensor, mmadParams);
                PipeBarrier<PIPE_M>();
                SetFlag<HardEvent::M_MTE1>(L0A_EVENT0 + aL0BufIter % 2);
                aL0BufIter++;
                SetFlag<HardEvent::M_MTE1>(L0B_EVENT0 + bL0BufIter % 2);
                bL0BufIter++;
            }
        }
        SetFlag<HardEvent::MTE1_MTE2>(KP_EVENT0 + kpL1BufIter % 2);
        kpL1BufIter++;
        SetFlag<HardEvent::MTE1_MTE2>(V_EVENT0 + (vL1BufIter % 2));
        vL1BufIter++;
    }
    SetFlag<HardEvent::M_FIX>(L0C_EVENT0 + cL0BufIter % 2);
    WaitFlag<HardEvent::M_FIX>(L0C_EVENT0 + cL0BufIter % 2);
    FixpipeParamsV220 fixParams;
    fixParams.nSize = 128;
    fixParams.mSize = msdIterNum * gSize; // 有效数据不足16行，只需要输出部分行即可
    fixParams.srcStride = ((fixParams.mSize + 15) / 16) * 16;
    fixParams.dstStride = 128;
    fixParams.ndNum = 1;
    Fixpipe(mm2ResGm[(loop % (PRE_LOAD_NUM)) * bmm2ResUbSize], cL0Tensor, fixParams);
    SetFlag<HardEvent::FIX_M>(L0C_EVENT0 + cL0BufIter % 2);
    cL0BufIter++;
}

template <typename NSAT>
__aicore__ inline bool NsaSelectAttentionInfer<NSAT>::IsFinish(uint32_t loop) {
  return (loop >= bn2s2LoopTimes);
}

template <typename NSAT>
__aicore__ inline void NsaSelectAttentionInfer<NSAT>::Process()
{
    if (aiCoreIdx < usedCoreNum) {
        SetLoopTimes();
        if ASCEND_IS_AIV {
            Duplicate(softmaxMaxDefaultUb, SOFTMAX_MIN_NUM, gSize * 32 / sizeof(T));
            Duplicate(softmaxSumDefaultUb, FLOAT_ZERO, gSize * 32 / sizeof(T));
        } else {
            // AllocEventID
            SetFlag<HardEvent::MTE1_MTE2>(KP_EVENT0);
            SetFlag<HardEvent::MTE1_MTE2>(KP_EVENT1);
            SetFlag<HardEvent::MTE1_MTE2>(V_EVENT0);
            SetFlag<HardEvent::MTE1_MTE2>(Q_EVENT1);

            SetFlag<HardEvent::M_MTE1>(L0A_EVENT0);
            SetFlag<HardEvent::M_MTE1>(L0A_EVENT1);
            SetFlag<HardEvent::M_MTE1>(L0B_EVENT0);
            SetFlag<HardEvent::M_MTE1>(L0B_EVENT1);

            SetFlag<HardEvent::FIX_M>(L0C_EVENT0);
            SetFlag<HardEvent::FIX_M>(L0C_EVENT1);
        }

        for (uint32_t i = 0; i < bn2s2LoopTimes; i = i + PRE_LOAD_NUM) {
            for (uint32_t j = 0; j < PRE_LOAD_NUM; j++) {
                uint32_t loop = i + j;
                if (i != 0) {
                    if (!IsFinish(loop - PRE_LOAD_NUM)) {
                        if ASCEND_IS_AIV {
                            CrossCoreWaitFlag(SYNC_C2_V2_FLAG);
                            ProcessVec2L(loop - PRE_LOAD_NUM);
                        }
                    }
                }

                if (!IsFinish(loop)) {
                    CalcParams(loop);
                    if ASCEND_IS_AIC {
                        ComputeMm1(loop);
                        CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_C1_V1_FLAG);
                    }
                }
            }

            for (uint32_t j = 0; j < PRE_LOAD_NUM; j++) {
                uint32_t loop = i + j;
                if (!IsFinish(loop)) {
                    if ASCEND_IS_AIV {
                        CrossCoreWaitFlag(SYNC_C1_V1_FLAG);
                        ProcessVec1L(loop);
                        CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(SYNC_V1_C2_FLAG);
                    }
                    if ASCEND_IS_AIC {
                        CrossCoreWaitFlag(SYNC_V1_C2_FLAG);
                        ComputeMm2(loop);
                        CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_C2_V2_FLAG);
                    }
                }
            }

            if (i + PRE_LOAD_NUM >= bn2s2LoopTimes) {
                for (uint32_t j = 0; j < PRE_LOAD_NUM; j++) {
                    uint32_t loop = i + j;
                    if (!IsFinish(loop)) {
                        if ASCEND_IS_AIV {
                            CrossCoreWaitFlag(SYNC_C2_V2_FLAG);
                            ProcessVec2L(loop);
                        }
                    }
                }
            }
        }

        if ASCEND_IS_AIC {
            // FreeEventID
            WaitFlag<HardEvent::MTE1_MTE2>(KP_EVENT0);
            WaitFlag<HardEvent::MTE1_MTE2>(KP_EVENT1);
            WaitFlag<HardEvent::MTE1_MTE2>(V_EVENT0);
            WaitFlag<HardEvent::MTE1_MTE2>(Q_EVENT1);

            WaitFlag<HardEvent::M_MTE1>(L0A_EVENT0);
            WaitFlag<HardEvent::M_MTE1>(L0A_EVENT1);
            WaitFlag<HardEvent::M_MTE1>(L0B_EVENT0);
            WaitFlag<HardEvent::M_MTE1>(L0B_EVENT1);

            WaitFlag<HardEvent::FIX_M>(L0C_EVENT0);
            WaitFlag<HardEvent::FIX_M>(L0C_EVENT1);
        }
    }
}