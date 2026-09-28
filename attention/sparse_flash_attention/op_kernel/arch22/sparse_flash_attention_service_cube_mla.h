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
 * \file sparse_flash_attention_service_cube_mla.h
 * \brief use 7 buffer for matmul l1, better pipeline
 */
#ifndef SPARSE_FLASH_ATTENTION_SERVICE_CUBE_MLA_H
#define SPARSE_FLASH_ATTENTION_SERVICE_CUBE_MLA_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "sparse_flash_attention_common.h"

struct PAShape {
    uint32_t blockSize;
    uint32_t headNum;             // 一般为kv的head num，对应n2
    uint32_t headDim;             // mla下rope为64，nope为512, 对应d
    uint32_t maxblockNumPerBatch; // block table 每一行的最大个数
    uint32_t actHeadDim;          // 实际拷贝col大小,考虑到N切块   s*d, 对应d
    uint32_t copyRowNum;          // 总共要拷贝的行数
    uint32_t copyRowNumAlign;
};

struct Position {
    uint32_t bIdx;
    uint32_t n2Idx;
    uint32_t s2Idx;
    uint32_t dIdx;
};

// 场景：query、queryRope、key、value GM to L1
// GM按ND格式存储
// L1按NZ格式存储
// GM的行、列、列的stride
template <typename T>
__aicore__ inline void DataCopyGmNDToL1(LocalTensor<T> &l1Tensor, GlobalTensor<T> &gmTensor, uint32_t rowAct,
                                        uint32_t rowAlign,
                                        uint32_t col,       // D
                                        uint32_t colStride) // D or N*D
{
    Nd2NzParams sfaCubeCopyParams;
    sfaCubeCopyParams.ndNum = 1;
    sfaCubeCopyParams.nValue = rowAct; // nd矩阵的行数
    // T为int4场景下，dValue = col / 2，srcDValue = colStride / 2
    sfaCubeCopyParams.dValue = col;          // nd矩阵的列数
    sfaCubeCopyParams.srcDValue = colStride; // 同一nd矩阵相邻行起始地址间的偏移
    sfaCubeCopyParams.dstNzC0Stride = rowAlign;
    sfaCubeCopyParams.dstNzNStride = 1;
    sfaCubeCopyParams.srcNdMatrixStride = 0;
    sfaCubeCopyParams.dstNzMatrixStride = 0;
    DataCopy(l1Tensor, gmTensor, sfaCubeCopyParams);
}

/*
    适用PA数据从GM拷贝到L1，支持ND、NZ数据；
    PA的layout分 BNBD（blockNum,N,blockSize,D） BBH（blockNum,blockSize,N*D
    BSH\BSND\TND 为BBH
    shape.copyRowNumAlign 需要16字节对齐，如拷贝k矩阵，一次拷贝128*512，遇到尾块 10*512 需对齐到16*512
*/
template <typename T, SFA_LAYOUT SRC_LAYOUT>
__aicore__ inline void DataCopyPA(LocalTensor<T> &dstTensor,  // l1
                                  GlobalTensor<T> &srcTensor, // gm
                                  GlobalTensor<int32_t> &blockTableGm,
                                  const PAShape &shape,     // blockSize, headNum, headDim
                                  const Position &startPos) // bacthIdx nIdx curSeqIdx
{
    uint64_t blockTableBaseOffset = startPos.bIdx * shape.maxblockNumPerBatch;
    uint32_t curS2Idx = startPos.s2Idx;
    uint32_t copyFinishRowCnt = 0;
    uint32_t blockElementCnt = 32 / sizeof(T);
    while (copyFinishRowCnt < shape.copyRowNum) {
        uint64_t reaminRowCnt = curS2Idx % shape.blockSize;  // 获取在单个块上超出的行数
        uint64_t blockIdOffset = curS2Idx / shape.blockSize; // 获取block table上的索引
        uint64_t idInBlockTable =
            blockTableGm.GetValue(blockTableBaseOffset + blockIdOffset); // 从block table上的获取编号
        // 计算可以拷贝行数
        uint32_t copyRowCnt = shape.blockSize - reaminRowCnt; // 一次只能处理一个Block
        if (copyFinishRowCnt + copyRowCnt > shape.copyRowNum) {
            copyRowCnt = shape.copyRowNum - copyFinishRowCnt; // 一个block未拷满
        }
        uint64_t dStride = shape.headDim;
        uint64_t offset = idInBlockTable * shape.blockSize * shape.headNum * shape.headDim; // PA的偏移

        if constexpr (SRC_LAYOUT == SFA_LAYOUT::BSND || SRC_LAYOUT == SFA_LAYOUT::TND) {
            offset += (uint64_t)(startPos.n2Idx * shape.headDim) + reaminRowCnt * shape.headDim * shape.headNum +
                      startPos.dIdx;
            dStride = shape.headDim * shape.headNum;
        } else {
            offset += (uint64_t)(startPos.n2Idx * shape.headDim * shape.blockSize) + reaminRowCnt * shape.headDim +
                      startPos.dIdx;
        }

        uint32_t dValue = shape.actHeadDim;
        uint32_t srcDValue = dStride;
        LocalTensor<T> tmpDstTensor = dstTensor[copyFinishRowCnt * blockElementCnt];
        GlobalTensor<T> tmpSrcTensor = srcTensor[offset];

        DataCopyGmNDToL1<T>(tmpDstTensor, tmpSrcTensor, copyRowCnt, shape.copyRowNumAlign, dValue, srcDValue);
        copyFinishRowCnt += copyRowCnt;
        curS2Idx += copyRowCnt;
    }
}

template <typename SFAT>
class SFAMatmulService {
public:
    // 中间计算数据类型为float, 高精度模式
    using T = float;
    using Q_T = typename SFAT::queryType;
    using KV_T = typename SFAT::kvType;
    using OUT_T = typename SFAT::outputType;
    using MM_OUT_T = T;

    __aicore__ inline SFAMatmulService(){};
    __aicore__ inline void InitParams(const ConstInfo &sfaCubeConstInfo);
    __aicore__ inline void InitMm1GlobalTensor(GlobalTensor<Q_T> queryGm, GlobalTensor<Q_T> qRopeGm,
                                               GlobalTensor<KV_T> keyGm, GlobalTensor<KV_T> kRopeGm,
                                               GlobalTensor<MM_OUT_T> mm1ResGm);
    __aicore__ inline void InitMm2GlobalTensor(GlobalTensor<KV_T> vec1ResGm, GlobalTensor<KV_T> valueGm,
                                               GlobalTensor<MM_OUT_T> mm2ResGm, GlobalTensor<OUT_T> attentionOutGm);
    __aicore__ inline void InitPageAttentionInfo(const GlobalTensor<KV_T> &kvMergeGm,
                                                 GlobalTensor<int32_t> blockTableGm, GlobalTensor<int32_t> topKGm,
                                                 uint32_t blockSize, uint32_t maxBlockNumPerBatch);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void UpdateKey(GlobalTensor<KV_T> keyGm);
    __aicore__ inline void UpdateValue(GlobalTensor<KV_T> valueGm);

    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void CalcTopKBlockInfo(const RunInfo &sfaCubeRunInfo, uint32_t &curTopKIdx,
                                             uint64_t &curOffsetInSparseBlock, uint32_t curSeqIdx, uint32_t &copyRowCnt,
                                             int64_t &idInTopK);
    __aicore__ inline void ComputeMm1(const RunInfo &sfaCubeRunInfo, const MSplitInfo sfaCubeSplitInfo);
    __aicore__ inline void ComputeMm1NoRope(const RunInfo &sfaCubeRunInfo, const MSplitInfo sfaCubeSplitInfo);
    __aicore__ inline void ComputeMm2(const RunInfo &sfaCubeRunInfo, const MSplitInfo sfaCubeSplitInfo);

private:
    static constexpr bool PAGE_ATTENTION = SFAT::pageAttention;
    static constexpr int TEMPLATE_MODE = SFAT::templateMode;
    static constexpr bool FLASH_DECODE = SFAT::flashDecode;
    static constexpr SFA_LAYOUT LAYOUT_T = SFAT::layout;
    static constexpr SFA_LAYOUT KV_LAYOUT_T = SFAT::kvLayout;

    static constexpr uint32_t M_SPLIT_SIZE = 128;     // m方向切分
    static constexpr uint32_t N_SPLIT_SIZE = 128;     // n方向切分
    static constexpr uint32_t N_WORKSPACE_SIZE = 512; // n方向切分

    static constexpr uint32_t L1_BLOCK_SIZE = (64 * (512 + 64) * sizeof(Q_T));
    static constexpr uint32_t L1_BLOCK_OFFSET = 64 * (512 + 64); // 72K的元素个数

    static constexpr uint32_t L0A_PP_SIZE = (32 * 1024);
    static constexpr uint32_t L0B_PP_SIZE = (32 * 1024);
    static constexpr uint32_t L0C_PP_SIZE = (64 * 1024);

    // mte2 <> mte1 EventID
    // L1 3buf, 使用3个eventId
    static constexpr uint32_t L1_EVENT0 = EVENT_ID2;
    static constexpr uint32_t L1_EVENT1 = EVENT_ID3;
    static constexpr uint32_t L1_EVENT2 = EVENT_ID4;
    static constexpr uint32_t L1_EVENT3 = EVENT_ID5;
    static constexpr uint32_t L1_EVENT4 = EVENT_ID6;
    static constexpr uint32_t L1_EVENT5 = EVENT_ID7;
    static constexpr uint32_t L1_EVENT6 = EVENT_ID1;

    // m <> mte1 EventID
    static constexpr uint32_t L0AB_EVENT0 = EVENT_ID3;
    static constexpr uint32_t L0AB_EVENT1 = EVENT_ID4;

    static constexpr IsResetLoad3dConfig LOAD3DV2_CONFIG = {true, true}; // isSetFMatrix isSetPadding;
    static constexpr uint32_t mte21QPIds[4] = {L1_EVENT0, L1_EVENT1, L1_EVENT2, L1_EVENT3}; // mte12复用
    static constexpr uint32_t sfaMte21KvEvents[3] = {L1_EVENT4, L1_EVENT5, L1_EVENT6};

    uint32_t kvCacheBlockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    ConstInfo sfaCubeConstInfo{};

    // L1分成3块buf, 用于记录
    uint32_t abL0BufIter = 0;
    uint32_t sfaL0CBufferIndex = 0;
    uint32_t qpL1BufIter = 0;
    uint32_t kvL1BufIter = -1;

    // mm1
    GlobalTensor<Q_T> queryGm;
    GlobalTensor<Q_T> qRopeGm;
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<KV_T> kRopeGm;
    GlobalTensor<MM_OUT_T> mm1ResGm;
    GlobalTensor<KV_T> kvMergeGm_;

    // mm2
    GlobalTensor<KV_T> vec1ResGm;
    GlobalTensor<KV_T> valueGm;
    GlobalTensor<MM_OUT_T> mm2ResGm;
    GlobalTensor<OUT_T> attentionOutGm;

    // block_table
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<int32_t> topKGm;

    TBuf<TPosition::A1> bufQPL1;
    TBuf<TPosition::A1> bufKVL1;
    TBuf<TPosition::A2> tmpBufL0A;
    TBuf<TPosition::B2> tmpBufL0B;
    TBuf<TPosition::CO1> tmpBufL0C;

    LocalTensor<Q_T> l1QPTensor;
    LocalTensor<Q_T> l1KVTensor;
    LocalTensor<KV_T> aL0TensorPingPong;
    LocalTensor<KV_T> bL0TensorPingPong;
    LocalTensor<MM_OUT_T> cL0TensorPingPong;

    // L0AB m <> mte1 EventID
    __aicore__ inline uint32_t Mte1MmABEventId(uint32_t idx)
    {
        return (L0AB_EVENT0 + idx);
    }

    __aicore__ inline uint32_t GetQPL1RealIdx(uint32_t mIdx, uint32_t k1Idx)
    {
        uint32_t idxMap[] = {0, 2}; // 确保0块和1块连在一起, 2和3块连在一起, 来保证同一m块的地址相连
        return idxMap[mIdx % 2] + k1Idx;
    }

    __aicore__ inline void CopyGmToL1(LocalTensor<KV_T> &l1Tensor, GlobalTensor<KV_T> &gmSrcTensor, uint32_t srcN,
                                      uint32_t srcD, uint32_t srcDstride);
    __aicore__ inline void CopyInMm1AToL1(LocalTensor<KV_T> &aL1Tensor, const RunInfo &sfaCubeRunInfo, uint32_t mSeqIdx,
                                          uint32_t mSizeAct, uint32_t headSize, uint32_t headOffset);
    __aicore__ inline void CopyInMm1ARopeToL1(LocalTensor<KV_T> &aL1Tensor, const RunInfo &sfaCubeRunInfo,
                                              uint32_t mSeqIdx, uint32_t mSizeAct);
    __aicore__ inline void CopyInMm1BToL1(LocalTensor<KV_T> &bL1Tensor, const uint64_t keyGmBaseOffset,
                                          uint32_t copyTotalRowCntAlign, uint32_t copyStartRowCnt,
                                          uint32_t nActCopyRowCount, uint32_t headSize);
    __aicore__ inline void CopyInMm1BRopeToL1(LocalTensor<KV_T> &bL1Tensor, const uint64_t keyGmBaseOffset,
                                              uint32_t copyTotalRowCntAlign, uint32_t copyStartRowCnt,
                                              uint32_t nActCopyRowCount, uint32_t headSize);
    __aicore__ inline void CopyInMm2AToL1(LocalTensor<KV_T> &aL1Tensor, const RunInfo &sfaCubeRunInfo, uint32_t mSeqIdx,
                                          uint32_t subMSizeAct, uint32_t nSize, uint32_t nOffset);
    __aicore__ inline void CopyInMm2BToL1(LocalTensor<KV_T> &bL1Tensor, const uint64_t valueGmBaseOffset,
                                          uint32_t copyTotalRowCntAlign, uint32_t copyStartRowCnt,
                                          uint32_t nActCopyRowCount, uint32_t copyStartColumnCount,
                                          uint32_t copyColumnCount);
    __aicore__ inline void LoadDataMm1A(LocalTensor<KV_T> &aL0Tensor, LocalTensor<KV_T> &aL1Tensor, uint32_t idx,
                                        uint32_t kSplitSize, uint32_t mSize, uint32_t kSize);
    __aicore__ inline void LoadDataMm1B(LocalTensor<KV_T> &bL0Tensor, LocalTensor<KV_T> &bL1Tensor, uint32_t idx,
                                        uint32_t kSplitSize, uint32_t kSize, uint32_t nSize);
};

template <typename SFAT>
__aicore__ inline void SFAMatmulService<SFAT>::InitParams(const ConstInfo &sfaCubeConstInfo)
{
    this->sfaCubeConstInfo = sfaCubeConstInfo;
}

template <typename SFAT>
__aicore__ inline void SFAMatmulService<SFAT>::InitMm1GlobalTensor(GlobalTensor<Q_T> queryGm, GlobalTensor<Q_T> qRopeGm,
                                                                   GlobalTensor<KV_T> keyGm, GlobalTensor<KV_T> kRopeGm,
                                                                   GlobalTensor<MM_OUT_T> mm1ResGm)
{
    // mm1
    this->queryGm = queryGm;
    this->qRopeGm = qRopeGm;
    this->keyGm = keyGm;
    this->kRopeGm = kRopeGm;
    this->mm1ResGm = mm1ResGm;
}

template <typename SFAT>
__aicore__ inline void SFAMatmulService<SFAT>::InitMm2GlobalTensor(GlobalTensor<KV_T> vec1ResGm,
                                                                   GlobalTensor<KV_T> valueGm,
                                                                   GlobalTensor<MM_OUT_T> mm2ResGm,
                                                                   GlobalTensor<OUT_T> attentionOutGm)
{
    // mm2
    this->vec1ResGm = vec1ResGm;
    this->valueGm = valueGm;
    this->mm2ResGm = mm2ResGm;
    this->attentionOutGm = attentionOutGm;
}

template <typename SFAT>
__aicore__ inline void SFAMatmulService<SFAT>::InitPageAttentionInfo(const GlobalTensor<KV_T> &kvMergeGm,
                                                                     GlobalTensor<int32_t> blockTableGm,
                                                                     GlobalTensor<int32_t> topKGm, uint32_t blockSize,
                                                                     uint32_t maxBlockNumPerBatch)
{
    this->blockTableGm = blockTableGm;
    this->topKGm = topKGm;
    this->kvCacheBlockSize = blockSize;
    this->maxBlockNumPerBatch = maxBlockNumPerBatch;
    this->kvMergeGm_ = kvMergeGm;
}

template <typename SFAT>
__aicore__ inline void SFAMatmulService<SFAT>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(bufQPL1, L1_BLOCK_SIZE * 4); // (64K + 8K) * 4
    l1QPTensor = bufQPL1.Get<Q_T>();
    pipe->InitBuffer(bufKVL1, L1_BLOCK_SIZE * 3); // (64K + 8K) * 3
    l1KVTensor = bufKVL1.Get<KV_T>();

    // L0A
    pipe->InitBuffer(tmpBufL0A, L0A_PP_SIZE * 2); // 64K
    aL0TensorPingPong = tmpBufL0A.Get<KV_T>();
    // L0B
    pipe->InitBuffer(tmpBufL0B, L0B_PP_SIZE * 2); // 64K
    bL0TensorPingPong = tmpBufL0B.Get<KV_T>();
    // L0C
    pipe->InitBuffer(tmpBufL0C, L0C_PP_SIZE * 2); // 128K
    cL0TensorPingPong = tmpBufL0C.Get<MM_OUT_T>();
}

template <typename SFAT>
__aicore__ inline void SFAMatmulService<SFAT>::UpdateKey(GlobalTensor<KV_T> keyGm)
{
    this->keyGm = keyGm;
}

template <typename SFAT>
__aicore__ inline void SFAMatmulService<SFAT>::UpdateValue(GlobalTensor<KV_T> valueGm)
{
    this->valueGm = valueGm;
}

template <typename SFAT>
__aicore__ inline void SFAMatmulService<SFAT>::AllocEventID()
{
    SetFlag<HardEvent::MTE1_MTE2>(L1_EVENT0);
    SetFlag<HardEvent::MTE1_MTE2>(L1_EVENT1);
    SetFlag<HardEvent::MTE1_MTE2>(L1_EVENT2);
    SetFlag<HardEvent::MTE1_MTE2>(L1_EVENT3);
    SetFlag<HardEvent::MTE1_MTE2>(L1_EVENT4);
    SetFlag<HardEvent::MTE1_MTE2>(L1_EVENT5);
    SetFlag<HardEvent::MTE1_MTE2>(L1_EVENT6);
    SetFlag<HardEvent::M_MTE1>(L0AB_EVENT0);
    SetFlag<HardEvent::M_MTE1>(L0AB_EVENT1);
}

template <typename SFAT>
__aicore__ inline void SFAMatmulService<SFAT>::FreeEventID()
{
    WaitFlag<HardEvent::MTE1_MTE2>(L1_EVENT0);
    WaitFlag<HardEvent::MTE1_MTE2>(L1_EVENT1);
    WaitFlag<HardEvent::MTE1_MTE2>(L1_EVENT2);
    WaitFlag<HardEvent::MTE1_MTE2>(L1_EVENT3);
    WaitFlag<HardEvent::MTE1_MTE2>(L1_EVENT4);
    WaitFlag<HardEvent::MTE1_MTE2>(L1_EVENT5);
    WaitFlag<HardEvent::MTE1_MTE2>(L1_EVENT6);
    WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT0);
    WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT1);
}

template <typename SFAT>
__aicore__ inline void SFAMatmulService<SFAT>::CopyGmToL1(LocalTensor<KV_T> &l1Tensor, GlobalTensor<KV_T> &gmSrcTensor,
                                                          uint32_t srcN, uint32_t srcD, uint32_t srcDstride)
{
    Nd2NzParams sfaCubeCopyParams;
    sfaCubeCopyParams.ndNum = 1;
    sfaCubeCopyParams.nValue = srcN; // 行数
    sfaCubeCopyParams.dValue = srcD;
    sfaCubeCopyParams.srcDValue = srcDstride;
    sfaCubeCopyParams.dstNzC0Stride = (srcN + 15) / 16 * 16; // 对齐到16 单位block
    sfaCubeCopyParams.dstNzNStride = 1;
    sfaCubeCopyParams.srcNdMatrixStride = 0;
    sfaCubeCopyParams.dstNzMatrixStride = 0;
    DataCopy(l1Tensor, gmSrcTensor, sfaCubeCopyParams);
}

template <typename SFAT>
__aicore__ inline void SFAMatmulService<SFAT>::CopyInMm1AToL1(LocalTensor<KV_T> &l1Tensor,
                                                              const RunInfo &sfaCubeRunInfo, uint32_t mSeqIdx,
                                                              uint32_t mSizeAct, uint32_t headSize, uint32_t headOffset)
{
    auto srcGm = queryGm[sfaCubeRunInfo.tensorAOffset + mSeqIdx * sfaCubeConstInfo.headDim + headOffset];
    CopyGmToL1(l1Tensor, srcGm, mSizeAct, headSize, sfaCubeConstInfo.headDim);
}

template <typename SFAT>
__aicore__ inline void SFAMatmulService<SFAT>::CopyInMm1ARopeToL1(LocalTensor<KV_T> &l1Tensor,
                                                                  const RunInfo &sfaCubeRunInfo, uint32_t mSeqIdx,
                                                                  uint32_t mSizeAct)
{
    auto srcGm = qRopeGm[sfaCubeRunInfo.tensorARopeOffset + mSeqIdx * sfaCubeConstInfo.headDimRope];
    CopyGmToL1(l1Tensor, srcGm, mSizeAct, sfaCubeConstInfo.headDimRope, sfaCubeConstInfo.headDimRope);
}

template <typename SFAT>
__aicore__ inline void SFAMatmulService<SFAT>::CopyInMm1BToL1(LocalTensor<KV_T> &bL1Tensor,
                                                              const uint64_t keyGmBaseOffset,
                                                              uint32_t copyTotalRowCntAlign, uint32_t copyStartRowCnt,
                                                              uint32_t nActCopyRowCount, uint32_t headSize)
{
    uint64_t dStride = sfaCubeConstInfo.headDim;
    if constexpr (KV_LAYOUT_T == SFA_LAYOUT::BSND || KV_LAYOUT_T == SFA_LAYOUT::TND) {
        dStride = sfaCubeConstInfo.headDim * sfaCubeConstInfo.kvHeadNum;
    }

    uint32_t blockElementCnt = 32 / sizeof(KV_T);

    Nd2NzParams mm1Nd2NzParamsForB;
    mm1Nd2NzParamsForB.nValue = nActCopyRowCount;
    mm1Nd2NzParamsForB.dValue = headSize;
    mm1Nd2NzParamsForB.srcDValue = dStride;
    mm1Nd2NzParamsForB.ndNum = 1;
    mm1Nd2NzParamsForB.dstNzC0Stride = copyTotalRowCntAlign;
    mm1Nd2NzParamsForB.dstNzNStride = 1;
    mm1Nd2NzParamsForB.srcNdMatrixStride = 0;
    mm1Nd2NzParamsForB.dstNzMatrixStride = 0;
    DataCopy(bL1Tensor[copyStartRowCnt * blockElementCnt], keyGm[keyGmBaseOffset], mm1Nd2NzParamsForB);
}

template <typename SFAT>
__aicore__ inline void SFAMatmulService<SFAT>::CopyInMm1BRopeToL1(LocalTensor<KV_T> &bL1Tensor,
                                                                  const uint64_t kRopeGmBaseOffset,
                                                                  uint32_t copyTotalRowCntAlign,
                                                                  uint32_t copyStartRowCnt, uint32_t nActCopyRowCount,
                                                                  uint32_t headSize)
{
    uint64_t dStride = sfaCubeConstInfo.headDimRope;
    if constexpr (KV_LAYOUT_T == SFA_LAYOUT::BSND || KV_LAYOUT_T == SFA_LAYOUT::TND) {
        dStride = sfaCubeConstInfo.headDimRope * sfaCubeConstInfo.kvHeadNum;
    }

    uint32_t blockElementCnt = 32 / sizeof(KV_T);

    Nd2NzParams mm1Nd2NzParamsForB;
    mm1Nd2NzParamsForB.nValue = nActCopyRowCount;
    mm1Nd2NzParamsForB.dValue = headSize;
    mm1Nd2NzParamsForB.ndNum = 1;
    mm1Nd2NzParamsForB.srcDValue = dStride;
    mm1Nd2NzParamsForB.dstNzC0Stride = copyTotalRowCntAlign;
    mm1Nd2NzParamsForB.dstNzNStride = 1;
    mm1Nd2NzParamsForB.srcNdMatrixStride = 0;
    mm1Nd2NzParamsForB.dstNzMatrixStride = 0;
    DataCopy(bL1Tensor[copyStartRowCnt * blockElementCnt], kRopeGm[kRopeGmBaseOffset], mm1Nd2NzParamsForB);
}

template <typename SFAT>
__aicore__ inline void SFAMatmulService<SFAT>::LoadDataMm1A(LocalTensor<KV_T> &aL0Tensor, LocalTensor<KV_T> &aL1Tensor,
                                                            uint32_t idx, uint32_t kSplitSize, uint32_t mSize,
                                                            uint32_t kSize)
{
    LocalTensor<KV_T> srcTensor = aL1Tensor[mSize * kSplitSize * idx];
    LoadData3DParamsV2<KV_T> sfaMm1LoadParams;
    // SetFmatrixParams
    sfaMm1LoadParams.padList[0] = 0;
    sfaMm1LoadParams.padList[1] = 0;
    sfaMm1LoadParams.padList[2] = 0;
    sfaMm1LoadParams.padList[3] = 255; // 尾部数据不影响滑窗的结果
    sfaMm1LoadParams.l1H = mSize / 16; // Hin=M1=8
    sfaMm1LoadParams.l1W = 16;         // Win=M0

    // SetLoadToA0Params
    sfaMm1LoadParams.mExtension = mSize; // M
    sfaMm1LoadParams.kExtension = kSize; // K
    sfaMm1LoadParams.strideW = 1;
    sfaMm1LoadParams.strideH = 1;
    sfaMm1LoadParams.filterW = 1;
    sfaMm1LoadParams.filterSizeW = (1 >> 8) & 255;
    sfaMm1LoadParams.filterH = 1;
    sfaMm1LoadParams.filterSizeH = (1 >> 8) & 255;
    sfaMm1LoadParams.mStartPt = 0;
    sfaMm1LoadParams.kStartPt = 0;
    sfaMm1LoadParams.dilationFilterW = 1;
    sfaMm1LoadParams.dilationFilterH = 1;
    sfaMm1LoadParams.enTranspose = 0;
    sfaMm1LoadParams.fMatrixCtrl = 0;
    sfaMm1LoadParams.channelSize = kSize; // Cin=K
    LoadData<KV_T, LOAD3DV2_CONFIG>(aL0Tensor, srcTensor, sfaMm1LoadParams);
}

template <typename SFAT>
__aicore__ inline void SFAMatmulService<SFAT>::LoadDataMm1B(LocalTensor<KV_T> &l0Tensor, LocalTensor<KV_T> &l1Tensor,
                                                            uint32_t idx, uint32_t kSplitSize, uint32_t kSize,
                                                            uint32_t nSize)
{
    // N 方向全载
    LocalTensor<KV_T> srcTensor = l1Tensor[nSize * kSplitSize * idx];

    LoadData2DParams sfaMm1LoadBParams;
    sfaMm1LoadBParams.startIndex = 0;
    sfaMm1LoadBParams.repeatTimes = (nSize + 15) / 16 * kSize / (32 / sizeof(KV_T));
    sfaMm1LoadBParams.srcStride = 1;
    sfaMm1LoadBParams.dstGap = 0;
    sfaMm1LoadBParams.ifTranspose = false;
    LoadData(l0Tensor, srcTensor, sfaMm1LoadBParams);
}

template <typename SFAT>
__aicore__ inline void SFAMatmulService<SFAT>::CopyInMm2AToL1(LocalTensor<KV_T> &aL1Tensor,
                                                              const RunInfo &sfaCubeRunInfo, uint32_t mSeqIdx,
                                                              uint32_t subMSizeAct, uint32_t nSize, uint32_t nOffset)
{
    auto srcGm = vec1ResGm[(sfaCubeRunInfo.loop % sfaCubeConstInfo.preLoadNum) * sfaCubeConstInfo.mmResUbSize +
                           mSeqIdx * sfaCubeRunInfo.actualSingleProcessSInnerSizeAlign + nOffset];
    CopyGmToL1(aL1Tensor, srcGm, subMSizeAct, nSize, sfaCubeRunInfo.actualSingleProcessSInnerSizeAlign);
}

template <typename SFAT>
__aicore__ inline void SFAMatmulService<SFAT>::CopyInMm2BToL1(LocalTensor<KV_T> &bL1Tensor,
                                                              const uint64_t valueGmBaseOffset,
                                                              uint32_t copyTotalRowCntAlign, uint32_t copyStartRowCnt,
                                                              uint32_t nActCopyRowCount, uint32_t copyStartColumnCount,
                                                              uint32_t copyColumnCount)
{
    uint64_t step = sfaCubeConstInfo.headDim;
    if constexpr (KV_LAYOUT_T == SFA_LAYOUT::BSND || KV_LAYOUT_T == SFA_LAYOUT::TND) {
        step = sfaCubeConstInfo.headDim * sfaCubeConstInfo.kvHeadNum;
    }

    uint32_t blockElementCnt = 32 / sizeof(KV_T);

    Nd2NzParams mm1Nd2NzParamsForB;
    mm1Nd2NzParamsForB.ndNum = 1;
    mm1Nd2NzParamsForB.nValue = nActCopyRowCount;
    mm1Nd2NzParamsForB.dValue = copyColumnCount;
    mm1Nd2NzParamsForB.srcDValue = step;
    mm1Nd2NzParamsForB.dstNzC0Stride = copyTotalRowCntAlign;
    mm1Nd2NzParamsForB.dstNzNStride = 1;
    mm1Nd2NzParamsForB.srcNdMatrixStride = 0;
    mm1Nd2NzParamsForB.dstNzMatrixStride = 0;
    DataCopy(bL1Tensor[copyStartRowCnt * blockElementCnt], valueGm[valueGmBaseOffset + copyStartColumnCount],
             mm1Nd2NzParamsForB);
}

template <typename SFAT>
__aicore__ inline void SFAMatmulService<SFAT>::CalcTopKBlockInfo(const RunInfo &sfaCubeRunInfo, uint32_t &curTopKIdx,
                                                                 uint64_t &curOffsetInSparseBlock, uint32_t curSeqIdx,
                                                                 uint32_t &copyRowCnt, int64_t &idInTopK)
{
    uint64_t blockBegin = idInTopK * sfaCubeConstInfo.sparseBlockSize;
    uint64_t blockEnd = (blockBegin + sfaCubeConstInfo.sparseBlockSize > sfaCubeRunInfo.threshold) ?
                            sfaCubeRunInfo.threshold :
                            blockBegin + sfaCubeConstInfo.sparseBlockSize;
    uint64_t blockLen = blockEnd - blockBegin;
    if (curOffsetInSparseBlock + copyRowCnt < blockLen) {
        curOffsetInSparseBlock += copyRowCnt;
        copyRowCnt = blockLen - curOffsetInSparseBlock;
    } else {
        for (uint64_t topkidx = curTopKIdx + 1; topkidx < sfaCubeConstInfo.sparseBlockCount; topkidx++) {
            int64_t sparseIndices = topKGm.GetValue(sfaCubeRunInfo.topKBaseOffset + topkidx);
            if (sparseIndices == -1) {
                break;
            }

            uint64_t blockBegin = sparseIndices * sfaCubeConstInfo.sparseBlockSize;
            if (blockBegin >= sfaCubeRunInfo.threshold) {
                continue;
            }
            uint64_t blockEnd = (blockBegin + sfaCubeConstInfo.sparseBlockSize > sfaCubeRunInfo.threshold) ?
                                    sfaCubeRunInfo.threshold :
                                    blockBegin + sfaCubeConstInfo.sparseBlockSize;
            uint64_t blockLen = blockEnd - blockBegin;
            curTopKIdx = topkidx;
            idInTopK = sparseIndices;
            curOffsetInSparseBlock = 0;
            copyRowCnt = blockLen;
            break;
        }
    }
}

template <typename SFAT>
__aicore__ inline void SFAMatmulService<SFAT>::ComputeMm1NoRope(const RunInfo &sfaCubeRunInfo,
                                                                const MSplitInfo sfaCubeSplitInfo)
{
    uint32_t mSize = sfaCubeSplitInfo.nBufferDealM;
    uint32_t mm1NrML1Size = M_SPLIT_SIZE;
    uint32_t mm1NrML1SizeAlign = SFAAlign(M_SPLIT_SIZE, 16U);
    uint32_t mm1NrML1Loops = (mSize + M_SPLIT_SIZE - 1) / M_SPLIT_SIZE;

    uint32_t nSize = sfaCubeRunInfo.actualSingleProcessSInnerSize;
    uint32_t mm1NrNL1Size = N_SPLIT_SIZE;
    uint32_t mm1NrNL1SizeAlign = SFAAlign(N_SPLIT_SIZE, 16U);
    uint32_t mm1NrNL1Loops = (nSize + N_SPLIT_SIZE - 1) / N_SPLIT_SIZE;

    constexpr uint32_t mm1NrKL1Loops = 2;
    constexpr uint32_t mm1NrKL0Size = 128;
    constexpr uint32_t mm1NrKL0Loops = 2;
    constexpr uint32_t MERGE_K_PITCH = 576;

    LocalTensor<KV_T> mm1NrBL1Tensor;
    LocalTensor<KV_T> kTensor;
    uint32_t mm1NrKa = 0, mm1NrKb = 0;

    uint32_t curTopKIdx = sfaCubeRunInfo.curTopKIdx;
    uint64_t curOffsetInSparseBlock = sfaCubeRunInfo.curOffsetInSparseBlock;
    uint32_t copyRowCnt = 0;
    int64_t idInTopK = topKGm.GetValue(sfaCubeRunInfo.topKBaseOffset + curTopKIdx);

    uint32_t curTopKIdxTmp = 0;
    uint64_t curOffsetInSparseBlockTmp = 0;
    uint32_t copyRowCntTmp = 0;
    int64_t idInTopKTmp = 0;

    for (uint32_t mm1NrNL1 = 0; mm1NrNL1 < mm1NrNL1Loops; mm1NrNL1++) {
        if (mm1NrNL1 == (mm1NrNL1Loops - 1)) {
            mm1NrNL1Size = nSize - (mm1NrNL1Loops - 1) * N_SPLIT_SIZE;
            mm1NrNL1SizeAlign = SFAAlign(mm1NrNL1Size, 16U);
        }
        curTopKIdxTmp = curTopKIdx;
        curOffsetInSparseBlockTmp = curOffsetInSparseBlock;
        copyRowCntTmp = copyRowCnt;
        idInTopKTmp = idInTopK;

        for (uint32_t mm1NrKL1 = 0; mm1NrKL1 < mm1NrKL1Loops; mm1NrKL1++) {
            kvL1BufIter++;
            mm1NrKb = kvL1BufIter % 3;
            WaitFlag<HardEvent::MTE1_MTE2>(sfaMte21KvEvents[mm1NrKb]);
            mm1NrBL1Tensor = l1KVTensor[mm1NrKb * L1_BLOCK_OFFSET];

            uint32_t curSeqIdx = sfaCubeRunInfo.s2BatchOffset + mm1NrNL1 * N_SPLIT_SIZE;
            uint32_t copyFinishRowCnt = 0;
            curTopKIdx = curTopKIdxTmp;
            curOffsetInSparseBlock = curOffsetInSparseBlockTmp;
            copyRowCnt = copyRowCntTmp;
            idInTopK = idInTopKTmp;
            if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
                Nd2NzParams sfaCubeCopyParams;
                sfaCubeCopyParams.ndNum = 1;
                sfaCubeCopyParams.nValue = mm1NrNL1Size;
                sfaCubeCopyParams.dValue = sfaCubeConstInfo.headDim >> 1;
                sfaCubeCopyParams.srcDValue = sfaCubeConstInfo.headDim;
                sfaCubeCopyParams.dstNzC0Stride = mm1NrNL1SizeAlign;
                sfaCubeCopyParams.dstNzNStride = 1;
                sfaCubeCopyParams.srcNdMatrixStride = 0;
                sfaCubeCopyParams.dstNzMatrixStride = 0;
                DataCopy(mm1NrBL1Tensor,
                         kvMergeGm_[sfaCubeRunInfo.loop % 4 * N_WORKSPACE_SIZE * MERGE_K_PITCH +
                                    mm1NrKL1 * (sfaCubeConstInfo.headDim >> 1) +
                                    mm1NrNL1 * N_SPLIT_SIZE * sfaCubeConstInfo.headDim],
                         sfaCubeCopyParams);
            } else {
                while (copyFinishRowCnt < mm1NrNL1Size) {
                    CalcTopKBlockInfo(sfaCubeRunInfo, curTopKIdx, curOffsetInSparseBlock, curSeqIdx, copyRowCnt,
                                      idInTopK);
                    if (copyFinishRowCnt + copyRowCnt > mm1NrNL1Size) {
                        copyRowCnt = mm1NrNL1Size - copyFinishRowCnt;
                    }
                    if constexpr (PAGE_ATTENTION) {
                        Position startPos;
                        startPos.bIdx = sfaCubeRunInfo.bIdx;
                        startPos.n2Idx = sfaCubeRunInfo.n2Idx;
                        startPos.s2Idx = idInTopK * sfaCubeConstInfo.sparseBlockSize + curOffsetInSparseBlock;
                        startPos.dIdx = mm1NrKL1 * 256;
                        PAShape shape;
                        shape.blockSize = kvCacheBlockSize;
                        shape.headNum = sfaCubeConstInfo.kvHeadNum;
                        shape.headDim = sfaCubeConstInfo.headDim;
                        shape.actHeadDim = 256;
                        shape.maxblockNumPerBatch = maxBlockNumPerBatch;
                        shape.copyRowNum = copyRowCnt;
                        shape.copyRowNumAlign = mm1NrNL1SizeAlign;
                        kTensor = mm1NrBL1Tensor[copyFinishRowCnt * 16];
                        DataCopyPA<KV_T, KV_LAYOUT_T>(kTensor, keyGm, blockTableGm, shape, startPos);
                    } else {
                        uint64_t keyOffset = sfaCubeRunInfo.tensorBOffset;
                        if constexpr (KV_LAYOUT_T == SFA_LAYOUT::BSND || KV_LAYOUT_T == SFA_LAYOUT::TND) {
                            keyOffset += (idInTopK * sfaCubeConstInfo.sparseBlockSize + curOffsetInSparseBlock) *
                                         sfaCubeConstInfo.kvHeadNum * sfaCubeConstInfo.headDim;
                        } else {
                            keyOffset += (idInTopK * sfaCubeConstInfo.sparseBlockSize + curOffsetInSparseBlock) *
                                         sfaCubeConstInfo.headDim;
                        }
                        CopyInMm1BToL1(mm1NrBL1Tensor, keyOffset + mm1NrKL1 * 256, mm1NrNL1SizeAlign, copyFinishRowCnt,
                                       copyRowCnt, 256);
                    }
                    copyFinishRowCnt += copyRowCnt;
                    curSeqIdx += copyRowCnt;
                }
            }

            SetFlag<HardEvent::MTE2_MTE1>(sfaMte21KvEvents[mm1NrKb]);
            WaitFlag<HardEvent::MTE2_MTE1>(sfaMte21KvEvents[mm1NrKb]);
            mm1NrML1Size = M_SPLIT_SIZE;
            mm1NrML1SizeAlign = SFAAlign(M_SPLIT_SIZE, 16U);
            for (uint32_t mm1NrML1 = 0; mm1NrML1 < mm1NrML1Loops; mm1NrML1++) {
                uint32_t aL1PaddingSize = 0;
                if (mm1NrML1 == (mm1NrML1Loops - 1)) {
                    mm1NrML1Size = mSize - (mm1NrML1Loops - 1) * M_SPLIT_SIZE;
                    mm1NrML1SizeAlign = SFAAlign(mm1NrML1Size, 16U);
                    aL1PaddingSize = (M_SPLIT_SIZE - mm1NrML1SizeAlign) * 256;
                }
                uint32_t mIdx = qpL1BufIter + mm1NrML1;
                mm1NrKa = GetQPL1RealIdx(mIdx, mm1NrKL1);
                LocalTensor<Q_T> mm1NrAL1Tensor =
                    l1QPTensor[mm1NrKa * L1_BLOCK_OFFSET + (1 - mm1NrKL1) * aL1PaddingSize];
                if (mm1NrNL1 == 0) {
                    if (mm1NrKL1 == 0) {
                        WaitFlag<HardEvent::MTE1_MTE2>(mte21QPIds[mm1NrKa]);
                        WaitFlag<HardEvent::MTE1_MTE2>(mte21QPIds[mm1NrKa + 1]);
                    }
                    CopyInMm1AToL1(mm1NrAL1Tensor, sfaCubeRunInfo,
                                   sfaCubeSplitInfo.nBufferStartM + mm1NrML1 * M_SPLIT_SIZE, mm1NrML1Size, 256,
                                   mm1NrKL1 * 256);
                    SetFlag<HardEvent::MTE2_MTE1>(mte21QPIds[mm1NrKa]);
                    WaitFlag<HardEvent::MTE2_MTE1>(mte21QPIds[mm1NrKa]);
                }

                LocalTensor mm1NrCL0Tensor =
                    cL0TensorPingPong[(sfaL0CBufferIndex % 2) * (L0C_PP_SIZE / sizeof(MM_OUT_T))];
                for (uint32_t mm1NrKL0 = 0; mm1NrKL0 < mm1NrKL0Loops; mm1NrKL0++) {
                    WaitFlag<HardEvent::M_MTE1>(Mte1MmABEventId(abL0BufIter % 2));
                    LocalTensor<KV_T> mm1NrAL0Tensor =
                        aL0TensorPingPong[(abL0BufIter % 2) * (L0A_PP_SIZE / sizeof(KV_T))];
                    LoadDataMm1A(mm1NrAL0Tensor, mm1NrAL1Tensor, mm1NrKL0, mm1NrKL0Size, mm1NrML1SizeAlign,
                                 mm1NrKL0Size);
                    LocalTensor<KV_T> mm1NrBL0Tensor =
                        bL0TensorPingPong[(abL0BufIter % 2) * (L0B_PP_SIZE / sizeof(KV_T))];
                    LoadDataMm1B(mm1NrBL0Tensor, mm1NrBL1Tensor, mm1NrKL0, mm1NrKL0Size, mm1NrKL0Size,
                                 mm1NrNL1SizeAlign);
                    SetFlag<HardEvent::MTE1_M>(Mte1MmABEventId(abL0BufIter % 2));
                    WaitFlag<HardEvent::MTE1_M>(Mte1MmABEventId(abL0BufIter % 2));

                    MmadParams mm1NrMmadParams;
                    mm1NrMmadParams.m = mm1NrML1SizeAlign;
                    mm1NrMmadParams.n = mm1NrNL1SizeAlign;
                    mm1NrMmadParams.k = mm1NrKL0Size;
                    mm1NrMmadParams.cmatrixInitVal = (mm1NrKL1 == 0 && mm1NrKL0 == 0);
                    mm1NrMmadParams.cmatrixSource = false;
                    mm1NrMmadParams.unitFlag = (mm1NrKL1 == 1 && mm1NrKL0 == (mm1NrKL0Loops - 1)) ? 0b11 : 0b10;
                    Mmad(mm1NrCL0Tensor, mm1NrAL0Tensor, mm1NrBL0Tensor, mm1NrMmadParams);
                    if ((mm1NrMmadParams.m / 16) * (mm1NrMmadParams.n / 16) < 10) {
                        PipeBarrier<PIPE_M>();
                    }
                    SetFlag<HardEvent::M_MTE1>(Mte1MmABEventId(abL0BufIter % 2));
                    abL0BufIter++;
                }

                if (mm1NrNL1 == (mm1NrNL1Loops - 1)) {
                    SetFlag<HardEvent::MTE1_MTE2>(mte21QPIds[mm1NrKa]);
                }
                if (mm1NrKL1 == 1) {
                    FixpipeParamsV220 mm1NrFixParams;
                    mm1NrFixParams.srcStride = mm1NrML1SizeAlign;
                    mm1NrFixParams.nSize = mm1NrNL1SizeAlign;
                    mm1NrFixParams.mSize = mm1NrML1SizeAlign;
                    mm1NrFixParams.dstStride = sfaCubeRunInfo.actualSingleProcessSInnerSizeAlign;
                    mm1NrFixParams.unitFlag = 0b11;
                    mm1NrFixParams.ndNum = 1;
                    Fixpipe(
                        mm1ResGm[(sfaCubeRunInfo.loop % (sfaCubeConstInfo.preLoadNum)) * sfaCubeConstInfo.mmResUbSize +
                                 mm1NrNL1 * N_SPLIT_SIZE +
                                 (sfaCubeSplitInfo.nBufferStartM + mm1NrML1 * M_SPLIT_SIZE) *
                                     sfaCubeRunInfo.actualSingleProcessSInnerSizeAlign],
                        mm1NrCL0Tensor, mm1NrFixParams);
                }
                if (mm1NrML1Loops == 2) {
                    sfaL0CBufferIndex++;
                }
            }
            SetFlag<HardEvent::MTE1_MTE2>(sfaMte21KvEvents[mm1NrKb]);
        }
        if (mm1NrML1Loops == 1) {
            sfaL0CBufferIndex++;
        }
    }
    qpL1BufIter += mm1NrML1Loops;
}

template <typename SFAT>
__aicore__ inline void SFAMatmulService<SFAT>::ComputeMm1(const RunInfo &sfaCubeRunInfo,
                                                          const MSplitInfo sfaCubeSplitInfo)
{
    // 最外层还需要一层m的循环
    uint32_t mSize = sfaCubeSplitInfo.nBufferDealM;
    uint32_t mL1Size = M_SPLIT_SIZE;
    uint32_t mL1SizeAlign = SFAAlign(M_SPLIT_SIZE, 16U);
    uint32_t sfaML1LoopCount = (mSize + M_SPLIT_SIZE - 1) / M_SPLIT_SIZE;

    uint32_t nSize = sfaCubeRunInfo.actualSingleProcessSInnerSize;
    uint32_t nL1Size = N_SPLIT_SIZE;
    uint32_t nL1SizeAlign = SFAAlign(N_SPLIT_SIZE, 16U);
    uint32_t nL1Loops = (nSize + N_SPLIT_SIZE - 1) / N_SPLIT_SIZE;

    uint32_t kSize = 576;
    uint32_t kL1Size = 288;
    uint32_t kL1Loops = 2; // 2 : 576/288, mla专用 这里不考虑d泛化

    uint32_t kL0Size = 96;
    uint32_t kL0Loops = (kL1Size + kL0Size - 1) / kL0Size; // 288 / 96 = 3 kloops

    LocalTensor<KV_T> bL1Tensor;
    LocalTensor<KV_T> kRopeTensor;
    LocalTensor<KV_T> kTensor;
    // ka表示左矩阵4buf选择哪一块buf, kb表示右矩阵3buf选择哪一块buf
    uint32_t sfaQueryBufferIndex = 0, sfaKvBufferIndex = 0;

    uint32_t curTopKIdx = sfaCubeRunInfo.curTopKIdx;
    uint64_t curOffsetInSparseBlock = sfaCubeRunInfo.curOffsetInSparseBlock; // sparse Block块内偏移
    uint32_t copyRowCnt = 0;
    int64_t idInTopK = topKGm.GetValue(sfaCubeRunInfo.topKBaseOffset + curTopKIdx);

    uint32_t curTopKIdxTmp = 0;
    uint64_t curOffsetInSparseBlockTmp = 0;
    uint32_t copyRowCntTmp = 0;
    int64_t idInTopKTmp = 0;

    // L1 切n切k切m
    for (uint32_t nL1 = 0; nL1 < nL1Loops; nL1++) { // L1切n, 512/128=4
        if (nL1 == (nL1Loops - 1)) {
            // 尾块重新计算size
            nL1Size = nSize - (nL1Loops - 1) * N_SPLIT_SIZE;
            nL1SizeAlign = SFAAlign(nL1Size, 16U);
        }
        curTopKIdxTmp = curTopKIdx;
        curOffsetInSparseBlockTmp = curOffsetInSparseBlock;
        copyRowCntTmp = copyRowCnt;
        idInTopKTmp = idInTopK;

        for (uint32_t kL1 = 0; kL1 < kL1Loops; kL1++) { // L1切k, 576/288, 这里不考虑d泛化
            kvL1BufIter++;
            uint32_t sfaKvBufferIndex = kvL1BufIter % 3;
            WaitFlag<HardEvent::MTE1_MTE2>(sfaMte21KvEvents[sfaKvBufferIndex]);
            // 从k当中取当前的块
            bL1Tensor = l1KVTensor[sfaKvBufferIndex * L1_BLOCK_OFFSET];
            // mm1拷贝主流程

            uint32_t curSeqIdx = sfaCubeRunInfo.s2BatchOffset + nL1 * N_SPLIT_SIZE;
            uint32_t copyFinishRowCnt = 0;
            curTopKIdx = curTopKIdxTmp;
            curOffsetInSparseBlock = curOffsetInSparseBlockTmp;
            copyRowCnt = copyRowCntTmp;
            idInTopK = idInTopKTmp;
            if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
                if (kL1 == 0) {
                    Nd2NzParams sfaCubeCopyParams;
                    sfaCubeCopyParams.ndNum = 1;
                    sfaCubeCopyParams.nValue = nL1Size; // 行数
                    sfaCubeCopyParams.dValue = sfaCubeConstInfo.headDim >> 1;
                    sfaCubeCopyParams.srcDValue = sfaCubeConstInfo.headDim;
                    sfaCubeCopyParams.dstNzC0Stride = nL1SizeAlign;
                    sfaCubeCopyParams.dstNzNStride = 1;
                    sfaCubeCopyParams.srcNdMatrixStride = 0;
                    sfaCubeCopyParams.dstNzMatrixStride = 0;
                    DataCopy(bL1Tensor,
                             kvMergeGm_[sfaCubeRunInfo.loop % 4 * N_WORKSPACE_SIZE * kSize +
                                        nL1 * N_SPLIT_SIZE * sfaCubeConstInfo.headDim],
                             sfaCubeCopyParams);
                    sfaCubeCopyParams.dValue = sfaCubeConstInfo.headDimRope >> 1;
                    sfaCubeCopyParams.srcDValue = sfaCubeConstInfo.headDimRope;
                    DataCopy(bL1Tensor[nL1SizeAlign * (sfaCubeConstInfo.headDim >> 1)],
                             kvMergeGm_[sfaCubeRunInfo.loop % 4 * N_WORKSPACE_SIZE * kSize +
                                        N_WORKSPACE_SIZE * sfaCubeConstInfo.headDim +
                                        nL1 * N_SPLIT_SIZE * sfaCubeConstInfo.headDimRope],
                             sfaCubeCopyParams);
                } else {
                    LocalTensor<Q_T> kTmpTensor = bL1Tensor[(sfaCubeConstInfo.headDimRope >> 1) * nL1SizeAlign];
                    Nd2NzParams sfaCubeCopyParams;
                    sfaCubeCopyParams.ndNum = 1;
                    sfaCubeCopyParams.nValue = nL1Size; // 行数
                    sfaCubeCopyParams.dValue = sfaCubeConstInfo.headDim >> 1;
                    sfaCubeCopyParams.srcDValue = sfaCubeConstInfo.headDim;
                    sfaCubeCopyParams.dstNzC0Stride = nL1SizeAlign;
                    sfaCubeCopyParams.dstNzNStride = 1;
                    sfaCubeCopyParams.srcNdMatrixStride = 0;
                    sfaCubeCopyParams.dstNzMatrixStride = 0;
                    DataCopy(
                        kTmpTensor,
                        kvMergeGm_[sfaCubeRunInfo.loop % 4 * N_WORKSPACE_SIZE * kSize +
                                   (sfaCubeConstInfo.headDim >> 1) + nL1 * N_SPLIT_SIZE * sfaCubeConstInfo.headDim],
                        sfaCubeCopyParams);
                    sfaCubeCopyParams.dValue = sfaCubeConstInfo.headDimRope >> 1;
                    sfaCubeCopyParams.srcDValue = sfaCubeConstInfo.headDimRope;
                    DataCopy(
                        bL1Tensor,
                        kvMergeGm_[sfaCubeRunInfo.loop % 4 * N_WORKSPACE_SIZE * kSize +
                                   N_WORKSPACE_SIZE * sfaCubeConstInfo.headDim + (sfaCubeConstInfo.headDimRope >> 1) +
                                   nL1 * N_SPLIT_SIZE * sfaCubeConstInfo.headDimRope],
                        sfaCubeCopyParams);
                }
            } else {
                while (copyFinishRowCnt < nL1Size) {
                    CalcTopKBlockInfo(sfaCubeRunInfo, curTopKIdx, curOffsetInSparseBlock, curSeqIdx, copyRowCnt,
                                      idInTopK);
                    if (copyFinishRowCnt + copyRowCnt > nL1Size) {
                        copyRowCnt = nL1Size - copyFinishRowCnt;
                    }

                    // BN2轴偏移
                    if constexpr (PAGE_ATTENTION) {
                        Position startPos;
                        startPos.bIdx = sfaCubeRunInfo.bIdx;
                        startPos.n2Idx = sfaCubeRunInfo.n2Idx;
                        startPos.s2Idx = idInTopK * sfaCubeConstInfo.sparseBlockSize + curOffsetInSparseBlock;
                        // 256、32等待7buf命名更改
                        startPos.dIdx = kL1 * 256; // mm1 右矩阵 bn2s2d, d为k轴不切; mm2 右矩阵, s2为k轴, d轴切分
                        Position ropeStartPos = startPos;
                        ropeStartPos.dIdx = kL1 * 32;
                        PAShape shape;
                        shape.blockSize = kvCacheBlockSize;
                        shape.headNum = sfaCubeConstInfo.kvHeadNum;
                        shape.headDim = sfaCubeConstInfo.headDim;
                        shape.actHeadDim = 256;
                        shape.maxblockNumPerBatch = maxBlockNumPerBatch;
                        shape.copyRowNum = copyRowCnt;
                        shape.copyRowNumAlign = nL1SizeAlign;
                        PAShape ropeShape = shape;
                        ropeShape.headDim = sfaCubeConstInfo.headDimRope;
                        ropeShape.actHeadDim = 32;
                        if (kL1 == 0) {
                            kTensor = bL1Tensor[copyFinishRowCnt * 16];
                            DataCopyPA<KV_T, KV_LAYOUT_T>(kTensor, keyGm, blockTableGm, shape, startPos);
                            kRopeTensor = bL1Tensor[(nL1SizeAlign * (BlockAlign<KV_T>(sfaCubeConstInfo.headDim) >> 1)) +
                                                    copyFinishRowCnt * 16];
                            DataCopyPA<KV_T, KV_LAYOUT_T>(kRopeTensor, kRopeGm, blockTableGm, ropeShape, ropeStartPos);
                        } else {
                            kRopeTensor = bL1Tensor[copyFinishRowCnt * 16];
                            DataCopyPA<KV_T, KV_LAYOUT_T>(kRopeTensor, kRopeGm, blockTableGm, ropeShape, ropeStartPos);
                            LocalTensor<Q_T> kTmpTensor = bL1Tensor[32 * nL1SizeAlign + copyFinishRowCnt * 16];
                            DataCopyPA<KV_T, KV_LAYOUT_T>(kTmpTensor, keyGm, blockTableGm, shape, startPos);
                        }
                    } else {
                        uint64_t keyOffset = sfaCubeRunInfo.tensorBOffset;
                        uint64_t kRopeOffset = sfaCubeRunInfo.tensorBRopeOffset;
                        if constexpr (KV_LAYOUT_T == SFA_LAYOUT::BSND || KV_LAYOUT_T == SFA_LAYOUT::TND) {
                            keyOffset += (idInTopK * sfaCubeConstInfo.sparseBlockSize + curOffsetInSparseBlock) *
                                         sfaCubeConstInfo.kvHeadNum * sfaCubeConstInfo.headDim;
                            kRopeOffset += (idInTopK * sfaCubeConstInfo.sparseBlockSize + curOffsetInSparseBlock) *
                                           sfaCubeConstInfo.kvHeadNum * sfaCubeConstInfo.headDimRope;
                        } else {
                            keyOffset += (idInTopK * sfaCubeConstInfo.sparseBlockSize + curOffsetInSparseBlock) *
                                         sfaCubeConstInfo.headDim;
                            kRopeOffset += (idInTopK * sfaCubeConstInfo.sparseBlockSize + curOffsetInSparseBlock) *
                                           sfaCubeConstInfo.headDimRope;
                        }

                        if (kL1 == 0) {
                            CopyInMm1BToL1(bL1Tensor, keyOffset, nL1SizeAlign, copyFinishRowCnt, copyRowCnt, 256);
                            kRopeTensor = bL1Tensor[nL1SizeAlign * (BlockAlign<KV_T>(sfaCubeConstInfo.headDim) >> 1)];
                            CopyInMm1BRopeToL1(kRopeTensor, kRopeOffset, nL1SizeAlign, copyFinishRowCnt, copyRowCnt,
                                               32);
                        } else {
                            kRopeTensor = bL1Tensor;
                            CopyInMm1BRopeToL1(kRopeTensor, kRopeOffset + 32, nL1SizeAlign, copyFinishRowCnt,
                                               copyRowCnt, 32);
                            LocalTensor<Q_T> kTmpTensor = bL1Tensor[nL1SizeAlign * 32];
                            CopyInMm1BToL1(kTmpTensor, keyOffset + 256, nL1SizeAlign, copyFinishRowCnt, copyRowCnt,
                                           256);
                        }
                    }

                    // 更新循环变量
                    copyFinishRowCnt += copyRowCnt;
                    curSeqIdx += copyRowCnt;
                }
            }

            SetFlag<HardEvent::MTE2_MTE1>(sfaMte21KvEvents[sfaKvBufferIndex]);
            WaitFlag<HardEvent::MTE2_MTE1>(sfaMte21KvEvents[sfaKvBufferIndex]);
            mL1Size = M_SPLIT_SIZE;
            mL1SizeAlign = SFAAlign(M_SPLIT_SIZE, 16U);
            for (uint32_t mL1 = 0; mL1 < sfaML1LoopCount; mL1++) {
                uint32_t aL1PaddingSize = 0; // 用于使左矩阵对齐到尾部, 以保证两块32K内存连续
                if (mL1 == (sfaML1LoopCount - 1)) {
                    // 尾块重新计算size
                    mL1Size = mSize - (sfaML1LoopCount - 1) * M_SPLIT_SIZE;
                    mL1SizeAlign = SFAAlign(mL1Size, 16U);
                    // mL1SizeAlign<128 kL1=0时需要偏移, 确保qRope能一半拷贝到当前tensor, 一半拷贝到下一个tensor
                    aL1PaddingSize = (M_SPLIT_SIZE - mL1SizeAlign) * 288;
                }

                // 左矩阵L1选择12块还是34块的index, 由m l1 index决定
                // 左矩阵L1选择12块或34块的前一块还是后一块, 由k l1 index决定
                uint32_t mIdx = qpL1BufIter + mL1;
                sfaQueryBufferIndex = GetQPL1RealIdx(mIdx, kL1);
                LocalTensor<Q_T> aL1Tensor =
                    l1QPTensor[sfaQueryBufferIndex * L1_BLOCK_OFFSET + (1 - kL1) * aL1PaddingSize]; // kL1=0时需要偏移
                if (nL1 == 0) { // mL1=0, mL1=1两次
                    if (kL1 == 0) {
                        WaitFlag<HardEvent::MTE1_MTE2>(mte21QPIds[sfaQueryBufferIndex]);
                        WaitFlag<HardEvent::MTE1_MTE2>(mte21QPIds[sfaQueryBufferIndex + 1]);
                        CopyInMm1AToL1(aL1Tensor, sfaCubeRunInfo, sfaCubeSplitInfo.nBufferStartM + mL1 * M_SPLIT_SIZE,
                                       mL1Size, 256, 0);
                        // 由于L1里面是NZ, 这里q rope的偏移为整块q nope切k的后大小, 256为headDim的一半
                        LocalTensor<Q_T> qRopeTensor = aL1Tensor[mL1SizeAlign * 256];
                        CopyInMm1ARopeToL1(qRopeTensor, sfaCubeRunInfo,
                                           sfaCubeSplitInfo.nBufferStartM + mL1 * M_SPLIT_SIZE, mL1Size);
                    } else {
                        // 32为rope headDim的一半
                        LocalTensor<Q_T> qTmpTensor = aL1Tensor[mL1SizeAlign * 32];
                        CopyInMm1AToL1(qTmpTensor, sfaCubeRunInfo, sfaCubeSplitInfo.nBufferStartM + mL1 * M_SPLIT_SIZE,
                                       mL1Size, 256, 256);
                    }
                    SetFlag<HardEvent::MTE2_MTE1>(mte21QPIds[sfaQueryBufferIndex]);
                    WaitFlag<HardEvent::MTE2_MTE1>(mte21QPIds[sfaQueryBufferIndex]);
                }

                // 使用unitflag同步
                LocalTensor cL0Tensor =
                    cL0TensorPingPong[(sfaL0CBufferIndex % 2) *
                                      (L0C_PP_SIZE / sizeof(MM_OUT_T))]; // 需要保证cL0BufIter和m步调一致
                for (uint32_t kL0 = 0; kL0 < kL0Loops; kL0++) {
                    WaitFlag<HardEvent::M_MTE1>(Mte1MmABEventId(abL0BufIter % 2));
                    LocalTensor<KV_T> aL0Tensor = aL0TensorPingPong[(abL0BufIter % 2) * (L0A_PP_SIZE / sizeof(KV_T))];
                    LoadDataMm1A(aL0Tensor, aL1Tensor, kL0, kL0Size, mL1SizeAlign, kL0Size);
                    LocalTensor<KV_T> bL0Tensor = bL0TensorPingPong[(abL0BufIter % 2) * (L0B_PP_SIZE / sizeof(KV_T))];
                    LoadDataMm1B(bL0Tensor, bL1Tensor, kL0, kL0Size, kL0Size, nL1SizeAlign);
                    SetFlag<HardEvent::MTE1_M>(Mte1MmABEventId(abL0BufIter % 2));
                    WaitFlag<HardEvent::MTE1_M>(Mte1MmABEventId(abL0BufIter % 2));

                    // m == 1的时候需要特殊处理
                    MmadParams sfaMmadParams;
                    sfaMmadParams.m = mL1SizeAlign;
                    sfaMmadParams.n = nL1SizeAlign;
                    sfaMmadParams.k = kL0Size;
                    sfaMmadParams.cmatrixInitVal = (kL1 == 0 && kL0 == 0);
                    sfaMmadParams.cmatrixSource = false;
                    sfaMmadParams.unitFlag =
                        (kL1 == 1 && kL0 == (kL0Loops - 1)) ? 0b11 : 0b10; // 累加最后一次翻转flag, 表示可以搬出
                    Mmad(cL0Tensor, aL0Tensor, bL0Tensor, sfaMmadParams);

                    if ((sfaMmadParams.m / 16) * (sfaMmadParams.n / 16) < 10) {
                        PipeBarrier<PIPE_M>();
                    }
                    SetFlag<HardEvent::M_MTE1>(Mte1MmABEventId(abL0BufIter % 2));
                    abL0BufIter++;
                }

                if (nL1 == (nL1Loops - 1)) {
                    SetFlag<HardEvent::MTE1_MTE2>(
                        mte21QPIds[sfaQueryBufferIndex]); // 反向同步, 表示L1中的A已经被mte1消费完
                }

                if (kL1 == 1) { // 最后一轮kL1循环
                    FixpipeParamsV220 fixParams;
                    fixParams.srcStride = mL1SizeAlign;
                    fixParams.nSize = nL1SizeAlign;
                    fixParams.mSize = mL1SizeAlign;
                    // 改成nSizeAlign
                    fixParams.dstStride = sfaCubeRunInfo.actualSingleProcessSInnerSizeAlign; // mm1ResGm两行之间的间隔
                    fixParams.unitFlag = 0b11;
                    fixParams.ndNum = 1; // 输出ND

                    // 输出偏移info.loop % (constInfo.preLoadNum)) * mmResUbSize是否在matmul里计算
                    Fixpipe(
                        mm1ResGm[(sfaCubeRunInfo.loop % (sfaCubeConstInfo.preLoadNum)) * sfaCubeConstInfo.mmResUbSize +
                                 nL1 * N_SPLIT_SIZE +
                                 (sfaCubeSplitInfo.nBufferStartM + mL1 * M_SPLIT_SIZE) *
                                     sfaCubeRunInfo.actualSingleProcessSInnerSizeAlign],
                        cL0Tensor, fixParams);
                }
                if (sfaML1LoopCount == 2) {
                    sfaL0CBufferIndex++;
                }
            }
            SetFlag<HardEvent::MTE1_MTE2>(sfaMte21KvEvents[sfaKvBufferIndex]); // 反向同步, 表示L1已经被mte1消费完
        }
        if (sfaML1LoopCount == 1) {
            sfaL0CBufferIndex++;
        }
    }
    qpL1BufIter += sfaML1LoopCount;
}

template <typename SFAT>
__aicore__ inline void SFAMatmulService<SFAT>::ComputeMm2(const RunInfo &sfaCubeRunInfo,
                                                          const MSplitInfo sfaCubeSplitInfo)
{
    uint32_t mSize = sfaCubeSplitInfo.nBufferDealM;
    uint32_t mSizeAlign = (mSize + 16 - 1) / 16;
    uint32_t sfaML1LoopCount = (mSize + M_SPLIT_SIZE - 1) / M_SPLIT_SIZE;
    uint32_t mL1SizeAlign = M_SPLIT_SIZE; // 16对齐
    uint32_t mL1Size = M_SPLIT_SIZE;      // m的实际大小

    uint32_t nSize = BlockAlign<KV_T>(sfaCubeConstInfo.headDim);
    uint32_t nL1Loops = (nSize + N_SPLIT_SIZE - 1) / N_SPLIT_SIZE;
    uint32_t nL1SizeAlign = N_SPLIT_SIZE; // 16对齐
    uint32_t nL1Size = N_SPLIT_SIZE;      // n的实际大小

    uint32_t kSize = sfaCubeRunInfo.actualSingleProcessSInnerSize;
    uint32_t kL1Size = 256;
    uint32_t kL1SizeAlign = SFAAlign(kL1Size, 16U);
    uint32_t kL1Loops = (kSize + kL1Size - 1) / kL1Size;
    uint32_t kL0Size = 128;
    uint32_t kL0Loops = (kL1Size + kL0Size - 1) / kL0Size;
    uint32_t kL0SizeAlign = kL0Size;
    LocalTensor<KV_T> bL1Tensor;
    LocalTensor<KV_T> subvTensor;

    // ka表示左矩阵4buf选择哪一块buf, kb表示右矩阵3buf选择哪一块buf
    uint32_t sfaQueryBufferIndex = 0, sfaKvBufferIndex = 0;
    uint32_t mBaseIdx = qpL1BufIter;
    for (uint32_t nL1 = 0; nL1 < nL1Loops; nL1++) { // n切L1
        if (nL1 == (nL1Loops - 1)) {
            // 尾块
            nL1Size = nSize - (nL1Loops - 1) * N_SPLIT_SIZE;
            nL1SizeAlign = SFAAlign(nL1Size, 16U);
        }

        // k l1写成一个循环, 和mm1保持一致
        kL1Size = 256;
        kL1SizeAlign = SFAAlign(kL1Size, 16U);

        uint32_t curTopKIdx = sfaCubeRunInfo.curTopKIdx;
        uint64_t curOffsetInSparseBlock = sfaCubeRunInfo.curOffsetInSparseBlock;
        uint32_t copyRowCnt = 0;
        int64_t idInTopK = topKGm.GetValue(sfaCubeRunInfo.topKBaseOffset + curTopKIdx);

        for (uint32_t k1 = 0; k1 < kL1Loops; k1++) { // k切L1, 这里套了一层l0来操作
            if (k1 == (kL1Loops - 1)) {
                // 尾块
                kL1Size = kSize - (kL1Loops - 1) * 256;
                kL1SizeAlign = SFAAlign(kL1Size, 16U);
            }
            kvL1BufIter++;
            uint32_t sfaKvBufferIndex = kvL1BufIter % 3;
            WaitFlag<HardEvent::MTE1_MTE2>(sfaMte21KvEvents[sfaKvBufferIndex]);
            bL1Tensor = l1KVTensor[sfaKvBufferIndex * L1_BLOCK_OFFSET];
            uint32_t kOffset = k1 * kL0Loops;
            kL0Size = 128;
            // 此处必须先初始化kL0Size, 再求kL0Loops, 否则由于循环会改变kL0Size大小, 导致kL0Loops错误
            kL0Loops = (kL1Size + kL0Size - 1) / kL0Size;
            kL0SizeAlign = kL0Size;
            for (uint32_t kL1 = kOffset; kL1 < kL0Loops + kOffset; kL1++) { // 128 循环搬pa
                if (kL1 == kOffset + kL0Loops - 1) {
                    // 尾块
                    kL0Size = kL1Size - (kL0Loops - 1) * kL0Size;
                    kL0SizeAlign = SFAAlign(kL0Size, 16U);
                }

                uint32_t curSeqIdx = sfaCubeRunInfo.s2BatchOffset + (kL1 - kOffset) * 128 + k1 * 256;
                uint32_t copyFinishRowCnt = 0;
                if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
                    Nd2NzParams sfaCubeCopyParams;
                    sfaCubeCopyParams.ndNum = 1;
                    sfaCubeCopyParams.nValue = kL0Size; // 行数
                    sfaCubeCopyParams.dValue = N_SPLIT_SIZE;
                    sfaCubeCopyParams.srcDValue = sfaCubeConstInfo.headDim;
                    sfaCubeCopyParams.dstNzC0Stride = kL0SizeAlign;
                    sfaCubeCopyParams.dstNzNStride = 1;
                    sfaCubeCopyParams.srcNdMatrixStride = 0;
                    sfaCubeCopyParams.dstNzMatrixStride = 0;
                    DataCopy(bL1Tensor[(kL1 - kOffset) * 128 * N_SPLIT_SIZE],
                             kvMergeGm_[sfaCubeRunInfo.loop % 4 * N_WORKSPACE_SIZE * 576 +
                                        kL1 * 128 * sfaCubeConstInfo.headDim + nL1 * N_SPLIT_SIZE],
                             sfaCubeCopyParams);
                } else {
                    while (copyFinishRowCnt < kL0Size) {
                        CalcTopKBlockInfo(sfaCubeRunInfo, curTopKIdx, curOffsetInSparseBlock, curSeqIdx, copyRowCnt,
                                          idInTopK);

                        if (copyFinishRowCnt + copyRowCnt > kL0Size) {
                            copyRowCnt = kL0Size - copyFinishRowCnt;
                        }

                        if constexpr (PAGE_ATTENTION) {
                            Position startPos;
                            startPos.bIdx = sfaCubeRunInfo.bIdx;
                            startPos.n2Idx = sfaCubeRunInfo.n2Idx;
                            startPos.s2Idx = idInTopK * sfaCubeConstInfo.sparseBlockSize + curOffsetInSparseBlock;
                            startPos.dIdx =
                                nL1 * N_SPLIT_SIZE; // mm1 右矩阵 bn2s2d, d为k轴不切; mm2 右矩阵, s2为k轴, d轴切分
                            PAShape shape;
                            shape.blockSize = kvCacheBlockSize;
                            shape.headNum = sfaCubeConstInfo.kvHeadNum;
                            shape.headDim = sfaCubeConstInfo.headDim;
                            shape.actHeadDim = nL1Size;
                            shape.maxblockNumPerBatch = maxBlockNumPerBatch;
                            shape.copyRowNum = copyRowCnt;
                            shape.copyRowNumAlign = kL0SizeAlign;
                            subvTensor = bL1Tensor[(kL1 - kOffset) * 128 * N_SPLIT_SIZE + copyFinishRowCnt * 16];
                            DataCopyPA<KV_T, KV_LAYOUT_T>(subvTensor, valueGm, blockTableGm, shape, startPos);
                        } else {
                            uint64_t valueOffset = sfaCubeRunInfo.tensorBOffset;
                            if constexpr (KV_LAYOUT_T == SFA_LAYOUT::BSND || KV_LAYOUT_T == SFA_LAYOUT::TND) {
                                valueOffset += (idInTopK * sfaCubeConstInfo.sparseBlockSize + curOffsetInSparseBlock) *
                                               sfaCubeConstInfo.kvHeadNum * sfaCubeConstInfo.headDim;
                            } else {
                                valueOffset += (idInTopK * sfaCubeConstInfo.sparseBlockSize + curOffsetInSparseBlock) *
                                               sfaCubeConstInfo.headDim;
                            }

                            subvTensor = bL1Tensor[(kL1 - kOffset) * 128 * N_SPLIT_SIZE];
                            CopyInMm2BToL1(subvTensor, valueOffset, kL0SizeAlign, copyFinishRowCnt, copyRowCnt,
                                           nL1 * N_SPLIT_SIZE, nL1Size);
                        }
                        // 更新循环变量
                        copyFinishRowCnt += copyRowCnt;
                        curSeqIdx += copyRowCnt;
                    }
                }
            }
            SetFlag<HardEvent::MTE2_MTE1>(sfaMte21KvEvents[sfaKvBufferIndex]);
            WaitFlag<HardEvent::MTE2_MTE1>(sfaMte21KvEvents[sfaKvBufferIndex]);
            mL1SizeAlign = M_SPLIT_SIZE;
            mL1Size = M_SPLIT_SIZE; // m的实际大小
            for (uint32_t mL1 = 0; mL1 < sfaML1LoopCount; mL1++) {
                if (mL1 == (sfaML1LoopCount - 1)) {
                    // 尾块
                    mL1Size = mSize - (sfaML1LoopCount - 1) * M_SPLIT_SIZE;
                    mL1SizeAlign = SFAAlign(mL1Size, 16U);
                }

                uint32_t mIdx = mBaseIdx + mL1;
                sfaQueryBufferIndex = GetQPL1RealIdx(mIdx, k1);
                LocalTensor<KV_T> aL1Tensor = l1QPTensor[sfaQueryBufferIndex * L1_BLOCK_OFFSET];
                if (nL1 == 0) {
                    WaitFlag<HardEvent::MTE1_MTE2>(mte21QPIds[sfaQueryBufferIndex]);
                    CopyInMm2AToL1(aL1Tensor, sfaCubeRunInfo, sfaCubeSplitInfo.nBufferStartM + mL1 * M_SPLIT_SIZE,
                                   mL1Size, kL1Size, 256 * k1);
                    SetFlag<HardEvent::MTE2_MTE1>(mte21QPIds[sfaQueryBufferIndex]);
                    WaitFlag<HardEvent::MTE2_MTE1>(mte21QPIds[sfaQueryBufferIndex]);
                }

                LocalTensor cL0Tensor =
                    cL0TensorPingPong[(sfaL0CBufferIndex % 2) *
                                      (L0C_PP_SIZE / sizeof(MM_OUT_T))]; // 需要保证cL0BufIter和m步调一致
                uint32_t baseK = 128;
                uint32_t baseN = 128;
                kL0Size = 128;
                kL0SizeAlign = kL0Size;
                for (uint32_t kL0 = 0; kL0 < kL0Loops; kL0++) {
                    if (kL0 + 1 == kL0Loops) {
                        kL0Size = kL1Size - (kL0Loops - 1) * kL0Size;
                        kL0SizeAlign = SFAAlign(kL0Size, 16U);
                    }
                    WaitFlag<HardEvent::M_MTE1>(Mte1MmABEventId(abL0BufIter % 2));
                    LocalTensor<KV_T> bL0Tensor = bL0TensorPingPong[(abL0BufIter % 2) * (L0B_PP_SIZE / sizeof(KV_T))];
                    LoadData3DParamsV2<KV_T> sfaMm2LoadBParams;
                    sfaMm2LoadBParams.l1H = kL0SizeAlign / 16; // 源操作数height
                    sfaMm2LoadBParams.l1W = 16;                // 源操作数weight=16，目的height=l1H*L1W
                    sfaMm2LoadBParams.padList[0] = 0;
                    sfaMm2LoadBParams.padList[1] = 0;
                    sfaMm2LoadBParams.padList[2] = 0;
                    sfaMm2LoadBParams.padList[3] = 255; // 尾部数据不影响滑窗的结果

                    sfaMm2LoadBParams.mStartPt = 0;              // 卷积核在目的操作数width维度的起点
                    sfaMm2LoadBParams.kStartPt = 0;              // 卷积核在目的操作数height维度的起点
                    sfaMm2LoadBParams.mExtension = kL0SizeAlign; // 在目的操作数height维度的传输长度
                    sfaMm2LoadBParams.kExtension = nL1SizeAlign; // 在目的操作数width维度的传输长度
                    sfaMm2LoadBParams.strideW = 1;
                    sfaMm2LoadBParams.strideH = 1;
                    sfaMm2LoadBParams.filterW = 1;
                    sfaMm2LoadBParams.filterSizeW = false; // 是否在filterW的基础上将卷积核width增加256个元素
                    sfaMm2LoadBParams.filterH = 1;
                    sfaMm2LoadBParams.filterSizeH = false; // 是否在filterH的基础上将卷积核height增加256个元素
                    sfaMm2LoadBParams.dilationFilterW = 1; // 卷积核width膨胀系数
                    sfaMm2LoadBParams.dilationFilterH = 1; // 卷积核height膨胀系数
                    sfaMm2LoadBParams.enTranspose = 1;     // 是否启用转置功能
                    sfaMm2LoadBParams.fMatrixCtrl =
                        0; // 使用FMATRIX_LEFT还是使用FMATRIX_RIGHT，=0使用FMATRIX_LEFT，=1使用FMATRIX_RIGHT 1
                    sfaMm2LoadBParams.channelSize =
                        nL1SizeAlign; // 源操作数的通道数。膨胀系数为1时，目的weight为filterW*filterH*channelSize
                    LoadData<KV_T, LOAD3DV2_CONFIG>(bL0Tensor, bL1Tensor[kL0 * baseK * baseN], sfaMm2LoadBParams);

                    LocalTensor<KV_T> aL0Tensor = aL0TensorPingPong[(abL0BufIter % 2) * (L0A_PP_SIZE / sizeof(KV_T))];
                    LoadData3DParamsV2<KV_T> sfaMm2LoadAParams;
                    sfaMm2LoadAParams.padList[0] = 0;
                    sfaMm2LoadAParams.padList[1] = 0;
                    sfaMm2LoadAParams.padList[2] = 0;
                    sfaMm2LoadAParams.padList[3] = 255;        // 尾部数据不影响滑窗的结果
                    sfaMm2LoadAParams.l1H = mL1SizeAlign / 16; // 源操作数height
                    sfaMm2LoadAParams.l1W = 16;                // 源操作数weight

                    sfaMm2LoadAParams.mExtension = mL1SizeAlign; // 在目的操作数height维度的传输长度
                    sfaMm2LoadAParams.kExtension = kL0SizeAlign; // 在目的操作数width维度的传输长度
                    sfaMm2LoadAParams.strideW = 1;               // 卷积核在源操作数width维度滑动的步长
                    sfaMm2LoadAParams.strideH = 1;               // 卷积核在源操作数height维度滑动的步长
                    sfaMm2LoadAParams.mStartPt = 0;              // 卷积核在目的操作数width维度的起点
                    sfaMm2LoadAParams.kStartPt = 0;              // 卷积核在目的操作数height维度的起点
                    sfaMm2LoadAParams.filterW = 1;               // 卷积核width
                    sfaMm2LoadAParams.filterSizeW = false; // 是否在filterW的基础上将卷积核width增加256个元素
                    sfaMm2LoadAParams.filterH = 1;         // 卷积核height
                    sfaMm2LoadAParams.filterSizeH = false; // 是否在filterH的基础上将卷积核height增加256个元素
                    sfaMm2LoadAParams.enTranspose = 0; // 是否启用转置功能，对整个目标矩阵进行转置
                    sfaMm2LoadAParams.fMatrixCtrl = 0;
                    sfaMm2LoadAParams.dilationFilterW = 1; // 卷积核width膨胀系数
                    sfaMm2LoadAParams.dilationFilterH = 1; // 卷积核height膨胀系数
                    sfaMm2LoadAParams.channelSize =
                        kL0SizeAlign; // 源操作数的通道数。膨胀系数为1时，目的weight为filterW*filterH*channelSize
                    LoadData<KV_T, LOAD3DV2_CONFIG>(aL0Tensor, aL1Tensor[kL0 * baseK * mL1SizeAlign],
                                                    sfaMm2LoadAParams);
                    SetFlag<HardEvent::MTE1_M>(Mte1MmABEventId(abL0BufIter % 2));
                    WaitFlag<HardEvent::MTE1_M>(Mte1MmABEventId(abL0BufIter % 2));

                    MmadParams sfaMmadParams;
                    sfaMmadParams.m = mL1SizeAlign;
                    sfaMmadParams.n = nL1SizeAlign;
                    sfaMmadParams.k = kL0Size;
                    sfaMmadParams.cmatrixInitVal = (kL0 == 0 && k1 == 0);
                    sfaMmadParams.cmatrixSource = false;
                    sfaMmadParams.unitFlag = ((k1 == (kL1Loops - 1)) && (kL0 == (kL0Loops - 1))) ? 0b11 : 0b10;

                    Mmad(cL0Tensor, aL0Tensor, bL0Tensor, sfaMmadParams);
                    if (((sfaMmadParams.m / 16) * (sfaMmadParams.n / 16)) < 10) {
                        PipeBarrier<PIPE_M>();
                    }
                    SetFlag<HardEvent::M_MTE1>(Mte1MmABEventId(abL0BufIter % 2));
                    abL0BufIter++;
                }

                if (nL1 == (nL1Loops - 1)) { // nL1最后一轮, 需要将B驻留在L1中, 用于下一轮的计算？
                    SetFlag<HardEvent::MTE1_MTE2>(
                        mte21QPIds[sfaQueryBufferIndex]); // 反向同步, 表示L1中的A已经被mte1消费完
                }

                if (k1 == (kL1Loops - 1)) {
                    if (nL1 == 0 && mL1 == 0) { // 第一次Fixpipe前等待
                        CrossCoreWaitFlag(sfaCubeConstInfo.syncV1NupdateC2);
                    }

                    SetAtomicAdd<MM_OUT_T>();
                    // ND
                    FixpipeParamsV220 fixParams;
                    fixParams.nSize = nL1SizeAlign;
                    fixParams.mSize = mL1SizeAlign;
                    fixParams.srcStride = mL1SizeAlign;
                    fixParams.dstStride = nSize; // mm2ResGm两行之间的间隔
                    fixParams.ndNum = 1;         // 输出ND
                    fixParams.unitFlag = 0b11;

                    uint64_t mm2Offset =
                        (sfaCubeSplitInfo.nBufferStartM + mL1 * M_SPLIT_SIZE) * nSize + nL1 * N_SPLIT_SIZE;
                    Fixpipe(mm2ResGm[(sfaCubeRunInfo.bn2IdxInCurCore % (sfaCubeConstInfo.preLoadNum)) *
                                         sfaCubeConstInfo.bmm2ResUbSize +
                                     mm2Offset],
                            cL0Tensor, fixParams);
                    SetAtomicNone();
                }

                if (sfaML1LoopCount == 2) {
                    sfaL0CBufferIndex++;
                }
            }
            SetFlag<HardEvent::MTE1_MTE2>(sfaMte21KvEvents[sfaKvBufferIndex]); // 反向同步, 表示L1已经被mte1消费完
        }
        // cL0BufIter已经不在使用
        if (sfaML1LoopCount == 1) {
            sfaL0CBufferIndex++;
        }
    }
    qpL1BufIter += sfaML1LoopCount;
}

#endif // SPARSE_FLASH_ATTENTION_SERVICE_CUBE_MLA_H
