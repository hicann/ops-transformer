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
 * \file dense_lightning_indexer_softmax_lse_service_cube_arch35.h
 * \brief A5架构适配的Cube服务头文件
 */

#ifndef __DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_SERVICE_CUBE_H___ARCH35
#define __DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_SERVICE_CUBE_H___ARCH35

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../dense_lightning_indexer_softmax_lse_common.h"

namespace DenseLISoftmaxLseKernel {
using namespace DenseLISoftmaxLseCommon;
using namespace AscendC;
template <typename T>
class DenseLISoftmaxLseMatmul {
public:
    using Q_T = typename T::queryType;
    using K_T = typename T::keyType;

    __aicore__ inline DenseLISoftmaxLseMatmul(){};
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitMm1GlobalTensor(const GlobalTensor<K_T> &keyGm, const GlobalTensor<Q_T> &queryGm,
                                               const GlobalTensor<float> &mm1ResGm);
    __aicore__ inline void InitParams(const ConstInfo &constInfo);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void ComputeMm1(const DenseLISoftmaxLseCommon::RunInfo &runInfo);

    static constexpr uint64_t KEY_BUF_NUM = 3;
    static constexpr uint64_t QUERY_BUF_NUM = 2;
    static constexpr uint64_t L0_BUF_NUM = 2;

    static constexpr uint32_t KEY_MTE1_MTE2_EVENT = EVENT_ID2;
    static constexpr uint32_t QUERY_MTE1_MTE2_EVENT = EVENT_ID5;
    static constexpr uint32_t M_MTE1_EVENT = EVENT_ID3;

    static constexpr uint32_t MTE2_MTE1_EVENT = EVENT_ID2;
    static constexpr uint32_t MTE1_M_EVENT = EVENT_ID2;

    static constexpr uint64_t M_BASIC_BLOCK = 256;
    static constexpr uint64_t D_BASIC_BLOCK = 128;
    static constexpr uint64_t S2_BASIC_BLOCK = 256;

    static constexpr uint64_t M_BASIC_BLOCK_L0 = 128;
    static constexpr uint64_t D_BASIC_BLOCK_L0 = 128;
    static constexpr uint64_t S2_BASIC_BLOCK_L0 = 128;

    static constexpr uint64_t QUERY_BUFFER_OFFSET = M_BASIC_BLOCK * D_BASIC_BLOCK;
    static constexpr uint64_t KEY_BUFFER_OFFSET = S2_BASIC_BLOCK * D_BASIC_BLOCK;
    static constexpr uint64_t L0AB_BUFFER_OFFSET = M_BASIC_BLOCK_L0 * D_BASIC_BLOCK_L0;
    static constexpr uint64_t L0C_BUFFER_OFFSET = M_BASIC_BLOCK_L0 * S2_BASIC_BLOCK_L0;

protected:
    __aicore__ inline void Fixp(uint64_t s1gGmOffset, uint64_t s2GmOffset, uint64_t s1gL0RealSize,
                                uint64_t s2L0RealSize, const DenseLISoftmaxLseCommon::RunInfo &runInfo);
    __aicore__ inline void ComuteL0c(uint64_t s1gL0RealSize, uint64_t s2L0RealSize,
                                     const DenseLISoftmaxLseCommon::RunInfo &runInfo);
    __aicore__ inline void LoadKeyToL0b(uint64_t s2L1Offset, uint64_t s2L1RealSize, uint64_t s2L0RealSize,
                                        const DenseLISoftmaxLseCommon::RunInfo &runInfo);
    __aicore__ inline void LoadQueryToL0a(uint64_t s1gL1Offset, uint64_t s1gL1RealSize, uint64_t s1gL0RealSize,
                                          const DenseLISoftmaxLseCommon::RunInfo &runInfo);
    __aicore__ inline void QueryNd2Nz(uint64_t s1gL1RealSize, uint64_t s1gL1Offset,
                                      const DenseLISoftmaxLseCommon::RunInfo &runInfo);
    __aicore__ inline void KeyNd2Nz(uint64_t s2L1RealSize, uint64_t s2GmOffset,
                                    const DenseLISoftmaxLseCommon::RunInfo &runInfo);
    GlobalTensor<K_T> keyGm_;
    GlobalTensor<Q_T> queryGm_;
    GlobalTensor<float> mm1ResGm_;

    TBuf<TPosition::A1> bufQL1_;
    LocalTensor<Q_T> queryL1_;
    TBuf<TPosition::B1> bufKeyL1_;
    LocalTensor<K_T> keyL1_;

    TBuf<TPosition::A2> bufQL0_;
    LocalTensor<Q_T> queryL0_;
    TBuf<TPosition::B2> bufKeyL0_;
    LocalTensor<K_T> keyL0_;

    TBuf<TPosition::CO1> bufL0C_;
    LocalTensor<float> cL0_;

    uint64_t keyL1BufIdx_ = 0;
    uint64_t queryL1Mte2BufIdx_ = 0;
    uint64_t queryL1Mte1BufIdx_ = 0;
    uint64_t l0BufIdx_ = 0;

    ConstInfo constInfo_;
};

template <typename T>
__aicore__ inline void DenseLISoftmaxLseMatmul<T>::InitParams(const ConstInfo &constInfo)
{
    constInfo_ = constInfo;
}

template <typename T>
__aicore__ inline void DenseLISoftmaxLseMatmul<T>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(bufQL1_, QUERY_BUF_NUM * M_BASIC_BLOCK * D_BASIC_BLOCK * sizeof(Q_T));
    queryL1_ = bufQL1_.Get<Q_T>();
    pipe->InitBuffer(bufKeyL1_, KEY_BUF_NUM * S2_BASIC_BLOCK * D_BASIC_BLOCK * sizeof(K_T));
    keyL1_ = bufKeyL1_.Get<K_T>();

    pipe->InitBuffer(bufQL0_, L0_BUF_NUM * M_BASIC_BLOCK_L0 * D_BASIC_BLOCK_L0 * sizeof(Q_T));
    queryL0_ = bufQL0_.Get<Q_T>();
    pipe->InitBuffer(bufKeyL0_, L0_BUF_NUM * D_BASIC_BLOCK_L0 * S2_BASIC_BLOCK_L0 * sizeof(K_T));
    keyL0_ = bufKeyL0_.Get<K_T>();

    pipe->InitBuffer(bufL0C_, L0_BUF_NUM * M_BASIC_BLOCK_L0 * S2_BASIC_BLOCK_L0 * sizeof(float));
    cL0_ = bufL0C_.Get<float>();
}

template <typename T>
__aicore__ inline void DenseLISoftmaxLseMatmul<T>::InitMm1GlobalTensor(const GlobalTensor<K_T> &keyGm,
                                                                       const GlobalTensor<Q_T> &queryGm,
                                                                       const GlobalTensor<float> &mm1ResGm)
{
    keyGm_ = keyGm;
    queryGm_ = queryGm;
    mm1ResGm_ = mm1ResGm;
}

template <typename T>
__aicore__ inline void DenseLISoftmaxLseMatmul<T>::ComputeMm1(const DenseLISoftmaxLseCommon::RunInfo &runInfo)
{
    uint64_t s2GmBaseOffset = runInfo.s2Idx * constInfo_.s2BaseSize;
    uint64_t s1gProcessSize = runInfo.actMBaseSize;
    uint64_t s2ProcessSize = runInfo.actualSingleProcessSInnerSize;
    for (uint64_t s2GmOffset = 0; s2GmOffset < s2ProcessSize; s2GmOffset += S2_BASIC_BLOCK) {
        WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + keyL1BufIdx_ % KEY_BUF_NUM);
        uint64_t s2L1RealSize =
            s2GmOffset + S2_BASIC_BLOCK > s2ProcessSize ? s2ProcessSize - s2GmOffset : S2_BASIC_BLOCK;
        KeyNd2Nz(s2L1RealSize, s2GmOffset, runInfo);
        SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
        WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
        for (uint64_t s1gGmOffset = 0; s1gGmOffset < s1gProcessSize; s1gGmOffset += M_BASIC_BLOCK) {
            uint64_t s1gL1RealSize =
                s1gGmOffset + M_BASIC_BLOCK > s1gProcessSize ? s1gProcessSize - s1gGmOffset : M_BASIC_BLOCK;
            if (runInfo.isFirstS2InnerLoop && s2GmOffset == 0) {
                queryL1Mte2BufIdx_++;
                queryL1Mte1BufIdx_ = queryL1Mte2BufIdx_;
                WaitFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + queryL1Mte2BufIdx_ % QUERY_BUF_NUM);
                QueryNd2Nz(s1gL1RealSize, s1gGmOffset, runInfo);
                SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
                WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
            } else {
                queryL1Mte1BufIdx_ =
                    queryL1Mte2BufIdx_ - (CeilDiv(s1gProcessSize, M_BASIC_BLOCK) - 1 - (s1gGmOffset > 0));
            }
            for (uint64_t s2L1Offset = 0; s2L1Offset < s2L1RealSize; s2L1Offset += S2_BASIC_BLOCK_L0) {
                uint64_t s2L0RealSize =
                    s2L1Offset + S2_BASIC_BLOCK_L0 > s2L1RealSize ? s2L1RealSize - s2L1Offset : S2_BASIC_BLOCK_L0;
                for (uint64_t s1gL1Offset = 0; s1gL1Offset < s1gL1RealSize; s1gL1Offset += M_BASIC_BLOCK_L0) {
                    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0BufIdx_ % L0_BUF_NUM);
                    uint64_t s1gL0RealSize =
                        s1gL1Offset + M_BASIC_BLOCK_L0 > s1gL1RealSize ? s1gL1RealSize - s1gL1Offset : M_BASIC_BLOCK_L0;
                    LoadQueryToL0a(s1gL1Offset, s1gL1RealSize, s1gL0RealSize, runInfo);
                    LoadKeyToL0b(s2L1Offset, s2L1RealSize, s2L0RealSize, runInfo);

                    SetFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
                    WaitFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
                    ComuteL0c(s1gL0RealSize, s2L0RealSize, runInfo);

                    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0BufIdx_ % L0_BUF_NUM);
                    Fixp(s1gGmOffset + s1gL1Offset, s2GmOffset + s2L1Offset, s1gL0RealSize, s2L0RealSize, runInfo);
                    l0BufIdx_++;
                }
            }
            if (s2GmOffset + S2_BASIC_BLOCK >= s2ProcessSize && runInfo.isLastS2InnerLoop) {
                SetFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + queryL1Mte1BufIdx_ % QUERY_BUF_NUM);
            }
        }

        SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + keyL1BufIdx_ % KEY_BUF_NUM);
        keyL1BufIdx_++;
    }
}

template <typename T>
__aicore__ inline void DenseLISoftmaxLseMatmul<T>::KeyNd2Nz(uint64_t s2L1RealSize, uint64_t s2GmOffset,
                                                            const DenseLISoftmaxLseCommon::RunInfo &runInfo)
{
    uint64_t s2L1Offset = 0;
    while (s2L1Offset < s2L1RealSize) {
        uint64_t keyGmOffset = runInfo.tensorKeyOffset + (s2GmOffset + s2L1Offset) * constInfo_.headDim;
        // 搬运按照S2_BASIC_BLOCK_L0*D_BASIC_BLOCK_L0的方式在l1上排布, 方便后续mte1
        // 根据s2的offset判断当前属于前一个L0分型还是后一个L0分型，暂时只支持两个分型
        uint64_t s2Mte2Size = (s2L1RealSize <= S2_BASIC_BLOCK_L0 || s2L1Offset >= S2_BASIC_BLOCK_L0) ?
                                  s2L1RealSize - s2L1Offset :
                                  S2_BASIC_BLOCK_L0 - s2L1Offset;

        Nd2NzParams nd2nzPara;
        nd2nzPara.ndNum = 1;
        nd2nzPara.nValue = s2Mte2Size; // 行数
        nd2nzPara.dValue = constInfo_.headDim;
        nd2nzPara.srcDValue = constInfo_.headDim;
        nd2nzPara.dstNzC0Stride = s2L1Offset >= S2_BASIC_BLOCK_L0 ?
                                      CeilAlign(s2L1RealSize - S2_BASIC_BLOCK_L0, (uint64_t)BLOCK_CUBE) :
                                      (s2L1RealSize > S2_BASIC_BLOCK_L0 ?
                                           S2_BASIC_BLOCK_L0 :
                                           CeilAlign(s2L1RealSize, (uint64_t)BLOCK_CUBE)); // 对齐到16 单位block
        nd2nzPara.dstNzNStride = 1;
        nd2nzPara.srcNdMatrixStride = 0;
        nd2nzPara.dstNzMatrixStride = 0;
        DataCopy(keyL1_[(keyL1BufIdx_ % KEY_BUF_NUM) * KEY_BUFFER_OFFSET +
                        (s2L1Offset >= S2_BASIC_BLOCK_L0 ?
                             S2_BASIC_BLOCK_L0 * D_BASIC_BLOCK_L0 + (s2L1Offset - S2_BASIC_BLOCK_L0) * BLOCK_CUBE :
                             s2L1Offset * BLOCK_CUBE)],
                 keyGm_[keyGmOffset], nd2nzPara);
        s2L1Offset += s2Mte2Size;
    }
}

// batch, s1, n2, g, d
template <typename T>
__aicore__ inline void DenseLISoftmaxLseMatmul<T>::QueryNd2Nz(uint64_t s1gL1RealSize, uint64_t s1gGmOffset,
                                                              const DenseLISoftmaxLseCommon::RunInfo &runInfo)
{
    uint64_t s1gL1Offset = 0;
    while (s1gL1Offset < s1gL1RealSize) {
        uint64_t queryGmOffset = runInfo.tensorQueryOffset + (s1gGmOffset + s1gL1Offset) * constInfo_.headDim;
        // M_BASIC_BLOCK_L0*D_BASIC_BLOCK_L0的方式在l1上排布, 方便后续mte1
        // 根据s2的offset判断当前属于前一个L0分型还是后一个L0分型，暂时只支持两个分型
        uint64_t s1Mte2Size = (s1gL1RealSize <= M_BASIC_BLOCK_L0 || s1gL1Offset >= M_BASIC_BLOCK_L0) ?
                                  s1gL1RealSize - s1gL1Offset :
                                  M_BASIC_BLOCK_L0 - s1gL1Offset;

        Nd2NzParams nd2nzPara;
        nd2nzPara.ndNum = 1;
        nd2nzPara.nValue = s1Mte2Size; // 行数
        nd2nzPara.dValue = constInfo_.headDim;
        nd2nzPara.srcDValue = constInfo_.headDim;
        nd2nzPara.dstNzC0Stride = s1gL1Offset >= M_BASIC_BLOCK_L0 ?
                                      CeilAlign(s1gL1RealSize - M_BASIC_BLOCK_L0, (uint64_t)BLOCK_CUBE) :
                                      (s1gL1RealSize > M_BASIC_BLOCK_L0 ?
                                           M_BASIC_BLOCK_L0 :
                                           CeilAlign(s1gL1RealSize, (uint64_t)BLOCK_CUBE)); // 对齐到16 单位block
        nd2nzPara.dstNzNStride = 1;
        nd2nzPara.srcNdMatrixStride = 0;
        nd2nzPara.dstNzMatrixStride = 0;
        DataCopy(queryL1_[(queryL1Mte2BufIdx_ % QUERY_BUF_NUM) * QUERY_BUFFER_OFFSET +
                          (s1gL1Offset >= M_BASIC_BLOCK_L0 ?
                               M_BASIC_BLOCK_L0 * D_BASIC_BLOCK_L0 + (s1gL1Offset - M_BASIC_BLOCK_L0) * BLOCK_CUBE :
                               s1gL1Offset * BLOCK_CUBE)],
                 queryGm_[queryGmOffset], nd2nzPara);
        s1gL1Offset += s1Mte2Size;
    }
}

template <typename T>
__aicore__ inline void DenseLISoftmaxLseMatmul<T>::LoadQueryToL0a(uint64_t s1gL1Offset, uint64_t s1gL1RealSize,
                                                                  uint64_t s1gL0RealSize,
                                                                  const DenseLISoftmaxLseCommon::RunInfo &runInfo)
{
    uint64_t keyL1Offset = s1gL1Offset >= M_BASIC_BLOCK_L0 ? M_BASIC_BLOCK_L0 * D_BASIC_BLOCK_L0 : 0;
    LoadData2DParams loadData2DParams;
    loadData2DParams.startIndex = 0;
    loadData2DParams.repeatTimes = CeilDiv(s1gL0RealSize, BLOCK_CUBE) * CeilDiv(constInfo_.headDim, BLOCK_CUBE);
    loadData2DParams.srcStride = 1;
    loadData2DParams.dstGap = 0;
    loadData2DParams.ifTranspose = false;
    LoadData(queryL0_[(l0BufIdx_ % L0_BUF_NUM) * L0AB_BUFFER_OFFSET],
             queryL1_[(queryL1Mte1BufIdx_ % QUERY_BUF_NUM) * QUERY_BUFFER_OFFSET + keyL1Offset], loadData2DParams);
}

template <typename T>
__aicore__ inline void DenseLISoftmaxLseMatmul<T>::LoadKeyToL0b(uint64_t s2L1Offset, uint64_t s2L1RealSize,
                                                                uint64_t s2L0RealSize,
                                                                const DenseLISoftmaxLseCommon::RunInfo &runInfo)
{
    uint64_t keyL1Offset = s2L1Offset >= S2_BASIC_BLOCK_L0 ? S2_BASIC_BLOCK_L0 * D_BASIC_BLOCK_L0 : 0;
    LoadData2DParams loadData2DParams;
    loadData2DParams.startIndex = 0;
    loadData2DParams.repeatTimes = CeilDiv(s2L0RealSize, BLOCK_CUBE) * CeilDiv(constInfo_.headDim, BLOCK_CUBE);
    loadData2DParams.srcStride = 1;
    loadData2DParams.dstGap = 0;
    loadData2DParams.ifTranspose = false;
    LoadData(keyL0_[(l0BufIdx_ % L0_BUF_NUM) * L0AB_BUFFER_OFFSET],
             keyL1_[(keyL1BufIdx_ % KEY_BUF_NUM) * KEY_BUFFER_OFFSET + keyL1Offset], loadData2DParams);
}

template <typename T>
__aicore__ inline void DenseLISoftmaxLseMatmul<T>::ComuteL0c(uint64_t s1gL0RealSize, uint64_t s2L0RealSize,
                                                             const DenseLISoftmaxLseCommon::RunInfo &runInfo)
{
    MmadParams mmadParams;
    mmadParams.m = CeilAlign(s1gL0RealSize, BLOCK_CUBE);
    mmadParams.n = s2L0RealSize;
    mmadParams.k = constInfo_.headDim;
    mmadParams.cmatrixInitVal = true;
    mmadParams.cmatrixSource = false;
    mmadParams.unitFlag = 0b11;
    Mmad(cL0_[(l0BufIdx_ % L0_BUF_NUM) * L0C_BUFFER_OFFSET], queryL0_[(l0BufIdx_ % L0_BUF_NUM) * L0AB_BUFFER_OFFSET],
         keyL0_[(l0BufIdx_ % L0_BUF_NUM) * L0AB_BUFFER_OFFSET], mmadParams);
    if ((mmadParams.m / 16) * (mmadParams.n / 16) < 10) {
        PipeBarrier<PIPE_M>();
    }
}

template <typename T>
__aicore__ inline void DenseLISoftmaxLseMatmul<T>::Fixp(uint64_t s1gGmOffset, uint64_t s2GmOffset,
                                                        uint64_t s1gL0RealSize, uint64_t s2L0RealSize,
                                                        const DenseLISoftmaxLseCommon::RunInfo &runInfo)
{
    AscendC::DataCopyCO12DstParams intriParams;
    intriParams.mSize = CeilAlign(s1gL0RealSize, BLOCK_CUBE);
    intriParams.nSize = s2L0RealSize;
    intriParams.dstStride = runInfo.actualSingleProcessSInnerSizeAlign;
    intriParams.srcStride = CeilAlign(s1gL0RealSize, BLOCK_CUBE);
    // set mode according to dtype
    intriParams.quantPre = QuantMode_t::NoQuant;
    intriParams.nz2ndEn = true;
    intriParams.unitFlag = 0b11; // 3 unitflag
    intriParams.reluPre = 1;
    AscendC::SetFixpipeNz2ndFlag(1, 1, 1);
    AscendC::DataCopy(mm1ResGm_[(runInfo.loop % 2) * constInfo_.mBaseSize * constInfo_.s2BaseSize +
                                s1gGmOffset * intriParams.dstStride + s2GmOffset],
                      cL0_[(l0BufIdx_ % L0_BUF_NUM) * L0C_BUFFER_OFFSET], intriParams);
}

template <typename T>
__aicore__ inline void DenseLISoftmaxLseMatmul<T>::AllocEventID()
{
    SetMMLayoutTransform(true);
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 0);
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 1);
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 2);

    SetFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + 0);
    SetFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + 1);

    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 0);
    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 1);
}

template <typename T>
__aicore__ inline void DenseLISoftmaxLseMatmul<T>::FreeEventID()
{
    SetMMLayoutTransform(false);
    WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 0);
    WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 1);
    WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 2);

    WaitFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + 0);
    WaitFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + 1);

    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 0);
    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 1);
}
} // namespace DenseLISoftmaxLseKernel
#endif // DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_SERVICE_CUBE_H
