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
 * \file allto_all_matmul_util.h
 * \brief
 */
#ifndef ALL_TO_ALL_MATMUL_UTIL
#define ALL_TO_ALL_MATMUL_UTIL

#include "allto_all_matmul_tiling.h"
#include "adv_api/hccl/hccl.h"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/tla_numeric_size.hpp"
#include <cstdint>

#define MC2_NON_QUANT 0
#define MC2_DYNAMIC_QUANT 1
#define MC2_STATIC_QUANT 2
#define MC2_UNKNOWN_QUANT 3

using namespace AscendC;

namespace {
constexpr static uint32_t BUFFER_NUM = 2U;
constexpr static uint32_t BLOCK_SIZE = 512U;
constexpr static int32_t MAX_BLOCK_COUNT = 2;
constexpr static int32_t FLAG_ZERO_IDX = 0;
constexpr static int32_t FLAG_ONE_IDX = 1;
constexpr static int32_t USED_UB_SIZE = 160 * 1024;
constexpr static int32_t FLAG_OFFSET = 180 * 1024 * 1024 / sizeof(int32_t);
constexpr static uint32_t UB_OFFSET = 97440; // 根据类型变动这个值
constexpr static uint32_t BLOCK_ALIGN_BYTES = 32U;
constexpr static uint32_t BLOCK_NUM_OF_UB_OFFSET = UB_OFFSET / BLOCK_ALIGN_BYTES;
constexpr static float MAX_INT8 = 127.0f;
constexpr static float MAX_INT4 = 7.0f;
} // namespace

template <typename T, size_t SIZE>
struct BaseBlock {
    static_assert((SIZE & (SIZE - 1)) == 0, "Invalid block size");
    static constexpr size_t size = Catlass::BytesToBits(SIZE) / Catlass::SizeOfBits<T>::value;

    static __aicore__ inline size_t Count(size_t len)
    {
        return (len + size - 1) / size;
    }

    static __aicore__ inline bool IsAligned(size_t len)
    {
        return len % size == 0;
    }

    static __aicore__ inline size_t AlignUp(size_t len)
    {
        return (len + size - 1) & ~(size - 1);
    }

    static __aicore__ inline size_t AlignDown(size_t len)
    {
        return len & ~(size - 1);
    }
};

template <typename T>
using Block32B = BaseBlock<T, 32>;

template <typename T>
using Block256B = BaseBlock<T, 256>;

template <typename T>
using Block512B = BaseBlock<T, 512>;

class CommBase {
public:
    __aicore__ explicit CommBase(){};

    template <typename T>
    __aicore__ inline void SetArgs(int32_t rank, int32_t rankSize, AlltoAllMatmulTilingData info)
    {
        blockIdx = GetBlockIdx();
        blockNum = GetBlockNum();
        aivIdx = GetSubBlockIdx();
        aicIdx = blockIdx / GetTaskRation();
        this->rank = rank;
        this->rankSize = rankSize;

        auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
        winContext_ = (__gm__ HcclCombineOpParam *)contextGM0;

        for (int i = 0; i < rankSize; i++) {
            buff[i] = (GM_ADDR)winContext_->windowsIn[i];
        }

        SetTiling(info);

        mPerLoop = m0 * pValue;
        tokenSize = k * rankSize;

        x1DataSize = 1LL * m * k;                    // 矩阵A大小
        allToAllSizePerRank = x1DataSize / rankSize; // 搬运到每个rank的数据量
        allToAllSizePerRankPerLoop = mPerLoop * k;   // 一次通信搬运到每个rank的数据量
        allToAllSizeAllRanksPerLoop = allToAllSizePerRankPerLoop * rankSize;     // 一次通信搬运的数据量
        pingPongBlockSize = static_cast<int64_t>(mPerLoop) * tokenSize;          // pingpong缓冲区大小
        allToAllSizePerCore = allToAllSizeAllRanksPerLoop / allToAllSendCoreNum; // 每个core搬运的数据量
        coreNumPerRank = allToAllSendCoreNum / rankSize; // 每个rank的用来搬运的core数量

        commCount = static_cast<int32_t>((x1DataSize + allToAllSizeAllRanksPerLoop - 1) / allToAllSizeAllRanksPerLoop);
        usedCoreNum = allToAllSendCoreNum + allToAllRecvCoreNum; // 总共的core数量

        if ASCEND_IS_AIV {
            TPipe pipe;
            pipe.InitBuffer(uBuf_, USED_UB_SIZE);
            pipe.InitBuffer(uBufSync_, 128);
            pipe.Destroy();
        }
    }

    __aicore__ inline void SetTiling(AlltoAllMatmulTilingData &info)
    {
        m = info.allToAllMatmulInfo.M;
        k = info.allToAllMatmulInfo.K;
        n = info.allToAllMatmulInfo.N;
        m0 = info.cocTiling.m0;
        k0 = info.cocTiling.k0;
        n0 = info.cocTiling.n0;
        quantSize = info.allToAllMatmulInfo.quantSize;
        dequantSize = info.allToAllMatmulInfo.dequantSize;
        quantScaleSize = info.allToAllMatmulInfo.quantScaleSize;
        isSegmentK = info.allToAllMatmulInfo.isSegmentK;
        isAlltoallOut = info.allToAllMatmulInfo.isAlltoallOut;
        isSmoothQuant = info.allToAllMatmulInfo.isSmoothQuant;
        copyTimes = info.allToAllMatmulInfo.segmentsNum;
        copyTensorSize = info.allToAllMatmulInfo.copyTensorSize;
        allToAllSendCoreNum = info.cocTiling.allToAllSendCoreNum;
        allToAllRecvCoreNum = info.cocTiling.allToAllRecvCoreNum;
        swizzlCount = info.cocTiling.swizzlCount;
        swizzlDirect = info.cocTiling.swizzlDirect;
        pValue = info.cocTiling.pValue;

        ubPingPongSize = info.cocTiling.ubMoveNum / 2;
        quantCoreNum = info.allToAllMatmulInfo.quantCoreNum;
    }

    template <typename T>
    __aicore__ inline void CopyGmToUbufAlignB16(LocalTensor<T> ubTensor, __gm__ T *src, uint16_t nBurst,
                                                uint32_t lenBurst, uint16_t srcStride, uint16_t dstStride)
    {
        DataCopyExtParams dataCopyParams(nBurst,    // blockCount
                                         lenBurst,  // blockLen
                                         srcStride, // srcStride
                                         dstStride, // dstStride
                                         0);
        GlobalTensor<T> gmTensor;
        gmTensor.SetGlobalBuffer(src);
        DataCopyPadExtParams<T> padParams;
        DataCopyPad(ubTensor, gmTensor, dataCopyParams, padParams);
    }

    template <typename T>
    __aicore__ inline void CopyUbufToGmAlignB16(__gm__ T *dst, LocalTensor<T> ubTensor, uint16_t nBurst,
                                                uint32_t lenBurst, uint16_t srcStride, uint16_t dstStride)
    {
        DataCopyExtParams dataCopyParams(nBurst,    // blockCount
                                         lenBurst,  // blockLen
                                         srcStride, // srcStride
                                         dstStride, // dstStride
                                         0);
        GlobalTensor<T> gmTensor;
        gmTensor.SetGlobalBuffer(dst);
        DataCopyPad(gmTensor, ubTensor, dataCopyParams);
    }

    __aicore__ inline void CheckBuffFlag(__gm__ int32_t *buff, int32_t flag)
    {
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        LocalTensor<int32_t> ubTensor = uBufSync_.AllocTensor<int32_t>();
        while (true) {
            CopyGmToUbufAlignB16(ubTensor, buff, 1, sizeof(int32_t), 0, 0);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID3);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID3); // Scalar等MTE2
            if (ubTensor(0) == flag) {
                break;
            }
        }
        uBufSync_.FreeTensor<int32_t>(ubTensor);
    }

    __aicore__ inline void SetBuffFlag(__gm__ int32_t *buff, int32_t flag)
    {
        SetFlag<HardEvent::MTE3_S>(EVENT_ID2);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID2);
        LocalTensor<int32_t> ubTensor = uBufSync_.AllocTensor<int32_t>();
        ubTensor(0) = flag;
        SetFlag<HardEvent::S_MTE3>(EVENT_ID2);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID2);
        CopyUbufToGmAlignB16(buff, ubTensor, 1, sizeof(int32_t), 0, 0);
    }

    __aicore__ inline void CopyTokensFromGMToGM(__gm__ int8_t *gmSrc, __gm__ int8_t *gmDst, uint32_t copyBytes,
                                                uint32_t srcKBytes, uint32_t dstKBytes)
    {
        LocalTensor<int8_t> ubTensor = uBuf_.AllocTensor<int8_t>();
        LocalTensor<int8_t> copyTensor0 = ubTensor;
        LocalTensor<int8_t> copyTensor1 = ubTensor[ub_offset];

        uint32_t tokenNum = copyBytes / srcKBytes;
        uint32_t blockPerToken = DivCeil(srcKBytes, BLOCK_ALIGN_BYTES);     // 每个token占用多少格子
        uint32_t copyTokenPerTime = BLOCK_NUM_OF_UB_OFFSET / blockPerToken; // 一次搬运多少token
        uint32_t copyTimes = DivCeil(tokenNum, copyTokenPerTime);
        uint32_t actualCopyTokenPerTime = copyTokenPerTime;

        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        for (int32_t copyIdx = 0; copyIdx < copyTimes; ++copyIdx) {
            if (copyIdx == copyTimes - 1) { // 最后一次可能不为copyTokenPerTime
                actualCopyTokenPerTime = tokenNum - (copyTimes - 1) * copyTokenPerTime;
            }
            auto eventId = (copyIdx & 1) ? EVENT_ID0 : EVENT_ID1;
            LocalTensor<int8_t> copyTensor = (copyIdx & 1) ? copyTensor0 : copyTensor1;
            WaitFlag<HardEvent::MTE3_MTE2>(eventId);
            CopyGmToUbufAlignB16(copyTensor, gmSrc, actualCopyTokenPerTime, srcKBytes, 0, 0);
            SetFlag<HardEvent::MTE2_MTE3>(eventId);
            WaitFlag<HardEvent::MTE2_MTE3>(eventId);
            CopyUbufToGmAlignB16(gmDst, copyTensor, actualCopyTokenPerTime, srcKBytes, 0, dstKBytes - srcKBytes);
            gmDst += dstKBytes * actualCopyTokenPerTime; // int_8占用一个字节，可直接作为地址偏移
            gmSrc += srcKBytes * actualCopyTokenPerTime;
            SetFlag<HardEvent::MTE3_MTE2>(eventId);
        }
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        uBuf_.FreeTensor<int8_t>(ubTensor);
    }

    __aicore__ inline void ResetIpcFlags(int32_t flagNum)
    {
        for (int32_t idx = 0; idx < flagNum; ++idx) {
            if (aicIdx == 0 && aivIdx == 0) {
                SetBuffFlag((__gm__ int32_t *)buff[rank] + FLAG_OFFSET + idx, 0);
            }
        }
    }

    __aicore__ inline void CrossRankSyncV1(int32_t flagIdx, int32_t flagVal)
    {
        if (aicIdx == 0 && aivIdx == 0) {
            SetBuffFlag((__gm__ int32_t *)buff[rank] + FLAG_OFFSET + flagIdx, flagVal);
        }
        if (aicIdx < rankSize && aivIdx == 0) {
            CheckBuffFlag((__gm__ int32_t *)buff[aicIdx] + FLAG_OFFSET + flagIdx, flagVal);
        }
    }

public:
    int32_t blockIdx;
    int32_t blockNum; // 取值为aiv数量
    int32_t aivIdx;   // 只能为0或1
    int32_t aicIdx;
    int32_t rank;
    int32_t rankSize;
    GM_ADDR buff[8];
    __gm__ HcclCombineOpParam *winContext_{nullptr};
    TBuf<AscendC::TPosition::VECCALC> uBuf_;
    TBuf<AscendC::TPosition::VECCALC> uBufSync_;

    uint64_t allToAllSizePerRank;
    uint32_t allToAllSizePerCore;
    int64_t pingPongBlockSize;
    int32_t mPerLoop;
    int32_t allToAllSizePerRankPerLoop;
    int32_t allToAllSizeAllRanksPerLoop;
    int32_t commCount;
    int32_t usedCoreNum;
    int32_t coreNumPerRank;
    int32_t tokenSize;
    int32_t allToAllSendCoreNum;
    int32_t allToAllRecvCoreNum;
    int32_t ub_offset;
    int32_t copyTimes;
    int32_t copyTensorSize;

    int32_t m0;
    int32_t k0;
    int32_t n0;
    int32_t swizzlCount;
    int32_t swizzlDirect;
    int32_t pValue;
    int32_t ubPingPongSize;

    uint32_t m;
    uint32_t k;
    uint32_t n;
    uint32_t kBytes;
    uint32_t tokenBytes;
    uint64_t quantSize;
    uint64_t dequantSize;
    uint64_t quantScaleSize;
    uint64_t x1DataSize;

    bool isSegmentK;
    bool isAlltoallOut;
    bool isSmoothQuant;
    uint32_t quantCoreNum;
};

__aicore__ inline void SetAndWaitAivSync(uint64_t flagIdx, int32_t pipeDepth = 2)
{
    AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(flagIdx + pipeDepth);
    AscendC::CrossCoreWaitFlag(flagIdx + pipeDepth);
}

__aicore__ inline void SetAicSync(uint64_t flagIdx)
{
    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagIdx);
}

template <typename T>
__aicore__ inline int64_t ElemNumToBytes(int64_t elemNum)
{
    return elemNum * sizeof(T);
}

// int4时，无法使用sizeof
template <>
__aicore__ inline int64_t ElemNumToBytes<AscendC::int4b_t>(int64_t elemNum)
{
    return elemNum / 2;
}
#endif
