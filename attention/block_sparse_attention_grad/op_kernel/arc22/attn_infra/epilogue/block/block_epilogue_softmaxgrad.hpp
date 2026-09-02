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
 * \file block_epliogue_softmaxgrad.h
 * \brief Block Epliogue Softmax Grad Kernel Implementation
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_SOFTAXGRAD_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_SOFTAXGRAD_HPP

#include "../../../attn_infra/arch/bsag_resource.hpp"
#include "../../../attn_infra/epilogue/bsag_epilogue_dispatch_policy.hpp"
#include "kernel_operator.h"

using namespace AscendC;

namespace NpuArch::Epilogue::Block {
template <typename InputDType, typename OutputDtype, uint32_t INPUT_LAYOUT>
class SoftmaxGrad {
public:
    using DispatchPolicy = EpilogueAtlasA2FAGPre;
    using ArchTag = typename DispatchPolicy::ArchTag;

    struct Params {
        // Data members
        GM_ADDR dout;
        GM_ADDR out;
        GM_ADDR actualQSeqlen;
        GM_ADDR softGradworkspace;
        GM_ADDR tilingData;

        // Methods
        __aicore__ inline Params() {}

        __aicore__ inline Params(GM_ADDR dout_, GM_ADDR out_, GM_ADDR actualQSeqlen_, GM_ADDR softGradworkspace_,
                                 GM_ADDR tilingData_
                                 // GM_ADDR doutWorkspace_,
                                 )
            : dout(dout_),
              out(out_),
              actualQSeqlen(actualQSeqlen_),
              softGradworkspace(softGradworkspace_),
              tilingData(tilingData_)
        {}
    };

    NpuArch::Arch::Resource<ArchTag> resource;

    constexpr static uint64_t BLOCK_BYTE_SIZE = 32;
    constexpr static uint64_t BLOCK_SIZE = 8; // 1个基本块（32字节） float元素个数
    constexpr static uint64_t SFMG_HIGH_PERF_N_FACTOR = 8;
    constexpr static uint64_t SFMG_HIGH_PERF_D_FACTOR = 64;
    constexpr static uint64_t STAGES = 1;
    constexpr static uint64_t DOUBLE_BUFFER = 2;
    constexpr static uint64_t INPUT_NUM = 2;
    constexpr static uint64_t BNSD = 1;
    constexpr static uint64_t TND = 0;

    uint64_t cBlockIdx;

    GlobalTensor<float> sfmgWorkspaceGm;
    GlobalTensor<InputDType> dyGm;
    GlobalTensor<InputDType> attenInGm;
    GM_ADDR actualSeqQlenAddr;

    LocalTensor<InputDType> doutTensor[STAGES];
    LocalTensor<InputDType> outTensor[STAGES];
    LocalTensor<float> doutFp32Tensor[STAGES];
    LocalTensor<float> outFp32Tensor[STAGES];
    LocalTensor<float> softmaxGradTensor[STAGES];
    LocalTensor<uint8_t> tempBuffer;

    uint64_t b = 0;
    uint64_t n1 = 0; // q_n
    uint64_t t1 = 0; // q_t
    uint64_t d = 0;  //
    uint64_t s1 = 0;
    uint64_t dAlign = 0; // 对齐的d

    uint64_t bIdx = 0; // batchIdx
    uint64_t nIdx = 0;
    uint64_t sIdx = 0;

    uint64_t dstOffset = 0;
    uint64_t transpseStride = 0;

    uint64_t usedCoreNums = 0;
    uint64_t normalCoreSize = 0;
    uint64_t tailCoreSize = 0;
    uint64_t singleLoopNBurstNum = 0;
    uint64_t normalCoreLoopTimes = 0;
    uint64_t normalCoreLastLoopNBurstNum = 0;
    uint64_t tailCoreLoopTimes = 0;
    uint64_t tailCoreLastLoopNBurstNum = 0;
    const BlockSparseAttentionGradTilingData *__restrict tilingData;

    __aicore__ inline SoftmaxGrad(Params const &params)
    {
        cBlockIdx = GetBlockIdx();
        GET_TILING_DATA_WITH_STRUCT(BlockSparseAttentionGradTilingData, tiling, params.tilingData);
        tilingData = &tiling;
        usedCoreNums = tilingData->usedVecCoreNum;
        if (cBlockIdx >= usedCoreNums) {
            return;
        }

        b = tilingData->batch;
        t1 = tilingData->qTotalSeqlen; // q_t
        n1 = tilingData->numHeads;     // q_n
        d = tilingData->headDim;
        s1 = tilingData->maxQSeqlen;

        uint64_t blockNums = BLOCK_BYTE_SIZE / sizeof(InputDType);
        dAlign = (d + blockNums - 1) / blockNums * blockNums;
        actualSeqQlenAddr = params.actualQSeqlen;

        // 计算 buffer 大小
        constexpr static uint64_t inputBufferLen = 24 * 1024;                    // castBuffer 24K*2=48K
        constexpr static uint64_t castBufferLen = 48 * 1024;                     // castBuffer 48K*2=96K
        uint64_t outputBufferLen = CeilDiv(castBufferLen, (uint64_t)dAlign) * 8; // 输出(s1,8)
        uint64_t tempBufferLen = 40 * 1024 - outputBufferLen;

        uint64_t offset = 0;
        uint64_t inputBufferLenEeachStage = inputBufferLen / STAGES; // 开启double buffer 后，每一个input的buffer len
        uint64_t castBufferLenEeachStage = castBufferLen / STAGES; // 开启double buffer 后，每一个case的buffer len
        uint64_t outBufferLenEeachStage = outputBufferLen / STAGES; // 开启double buffer 后，每一个case的buffer len
        for (uint64_t i = 0; i < STAGES; i++) {
            doutTensor[i] = resource.ubBuf.template GetBufferByByte<InputDType>(inputBufferLenEeachStage * i);
            outTensor[i] =
                resource.ubBuf.template GetBufferByByte<InputDType>(inputBufferLen + inputBufferLenEeachStage * i);
            doutFp32Tensor[i] = resource.ubBuf.template GetBufferByByte<float>(inputBufferLen * INPUT_NUM +
                                                                               castBufferLenEeachStage * i);
            outFp32Tensor[i] = resource.ubBuf.template GetBufferByByte<float>(
                inputBufferLen * INPUT_NUM + castBufferLen + castBufferLenEeachStage * i);
            softmaxGradTensor[i] = resource.ubBuf.template GetBufferByByte<float>(
                inputBufferLen * INPUT_NUM + castBufferLen * INPUT_NUM + outBufferLenEeachStage * i);
        }
        tempBuffer = resource.ubBuf.template GetBufferByByte<uint8_t>((inputBufferLen + castBufferLen) * INPUT_NUM +
                                                                      outputBufferLen);

        // 初始化 GM
        dyGm.SetGlobalBuffer((__gm__ InputDType *)params.dout);
        attenInGm.SetGlobalBuffer((__gm__ InputDType *)params.out);
        sfmgWorkspaceGm.SetGlobalBuffer((__gm__ float *)params.softGradworkspace);

        uint64_t normalAxisSize = 0;
        if (INPUT_LAYOUT == TND) {
            normalAxisSize = t1 * n1;
        } else {
            normalAxisSize = b * n1 * s1;
        }

        // 计算单核的计算量
        normalCoreSize = normalAxisSize / usedCoreNums;
        tailCoreSize = normalAxisSize - (usedCoreNums - 1) * normalCoreSize;

        // 计算单loop的计算量及loop次数
        singleLoopNBurstNum = inputBufferLenEeachStage / sizeof(InputDType) / dAlign; // 1次loop可以处理最大s行数
        normalCoreLoopTimes = CeilDiv(normalCoreSize, singleLoopNBurstNum);           // loop次数
        normalCoreLastLoopNBurstNum =
            normalCoreSize - (normalCoreLoopTimes - 1) * singleLoopNBurstNum; // 尾循环处理行数

        tailCoreLoopTimes = CeilDiv(tailCoreSize, singleLoopNBurstNum);
        tailCoreLastLoopNBurstNum = tailCoreSize - (tailCoreLoopTimes - 1) * singleLoopNBurstNum;

        if constexpr (INPUT_LAYOUT == TND) {
            transpseStride = (n1 * d - d) * sizeof(InputDType);
        } else if constexpr (INPUT_LAYOUT == BNSD) {
            transpseStride = 0;
        }
    }

    __aicore__ inline ~SoftmaxGrad() {}

    template <int32_t CORE_TYPE = g_coreType>
    __aicore__ inline void operator()(uint64_t startIdx, uint64_t singleCoreCount);

    template <>
    __aicore__ inline void operator()<AscendC::AIC>(uint64_t startIdx, uint64_t singleCoreCount)
    {}

    /*
     * brief: softmaxgrad execute
     *
     * startIdx : input 开始序号
     * singleCoreRowCount: 当前核需要处理的行数，BNSD格式，bns进行并轴；tnd, tn进行并轴
     */
    template <>
    __aicore__ inline void operator()<AscendC::AIV>(uint64_t startIdx, uint64_t singleCoreRowCount)
    {
        if (cBlockIdx >= usedCoreNums || singleCoreRowCount == 0) {
            return;
        }

        normalCoreLoopTimes = CeilDiv(singleCoreRowCount, singleLoopNBurstNum); // loop次数
        normalCoreLastLoopNBurstNum =
            singleCoreRowCount - (normalCoreLoopTimes - 1) * singleLoopNBurstNum; // 尾循环处理行数

        uint64_t singleCoreLoop = normalCoreLoopTimes;
        uint64_t singleCoreLastLoopNBurstNum = normalCoreLastLoopNBurstNum; // 普通单核最后一次loop处理多少个D

        uint64_t nBurst = singleLoopNBurstNum; // single loop nums
        uint64_t curS = s1;                    // tnd场景会在InitIndex中重新赋值
        uint32_t ping = 0;

        uint64_t layoutFlag = 1;
        if constexpr (INPUT_LAYOUT == TND) {
            layoutFlag = n1;
        }
        for (uint64_t i = 0; i < singleCoreLoop; i++) {
            if (i == singleCoreLoop - 1) {
                nBurst = singleCoreLastLoopNBurstNum;
            }
            auto eventId = ping ? EVENT_ID3 : EVENT_ID2;

            set_flag(PIPE_MTE3, PIPE_MTE2, eventId);
            wait_flag(PIPE_MTE3, PIPE_MTE2, eventId);

            // copyIn
            if (i == 0) {
                InitIndex((startIdx + i * singleLoopNBurstNum * layoutFlag) * d, curS, actualSeqQlenAddr);
                CopyInSfmg(nBurst, curS, actualSeqQlenAddr, ping);
            }

            set_flag(PIPE_MTE2, PIPE_V, eventId);
            wait_flag(PIPE_MTE2, PIPE_V, eventId);

            // cast 1
            uint64_t calcSize = nBurst * dAlign;
            Cast(doutFp32Tensor[ping], doutTensor[ping], RoundMode::CAST_NONE, calcSize);
            AscendC::PipeBarrier<PIPE_V>();

            // cast 2
            Cast(outFp32Tensor[ping], outTensor[ping], RoundMode::CAST_NONE, calcSize);
            AscendC::PipeBarrier<PIPE_V>();

            // pre copyIn next nBurst
            if (i < singleCoreLoop - 1) {
                set_flag(PIPE_V, PIPE_MTE2, eventId);
                wait_flag(PIPE_V, PIPE_MTE2, eventId);

                uint64_t nextNBurst = i == singleCoreLoop - 2 ? singleCoreLastLoopNBurstNum : nBurst;
                InitIndex((startIdx + (i + 1) * singleLoopNBurstNum * layoutFlag) * d, curS, actualSeqQlenAddr);
                CopyInSfmg(nextNBurst, curS, actualSeqQlenAddr, ping);
            }

            if (i != 0) {
                set_flag(PIPE_MTE3, PIPE_V, eventId);
                wait_flag(PIPE_MTE3, PIPE_V, eventId);
            }

            // sfmg
            Duplicate<float>(softmaxGradTensor[ping], 0.0, nBurst * 8);
            AscendC::PipeBarrier<PIPE_V>();

            uint32_t shapeArray[] = {static_cast<uint32_t>(nBurst), static_cast<uint32_t>(dAlign)};
            doutFp32Tensor[ping].SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
            outFp32Tensor[ping].SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
            uint32_t outShapeArray[] = {static_cast<uint32_t>(nBurst), BLOCK_BYTE_SIZE / sizeof(float)};
            softmaxGradTensor[ping].SetShapeInfo(ShapeInfo(2, outShapeArray, DataFormat::ND));

            bool isBasicBlock = (nBurst % SFMG_HIGH_PERF_N_FACTOR == 0) && (dAlign % SFMG_HIGH_PERF_D_FACTOR == 0);
            if (likely(isBasicBlock)) {
                SoftmaxGradFront<float, true>(softmaxGradTensor[ping], doutFp32Tensor[ping], outFp32Tensor[ping],
                                              tempBuffer, tilingData->softmaxGradTilingData);
            } else {
                SoftmaxGradFront<float, false>(softmaxGradTensor[ping], doutFp32Tensor[ping], outFp32Tensor[ping],
                                               tempBuffer, tilingData->softmaxGradTilingData);
            }
            AscendC::PipeBarrier<PIPE_V>();

            set_flag(PIPE_V, PIPE_MTE3, eventId);
            wait_flag(PIPE_V, PIPE_MTE3, eventId);

            // copyOut
            uint64_t sfmgOutputOffset = (startIdx + i * singleLoopNBurstNum * layoutFlag) * BLOCK_SIZE;
            if constexpr (INPUT_LAYOUT == TND) {
                DataCopyPad(sfmgWorkspaceGm[sfmgOutputOffset], softmaxGradTensor[ping],
                            {static_cast<uint16_t>(nBurst), static_cast<uint32_t>(BLOCK_SIZE * sizeof(float)), 0,
                             static_cast<uint32_t>((n1 - 1) * 8 * 4), 0});
            } else {
                DataCopy(sfmgWorkspaceGm[sfmgOutputOffset], softmaxGradTensor[ping], nBurst * BLOCK_SIZE);
            }

            if (STAGES == DOUBLE_BUFFER) {
                ping = 1 - ping;
            }
        }
    }

    /*
     * 根据当前 Core 的起始偏移量（startIdx），计算要处理的维度索引（bIdx/nIdx/sIdx）；
     *
     * startIdx : input 开始序号
     * curS: 当前batch的seqlen, 主要要针对tnd格式，s不等场景
     * seqS: actual seqlen list
     */
    __aicore__ inline void InitIndex(uint64_t startIdx, uint64_t &curS, GM_ADDR seqS)
    {
        if constexpr (INPUT_LAYOUT == TND) {
            uint64_t prefixSum = 0;
            for (int64_t bDimIdx = 0; bDimIdx < b; bDimIdx++) {
                uint64_t curBatchLen = ((__gm__ int64_t *)seqS)[bDimIdx];
                uint64_t curBatchElements = curBatchLen * n1 * d;

                if (startIdx < prefixSum + curBatchElements) {
                    bIdx = bDimIdx;
                    curS = curBatchLen;

                    uint64_t bTail = startIdx - prefixSum;
                    sIdx = bTail / (n1 * d);
                    uint64_t sTail = bTail % (n1 * d);
                    nIdx = sTail / d;
                    break;
                }
                prefixSum += curBatchElements;
            }
        } else {
            bIdx = startIdx / (n1 * s1 * d);
            int64_t bTail = startIdx % (n1 * s1 * d);
            nIdx = bTail / (s1 * d);
            int64_t nTail = bTail % (s1 * d);
            sIdx = nTail / d;
        }
    }

    __aicore__ inline void CopyInSfmg(uint64_t leftNburst, uint64_t &curS, GM_ADDR seqS, uint32_t ping)
    {
        uint64_t dstOffset = 0;
        while (leftNburst > 0) {
            uint64_t curNburst = 0;
            if (curS - sIdx < leftNburst) { // 需要借N或借B
                curNburst = curS - sIdx;
                DoCopyIn(curS, curNburst, dstOffset, seqS, ping);
                leftNburst = leftNburst - curNburst;
                sIdx = 0;
                if (nIdx < n1 - 1) { // 需要借N
                    nIdx += 1;
                } else {
                    nIdx = 0;
                    if (bIdx < b - 1) { // 需要借B
                        bIdx += 1;
                        if constexpr (INPUT_LAYOUT == TND) {
                            curS = ((__gm__ uint64_t *)seqS)[bIdx];
                        } else {
                            curS = s1;
                        }
                    } else { // 没有轴可以借了，end
                        leftNburst = 0;
                    }
                }
            } else { // 当前S够用
                curNburst = leftNburst;
                DoCopyIn(curS, curNburst, dstOffset, seqS, ping);
                sIdx = sIdx + leftNburst;
                leftNburst = 0;
            }
            dstOffset = dstOffset + curNburst * dAlign;
        }
    }

    __aicore__ inline void DoCopyIn(uint64_t curS, uint64_t curNBurst, uint64_t dstOffset, GM_ADDR seqS, uint32_t ping)
    {
        uint64_t srcOffset = 0;
        if constexpr (INPUT_LAYOUT == TND) {
            uint64_t prefixSum = 0;
            for (uint64_t b = 0; b < bIdx; b++) {
                prefixSum += ((__gm__ uint64_t *)seqS)[b];
            }
            uint64_t bOffset = prefixSum * n1 * d;
            srcOffset = bOffset + (sIdx * n1 + nIdx) * d;
        } else {
            if constexpr (INPUT_LAYOUT == BNSD) {
                srcOffset = bIdx * (n1 * s1 * d) + nIdx * (s1 * d) + sIdx * d;
            }
        }
        DataCopyPad(doutTensor[ping][dstOffset], dyGm[srcOffset],
                    {static_cast<uint16_t>(curNBurst), static_cast<uint32_t>(d * sizeof(InputDType)),
                     static_cast<uint32_t>(transpseStride), 0, 0},
                    {true, 0, static_cast<uint8_t>((dAlign - d)), 0});
        DataCopyPad(outTensor[ping][dstOffset], attenInGm[srcOffset],
                    {static_cast<uint16_t>(curNBurst), static_cast<uint32_t>(d * sizeof(InputDType)),
                     static_cast<uint32_t>(transpseStride), 0, 0},
                    {true, 0, static_cast<uint8_t>((dAlign - d)), 0});
    }
};

} // namespace NpuArch::Epilogue::Block

#endif // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_SOFTAXGRAD_HPP
