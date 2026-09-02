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
 * \file block_epliogue_simply_softmax.h
 * \brief Block Epliogue Simply Softmax Kernel Implementation
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_SIMPLY_SOFTMAX_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_SIMPLY_SOFTMAX_HPP

#include "../../../attn_infra/arch/bsag_resource.hpp"
#include "../../../attn_infra/epilogue/bsag_epilogue_dispatch_policy.hpp"
#include "kernel_operator.h"

using namespace AscendC;

template <typename InDtype>
struct SimplySoftMaxInfo {
    LocalTensor<float> sTensor;
    LocalTensor<float> lseTensor;
    LocalTensor<float> lseBrocTensor;
    LocalTensor<float> pFp32Tensor;
    LocalTensor<InDtype> pFp16Tensor;

    GlobalTensor<float> sGm;
    GlobalTensor<float> lseGm;
    GlobalTensor<InDtype> pGm;
};

template <typename InDtype>
struct CalDsInfo {
    LocalTensor<float> dpFp32Tensor;
    LocalTensor<float> softmaxGradTensor;
    LocalTensor<float> pFp32Tensor;
    LocalTensor<InDtype> dsFp16Tensor;

    GlobalTensor<float> dpGm;
    GlobalTensor<float> softmaxGradGm;
    GlobalTensor<InDtype> dsGm;
};

namespace NpuArch::Epilogue::Block {
template <typename InputDType, typename OutputDtype, uint32_t INPUT_LAYOUT>
class SimpltSoftmax {
public:
    using DispatchPolicy = EpilogueAtlasA2FAGPre;
    using ArchTag = typename DispatchPolicy::ArchTag;

    struct Params {
        // Data members
        GM_ADDR s;          // 连续
        GM_ADDR softmaxLse; // 需要跳着搬运
        GM_ADDR dp;         // 连续
        GM_ADDR actualQSeqlen;
        GM_ADDR actualKvSeqlen;
        GM_ADDR softGradworkspace; // 需要跳着搬运
        GM_ADDR pWorkspace;        // 连续
        GM_ADDR dsWorkspace;       // 连续
        GM_ADDR tilingData;
        uint64_t actualRow = 0;
        uint64_t actualCol = 0;
        uint64_t processNums = 0;
        uint64_t curCoreBatch = 0;
        uint64_t curCoreN1Idx = 0;
        uint64_t curCoreS1Idx = 0;
        uint64_t curT1Idx = 0;

        // Methods
        __aicore__ inline Params() {}

        __aicore__ inline Params(GM_ADDR s_, GM_ADDR softmaxLse_, GM_ADDR dp_, GM_ADDR actualQSeqlen_,
                                 GM_ADDR actualKvSeqlen_, GM_ADDR softGradworkspace_, GM_ADDR pWorkspace_,
                                 GM_ADDR dsWorkspace_, GM_ADDR tilingData_, uint64_t acutualRow_, uint64_t actualCol_,
                                 uint64_t processNums_, uint64_t curBatch_, uint64_t curN1_, uint64_t curS1_,
                                 uint64_t curT1_)
            : s(s_),
              softmaxLse(softmaxLse_),
              dp(dp_),
              actualQSeqlen(actualQSeqlen_),
              actualKvSeqlen(actualKvSeqlen_),
              softGradworkspace(softGradworkspace_),
              pWorkspace(pWorkspace_),
              dsWorkspace(dsWorkspace_),
              tilingData(tilingData_),
              actualRow(acutualRow_),
              actualCol(actualCol_),
              processNums(processNums_),
              curCoreBatch(curBatch_),
              curCoreN1Idx(curN1_),
              curCoreS1Idx(curS1_),
              curT1Idx(curT1_)
        {}
    };

    NpuArch::Arch::Resource<ArchTag> resource;

    constexpr static uint64_t STAGES = 1;
    constexpr static uint64_t INPUT_NUM = 2;
    constexpr static uint64_t DOUBLE_BUFFER = 2;
    constexpr static uint64_t BNSD = 1;
    constexpr static uint64_t TND = 0;
    constexpr static uint64_t proceeM = 128;
    constexpr static uint64_t proceeK = 128;
    constexpr static uint64_t BRCB_BASE_NUM = 8;
    constexpr static uint64_t REAPTE_BYTE = 256;

    constexpr static uint64_t BLOCK_BYTE_SIZE = 32;
    constexpr static uint64_t BLOCK_FP32_NUM = 8;
    constexpr static uint64_t BLOCK_16_NUM = 16;
    constexpr static uint64_t SFMG_HIGH_PERF_N_FACTOR = 8;
    constexpr static uint64_t SFMG_HIGH_PERF_D_FACTOR = 64;
    constexpr static uint64_t baseM = 64;

    uint64_t cBlockIdx = 0;
    uint64_t cubeCoreIdx = 0;
    uint64_t vecCoreIdx = 0;
    uint64_t row = 0; // 当前core需要处理q方向的s数
    uint64_t col = 0; // 当前core需要处理kv方向的s数
    uint64_t align32Col = 0;
    uint64_t align16Col = 0;
    uint64_t alignCol = 0;
    uint64_t alignRow = 0;
    uint64_t curCoreBatch = 0;
    uint64_t curCoreN1Idx = 0; // q_n
    uint64_t curT1Idx = 0;     // q_t
    uint64_t curCoreS1Idx = 0; // q_s
    uint64_t maxQSeqlen = 0;
    uint64_t maxKvSeqlen = 0;
    uint64_t n1 = 0; // q_n
    uint64_t transpseStride = 0;

    uint64_t usedVecCoreNums = 0;
    uint64_t p16BaseBufLen = 0;
    uint64_t p32BaseBufLen = 0;
    float scaleValue = 0.0f;

    GlobalTensor<float> sGm;                 // (N s1 s2)
    GlobalTensor<float> softmaxLseGm;        // (N s1 1)
    GlobalTensor<float> dpGm;                // (N s1 s2)
    GlobalTensor<InputDType> pWorkspaceGm;   // (N s1 s2)
    GlobalTensor<float> softGradworkspaceGm; // (N s1 8)
    GlobalTensor<InputDType> dsWorkspaceGm;  // (N s1 s2)

    GM_ADDR actualQSeqlen;
    GM_ADDR actualKvSeqlen;

    LocalTensor<float> sTensor[STAGES];
    LocalTensor<float> lseTensor[STAGES];
    LocalTensor<float> lseBrocTensor[STAGES];
    LocalTensor<float> pFp32Tensor[STAGES];
    LocalTensor<InputDType> p16Tensor[STAGES];
    LocalTensor<float> dpFp32Tensor[STAGES];
    LocalTensor<float> softmaxGradTensor[STAGES];
    LocalTensor<float> dsTensor[STAGES];
    LocalTensor<InputDType> ds16Tensor[STAGES];

    __aicore__ inline SimpltSoftmax()
    {
        // baseM =  128;
        // 分核 一个core 最大 128 * 128 一个vec 64 * 128
        uint64_t sBufferLen = baseM * 128 * sizeof(float);               // max
        uint64_t lBufferLen = baseM * sizeof(float);                     // max
        uint64_t lBrobBufferLen = BRCB_BASE_NUM * baseM * sizeof(float); // max
        uint64_t p32BufferLen = baseM * 128 * sizeof(float);
        uint64_t dpBufLen = sBufferLen;
        uint64_t p16BufLen = baseM * 128 * sizeof(InputDType);
        uint64_t ds16BufLen = p16BufLen;
        uint64_t dBufLen = BRCB_BASE_NUM * baseM * sizeof(float);
        p16BaseBufLen = p16BufLen;
        p32BaseBufLen = p32BufferLen;

        uint64_t sftBufferAlign = sBufferLen + p16BufLen + lBrobBufferLen + lBufferLen * BRCB_BASE_NUM;

        for (uint64_t i = 0; i < STAGES; i++) {
            // 第一轮 softmax 计算空间划分
            // 由于分核关系，所以的输入都能放入ub中
            // 空间示意图： S32(P32) || p16 || lseBroc || lse (align) || dpFp32(ds) || softmaxgrad || ds16
            sTensor[i] = resource.ubBuf.template GetBufferByByte<float>((sBufferLen / 2) * i);
            pFp32Tensor[i] = sTensor[i]; // 复用s
            p16Tensor[i] = resource.ubBuf.template GetBufferByByte<InputDType>(sBufferLen + (p16BufLen / 2) * i);
            lseBrocTensor[i] =
                resource.ubBuf.template GetBufferByByte<float>(sBufferLen + p16BufLen + (lBrobBufferLen / 2) * i);
            lseTensor[i] = resource.ubBuf.template GetBufferByByte<float>(sBufferLen + p16BufLen + lBrobBufferLen +
                                                                          (lBufferLen / 2) * i);

            // 第二轮 ds 计算空间划分
            dpFp32Tensor[i] = resource.ubBuf.template GetBufferByByte<float>(sftBufferAlign + (dpBufLen / 2) * i);
            softmaxGradTensor[i] =
                resource.ubBuf.template GetBufferByByte<float>(sftBufferAlign + dpBufLen + (dBufLen / 2) * i);
            dsTensor[i] = dpFp32Tensor[i];
            ds16Tensor[i] = resource.ubBuf.template GetBufferByByte<InputDType>(sftBufferAlign + dpBufLen + dBufLen +
                                                                                (p16BufLen / 2) * i);
        }
    }

    __aicore__ inline ~SimpltSoftmax() {}

    template <int32_t CORE_TYPE = g_coreType>
    __aicore__ inline void operator()(Params const &params);

    template <>
    __aicore__ inline void operator()<AscendC::AIC>(Params const &params)
    {}

    __aicore__ inline void Init(Params const &params)
    {
        vecCoreIdx = GetBlockIdx();
        cBlockIdx = vecCoreIdx;
        __gm__ BlockSparseAttentionGradTilingData *tilingData =
            reinterpret_cast<__gm__ BlockSparseAttentionGradTilingData *>(params.tilingData);
        usedVecCoreNums = tilingData->usedVecCoreNum;

        if (cBlockIdx >= usedVecCoreNums) {
            return;
        }

        maxQSeqlen = tilingData->maxQSeqlen;
        maxKvSeqlen = tilingData->maxKvSeqlen;
        n1 = tilingData->numHeads; // q_n
        col = params.actualCol;
        align32Col = (col + BLOCK_FP32_NUM - 1) / BLOCK_FP32_NUM * BLOCK_FP32_NUM; // fp32 对齐后的列数
        align16Col = (col + BLOCK_16_NUM - 1) / BLOCK_16_NUM * BLOCK_16_NUM;
        alignCol = (align32Col % BLOCK_16_NUM != 0) ? align16Col : align32Col;
        curCoreBatch = params.curCoreBatch;
        curCoreN1Idx = params.curCoreN1Idx;
        curT1Idx = params.curT1Idx;
        curCoreS1Idx = params.curCoreS1Idx;
        actualQSeqlen = params.actualQSeqlen;
        actualKvSeqlen = params.actualKvSeqlen;
        scaleValue = tilingData->scaleValue;

        uint64_t s2 = ((__gm__ uint64_t *)actualKvSeqlen)[curCoreBatch];
        row = params.actualRow;
        if (row <= 0) {
            return;
        }

        if constexpr (INPUT_LAYOUT == TND) {
            transpseStride = (n1 * 1 - 1) * sizeof(float);
        }

        // 初始化 GM
        sGm.SetGlobalBuffer((__gm__ float *)params.s);
        softmaxLseGm.SetGlobalBuffer((__gm__ float *)params.softmaxLse);
        dpGm.SetGlobalBuffer((__gm__ float *)params.dp);
        pWorkspaceGm.SetGlobalBuffer((__gm__ InputDType *)params.pWorkspace);
        softGradworkspaceGm.SetGlobalBuffer((__gm__ float *)params.softGradworkspace);
        dsWorkspaceGm.SetGlobalBuffer((__gm__ InputDType *)params.dsWorkspace);
    }

    template <>
    __aicore__ inline void operator()<AscendC::AIV>(Params const &params)
    {
        Init(params);

        if (cBlockIdx >= usedVecCoreNums || row <= 0) {
            return;
        }

        // col <= 128
        // 计算单loop的计算量及loop次数
        uint64_t eleBaseBuffNum = p32BaseBufLen / STAGES / sizeof(float); // 基本buffer块的元素数量
        uint64_t bufferRows = baseM / STAGES;                             // 一次lopp可以执行的最多row行数
        uint64_t rowLoopTimes = row / bufferRows;
        uint64_t tailRowNum = row - rowLoopTimes * bufferRows;

        uint64_t ping = 0;

        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);

        // 不包含尾行处理
        for (uint64_t i = 0; i < rowLoopTimes; i++) {
            uint64_t curS1Idx = curCoreS1Idx + i * bufferRows;
            int32_t gmRowOffset = i * bufferRows * col;
            compute(gmRowOffset, bufferRows, col, curS1Idx, ping);
            if (STAGES == DOUBLE_BUFFER) {
                ping = 1 - ping;
            }
        }

        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);

        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
        if (tailRowNum > 0) {
            uint64_t curS1Idx = curCoreS1Idx + rowLoopTimes * bufferRows;
            int32_t gmOffset = rowLoopTimes * bufferRows * col;
            uint64_t tempRow = tailRowNum;
            compute(gmOffset, tempRow, col, curS1Idx, ping);
            if (STAGES == DOUBLE_BUFFER) {
                ping = 1 - ping;
            }
        }
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
    }

    __aicore__ inline void compute(int32_t gmOffset, uint64_t row, uint64_t col, uint64_t curS1Idx, uint64_t ping)
    {
        struct SimplySoftMaxInfo<InputDType> runSftInfo = {
            sTensor[ping], lseTensor[ping], lseBrocTensor[ping], pFp32Tensor[ping], p16Tensor[ping], sGm[gmOffset],
                softmaxLseGm, pWorkspaceGm[gmOffset]
        };
        struct CalDsInfo<InputDType> runDsInfo = {
            dpFp32Tensor[ping], softmaxGradTensor[ping], pFp32Tensor[ping], ds16Tensor[ping], dpGm[gmOffset],
                softGradworkspaceGm, dsWorkspaceGm[gmOffset]
        };

        CalSimplySoft(runSftInfo, row, col, curS1Idx, ping);
        CalDs(runDsInfo, row, col, curS1Idx, ping);
    }

    /*
     * brief: simply softmax
     * runSftInfo : 计算需要的ub上的tensor
     * row: 需要计算的行数
     * col: 需要计算的列数
     * curS1Idx: 当前计算的query的 s 维度的idx
     */
    __aicore__ inline void CalSimplySoft(struct SimplySoftMaxInfo<InputDType> runSftInfo, uint64_t row, uint64_t col,
                                         uint64_t curS1Idx, uint64_t ping)
    {
        // simply softtmax
        LocalTensor<float> &sLocal = runSftInfo.sTensor;
        LocalTensor<float> &lse = runSftInfo.lseTensor;
        LocalTensor<float> &lseFp32Brc = runSftInfo.lseBrocTensor;
        LocalTensor<float> &p32Local = runSftInfo.pFp32Tensor;
        LocalTensor<InputDType> &p16Local = runSftInfo.pFp16Tensor;

        GlobalTensor<float> s = runSftInfo.sGm;
        GlobalTensor<float> lseGm = runSftInfo.lseGm;
        GlobalTensor<InputDType> pGm = runSftInfo.pGm;

        uint64_t countAlign = row * alignCol;

        auto eventId = ping ? EVENT_ID4 : EVENT_ID5;

        wait_flag(PIPE_MTE3, PIPE_MTE2, eventId);

        if (align32Col * sizeof(InputDType) % BLOCK_BYTE_SIZE == 0) {
            DataCopyPad(sLocal, s, {static_cast<uint16_t>(row), static_cast<uint32_t>(col * sizeof(float)), 0, 0, 0},
                        {true, 0, static_cast<uint8_t>(align32Col - col), 0});
        } else {
            DataCopyPad(sLocal, s, {static_cast<uint16_t>(row), static_cast<uint32_t>(col * sizeof(float)), 0, 1, 0},
                        {true, 0, static_cast<uint8_t>(align32Col - col), 0});
        }

        LseCopy(lseGm, lse, lseFp32Brc, row, curS1Idx);

        set_flag(PIPE_MTE2, PIPE_V, eventId);
        wait_flag(PIPE_MTE2, PIPE_V, eventId);

        uint8_t repeatimes = CeilDiv(row, BRCB_BASE_NUM);
        if constexpr (INPUT_LAYOUT == BNSD) {
            Brcb(lseFp32Brc, lse, repeatimes, {1, 8});
            AscendC::PipeBarrier<PIPE_V>();
        }

        Muls(sLocal, sLocal, (float)scaleValue, countAlign);
        AscendC::PipeBarrier<PIPE_V>();

        SubBrcb(p32Local, sLocal, lseFp32Brc, row, alignCol);
        AscendC::PipeBarrier<PIPE_V>();

        Exp(p32Local, p32Local, countAlign);
        AscendC::PipeBarrier<PIPE_V>();

        Cast(p16Local, p32Local, AscendC::RoundMode::CAST_ROUND, countAlign);
        AscendC::PipeBarrier<PIPE_V>();

        set_flag(PIPE_V, PIPE_MTE3, eventId);
        wait_flag(PIPE_V, PIPE_MTE3, eventId);

        DataCopyPad(pGm, p16Local,
                    {static_cast<uint16_t>(row), static_cast<uint32_t>(col * sizeof(InputDType)), 0, 0, 0});
    }

    /*
     * brief: cal ds = p * (dp - D)
     * runDsInfo : 计算需要的ub上的tensor
     * row: 需要计算的行数
     * col: 需要计算的列数
     * curS1Idx: 当前计算的query的 s 维度的idx
     */
    __aicore__ inline void CalDs(struct CalDsInfo<InputDType> runDsInfo, uint64_t row, uint64_t col, uint64_t curS1Idx,
                                 uint64_t ping)
    {
        LocalTensor<float> dpLocal = runDsInfo.dpFp32Tensor;
        LocalTensor<float> dLocal = runDsInfo.softmaxGradTensor;
        LocalTensor<InputDType> ds16Tensor = runDsInfo.dsFp16Tensor;
        LocalTensor<float> &p32Local = runDsInfo.pFp32Tensor;

        GlobalTensor<float> dp = runDsInfo.dpGm;
        GlobalTensor<float> d = runDsInfo.softmaxGradGm;
        GlobalTensor<InputDType> ds = runDsInfo.dsGm;

        uint64_t countAlign = row * alignCol;

        auto eventId = ping ? EVENT_ID4 : EVENT_ID5;

        if (align32Col * sizeof(InputDType) % BLOCK_BYTE_SIZE == 0) {
            DataCopyPad(dpLocal, dp, {static_cast<uint16_t>(row), static_cast<uint32_t>(col * sizeof(float)), 0, 0, 0},
                        {true, 0, static_cast<uint8_t>(align32Col - col), 0});
        } else {
            DataCopyPad(dpLocal, dp, {static_cast<uint16_t>(row), static_cast<uint32_t>(col * sizeof(float)), 0, 1, 0},
                        {true, 0, static_cast<uint8_t>(align32Col - col), 0});
        }
        CopyDIn(d, dLocal, row, curS1Idx);

        set_flag(PIPE_MTE2, PIPE_V, eventId);
        wait_flag(PIPE_MTE2, PIPE_V, eventId);

        SubBrcb(dpLocal, dpLocal, dLocal, row, alignCol);
        AscendC::PipeBarrier<PIPE_V>();

        Mul(dpLocal, p32Local, dpLocal, countAlign);
        AscendC::PipeBarrier<PIPE_V>();

        Cast(ds16Tensor, dpLocal, AscendC::RoundMode::CAST_ROUND, countAlign);

        set_flag(PIPE_V, PIPE_MTE3, eventId);
        wait_flag(PIPE_V, PIPE_MTE3, eventId);

        DataCopyPad(ds, ds16Tensor,
                    {static_cast<uint16_t>(row), static_cast<uint32_t>(col * sizeof(InputDType)), 0, 0, 0});
        set_flag(PIPE_MTE3, PIPE_MTE2, eventId);
    }

    /*
     * lse copy and brocast
     * lse input shape (b n s 1) or (t n 1)
     * out shape (b n s 8) or (n s 8)
     * dtype float
     */
    __aicore__ inline void LseCopy(GlobalTensor<float> &LseGm, LocalTensor<float> &lse, LocalTensor<float> &lseFp32Brc,
                                   uint64_t count, uint64_t curS1Idx)
    {
        uint64_t startOffset = 0;
        auto eventId = EVENT_ID5;
        if constexpr (INPUT_LAYOUT == TND) {
            uint64_t prefixSum = 0;
            for (uint64_t b = 0; b < curCoreBatch; b++) {
                prefixSum += ((__gm__ int64_t *)actualQSeqlen)[b];
            }
            uint64_t bOffset = n1 * prefixSum;
            startOffset = bOffset + curS1Idx * n1 + curCoreN1Idx;
            // 对于TND 格式来说， 会进行类似 (s n) -> (n s) 的transpose转换
            DataCopyPad(lseFp32Brc, LseGm[startOffset],
                        {static_cast<uint16_t>(count), static_cast<uint32_t>(1 * sizeof(float)),
                         static_cast<uint32_t>(transpseStride), 0, 0},
                        {false, 0, 0, 0});
        } else {
            startOffset = curCoreBatch * (n1 * maxQSeqlen) + curCoreN1Idx * maxQSeqlen + curS1Idx;
            DataCopyPad(lse, LseGm[startOffset],
                        {static_cast<uint16_t>(1), static_cast<uint32_t>(count * sizeof(float)),
                         static_cast<uint32_t>(0), 0, 0},
                        {false, 0, 0, 0});
        }
    }

    /*
     * brief: Compute the elementwise multiplication of a tensor of shape (m, n) and a tensor of shape
     * ubIn0:[m, n], ubIn1[m, 8]
     */
    __aicore__ inline void SubBrcb(LocalTensor<float> const &ubOut, LocalTensor<float> const &ubIn0,
                                   LocalTensor<float> const &ubIn1, uint64_t row, uint64_t col)
    {
        // 分核逻辑的关系，矩阵shape不会超过[128,128], 若启动俩个vector，shape不会超过[64, 128]
        // 所以，一次sub， 最大可计算[128, 64], 迭代次数为行数, 一次迭代计算元素大小最多为64
        // 所以，对列按照64切分，循环sub计算
        uint32_t countEachRepeat = REAPTE_BYTE / sizeof(float);           // 每次迭代计算的元素 64
        uint32_t colLoop = (col + countEachRepeat - 1) / countEachRepeat; // 上取整
        uint32_t remain = col % countEachRepeat;
        uint64_t mask = countEachRepeat; // 参与计算的元素个数
        uint8_t repeatTimes = row;
        AscendC::BinaryRepeatParams repeatParams;
        repeatParams.dstBlkStride = 1;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1BlkStride = 0;
        repeatParams.dstRepStride = col / BLOCK_FP32_NUM;
        repeatParams.src0RepStride = col / BLOCK_FP32_NUM;
        repeatParams.src1RepStride = 1;

        for (uint32_t i = 0; i < colLoop; i++) {
            if (i == colLoop - 1 && remain != 0) {
                mask = remain;
            }
            AscendC::Sub(ubOut[i * countEachRepeat], ubIn0[i * countEachRepeat], ubIn1[0], mask, repeatTimes,
                         repeatParams);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    /*
     * D shape ((n s1 8) or (b n s1 8) fp32
     */
    __aicore__ inline void CopyDIn(GlobalTensor<float> &d, LocalTensor<float> &dLocal, uint64_t count,
                                   uint64_t curS1Idx)
    {
        uint64_t startOffset = 0;
        uint32_t srcStride = 0;
        if constexpr (INPUT_LAYOUT == TND) {
            uint64_t prefixSum = 0;
            for (uint64_t b = 0; b < curCoreBatch; b++) {
                prefixSum += ((__gm__ int64_t *)actualQSeqlen)[b];
            }
            uint64_t bOffset = prefixSum * n1 * BRCB_BASE_NUM;
            startOffset = bOffset + curS1Idx * n1 * BRCB_BASE_NUM + curCoreN1Idx * BRCB_BASE_NUM;
            srcStride = static_cast<uint32_t>((n1 - 1) * BRCB_BASE_NUM * sizeof(float));
        } else {
            startOffset = curCoreBatch * (n1 * maxQSeqlen * BRCB_BASE_NUM) + curCoreN1Idx * maxQSeqlen * BRCB_BASE_NUM +
                          curS1Idx * BRCB_BASE_NUM;
        }
        DataCopyPad(
            dLocal, d[startOffset],
            {static_cast<uint16_t>(count), static_cast<uint32_t>(BRCB_BASE_NUM * sizeof(float)), srcStride, 0, 0},
            {false, 0, 0, 0});
    }
};

} // namespace NpuArch::Epilogue::Block

#endif // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_SIMPLY_SOFTMAX_HPP
