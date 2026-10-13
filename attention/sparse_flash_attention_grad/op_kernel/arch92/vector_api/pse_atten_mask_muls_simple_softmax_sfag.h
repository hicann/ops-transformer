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
 * \file pse_atten_mask_muls_simple_softmax_sfag.h
 */

#ifndef PSE_ATTEN_MASK_MULS_SIMPLE_SOFTMAX__SFAG
#define PSE_ATTEN_MASK_MULS_SIMPLE_SOFTMAX__SFAG
#include "../sparse_flash_attention_grad_arch92_common.h"
#include "vf_muls_sel_simple_softmax_sfag.h"

using namespace commondef;

template <typename T2, const uint32_t BASE_M = 64>
__aicore__ inline void CopyInMaxSum(FagConstInfo& constInfo, FagRunInfo& runInfo,
                                    TQue<QuePosition::VECIN, 1>& maxSumInQue, GlobalTensor<T2>& maxGm,
                                    GlobalTensor<T2>& sumGm)
{
    if (runInfo.gRealSize == 0) {
        return;
    }
    int64_t maxSumGmOffset = runInfo.t1Index * constInfo.commonConstInfo.gSize;
    LocalTensor<T2> maxSumTensor = maxSumInQue.AllocTensor<T2>();

    DataCopyPad(maxSumTensor, sumGm[maxSumGmOffset],
                {static_cast<uint16_t>(runInfo.gRealSize), static_cast<uint16_t>(sizeof(float)), 0, 0},
                {false, 0, 0, 0});
    DataCopyPad(maxSumTensor[BASE_M * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2)], maxGm[maxSumGmOffset],
                {static_cast<uint16_t>(runInfo.gRealSize), static_cast<uint16_t>(sizeof(float)), 0, 0},
                {false, 0, 0, 0});
    maxSumInQue.EnQue(maxSumTensor);
}

// CopyInSinksAndMaxSum：sinks + max + sum 从 GM 拷到 TQue（MTE2 + EnQue）
// DataCopyPad {1, g*4, 0, 0} 单块连续搬运——非对齐 g(如 N1=24 → 96B) 也能连续搬进 TQue。
// 不能用 {g,4,0,0} 碎块（每 4B 块补齐 32B → 数据摊成 1-float-per-32B，连续读拿到 dummy）。
// 不能复用 maxSumQue——它是 {g,4,0,0} 摊开布局（BRC 读），sink 微指令用 default LoadAlign 连续读，布局不兼容。
template <typename T2, const uint32_t BASE_M = 64>
__aicore__ inline void CopyInSinksAndMaxSum(FagConstInfo& constInfo, FagRunInfo& runInfo, GlobalTensor<T2>& sinksGm,
                                            GlobalTensor<T2>& maxGm, GlobalTensor<T2>& sumGm,
                                            TQue<QuePosition::VECIN, 1>& sinksInQue,
                                            TQue<QuePosition::VECIN, 1>& maxForSinkQue,
                                            TQue<QuePosition::VECIN, 1>& sumForSinkQue)
{
    if (runInfo.gRealSize == 0) {
        return;
    }
    int64_t gmOffset = 0;
    int64_t maxSumGmOffset = runInfo.t1Index * constInfo.commonConstInfo.gSize;
    // 单块、blockLen=g*sizeof(T) 字节；框架只在末尾补 dummy 到 32B，真数据连续在 dst[0:g)。
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(runInfo.gRealSize * sizeof(T2));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;

    LocalTensor<T2> sinksTensor = sinksInQue.AllocTensor<T2>();
    LocalTensor<T2> maxTensor = maxForSinkQue.AllocTensor<T2>();
    LocalTensor<T2> sumTensor = sumForSinkQue.AllocTensor<T2>();
    DataCopyPad(sinksTensor, sinksGm[gmOffset], copyParams, {false, 0, 0, 0});
    DataCopyPad(maxTensor, maxGm[maxSumGmOffset], copyParams, {false, 0, 0, 0});
    DataCopyPad(sumTensor, sumGm[maxSumGmOffset], copyParams, {false, 0, 0, 0});
    sinksInQue.EnQue(sinksTensor);
    maxForSinkQue.EnQue(maxTensor);
    sumForSinkQue.EnQue(sumTensor);
}

/*************************
Function： VF计算函数，实现Pse + AttenMask + Muls + SimpleSoftmax计算
baseParams：循环定参，入参
runInfo: 循环变参，入参
attenMaskInfo：attenMask相关参数，入参
maxSumInQue：maxSum分配Que，入参
attenMaskInQue：attenMask分配Que，入参
pseInQue：pse分配Que，入参
dstTensor：返回计算结果，出参
srcTensor：VF输入，入参
*************************/
template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK = 0, const uint32_t IS_PSE = 0,
          const uint32_t IS_DETER_OLD = 0, const uint32_t BASE_M = 64, const uint32_t BASE_N = 128>
__aicore__ inline void CalculatePseMulsSelSimpleSoftMax(FagConstInfo& constInfo, FagRunInfo& runInfo, PseInfo& pseInfo,
                                                        AttenMaskInfo& attenMaskInfo,
                                                        TQue<QuePosition::VECIN, 1>& maxSumInQue,
                                                        LocalTensor<T2>& dstTensor, LocalTensor<T2>& srcTensor,
                                                        __gm__ uint8_t* pseSlope)
{
    if (runInfo.gRealSize == 0) {
        return;
    }
    LocalTensor<uint8_t> attenMaskTensor;
    LocalTensor<T1> pseTensor;
    // Compute
    LocalTensor<T2> maxSumTensor = maxSumInQue.DeQue<T2>();

    if (runInfo.commonRunInfo.s2RealSize > 64) {
        AscendC::MulsSelSimpleSoftMax<T1, T2, 128, IS_ATTEN_MASK, IS_PSE, IS_DETER_OLD>(
            dstTensor, maxSumTensor, maxSumTensor[BASE_M * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2)], srcTensor, pseTensor,
            attenMaskTensor, constInfo.scaleValue, constInfo.attenMaskMinValue, runInfo.gRealSize,
            runInfo.commonRunInfo.s2RealSize);
    } else {
        AscendC::MulsSelSimpleSoftMax<T1, T2, 64, IS_ATTEN_MASK, IS_PSE, IS_DETER_OLD>(
            dstTensor, maxSumTensor, maxSumTensor[BASE_M * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2)], srcTensor, pseTensor,
            attenMaskTensor, constInfo.scaleValue, constInfo.attenMaskMinValue, runInfo.gRealSize,
            runInfo.commonRunInfo.s2RealSize);
    }
    maxSumInQue.FreeTensor(maxSumTensor);
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK = 0, const uint32_t IS_PSE = 0,
          const uint32_t IS_DETER_OLD = 0, const uint32_t BASE_M = 64, const uint32_t BASE_N = 128>
__aicore__ inline void CalculatePseMulsSelSimpleSoftMaxReuse(FagConstInfo& constInfo, FagRunInfo& runInfo,
                                                             PseInfo& pseInfo, AttenMaskInfo& attenMaskInfo,
                                                             LocalTensor<T2>& maxSumTensor, LocalTensor<T2>& dstTensor,
                                                             LocalTensor<T2>& srcTensor, __gm__ uint8_t* pseSlope)
{
    LocalTensor<uint8_t> attenMaskTensor;
    LocalTensor<T1> pseTensor;
    if (runInfo.commonRunInfo.s2RealSize > 64) {
        AscendC::MulsSelSimpleSoftMax<T1, T2, 128, IS_ATTEN_MASK, IS_PSE, IS_DETER_OLD>(
            dstTensor, maxSumTensor, maxSumTensor[BASE_M * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2)], srcTensor, pseTensor,
            attenMaskTensor, constInfo.scaleValue, constInfo.attenMaskMinValue, runInfo.gRealSize,
            runInfo.commonRunInfo.s2RealSize);
    } else {
        AscendC::MulsSelSimpleSoftMax<T1, T2, 64, IS_ATTEN_MASK, IS_PSE, IS_DETER_OLD>(
            dstTensor, maxSumTensor, maxSumTensor[BASE_M * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2)], srcTensor, pseTensor,
            attenMaskTensor, constInfo.scaleValue, constInfo.attenMaskMinValue, runInfo.gRealSize,
            runInfo.commonRunInfo.s2RealSize);
    }
}

// 微指令 VF：P_sink = exp(sink - max) / sum
// 寄存器内运算（GetPhyAddr + LoadAlign 走原始 UB 地址，绕开"标准 Vector API 读 TQue 只读 element 0"）。
// sink/max/sum 从 sink 专用 TQue（DataCopyPad {1,rowNum*4,0,0} 单块连续）default LoadAlign 连续读取。
// Div 与参考实现 MulsSelSimpleSoftMax 同款；pregTail = UpdateMask(剩余行数) 控制每段有效元素数。
// rowNum 最大为 BASE_M=128（单 AIV 吃满整个 G），一个 fp32 VREG 只有 64 lane，因此按 64 行分段循环；
// POST_MODE_UPDATE 会在每次 Load/Store 后把地址前进 FP32_LANES_PER_VREG 个元素。
template <typename T>
__simd_vf__ inline void SinkSoftMaxVF(uint64_t sinkAddr, uint64_t maxAddr, uint64_t sumAddr, uint64_t dstAddr,
                                      uint32_t rowNum)
{
    constexpr uint32_t FP32_LANES_PER_VREG = 64;
    RegTensor<float> vregSink, vregMax, vregSum, vregSub, vregExp, vregDiv;
    for (uint32_t done = 0; done < rowNum; done += FP32_LANES_PER_VREG) {
        // 尾段 [rowNum-done, 64) lane 读到的是 TQue UB 后续内容，pregTail 会屏蔽掉
        uint32_t tailSize = rowNum - done;
        MaskReg pregTail = UpdateMask<float>(tailSize);
        LoadAlign<float, Reg::PostLiteral::POST_MODE_UPDATE>(vregSink, (__ubuf__ float*&)sinkAddr, FP32_LANES_PER_VREG);
        LoadAlign<float, Reg::PostLiteral::POST_MODE_UPDATE>(vregMax, (__ubuf__ float*&)maxAddr, FP32_LANES_PER_VREG);
        LoadAlign<float, Reg::PostLiteral::POST_MODE_UPDATE>(vregSum, (__ubuf__ float*&)sumAddr, FP32_LANES_PER_VREG);
        Sub(vregSub, vregSink, vregMax, pregTail); // sink - max
        Exp(vregExp, vregSub, pregTail);           // exp(sink-max)
        Div(vregDiv, vregExp, vregSum, pregTail);  // exp / sum
        StoreAlign<float, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_NORM_B32>(
            (__ubuf__ float*&)dstAddr, vregDiv, FP32_LANES_PER_VREG, pregTail);
    }
}

// CalculateSinkSimpleSoftMax：P_sink = exp(sink-max)/sum，微指令直接读 sink 专用 TQue
// sinksInQue/maxForSinkQue/sumForSinkQue 已由 CopyInSinksAndMaxSum 用 DataCopyPad 单块连续搬入并 EnQue。
// TQue DeQue 自动 MTE2→V 同步；微指令 LoadAlign 读 TQue（GetPhyAddr 原始地址）。
// 结果 StoreAlign 写 sinkTensor(TBuf)，V3 用标准 Mul 读（TBuf 上标准 Vector API 可靠）。
template <typename T2, const uint32_t BASE_M = 64>
__aicore__ inline void CalculateSinkSimpleSoftMax(FagConstInfo& constInfo, FagRunInfo& runInfo,
                                                  TQue<QuePosition::VECIN, 1>& sinksInQue,
                                                  TQue<QuePosition::VECIN, 1>& maxForSinkQue,
                                                  TQue<QuePosition::VECIN, 1>& sumForSinkQue,
                                                  LocalTensor<T2>& sinkTensor)
{
    LocalTensor<T2> sinkInTensor = sinksInQue.DeQue<T2>();
    LocalTensor<T2> maxTensor = maxForSinkQue.DeQue<T2>();
    LocalTensor<T2> sumTensor = sumForSinkQue.DeQue<T2>();
    uint64_t sinkAddr = sinkInTensor.GetPhyAddr();
    uint64_t maxAddr = maxTensor.GetPhyAddr();
    uint64_t sumAddr = sumTensor.GetPhyAddr();
    uint64_t dstAddr = sinkTensor.GetPhyAddr();
    SinkSoftMaxVF<float>(sinkAddr, maxAddr, sumAddr, dstAddr, runInfo.gRealSize);
    sinksInQue.FreeTensor(sinkInTensor);
    maxForSinkQue.FreeTensor(maxTensor);
    sumForSinkQue.FreeTensor(sumTensor);
}

// 微指令 VF：SinkNegRowSumVF —— 逐元素 dSink 的"行内负和"
// rowSumNeg[h] = -Σ_j P[h,j]·dp[h,j]（P∈mm2、dp∈mm1，fp32 [gRows, s2RealSize]，行步长 srcN=128 元素）
// 必须在 BroadcastSubMul 覆盖 mm1(dp) 之前调用（否则 dp 已被 P⊙(dp-Di) 覆盖）。
// 结构同参考 flash_attention_score_grad vf_cal_sink.h 的 CalculateSinkVF：外层 m 行循环(uint16_t)
// + 内层主 64-chunk 循环 + 直线尾块(恒执行一次, UpdateMask 控制实尾)；
// 归约/标量写沿用本算子 vf_softmax_grad_front_cast_aligned512_f16.h 的
// ReduceSum + StoreUnAlign(POST_MODE_UPDATE 自动推进 dstAddr) + vstas flush。
// 负号折进累加：vregRes 每行 Duplicate(0)，Sub(vregRes, vregRes, vregReduceSum, pregAccu=VL1) 保护 lane0，
// 免去额外的 Muls 取负。每 head 标量结果顺序写 dst[0..gRows-1]。
template <typename T, uint16_t srcN>
__simd_vf__ inline void SinkNegRowSumVF(uint64_t dstAddr, uint64_t pAddr, uint64_t dpAddr, uint64_t pTailAddr,
                                        uint64_t dpTailAddr, uint16_t gRows, uint16_t loopTimes, uint32_t realTailSize)
{
    RegTensor<T> vregP;
    RegTensor<T> vregDp;
    RegTensor<T> vregMul;
    RegTensor<T> vregReduceSum;
    RegTensor<T> vregRes;
    UnalignReg uregRes;
    MaskReg pregFullExe = CreateMask<T, MaskPattern::ALL>();
    MaskReg pregAccu = CreateMask<T, MaskPattern::VL1>();
    MaskReg pregTailExe = UpdateMask<float>(realTailSize);
    for (uint16_t m = 0; m < gRows; m++) {
        Duplicate(vregRes, 0);
        for (uint16_t n = 0; n < loopTimes; n++) {
            LoadAlign(vregP, ((__ubuf__ T*&)pAddr + m * srcN + n * 64));
            LoadAlign(vregDp, ((__ubuf__ T*&)dpAddr + m * srcN + n * 64));
            Mul(vregMul, vregP, vregDp, pregFullExe);
            ReduceSum(vregReduceSum, vregMul, pregFullExe);
            Sub(vregRes, vregRes, vregReduceSum, pregAccu);
        }
        // 尾块（直线，恒执行一次）：整 64 尾(realTailSize==64) 或部分尾(UpdateMask)
        LoadAlign(vregP, ((__ubuf__ T*&)pTailAddr + m * srcN));
        LoadAlign(vregDp, ((__ubuf__ T*&)dpTailAddr + m * srcN));
        Mul(vregMul, vregP, vregDp, pregTailExe);
        ReduceSum(vregReduceSum, vregMul, pregTailExe);
        Sub(vregRes, vregRes, vregReduceSum, pregAccu);
        // 每 head 标量写 dst[m]，POST_MODE_UPDATE 自动推进 dstAddr 一个元素
        StoreUnAlign<T, Reg::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T*&)dstAddr), vregRes, uregRes, 1);
    }
    vstas(uregRes, ((__ubuf__ T*&)dstAddr), 0, POST_UPDATE);
}

// CalculateSinkNegRowSum：逐元素 dSink 行内负和 wrapper（__aicore__ 侧算循环参数，VF 侧只跑 uint16_t 循环）
// guard gRows==0 / realN==0（realN==0 时 CeilDivision-1 会下溢成 65535，尾指针越界，必须拦）。
// 尾指针 = 基址 + loopTimes*64 元素（= 主 chunk 覆盖后的下一 chunk 起始），与 vf_cal_sink.h 一致。
template <typename T2, const uint32_t BASE_N = 128>
__aicore__ inline void CalculateSinkNegRowSum(LocalTensor<T2>& dstTensor, LocalTensor<T2>& pTensor,
                                              LocalTensor<T2>& dpTensor, uint16_t gRows, uint16_t realN)
{
    if (gRows == 0 || realN == 0) {
        return;
    }
    const uint16_t fullExeSize = 64;
    uint16_t loopTimes = CeilDivision(realN, fullExeSize) - 1;
    uint16_t tailSize = realN % fullExeSize;
    uint16_t realTailSize = tailSize == 0 ? fullExeSize : tailSize;
    uint64_t dstLocalInt = dstTensor.GetPhyAddr();
    uint64_t pLocalInt = pTensor.GetPhyAddr();
    uint64_t dpLocalInt = dpTensor.GetPhyAddr();
    uint64_t pLocalIntTail = pTensor.GetPhyAddr() + (uint64_t)loopTimes * fullExeSize * sizeof(T2);
    uint64_t dpLocalIntTail = dpTensor.GetPhyAddr() + (uint64_t)loopTimes * fullExeSize * sizeof(T2);
    SinkNegRowSumVF<T2, static_cast<uint16_t>(BASE_N)>(dstLocalInt, pLocalInt, dpLocalInt, pLocalIntTail,
                                                       dpLocalIntTail, gRows, loopTimes, realTailSize);
}

#endif
