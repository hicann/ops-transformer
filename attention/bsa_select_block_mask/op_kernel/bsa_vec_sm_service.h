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
 * \file bsa_vec_sm_service.h
 * \brief
 */
#ifndef BSA_VEC_SM_SERVICE_H
#define BSA_VEC_SM_SERVICE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "bsa_select_block_mask_common.h"
#include "bsa_select_block_mask_tiling_data.h"

template <typename BSAT>
class BSAVecSmService {
public:
    using T = float;                    //< 中间计算类型 (float, FP32)
    using IN_T = typename BSAT::inputT; //< 输入数据类型 (FP16/BF16)
    using OUT_T = half;                 //< 输出数据类型 (INT8)

    static constexpr BSALayout LAYOUT_Q = BSAT::layoutQ;   // BNSD or TND
    static constexpr BSALayout LAYOUT_KV = BSAT::layoutKV; // BNSD or TND

    __aicore__ inline BSAVecSmService(){};
    __aicore__ inline void InitParams(const BSAConstInfo &constInfo,
                                      const optiling::BSASelectBlockMaskTilingData *__restrict tilingData);
    __aicore__ inline void InitBuffers(TBuf<> *uBuf_);

    __aicore__ inline void InitGM(GlobalTensor<T> &ScoreFp32Gm, GlobalTensor<OUT_T> &attnScoreFp16Gm);

    __aicore__ inline void OnlineSoftmaxFirstPassChunk(uint32_t qChunkStart, uint32_t qChunkSize, uint32_t kChunkStart,
                                                       uint32_t kChunkSize, uint32_t validYBlocks);

    __aicore__ inline void SoftmaxSecondPassAndCast(uint32_t qChunkStart, uint32_t qChunkSize, uint32_t kChunkStart,
                                                    uint32_t kChunkSize, uint32_t batchIdx, uint32_t headIdx,
                                                    uint32_t validYBlocks);

private:
    BSAConstInfo constInfo;
    const optiling::BSASelectBlockMaskTilingData *__restrict tilingData;

    static constexpr int32_t BUFFER = 1;
    uint32_t vecSubBlockIdx = 0; // vector核在aic的序号： 0 or 1

    // ---- GM 全局张量引用 ----
    GlobalTensor<OUT_T> attnScoreFp16GmTensor; //< 注意力分数 [xBlocks * yBlocks] (FP32→FP16)
    GlobalTensor<T> scoreFp32GmTensor;         //< Softmax 临时缓冲 (FP32)
    TBuf<> uBuf_;                              //< 统一 UB 缓冲区 (192KB)

    LocalTensor<uint8_t> reduceSharedTemp; // reduce相关 高阶接口的临时存储空间，暂定24k
    LocalTensor<T> tmpMax;                 //< Softmax Max (FP32) * 8 (即FLOAT_DATA_BLOCK_NUM)
    LocalTensor<T> tmpSum;                 //< Softmax Sum (FP32) * 8 (即FLOAT_DATA_BLOCK_NUM)
    LocalTensor<T> globalMax;              //< Softmax old Max (FP32) 在一轮k里会驻守 <= 64
    LocalTensor<T> globalSum;              //< Softmax old Sum (FP32) 在一轮k里会驻守 <= 64
    LocalTensor<T> globalMaxBroc;          //< Softmax Max (FP32) 在一轮k里会驻守 <= 64
    LocalTensor<T> globalSumBroc;          //< Softmax Max (FP32) 在一轮k里会驻守 <= 64
    LocalTensor<T> diffMax;

    LocalTensor<T> scoreUb;         // out shape [x, y] < [64, 128 * 5]
    LocalTensor<OUT_T> scoreFp16Ub; // 上一轮矩阵乘结果 input [x, y] < [64, 128 * 5]

    // ---- 硬件同步事件 ----
    event_t eventSft = (event_t)3;
};

template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::InitParams(
    const BSAConstInfo &constInfo, const optiling::BSASelectBlockMaskTilingData *__restrict tilingData)
{
    this->constInfo = constInfo;
    this->tilingData = tilingData;
    vecSubBlockIdx = constInfo.subBlockIdx; // 0 or 1
}

template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::InitBuffers(TBuf<> *uBuf_)
{
    // ub 划分
    // 根据UB计算，最多可以5次cube，一次vector计算， 即score ub 占用64 * 128 * 5 * 4 = 160k
    // 1个cube计算128 * 128， 计算5次
    // 一个vector 计算64 * 128 * 5
    // max 64 * 4 = 0.25k
    // sum 64 * 4 = 0.25k
    // dif (max - global_max) 64 * 4 = 0.25k
    // score 64 * 128 * 4 * 5 = 160 k // 1 个cube核计算量均分给俩个vectore核
    // global Max 64 * 4 = 0.25k
    // global L 64 * 4 = 0.25kk
    // max_broc 64 * 8 * 4 = 2k
    // sum_broc 64 * 8 * 4 = 2k
    // shareTempReduce 24k

    uint32_t ubOffset = 0;
    uint32_t chunkSize = Q_CHUNK_SIZE;

    globalMax = uBuf_->GetWithOffset<T>(chunkSize, ubOffset);
    ubOffset += chunkSize * sizeof(T);
    ubOffset = BSAAlignTo(ubOffset, static_cast<uint32_t>(VEC_ALIGN_SIZE));

    globalMaxBroc = uBuf_->GetWithOffset<T>(chunkSize * FLOAT_DATA_BLOCK_NUM, ubOffset);
    ubOffset += chunkSize * FLOAT_DATA_BLOCK_NUM * sizeof(T);
    ubOffset = BSAAlignTo(ubOffset, static_cast<uint32_t>(VEC_ALIGN_SIZE));

    globalSum = uBuf_->GetWithOffset<T>(chunkSize, ubOffset);
    ubOffset += chunkSize * sizeof(T);
    ubOffset = BSAAlignTo(ubOffset, static_cast<uint32_t>(VEC_ALIGN_SIZE));

    globalSumBroc = uBuf_->GetWithOffset<T>(chunkSize * FLOAT_DATA_BLOCK_NUM, ubOffset);
    ubOffset += chunkSize * FLOAT_DATA_BLOCK_NUM * sizeof(T);
    ubOffset = BSAAlignTo(ubOffset, static_cast<uint32_t>(VEC_ALIGN_SIZE));

    tmpMax = uBuf_->GetWithOffset<T>(chunkSize, ubOffset);
    ubOffset += chunkSize * sizeof(T);
    ubOffset = BSAAlignTo(ubOffset, static_cast<uint32_t>(VEC_ALIGN_SIZE));

    diffMax = uBuf_->GetWithOffset<T>(chunkSize, ubOffset);
    ubOffset += chunkSize * sizeof(T);
    ubOffset = BSAAlignTo(ubOffset, static_cast<uint32_t>(VEC_ALIGN_SIZE));

    tmpSum = uBuf_->GetWithOffset<T>(chunkSize, ubOffset);
    ubOffset += chunkSize * sizeof(T);
    ubOffset = BSAAlignTo(ubOffset, static_cast<uint32_t>(VEC_ALIGN_SIZE));

    uint32_t scoreElements = (Q_CHUNK_SIZE / 2) * K_CHUNK_SIZE * CV_EXEC_RATIO;
    scoreUb = uBuf_->GetWithOffset<T>(scoreElements, ubOffset);
    scoreFp16Ub = uBuf_->GetWithOffset<OUT_T>(scoreElements, ubOffset); // 复用fp32空间

    ubOffset += scoreElements * sizeof(T);
    ubOffset = BSAAlignTo(ubOffset, static_cast<uint32_t>(VEC_ALIGN_SIZE));

    uint32_t sharedReduceEles = BSAConstInfo::BUFFER_SIZE_BYTE_24K / sizeof(uint8_t);
    reduceSharedTemp = uBuf_->GetWithOffset<uint8_t>(sharedReduceEles, ubOffset);
    ubOffset += sharedReduceEles * sizeof(uint8_t);
    ubOffset = BSAAlignTo(ubOffset, static_cast<uint32_t>(VEC_ALIGN_SIZE));
}

template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::InitGM(GlobalTensor<T> &ScoreFp32Gm, GlobalTensor<OUT_T> &attnScoreFp16Gm)
{
    this->attnScoreFp16GmTensor = attnScoreFp16Gm;
    this->scoreFp32GmTensor = ScoreFp32Gm;
}

__aicore__ inline void SubBroadcast(LocalTensor<float> const &ubOut, LocalTensor<float> const &ubIn0,
                                    LocalTensor<float> const &ubIn1, uint64_t row, uint64_t col)
{
    // 默认 col 安装32字节对齐
    // 分核逻辑的关系，矩阵shape不会超过[128, 128 * 5], 若启动俩个vector，shape不会超过[64, 128 * 5]
    // 所以，一次sub， 最大可计算[255, 64], 迭代次数为行数, 一次迭代计算元素大小最多为64
    // 所以，对列按照64切分，循环sub计算
    uint32_t countEachRepeat = 256 / sizeof(float);                   // 每次迭代计算的元素 64
    uint32_t colLoop = (col + countEachRepeat - 1) / countEachRepeat; // 上取整
    uint32_t remain = col % countEachRepeat;
    uint64_t mask = countEachRepeat; // 参与计算的元素个数
    uint8_t repeatTimes = row;
    AscendC::BinaryRepeatParams repeatParams;
    repeatParams.dstBlkStride = 1;
    repeatParams.src0BlkStride = 1;
    repeatParams.src1BlkStride = 0;
    repeatParams.dstRepStride = col / FLOAT_DATA_BLOCK_NUM;
    repeatParams.src0RepStride = col / FLOAT_DATA_BLOCK_NUM;
    repeatParams.src1RepStride = 1;

    for (uint32_t i = 0; i < colLoop; i++) {
        if (i == colLoop - 1 && remain != 0) {
            mask = remain;
        }
        AscendC::Sub(ubOut[i * countEachRepeat], ubIn0[i * countEachRepeat], ubIn1[0], mask, repeatTimes, repeatParams);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void DivBroadcast(LocalTensor<float> const &ubOut, LocalTensor<float> const &ubIn0,
                                    LocalTensor<float> const &ubIn1, uint64_t row, uint64_t col)
{
    // 默认 col 安装32字节对齐
    // 分核逻辑的关系，矩阵shape不会超过[128, 128 * 5], 若启动俩个vector，shape不会超过[64, 128 * 5]
    // 所以，一次div， 最大可计算[255, 64], 迭代次数为行数, 一次迭代计算元素大小最多为64
    // 所以，对列按照64切分，循环div计算
    uint32_t countEachRepeat = 256 / sizeof(float);                   // 每次迭代计算的元素 64
    uint32_t colLoop = (col + countEachRepeat - 1) / countEachRepeat; // 上取整
    uint32_t remain = col % countEachRepeat;
    uint64_t mask = countEachRepeat; // 参与计算的元素个数
    uint8_t repeatTimes = row;
    AscendC::BinaryRepeatParams repeatParams;
    repeatParams.dstBlkStride = 1;
    repeatParams.src0BlkStride = 1;
    repeatParams.src1BlkStride = 0;
    repeatParams.dstRepStride = col / FLOAT_DATA_BLOCK_NUM;
    repeatParams.src0RepStride = col / FLOAT_DATA_BLOCK_NUM;
    repeatParams.src1RepStride = 1;

    for (uint32_t i = 0; i < colLoop; i++) {
        if (i == colLoop - 1 && remain != 0) {
            mask = remain;
        }
        AscendC::Div(ubOut[i * countEachRepeat], ubIn0[i * countEachRepeat], ubIn1[0], mask, repeatTimes, repeatParams);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

/**
 * @brief Online Softmax 第一遍——对一个 K-Chunk 的 Score 块完成局部 softmax 的 m/l 累积
 * @details 采用 Online Softmax 算法（两遍法），避免全尺寸 score 矩阵的物化。
 *          第一遍对 score 矩阵的一个子块 [qChunkStart:qChunkStart+qChunkSize, kChunkStart:kChunkStart+kChunkSize]
 *          逐行更新 m（逐行最大值）和 l（exp 和）：
 *          - 从 softmaxTmpGm（FP32）读入 score_partial 子块
 *          - m_new = max(m, row_max(score_partial))
 *          - l_new = l × exp(m - m_new) + row_sum(exp(score_partial - m_new))
 *          - 将原始 score_partial 写回 softmaxTmpGm（用于第二遍归一化）
 *          - m = m_new, l = l_new
 *          最终 mFinal 和 lFinal 通过引用返回，跨所有 K-Chunk 累积。
 * @tparam BSAT 算子模板类型
 * @param qChunkStart Q 维度 chunk 起始偏移
 * @param qChunkSize Q 维度 chunk 大小（block 数）
 * @param kChunkStart K 维度 chunk 起始偏移
 * @param kChunkSize K 维度 chunk 大小（block 数）
 * @param[out] mFinal 跨所有 K-Chunk 累积后的逐行最大值（FP32）
 * @param[out] lFinal 跨所有 K-Chunk 累积后的逐行 exp 和（FP32）
 */
template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::OnlineSoftmaxFirstPassChunk(uint32_t qChunkStart, uint32_t qChunkSize,
                                                                          uint32_t kChunkStart, uint32_t kChunkSize,
                                                                          uint32_t validYBlocks)
{
    // 更新max
    // M = ReduceMax(score_partial)
    // m_new = max(old_m, M) # 复用old_m
    // 获取vector 实际执行计算的start row 和 excute row
    // FIX: remove duplicate type specifier on actualQExcuteRow
    uint32_t actualQRowStart = 0, actualQExcuteRow = 0;
    if (vecSubBlockIdx == 0) {
        actualQExcuteRow = qChunkSize / 2 + qChunkSize % 2;
        actualQRowStart = constInfo.aicIdx * Q_CHUNK_SIZE;
    } else {
        actualQRowStart = constInfo.aicIdx * Q_CHUNK_SIZE + qChunkSize / 2 + qChunkSize % 2;
        actualQExcuteRow = qChunkSize / 2;
    }

    // qChunkSize 为 1 的场景
    if (actualQExcuteRow == 0) {
        return;
    }

    // 32字节对齐
    uint32_t kChunkSizeAlign = (kChunkSize + FLOAT_DATA_BLOCK_NUM - 1) / FLOAT_DATA_BLOCK_NUM * FLOAT_DATA_BLOCK_NUM;

    SetFlag<HardEvent::MTE3_MTE2>(eventSft);
    WaitFlag<HardEvent::MTE3_MTE2>(eventSft);

    SetFlag<HardEvent::V_MTE2>(eventSft);
    WaitFlag<HardEvent::V_MTE2>(eventSft);

    // score 矩阵workspace存储还需再想想
    DataCopyPad(scoreUb, scoreFp32GmTensor[actualQRowStart * constInfo.yBlocks + kChunkStart],
                {static_cast<uint16_t>(actualQExcuteRow), static_cast<uint32_t>(kChunkSize * sizeof(float)),
                 static_cast<uint32_t>((constInfo.yBlocks - kChunkSize) * sizeof(float)), 0, 0},
                {true, 0, static_cast<uint8_t>(kChunkSizeAlign - kChunkSize), SOFTMAX_NEG_INF});

    SetFlag<HardEvent::MTE2_V>(eventSft);
    WaitFlag<HardEvent::MTE2_V>(eventSft);

    Muls(scoreUb, scoreUb, (float)(constInfo.scaleValue), actualQExcuteRow * kChunkSizeAlign);
    AscendC::PipeBarrier<PIPE_V>();

    // 更新new max
    uint32_t maxShape[] = {actualQExcuteRow, kChunkSizeAlign};
    bool isSrcInnerPad = true;

    AscendC::ReduceMax<T, AscendC::Pattern::Reduce::AR, false>(tmpMax, scoreUb, reduceSharedTemp, maxShape,
                                                               isSrcInnerPad);
    AscendC::PipeBarrier<PIPE_V>();

    // 更新global sum = global_sum * exp(global_max - newMax)  + exp(score - newMax).sum(dim=-1)

    // step 1: 求t3 = global_sum * exp(global_max - newMax)
    // 即：newMax = max(global_max, newMax)
    // t1 = global_max - newMax
    // t2 = torch.exp(t1)
    // t3 = global_sum * t2

    if (kChunkStart != 0) {
        AscendC::Max(tmpMax, tmpMax, globalMax, actualQExcuteRow);
    } else {
        // 第一次k循环不用max
        AscendC::Duplicate(globalMax, float(SOFTMAX_NEG_INF), actualQExcuteRow);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate(globalSum, float(0.0f), actualQExcuteRow);
    }
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::Sub(diffMax, globalMax, tmpMax, actualQExcuteRow);
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::Exp(diffMax, diffMax, actualQExcuteRow);
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::Mul(globalSum, globalSum, diffMax, actualQExcuteRow);
    AscendC::PipeBarrier<PIPE_V>();

    // step 2: 求t3 = global_sum * exp(score - newMax)
    // 即 t4 = score - tmpMax.unsqueeze(1)
    // t5 = torch.exp(t4)
    // t6 = t5.sum(dim=1)
    // global_sum = t3 + t6
    uint8_t repeatimes = BSACeilDiv(actualQExcuteRow, BROC_BASE_NUM);
    Brcb(globalMaxBroc, tmpMax, repeatimes, {1, 8});
    AscendC::PipeBarrier<PIPE_V>();

    SubBroadcast(scoreUb, scoreUb, globalMaxBroc, actualQExcuteRow, kChunkSizeAlign);
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::Exp(scoreUb, scoreUb, actualQExcuteRow * kChunkSizeAlign);
    AscendC::PipeBarrier<PIPE_V>();

    uint32_t sumShape[] = {actualQExcuteRow, kChunkSizeAlign};
    if (kChunkSizeAlign <= FLOAT_REPEAT_NUM) {
        // Set the mask in the instruction instead of relying on the high-level reuse implementation's mask state.
        AscendC::WholeReduceSum<T, true>(tmpSum, scoreUb, static_cast<int32_t>(kChunkSizeAlign),
                                         static_cast<int32_t>(actualQExcuteRow), 1, 1,
                                         static_cast<int32_t>(kChunkSizeAlign / FLOAT_DATA_BLOCK_NUM));
    } else {
        AscendC::ReduceSum<T, AscendC::Pattern::Reduce::AR, true>(tmpSum, scoreUb, reduceSharedTemp, sumShape,
                                                                  isSrcInnerPad);
    }
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::Add(globalSum, tmpSum, globalSum, actualQExcuteRow);
    AscendC::PipeBarrier<PIPE_V>();

    // 最后一次k循环第二轮循环存储sumbroc
    if (kChunkStart + kChunkSize >= validYBlocks) {
        Brcb(globalSumBroc, globalSum, repeatimes, {1, 8});
        AscendC::PipeBarrier<PIPE_V>();
    }

    // 更新globalMax
    AscendC::Max(globalMax, tmpMax, globalMax, actualQExcuteRow);
}

template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::SoftmaxSecondPassAndCast(uint32_t qChunkStart, uint32_t qChunkSize,
                                                                       uint32_t kChunkStart, uint32_t kChunkSize,
                                                                       uint32_t batchIdx, uint32_t headIdx,
                                                                       uint32_t validYBlocks)
{
    uint32_t actualQRowStart = 0, actualQExcuteRow = 0;
    if (vecSubBlockIdx == 0) {
        actualQExcuteRow = qChunkSize / 2 + qChunkSize % 2;
        actualQRowStart = constInfo.aicIdx * Q_CHUNK_SIZE;
    } else {
        actualQRowStart = constInfo.aicIdx * Q_CHUNK_SIZE + qChunkSize / 2 + qChunkSize % 2;
        actualQExcuteRow = qChunkSize / 2;
    }

    // qChunkSize 为 1 的场景
    if (actualQExcuteRow == 0) {
        return;
    }

    // 32字节对齐 (FP32)
    uint32_t kChunkSizeAlign32 = (kChunkSize + FLOAT_DATA_BLOCK_NUM - 1) / FLOAT_DATA_BLOCK_NUM * FLOAT_DATA_BLOCK_NUM;
    // 32字节对齐 (FP16)
    uint32_t kChunkSizeAlign16 = (kChunkSize + FP16_DATA_BLOCK_NUM - 1) / FP16_DATA_BLOCK_NUM * FP16_DATA_BLOCK_NUM;
    uint32_t kChunkSizeAlign = kChunkSizeAlign16;

    SetFlag<HardEvent::MTE3_MTE2>(eventSft);
    WaitFlag<HardEvent::MTE3_MTE2>(eventSft);

    SetFlag<HardEvent::V_MTE2>(eventSft);
    WaitFlag<HardEvent::V_MTE2>(eventSft);

    if (kChunkSizeAlign32 * sizeof(IN_T) % VEC_ALIGN_SIZE == 0) {
        // CAST TO FP16时候保持32字节对齐
        DataCopyPad(scoreUb, scoreFp32GmTensor[actualQRowStart * constInfo.yBlocks + kChunkStart],
                    {static_cast<uint16_t>(actualQExcuteRow), static_cast<uint32_t>(kChunkSize * sizeof(float)),
                     static_cast<uint32_t>((constInfo.yBlocks - kChunkSize) * sizeof(float)), 0, 0},
                    {true, 0, static_cast<uint8_t>(kChunkSizeAlign32 - kChunkSize), SOFTMAX_NEG_INF});
    } else {
        // CAST TO FP16时候未保持32字节对齐，需要加一个1个fp32的block
        DataCopyPad(scoreUb, scoreFp32GmTensor[actualQRowStart * constInfo.yBlocks + kChunkStart],
                    {static_cast<uint16_t>(actualQExcuteRow), static_cast<uint32_t>(kChunkSize * sizeof(float)),
                     static_cast<uint32_t>((constInfo.yBlocks - kChunkSize) * sizeof(float)), 1, 0},
                    {true, 0, static_cast<uint8_t>(kChunkSizeAlign32 - kChunkSize), SOFTMAX_NEG_INF});
    }

    SetFlag<HardEvent::MTE2_V>(eventSft);
    WaitFlag<HardEvent::MTE2_V>(eventSft);

    Muls(scoreUb, scoreUb, (float)(constInfo.scaleValue), actualQExcuteRow * kChunkSizeAlign);
    AscendC::PipeBarrier<PIPE_V>();

    SubBroadcast(scoreUb, scoreUb, globalMaxBroc, actualQExcuteRow, kChunkSizeAlign16);
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::Exp(scoreUb, scoreUb, actualQExcuteRow * kChunkSizeAlign16);
    AscendC::PipeBarrier<PIPE_V>();

    DivBroadcast(scoreUb, scoreUb, globalSumBroc, actualQExcuteRow, kChunkSizeAlign16);
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::Cast(scoreFp16Ub, scoreUb, AscendC::RoundMode::CAST_ROUND, actualQExcuteRow * kChunkSizeAlign16);
    AscendC::PipeBarrier<PIPE_V>();

    SetFlag<HardEvent::V_MTE3>(eventSft);
    WaitFlag<HardEvent::V_MTE3>(eventSft);

    uint32_t sftOutQRowOffset = 0;
    if (vecSubBlockIdx == 0) {
        sftOutQRowOffset = qChunkStart;
    } else {
        sftOutQRowOffset = qChunkStart + qChunkSize / 2 + qChunkSize % 2;
    }

    uint64_t attnScoreOffset = static_cast<uint64_t>(sftOutQRowOffset) * validYBlocks + kChunkStart;
    GlobalTensor<OUT_T> attnScoreChunkGm;
    attnScoreChunkGm.SetGlobalBuffer((__gm__ OUT_T *)(attnScoreFp16GmTensor.GetPhyAddr() + attnScoreOffset));
    DataCopyPad(attnScoreChunkGm[0], scoreFp16Ub,
                {static_cast<uint16_t>(actualQExcuteRow), static_cast<uint32_t>(kChunkSize * sizeof(OUT_T)), 0,
                 static_cast<uint32_t>((validYBlocks - kChunkSize) * sizeof(OUT_T)), 0});
}

#endif // BSA_VEC_SM_SERVICE_H
