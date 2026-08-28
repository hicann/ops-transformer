/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file bsa_vec_post_pool_service.h
 * \brief Post-Softmax Mean Pooling Service — softmax 后二次 mean pooling（输出粗粒度 pooledScore 供 TopK）
 *        （向量化 + 多核实现：px 行组按 aivIdx/aivNum 划分到各 AIV，R 维用 ReduceSum(RA) 折叠，C 维标量尾折叠）
 */
#ifndef BSA_VEC_POST_POOL_SERVICE_H
#define BSA_VEC_POST_POOL_SERVICE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "bsa_select_block_mask_common.h"
#include "bsa_select_block_mask_tiling_data.h"

// post-pool 优化路径支持的 postBlockShapeY 上限（列方向单跨度归约宽度 = colSum 缓冲容量）。
// 超过该值由 base 侧回退标量路径（现实规格 post ≤ 128，远小于该上限）。
static constexpr uint32_t BSA_POST_POOL_COL_CAP = 2048;

template <typename BSAT>
class BSAVecPostPoolService {
public:
    using T = float;
    using IN_T = half;
    using OUT_T = half;

    __aicore__ inline BSAVecPostPoolService(){};

    __aicore__ inline void InitParams(const BSAConstInfo &constInfo,
                                      const optiling::BSASelectBlockMaskTilingData *__restrict tilingData);
    __aicore__ inline void InitBuffers(TBuf<> *uBuf_);
    __aicore__ inline void InitPostPoolGM(GlobalTensor<IN_T> &attnScoreGm, GlobalTensor<OUT_T> &pooledScoreGm);

    // 多核入口：按 aivIdx/aivNum 将 postX 方向的 px 行组划分到各 AIV
    __aicore__ inline void PostPoolRange(uint32_t batchIdx, uint32_t headIdx, uint32_t validXBlocks,
                                         uint32_t validYBlocks);
    // 标量兜底路径的 scratch 访问（与 post-pool 相位同一 UB 区域，postBlockShapeY > 上限时由 base 侧调用）
    __aicore__ inline LocalTensor<IN_T> GetScalarScratch();

private:
    // ---- post-pool 相位缓冲（uBuf_ 从 offset 0 carve）----
    static constexpr uint32_t TILE_ELEMS = 16384; // readUb(half)/colFp32Ub(fp32) 元素上限：32KB + 64KB
    static constexpr uint32_t REDUCE_TMP_BYTES = BSAConstInfo::BUFFER_SIZE_BYTE_24K;
    static constexpr uint32_t POOLED_ROW_ELEMS = BSA_POST_POOL_COL_CAP / 8; // SC/C ≤ 256（C ≥ 8）

    __aicore__ inline void CalcPostPxRange(uint32_t &pxStart, uint32_t &pxEnd, uint32_t validPostX);

    BSAConstInfo constInfo;
    const optiling::BSASelectBlockMaskTilingData *__restrict tilingData;

    GlobalTensor<IN_T> attnScoreGmTensor;
    GlobalTensor<OUT_T> pooledScoreGmTensor;

    TBuf<> *uBufPtr = nullptr;
    // post-pool 相位
    LocalTensor<IN_T> readUb;
    LocalTensor<T> colFp32Ub;
    LocalTensor<T> colSumUb;
    LocalTensor<T> colSumTmpUb;
    LocalTensor<uint8_t> reduceSharedTemp;
    LocalTensor<OUT_T> pooledRowUb;
};

template <typename BSAT>
__aicore__ inline void BSAVecPostPoolService<BSAT>::InitParams(
    const BSAConstInfo &constInfo, const optiling::BSASelectBlockMaskTilingData *__restrict tilingData)
{
    this->constInfo = constInfo;
    this->tilingData = tilingData;
}

template <typename BSAT>
__aicore__ inline void BSAVecPostPoolService<BSAT>::InitBuffers(TBuf<> *uBuf_)
{
    uBufPtr = uBuf_;
    uint32_t ubOffset = 0;

    // ================= post-pool 相位（合计 ~136.5KB < 192KB）=================
    readUb = uBuf_->GetWithOffset<IN_T>(TILE_ELEMS, ubOffset);
    ubOffset += TILE_ELEMS * sizeof(IN_T);
    ubOffset = BSAAlignTo(ubOffset, static_cast<uint32_t>(VEC_ALIGN_SIZE));

    colFp32Ub = uBuf_->GetWithOffset<T>(TILE_ELEMS, ubOffset);
    ubOffset += TILE_ELEMS * sizeof(T);
    ubOffset = BSAAlignTo(ubOffset, static_cast<uint32_t>(VEC_ALIGN_SIZE));

    colSumUb = uBuf_->GetWithOffset<T>(BSA_POST_POOL_COL_CAP, ubOffset);
    ubOffset += BSA_POST_POOL_COL_CAP * sizeof(T);
    colSumTmpUb = uBuf_->GetWithOffset<T>(BSA_POST_POOL_COL_CAP, ubOffset);
    ubOffset += BSA_POST_POOL_COL_CAP * sizeof(T);
    ubOffset = BSAAlignTo(ubOffset, static_cast<uint32_t>(VEC_ALIGN_SIZE));

    reduceSharedTemp = uBuf_->GetWithOffset<uint8_t>(REDUCE_TMP_BYTES, ubOffset);
    ubOffset += REDUCE_TMP_BYTES;
    ubOffset = BSAAlignTo(ubOffset, static_cast<uint32_t>(VEC_ALIGN_SIZE));

    pooledRowUb = uBuf_->GetWithOffset<OUT_T>(POOLED_ROW_ELEMS, ubOffset);
}

template <typename BSAT>
__aicore__ inline void BSAVecPostPoolService<BSAT>::InitPostPoolGM(GlobalTensor<IN_T> &attnScoreGm,
                                                                   GlobalTensor<OUT_T> &pooledScoreGm)
{
    this->attnScoreGmTensor = attnScoreGm;
    this->pooledScoreGmTensor = pooledScoreGm;
}

template <typename BSAT>
__aicore__ inline LocalTensor<typename BSAVecPostPoolService<BSAT>::IN_T>
BSAVecPostPoolService<BSAT>::GetScalarScratch()
{
    return readUb;
}

template <typename BSAT>
__aicore__ inline void BSAVecPostPoolService<BSAT>::CalcPostPxRange(uint32_t &pxStart, uint32_t &pxEnd,
                                                                    uint32_t validPostX)
{
    pxStart = 0;
    pxEnd = 0;
    if (validPostX == 0) {
        return;
    }
    // 与 radix TopK 相同的多核基准（aivIdx / aivNum）
    uint32_t coreIdx = constInfo.aivIdx;
    uint32_t actualCores = BSAMin(validPostX, constInfo.aivNum);
    if (coreIdx >= actualCores) {
        return;
    }
    uint32_t base = validPostX / actualCores;
    uint32_t extra = validPostX % actualCores;
    if (coreIdx < extra) {
        pxStart = coreIdx * (base + 1);
        pxEnd = pxStart + base + 1;
    } else {
        pxStart = extra * (base + 1) + (coreIdx - extra) * base;
        pxEnd = pxStart + base;
    }
}

template <typename BSAT>
__aicore__ inline void BSAVecPostPoolService<BSAT>::PostPoolRange(uint32_t batchIdx, uint32_t headIdx,
                                                                  uint32_t validXBlocks, uint32_t validYBlocks)
{
    uint32_t R = constInfo.postBlockShapeX;
    uint32_t C = constInfo.postBlockShapeY;
    uint32_t validPostX = BSACeilDiv(validXBlocks, R);
    uint32_t validPostY = BSACeilDiv(validYBlocks, C);

    uint32_t pxStart, pxEnd;
    CalcPostPxRange(pxStart, pxEnd, validPostX);
    if (pxStart >= pxEnd) {
        return;
    }

    // attnScore workspace 为单 head 复用的紧凑 scratch [validXBlocks × validYBlocks]（head 偏移 0）；
    // pooledScore 逐 head 布局：head 间距为最大粗粒度网格 postXBlocks*postYBlocks，
    // head 内部为紧凑 [validPostX × validPostY]（与 TopK 线性读取一致）。
    uint64_t pooledHeadOffset = static_cast<uint64_t>(batchIdx) * constInfo.numHeads;
    pooledHeadOffset = (pooledHeadOffset + headIdx) * constInfo.postXBlocks * constInfo.postYBlocks;

    // 列跨度 SC：C 的倍数且 16 对齐（保证 2D 搬运 blockLen 为 32B 倍数、UB 行布局紧凑无隐藏 padding）
    uint32_t spanGroups = BSA_POST_POOL_COL_CAP / C;
    if (C % 16 != 0 && spanGroups > 1 && (spanGroups % 2) != 0) {
        spanGroups -= 1; // C 仅 8 对齐时取偶数倍组，保证 SC % 16 == 0
    }
    uint32_t SC = spanGroups * C;

    for (uint32_t px = pxStart; px < pxEnd; px++) {
        uint32_t xStart = px * R;
        uint32_t numFineRows = BSAMin(R, validXBlocks - xStart);

        for (uint32_t colBase = 0; colBase < validYBlocks; colBase += SC) {
            uint32_t scValid = BSAMin(SC, validYBlocks - colBase);
            uint32_t sPad = (scValid + 15) / 16 * 16; // 16 half 对齐 = 32B
            uint32_t rightPad = sPad - scValid;

            // 行子块：rows * sPad ≤ TILE_ELEMS
            uint32_t rowBatch = BSAMax<uint32_t>(1, TILE_ELEMS / sPad);
            bool colSumValid = false;

            for (uint32_t rowBase = 0; rowBase < numFineRows; rowBase += rowBatch) {
                uint32_t rows = BSAMin(rowBatch, numFineRows - rowBase);
                uint64_t gmRowBase = static_cast<uint64_t>(xStart + rowBase) * validYBlocks + colBase;

                // 逐行 1D 搬运：显式控制 UB 侧行偏移（sPad 对齐），语义无歧义；
                // rightPad 显式补零到 sPad（ReduceSum 列数须 8 对齐）
                DataCopyPadExtParams<IN_T> pad1D{true, 0, static_cast<uint8_t>(rightPad), 0};
                for (uint32_t r = 0; r < rows; r++) {
                    DataCopyPad(readUb[r * sPad], attnScoreGmTensor[gmRowBase + r * validYBlocks],
                                DataCopyExtParams(1, scValid * sizeof(IN_T), 0, 0, 0), pad1D);
                }
                AscendC::PipeBarrier<PIPE_ALL>(); // MTE2 → V/标量（N4 教训：标量/向量消费前须全屏障）

                Cast(colFp32Ub, readUb, RoundMode::CAST_NONE, rows * sPad);
                AscendC::PipeBarrier<PIPE_V>();

                uint32_t shape[] = {rows, sPad};
                bool isSrcInnerPad = (rows % FLOAT_DATA_BLOCK_NUM == 0);
                if (!colSumValid) {
                    AscendC::ReduceSum<T, AscendC::Pattern::Reduce::RA, true>(colSumUb, colFp32Ub, reduceSharedTemp,
                                                                              shape, isSrcInnerPad);
                    colSumValid = true;
                } else {
                    AscendC::ReduceSum<T, AscendC::Pattern::Reduce::RA, true>(colSumTmpUb, colFp32Ub, reduceSharedTemp,
                                                                              shape, isSrcInnerPad);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(colSumUb, colSumUb, colSumTmpUb, sPad);
                }
                AscendC::PipeBarrier<PIPE_ALL>(); // V → 标量折叠
            }

            // C 维标量尾折叠：列和 → 组均值（每列仅读一次，总量 = validYBlocks）
            uint32_t groupBase = colBase / C;
            uint32_t groupsInSpan = BSACeilDiv(scValid, C);
            for (uint32_t g = 0; g < groupsInSpan; g++) {
                uint32_t py = groupBase + g;
                uint32_t cActual = BSAMin(C, validYBlocks - py * C);
                float totalSum = 0.0f;
                for (uint32_t c = 0; c < cActual; c++) {
                    totalSum += colSumUb.GetValue(g * C + c);
                }
                float count = static_cast<float>(static_cast<int32_t>(numFineRows * cActual));
                pooledRowUb.SetValue(g, static_cast<OUT_T>(totalSum / count));
            }
            AscendC::PipeBarrier<PIPE_ALL>(); // 标量写 → MTE3 读

            // 精确长度写出（CopyOut 自动丢弃 dummy，无跨核越界踩踏）；跨核 px 行组不相交。
            // 调用点（base.h）在返回后统一 PipeBarrier<PIPE_ALL> + SyncAll，覆盖本 MTE3 排空
            DataCopyPad(pooledScoreGmTensor[pooledHeadOffset + static_cast<uint64_t>(px) * validPostY + groupBase],
                        pooledRowUb, DataCopyExtParams(1, groupsInSpan * sizeof(OUT_T), 0, 0, 0));
        }
    }
}

#endif // BSA_VEC_POST_POOL_SERVICE_H
