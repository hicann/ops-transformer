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
 * \file moe_finalize_routing_v2_row_k_h_full_load.h
 * \brief
 */

#ifndef MOE_FINALIZE_ROUTING_V2_ROW_K_H_FULL_LOAD_H_
#define MOE_FINALIZE_ROUTING_V2_ROW_K_H_FULL_LOAD_H_

#include "moe_finalize_routing_v2_base.h"
namespace MoeFinalizeRoutingV2Regbase {
using namespace AscendC;
template <typename T, typename S, int32_t dropPadMode>
class MoeFinalizeRoutingV2RowKHFullLoad {
public:
    __aicore__ inline MoeFinalizeRoutingV2RowKHFullLoad() {}

    // expandX/bias 全部载入ub，避免多次小段搬运, 此时对应的Que不开db
    __aicore__ inline void Init(GM_ADDR expandedX, GM_ADDR expandedRowIdx, GM_ADDR x1, GM_ADDR x2, GM_ADDR bias,
                                GM_ADDR scales, GM_ADDR expertIdx, GM_ADDR x, GM_ADDR constExpertAlpha1,
                                GM_ADDR constExpertAlpha2, GM_ADDR v, GM_ADDR y, GM_ADDR workspace,
                                const MoeFinalizeRoutingV2RegbaseTilingData *tilingDataPtr, TPipe *pipePtr)
    {
        pipe = pipePtr;
        tilingData = tilingDataPtr;

        hasX1 = (x1 != nullptr);
        hasX2 = (x2 != nullptr);
        hasBiasAndExpertIdx = (bias != nullptr) && (expertIdx != nullptr);
        hasExpertIdx = expertIdx != nullptr;
        hasScales = scales != nullptr;
        hasX = (x != nullptr);
        hasConstExpert = (constExpertAlpha1 != nullptr) && (constExpertAlpha2 != nullptr) && (v != nullptr);

        SetGlobalBuffers(expandedX, expandedRowIdx, x1, x2, bias, scales, expertIdx, x, constExpertAlpha1,
                         constExpertAlpha2, v, y);
        InitQueBuffers();
    }

    __aicore__ inline void Process()
    {
        LoadExpandedX();
        LoadBiasAndAllocOptionalTensors();

        int64_t rowOuterLoop =
            (GetBlockIdx() == GetBlockNum() - 1) ? tilingData->rowLoopOfTailBlock : tilingData->rowLoopOfFormerBlock;
        int64_t tailRowFactor = (GetBlockIdx() == GetBlockNum() - 1) ? tilingData->tailRowFactorOfTailBlock :
                                                                       tilingData->tailRowFactorOfFormerBlock;
        for (int64_t rowOuterIdx = 0; rowOuterIdx < rowOuterLoop; rowOuterIdx += 1) {
            int64_t rowInnerLoop = (rowOuterIdx == rowOuterLoop - 1) ? tailRowFactor : tilingData->rowFactor;
            ProcessLoopBody(rowOuterIdx, rowInnerLoop);
        }

        FreePersistentTensors();
    }

private:
    __aicore__ inline void LoadExpandedX()
    {
        expandedXLocal = expandedXQue.AllocTensor<T>();
        if constexpr (dropPadMode == DROPLESS_COLUMN || dropPadMode == DROPLESS_ROW) {
            CopyIn(expandedXGm, expandedXLocal, 1, tilingData->row * tilingData->k * tilingData->h);
        } else {
            CopyIn(expandedXGm, expandedXLocal, 1, tilingData->e * tilingData->c * tilingData->h);
        }
        expandedXQue.EnQue(expandedXLocal);
        expandedXLocal = expandedXQue.DeQue<T>();
    }

    __aicore__ inline void LoadBiasAndAllocOptionalTensors()
    {
        if (hasBiasAndExpertIdx) {
            biasLocal = biasQue.AllocTensor<T>();
            CopyIn(biasGm, biasLocal, 1, tilingData->e * tilingData->h);
            biasQue.EnQue(biasLocal);
            biasLocal = biasQue.DeQue<T>();
        }
        if (hasX) {
            xLocal = xQue.AllocTensor<T>();
        }
        if (hasConstExpert) {
            constExpertAlpha1Local = constExpertAlpha1Que.AllocTensor<T>();
            constExpertAlpha2Local = constExpertAlpha2Que.AllocTensor<T>();
            vLocal = vQue.AllocTensor<T>();
        }
    }

    __aicore__ inline void ProcessLoopBody(int64_t rowOuterIdx, int64_t rowInnerLoop)
    {
        ProcessInputWithX(rowOuterIdx, rowInnerLoop);

        yLocal = yQue.AllocTensor<float>();
        ProcessX1AndX2(yLocal, x1Local, x2Local, rowInnerLoop * tilingData->h, hasX1, hasX2);
        ProcessYWithInput(rowOuterIdx, rowInnerLoop);
        yQue.EnQue(yLocal);

        if (hasX1) {
            x1Que.FreeTensor(x1Local);
        }
        if (hasScales) {
            scalesQue.FreeTensor(scalesLocal);
        }
        if (hasX2) {
            x2Que.FreeTensor(x2Local);
        }
        yLocal = yQue.DeQue<float>();
        int64_t yGmOffset = GetBlockIdx() * tilingData->rowOfFormerBlock * tilingData->h +
                            rowOuterIdx * tilingData->rowFactor * tilingData->h;
        CopyOut(yLocal.template ReinterpretCast<T>(), yGm[yGmOffset], 1, rowInnerLoop * tilingData->h);
        yQue.FreeTensor(yLocal);
    }

    __aicore__ inline void FreePersistentTensors()
    {
        expandedXQue.FreeTensor(expandedXLocal);
        if (hasBiasAndExpertIdx) {
            biasQue.FreeTensor(biasLocal);
        }
        if (hasX) {
            xQue.FreeTensor(xLocal);
        }
        if (hasConstExpert) {
            constExpertAlpha1Que.FreeTensor(constExpertAlpha1Local);
            constExpertAlpha2Que.FreeTensor(constExpertAlpha2Local);
            vQue.FreeTensor(vLocal);
        }
    }
    __aicore__ inline void ProcessInputWithX(int64_t rowOuterIdx, int64_t rowInnerLoop)
    {
        if (hasX2) {
            x2Local = x2Que.AllocTensor<T>();
            int64_t x2GmOffset = GetBlockIdx() * tilingData->rowOfFormerBlock * tilingData->h +
                                 rowOuterIdx * tilingData->rowFactor * tilingData->h;
            CopyIn(x2Gm[x2GmOffset], x2Local, 1, rowInnerLoop * tilingData->h);
            x2Que.EnQue(x2Local);
            x2Local = x2Que.DeQue<T>();
        }
        if (hasScales) {
            scalesLocal = scalesQue.AllocTensor<S>();
            int64_t scaleGmOffset = GetBlockIdx() * tilingData->rowOfFormerBlock * tilingData->k +
                                    rowOuterIdx * tilingData->rowFactor * tilingData->k;
            CopyIn(scalesGm[scaleGmOffset], scalesLocal, 1, rowInnerLoop * tilingData->k);
            scalesQue.EnQue(scalesLocal);
            scalesLocal = scalesQue.DeQue<S>();
        }
        if (hasX1) {
            x1Local = x1Que.AllocTensor<T>();
            int64_t x1GmOffset = GetBlockIdx() * tilingData->rowOfFormerBlock * tilingData->h +
                                 rowOuterIdx * tilingData->rowFactor * tilingData->h;
            CopyIn(x1Gm[x1GmOffset], x1Local, 1, rowInnerLoop * tilingData->h);
            x1Que.EnQue(x1Local);
            x1Local = x1Que.DeQue<T>();
        }
    }

    __aicore__ inline void ProcessYWithInput(int64_t rowOuterIdx, int64_t rowInnerLoop)
    {
        for (int64_t rowInnerIdx = 0; rowInnerIdx < rowInnerLoop; rowInnerIdx += 1) {
            for (int64_t kIdx = 0; kIdx < tilingData->k; kIdx += 1) {
                ProcessInnerLoopBody(rowOuterIdx, rowInnerIdx, kIdx);
            }
        }
        CastYOutput(rowInnerLoop);
    }

    __aicore__ inline void ProcessInnerLoopBody(int64_t rowOuterIdx, int64_t rowInnerIdx, int64_t kIdx)
    {
        SetOffsetForExpandedRowIdx(rowOuterIdx, kIdx, rowInnerIdx);
        int64_t expandedRowIdxGmValue = expandedRowIdxGm.GetValue(expandedRowIdxOffset);
        if (expandedRowIdxGmValue == INVALID_IDX) {
            return;
        }
        if constexpr (dropPadMode != DROP_PAD_COLUMN && dropPadMode != DROP_PAD_ROW) {
            if (expandedRowIdxGmValue >= tilingData->activeNum) {
                return;
            }
        }
        int64_t expertIdx = -1;
        if (hasExpertIdx) {
            SetExpertIdxOffset(rowOuterIdx, rowInnerIdx, kIdx);
            expertIdx = expertIdxGm.GetValue(expertIdxOffset);
        }

        if (hasBiasAndExpertIdx) {
            biasGmOffset = expertIdx * tilingData->h;
        }
        if (hasScales) {
            int64_t scaleOffset = rowInnerIdx * tilingData->k + kIdx;
            scale = scalesLocal.GetValue(scaleOffset);
        }
        LocalTensor<T> innerBiasLocal = hasBiasAndExpertIdx ? biasLocal[biasGmOffset] : expandedXLocal;
        if (hasX && hasExpertIdx) {
            if (expertIdx >= tilingData->zeroExpertStart && expertIdx < tilingData->zeroExpertEnd) {
                return;
            }
            if (expertIdx >= tilingData->copyExpertStart && expertIdx < tilingData->copyExpertEnd) {
                ProcessCopyExpert(rowOuterIdx, rowInnerIdx, innerBiasLocal, scale);
            } else if (hasConstExpert && expertIdx >= tilingData->constantExpertStart &&
                       expertIdx < tilingData->constantExpertEnd) {
                ProcessConstantExpert(rowOuterIdx, rowInnerIdx, expertIdx, innerBiasLocal, scale);
            } else {
                ProcessExpandXBiasScale<T, S>(yLocal[rowInnerIdx * tilingData->h],
                                              expandedXLocal[expandedRowIdxGmValue * tilingData->h], innerBiasLocal,
                                              scale, tilingData->h, hasBiasAndExpertIdx, hasScales);
            }
        } else {
            ProcessExpandXBiasScale<T, S>(yLocal[rowInnerIdx * tilingData->h],
                                          expandedXLocal[expandedRowIdxGmValue * tilingData->h], innerBiasLocal, scale,
                                          tilingData->h, hasBiasAndExpertIdx, hasScales);
        }
    }

    __aicore__ inline void ProcessCopyExpert(int64_t rowOuterIdx, int64_t rowInnerIdx,
                                             const LocalTensor<T> &innerBiasLocal, S scale)
    {
        int64_t xGmOffset = GetBlockIdx() * tilingData->rowOfFormerBlock * tilingData->h +
                            rowOuterIdx * tilingData->rowFactor * tilingData->h + rowInnerIdx * tilingData->h;
        CopyIn(xGm[xGmOffset], xLocal, 1, tilingData->h);
        xQue.EnQue(xLocal);
        xLocal = xQue.DeQue<T>();
        ProcessExpandXBiasScale<T, S>(yLocal[rowInnerIdx * tilingData->h], xLocal, innerBiasLocal, scale, tilingData->h,
                                      hasBiasAndExpertIdx, hasScales);
    }

    __aicore__ inline void ProcessConstantExpert(int64_t rowOuterIdx, int64_t rowInnerIdx, int64_t expertIdx,
                                                 const LocalTensor<T> &innerBiasLocal, S scale)
    {
        int64_t xGmOffset = GetBlockIdx() * tilingData->rowOfFormerBlock * tilingData->h +
                            rowOuterIdx * tilingData->rowFactor * tilingData->h + rowInnerIdx * tilingData->h;
        CopyIn(xGm[xGmOffset], xLocal, 1, tilingData->h);

        int64_t constExpertGmOffset = (expertIdx - tilingData->constantExpertStart) * tilingData->h;
        CopyIn(constExpertAlpha1Gm[constExpertGmOffset], constExpertAlpha1Local, 1, tilingData->h);
        CopyIn(constExpertAlpha2Gm[constExpertGmOffset], constExpertAlpha2Local, 1, tilingData->h);
        CopyIn(vGm[constExpertGmOffset], vLocal, 1, tilingData->h);
        PipeBarrier<PIPE_ALL>();
        vLocal = vLocal * constExpertAlpha2Local;
        xLocal = xLocal * constExpertAlpha1Local;
        xLocal = xLocal + vLocal;
        constExpertAlpha1Que.EnQue(constExpertAlpha1Local);
        constExpertAlpha1Local = constExpertAlpha1Que.DeQue<T>();
        constExpertAlpha2Que.EnQue(constExpertAlpha2Local);
        constExpertAlpha2Local = constExpertAlpha2Que.DeQue<T>();
        vQue.EnQue(vLocal);
        vLocal = vQue.DeQue<T>();
        xQue.EnQue(xLocal);
        xLocal = xQue.DeQue<T>();
        ProcessExpandXBiasScale<T, S>(yLocal[rowInnerIdx * tilingData->h], xLocal, innerBiasLocal, scale, tilingData->h,
                                      hasBiasAndExpertIdx, hasScales);
    }

    __aicore__ inline void CastYOutput(int64_t rowInnerLoop)
    {
        if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
            Cast(yLocal.template ReinterpretCast<T>(), yLocal, RoundMode::CAST_RINT, rowInnerLoop * tilingData->h);
        }
    }

    __aicore__ inline void SetOffsetForExpandedRowIdx(int64_t rowOuterIdx, int64_t kIdx, int64_t rowInnerIdx)
    {
        if constexpr (dropPadMode != DROPLESS_COLUMN && dropPadMode != DROP_PAD_COLUMN) {
            expandedRowIdxOffset = GetBlockIdx() * tilingData->rowOfFormerBlock * tilingData->k +
                                   rowOuterIdx * tilingData->rowFactor * tilingData->k + rowInnerIdx * tilingData->k +
                                   kIdx;
            return;
        }
        expandedRowIdxOffset = GetBlockIdx() * tilingData->rowOfFormerBlock + rowOuterIdx * tilingData->rowFactor +
                               rowInnerIdx + kIdx * tilingData->row;
        return;
    }

    __aicore__ inline void SetExpertIdxOffset(int64_t rowOuterIdx, int64_t rowInnerIdx, int64_t kIdx)
    {
        if constexpr (dropPadMode == DROPLESS_COLUMN || dropPadMode == DROP_PAD_COLUMN) {
            expertIdxOffset = GetBlockIdx() * tilingData->rowOfFormerBlock * tilingData->k +
                              rowOuterIdx * tilingData->rowFactor * tilingData->k + rowInnerIdx * tilingData->k + kIdx;
        } else {
            expertIdxOffset = expandedRowIdxOffset;
        }
    }

    __aicore__ inline void SetGlobalBuffers(GM_ADDR expandedX, GM_ADDR expandedRowIdx, GM_ADDR x1, GM_ADDR x2,
                                            GM_ADDR bias, GM_ADDR scales, GM_ADDR expertIdx, GM_ADDR x,
                                            GM_ADDR constExpertAlpha1, GM_ADDR constExpertAlpha2, GM_ADDR v, GM_ADDR y)
    {
        expandedXGm.SetGlobalBuffer((__gm__ T *)expandedX);
        expandedRowIdxGm.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx);
        x1Gm.SetGlobalBuffer((__gm__ T *)x1);
        x2Gm.SetGlobalBuffer((__gm__ T *)x2);
        biasGm.SetGlobalBuffer((__gm__ T *)bias);
        scalesGm.SetGlobalBuffer((__gm__ S *)scales);
        expertIdxGm.SetGlobalBuffer((__gm__ int32_t *)expertIdx);
        xGm.SetGlobalBuffer((__gm__ T *)x);
        constExpertAlpha1Gm.SetGlobalBuffer((__gm__ T *)constExpertAlpha1);
        constExpertAlpha2Gm.SetGlobalBuffer((__gm__ T *)constExpertAlpha2);
        vGm.SetGlobalBuffer((__gm__ T *)v);
        yGm.SetGlobalBuffer((__gm__ T *)y);
    }

    __aicore__ inline void InitQueBuffers()
    {
        int32_t eHAligned = RoundUp<T>(tilingData->e * tilingData->h);
        int32_t rowFactorHAlignedFloat = RoundUp<float>(tilingData->rowFactor * tilingData->h);
        int32_t rowFactorKAlignedT = RoundUp<T>(tilingData->rowFactor * tilingData->k);
        int32_t rowFactorHAlignedT = RoundUp<T>(tilingData->rowFactor * tilingData->h);
        if constexpr (dropPadMode == DROPLESS_COLUMN || dropPadMode == DROPLESS_ROW) {
            int32_t rowsKHAligned = RoundUp<T>(tilingData->row * tilingData->k * tilingData->h);
            pipe->InitBuffer(expandedXQue, 1, rowsKHAligned * sizeof(T));
        } else {
            int32_t eCHAligned = RoundUp<T>(tilingData->e * tilingData->c * tilingData->h);
            pipe->InitBuffer(expandedXQue, 1, eCHAligned * sizeof(T));
        }
        pipe->InitBuffer(yQue, DOUBLE_BUFFER, rowFactorHAlignedFloat * sizeof(float));
        if (hasX1) {
            pipe->InitBuffer(x1Que, DOUBLE_BUFFER, rowFactorHAlignedT * sizeof(T));
        }
        if (hasBiasAndExpertIdx) {
            pipe->InitBuffer(biasQue, 1, eHAligned * sizeof(T));
        }
        if (hasX2) {
            pipe->InitBuffer(x2Que, DOUBLE_BUFFER, rowFactorHAlignedT * sizeof(T));
        }
        if (hasScales) {
            pipe->InitBuffer(scalesQue, DOUBLE_BUFFER, rowFactorKAlignedT * sizeof(S));
        }
        if (hasX) {
            pipe->InitBuffer(xQue, DOUBLE_BUFFER, tilingData->h * sizeof(T));
        }
        if (hasConstExpert) {
            pipe->InitBuffer(constExpertAlpha1Que, DOUBLE_BUFFER, tilingData->h * sizeof(T));
            pipe->InitBuffer(constExpertAlpha2Que, DOUBLE_BUFFER, tilingData->h * sizeof(T));
            pipe->InitBuffer(vQue, DOUBLE_BUFFER, tilingData->h * sizeof(T));
        }
    }

    TPipe *pipe;
    const MoeFinalizeRoutingV2RegbaseTilingData *tilingData;

    S scale;
    bool hasX1{false};
    bool hasX2{false};
    bool hasBiasAndExpertIdx{false};
    bool hasExpertIdx{false};
    bool hasScales{false};
    bool hasX{false};
    bool hasConstExpert{false};
    int64_t biasGmOffset{0};

    GlobalTensor<T> expandedXGm;
    GlobalTensor<int32_t> expandedRowIdxGm;
    GlobalTensor<T> x1Gm;
    GlobalTensor<T> x2Gm;
    GlobalTensor<T> biasGm;
    GlobalTensor<S> scalesGm;
    GlobalTensor<int32_t> expertIdxGm;
    GlobalTensor<T> xGm;
    GlobalTensor<T> constExpertAlpha1Gm;
    GlobalTensor<T> constExpertAlpha2Gm;
    GlobalTensor<T> vGm;

    GlobalTensor<T> yGm;

    TQue<QuePosition::VECIN, 1> expandedXQue;
    TQue<QuePosition::VECIN, 1> biasQue;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> x1Que;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> x2Que;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> scalesQue;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> yQue;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> xQue;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> constExpertAlpha1Que;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> constExpertAlpha2Que;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> vQue;

    LocalTensor<T> expandedXLocal;
    LocalTensor<T> x1Local;
    LocalTensor<T> x2Local;
    LocalTensor<T> biasLocal;
    LocalTensor<S> scalesLocal;
    LocalTensor<float> yLocal;
    LocalTensor<T> xLocal;
    LocalTensor<T> constExpertAlpha1Local;
    LocalTensor<T> constExpertAlpha2Local;
    LocalTensor<T> vLocal;

    int64_t expandedRowIdxOffset{0};
    int64_t expertIdxOffset{0};
};
} // namespace MoeFinalizeRoutingV2Regbase

#endif
