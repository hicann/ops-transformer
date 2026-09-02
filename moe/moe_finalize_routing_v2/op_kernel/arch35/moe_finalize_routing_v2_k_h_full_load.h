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
 * \file moe_finalize_routing_v2_k_h_full_load.h
 * \brief
 */

#ifndef MOE_FINALIZE_ROUTING_V2_K_H_FULL_LOAD_H_
#define MOE_FINALIZE_ROUTING_V2_K_H_FULL_LOAD_H_

#include "moe_finalize_routing_v2_base.h"
namespace MoeFinalizeRoutingV2Regbase {
using namespace AscendC;

constexpr int64_t BATCH_COPY_SIZE = 16;
constexpr int64_t BATCH_COPY_EXPERT_NUM = 4;

template <typename T, typename S, int32_t dropPadMode>
class MoeFinalizeRoutingV2KHFullLoad {
public:
    __aicore__ inline MoeFinalizeRoutingV2KHFullLoad() {}

    // expandedX/bias k*h全载，需要fork次搬入
    __aicore__ inline void Init(GM_ADDR expandedX, GM_ADDR expandedRowIdx, GM_ADDR x1, GM_ADDR x2, GM_ADDR bias,
                                GM_ADDR scales, GM_ADDR expertIdx, GM_ADDR x, GM_ADDR constExpertAlpha1,
                                GM_ADDR constExpertAlpha2, GM_ADDR v, GM_ADDR y, GM_ADDR workspace,
                                const MoeFinalizeRoutingV2RegbaseTilingData *tilingDataPtr, TPipe *pipePtr)
    {
        pipe = pipePtr;
        tilingData = tilingDataPtr;

        k = tilingData->k;
        h = tilingData->h;
        hAligned = tilingData->hAligned;
        khAligned = k * hAligned;
        hasX1 = x1 != nullptr;
        hasX2 = x2 != nullptr;
        hasExpertIdx = (expertIdx != nullptr);
        hasBiasAndExpertIdx = (bias != nullptr) && (expertIdx != nullptr);
        hasScales = scales != nullptr;
        hasX = (x != nullptr);
        hasConstExpert = (constExpertAlpha1 != nullptr) && (constExpertAlpha2 != nullptr) && (v != nullptr);
        k1 = k == 1;
        k4 = k <= BATCH_COPY_EXPERT_NUM;

        SetGlobalBuffers(x1, x2, bias, scales, expertIdx, y, expandedX, x, constExpertAlpha1, constExpertAlpha2, v,
                         expandedRowIdx);
        InitQueBuffers();
    }

    __aicore__ inline void Process()
    {
        int64_t tailRowFactor = (GetBlockIdx() == GetBlockNum() - 1) ? tilingData->tailRowFactorOfTailBlock :
                                                                       tilingData->tailRowFactorOfFormerBlock;
        int64_t rowOuterLoop =
            (GetBlockIdx() == GetBlockNum() - 1) ? tilingData->rowLoopOfTailBlock : tilingData->rowLoopOfFormerBlock;
        for (int64_t rowOuterIdx = 0; rowOuterIdx < rowOuterLoop; rowOuterIdx += 1) {
            int64_t rowInnerLoop = (rowOuterIdx == rowOuterLoop - 1) ? tailRowFactor : tilingData->rowFactor;
            ProcessInputWithX(rowOuterIdx, rowInnerLoop);
            ProcessWithId(rowOuterIdx, rowInnerLoop);

            yLocal = yQue.AllocTensor<float>();
            ProcessX1AndX2(yLocal, x1Local, x2Local, rowInnerLoop * h, hasX1, hasX2);
            if (k1) {
                ProcessYWithInputK1(rowOuterIdx, rowInnerLoop);
            } else if (k4 && tilingData->rowFactor >= BATCH_COPY_SIZE) {
                ProcessYWithInputK4(rowOuterIdx, rowInnerLoop);
            } else {
                ProcessYWithInput(rowOuterIdx, rowInnerLoop);
            }

            yQue.EnQue(yLocal);

            if (hasX1) {
                x1Que.FreeTensor(x1Local);
            }
            if (hasX2) {
                x2Que.FreeTensor(x2Local);
            }
            if (hasScales) {
                scalesQue.FreeTensor(scalesLocal);
            }

            if (k1) {
                expandedRowIdxQue.FreeTensor(expandedRowIdxLocal);
                if (hasBiasAndExpertIdx) {
                    expertIdxQue.FreeTensor(expertIdxLocal);
                }
            }
            yLocal = yQue.DeQue<float>();
            int64_t yGmOffset =
                GetBlockIdx() * tilingData->rowOfFormerBlock * h + rowOuterIdx * tilingData->rowFactor * h;
            CopyOut(yLocal.template ReinterpretCast<T>(), yGm[yGmOffset], 1, rowInnerLoop * h);
            yQue.FreeTensor(yLocal);
        }
    }

private:
    __aicore__ inline void SetGlobalBuffers(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR scales, GM_ADDR expertIdx,
                                            GM_ADDR y, GM_ADDR expandedX, GM_ADDR x, GM_ADDR constExpertAlpha1,
                                            GM_ADDR constExpertAlpha2, GM_ADDR v, GM_ADDR expandedRowIdx)
    {
        x1Gm.SetGlobalBuffer((__gm__ T *)x1);
        x2Gm.SetGlobalBuffer((__gm__ T *)x2);
        biasGm.SetGlobalBuffer((__gm__ T *)bias);
        scalesGm.SetGlobalBuffer((__gm__ S *)scales);
        expertIdxGm.SetGlobalBuffer((__gm__ int32_t *)expertIdx);
        yGm.SetGlobalBuffer((__gm__ T *)y);
        expandedXGm.SetGlobalBuffer((__gm__ T *)expandedX);
        xGm.SetGlobalBuffer((__gm__ T *)x);
        constExpertAlpha1Gm.SetGlobalBuffer((__gm__ T *)constExpertAlpha1);
        constExpertAlpha2Gm.SetGlobalBuffer((__gm__ T *)constExpertAlpha2);
        vGm.SetGlobalBuffer((__gm__ T *)v);
        expandedRowIdxGm.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx);
    }

    __aicore__ inline void InitQueBuffers()
    {
        int32_t rowFactorHAlignedT = RoundUp<T>(tilingData->rowFactor * h);
        int32_t rowFactorHAlignedFloat = RoundUp<float>(tilingData->rowFactor * h);
        int32_t rowFactorKAlignedT = RoundUp<T>(tilingData->rowFactor * k);
        int32_t rowFactorKAlignedTInt32 = RoundUp<int32_t>(tilingData->rowFactor * k);
        if (k1) {
            pipe->InitBuffer(expandedXQue, DOUBLE_BUFFER, tilingData->rowFactor * hAligned * sizeof(T));
            pipe->InitBuffer(expandedRowIdxQue, DOUBLE_BUFFER, rowFactorKAlignedTInt32 * sizeof(int32_t));
        } else if (k4 && tilingData->rowFactor >= BATCH_COPY_SIZE) {
            pipe->InitBuffer(expandedXQue, DOUBLE_BUFFER, BATCH_COPY_SIZE * k * hAligned * sizeof(T));
        } else {
            pipe->InitBuffer(expandedXQue, DOUBLE_BUFFER, k * hAligned * sizeof(T));
        }
        pipe->InitBuffer(yQue, DOUBLE_BUFFER, rowFactorHAlignedFloat * sizeof(float));
        if (hasBiasAndExpertIdx) {
            if (k1) {
                pipe->InitBuffer(biasQue, DOUBLE_BUFFER, tilingData->rowFactor * hAligned * sizeof(T));
                pipe->InitBuffer(expertIdxQue, DOUBLE_BUFFER, rowFactorKAlignedTInt32 * sizeof(int32_t));
            } else if (k4 && tilingData->rowFactor >= BATCH_COPY_SIZE) {
                pipe->InitBuffer(biasQue, DOUBLE_BUFFER, BATCH_COPY_SIZE * k * hAligned * sizeof(T));
            } else {
                pipe->InitBuffer(biasQue, DOUBLE_BUFFER, k * hAligned * sizeof(T));
            }
        }
        if (hasX1) {
            pipe->InitBuffer(x1Que, DOUBLE_BUFFER, rowFactorHAlignedT * sizeof(T));
        }
        if (hasX2) {
            pipe->InitBuffer(x2Que, DOUBLE_BUFFER, rowFactorHAlignedT * sizeof(T));
        }
        if (hasScales) {
            pipe->InitBuffer(scalesQue, DOUBLE_BUFFER, rowFactorKAlignedT * sizeof(S));
        }
        if (hasX) {
            pipe->InitBuffer(xBuf, h * sizeof(T));
        }
        if (hasConstExpert) {
            pipe->InitBuffer(constExpertAlpha1Buf, h * sizeof(T));
            pipe->InitBuffer(constExpertAlpha2Buf, h * sizeof(T));
            pipe->InitBuffer(vBuf, h * sizeof(T));
        }
    }

    __aicore__ inline void ProcessWithId(int64_t rowOuterIdx, int64_t rowInnerLoop)
    {
        // k == 1
        if (k1) {
            expandedRowIdxLocal = expandedRowIdxQue.AllocTensor<int32_t>();
            expandedRowIdxOffset = GetBlockIdx() * tilingData->rowOfFormerBlock + rowOuterIdx * tilingData->rowFactor;
            CopyIn(expandedRowIdxGm[expandedRowIdxOffset], expandedRowIdxLocal, 1, rowInnerLoop);
            expandedRowIdxQue.EnQue(expandedRowIdxLocal);
            expandedRowIdxLocal = expandedRowIdxQue.DeQue<int32_t>();

            if (hasBiasAndExpertIdx) {
                expertIdxLocal = expertIdxQue.AllocTensor<int32_t>();
                expertIdxOffset = GetBlockIdx() * tilingData->rowOfFormerBlock + rowOuterIdx * tilingData->rowFactor;
                CopyIn(expertIdxGm[expertIdxOffset], expertIdxLocal, 1, rowInnerLoop);
                expertIdxQue.EnQue(expertIdxLocal);
                expertIdxLocal = expertIdxQue.DeQue<int32_t>();
            }
        }
    }

    __aicore__ inline void ProcessInputWithX(int64_t rowOuterIdx, int64_t rowInnerLoop)
    {
        if (hasX1) {
            x1Local = x1Que.AllocTensor<T>();
            int64_t x1GmOffset =
                GetBlockIdx() * tilingData->rowOfFormerBlock * h + rowOuterIdx * tilingData->rowFactor * h;
            CopyIn(x1Gm[x1GmOffset], x1Local, 1, rowInnerLoop * h);
            x1Que.EnQue(x1Local);
            x1Local = x1Que.DeQue<T>();
        }
        if (hasScales) {
            scalesLocal = scalesQue.AllocTensor<S>();
            int64_t scaleGmOffset =
                GetBlockIdx() * tilingData->rowOfFormerBlock * k + rowOuterIdx * tilingData->rowFactor * k;
            CopyIn(scalesGm[scaleGmOffset], scalesLocal, 1, rowInnerLoop * k);
            scalesQue.EnQue(scalesLocal);
            scalesLocal = scalesQue.DeQue<S>();
        }
        if (hasX2) {
            x2Local = x2Que.AllocTensor<T>();
            int64_t x2GmOffset =
                GetBlockIdx() * tilingData->rowOfFormerBlock * h + rowOuterIdx * tilingData->rowFactor * h;
            CopyIn(x2Gm[x2GmOffset], x2Local, 1, rowInnerLoop * h);
            x2Que.EnQue(x2Local);
            x2Local = x2Que.DeQue<T>();
        }
    }

    __aicore__ inline void LoadBatchAsync(int64_t rowOuterIdx, int64_t batchStart, int64_t batchSize)
    {
        expandedXLocal = expandedXQue.AllocTensor<T>();
        if (hasBiasAndExpertIdx) {
            biasLocal = biasQue.AllocTensor<T>();
        }

        int64_t offset = 0;
        for (int64_t i = 0; i < batchSize; i++) {
            if (hasX) {
                xLocal = xBuf.Get<T>();
            }
            if (hasConstExpert) {
                constExpertAlpha1Local = constExpertAlpha1Buf.Get<T>();
                constExpertAlpha2Local = constExpertAlpha2Buf.Get<T>();
                vLocal = vBuf.Get<T>();
            }

            int64_t rowInnerIdx = batchStart + i;
            ProcessExpandedXBiasAndScale(rowOuterIdx, rowInnerIdx, offset);
            validKArr[i] = validK;
            offset += tilingData->k * hAligned;
        }

        if (hasBiasAndExpertIdx) {
            biasQue.EnQue(biasLocal);
        }

        expandedXQue.EnQue(expandedXLocal);
    }

    __aicore__ inline void ComputeBatch(int64_t rowOuterIdx, int64_t batchStart, int64_t batchSize)
    {
        int64_t offset = 0;
        expandedXLocal = expandedXQue.DeQue<T>();

        if (hasBiasAndExpertIdx) {
            biasLocal = biasQue.DeQue<T>();
        }

        for (int64_t i = 0; i < batchSize; i++) {
            int64_t rowInnerIdx = batchStart + i;

            LocalTensor<S> innerScaleLocal;
            if (hasScales) {
                innerScaleLocal = scalesLocal[rowInnerIdx * tilingData->k];
            }

            ProcessExpandXBiasScaleOptimized<T, S>(yLocal[rowInnerIdx * tilingData->h], expandedXLocal[offset],
                                                   biasLocal[offset], innerScaleLocal, validKArr[i], tilingData->h,
                                                   hasBiasAndExpertIdx, hasScales);

            offset += tilingData->k * hAligned;
        }

        if (hasBiasAndExpertIdx) {
            biasQue.FreeTensor(biasLocal);
        }
        expandedXQue.FreeTensor(expandedXLocal);
    }

    __aicore__ inline void ProcessYWithInputK4(int64_t rowOuterIdx, int64_t rowInnerLoop)
    {
        constexpr int64_t BATCH_SIZE = BATCH_COPY_SIZE;
        int64_t numBatches = (rowInnerLoop + BATCH_SIZE - 1) / BATCH_SIZE;

        if (numBatches > 0) {
            int64_t batchStart = 0;
            int64_t batchSize = min(BATCH_SIZE, rowInnerLoop - batchStart);
            LoadBatchAsync(rowOuterIdx, batchStart, batchSize);
        }

        for (int64_t batchIdx = 0; batchIdx < numBatches; batchIdx++) {
            int64_t batchStart = batchIdx * BATCH_SIZE;
            int64_t batchSize = min(BATCH_SIZE, rowInnerLoop - batchStart);

            ComputeBatch(rowOuterIdx, batchStart, batchSize);

            if (batchIdx + 1 < numBatches) {
                int64_t nextBatchStart = (batchIdx + 1) * BATCH_SIZE;
                int64_t nextBatchSize = min(BATCH_SIZE, rowInnerLoop - nextBatchStart);
                LoadBatchAsync(rowOuterIdx, nextBatchStart, nextBatchSize);
            }
        }

        if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
            Cast(yLocal.template ReinterpretCast<T>(), yLocal, RoundMode::CAST_RINT, rowInnerLoop * tilingData->h);
        }
    }

    __aicore__ inline void AllocTensorsForProcessY()
    {
        expandedXLocal = expandedXQue.AllocTensor<T>();
        if (hasBiasAndExpertIdx) {
            biasLocal = biasQue.AllocTensor<T>();
        }
        if (hasX) {
            xLocal = xBuf.Get<T>();
        }
        if (hasConstExpert) {
            constExpertAlpha1Local = constExpertAlpha1Buf.Get<T>();
            constExpertAlpha2Local = constExpertAlpha2Buf.Get<T>();
            vLocal = vBuf.Get<T>();
        }
    }

    __aicore__ inline void EnQueDeQueTensors()
    {
        expandedXQue.EnQue(expandedXLocal);
        expandedXLocal = expandedXQue.DeQue<T>();
        if (hasBiasAndExpertIdx) {
            biasQue.EnQue(biasLocal);
            biasLocal = biasQue.DeQue<T>();
        }
    }

    __aicore__ inline void FreeTensorsForProcessY()
    {
        expandedXQue.FreeTensor(expandedXLocal);
        if (hasBiasAndExpertIdx) {
            biasQue.FreeTensor(biasLocal);
        }
    }

    __aicore__ inline void ProcessYWithInput(int64_t rowOuterIdx, int64_t rowInnerLoop)
    {
        for (int64_t rowInnerIdx = 0; rowInnerIdx < rowInnerLoop; rowInnerIdx += 1) {
            AllocTensorsForProcessY();
            ProcessExpandedXBiasAndScale(rowOuterIdx, rowInnerIdx, 0);
            EnQueDeQueTensors();
            LocalTensor<S> innerScaleLocal;
            if (hasScales) {
                event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
                SetFlag<HardEvent::S_V>(eventId);
                WaitFlag<HardEvent::S_V>(eventId);
                innerScaleLocal = scalesLocal[rowInnerIdx * tilingData->k];
            }
            ProcessExpandXBiasScaleOptimized<T, S>(yLocal[rowInnerIdx * tilingData->h], expandedXLocal, biasLocal,
                                                   innerScaleLocal, validK, tilingData->h, hasBiasAndExpertIdx,
                                                   hasScales);
            FreeTensorsForProcessY();
        }
        if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
            Cast(yLocal.template ReinterpretCast<T>(), yLocal, RoundMode::CAST_RINT, rowInnerLoop * tilingData->h);
        }
    }

    __aicore__ inline bool IsValidExpandedRowIdx(int64_t expandedRowIdxValue)
    {
        if (expandedRowIdxValue == INVALID_IDX) {
            return false;
        }
        if constexpr (dropPadMode != DROP_PAD_COLUMN && dropPadMode != DROP_PAD_ROW) {
            if (expandedRowIdxValue >= tilingData->activeNum) {
                return false;
            }
        }
        return true;
    }

    __aicore__ inline int64_t GetExpertIdxValue(int64_t rowInnerIdx, int64_t rowOuterIdx, int64_t kIdx)
    {
        if (!hasExpertIdx) {
            return -1;
        }
        SetOffsetOfExpertIdx(rowInnerIdx, rowOuterIdx, kIdx);
        return expertIdxGm.GetValue(expertIdxOffset);
    }

    __aicore__ inline bool LoadExpandedXForK(int64_t rowOuterIdx, int64_t rowInnerIdx, int64_t expertIdx,
                                             int64_t expandedRowIdxGmValue, int64_t offset)
    {
        if (hasX && hasExpertIdx) {
            if (expertIdx >= tilingData->zeroExpertStart && expertIdx < tilingData->zeroExpertEnd) {
                return false;
            }
            if (expertIdx >= tilingData->copyExpertStart && expertIdx < tilingData->copyExpertEnd) {
                // x = x[i]
                int64_t xGmOffset = GetBlockIdx() * tilingData->rowOfFormerBlock * h +
                                    rowOuterIdx * tilingData->rowFactor * h + rowInnerIdx * h;
                CopyIn(xGm[xGmOffset], expandedXLocal[offset + validK * tilingData->hAligned], 1, tilingData->h);
            } else if (hasConstExpert && expertIdx >= tilingData->constantExpertStart &&
                       expertIdx < tilingData->constantExpertEnd) {
                // x = a1 * x[i] +  a2 * v
                int64_t xGmOffset = GetBlockIdx() * tilingData->rowOfFormerBlock * h +
                                    rowOuterIdx * tilingData->rowFactor * h + rowInnerIdx * h;
                CopyIn(xGm[xGmOffset], xLocal, 1, tilingData->h);
                // 不需要有偏移，用完就下一个循环覆盖掉就行
                int64_t constExpertGmOffset = (expertIdx - tilingData->constantExpertStart) * h;
                CopyIn(constExpertAlpha1Gm[constExpertGmOffset], constExpertAlpha1Local, 1, tilingData->h);
                CopyIn(constExpertAlpha2Gm[constExpertGmOffset], constExpertAlpha2Local, 1, tilingData->h);
                CopyIn(vGm[constExpertGmOffset], vLocal, 1, tilingData->h);
                event_t mte2v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                SetFlag<HardEvent::MTE2_V>(mte2v);
                WaitFlag<HardEvent::MTE2_V>(mte2v);
                vLocal = vLocal * constExpertAlpha2Local;
                xLocal = xLocal * constExpertAlpha1Local;
                xLocal = xLocal + vLocal;
                AscendC::Copy(expandedXLocal[offset + validK * tilingData->hAligned], xLocal, tilingData->h);
                event_t v2mte = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
                SetFlag<HardEvent::V_MTE2>(v2mte);
                WaitFlag<HardEvent::V_MTE2>(v2mte);
            } else {
                CopyIn(expandedXGm[expandedRowIdxGmValue * tilingData->h],
                       expandedXLocal[offset + validK * tilingData->hAligned], 1, tilingData->h);
            }
        } else {
            CopyIn(expandedXGm[expandedRowIdxGmValue * tilingData->h],
                   expandedXLocal[offset + validK * tilingData->hAligned], 1, tilingData->h);
        }
        return true;
    }

    __aicore__ inline void ProcessExpandedXBiasAndScale(int64_t rowOuterIdx, int64_t rowInnerIdx, int64_t offset)
    {
        validK = 0;
        for (int64_t kIdx = 0; kIdx < tilingData->k; kIdx += 1) {
            SetExpandedRowIdxOffset(rowOuterIdx, rowInnerIdx, kIdx);
            int64_t expandedRowIdxGmValue = expandedRowIdxGm.GetValue(expandedRowIdxOffset);
            if (!IsValidExpandedRowIdx(expandedRowIdxGmValue)) {
                continue;
            }
            int64_t expertIdx = GetExpertIdxValue(rowInnerIdx, rowOuterIdx, kIdx);
            if (!LoadExpandedXForK(rowOuterIdx, rowInnerIdx, expertIdx, expandedRowIdxGmValue, offset)) {
                continue;
            }

            if (hasBiasAndExpertIdx) {
                int64_t biasGmOffset = expertIdx * tilingData->h;
                CopyIn(biasGm[biasGmOffset], biasLocal[offset + validK * tilingData->hAligned], 1, tilingData->h);
            }
            // 由于会将k轴融到VF中，所以drop_pad场景需要将有效的Scale按序设置
            if (hasScales) {
                S scaleValue = scalesLocal[rowInnerIdx * tilingData->k].GetValue(kIdx);
                scalesLocal[rowInnerIdx * tilingData->k].SetValue(validK, scaleValue);
            }
            validK += 1;
        }
    }

    __aicore__ inline void ProcessYWithInputK1(int64_t rowOuterIdx, int64_t rowInnerLoop)
    {
        expandedXLocal = expandedXQue.AllocTensor<T>();
        if (hasBiasAndExpertIdx) {
            biasLocal = biasQue.AllocTensor<T>();
        }
        if (hasX) {
            xLocal = xBuf.Get<T>();
        }
        if (hasConstExpert) {
            constExpertAlpha1Local = constExpertAlpha1Buf.Get<T>();
            constExpertAlpha2Local = constExpertAlpha2Buf.Get<T>();
            vLocal = vBuf.Get<T>();
        }
        event_t sWaitMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(sWaitMte2);
        WaitFlag<HardEvent::MTE2_S>(sWaitMte2);
        int64_t hOffset = 0;
        int64_t kOffset = 0;
        int64_t khAlignedOffset = 0;
        for (int64_t rowInnerIdx = 0; rowInnerIdx < rowInnerLoop; rowInnerIdx += 1) {
            ProcessExpandedXBiasAndScaleK1(rowOuterIdx, rowInnerIdx, kOffset, khAlignedOffset);
            expandedXQue.EnQue(expandedXLocal);
            expandedXLocal = expandedXQue.DeQue<T>();
            if (hasBiasAndExpertIdx) {
                biasQue.EnQue(biasLocal);
                biasLocal = biasQue.DeQue<T>();
            }
            ProcessExpandXBiasScaleOptimized<T, S>(yLocal[hOffset], expandedXLocal[khAlignedOffset],
                                                   biasLocal[khAlignedOffset], scalesLocal[kOffset], 1, h,
                                                   hasBiasAndExpertIdx, hasScales);
            hOffset += h;
            kOffset += 1;
            khAlignedOffset += khAligned;
        }

        expandedXQue.FreeTensor(expandedXLocal);
        if (hasBiasAndExpertIdx) {
            biasQue.FreeTensor(biasLocal);
        }
        if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
            Cast(yLocal.template ReinterpretCast<T>(), yLocal, RoundMode::CAST_RINT, rowInnerLoop * h);
        }
    }

    __aicore__ inline void ProcessExpandedXBiasAndScaleK1(int64_t rowOuterIdx, int64_t rowInnerIdx, int64_t kOffset,
                                                          int64_t khAlignedOffset)
    {
        int64_t expandedRowIdxValue = expandedRowIdxLocal.GetValue(rowInnerIdx);
        if (expandedRowIdxValue == INVALID_IDX) {
            return;
        }
        if constexpr (dropPadMode != DROP_PAD_COLUMN && dropPadMode != DROP_PAD_ROW) {
            if (expandedRowIdxValue >= tilingData->activeNum) {
                return;
            }
        }
        int64_t expertIdx = -1;
        if (hasExpertIdx) {
            expertIdx = expertIdxLocal.GetValue(rowInnerIdx);
        }
        if (hasX && hasExpertIdx) {
            if (expertIdx >= tilingData->zeroExpertStart && expertIdx < tilingData->zeroExpertEnd) {
                // 直接不计算
                return;
            }
            if (expertIdx >= tilingData->copyExpertStart && expertIdx < tilingData->copyExpertEnd) {
                // x = x[i]
                int64_t xGmOffset = GetBlockIdx() * tilingData->rowOfFormerBlock * tilingData->h +
                                    rowOuterIdx * tilingData->rowFactor * tilingData->h + rowInnerIdx * tilingData->h;
                CopyIn(xGm[xGmOffset], expandedXLocal[khAlignedOffset], 1, tilingData->h);
            } else if (expertIdx >= tilingData->constantExpertStart && expertIdx < tilingData->constantExpertEnd) {
                // x = a1 * x[i] +  a2 * v
                // 不需要有偏移，用完就下一个循环覆盖掉就行
                int64_t xGmOffset = GetBlockIdx() * tilingData->rowOfFormerBlock * tilingData->h +
                                    rowOuterIdx * tilingData->rowFactor * tilingData->h + rowInnerIdx * tilingData->h;
                CopyIn(xGm[xGmOffset], xLocal, 1, tilingData->h);
                int64_t constExpertGmOffset = (expertIdx - tilingData->constantExpertStart) * tilingData->h;
                CopyIn(constExpertAlpha1Gm[constExpertGmOffset], constExpertAlpha1Local, 1, tilingData->h);
                CopyIn(constExpertAlpha2Gm[constExpertGmOffset], constExpertAlpha2Local, 1, tilingData->h);
                CopyIn(vGm[constExpertGmOffset], vLocal, 1, tilingData->h);
                event_t mte2v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                SetFlag<HardEvent::MTE2_V>(mte2v);
                WaitFlag<HardEvent::MTE2_V>(mte2v);
                vLocal = vLocal * constExpertAlpha2Local;
                xLocal = xLocal * constExpertAlpha1Local;
                xLocal = xLocal + vLocal;
                AscendC::Copy(expandedXLocal[khAlignedOffset], xLocal, tilingData->h);
            } else {
                CopyIn(expandedXGm[expandedRowIdxValue * h], expandedXLocal[khAlignedOffset], 1, h);
            }
        } else {
            CopyIn(expandedXGm[expandedRowIdxValue * h], expandedXLocal[khAlignedOffset], 1, h);
        }
        if (hasBiasAndExpertIdx) {
            int64_t biasGmOffset = expertIdx * h;
            CopyIn(biasGm[biasGmOffset], biasLocal[khAlignedOffset], 1, h);
        }

        if (hasScales) {
            S scaleValue = scalesLocal[kOffset].GetValue(0);
            scalesLocal[kOffset].SetValue(0, scaleValue);
        }
    }

    __aicore__ inline void SetExpandedRowIdxOffset(int64_t rowOuterIdx, int64_t rowInnerIdx, int64_t kIdx)
    {
        if constexpr (dropPadMode == DROPLESS_COLUMN || dropPadMode == DROP_PAD_COLUMN) {
            expandedRowIdxOffset = GetBlockIdx() * tilingData->rowOfFormerBlock + rowOuterIdx * tilingData->rowFactor +
                                   rowInnerIdx + kIdx * tilingData->row;
        } else {
            expandedRowIdxOffset = GetBlockIdx() * tilingData->rowOfFormerBlock * tilingData->k +
                                   rowOuterIdx * tilingData->rowFactor * tilingData->k + rowInnerIdx * tilingData->k +
                                   kIdx;
        }
    }

    __aicore__ inline void SetOffsetOfExpertIdx(int64_t rowInnerIdx, int64_t rowOuterIdx, int64_t kIdx)
    {
        if constexpr (dropPadMode == DROP_PAD_COLUMN || dropPadMode == DROPLESS_COLUMN) {
            expertIdxOffset = rowOuterIdx * tilingData->rowFactor * k +
                              GetBlockIdx() * tilingData->rowOfFormerBlock * k + rowInnerIdx * k + kIdx;
        } else {
            expertIdxOffset = expandedRowIdxOffset;
        }
    }

    TPipe *pipe;
    const MoeFinalizeRoutingV2RegbaseTilingData *tilingData;

    LocalTensor<T> expandedXLocal;
    LocalTensor<int32_t> expandedRowIdxLocal;
    LocalTensor<T> x1Local;
    LocalTensor<T> x2Local;
    LocalTensor<T> biasLocal;
    LocalTensor<int32_t> expertIdxLocal;

    LocalTensor<T> xLocal;
    LocalTensor<T> constExpertAlpha1Local;
    LocalTensor<T> constExpertAlpha2Local;
    LocalTensor<T> vLocal;

    LocalTensor<S> scalesLocal;
    LocalTensor<float> yLocal;

    bool hasX1{false};
    bool hasX2{false};
    bool hasBiasAndExpertIdx{false};
    bool hasExpertIdx{false};
    bool hasScales{false};
    bool k1{false};
    bool k4{false};
    bool hasX{false};
    bool hasConstExpert{false};

    int16_t validK{0};
    int16_t validKArr[BATCH_COPY_SIZE]{0}; // k <= 4使用
    int64_t expandedRowIdxOffset{0};
    int64_t expertIdxOffset{0};
    int64_t k{0};
    int64_t h{0};
    int64_t hAligned{0};
    int64_t khAligned{0};

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

    TQue<QuePosition::VECIN, DOUBLE_BUFFER> expandedXQue;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> expandedRowIdxQue;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> biasQue;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> expertIdxQue;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> x1Que;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> x2Que;
    TBuf<QuePosition::VECCALC> xBuf;
    TBuf<QuePosition::VECCALC> constExpertAlpha1Buf;
    TBuf<QuePosition::VECCALC> constExpertAlpha2Buf;
    TBuf<QuePosition::VECCALC> vBuf;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> scalesQue;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> yQue;
};
} // namespace MoeFinalizeRoutingV2Regbase

#endif
