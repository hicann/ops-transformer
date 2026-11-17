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
 * \file addr_compute_det.h
 * \brief
 */

#ifndef __ADDR_COMPUTE_DET_H__
#define __ADDR_COMPUTE_DET_H__

#include "flash_attention_score_grad_common_header.h"
using namespace AscendC;

namespace FAG_DET {
template <typename FAGT>
class AddrComputeDet {
    using TILING_CLASS = typename FAGT::tiling_class;
    using SEQLEN_TYPE = typename FAGT::seqlen_type;
    using INPUT_TYPE = typename FAGT::input_type;
    static constexpr bool DROP_ENABLE = FAGT::drop_enable;
    static constexpr bool DETERMINISTIC_ENABLE = FAGT::deterministic_enable;

public:
    __aicore__ inline void GetCubeAddrInfo(CubeAddrInfoDet* cubeAddrInfo)
    {
        globalCubeAddr = cubeAddrInfo;
        globalCubeAddr->blockLength = 0;
        globalCubeAddr->atmoicAdd = 0;
        if (useDenseSchedule) {
            const int32_t waveStart = curCubeIdx;
            const int32_t waveEnd = min(waveStart + int32_t(cubeCoreNum), totalTiles);
            const int32_t t = waveStart + cubeCoreIdx;
            if (t < totalTiles) {
                const int32_t head = t / headTiles;
                bIdx = head / dimN1;
                n1Idx = head % dimN1;
                const int32_t major = (t / seqTiles % seqTiles) * 512;
                const int32_t minor = (t % seqTiles) * 512;
                s1Idx = useQFirst ? minor : major;
                s2Idx = useQFirst ? major : minor;
                curCubeIdx = t;
                CubeRecord(min(512, seqLength - s1Idx), min(512, seqLength - s2Idx));
                // Only a repeated KV key in this wave needs vector accumulation.
                const bool hasPrev = useQFirst ? t > waveStart && t % seqTiles != 0 :
                                                 t - seqTiles >= waveStart && (t - seqTiles) / headTiles == head;
                const bool hasNext = useQFirst ? t + 1 < waveEnd && (t + 1) % seqTiles != 0 :
                                                 t + seqTiles < waveEnd && (t + seqTiles) / headTiles == head;
                globalCubeAddr->atmoicAdd = !hasPrev && !hasNext;
            }
            curCubeIdx = waveEnd;
            return;
        }
        s2GroupNum = 0;
        for (int32_t i = 0; i < cubeCoreNum; i++) {
            globalCubeAddr->detS2GroupList[i].accumNum = 0;
        }
        ComputeAddrInfo<false>();
        for (int32_t i = 0; i < s2GroupNum; i++) {
            if (globalCubeAddr->detS2GroupList[i].accumNum == 1 &&
                globalCubeAddr->detS2GroupList[i].accumList[0] == cubeCoreIdx) {
                // k/v 如果累加数量只有1，则直接通过atmoicAdd指令累加，不再经过vec
                globalCubeAddr->atmoicAdd = 1;
            }
        }
    }

    __aicore__ inline void GetVecAddrInfo(VecAddrInfoDet* vecAddrInfo)
    {
        globalVecAddr = vecAddrInfo;
        if (globalVecAddr->taskId == 0) {
            globalVecAddr->attenMaskDimS2 = sparseMode == 1 ? maxSeqK : ATTEN_MASK_DIM_S2;
            globalVecAddr->enableCausalOpt = enableCausalOpt;
        }
        globalVecAddr->blockLength = 0;
        s1GroupNum = 0;
        s2GroupNum = 0;
        if (useDenseSchedule) {
            if (useQFirst) {
                GetDenseQFirstVecAddrInfo();
                return;
            }
            const int32_t waveStart = curCubeIdx;
            const int32_t waveEnd = min(waveStart + int32_t(cubeCoreNum), totalTiles);
            // Keep Q rows in first-appearance order, including partial final rows.
            for (int32_t qrow = waveStart / seqTiles; qrow * seqTiles < waveEnd; qrow++) {
                auto& group = globalVecAddr->detS1GroupList[s1GroupNum++];
                const int32_t first = max(waveStart, qrow * seqTiles);
                const int32_t last = min(waveEnd, (qrow + 1) * seqTiles);
                group.outAddr = (int64_t(qrow / seqTiles) * seqLength + (qrow % seqTiles) * 512) * dimD;
                group.recoderS = min(512, seqLength - (qrow % seqTiles) * 512);
                group.accumNum = 0;
                group.existFirstBlock = first == qrow * seqTiles;
                group.existLastBlock = last == (qrow + 1) * seqTiles;
                for (int32_t t = first; t < last; t++) {
                    group.accumList[group.accumNum++] = t % cubeCoreNum;
                }
            }
            for (int32_t j = 0; j < waveEnd - waveStart; j++) {
                const int32_t t = waveStart + j;
                const int32_t h = t / headTiles;
                const int32_t k = t % seqTiles;
                const int64_t outAddr = (int64_t(h) * seqLength + k * 512) * dimD;
                int32_t gidx = s2GroupNum;
                if (2 * seqTiles >= static_cast<int32_t>(cubeCoreNum)) {
                    // At most one repeat per key. The first seqTiles slots are unique.
                    if (j >= seqTiles && (t - seqTiles) / headTiles == h) {
                        gidx = j - seqTiles;
                    }
                } else {
                    gidx = 0;
                    while (gidx < s2GroupNum && globalVecAddr->detS2GroupList[gidx].outAddr != outAddr) {
                        gidx++;
                    }
                }
                if (gidx == s2GroupNum) {
                    s2GroupNum++;
                    auto& group = globalVecAddr->detS2GroupList[gidx];
                    group.outAddr = outAddr;
                    group.recoderS = min(512, seqLength - k * 512);
                    group.accumNum = 0;
                    group.existFirstBlock = 0;
                    group.existLastBlock = 0;
                }
                auto& group = globalVecAddr->detS2GroupList[gidx];
                group.accumList[group.accumNum++] = j;
            }
            const int32_t t = waveStart + cubeCoreIdx;
            if (t < totalTiles) {
                const int32_t head = t / headTiles;
                bIdx = head / dimN1;
                n1Idx = head % dimN1;
                s1Idx = (t / seqTiles % seqTiles) * 512;
                s2Idx = (t % seqTiles) * 512;
                curCubeIdx = t;
                VecRecord(min(512, seqLength - s1Idx), min(512, seqLength - s2Idx));
            }
            curCubeIdx = waveEnd;
            globalVecAddr->curCubeIdx = waveEnd % cubeCoreNum;
            globalVecAddr->s1GroupNum = s1GroupNum;
            globalVecAddr->s2GroupNum = s2GroupNum;
            return;
        }
        for (int32_t i = 0; i < cubeCoreNum; i++) {
            globalVecAddr->detS1GroupList[i].accumNum = 0;
            globalVecAddr->detS1GroupList[i].existFirstBlock = 0;
            globalVecAddr->detS1GroupList[i].existLastBlock = 0;
            globalVecAddr->detS2GroupList[i].accumNum = 0;
        }
        ComputeAddrInfo<true>();
        globalVecAddr->curCubeIdx = curCubeIdx % cubeCoreNum;
        globalVecAddr->s1GroupNum = s1GroupNum;
        globalVecAddr->s2GroupNum = s2GroupNum;
    }

    __aicore__ void Init(__gm__ uint8_t* seqLenQ, __gm__ uint8_t* seqLenK, uint32_t cubeCoreNum, uint32_t cubeCoreIdx,
                         const TILING_CLASS* tilingData)
    {
        this->seqLenQ = seqLenQ;
        this->seqLenK = seqLenK;
        this->cubeCoreIdx = cubeCoreIdx;
        this->cubeCoreNum = cubeCoreNum;
        this->dimB = tilingData->basicDetTensorTilingData.b;
        this->dimN2 = tilingData->basicDetTensorTilingData.n2;
        this->dimG = tilingData->basicDetTensorTilingData.g;
        this->dimN1 = dimN2 * dimG;
        this->dimD = tilingData->basicDetTensorTilingData.d;
        this->sparseMode = tilingData->basicDetTensorTilingData.sparseMode;
        useDenseSchedule = dimD > 128 && sparseMode == 0;
        this->preToken = tilingData->basicDetTensorTilingData.preTockens;
        this->nextToken = tilingData->basicDetTensorTilingData.nextTockens;
        this->dqPostAbsorb = tilingData->basicDetTensorTilingData.dqPostAbsorb;
        this->layout = tilingData->basicDetTensorTilingData.layout;
        if (dimD > 128) {
            dqPostAbsorb = 1;
        }
        if (useDenseSchedule) {
            seqLength = tilingData->basicDetTensorTilingData.s1;
            seqTiles = CeilDiv(seqLength, 512);
            // Near-full tails keep the same Cube sub-block count as full tiles.
            constexpr int32_t tileLength = 512;
            const int32_t seqTail = seqLength % tileLength;
            const int32_t waveSize = static_cast<int32_t>(cubeCoreNum);
            useQFirst = seqTail > tileLength - SIZE_128 && (seqTiles % waveSize == 0 || 2 * seqTiles == waveSize);
            headTiles = seqTiles * seqTiles;
            totalTiles = dimB * dimN1 * headTiles;
        }
        if (layout != TND) {
            dimS1 = tilingData->basicDetTensorTilingData.s1;
            dimS2 = tilingData->basicDetTensorTilingData.s2;
        }

        UpdateSeqLen();

        if (sparseMode == 1) {
            for (int32_t i = 0; i < dimB; i++) {
                maxSeqK = max(maxSeqK, getSeqLen(i, seqLenK, false));
            }
        } else if (sparseMode == 3) {
            for (int32_t i = 0; i < dimB; i++) {
                if (getSeqLen(i, seqLenQ, true) != getSeqLen(i, seqLenK, false)) {
                    // sparsemode=3，qk不等长情况不使能优化
                    return;
                }
            }
            enableCausalOpt = 1;
        }
    }

private:
    __aicore__ __attribute__((noinline)) void GetDenseQFirstVecAddrInfo()
    {
        const int32_t waveStart = curCubeIdx;
        const int32_t waveEnd = min(waveStart + int32_t(cubeCoreNum), totalTiles);
        for (int32_t kv = waveStart / seqTiles; kv * seqTiles < waveEnd; kv++) {
            auto& group = globalVecAddr->detS2GroupList[s2GroupNum++];
            const int32_t first = max(waveStart, kv * seqTiles);
            const int32_t last = min(waveEnd, (kv + 1) * seqTiles);
            group.outAddr = (int64_t(kv / seqTiles) * seqLength + (kv % seqTiles) * 512) * dimD;
            group.recoderS = min(512, seqLength - (kv % seqTiles) * 512);
            group.accumNum = 0;
            group.existFirstBlock = 0;
            group.existLastBlock = 0;
            for (int32_t t = first; t < last; t++) {
                group.accumList[group.accumNum++] = t - waveStart;
            }
        }

        int32_t nextHeadStart = (waveStart / headTiles + 1) * headTiles;
        int32_t headGroupStart = 0;
        int32_t nextQGroup = 0;
        for (int32_t t = waveStart; t < waveEnd; t++) {
            if (t == nextHeadStart) {
                nextHeadStart += headTiles;
                headGroupStart = s1GroupNum;
                nextQGroup = headGroupStart;
            }
            const int32_t gidx = nextQGroup++;
            if (nextQGroup == headGroupStart + seqTiles) {
                nextQGroup = headGroupStart;
            }
            auto& group = globalVecAddr->detS1GroupList[gidx];
            if (gidx == s1GroupNum) {
                s1GroupNum++;
                const int32_t q = t % seqTiles;
                group.outAddr = (int64_t(t / headTiles) * seqLength + q * 512) * dimD;
                group.recoderS = min(512, seqLength - q * 512);
                group.accumNum = 0;
                group.existFirstBlock = 0;
                group.existLastBlock = 0;
            }
            const int32_t k = t / seqTiles % seqTiles;
            group.existFirstBlock |= k == 0;
            group.existLastBlock |= k == seqTiles - 1;
            group.accumList[group.accumNum++] = t - waveStart;
        }
        const int32_t t = waveStart + cubeCoreIdx;
        if (t < totalTiles) {
            const int32_t head = t / headTiles;
            bIdx = head / dimN1;
            n1Idx = head % dimN1;
            s1Idx = (t % seqTiles) * 512;
            s2Idx = (t / seqTiles % seqTiles) * 512;
            curCubeIdx = t;
            VecRecord(min(512, seqLength - s1Idx), min(512, seqLength - s2Idx));
        }
        curCubeIdx = waveEnd;
        globalVecAddr->curCubeIdx = waveEnd % cubeCoreNum;
        globalVecAddr->s1GroupNum = s1GroupNum;
        globalVecAddr->s2GroupNum = s2GroupNum;
    }

    __gm__ uint8_t* seqLenQ;
    __gm__ uint8_t* seqLenK;
    CubeAddrInfoDet* globalCubeAddr; // 用于存储Cube计算相关的地址信息
    VecAddrInfoDet* globalVecAddr;   // 用于存储Vector计算相关的地址信息
    uint32_t cubeCoreIdx{0};         // 当前核所对应的Cube核的下标
    uint32_t cubeCoreNum{0};         // Cube核的数量
    uint32_t layout{0};              // 新增：数据布局格式

    SEQLEN_TYPE maxSeqK{0};
    int32_t dimB{0};              // batch
    SEQLEN_TYPE dimS1{0};         // 当前处理的batch的s1
    SEQLEN_TYPE dimS2{0};         // 当前处理的batch的s2
    int32_t dimN1{0};             // query的HeadNum
    int32_t dimN2{0};             // key/vaule的HeadNum
    int32_t dimG{0};              // group，dimN1=dimN2*dimG
    int32_t dimD{0};              // headDim
    int32_t bIdx{0};              // 当前batch计算到的位置
    int32_t s1Idx{0};             // 当前s1方向计算到的位置
    int32_t s2Idx{0};             // 当前s2方向计算到的位置
    int32_t n1Idx{0};             // 当前n1方向计算到的位置
    SEQLEN_TYPE lastBatchQSum{0}; // 已处理的s1的总和
    SEQLEN_TYPE lastBatchKSum{0}; // 已处理的s2的总和
    int32_t curCubeIdx{0}; // 用于分配核时的中间变量，若curCubeIdx==cubeCoreIdx,则表示该核需要计算数据
    int32_t loopFinish{0}; // 用于控制Loop结束的中间变量，当所有Cube核在当前Loop都分配到任务时，一次Loop结束
    int32_t enableCausalOpt{0};
    int32_t dqPostAbsorb{0};
    bool useDenseSchedule{false};
    bool useQFirst{false};
    int32_t seqTiles{0}, headTiles{0}, totalTiles{0}, seqLength{0};
    // 稀疏计算参数
    uint32_t sparseMode{0}; // 稀疏模式
    int64_t preToken{0};
    int64_t nextToken{0};
    int64_t sparseLeftBound{0};  // preToken直线计算公式: s1Idx = s2Idx + sparseLeftBound
    int64_t sparseRightBound{0}; // nextToken直线计算公式: s1Idx = s2Idx + sparseRightBound
    // 确定性计算参数
    int32_t s1GroupNum{0}; // 当前s1方向计算到需要累加的分组个数
    int32_t s2GroupNum{0}; // 当前s2方向计算到需要累加的分组个数

    __aicore__ inline void UpdateSeqLen()
    {
        if (layout == TND) {
            dimS1 = getSeqLen(bIdx, seqLenQ, true);
            dimS2 = getSeqLen(bIdx, seqLenK, false);
            while ((dimS1 == 0 || dimS2 == 0) && bIdx < dimB - 1) {
                bIdx++;
                dimS1 = getSeqLen(bIdx, seqLenQ, true);
                dimS2 = getSeqLen(bIdx, seqLenK, false);
            }
            if (bIdx > 0) {
                lastBatchQSum = getTotalLen(bIdx - 1, seqLenQ);
                lastBatchKSum = getTotalLen(bIdx - 1, seqLenK);
            }
        } else {
            if (bIdx > 0) {
                lastBatchQSum = bIdx * dimS1;
                lastBatchKSum = bIdx * dimS2;
            }
        }

        sparseLeftBound = dimS1 - dimS2 + preToken + 1;
        sparseRightBound = dimS1 - dimS2 - nextToken;
    }

    __aicore__ inline SEQLEN_TYPE getSeqLen(int32_t i, __gm__ uint8_t* seq_Len, bool isQ = true)
    {
        if (layout != TND) {
            return isQ ? dimS1 : dimS2;
        }

        SEQLEN_TYPE actualSeqlen;
        if (i == 0) {
            actualSeqlen = ((__gm__ SEQLEN_TYPE*)seq_Len)[0];
        } else {
            actualSeqlen = ((__gm__ SEQLEN_TYPE*)seq_Len)[i] - ((__gm__ SEQLEN_TYPE*)seq_Len)[i - 1];
        }
        return actualSeqlen;
    }

    __aicore__ inline SEQLEN_TYPE getTotalLen(int32_t i, __gm__ uint8_t* seq_Len)
    {
        SEQLEN_TYPE actualTotalSeqlen = ((__gm__ SEQLEN_TYPE*)seq_Len)[i];
        return actualTotalSeqlen;
    }

    __aicore__ inline uint64_t getLeftAddr(int32_t lastBatchSum, int32_t s1Idx, int32_t n1Idx)
    {
        if (layout == BSNGD) {
            return bIdx * dimS1 * dimN1 * dimD + (s1Idx * dimN1 * dimD) + (n1Idx * dimD);
        } else if (layout == SBNGD) {
            return s1Idx * dimB * dimN1 * dimD + (bIdx * dimN1 * dimD) + (n1Idx * dimD);
        } else if (layout == BNGSD) {
            return bIdx * dimN1 * dimS1 * dimD + (n1Idx * dimS1 * dimD) + (s1Idx * dimD);
        } else {
            return lastBatchSum * dimN1 * dimD + (s1Idx * dimN1 * dimD) + (n1Idx * dimD);
        }
    }

    __aicore__ inline uint64_t getRightAddr(int32_t lastBatchSum, int32_t s2Idx, int32_t n1Idx)
    {
        if (layout == BSNGD) { // BSH格式
            return bIdx * dimS2 * dimN2 * dimD + (s2Idx * dimN2 * dimD) + ((n1Idx / dimG) * dimD);
        } else if (layout == SBNGD) {
            return s2Idx * dimB * dimN2 * dimD + (bIdx * dimN2 * dimD) + ((n1Idx / dimG) * dimD);
        } else if (layout == BNGSD) {
            return bIdx * dimN2 * dimS2 * dimD + ((n1Idx / dimG) * dimS2 * dimD) + (s2Idx * dimD);
        } else {
            return lastBatchSum * dimN2 * dimD + (s2Idx * dimN2 * dimD) + ((n1Idx / dimG) * dimD);
        }
    }

    __aicore__ inline int32_t GetRecoderS(int32_t sIdx, int32_t sLen)
    {
        return sIdx + 512 < sLen ? 512 : (sLen - sIdx);
    }

    __aicore__ inline void UpdateLoopIdx()
    {
        curCubeIdx++;
        if (curCubeIdx % cubeCoreNum == 0) {
            loopFinish = 1;
        }
    }

    template <bool IS_DQ>
    __aicore__ inline void DetInfoProcess(DetGroup (&detGroupList)[24], int32_t recoderS1, int32_t recoderS2,
                                          int32_t& groupNum, uint64_t outAddr)
    {
        uint32_t groupIdx;
        int32_t find{0};
        int32_t recoderS;
        if constexpr (IS_DQ) {
            recoderS = recoderS1;
        } else {
            recoderS = recoderS2;
        }

        for (uint32_t i = 0; i < groupNum; i++) {
            if (detGroupList[i].outAddr == outAddr) {
                find = 1;
                groupIdx = i;
                break;
            }
        }

        if (!find) {
            groupNum++;
            groupIdx = groupNum - 1;
            detGroupList[groupIdx].outAddr = outAddr;
            detGroupList[groupIdx].recoderS = recoderS;
        }

        uint32_t accIdx = detGroupList[groupIdx].accumNum;
        detGroupList[groupIdx].accumList[accIdx] = curCubeIdx % cubeCoreNum;
        detGroupList[groupIdx].accumNum++;

        if constexpr (IS_DQ) {
            if (!dqPostAbsorb) {
                return;
            }
            if (s2Idx == 0) {
                detGroupList[groupIdx].existFirstBlock = 1;
            }
            if ((s2Idx + recoderS2 == dimS2)) {
                detGroupList[groupIdx].existLastBlock = 1;
            } else if (sparseMode == 3 &&
                       isRightDownOverTouchRight(s1Idx + recoderS1, s2Idx + recoderS2, sparseRightBound)) {
                detGroupList[groupIdx].existLastBlock = 1;
            }
        }
    }

    template <bool IS_AIV>
    __aicore__ inline void DetRecord(int32_t recoderS1, int32_t recoderS2)
    {
        // 保存所有核的确定性信息
        if constexpr (IS_AIV) {
            DetInfoProcess<true>(globalVecAddr->detS1GroupList, recoderS1, recoderS2, s1GroupNum,
                                 getLeftAddr(lastBatchQSum, s1Idx, n1Idx));
            DetInfoProcess<false>(globalVecAddr->detS2GroupList, recoderS1, recoderS2, s2GroupNum,
                                  getRightAddr(lastBatchKSum, s2Idx, n1Idx));
        } else {
            DetInfoProcess<false>(globalCubeAddr->detS2GroupList, recoderS1, recoderS2, s2GroupNum,
                                  getRightAddr(lastBatchKSum, s2Idx, n1Idx));
        }
    }

    __aicore__ inline void CubeRecord(int32_t recoderS1, int32_t recoderS2)
    {
        if (curCubeIdx % cubeCoreNum != cubeCoreIdx) {
            // 非确定性信息只保存当前核
            return;
        }

        globalCubeAddr->s1GmAddr = getLeftAddr(lastBatchQSum, s1Idx, n1Idx);
        globalCubeAddr->s2GmAddr = getRightAddr(lastBatchKSum, s2Idx, n1Idx);
        globalCubeAddr->s1Len = recoderS1;
        globalCubeAddr->s2Len = recoderS2;
        globalCubeAddr->s1Idx = s1Idx;
        globalCubeAddr->s2Idx = s2Idx;
        globalCubeAddr->sparseLeftBound = sparseLeftBound;
        globalCubeAddr->sparseRightBound = sparseRightBound;
        globalCubeAddr->enableCausalOpt = enableCausalOpt;
        globalCubeAddr->blockLength++;
    }

    __aicore__ inline void VecRecord(int32_t recoderS1, int32_t recoderS2)
    {
        if (curCubeIdx % cubeCoreNum != cubeCoreIdx) {
            // 非确定性信息只保存当前核
            return;
        }

        int32_t blockId = 0;
        int32_t s1Tail = (recoderS1 % 128 == 0) ? 128 : recoderS1 % 128;
        int32_t s2Tail = (recoderS2 % 128 == 0) ? 128 : recoderS2 % 128;
        int32_t s1Loop = CeilDiv(recoderS1, 128);
        int32_t s2Loop = CeilDiv(recoderS2, 128);
        globalVecAddr->batchIdx = bIdx;

        for (int32_t x = 0; x < s2Loop; x++) {
            for (int32_t y = 0; y < s1Loop; y++) {
                int32_t curS1Idx = s1Idx + y * 128;
                int32_t curS2Idx = s2Idx + x * 128;
                int32_t curS1Len = (y == s1Loop - 1) ? s1Tail : 128;
                int32_t curS2Len = (x == s2Loop - 1) ? s2Tail : 128;
                auto vecPhyAddr = &globalVecAddr->VecBlkInfo[blockId];
                if (IsFullMaskBlock(curS1Idx, curS2Idx, curS1Len, curS2Len, sparseLeftBound, sparseRightBound,
                                    sparseMode)) {
                    if (enableCausalOpt) {
                        continue;
                    }
                    vecPhyAddr->isFullMask = true;
                } else {
                    vecPhyAddr->isFullMask = false;
                }

                vecPhyAddr->n2Idx = n1Idx / dimG;
                vecPhyAddr->gIdx = n1Idx % dimG;
                vecPhyAddr->s1Idx = curS1Idx;
                vecPhyAddr->s2Idx = curS2Idx;
                vecPhyAddr->s1Len = curS1Len;
                vecPhyAddr->s2Len = curS2Len;
                vecPhyAddr->cubeBlockOffset = blockId * BASE_BLOCK_SIZE;
                blockId++;

                // cal Attenmask Info
                if (sparseMode == 0 || vecPhyAddr->isFullMask) {
                    continue;
                } else if (sparseMode == 1) {
                    vecPhyAddr->calNextToken = 1;
                    vecPhyAddr->nextTokenOffset = curS1Idx * maxSeqK + curS2Idx;
                } else {
                    if (isRightUpOverRight(curS1Idx, curS2Idx + curS2Len, sparseRightBound)) {
                        int32_t sparseS2Idx = (curS1Idx - sparseRightBound - curS2Idx) % 128;
                        vecPhyAddr->calNextToken = 1;
                        vecPhyAddr->nextTokenOffset = sparseS2Idx >= 0 ?
                                                          ATTEN_MASK_DEFAULT_OFFSET - sparseS2Idx :
                                                          ATTEN_MASK_DEFAULT_OFFSET + sparseS2Idx * ATTEN_MASK_DIM_S2;
                    } else {
                        vecPhyAddr->calNextToken = 0;
                    }

                    if (sparseMode == 4 && isLeftDownBeyondLeft(curS1Idx + curS1Len, curS2Idx, sparseLeftBound)) {
                        int32_t sparseS2Idx = (curS1Idx - sparseLeftBound - curS2Idx) % 128;
                        vecPhyAddr->calPreToken = 1;
                        vecPhyAddr->preTokenOffset = sparseS2Idx >= 0 ?
                                                         ATTEN_MASK_DEFAULT_OFFSET - sparseS2Idx :
                                                         ATTEN_MASK_DEFAULT_OFFSET + sparseS2Idx * ATTEN_MASK_DIM_S2;
                    } else {
                        vecPhyAddr->calPreToken = 0;
                    }
                }
            }
        }
        globalVecAddr->blockLength = blockId;
    }

    __aicore__ inline int32_t InitBNIndex()
    {
        int32_t recoderS1 = GetRecoderS(s1Idx, dimS1);
        bool updateS1Idx;
        if (sparseMode == 0 || sparseMode == 1) {
            updateS1Idx = (s2Idx == dimS2) ? true : false;
        } else {
            // 当矩阵左下角超过右边界，表示当前S2Idx已遍历到最后
            updateS1Idx = isLeftDownOverTouchRight(s1Idx + recoderS1, s2Idx, sparseRightBound);
        }

        if (updateS1Idx) {
            // 处理一轮计算结束时，S2正好全部遍历完的情况。
            s1Idx += recoderS1;
            s2Idx = 0;
        }

        if (s1Idx < dimS1) {
            return false;
        }

        if (n1Idx < dimN1 - 1) {
            s1Idx = 0;
            s2Idx = 0;
            n1Idx++;
            return false;
        }

        if (bIdx < dimB - 1) {
            s1Idx = 0;
            s2Idx = 0;
            n1Idx = 0;
            bIdx++;
            UpdateSeqLen();
            return false;
        }

        return true;
    }

    template <bool IS_AIV>
    __aicore__ inline void ComputeAddrInfo()
    {
        while (true) {
            if (InitBNIndex()) {
                break;
            }

            while (s1Idx < dimS1) {
                int32_t recoderS1 = GetRecoderS(s1Idx, dimS1);
                int32_t recoderS2 = GetRecoderS(s2Idx, dimS2);

                if (recoderS2 == 0) {
                    s1Idx += recoderS1;
                    s2Idx = 0;
                    continue;
                }

                if (!IsFullMaskBlock(s1Idx, s2Idx, recoderS1, recoderS2, sparseLeftBound, sparseRightBound,
                                     sparseMode)) {
                    DetRecord<IS_AIV>(recoderS1, recoderS2);
                    if constexpr (IS_AIV) {
                        VecRecord(recoderS1, recoderS2);
                    } else {
                        CubeRecord(recoderS1, recoderS2);
                    }
                    UpdateLoopIdx();
                }
                s2Idx += recoderS2;
                if (loopFinish) {
                    break;
                }
            }

            if (loopFinish) {
                loopFinish = 0;
                break;
            }
        }
    }
};
} // namespace FAG_DET

#endif // __ADDR_COMPUTE_DET_H__
