/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_gating_top_k_softmax_v2_perf_arch35.h
 * \brief
 */
#ifndef MOE_GATING_TOP_K_SOFTMAX_V2_PERF_REGBASE
#define MOE_GATING_TOP_K_SOFTMAX_V2_PERF_REGBASE

#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"

namespace MoeGatingTopKSoftmaxV2 {
using namespace AscendC;

constexpr int32_t DB_BUFFER_NUM = 2;
constexpr int32_t BLOCK_BYTES = 32;
constexpr int32_t BLOCK_B32_SIZE = 8;
constexpr int32_t REPEAT_B32_SIZE = 64;
constexpr int32_t REDUCE_MAX_SIZE = 64;
constexpr int32_t CONSTANT_TWO = 2;
constexpr int32_t CONSTANT_THREE = 3;
constexpr int32_t CONSTANT_FOUR = 4;
constexpr int32_t CONSTANT_FIVE = 5;
constexpr int32_t CONSTANT_SIX = 6;
constexpr int32_t CONSTANT_SEVEN = 7;
constexpr int32_t CONSTANT_EIGHT = 8;
constexpr int32_t ZERO_MASK = 0b0;
constexpr int32_t SORT_UNIT = 32;
constexpr int32_t MERGE_UNIT = 128;
constexpr int32_t MERGE_LIST_MAX_NUM = 4;
constexpr int32_t MERGE_TWO = 0b0011;
constexpr int32_t MERGE_FOUR = 0b1111;
constexpr int32_t VL_FP32 = Ops::Base::GetVRegSize() / sizeof(float);
constexpr MicroAPI::CastTrait castTraitINT82INT32 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                     MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};

template <typename T, int32_t renorm>
class MoeGatingTopKSoftmaxV2Perf {
public:
    __aicore__ inline MoeGatingTopKSoftmaxV2Perf(){};
    __aicore__ inline void Init(GM_ADDR gating, GM_ADDR finished, GM_ADDR out, GM_ADDR indicesOut, GM_ADDR softmaxOut,
                                GM_ADDR workspace,
                                const MoeGatingTopKSoftmaxV2PerfRegbaseTilingData *__restrict tilingData)
    {
        // init gm inputs
        int64_t formerblockLength = tilingData->blockFormer * tilingData->col;
        this->softmaxFlag = tilingData->softmaxFlag;
        int64_t blockLength =
            (GetBlockIdx() != tilingData->blockNum - 1) ? formerblockLength : tilingData->blockTail * tilingData->col;
        gatingTensorGM.SetGlobalBuffer((__gm__ T *)gating + formerblockLength * GetBlockIdx(), blockLength);
        if (finished != nullptr) {
            exitFinished = true;
            int64_t blockLengthFinished =
                (GetBlockIdx() != tilingData->blockNum - 1) ? tilingData->blockFormer : tilingData->blockTail;
            finishedTensorGM.SetGlobalBuffer((__gm__ bool *)finished + tilingData->blockFormer * GetBlockIdx(),
                                             blockLengthFinished);
        }
        // init gm outputs
        int64_t outFormerBlockLength = tilingData->blockFormer * tilingData->k;
        int64_t outBlockLength =
            (GetBlockIdx() != tilingData->blockNum - 1) ? outFormerBlockLength : tilingData->blockTail * tilingData->k;
        outTensorGM.SetGlobalBuffer((__gm__ T *)out + outFormerBlockLength * GetBlockIdx(), outBlockLength);
        indicesOutTensorGM.SetGlobalBuffer((__gm__ int32_t *)indicesOut + outFormerBlockLength * GetBlockIdx(),
                                           outBlockLength);
        softmaxResultGM.SetGlobalBuffer((__gm__ float *)softmaxOut + formerblockLength * GetBlockIdx(), blockLength);
        // init queues
        int32_t bufferSize = tilingData->bufferElemSize * sizeof(int32_t);
        pipe.InitBuffer(gatingQueue, DB_BUFFER_NUM, tilingData->bufferElemSize * sizeof(T));
        pipe.InitBuffer(finishedQueue, DB_BUFFER_NUM,
                        (tilingData->ubFormer * sizeof(bool) + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES);
        pipe.InitBuffer(topKOutsQueue, 1, bufferSize * CONSTANT_TWO);
        pipe.InitBuffer(tmpTensor, bufferSize * CONSTANT_TWO);
        pipe.InitBuffer(softmaxResultOutQueue, 1, bufferSize);
    }

    __aicore__ inline void Process(const MoeGatingTopKSoftmaxV2PerfRegbaseTilingData *__restrict tilingData)
    {
        int32_t ubLoopCount = (GetBlockIdx() == tilingData->blockNum - 1) ? tilingData->ubLoopOfTailBlock :
                                                                            tilingData->ubLoopOfFormerBlock;
        int32_t tailRowsNum = (GetBlockIdx() == tilingData->blockNum - 1) ? tilingData->ubTailOfTailBlock :
                                                                            tilingData->ubTailOfFormerBlock;
        // preload
        CopyIn(0, (0 == ubLoopCount - 1) ? tailRowsNum : tilingData->ubFormer, tilingData);
        for (int32_t i = 0; i < ubLoopCount - 1; i++) {
            ComputePhase1(i, tilingData->ubFormer, tilingData);
            CopyIn(i + 1, (i == ubLoopCount - 1 - 1) ? tailRowsNum : tilingData->ubFormer, tilingData);
            CopyOutPhase0(i, tilingData->ubFormer, tilingData);
        }
        ComputePhase1(ubLoopCount - 1, tailRowsNum, tilingData);
        CopyOutPhase0(ubLoopCount - 1, tailRowsNum, tilingData);
    }

private:
    __aicore__ inline void CopyIn(const int32_t OuterIdx, const int32_t curRowsNum,
                                  const MoeGatingTopKSoftmaxV2PerfRegbaseTilingData *__restrict tilingData)
    {
        LocalTensor<T> gatingLocal = gatingQueue.AllocTensor<T>();
        DataCopyPadParams padParams{false, 0, 0, 0};
        DataCopyParams intriParams;
        if (tilingData->colBytesAlign - tilingData->col != 0) {
            intriParams.blockCount = curRowsNum;
            intriParams.blockLen = tilingData->col * sizeof(T);
        } else {
            intriParams.blockCount = 1;
            intriParams.blockLen = curRowsNum * tilingData->col * sizeof(T);
        }
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
        DataCopyPad(gatingLocal, gatingTensorGM[tilingData->ubFormer * tilingData->col * OuterIdx], intriParams,
                    padParams);
        gatingQueue.EnQue(gatingLocal);

        if (exitFinished) {
            LocalTensor<bool> finishedLocal = finishedQueue.AllocTensor<bool>();
            DataCopyParams intriParamsFinished;
            intriParamsFinished.blockCount = 1;
            intriParamsFinished.blockLen = curRowsNum * sizeof(bool);
            intriParamsFinished.srcStride = 0;
            intriParamsFinished.dstStride = 0;
            DataCopyPad(finishedLocal, finishedTensorGM[tilingData->ubFormer * OuterIdx], intriParamsFinished,
                        padParams);
            finishedQueue.EnQue(finishedLocal);
        }
    }

    __aicore__ inline void Rearrange(const LocalTensor<float> &dst, const LocalTensor<float> &src,
                                     const int32_t curRowsNum,
                                     const MoeGatingTopKSoftmaxV2PerfRegbaseTilingData *__restrict tilingData)
    {
        // softmax is 32 bytes aligned while topk must be 32 element aligned
        DataCopyParams intriParams;
        intriParams.blockCount = curRowsNum;
        intriParams.blockLen = tilingData->colBytesAlign / CONSTANT_EIGHT;
        intriParams.srcStride = 0;
        intriParams.dstStride = (tilingData->colAlign - tilingData->colBytesAlign) / CONSTANT_EIGHT;
        DataCopy(dst, src, intriParams);
    }

    __aicore__ inline void DuplicatePad(const LocalTensor<float> &dst, const int32_t curRowsNum,
                                        const MoeGatingTopKSoftmaxV2PerfRegbaseTilingData *__restrict tilingData)
    {
        if (tilingData->colAlign - tilingData->col != 0) {
            uint32_t tmp = 0xFF800000; // -inf
            uint64_t mask[2] = {(((uint64_t)1 << (tilingData->colAlign - tilingData->col)) - 1)
                                    << (SORT_UNIT - (tilingData->colAlign - tilingData->col)),
                                0};
            Duplicate(dst[tilingData->colAlign - SORT_UNIT], *((float *)&tmp), mask, curRowsNum, 1,
                      tilingData->colAlign / CONSTANT_EIGHT);
        }
    }

    __aicore__ inline void InitIndex(const LocalTensor<int32_t> &dst, const int32_t curRowsNum,
                                     const MoeGatingTopKSoftmaxV2PerfRegbaseTilingData *__restrict tilingData)
    {
        __ubuf__ int32_t *ubDst = (__ubuf__ int32_t *)dst.GetPhyAddr();
        uint16_t loops = curRowsNum;
        uint16_t repeatTime = Ops::Base::CeilDiv(tilingData->colAlign, static_cast<uint32_t>(REPEAT_B32_SIZE));

        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<int32_t> vreg;
            MicroAPI::MaskReg mask;
            uint32_t sregMask = tilingData->colAlign;
            for (uint16_t i = 0; i < repeatTime; i++) {
                mask = MicroAPI::UpdateMask<int32_t>(sregMask);
                MicroAPI::Arange(vreg, i * REPEAT_B32_SIZE);
                for (uint16_t j = 0; j < loops; j++) {
                    MicroAPI::StoreAlign(ubDst + i * REPEAT_B32_SIZE + j * tilingData->colAlign, vreg, mask);
                }
            }
        }
    }

    __aicore__ inline void SortFP32Perf(const LocalTensor<float> &dst, const LocalTensor<float> &src0,
                                        const LocalTensor<int32_t> &src1, const int32_t curRowsNum,
                                        const MoeGatingTopKSoftmaxV2PerfRegbaseTilingData *__restrict tilingData)
    {
        Sort32(dst, src0, src1.ReinterpretCast<uint32_t>(), tilingData->colAlign * curRowsNum / SORT_UNIT);
    }

    __aicore__ inline void MergeSortFP32PerfCopy(const LocalTensor<float> &dst, const LocalTensor<float> &src,
                                                 const int32_t curRowsNum, const int32_t srcColsNum,
                                                 const int32_t dstColsNum)
    {
        if (srcColsNum != dstColsNum * MERGE_LIST_MAX_NUM) {
            uint32_t tmp = 0xFF800000; // -inf
            Duplicate(dst, *((float *)&tmp), dstColsNum * CONSTANT_TWO * curRowsNum);
        }
        DataCopyParams intriParams;
        intriParams.blockCount = curRowsNum;
        intriParams.blockLen = BLOCK_B32_SIZE * CONSTANT_TWO / CONSTANT_EIGHT;
        intriParams.srcStride = (srcColsNum - BLOCK_B32_SIZE) * CONSTANT_TWO / CONSTANT_EIGHT;
        intriParams.dstStride = (dstColsNum - BLOCK_B32_SIZE) * CONSTANT_TWO / CONSTANT_EIGHT;
        // 32 -> 8
        for (int32_t i = 0; i < srcColsNum / SORT_UNIT; i++) {
            DataCopy(dst[BLOCK_B32_SIZE * CONSTANT_TWO * i], src[SORT_UNIT * CONSTANT_TWO * i], intriParams);
        }
    }

    __aicore__ inline void MergeSortFP32PerfBlockMerge(const LocalTensor<float> &dst, const LocalTensor<float> &src,
                                                       const int32_t repeatTimes)
    {
        MrgSort4Info params;
        MrgSortSrcList<float> srcList;
        params.ifExhaustedSuspension = false;
        params.elementLengths[0] = BLOCK_B32_SIZE;
        params.elementLengths[1] = BLOCK_B32_SIZE;
        params.elementLengths[CONSTANT_TWO] = BLOCK_B32_SIZE;
        params.elementLengths[CONSTANT_THREE] = BLOCK_B32_SIZE;
        params.validBit = MERGE_FOUR;
        params.repeatTimes = repeatTimes;
        srcList.src1 = src[0];
        srcList.src2 = src[BLOCK_B32_SIZE * CONSTANT_TWO];
        srcList.src3 = src[BLOCK_B32_SIZE * CONSTANT_TWO * CONSTANT_TWO];
        srcList.src4 = src[BLOCK_B32_SIZE * CONSTANT_TWO * CONSTANT_THREE];
        MrgSort(dst, srcList, params);
    }

    __aicore__ inline void MergeSortFP32Perf2To1(const LocalTensor<float> &dst, const LocalTensor<float> &src,
                                                 const LocalTensor<float> &tmpBuffer, const int32_t curRowsNum,
                                                 const int32_t tailOffset)
    {
        // former + tail -> 1
        MrgSort4Info params;
        MrgSortSrcList<float> srcList;
        params.ifExhaustedSuspension = false;
        params.elementLengths[0] = BLOCK_B32_SIZE;
        params.elementLengths[1] = BLOCK_B32_SIZE;
        params.validBit = MERGE_TWO;
        params.repeatTimes = 1;
        for (int32_t i = 0; i < curRowsNum; i++) {
            srcList.src1 = src[i * BLOCK_B32_SIZE * MERGE_LIST_MAX_NUM * CONSTANT_TWO];
            srcList.src2 = src[i * BLOCK_B32_SIZE * MERGE_LIST_MAX_NUM * CONSTANT_TWO + tailOffset];
            MrgSort(tmpBuffer[i * BLOCK_B32_SIZE * CONSTANT_TWO * CONSTANT_TWO], srcList, params);
        }
        // so that col == BLOCK_B32_SIZE * MERGE_LIST_MAX_NUM
        DataCopyParams intriParams;
        intriParams.blockCount = curRowsNum;
        intriParams.blockLen = BLOCK_B32_SIZE * CONSTANT_TWO * CONSTANT_TWO / CONSTANT_EIGHT;
        intriParams.srcStride = 0;
        intriParams.dstStride =
            (BLOCK_B32_SIZE * MERGE_LIST_MAX_NUM - BLOCK_B32_SIZE * CONSTANT_TWO) * CONSTANT_TWO / CONSTANT_EIGHT;
        DataCopy(dst, tmpBuffer, intriParams);
    }

    __aicore__ inline void MergeSortFP32Perf(const LocalTensor<float> &dst, const LocalTensor<float> &src,
                                             const LocalTensor<float> &tmpBuffer, const int32_t curRowsNum,
                                             const MoeGatingTopKSoftmaxV2PerfRegbaseTilingData *__restrict tilingData)
    {
        if (tilingData->colAlign <= MERGE_UNIT * MERGE_LIST_MAX_NUM) {
            int32_t curColsNum = tilingData->colAlign;
            if (tilingData->colAlign > MERGE_UNIT) {
                // 16 -> 4
                MergeSortFP32PerfCopy(tmpBuffer, src, curRowsNum, tilingData->colAlign, MERGE_UNIT);
                MergeSortFP32PerfBlockMerge(src, tmpBuffer, curRowsNum * MERGE_LIST_MAX_NUM);
                curColsNum = MERGE_UNIT;
            }
            // 4 -> 1
            MergeSortFP32PerfCopy(tmpBuffer, src, curRowsNum, curColsNum, BLOCK_B32_SIZE * MERGE_LIST_MAX_NUM);
            MergeSortFP32PerfBlockMerge(dst, tmpBuffer, curRowsNum);
        } else {
            // former is 512
            DataCopyParams intriParams;
            intriParams.blockCount = curRowsNum;
            intriParams.blockLen = BLOCK_B32_SIZE * CONSTANT_TWO / CONSTANT_EIGHT;
            intriParams.srcStride = (tilingData->colAlign - BLOCK_B32_SIZE) * CONSTANT_TWO / CONSTANT_EIGHT;
            intriParams.dstStride = (MERGE_UNIT - BLOCK_B32_SIZE) * CONSTANT_TWO / CONSTANT_EIGHT;
            for (int32_t i = 0; i < MERGE_UNIT * MERGE_LIST_MAX_NUM / SORT_UNIT; i++) {
                DataCopy(tmpBuffer[BLOCK_B32_SIZE * CONSTANT_TWO * i], src[SORT_UNIT * CONSTANT_TWO * i], intriParams);
            }
            // tail is colAlign - 512
            int32_t tailColsNum = tilingData->colAlign - MERGE_UNIT * MERGE_LIST_MAX_NUM;
            int32_t dstColsNum = tailColsNum > MERGE_UNIT ? MERGE_UNIT : BLOCK_B32_SIZE * MERGE_LIST_MAX_NUM;
            int32_t tailOffset = MERGE_UNIT * CONSTANT_TWO * curRowsNum;
            if (tailColsNum != dstColsNum * MERGE_LIST_MAX_NUM) {
                uint32_t tmp = 0xFF800000; // -inf
                Duplicate(tmpBuffer[tailOffset], *((float *)&tmp), dstColsNum * CONSTANT_TWO * curRowsNum);
            }
            intriParams.dstStride = (dstColsNum - BLOCK_B32_SIZE) * CONSTANT_TWO / CONSTANT_EIGHT;
            for (int32_t i = 0; i < tailColsNum / SORT_UNIT; i++) {
                DataCopy(tmpBuffer[BLOCK_B32_SIZE * CONSTANT_TWO * i + tailOffset],
                         src[SORT_UNIT * CONSTANT_TWO * i + MERGE_UNIT * MERGE_LIST_MAX_NUM * CONSTANT_TWO],
                         intriParams);
            }
            // former merge
            MergeSortFP32PerfBlockMerge(dst, tmpBuffer, curRowsNum * MERGE_LIST_MAX_NUM);
            MergeSortFP32PerfCopy(tmpBuffer, dst, curRowsNum, MERGE_UNIT, BLOCK_B32_SIZE * MERGE_LIST_MAX_NUM);
            MergeSortFP32PerfBlockMerge(dst, tmpBuffer, curRowsNum);
            // tail merge
            if (tailColsNum > MERGE_UNIT) {
                MergeSortFP32PerfBlockMerge(dst[tailOffset], tmpBuffer[tailOffset], curRowsNum * MERGE_LIST_MAX_NUM);
                MergeSortFP32PerfCopy(tmpBuffer[tailOffset], dst[tailOffset], curRowsNum, MERGE_UNIT,
                                      BLOCK_B32_SIZE * MERGE_LIST_MAX_NUM);
            }
            MergeSortFP32PerfBlockMerge(dst[tailOffset], tmpBuffer[tailOffset], curRowsNum);
            MergeSortFP32Perf2To1(dst, dst, tmpBuffer, curRowsNum, tailOffset);
        }
    }

    __aicore__ inline void ExtractKFP32Perf(const LocalTensor<float> &dstValues, const LocalTensor<int32_t> &dstIndices,
                                            const LocalTensor<int32_t> &src, const int32_t curRowsNum,
                                            const MoeGatingTopKSoftmaxV2PerfRegbaseTilingData *__restrict tilingData)
    {
        __ubuf__ float *ubValue = (__ubuf__ float *)dstValues.GetPhyAddr();
        __ubuf__ int32_t *ubIndex = (__ubuf__ int32_t *)dstIndices.GetPhyAddr();
        __ubuf__ int32_t *ubSrc = (__ubuf__ int32_t *)src.GetPhyAddr();
        uint16_t loops = curRowsNum;
        uint32_t sregMask;

        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<int32_t> vreg0;
            MicroAPI::RegTensor<int32_t> vreg1;
            MicroAPI::UnalignRegForStore ureg0;
            MicroAPI::UnalignRegForStore ureg1;
            MicroAPI::MaskReg mask;

            for (uint16_t i = 0; i < loops; i++) {
                MicroAPI::LoadAlign<int32_t, MicroAPI::LoadDist::DIST_DINTLV_B32>(vreg0, vreg1, ubSrc + i * VL_FP32);
                if constexpr (renorm == 0) {
                    MicroAPI::StoreUnAlign(ubValue, (MicroAPI::RegTensor<float> &)vreg0, ureg0, tilingData->k);
                } else {
                    sregMask = tilingData->kAlign;
                    mask = MicroAPI::UpdateMask<float>(sregMask);
                    MicroAPI::StoreAlign(ubValue + i * tilingData->kAlign, (MicroAPI::RegTensor<float> &)vreg0, mask);
                }
                MicroAPI::StoreUnAlign(ubIndex, vreg1, ureg1, tilingData->k);
            }
            if constexpr (renorm == 0) {
                MicroAPI::StoreUnAlignPost(ubValue, ureg0, 0);
            }
            MicroAPI::StoreUnAlignPost(ubIndex, ureg1, 0);
        }
    }

    __aicore__ inline void UpdateIndices(const LocalTensor<int32_t> &src, const int32_t curRowsNum,
                                         const MoeGatingTopKSoftmaxV2PerfRegbaseTilingData *__restrict tilingData)
    {
        LocalTensor<bool> finishedLocal = finishedQueue.DeQue<bool>();
        __ubuf__ int8_t *finishedUb = (__ubuf__ int8_t *)finishedLocal.GetPhyAddr();
        __ubuf__ int32_t *srcUb = (__ubuf__ int32_t *)src.GetPhyAddr();
        __ubuf__ int32_t *dstUb = (__ubuf__ int32_t *)src.GetPhyAddr();

        uint16_t loops = curRowsNum;
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<int8_t> vreg0;
            MicroAPI::RegTensor<int32_t> vreg1;
            MicroAPI::RegTensor<int32_t> vreg2;
            MicroAPI::RegTensor<int32_t> vregSrc;
            MicroAPI::UnalignRegForLoad uregX;
            MicroAPI::UnalignRegForLoad uregSrc;
            MicroAPI::UnalignRegForStore uregDst;
            MicroAPI::MaskReg preg1;
            MicroAPI::MaskReg preg2;

            uint32_t finCount = 1;
            uint32_t srcCount = tilingData->k;
            preg1 = MicroAPI::UpdateMask<int8_t>(finCount);
            preg2 = MicroAPI::UpdateMask<int32_t>(srcCount);

            MicroAPI::LoadUnAlignPre(uregX, finishedUb);
            MicroAPI::LoadUnAlignPre(uregSrc, srcUb);
            for (uint16_t i = 0; i < loops; i++) {
                MicroAPI::LoadUnAlign(vreg0, uregX, finishedUb, 1);
                MicroAPI::LoadUnAlign(vregSrc, uregSrc, srcUb, tilingData->k);
                MicroAPI::Cast<int32_t, int8_t, castTraitINT82INT32>(vreg1, vreg0, preg1);
                MicroAPI::Muls(vreg1, vreg1, int32_t(tilingData->col), preg1);
                MicroAPI::Duplicate(vreg2, vreg1, preg2);
                MicroAPI::Max(vreg2, vreg2, vregSrc, preg2);
                MicroAPI::StoreUnAlign(dstUb, vreg2, uregDst, tilingData->k);
            }
            MicroAPI::StoreUnAlignPost(dstUb, uregDst, 0);
        }
        finishedQueue.FreeTensor(finishedLocal);
    }

    __aicore__ inline void TopKFP32Perf(const LocalTensor<float> &dstValues, const LocalTensor<int32_t> &dstIndices,
                                        const LocalTensor<float> &src, const LocalTensor<float> &tmpBuffer,
                                        const int32_t curRowsNum,
                                        const MoeGatingTopKSoftmaxV2PerfRegbaseTilingData *__restrict tilingData)
    {
        if (tilingData->k == 1 && tilingData->colBytesAlign <= REDUCE_MAX_SIZE) {
            WholeReduceMax(tmpBuffer, src, tilingData->col, curRowsNum, BLOCK_BYTES, 1,
                           tilingData->colBytesAlign / CONSTANT_EIGHT, ReduceOrder::ORDER_VALUE_INDEX);
        } else {
            Rearrange(dstValues, src, curRowsNum, tilingData);
            DuplicatePad(dstValues, curRowsNum, tilingData);
            InitIndex(dstIndices, curRowsNum, tilingData);
            SortFP32Perf(tmpBuffer, dstValues, dstIndices, curRowsNum, tilingData);
            if (tilingData->colAlign > SORT_UNIT) {
                MergeSortFP32Perf(tmpBuffer, tmpBuffer, dstValues, curRowsNum, tilingData);
            }
        }
        ExtractKFP32Perf(dstValues, dstIndices, tmpBuffer.ReinterpretCast<int32_t>(), curRowsNum, tilingData);
        if (exitFinished) {
            UpdateIndices(dstIndices, curRowsNum, tilingData);
        }
    }

    __aicore__ inline void CopyOutSoftmax(const int32_t outerIdx, const int32_t curRowsNum,
                                          LocalTensor<float> &softmaxResultOutLocal,
                                          const MoeGatingTopKSoftmaxV2PerfRegbaseTilingData *__restrict tilingData)
    {
        softmaxResultOutQueue.EnQue<float>(softmaxResultOutLocal);
        softmaxResultOutLocal = softmaxResultOutQueue.DeQue<float>();
        DataCopyParams intriParams;
        if (tilingData->colBytesAlign != tilingData->col) {
            intriParams.blockCount = curRowsNum;
            intriParams.blockLen = tilingData->col * sizeof(float);
        } else {
            intriParams.blockCount = 1;
            intriParams.blockLen = curRowsNum * tilingData->col * sizeof(float);
        }
        DataCopyPad(softmaxResultGM[tilingData->ubFormer * tilingData->col * outerIdx], softmaxResultOutLocal,
                    intriParams);
        auto eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventID);
        WaitFlag<HardEvent::MTE3_V>(eventID);
    }

    __aicore__ inline void GatherSoftmaxResult(const LocalTensor<int32_t> &softmaxResult, const int32_t curRowsNum,
                                               const MoeGatingTopKSoftmaxV2PerfRegbaseTilingData *__restrict tilingData)
    {
        __ubuf__ float *ubSrc = (__ubuf__ float *)softmaxResult.GetPhyAddr();
        __ubuf__ float *ubDst = (__ubuf__ float *)softmaxResult.GetPhyAddr();
        uint16_t loops = curRowsNum;

        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<float> vregx;
            MicroAPI::UnalignRegForStore ureg;
            for (uint16_t i = 0; i < loops; i++) {
                MicroAPI::LoadAlign(vregx, ubSrc + i * BLOCK_B32_SIZE);
                MicroAPI::StoreUnAlign(ubDst, vregx, ureg, tilingData->k);
            }
            MicroAPI::StoreUnAlignPost(ubDst, ureg, 0);
        }
    }

    __aicore__ inline void ComputeRenorm(LocalTensor<T> &gatingLocal, const LocalTensor<int32_t> &topKOutsLocal,
                                         const LocalTensor<float> &tmpTensorLocal, const int32_t curRowsNum,
                                         const MoeGatingTopKSoftmaxV2PerfRegbaseTilingData *__restrict tilingData)
    {
        SoftMaxShapeInfo softmaxShapeInfo;
        softmaxShapeInfo.srcK =
            (tilingData->k * sizeof(float) + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES / sizeof(float);
        softmaxShapeInfo.srcM = curRowsNum;
        softmaxShapeInfo.oriSrcK = tilingData->k;
        softmaxShapeInfo.oriSrcM = curRowsNum;

        if constexpr (IsSameType<T, float>::value) {
            TopKFP32Perf(topKOutsLocal.ReinterpretCast<float>(), topKOutsLocal[tilingData->bufferElemSize], gatingLocal,
                         tmpTensorLocal, curRowsNum, tilingData);
            gatingQueue.FreeTensor(gatingLocal);
            SoftMax<float, true, false>(topKOutsLocal.ReinterpretCast<float>(), topKOutsLocal.ReinterpretCast<float>(),
                                        tilingData->softmaxTilingData, softmaxShapeInfo);
            PipeBarrier<PIPE_ALL>();
            GatherSoftmaxResult(topKOutsLocal, curRowsNum, tilingData);
        } else {
            if (tilingData->col < CONSTANT_EIGHT) {
                Cast(tmpTensorLocal, gatingLocal, RoundMode::CAST_NONE, tilingData->colBytesAlign, curRowsNum,
                     {1, 1, 1, 1});
            } else {
                Cast(tmpTensorLocal, gatingLocal, RoundMode::CAST_NONE, curRowsNum * tilingData->colBytesAlign);
            }
            gatingQueue.FreeTensor(gatingLocal);
            TopKFP32Perf(topKOutsLocal.ReinterpretCast<float>(), topKOutsLocal[tilingData->bufferElemSize],
                         tmpTensorLocal, tmpTensorLocal[tilingData->bufferElemSize], curRowsNum, tilingData);
            SoftMax<float, true, false>(topKOutsLocal.ReinterpretCast<float>(), topKOutsLocal.ReinterpretCast<float>(),
                                        tilingData->softmaxTilingData, softmaxShapeInfo);
            PipeBarrier<PIPE_ALL>();
            GatherSoftmaxResult(topKOutsLocal, curRowsNum, tilingData);
            Cast(topKOutsLocal.ReinterpretCast<T>(), topKOutsLocal.ReinterpretCast<float>(), RoundMode::CAST_RINT,
                 tilingData->k * curRowsNum);
        }
    }

    __aicore__ inline void ComputePhase1(const int32_t OuterIdx, const int32_t curRowsNum,
                                         const MoeGatingTopKSoftmaxV2PerfRegbaseTilingData *__restrict tilingData)
    {
        LocalTensor<T> gatingLocal = gatingQueue.DeQue<T>();
        LocalTensor<int32_t> topKOutsLocal = topKOutsQueue.AllocTensor<int32_t>();
        LocalTensor<float> tmpTensorLocal = tmpTensor.Get<float>();
        LocalTensor<float> softmaxResultOutLocal = softmaxResultOutQueue.AllocTensor<float>();
        if constexpr (renorm == 0) {
            SoftMaxShapeInfo softmaxShapeInfo;
            softmaxShapeInfo.srcK =
                (tilingData->col * sizeof(float) + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES / sizeof(float);
            softmaxShapeInfo.srcM = curRowsNum;
            softmaxShapeInfo.oriSrcK = tilingData->col;
            softmaxShapeInfo.oriSrcM = curRowsNum;

            if constexpr (IsSameType<T, float>::value) {
                SoftMax<float, true, false>(softmaxResultOutLocal, gatingLocal, tilingData->softmaxTilingData,
                                            softmaxShapeInfo);
                PipeBarrier<PIPE_ALL>();
                gatingQueue.FreeTensor(gatingLocal);
                if (this->softmaxFlag == 1) {
                    CopyOutSoftmax(OuterIdx, curRowsNum, softmaxResultOutLocal, tilingData);
                }
                TopKFP32Perf(topKOutsLocal.ReinterpretCast<float>(), topKOutsLocal[tilingData->bufferElemSize],
                             softmaxResultOutLocal, tmpTensorLocal, curRowsNum, tilingData);
            } else {
                if (tilingData->col < CONSTANT_EIGHT) {
                    Cast(tmpTensorLocal, gatingLocal, RoundMode::CAST_NONE, tilingData->colBytesAlign, curRowsNum,
                         {1, 1, 1, 1});
                } else {
                    Cast(tmpTensorLocal, gatingLocal, RoundMode::CAST_NONE, curRowsNum * tilingData->colBytesAlign);
                }
                gatingQueue.FreeTensor(gatingLocal);
                SoftMax<float, true, false>(softmaxResultOutLocal, tmpTensorLocal, tilingData->softmaxTilingData,
                                            softmaxShapeInfo);
                PipeBarrier<PIPE_ALL>();
                if (this->softmaxFlag == 1) {
                    CopyOutSoftmax(OuterIdx, curRowsNum, softmaxResultOutLocal, tilingData);
                }
                TopKFP32Perf(topKOutsLocal.ReinterpretCast<float>(), topKOutsLocal[tilingData->bufferElemSize],
                             softmaxResultOutLocal, tmpTensorLocal, curRowsNum, tilingData);
                Cast(topKOutsLocal.ReinterpretCast<T>(), topKOutsLocal.ReinterpretCast<float>(), RoundMode::CAST_RINT,
                     curRowsNum * tilingData->k);
            }
        } else {
            ComputeRenorm(gatingLocal, topKOutsLocal, tmpTensorLocal, curRowsNum, tilingData);
        }
        topKOutsQueue.EnQue(topKOutsLocal);
        softmaxResultOutQueue.FreeTensor(softmaxResultOutLocal);
    }

    __aicore__ inline void CopyOutPhase0(const int32_t OuterIdx, const int32_t curRowsNum,
                                         const MoeGatingTopKSoftmaxV2PerfRegbaseTilingData *__restrict tilingData)
    {
        LocalTensor<int32_t> topKOutsLocal = topKOutsQueue.DeQue<int32_t>();
        int64_t gmIndex = tilingData->ubFormer * tilingData->k * OuterIdx;
        DataCopyParams intriParams;
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
        intriParams.blockCount = 1;
        intriParams.blockLen = curRowsNum * tilingData->k * sizeof(int32_t);
        DataCopyPad(indicesOutTensorGM[gmIndex], topKOutsLocal[tilingData->bufferElemSize], intriParams);

        auto eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventID);
        WaitFlag<HardEvent::V_MTE3>(eventID);
        intriParams.blockCount = 1;
        intriParams.blockLen = curRowsNum * tilingData->k * sizeof(T);
        DataCopyPad(outTensorGM[gmIndex], topKOutsLocal.ReinterpretCast<T>(), intriParams);
        topKOutsQueue.FreeTensor(topKOutsLocal);
    }

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, DB_BUFFER_NUM> gatingQueue;
    TQue<QuePosition::VECIN, DB_BUFFER_NUM> finishedQueue;
    TQue<QuePosition::VECOUT, 1> topKOutsQueue;
    TQue<QuePosition::VECOUT, 1> softmaxResultOutQueue;

    TBuf<> tmpTensor;

    GlobalTensor<T> gatingTensorGM;
    GlobalTensor<bool> finishedTensorGM;
    GlobalTensor<T> outTensorGM;
    GlobalTensor<int32_t> indicesOutTensorGM;
    GlobalTensor<float> softmaxResultGM;

    bool exitFinished{false};
    int64_t softmaxFlag;
};
} // namespace MoeGatingTopKSoftmaxV2
#endif
