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
 * \file swin_transformer_ln_qkv_quant.cpp
 * \brief
 */

#include "swin_transformer_ln_qkv_quant_base.h"

using namespace AscendC;
constexpr int32_t DATA_COPY_UINT32 = 32;

template <typename aDType, typename bDType, typename cDType, bool aTrans, bool bTrans, bool isReuseSource = false>
class SwinTransformerLnQkvQuantNormalKernel
    : public SwinTransformerLnQkvQuantBase<aDType, bDType, cDType, aTrans, bTrans, false> {
public:
    __aicore__ inline SwinTransformerLnQkvQuantNormalKernel(){};

    __aicore__ inline void Init(GM_ADDR inputxGm, GM_ADDR gammaGm, GM_ADDR betaGm, GM_ADDR weight, GM_ADDR bias,
                                GM_ADDR quantScale, GM_ADDR quantOffset, GM_ADDR dequantScale, GM_ADDR queryOutput,
                                GM_ADDR keyOutput, GM_ADDR valueOutput,
                                const SwinTransformerLnQkvQuantTilingData* tiling, GM_ADDR userWorkspace, TPipe* tPipe)
    {
        bLength = tiling->opBaseInfo.bSize;
        sLength = tiling->opBaseInfo.sSize;
        hLength = tiling->opBaseInfo.hSize;
        lnBaseM = tiling->opBaseInfo.lnBaseM;
        lnBaseK = tiling->opBaseInfo.lnBaseK;
        mmBaseN = tiling->weightN;
        windowSize = tiling->opBaseInfo.wWinSize;
        sizePerHead = tiling->opBaseInfo.sizePerHead;
        patchHeight = tiling->opBaseInfo.patchHeight;
        heightWindow = tiling->opBaseInfo.hWinSize;
        patchWeight = tiling->opBaseInfo.patchWeight;
        batchNum = 1;
        headNum = tiling->opBaseInfo.headNum;
        epsilon = tiling->epsilon;

        avgFactor = avgFactor / hLength;
        lnBaseM16Align = tiling->opBaseInfo.lnBufferM;
        lnBufferK = tiling->opBaseInfo.lnBufferK;
        mmSizeN = tiling->mmInfo.mmSizeN;
        mmSizeM = tiling->mmInfo.mmSizeM;
        mmSizeK = tiling->mmInfo.mmSizeK;
        subMLoop = tiling->opBaseInfo.lnMSubLoop;
        splitNFlag = (mmSizeN != mmBaseN);
        nSplitPart = mmBaseN / mmSizeN;

        weightGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bDType*>(weight), lnBaseK * mmBaseN);
        biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(bias), mmBaseN);
        gammaGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(gammaGm), hLength);
        betaGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(betaGm), hLength);
        scaleGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(quantScale), hLength);
        offsetGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(quantOffset), hLength);
        quantGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(dequantScale), mmBaseN);

        curBlockIdx = AscendC::GetBlockIdx();
        int64_t blockDim = AscendC::GetBlockNum();

        bool isTailBlock = (curBlockIdx == (blockDim - 1));
        winOffsetNumPerCore = tiling->opBaseInfo.singleCoreLnBsSize * curBlockIdx;
        uint32_t inputSize = tiling->size;
        if (isTailBlock) {
            lnBsSize = bLength * sLength - winOffsetNumPerCore;
        } else {
            lnBsSize = tiling->opBaseInfo.singleCoreLnBsSize;
        }
        loopNum = DivUp(lnBsSize, mmSizeM);
        inputXGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(inputxGm), inputSize);
        qOutputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ cDType*>(queryOutput), inputSize);
        kOutputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ cDType*>(keyOutput), inputSize);
        vOutputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ cDType*>(valueOutput), inputSize);
        mmOutWorkspace.SetGlobalBuffer(reinterpret_cast<__gm__ cDType*>(userWorkspace), bLength * sLength * mmBaseN);

        tPipe->InitBuffer(gammaTmpBuf, hLength * sizeof(half));
        tPipe->InitBuffer(betaTmpBuf, hLength * sizeof(half));
        tPipe->InitBuffer(quantScaleTmpBuf, hLength * sizeof(half));
        tPipe->InitBuffer(quantOffsetTmpBuf, hLength * sizeof(half));
        tPipe->InitBuffer(fp32Gamma, hLength * sizeof(float));
        tPipe->InitBuffer(fp32Beta, hLength * sizeof(float));

        tPipe->InitBuffer(lnSharedBuffer, tiling->tmpShareBufferForLn);
        tPipe->InitBuffer(ubShareForQuant, tiling->tmpBufferForQuant);
        tPipe->InitBuffer(inBufAL1, RoundUp(mmSizeM, BLOCK_CUBE) * mmSizeK * sizeof(int8_t));
        tPipe->InitBuffer(inQueueb, 1, lnBaseM16Align * lnBufferK * sizeof(half));
        REGIST_MATMUL_OBJ(tPipe, GetSysWorkSpacePtr(), mm, &tiling->mmTilingParams);
        AscendC::LocalTensor<uint8_t> sharedUbBuffer = lnSharedBuffer.Get<uint8_t>();
        mm.SetLocalWorkspace(sharedUbBuffer[mmSizeM * mmSizeN * sizeof(half)]);
    }

    __aicore__ inline void Process(const SwinTransformerLnQkvQuantTilingData* tiling)
    {
        AscendC::LocalTensor<half> fp16Gamma = gammaTmpBuf.Get<half>();
        AscendC::LocalTensor<half> fp16Beta = betaTmpBuf.Get<half>();
        AscendC::DataCopy(fp16Gamma, gammaGlobal, hLength);
        AscendC::DataCopy(fp16Beta, betaGlobal, hLength);

        AscendC::LocalTensor<half> quantScale = quantScaleTmpBuf.Get<half>();
        AscendC::LocalTensor<half> quantOffset = quantOffsetTmpBuf.Get<half>();
        AscendC::DataCopy(quantScale, scaleGlobal, hLength);
        AscendC::DataCopy(quantOffset, offsetGlobal, hLength);

        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

        mm.SetTensorB(weightGlobal, bTrans);
        mm.SetBias(biasGlobal);
        mm.SetQuantVector(quantGlobal);
        uint32_t gmCopyInOffset = winOffsetNumPerCore * hLength;
        uint32_t gmCopyOutOffset = gmCopyInOffset;
        AscendC::LocalTensor<float> sharedTmpBuffer = lnSharedBuffer.Get<float>();
        uint32_t mmOutSize = lnBaseM16Align * lnBufferK / 2; // fp32 -> fp16 is 2
        AscendC::LocalTensor<float> xFp32 = sharedTmpBuffer[mmOutSize];
        AscendC::LocalTensor<float> tmpSub = sharedTmpBuffer[lnBufferK * lnBaseM16Align + mmOutSize];
        AscendC::LocalTensor<float> mean = sharedTmpBuffer[lnBufferK * lnBaseM16Align * 2 + mmOutSize];

        AscendC::LocalTensor<float> fp32GammaLocal = fp32Gamma.Get<float>();
        AscendC::LocalTensor<float> fp32BetaLocal = fp32Beta.Get<float>();
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        Cast(fp32GammaLocal, fp16Gamma, AscendC::RoundMode::CAST_NONE, lnBaseK);
        Cast(fp32BetaLocal, fp16Beta, AscendC::RoundMode::CAST_NONE, lnBaseK);
        PipeBarrier<PIPE_V>();
        AscendC::LocalTensor<int8_t> aL1Buffer = inBufAL1.Get<int8_t>();
        AscendC::LocalTensor<int8_t> mmANz = lnSharedBuffer.Get<int8_t>();
        AscendC::LocalTensor<int8_t> matAInt8Ub = mmANz[lnBaseM16Align * lnBufferK * sizeof(half)];

        DataCopyParams intriParams2L1;
        AscendC::GlobalTensor<half> dstGlobal[3] = {qOutputGlobal, kOutputGlobal, vOutputGlobal}; // q k v sum is 3
        event_t eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        for (uint32_t loopIdx = 0; loopIdx < loopNum; ++loopIdx) {
            actualMmBaseM = (lnBsSize - loopIdx * mmSizeM > mmSizeM) ? mmSizeM : (lnBsSize - loopIdx * mmSizeM);
            subMLoop = DivUp(actualMmBaseM, lnBaseM16Align);
            for (uint32_t subMLoopIdx = 0; subMLoopIdx < subMLoop; ++subMLoopIdx) {
                if (subMLoop == 1) {
                    actualLnBaseM = (lnBsSize - loopIdx * mmSizeM - subMLoopIdx * lnBaseM > lnBaseM) ?
                                        lnBaseM :
                                        (lnBsSize - loopIdx * mmSizeM - subMLoopIdx * lnBaseM);
                } else {
                    actualLnBaseM = (lnBsSize - loopIdx * mmSizeM - subMLoopIdx * lnBaseM16Align > lnBaseM16Align) ?
                                        lnBaseM16Align :
                                        (lnBsSize - loopIdx * mmSizeM - subMLoopIdx * lnBaseM16Align);
                    actualLnBaseM = ((mmSizeM - subMLoopIdx * lnBaseM16Align) < actualLnBaseM) ?
                                        (mmSizeM - subMLoopIdx * lnBaseM16Align) :
                                        actualLnBaseM;
                }
                uint32_t mAlign16 = (matmul::CeilDiv(actualLnBaseM, BLOCK_CUBE) * BLOCK_CUBE);

                CopyIn(loopIdx, subMLoopIdx, gmCopyInOffset, subMLoop);

                if (subMLoopIdx == 0) {
                    WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
                }
                if (lnBaseK < 256) { // repeat max is 256
                    LnCompute(xFp32, mean, tmpSub, fp32GammaLocal, fp32BetaLocal);
                } else {
                    LnComputeNlarge(xFp32, mean, tmpSub, fp32GammaLocal, fp32BetaLocal);
                }
                PipeBarrier<PIPE_V>();
                Quant(matAInt8Ub, loopIdx, gmCopyInOffset + subMLoopIdx * lnBaseM16Align * lnBaseK);
                PipeBarrier<PIPE_V>();
                this->VecND2NZ(mmANz, matAInt8Ub, actualLnBaseM, lnBaseK, lnBaseK);

                event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
                SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
                WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
                intriParams2L1.blockCount = matmul::CeilDiv(lnBaseK, BLOCK_SIZE_32);
                intriParams2L1.blockLen = mAlign16;
                intriParams2L1.srcStride = 0;
                intriParams2L1.dstStride = matmul::CeilDiv(actualMmBaseM, BLOCK_CUBE) * BLOCK_CUBE - mAlign16;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200 || \
    (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
                if (subMLoop == 1) {
                    DataCopy(aL1Buffer[subMLoopIdx * lnBaseK * actualLnBaseM], mmANz, mAlign16 * lnBaseK);
                } else {
                    DataCopy(aL1Buffer[subMLoopIdx * lnBaseM16Align * BLOCK_SIZE_32], mmANz, intriParams2L1);
                }
#endif
                PipeBarrier<PIPE_ALL>();
            }
            if (splitNFlag) {
                for (uint32_t mmNSubIdx = 0; mmNSubIdx < nSplitPart; ++mmNSubIdx) {
                    mm.SetBias(biasGlobal[mmSizeN * mmNSubIdx]);
                    mm.SetQuantVector(quantGlobal[mmSizeN * mmNSubIdx]);
                    Matmul(loopIdx, gmCopyOutOffset, mmNSubIdx);
                    event_t eventIdVToMte3Out = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
                    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3Out);
                    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3Out);
                    CopyOutForSplitN(loopIdx, gmCopyOutOffset, dstGlobal[mmNSubIdx]);
                }
            } else {
                Matmul(loopIdx, gmCopyOutOffset, 0);
                event_t eventIdVToMte3Out = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
                SetFlag<HardEvent::V_MTE3>(eventIdVToMte3Out);
                WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3Out);
                CopyOut(loopIdx, gmCopyOutOffset);
            }
            SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        }
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
    }

private:
    __aicore__ inline void GetGmOffset(uint32_t innerLoopIdx, uint32_t& outputOffset)
    {
        uint32_t curLoopGlobalIdx = innerLoopIdx;
        uint32_t batchIdx = curLoopGlobalIdx / (patchHeight * heightWindow * patchWeight);
        uint32_t heightPatchIdx =
            (curLoopGlobalIdx - batchIdx * (patchHeight * heightWindow * patchWeight)) / (heightWindow * patchWeight);
        uint32_t windowHeightIdx = (curLoopGlobalIdx - batchIdx * (patchHeight * heightWindow * patchWeight) -
                                    heightPatchIdx * (heightWindow * patchWeight)) /
                                   patchWeight;
        uint32_t weightPatchIdx = curLoopGlobalIdx - batchIdx * (patchHeight * heightWindow * patchWeight) -
                                  heightPatchIdx * (heightWindow * patchWeight) - windowHeightIdx * patchWeight;
        outputOffset = (batchIdx * patchHeight * patchWeight + heightPatchIdx * patchWeight + weightPatchIdx) *
                       headNum * heightWindow * windowSize * sizePerHead;
        outputOffset += windowHeightIdx * windowSize * sizePerHead;
    }

    __aicore__ inline void LnCompute(LocalTensor<float> xLocalFp32, LocalTensor<float> meanTmp,
                                     LocalTensor<float> tmpSub, LocalTensor<float> fp32Gamma,
                                     LocalTensor<float> fp32Beta)
    {
        GenLnParams();
        LocalTensor<half> inputXLocal = inQueueb.DeQue<half>();
        AscendC::LocalTensor<half> lnOutLocalFp16 = lnSharedBuffer.Get<half>();

        constexpr uint64_t mask = 64; // fp32 blockNum is 64
        int32_t roll = lnBaseK / (BLOCK_UINT / sizeof(half));
        Duplicate(meanTmp, float(0.0), lnBaseM16Align);
        addRoll = actualLnBaseM / mask;
        int32_t addRollTail = actualLnBaseM - addRoll * mask;
        int32_t bakRoll = lnBaseK / mask;
        int32_t bakRollTail = lnBaseK - bakRoll * mask;
        PipeBarrier<PIPE_V>();
        constexpr uint64_t blockNumTrans = BLOCK_SIZE_16;

        uint64_t dstLocalList[BLOCK_SIZE_16];
        uint64_t srcLocalList[BLOCK_SIZE_16];
        int dstOffset;
        int srcOffset;
        for (int32_t j = 0; j < roll; j++) {
            dstOffset = 0;
            srcOffset = 0;
            for (int32_t i = 0; i < blockNumTrans; i++) {
                dstOffset = (BLOCK_SIZE_16 * lnBaseM16Align * j + i * lnBaseM16Align);
                srcOffset = (BLOCK_SIZE_16 * j + i * lnBufferK);
                dstLocalList[i] = (uint64_t)(lnOutLocalFp16[dstOffset].GetPhyAddr());
                srcLocalList[i] = (uint64_t)(inputXLocal[srcOffset].GetPhyAddr());
            }
            TransDataTo5HD<half>(dstLocalList, srcLocalList, transParams);
        }
        PipeBarrier<PIPE_V>();
        Cast(xLocalFp32, lnOutLocalFp16, AscendC::RoundMode::CAST_NONE, lnBaseM16Align * lnBufferK);
        PipeBarrier<PIPE_V>();
        for (int32_t ll = 0; ll < addRoll; ll++) {
            Add(meanTmp[mask * ll], xLocalFp32[mask * ll], meanTmp[mask * ll], mask, lnBufferK, params);
        }
        if (addRollTail > 0) {
            Add(meanTmp[mask * addRoll], xLocalFp32[mask * addRoll], meanTmp[mask * addRoll], addRollTail, lnBufferK,
                params);
        }
        PipeBarrier<PIPE_V>();
        Muls(meanTmp, meanTmp, avgFactor, actualLnBaseM);

        PipeBarrier<PIPE_V>();
        for (int32_t ll = 0; ll < addRoll; ll++) {
            Sub(tmpSub[mask * ll], xLocalFp32[mask * ll], meanTmp[mask * ll], mask, lnBufferK, params1);
        }
        if (addRollTail > 0) {
            Sub(tmpSub[mask * addRoll], xLocalFp32[mask * addRoll], meanTmp[mask * addRoll], addRollTail, lnBufferK,
                params1);
        }
        PipeBarrier<PIPE_V>();
        Mul(xLocalFp32, tmpSub, tmpSub, lnBufferK * lnBaseM16Align);
        PipeBarrier<PIPE_V>();
        Duplicate(meanTmp, float(0.0), lnBaseM16Align);
        PipeBarrier<PIPE_V>();
        for (int32_t ll = 0; ll < addRoll; ll++) {
            Add(meanTmp[mask * ll], xLocalFp32[mask * ll], meanTmp[mask * ll], mask, lnBufferK, params);
        }
        if (addRollTail > 0) {
            Add(meanTmp[mask * addRoll], xLocalFp32[mask * addRoll], meanTmp[mask * addRoll], addRollTail, lnBufferK,
                params);
        }
        PipeBarrier<PIPE_V>();
        Muls(meanTmp, meanTmp, avgFactor, actualLnBaseM);
        PipeBarrier<PIPE_V>();
        Adds(meanTmp, meanTmp, epsilon, actualLnBaseM);
        PipeBarrier<PIPE_V>();
        Sqrt(meanTmp, meanTmp, actualLnBaseM);
        PipeBarrier<PIPE_V>();
        for (int32_t ll = 0; ll < addRoll; ll++) {
            Div(xLocalFp32[mask * ll], tmpSub[mask * ll], meanTmp[mask * ll], mask, lnBufferK, params1);
        }
        if (addRollTail > 0) {
            Div(xLocalFp32[mask * addRoll], tmpSub[mask * addRoll], meanTmp[mask * addRoll], addRollTail, lnBufferK,
                params1);
        }
        PipeBarrier<PIPE_V>();

        for (int32_t k1 = 0; k1 < roll; k1++) {
            dstOffset = 0;
            srcOffset = 0;
            for (int32_t i = 0; i < blockNumTrans; i++) {
                dstOffset = BLOCK_SIZE_16 * k1 + lnBufferK * (i / 2) + (i % 2) * 8; // 2 8 is for float32
                srcOffset = BLOCK_SIZE_16 * lnBaseM16Align * k1 + i * lnBaseM16Align;
                dstLocalList[i] = (uint64_t)(tmpSub[dstOffset].GetPhyAddr());
                srcLocalList[i] = (uint64_t)(xLocalFp32[srcOffset].GetPhyAddr());
            }
            TransDataTo5HD<float>(dstLocalList, srcLocalList, transParams1);
        }
        PipeBarrier<PIPE_V>();
        for (int32_t ll = 0; ll < bakRoll; ll++) {
            Mul(xLocalFp32[mask * ll], tmpSub[mask * ll], fp32Gamma[mask * ll], mask, actualLnBaseM, params2);
        }
        if (bakRollTail > 0) {
            Mul(xLocalFp32[mask * bakRoll], tmpSub[mask * bakRoll], fp32Gamma[mask * bakRoll], bakRollTail,
                actualLnBaseM, params2);
        }
        PipeBarrier<PIPE_V>();
        for (int32_t ll = 0; ll < bakRoll; ll++) {
            Add(tmpSub[mask * ll], xLocalFp32[mask * ll], fp32Beta[mask * ll], mask, actualLnBaseM, params2);
        }
        if (bakRollTail > 0) {
            Add(tmpSub[mask * bakRoll], xLocalFp32[mask * bakRoll], fp32Beta[mask * bakRoll], bakRollTail,
                actualLnBaseM, params2);
        }
        PipeBarrier<PIPE_V>();
        Cast(lnOutLocalFp16, tmpSub, AscendC::RoundMode::CAST_NONE, actualLnBaseM * lnBaseK);
        inQueueb.FreeTensor(inputXLocal);
    }
    __aicore__ inline void LnComputeNlarge(LocalTensor<float> xLocalFp32, LocalTensor<float> meanTmp,
                                           LocalTensor<float> tmpSub, LocalTensor<float> fp32Gamma,
                                           LocalTensor<float> fp32Beta)
    {
        GenLnParamsNlarge();
        LocalTensor<half> inputXLocal = inQueueb.DeQue<half>();
        constexpr uint64_t mask = 64;                                  // fp32 blockNum is 64
        constexpr uint32_t alignForFloat = BLOCK_UINT / sizeof(float); // 8
        constexpr uint64_t repeatMax = 255;                            // max repeta Times is 255
        Duplicate(meanTmp, float(0.0), lnBaseM16Align);
        int32_t roll = lnBaseM16Align / BLOCK_SIZE_16;
        int32_t roll8 = lnBaseM16Align / (BLOCK_UINT / sizeof(float));
        int32_t addRollM = lnBaseM16Align / mask;
        int32_t addRollMTail = lnBaseM16Align - addRollM * mask;
        int32_t rollK = lnBufferK / repeatMax;
        int32_t rollKTail = lnBufferK - rollK * repeatMax;
        int32_t mAlign = RoundUp(lnBaseM16Align, alignForFloat);

        int32_t bakRoll = lnBaseM16Align / repeatMax;
        int32_t bakRollTail = lnBaseM16Align - bakRoll * repeatMax;
        int32_t bakRollK = lnBufferK / mask;
        int32_t bakRollKTail = lnBufferK - bakRollK * mask;
        PipeBarrier<PIPE_V>();
        AscendC::LocalTensor<half> lnOutLocalFp16 = lnSharedBuffer.Get<half>();
        constexpr uint64_t blockNumTrans = BLOCK_SIZE_16;
        uint64_t dstLocalList[BLOCK_SIZE_16];
        uint64_t srcLocalList[BLOCK_SIZE_16];
        int dstOffset;
        int srcOffset;
        for (int32_t j = 0; j < roll; j++) {
            dstOffset = 0;
            srcOffset = 0;
            for (int32_t i = 0; i < blockNumTrans; i++) {
                dstOffset = (BLOCK_SIZE_16 * j + i * lnBaseM16Align);
                srcOffset = (BLOCK_SIZE_16 * j * lnBufferK + i * lnBufferK);
                dstLocalList[i] = (uint64_t)(lnOutLocalFp16[dstOffset].GetPhyAddr());
                srcLocalList[i] = (uint64_t)(inputXLocal[srcOffset].GetPhyAddr());
            }
            TransDataTo5HD<half>(dstLocalList, srcLocalList, transParams);
        }
        PipeBarrier<PIPE_V>();
        Cast(xLocalFp32, lnOutLocalFp16, AscendC::RoundMode::CAST_NONE, lnBaseM16Align * lnBufferK);
        PipeBarrier<PIPE_V>();

        for (int32_t lll = 0; lll < addRollM; lll++) {
            for (int32_t ll = 0; ll < rollK; ll++) {
                Add(meanTmp[mask * lll], xLocalFp32[repeatMax * lnBaseM16Align * ll + mask * lll], meanTmp[mask * lll],
                    mask, repeatMax, params);
            }
            if (rollKTail > 0) {
                Add(meanTmp[mask * lll], xLocalFp32[repeatMax * lnBaseM16Align * rollK + mask * lll],
                    meanTmp[mask * lll], mask, rollKTail, params);
            }
        }
        if (addRollMTail > 0) {
            for (int32_t ll = 0; ll < rollK; ll++) {
                Add(meanTmp[mask * addRollM], xLocalFp32[repeatMax * lnBaseM16Align * ll + mask * addRollM],
                    meanTmp[mask * addRollM], addRollMTail, repeatMax, params);
            }
            if (rollKTail > 0) {
                Add(meanTmp[mask * addRollM], xLocalFp32[repeatMax * lnBaseM16Align * rollK + mask * addRollM],
                    meanTmp[mask * addRollM], addRollMTail, rollKTail, params);
            }
        }
        PipeBarrier<PIPE_V>();
        Muls(meanTmp, meanTmp, avgFactor, mAlign);
        PipeBarrier<PIPE_V>();
        for (int32_t lll = 0; lll < addRollM; lll++) {
            for (int32_t ll = 0; ll < rollK; ll++) {
                Sub(tmpSub[repeatMax * ll * lnBaseM16Align + mask * lll],
                    xLocalFp32[repeatMax * lnBaseM16Align * ll + mask * lll], meanTmp[mask * lll], mask, repeatMax,
                    params1);
            }
            if (rollKTail > 0) {
                Sub(tmpSub[repeatMax * rollK * lnBaseM16Align + mask * lll],
                    xLocalFp32[repeatMax * lnBaseM16Align * rollK + mask * lll], meanTmp[mask * lll], mask, rollKTail,
                    params1);
            }
        }
        if (addRollMTail > 0) {
            for (int32_t ll = 0; ll < rollK; ll++) {
                Sub(tmpSub[repeatMax * ll * lnBaseM16Align + mask * addRollM],
                    xLocalFp32[repeatMax * lnBaseM16Align * ll + mask * addRollM], meanTmp[mask * addRollM],
                    addRollMTail, repeatMax, params1);
            }
            if (rollKTail > 0) {
                Sub(tmpSub[repeatMax * rollK * lnBaseM16Align + mask * addRollM],
                    xLocalFp32[repeatMax * lnBaseM16Align * rollK + mask * addRollM], meanTmp[mask * addRollM],
                    addRollMTail, rollKTail, params1);
            }
        }
        PipeBarrier<PIPE_V>();
        Mul(xLocalFp32, tmpSub, tmpSub, lnBufferK * lnBaseM16Align);
        PipeBarrier<PIPE_V>();
        Duplicate(meanTmp, float(0.0), lnBaseM16Align);
        PipeBarrier<PIPE_V>();
        for (int32_t lll = 0; lll < addRollM; lll++) {
            for (int32_t ll = 0; ll < rollK; ll++) {
                Add(meanTmp[mask * lll], xLocalFp32[repeatMax * lnBaseM16Align * ll + mask * lll], meanTmp[mask * lll],
                    mask, repeatMax, params);
            }
            if (rollKTail > 0) {
                Add(meanTmp[mask * lll], xLocalFp32[repeatMax * lnBaseM16Align * rollK + mask * lll],
                    meanTmp[mask * lll], mask, rollKTail, params);
            }
        }
        if (addRollMTail > 0) {
            for (int32_t ll = 0; ll < rollK; ll++) {
                Add(meanTmp[mask * addRollM], xLocalFp32[repeatMax * lnBaseM16Align * ll + mask * addRollM],
                    meanTmp[mask * addRollM], addRollMTail, repeatMax, params);
            }
            if (rollKTail > 0) {
                Add(meanTmp[mask * addRollM], xLocalFp32[repeatMax * lnBaseM16Align * rollK + mask * addRollM],
                    meanTmp[mask * addRollM], addRollMTail, rollKTail, params);
            }
        }
        Muls(meanTmp, meanTmp, avgFactor, mAlign);
        PipeBarrier<PIPE_V>();
        Adds(meanTmp, meanTmp, epsilon, mAlign);
        PipeBarrier<PIPE_V>();
        Sqrt(meanTmp, meanTmp, mAlign);
        PipeBarrier<PIPE_V>();
        for (int32_t lll = 0; lll < addRollM; lll++) {
            for (int32_t ll = 0; ll < rollK; ll++) {
                Div(xLocalFp32[repeatMax * ll * lnBaseM16Align + mask * lll],
                    tmpSub[repeatMax * lnBaseM16Align * ll + mask * lll], meanTmp[mask * lll], mask, repeatMax,
                    params1);
            }
            if (rollKTail > 0) {
                Div(xLocalFp32[repeatMax * rollK * lnBaseM16Align + mask * lll],
                    tmpSub[repeatMax * lnBaseM16Align * rollK + mask * lll], meanTmp[mask * lll], mask, rollKTail,
                    params1);
            }
        }
        if (addRollMTail > 0) {
            for (int32_t ll = 0; ll < rollK; ll++) {
                Div(xLocalFp32[repeatMax * ll * lnBaseM16Align + mask * addRollM],
                    tmpSub[repeatMax * lnBaseM16Align * ll + mask * addRollM], meanTmp[mask * addRollM], addRollMTail,
                    repeatMax, params1);
            }
            if (rollKTail > 0) {
                Div(xLocalFp32[repeatMax * rollK * lnBaseM16Align + mask * addRollM],
                    tmpSub[repeatMax * lnBaseM16Align * rollK + mask * addRollM], meanTmp[mask * addRollM],
                    addRollMTail, rollKTail, params1);
            }
        }
        PipeBarrier<PIPE_V>();
        for (int32_t k1 = 0; k1 < roll8; k1++) {
            dstOffset = 0;
            srcOffset = 0;
            for (int32_t i = 0; i < blockNumTrans; i++) {
                dstOffset =
                    alignForFloat * k1 * lnBufferK + lnBufferK * (i / 2) + (i % 2) * alignForFloat; // 2 if for float
                srcOffset = alignForFloat * k1 + i * lnBaseM16Align;
                dstLocalList[i] = (uint64_t)(tmpSub[dstOffset].GetPhyAddr());
                srcLocalList[i] = (uint64_t)(xLocalFp32[srcOffset].GetPhyAddr());
            }
            TransDataTo5HD<float>(dstLocalList, srcLocalList, transParams1);
        }
        PipeBarrier<PIPE_V>();
        for (int32_t lll = 0; lll < bakRollK; lll++) {
            for (int32_t ll = 0; ll < bakRoll; ll++) {
                Mul(xLocalFp32[repeatMax * ll * lnBufferK + mask * lll],
                    tmpSub[repeatMax * lnBufferK * ll + mask * lll], fp32Gamma[mask * lll], mask, repeatMax, params2);
            }
            if (bakRollTail > 0) {
                Mul(xLocalFp32[repeatMax * bakRoll * lnBufferK + mask * lll],
                    tmpSub[repeatMax * lnBufferK * bakRoll + mask * lll], fp32Gamma[mask * lll], mask, bakRollTail,
                    params2);
            }
        }
        if (bakRollKTail > 0) {
            for (int32_t ll = 0; ll < bakRoll; ll++) {
                Mul(xLocalFp32[repeatMax * ll * lnBufferK + mask * bakRollK],
                    tmpSub[repeatMax * lnBufferK * ll + mask * bakRollK], fp32Gamma[mask * bakRollK], bakRollKTail,
                    repeatMax, params2);
            }
            if (bakRollTail > 0) {
                Mul(xLocalFp32[repeatMax * bakRoll * lnBufferK + mask * bakRollK],
                    tmpSub[repeatMax * lnBufferK * bakRoll + mask * bakRollK], fp32Gamma[mask * bakRollK], bakRollKTail,
                    bakRollTail, params2);
            }
        }
        PipeBarrier<PIPE_V>();
        for (int32_t lll = 0; lll < bakRollK; lll++) {
            for (int32_t ll = 0; ll < bakRoll; ll++) {
                Add(tmpSub[repeatMax * ll * lnBufferK + mask * lll],
                    xLocalFp32[repeatMax * lnBufferK * ll + mask * lll], fp32Beta[mask * lll], mask, repeatMax,
                    params2);
            }
            if (bakRollTail > 0) {
                Add(tmpSub[repeatMax * bakRoll * lnBufferK + mask * lll],
                    xLocalFp32[repeatMax * lnBufferK * bakRoll + mask * lll], fp32Beta[mask * lll], mask, bakRollTail,
                    params2);
            }
        }
        if (bakRollKTail > 0) {
            for (int32_t ll = 0; ll < bakRoll; ll++) {
                Add(tmpSub[repeatMax * ll * lnBufferK + mask * bakRollK],
                    xLocalFp32[repeatMax * lnBufferK * ll + mask * bakRollK], fp32Beta[mask * bakRollK], bakRollKTail,
                    repeatMax, params2);
            }
            if (bakRollTail > 0) {
                Add(tmpSub[repeatMax * bakRoll * lnBufferK + mask * bakRollK],
                    xLocalFp32[repeatMax * lnBufferK * bakRoll + mask * bakRollK], fp32Beta[mask * bakRollK],
                    bakRollKTail, bakRollTail, params2);
            }
        }
        PipeBarrier<PIPE_V>();
        Cast(lnOutLocalFp16, tmpSub, AscendC::RoundMode::CAST_NONE, actualLnBaseM * lnBaseK);
        inQueueb.FreeTensor(inputXLocal);
    }

    __aicore__ inline void CopyIn(uint32_t innerLoopIdx, uint32_t subMLoopIdx, uint32_t gmCopyInOffset,
                                  uint32_t subMLoop)
    {
        uint32_t offset = gmCopyInOffset + innerLoopIdx * mmSizeM * lnBaseK;
        if (subMLoop > 1) {
            offset += subMLoopIdx * lnBaseM16Align * lnBaseK;
        }
        AscendC::LocalTensor<half> bLocal = inQueueb.AllocTensor<half>();
        AscendC::DataCopy(bLocal, inputXGlobal[offset], actualLnBaseM * lnBaseK);
        inQueueb.EnQue<half>(bLocal);
    }

    __aicore__ inline void CopyOut(uint32_t innerLoopIdx, uint32_t gmCopyOutOffset)
    {
        AscendC::LocalTensor<half> mmOutTmpFp16 = lnSharedBuffer.Get<half>();

        DataCopyParams intriParamsForSplit;
        intriParamsForSplit.blockCount = windowSize;
        intriParamsForSplit.blockLen = sizePerHead * sizeof(half) / DATA_COPY_UINT32;
        intriParamsForSplit.srcStride = mmBaseN * sizeof(half) / DATA_COPY_UINT32 - intriParamsForSplit.blockLen;
        intriParamsForSplit.dstStride = 0;
        uint32_t gmCopyOut = 0;
        for (uint32_t copyIdx = 0; copyIdx < (actualMmBaseM / windowSize); ++copyIdx) {
            GetGmOffset((winOffsetNumPerCore / windowSize) + innerLoopIdx * (mmSizeM / windowSize) + copyIdx,
                        gmCopyOut);
            for (uint32_t headIdx = 0; headIdx < headNum; ++headIdx) {
                AscendC::DataCopy(qOutputGlobal[gmCopyOut + headIdx * heightWindow * windowSize * sizePerHead],
                                  mmOutTmpFp16[headIdx * sizePerHead + windowSize * mmBaseN * copyIdx],
                                  intriParamsForSplit);
                AscendC::DataCopy(kOutputGlobal[gmCopyOut + headIdx * heightWindow * windowSize * sizePerHead],
                                  mmOutTmpFp16[lnBaseK + headIdx * sizePerHead + windowSize * mmBaseN * copyIdx],
                                  intriParamsForSplit);
                AscendC::DataCopy(
                    vOutputGlobal[gmCopyOut + headIdx * heightWindow * windowSize * sizePerHead],
                    mmOutTmpFp16[lnBaseK * 2 + headIdx * sizePerHead + windowSize * mmBaseN * copyIdx], // offset 2
                    intriParamsForSplit);
            }
        }
    }
    __aicore__ inline void CopyOutForSplitN(uint32_t innerLoopIdx, uint32_t gmCopyOutOffset, GlobalTensor<half>& dst)
    {
        AscendC::LocalTensor<half> mmOutTmpFp16 = lnSharedBuffer.Get<half>();

        DataCopyParams intriParamsForSplit;
        intriParamsForSplit.blockCount = windowSize;
        intriParamsForSplit.blockLen = sizePerHead * sizeof(half) / DATA_COPY_UINT32;
        intriParamsForSplit.srcStride = mmSizeN * sizeof(half) / DATA_COPY_UINT32 - intriParamsForSplit.blockLen;
        intriParamsForSplit.dstStride = 0;
        uint32_t gmCopyOut = 0;
        for (uint32_t copyIdx = 0; copyIdx < (actualMmBaseM / windowSize); ++copyIdx) {
            GetGmOffset((winOffsetNumPerCore / windowSize) + innerLoopIdx * (mmSizeM / windowSize) + copyIdx,
                        gmCopyOut);
            for (uint32_t headIdx = 0; headIdx < headNum; ++headIdx) {
                AscendC::DataCopy(dst[gmCopyOut + headIdx * heightWindow * windowSize * sizePerHead],
                                  mmOutTmpFp16[headIdx * sizePerHead + windowSize * mmSizeN * copyIdx],
                                  intriParamsForSplit);
            }
        }
    }
    __aicore__ inline void Quant(AscendC::LocalTensor<int8_t>& matAInt8Ub, uint32_t innerLoopIdx,
                                 uint32_t gmCopyOutOffset)
    {
        AscendC::LocalTensor<half> scale = quantScaleTmpBuf.Get<half>();
        AscendC::LocalTensor<half> offset = quantOffsetTmpBuf.Get<half>();
        AscendC::LocalTensor<uint8_t> quantTmpBuffer = ubShareForQuant.Get<uint8_t>();
        AscendC::LocalTensor<half> lnOutLocalFp16 = lnSharedBuffer.Get<half>();

        AscendC::AscendQuant(matAInt8Ub, lnOutLocalFp16, quantTmpBuffer, scale, offset, lnBaseK, lnBaseK,
                             actualLnBaseM * lnBaseK);
    }
    __aicore__ inline void Matmul(uint32_t innerLoopIdx, uint32_t gmCopyOutOffset, uint32_t mmSubNIdx)
    {
        AscendC::LocalTensor<int8_t> aL1Buffer = inBufAL1.Get<int8_t>();
        AscendC::LocalTensor<half> mmOutTmpFp16 = lnSharedBuffer.Get<half>();
        mm.SetTensorA(aL1Buffer);
        if (splitNFlag) {
            if (bTrans) {
                mm.SetTensorB(weightGlobal[mmSizeK * mmSizeN * mmSubNIdx], bTrans);
            } else {
                mm.SetTensorB(weightGlobal[mmSizeK * mmSubNIdx], bTrans);
            }
        }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200 || \
    (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
        mm.SetTail(actualMmBaseM, -1, -1);
        mm.template IterateAll<true>(mmOutTmpFp16, false);
#endif
        mm.End();
    }
    __aicore__ inline void GenLnParams()
    {
        uint8_t addSrcRep = lnBaseM16Align / (BLOCK_UINT / sizeof(float));
        uint8_t bakRep = lnBaseK / (BLOCK_UINT / sizeof(float));
        uint16_t srcRep = BLOCK_SIZE_16 * lnBufferK / BLOCK_SIZE_16;
        uint8_t bakRepeat = lnBaseM16Align / (BLOCK_UINT / sizeof(float));
        uint8_t repeat = lnBaseM16Align / (BLOCK_UINT / sizeof(half));
        params.dstBlkStride = 1;
        params.src0BlkStride = 1;
        params.src1BlkStride = 1;
        params.dstRepStride = 0;
        params.src0RepStride = addSrcRep;
        params.src1RepStride = 0;

        params1.dstBlkStride = 1;
        params1.src0BlkStride = 1;
        params1.src1BlkStride = 1;
        params1.dstRepStride = addSrcRep;
        params1.src0RepStride = addSrcRep;
        params1.src1RepStride = 0;

        params2.dstBlkStride = 1;
        params2.src0BlkStride = 1;
        params2.src1BlkStride = 1;
        params2.dstRepStride = bakRep;
        params2.src0RepStride = bakRep;
        params2.src1RepStride = 0;

        transParams.dstHighHalf = 0;
        transParams.srcHighHalf = 0;
        transParams.repeatTimes = repeat;
        transParams.dstRepStride = 1;
        transParams.srcRepStride = srcRep;
        if (repeat == 1) {
            transParams.dstRepStride = 0;
            transParams.srcRepStride = 0;
        }
        transParams1.dstHighHalf = 0;
        transParams1.srcHighHalf = 0;
        transParams1.repeatTimes = bakRepeat;
        transParams1.dstRepStride = lnBufferK;
        transParams1.srcRepStride = 1;
        if (bakRepeat == 1) {
            transParams1.dstRepStride = 0;
            transParams1.srcRepStride = 0;
        }
    }
    __aicore__ inline void GenLnParamsNlarge()
    {
        uint8_t addSrcRep = lnBaseM16Align / (BLOCK_UINT / sizeof(float));
        uint8_t bakRep = lnBaseK / (BLOCK_UINT / sizeof(float));

        uint8_t bakRepeat = lnBaseK / BLOCK_SIZE_16;
        uint8_t repeat = lnBaseK / BLOCK_SIZE_16;
        params.dstBlkStride = 1;
        params.src0BlkStride = 1;
        params.src1BlkStride = 1;
        params.dstRepStride = 0;
        params.src0RepStride = addSrcRep;
        params.src1RepStride = 0;

        params1.dstBlkStride = 1;
        params1.src0BlkStride = 1;
        params1.src1BlkStride = 1;
        params1.dstRepStride = addSrcRep;
        params1.src0RepStride = addSrcRep;
        params1.src1RepStride = 0;

        params2.dstBlkStride = 1;
        params2.src0BlkStride = 1;
        params2.src1BlkStride = 1;
        params2.dstRepStride = bakRep;
        params2.src0RepStride = bakRep;
        params2.src1RepStride = 0;

        transParams.dstHighHalf = 0;
        transParams.srcHighHalf = 0;
        transParams.repeatTimes = repeat;
        transParams.dstRepStride = lnBaseM16Align;
        transParams.srcRepStride = 1;
        if (repeat == 1) {
            transParams.dstRepStride = 0;
            transParams.srcRepStride = 0;
        }
        transParams1.dstHighHalf = 0;
        transParams1.srcHighHalf = 0;
        transParams1.repeatTimes = bakRepeat;
        transParams1.dstRepStride = 2;                  // 2 is for fp32 in block unit
        transParams1.srcRepStride = lnBaseM16Align * 2; // 2 is for fp32
        if (bakRepeat == 1) {
            transParams1.dstRepStride = 0;
            transParams1.srcRepStride = 0;
        }
    }

private:
    AscendC::GlobalTensor<aDType> inputXGlobal;
    AscendC::GlobalTensor<aDType> gammaGlobal;
    AscendC::GlobalTensor<aDType> betaGlobal;
    AscendC::GlobalTensor<bDType> weightGlobal;
    AscendC::GlobalTensor<int32_t> biasGlobal;
    AscendC::GlobalTensor<half> scaleGlobal;
    AscendC::GlobalTensor<half> offsetGlobal;
    AscendC::GlobalTensor<uint64_t> quantGlobal;

    AscendC::GlobalTensor<cDType> qOutputGlobal;
    AscendC::GlobalTensor<cDType> kOutputGlobal;
    AscendC::GlobalTensor<cDType> vOutputGlobal;
    AscendC::GlobalTensor<cDType> mmOutWorkspace;

    AscendC::TBuf<AscendC::TPosition::VECCALC> lnSharedBuffer;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ubShareForMmOut;
    AscendC::TBuf<AscendC::TPosition::VECCALC> lnOutBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> matAInt8;
    AscendC::TBuf<AscendC::TPosition::VECCALC> matAInt8Nz;
    AscendC::TBuf<AscendC::TPosition::VECIN> gammaTmpBuf;
    AscendC::TBuf<AscendC::TPosition::VECIN> betaTmpBuf;
    AscendC::TBuf<AscendC::TPosition::VECIN> fp32Gamma;
    AscendC::TBuf<AscendC::TPosition::VECIN> fp32Beta;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueb; // 1 buffer

    AscendC::TBuf<AscendC::QuePosition::VECCALC> ubShareForQuant;
    AscendC::TBuf<AscendC::TPosition::A1> inBufAL1;
    AscendC::TBuf<AscendC::TPosition::VECIN> quantScaleTmpBuf;
    AscendC::TBuf<AscendC::TPosition::VECIN> quantOffsetTmpBuf;

    uint32_t bLength;
    uint32_t sLength;
    uint32_t hLength;
    uint32_t lnBaseM;
    uint32_t lnBaseK;
    uint32_t lnBaseM16Align;
    uint32_t lnBufferK;
    uint32_t mmBaseN;
    uint32_t windowSize;
    uint32_t batchNum;
    uint32_t patchHeight;
    uint32_t heightWindow;
    uint32_t patchWeight;
    uint32_t sizePerHead;
    uint32_t headNum;
    float epsilon;
    float avgFactor = 1.0f;
    uint32_t blockNum;
    uint32_t curBlockIdx;
    uint32_t winOffsetNumPerCore;
    uint32_t mmSizeM;
    uint32_t mmSizeN;
    uint32_t mmSizeK;
    uint32_t loopNum;
    uint32_t subMLoop;
    int64_t lnLoopSingleCore;
    int32_t addRoll;
    uint32_t nSplitPart;
    uint32_t tailM;

    int64_t actualLnBaseM;
    int64_t actualMmBaseM;

    uint32_t lnBsSize;
    bool splitNFlag;
    using aType = matmul::MatmulType<AscendC::TPosition::TSCM, CubeFormat::NZ, bDType>;
    using bType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, bDType, bTrans>;
    using cType = matmul::MatmulType<AscendC::TPosition::VECCALC, CubeFormat::ND, cDType>;
    using biasType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, int32_t>;
    matmul::Matmul<aType, bType, cType, biasType> mm;
    BinaryRepeatParams params = {1, 1, 1, 0, 0, 0};
    BinaryRepeatParams params1 = {1, 1, 1, 0, 0, 0};
    BinaryRepeatParams params2 = {1, 1, 1, 0, 0, 0};
    TransDataTo5HDParams transParams = {0, 0, 1, 0, 0};
    TransDataTo5HDParams transParams1 = {0, 0, 1, 0, 0};
};

extern "C" __global__ __aicore__ void swin_transformer_ln_qkv_quant(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta,
                                                                    GM_ADDR weight, GM_ADDR bias, GM_ADDR quant_scale,
                                                                    GM_ADDR quant_offset, GM_ADDR dequant_scale,
                                                                    GM_ADDR query_output, GM_ADDR key_output,
                                                                    GM_ADDR value_output, GM_ADDR workspace,
                                                                    GM_ADDR tiling)
{
    AscendC::SetSysWorkspace(workspace);
    GET_TILING_DATA(tiling_data, tiling);
    __gm__ uint8_t* userWorkspace = AscendC::GetUserWorkspace(workspace);
    TPipe tpipe;
    if (TILING_KEY_IS(100000UL)) {
        SwinTransformerLnQkvQuantNormalKernel<half, int8_t, half, false, true> op;
        op.Init(x, gamma, beta, weight, bias, quant_scale, quant_offset, dequant_scale, query_output, key_output,
                value_output, &tiling_data, userWorkspace, &tpipe);
        op.Process(&tiling_data);
    } else if (TILING_KEY_IS(000000UL)) {
        SwinTransformerLnQkvQuantNormalKernel<half, int8_t, half, false, false> op;
        op.Init(x, gamma, beta, weight, bias, quant_scale, quant_offset, dequant_scale, query_output, key_output,
                value_output, &tiling_data, userWorkspace, &tpipe);
        op.Process(&tiling_data);
    }
}
