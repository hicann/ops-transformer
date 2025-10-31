/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file swin_transformer_ln_qkv.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
using namespace AscendC;
using namespace matmul;

#define DATA_COPY_UNIT 32U // Datacopy block (unit: 32 bytes)
#define S_SIZE_FOR_LN 64U     // 1 x n x 128 is 64 in a single calculation of ln.

static constexpr uint32_t BATCH_NUM = 8;
static constexpr uint32_t SEQ_LENGTH = 128;

template <typename dataType, bool isReuseSource = false> class TransformQKVKernel {
public:
    __aicore__ inline TransformQKVKernel() {}
    __aicore__ inline void SetBasePara(const SwinTransformerLnQKVTilingData *tiling, DataFormat dataFormat)
    {
        bLength = tiling->layernormTilingParams.bLength;
        sLength = tiling->layernormTilingParams.sLength;
        hLength = tiling->layernormTilingParams.hLength;
        hLengthFloat = 128.0; //  The default value is 128.0.
        epsilon = 0.00001;      //  The default value is 0.00001
        innerLoopNum = tiling->layernormTilingParams.innerLoopNum;
        workspaceLength = tiling->workspaceSize;
        loopSize = tiling->layernormTilingParams.loopSize;
        this->dataFormat = dataFormat;
    }
    __aicore__ inline void Init(GM_ADDR inputX_gm, GM_ADDR gamm_gm, GM_ADDR beta_gm, GM_ADDR weight_gm, GM_ADDR bias_gm,
        GM_ADDR query_output, GM_ADDR key_output, GM_ADDR value_output, const SwinTransformerLnQKVTilingData *tiling,
        __gm__ uint8_t *userWorkspace, DataFormat dataFormat)
    {
        SetBasePara(tiling, dataFormat);
        loopNum = (GetBlockIdx() < tiling->opBaseInfo.remainderBlockNum) ? (tiling->opBaseInfo.baseLoopNum + 1) :
                                                                           tiling->opBaseInfo.baseLoopNum;
        dataSizePerBlock = loopSize * loopNum * sizeof(dataType);
        uint32_t elementPerBlock = tiling->layernormTilingParams.elementPerBlock;
        useVectorNum = tiling->useVectorNum;
        bshLength = loopSize;
        bsLength = tiling->layernormTilingParams.bsLength;
        shLength = tiling->layernormTilingParams.shLength;
        constexpr uint32_t weightLength = 128 * 384;
        constexpr uint32_t biasLength = 384;

        uint32_t normalBlockElementOffset = tiling->layernormTilingParams.normalBlockElementOffset;

        uint32_t gmOffset;
        if (GetBlockIdx() < tiling->opBaseInfo.remainderBlockNum) {
            gmOffset = tiling->layernormTilingParams.remainderElementPerBlock * GetBlockIdx();
        } else {
            gmOffset =
                normalBlockElementOffset + (GetBlockIdx() - tiling->opBaseInfo.remainderBlockNum) * elementPerBlock;
        }
        uint32_t elementCurBlock = (GetBlockIdx() < tiling->opBaseInfo.remainderBlockNum) ?
            tiling->layernormTilingParams.remainderElementPerBlock :
            elementPerBlock;
        inputXGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(inputX_gm + gmOffset * 2),
            elementCurBlock); // datasize indicates the number of elements.
        gammGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(gamm_gm), hLength);
        betaGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(beta_gm), hLength);
        uint32_t max_loop = tiling->opBaseInfo.baseLoopNum + 1;
        userWorkspaceGlobal.SetGlobalBuffer(
            reinterpret_cast<__gm__ dataType *>(userWorkspace + GetBlockIdx() * 8 * 256 * 128 * max_loop * 2),
            8 * 256 * 128 * max_loop); // 256K *sizoef(half) *2

        gmGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(userWorkspace +
            GetBlockIdx() * 8 * 256 * 384 * max_loop * 2 + 200 * 1024 * 1024),
            8 * 256 * 384 * max_loop); // 256K *sizoef(half) *2
        weightGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(weight_gm), weightLength);
        biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(bias_gm), biasLength);
        qOutputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(query_output + gmOffset * 2),
            elementCurBlock);
        kOutputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(key_output + gmOffset * 2), elementCurBlock);
        vOutputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(value_output + gmOffset * 2),
            elementCurBlock);

        pipe.InitBuffer(inQueueX, 2, sizeof(dataType) * shLength / 2); // 64K 2 pingpang

        pipe.InitBuffer(outQueue, 2, sizeof(dataType) * shLength / 2); // 64K  2 pingpang

        pipe.InitBuffer(fp32GammaTmpBuf, BATCH_NUM * 128 * sizeof(float));
        pipe.InitBuffer(fp32BetaTmpBuf, BATCH_NUM * 128 * sizeof(float));
        pipe.InitBuffer(calcTmpBuf, 6 * 8 * 128 * sizeof(float));
        matmulTiling = tiling->mmTilingParams;
        avg_factor_ = avg_factor_ / 128;
    }
    __aicore__ inline void LaunchData()
    {
        LocalTensor<float> fp32_gamma = fp32GammaTmpBuf.Get<float>(BATCH_NUM * 128);
        LocalTensor<float> fp32_beta = fp32BetaTmpBuf.Get<float>(BATCH_NUM * 128);
        LocalTensor<dataType> half_tmp = calcTmpBuf.Get<half>(128 * 2);
        DataCopy(half_tmp[0], gammGlobal, SEQ_LENGTH);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        uint32_t castIdx = 0;
        while (castIdx < BATCH_NUM) {
            Cast(fp32_gamma[SEQ_LENGTH * castIdx], half_tmp[0], AscendC::RoundMode::CAST_NONE, SEQ_LENGTH);
            PipeBarrier<PIPE_V>();
            ++castIdx;
        }
        DataCopy(half_tmp[SEQ_LENGTH], betaGlobal, SEQ_LENGTH);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        castIdx = 0;
        while (castIdx < BATCH_NUM) {
            Cast(fp32_beta[SEQ_LENGTH * castIdx], half_tmp[SEQ_LENGTH], AscendC::RoundMode::CAST_NONE, SEQ_LENGTH);
            PipeBarrier<PIPE_V>();
            ++castIdx;
        }
    }
    __aicore__ inline void DataCopyTranspose(uint32_t srcOffset, GlobalTensor<dataType> outputGlobal, DataCopyParams intriParamsForQkv, DataCopyParams intriParamsForWeight)
    {
        constexpr uint32_t offset[8] = {0, 32, 64, 96, 8192, 8224, 8256, 8288};
        constexpr uint32_t offsetPerloop = 128 * 384;
        constexpr uint32_t offsetQGm = 128 * 128;
        constexpr uint32_t offsetTrans = 32 * 64;
        for (uint32_t copyIdx = 0; copyIdx < 16 * loopNum; copyIdx++) { // 16 datacopy buffer
            auto weigthB = inQueueX.AllocTensor<dataType>();
            DataCopy(weigthB, gmGlobal[copyIdx * offsetPerloop + srcOffset], intriParamsForWeight);
            inQueueX.EnQue(weigthB);
            LocalTensor<dataType> inputXLocal = inQueueX.DeQue<dataType>();
            LocalTensor<dataType> tensorQkvOutput2 = outQueue.AllocTensor<dataType>();
            for (uint32_t qkvCopyOutTransIdx = 0; qkvCopyOutTransIdx < 8; qkvCopyOutTransIdx++) { // 8 copyout times
                DataCopy(tensorQkvOutput2[qkvCopyOutTransIdx * offsetTrans], inputXLocal[offset[qkvCopyOutTransIdx]],
                    intriParamsForQkv);
            }
            outQueue.EnQue(tensorQkvOutput2);
            inQueueX.FreeTensor(weigthB);
            LocalTensor<dataType> outputXLocal = outQueue.DeQue<dataType>();
            DataCopy(outputGlobal[copyIdx * offsetQGm], outputXLocal, offsetQGm);
            outQueue.FreeTensor(outputXLocal);
        }
    }
    __aicore__ inline void Process()
    {
        constexpr uint32_t srcStrideParam = 4;
        constexpr uint32_t numPerLoop = 16;
        constexpr uint32_t qkvCopyOutTransNum = BATCH_NUM;
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &matmulTiling);
        if (GetBlockIdx() >= useVectorNum) {
            return;
        }
        DataCopyParams intriParamsForQkv;
        intriParamsForQkv.blockCount = 64;    // 64 block = 64*32
        intriParamsForQkv.blockLen = DATA_COPY_UNIT * sizeof(dataType) / DATA_COPY_UNIT;
        intriParamsForQkv.srcStride = (srcStrideParam - 1) * intriParamsForQkv.blockLen;
        intriParamsForQkv.dstStride = 0; //   [64,1,4,32] -> [4,64,32]

        DataCopyParams intriParamsForWeight;
        intriParamsForWeight.blockCount = SEQ_LENGTH;                                   // 128
        intriParamsForWeight.blockLen = SEQ_LENGTH * sizeof(dataType) / DATA_COPY_UNIT; // 128
        intriParamsForWeight.srcStride = 2 * intriParamsForWeight.blockLen;
        intriParamsForWeight.dstStride = 0;
        mm.SetTensorB(weightGlobal); // Set the right matrix B.
        mm.SetBias(biasGlobal);      // Setting the Bias
        constexpr uint32_t offsetMm = 8 * 256 * SEQ_LENGTH;
        constexpr uint32_t offsetMmRes = 8 * 256 * 384;
        LaunchData();

        //  loopNum  = [B], 32
        for (uint32_t innerLoopIdx = 0; innerLoopIdx < loopNum; innerLoopIdx++) { //   8
            for (uint32_t loopIdx = 0; loopIdx < numPerLoop; loopIdx++) {
                CopyIn(innerLoopIdx * numPerLoop + loopIdx);
                Compute(innerLoopIdx * numPerLoop + loopIdx); //           [8, 32,8,128]
                CopyOut(innerLoopIdx * numPerLoop + loopIdx); // After done [32, 8, 8, 128] on GM
            }
            mm.SetTensorA(userWorkspaceGlobal[innerLoopIdx * offsetMm]); // Setting Left Matrix A
            if (innerLoopIdx != (loopNum - 1)) {
                mm.template IterateAll<false>(gmGlobal[innerLoopIdx * offsetMmRes]);
            } else {
                mm.template IterateAll<true>(gmGlobal[innerLoopIdx * offsetMmRes]);
            }
            mm.End();
        }
        DataCopyTranspose(0, qOutputGlobal, intriParamsForQkv, intriParamsForWeight);
        DataCopyTranspose(SEQ_LENGTH, kOutputGlobal, intriParamsForQkv, intriParamsForWeight);
        DataCopyTranspose(SEQ_LENGTH * 2, vOutputGlobal, intriParamsForQkv, intriParamsForWeight); // 2 times seq
    } // Process

private:
    __aicore__ inline void CopyIn(uint32_t innerLoopIdx)
    {
        LocalTensor<dataType> inputXLocal = inQueueX.AllocTensor<dataType>();
        DataCopy(inputXLocal, inputXGlobal[(SEQ_LENGTH * SEQ_LENGTH) * innerLoopIdx], shLength / 2); // innerloop need div 2
        inQueueX.EnQue(inputXLocal);
    }
    __aicore__ inline void ComputeLayernorm(uint32_t subTaskElement, AscendC::LocalTensor<float> sum, \
                                            AscendC::LocalTensor<float> sqx, AscendC::LocalTensor<float> fp32x, \
                                            AscendC::LocalTensor<float> tmpSub, AscendC::LocalTensor<float> mean)
    {
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        for (uint32_t idx = 0 ;idx < BATCH_NUM ;++idx) {
            float delta = sum.GetValue(idx * 64) + sum.GetValue(idx * 64 + 32);   // transpose for 64 , 32
            float variance = delta * avg_factor_;
            sum.SetValue(idx, variance + epsilon);
        }
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Sqrt(sum, sum, BATCH_NUM);
        PipeBarrier<PIPE_V>();
        for (uint32_t idx = 0 ;idx < BATCH_NUM ;++idx) {
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            float factor = sum.GetValue(idx);

            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);
            PipeBarrier<PIPE_V>();
            Duplicate(sqx[idx * SEQ_LENGTH], factor, 64, 4, 1, 8);   // transpose for 64, 4, 8
            PipeBarrier<PIPE_V>();
        }

        Sub(tmpSub, fp32x, mean, subTaskElement);
        PipeBarrier<PIPE_V>();

        Div(fp32x, tmpSub, sqx, subTaskElement);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Compute(uint32_t innerLoopIdx)
    {
        LocalTensor<dataType> inputXLocal = inQueueX.DeQue<dataType>();

        LocalTensor<dataType> outputLocal = outQueue.AllocTensor<dataType>();
        LocalTensor<float> fp32_gamma = fp32GammaTmpBuf.Get<float>(BATCH_NUM * SEQ_LENGTH);
        LocalTensor<float> fp32_beta = fp32BetaTmpBuf.Get<float>(BATCH_NUM * SEQ_LENGTH);
        LocalTensor<float> buf = calcTmpBuf.Get<float>();
        uint32_t subtaskIdx = 0;
        constexpr uint32_t subTaskElement = 8 * SEQ_LENGTH;   // need do 8 compute in each
        AscendC::LocalTensor<float> fp32x = buf[0];
        AscendC::LocalTensor<float> sqx = buf[1 * subTaskElement];
        AscendC::LocalTensor<float> sum = buf[2  * subTaskElement];
        AscendC::LocalTensor<float> work = buf[3 * subTaskElement];
        AscendC::LocalTensor<float> mean = buf[4 * subTaskElement];
        AscendC::LocalTensor<float> tmpSub = buf[5 * subTaskElement];
        while (subtaskIdx < (SEQ_LENGTH / 8)) {  // need do 8 compute in each
            Cast(fp32x, inputXLocal[subtaskIdx * subTaskElement], AscendC::RoundMode::CAST_NONE, subTaskElement);
            PipeBarrier<PIPE_V>();
            BlockReduceSum<float>(work, fp32x, 16, 64, 8, 1, 8);  // 16 element in one block, 64 num, 8 block
            PipeBarrier<PIPE_V>();
            BlockReduceSum<float>(sum, work, 16, 8, 4, 1, 8);    // 16 element in one block, 8 num, 4 block
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            for (uint32_t idx = 0 ;idx < BATCH_NUM ;++idx) {
                float hSum = sum.GetValue(idx * 64) + sum.GetValue(idx * 64 +32);
                float meanScalar = hSum * avg_factor_;
                SetFlag<HardEvent::S_V>(EVENT_ID0);
                WaitFlag<HardEvent::S_V>(EVENT_ID0);
                PipeBarrier<PIPE_V>();

                Duplicate(mean[idx * SEQ_LENGTH], meanScalar, 64,4,1,8); // copy 64 element, 4 num, 8 block
                PipeBarrier<PIPE_V>();
            }

            Sub(tmpSub, fp32x, mean, subTaskElement);
            PipeBarrier<PIPE_V>();
            Mul(sqx, tmpSub, tmpSub, subTaskElement);
            PipeBarrier<PIPE_V>();

            BlockReduceSum<float>(work, sqx, 16, 64, 8, 1, 8);   // 16 element in one block, 64 num, 8 block
            PipeBarrier<PIPE_V>();
            BlockReduceSum<float>(sum, work, 16, 8, 4, 1, 8);    // 16 element in one block, 8 num, 4 block
            ComputeLayernorm(subTaskElement, sum, sqx, fp32x, tmpSub, mean);

            Mul(sqx, fp32x, fp32_gamma, subTaskElement);
            PipeBarrier<PIPE_V>();
            Add(fp32x, sqx , fp32_beta , subTaskElement);
            PipeBarrier<PIPE_V>();
            Cast(outputLocal[subtaskIdx * subTaskElement], fp32x, AscendC::RoundMode::CAST_RINT, subTaskElement);
            ++subtaskIdx;
        }

        outQueue.EnQue<dataType>(outputLocal);
        inQueueX.FreeTensor(inputXLocal);
    }
    __aicore__ inline void CopyOut(uint32_t innerLoopIdx)
    {
        DataCopyParams intriParams;
        constexpr uint32_t bufferNum = 2; // 256 * 128 devided into 2 128 * 128

        uint32_t blockLen = 8 * hLength; // 8 * 128
        intriParams.blockCount = 16; // 16 (8 x 128) small matrices
        intriParams.blockLen = blockLen * sizeof(dataType) / DATA_COPY_UNIT;
        intriParams.srcStride = 0;
        intriParams.dstStride = (innerLoopNum - 1) * intriParams.blockLen; // Size of the third dimension
        LocalTensor<dataType> outputLocal = outQueue.DeQue<dataType>();

        uint32_t blocIdx = innerLoopIdx / 16;        // 8 * 2 = 16
        uint32_t curBlockIdx = innerLoopIdx % 16;    // 8 * 2 = 16
        uint32_t offset = 0;
        offset = blockLen * (curBlockIdx / bufferNum) + (curBlockIdx % bufferNum) * 16 * 64 * SEQ_LENGTH + // 16 = 8 * 2, 64 is channel number,
                        blocIdx * 8 * 256 * SEQ_LENGTH; // 8 * 256 * seq is loopsize

        DataCopy(userWorkspaceGlobal[offset], outputLocal, intriParams);
        outQueue.FreeTensor(outputLocal);
    }

private:
    GlobalTensor<dataType> inputXGlobal;
    GlobalTensor<dataType> gammGlobal;
    GlobalTensor<dataType> betaGlobal;
    GlobalTensor<dataType> weightGlobal;
    GlobalTensor<dataType> biasGlobal;
    GlobalTensor<dataType> qOutputGlobal;
    GlobalTensor<dataType> kOutputGlobal;
    GlobalTensor<dataType> vOutputGlobal;
    GlobalTensor<dataType> userWorkspaceGlobal;
    GlobalTensor<dataType> gmGlobal;
    using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, dataType>;
    using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, dataType>;
    using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, dataType>;
    using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, dataType>;
    Matmul<aType, bType, cType, biasType> mm;

    TPipe pipe;
    TCubeTiling matmulTiling;
    TQue<QuePosition::VECIN, 2> inQueueX; // 2 buffer_num
    TQue<QuePosition::VECIN, 2> inQueueGamma; // 2 buffer_num
    TQue<QuePosition::VECIN, 2> inQueueBeta; // 2 buffer_num
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp32GammaTmpBuf;                // 32 * 128 * sizeof(fp32) = 16K
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp32BetaTmpBuf;                // 32 * 128 * sizeof(fp32) = 16k
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcTmpBuf;                   // 128 * 128 * sizeof(fp32) = 64k
    TQue<QuePosition::VECOUT, 2> outQueue; // 2 buffer_num
    TQue<QuePosition::VECOUT, 2> outQueueMean; // 2 buffer_num
    TQue<QuePosition::VECOUT, 2> outQueueVariance; // 2 buffer_num

    float avg_factor_{1.0}; // num_col_的倒数
    uint32_t bLength;
    uint32_t sLength;
    uint32_t hLength;
    float hLengthFloat;
    uint32_t loopNum = 1;
    uint32_t innerLoopNum = 1;
    uint32_t loopSize;
    uint32_t workspaceLength;
    float epsilon;
    DataFormat dataFormat;
    uint32_t useVectorNum;
    uint32_t bshLength;
    uint32_t bsLength;
    uint32_t shLength;
    uint32_t dataSizePerBlock;
    uint32_t outputQkvBaseLen;
    uint32_t stackBufferSize = 0;
};
#ifdef __CCE_KT_TEST__
    REGISTER_TILING_DEFAULT(SwinTransformerLnQKVTilingData);
#endif
extern "C" __global__ __aicore__ void swin_transformer_ln_qkv(GM_ADDR inputX, GM_ADDR gamma, GM_ADDR beta,
    GM_ADDR weight, GM_ADDR bias, GM_ADDR query_output, GM_ADDR key_output, GM_ADDR value_output, GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    __gm__ uint8_t *userWorkspace = GetUserWorkspace(workspace);

    DataFormat dataFormat = DataFormat::NZ;

    TransformQKVKernel<half> op;
    op.Init(inputX, gamma, beta, weight, bias, query_output, key_output, value_output, &tiling_data, userWorkspace,
        dataFormat);
    op.Process();
}