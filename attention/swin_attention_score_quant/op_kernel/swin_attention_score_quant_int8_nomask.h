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
 * \file swin_attention_score_quant_int8_nomask.h
 * \brief
 */
#ifndef SWIN_ATTENTION_SCORE_QUANT_INT8_NOMASK_H
#define SWIN_ATTENTION_SCORE_QUANT_INT8_NOMASK_H

#include "swin_attention_score_quant.h"

template <typename DTYPE_OUT, typename DTYPE_SCALE, typename DTYPE_BIAS, typename DTYPE_MASK>
class SwinAttentionScoreQuant<int8_t, DTYPE_OUT, DTYPE_SCALE, DTYPE_BIAS, DTYPE_MASK, false> {
    using DTYPE_IN = int8_t;

public:
    __aicore__ inline SwinAttentionScoreQuant(){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                __gm__ uint8_t *scaleQuant, __gm__ uint8_t *scaleDequant1,
                                __gm__ uint8_t *scaleDequant2, __gm__ uint8_t *biasQuant, __gm__ uint8_t *biasDequant1,
                                __gm__ uint8_t *biasDequant2, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
                                const SwinAttentionScoreQuantTilingData *tiling, AscendC::TPipe *tPipe)
    {
        coreIdx = AscendC::GetBlockIdx();
        blockDim = AscendC::GetBlockNum();
        queryGm.SetGlobalBuffer((__gm__ DTYPE_IN *)query);
        keyGm.SetGlobalBuffer((__gm__ DTYPE_IN *)key);
        valueGm.SetGlobalBuffer((__gm__ DTYPE_IN *)value);
        scaleQuantGm.SetGlobalBuffer((__gm__ DTYPE_MASK *)scaleQuant);
        scaleDequant1Gm.SetGlobalBuffer((__gm__ DTYPE_SCALE *)scaleDequant1);
        scaleDequant2Gm.SetGlobalBuffer((__gm__ DTYPE_SCALE *)scaleDequant2);
        biasQuantGm.SetGlobalBuffer((__gm__ DTYPE_MASK *)biasQuant);
        biasDequant1Gm.SetGlobalBuffer((__gm__ DTYPE_BIAS *)biasDequant1);
        biasDequant2Gm.SetGlobalBuffer((__gm__ DTYPE_BIAS *)biasDequant2);
        attentionOutGm.SetGlobalBuffer((__gm__ DTYPE_OUT *)attentionOut);

        tilingData = const_cast<SwinAttentionScoreQuantTilingData *>(tiling);
        pipe = tPipe;

        coreLoops = tilingData->coreLoops;
        b = tilingData->dimB;
        n = tilingData->dimN;
        s = tilingData->dimS;
        h = tilingData->dimH;
        round16S = RoundUp(s, BLOCK_SIZE_16);
        round32S = RoundUp(s, BLOCK_SIZE_32);
        singleCoreM = tilingData->qkBmmTilingData.singleCoreM;
        numPerS = DivUp(s, singleCoreM);
        singleMTail = s - singleCoreM * (numPerS - 1);
        qSize = tilingData->qSize;
        kSize = tilingData->kSize;
        pSize = tilingData->pSize;
        vSize = tilingData->vSize;

        pipe->InitBuffer(softMaxTBuf, singleCoreM * BLOCK_SIZE_32);
        pipe->InitBuffer(softSumTBuf, singleCoreM * BLOCK_SIZE_32);
        pipe->InitBuffer(cubeSharedTBuf, tilingData->cubeSharedUbSize);
        pipe->InitBuffer(vecSharedTBuf, tilingData->vecSharedUbSize);
        pipe->InitBuffer(scaleQuantTBuf, round16S * sizeof(DTYPE_MASK));
        pipe->InitBuffer(biasQuantTBuf, round16S * sizeof(DTYPE_MASK));
        pipe->InitBuffer(pUbTBuf, singleCoreM * round16S * sizeof(DTYPE_MASK));
        pipe->InitBuffer(qL1TQue, SWIN_BUFFER_NUM, qSize * sizeof(DTYPE_IN));
        pipe->InitBuffer(kL1TQue, SWIN_BUFFER_NUM, kSize * sizeof(DTYPE_IN));
        pipe->InitBuffer(pL1TQue, SWIN_BUFFER_NUM, pSize * sizeof(DTYPE_IN));
        pipe->InitBuffer(vL1TQue, SWIN_BUFFER_NUM, vSize * sizeof(DTYPE_IN));
    }
    __aicore__ inline void Process()
    {
        AscendC::LocalTensor<uint8_t> qkBmmFormatUb = cubeSharedTBuf.template Get<uint8_t>();
        AscendC::LocalTensor<uint8_t> pvBmmFormatUb = cubeSharedTBuf.template Get<uint8_t>();
        AscendC::LocalTensor<DTYPE_MASK> scaleQuantUb = scaleQuantTBuf.template Get<DTYPE_MASK>();
        AscendC::LocalTensor<DTYPE_MASK> biasQuantUb = biasQuantTBuf.template Get<DTYPE_MASK>();
        qkBmm.SetLocalWorkspace(qkBmmFormatUb);
        pvBmm.SetLocalWorkspace(pvBmmFormatUb);

        AscendC::DataCopy(scaleQuantUb, scaleQuantGm, round16S);
        AscendC::DataCopy(biasQuantUb, biasQuantGm, round16S);

        qkBmm.SetBias(biasDequant1Gm);
        qkBmm.SetQuantVector(scaleDequant1Gm);
        pvBmm.SetBias(biasDequant2Gm);
        pvBmm.SetQuantVector(scaleDequant2Gm);

        for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += blockDim) {
            AscendC::LocalTensor<DTYPE_IN> qL1Tensor = qL1TQue.template AllocTensor<DTYPE_IN>();
            AscendC::LocalTensor<DTYPE_IN> kL1Tensor = kL1TQue.template AllocTensor<DTYPE_IN>();
            AscendC::LocalTensor<DTYPE_IN> pL1Tensor = pL1TQue.template AllocTensor<DTYPE_IN>();
            AscendC::LocalTensor<DTYPE_IN> vL1Tensor = vL1TQue.template AllocTensor<DTYPE_IN>();
            uint32_t bIdx = loopIdx / numPerS;
            uint32_t sIdx = loopIdx % numPerS;
            uint32_t bOffset = bIdx * s * h;
            actualM = (sIdx == numPerS - 1) ? singleMTail : singleCoreM;
            roundM = RoundUp(actualM, BLOCK_NUM_PER_FRACTAL);
            Stage1BmmQK(bIdx, sIdx, bOffset, qL1Tensor, kL1Tensor);
            qL1TQue.FreeTensor(qL1Tensor);
            kL1TQue.FreeTensor(kL1Tensor);
            Stage2Vec(bIdx, sIdx, bOffset, pL1Tensor, vL1Tensor);
            Stage3BmmPV(bIdx, sIdx, bOffset, pL1Tensor, vL1Tensor);
            pL1TQue.FreeTensor(pL1Tensor);
            vL1TQue.FreeTensor(vL1Tensor);
        }
    }
    // define qkBmm
    using qType = matmul::MatmulType<AscendC::TPosition::TSCM, CubeFormat::NZ, DTYPE_IN, false>;
    using kType = matmul::MatmulType<AscendC::TPosition::TSCM, CubeFormat::NZ, DTYPE_IN, true>;
    using qkType = matmul::MatmulType<AscendC::TPosition::VECCALC, CubeFormat::NZ, DTYPE_MASK>;
    using qkBiasType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS>;
    matmul::Matmul<qType, kType, qkType, qkBiasType> qkBmm;
    // define pvBmm
    using pType = matmul::MatmulType<AscendC::TPosition::TSCM, CubeFormat::NZ, DTYPE_IN, false>;
    using vType = matmul::MatmulType<AscendC::TPosition::TSCM, CubeFormat::NZ, DTYPE_IN, true>;
    using pvType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_OUT>;
    using pvBiasType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS>;
    matmul::Matmul<pType, vType, pvType, pvBiasType, CFG_ENVECND2NZ> pvBmm;

    const uint32_t C0_SIZE = BLOCK_SIZE_32 / sizeof(DTYPE_IN);

protected:
    __aicore__ inline void Stage1BmmQK(uint32_t bIdx, uint32_t sIdx, uint32_t bOffset,
                                       AscendC::LocalTensor<DTYPE_IN> &qL1Tensor,
                                       AscendC::LocalTensor<DTYPE_IN> &kL1Tensor)
    {
        AscendC::LocalTensor<DTYPE_MASK> pUbTensor = pUbTBuf.template Get<DTYPE_MASK>();

        CopyND2NZOnTheFly(qL1Tensor, queryGm[bOffset + sIdx * singleCoreM * h], 0, 0, actualM, h, h);
        CopyND2NZOnTheFly(kL1Tensor, keyGm[bOffset], 0, 0, s, h, h);

        qkBmm.SetTail(actualM, s, h);
        qkBmm.SetOrgShape(roundM, round32S, h);
        qkBmm.SetTensorA(qL1Tensor);
        qkBmm.SetTensorB(kL1Tensor, true);
        qkBmm.IterateAll(pUbTensor);
        qkBmm.End();
    }

    __aicore__ inline void Stage2AscendQuantNz(AscendC::LocalTensor<DTYPE_IN> &dst,
                                               AscendC::LocalTensor<DTYPE_MASK> &src,
                                               AscendC::LocalTensor<DTYPE_MASK> &scaleQuantUb,
                                               AscendC::LocalTensor<DTYPE_MASK> &biasQuantUb)
    {
        uint64_t mask0 = 128;
        uint64_t mask1[2] = {0xffffffff, 0};
        bool halfBlockIn = true;
        for (uint32_t j = 0; j < round16S / BLOCK_SIZE_16; j++) {
            Mul(src[j * roundM * BLOCK_SIZE_16], src[j * roundM * BLOCK_SIZE_16], scaleQuantUb[j * BLOCK_SIZE_16],
                mask0, roundM / BLOCK_NUM_PER_VEC, {1, 1, 0, BLOCK_NUM_PER_VEC, BLOCK_NUM_PER_VEC, 0});
            Add(src[j * roundM * BLOCK_SIZE_16], src[j * roundM * BLOCK_SIZE_16], biasQuantUb[j * BLOCK_SIZE_16], mask0,
                roundM / BLOCK_NUM_PER_VEC, {1, 1, 0, BLOCK_NUM_PER_VEC, BLOCK_NUM_PER_VEC, 0});
        }
        for (uint32_t j = 0; j < round16S / BLOCK_SIZE_16; j += 2) { // 将2个BLOCK的fp16数据转成1个BLOCK的int8数据
            Cast(dst[j / 2 * roundM * BLOCK_SIZE_32], src[j * roundM * BLOCK_SIZE_16], AscendC::RoundMode::CAST_NONE,
                 mask1, roundM, {1, (uint16_t)roundM, 1, 1}); // 输出每次循环偏移2列
        }
    }

    __aicore__ inline void Stage2Vec(uint32_t bIdx, uint32_t sIdx, uint32_t bOffset,
                                     AscendC::LocalTensor<DTYPE_IN> &pL1Tensor,
                                     AscendC::LocalTensor<DTYPE_IN> &vL1Tensor)
    {
        AscendC::LocalTensor<DTYPE_MASK> pUbTensor = pUbTBuf.template Get<DTYPE_MASK>();
        AscendC::LocalTensor<DTYPE_IN> pInt8 = pUbTBuf.template Get<DTYPE_IN>();
        AscendC::LocalTensor<DTYPE_MASK> softmaxMaxUb = softMaxTBuf.template Get<DTYPE_MASK>();
        AscendC::LocalTensor<DTYPE_MASK> softmaxSumUb = softSumTBuf.template Get<DTYPE_MASK>();
        AscendC::LocalTensor<DTYPE_MASK> scaleQuantUb = scaleQuantTBuf.template Get<DTYPE_MASK>();
        AscendC::LocalTensor<DTYPE_MASK> biasQuantUb = biasQuantTBuf.template Get<DTYPE_MASK>();
        AscendC::LocalTensor<uint8_t> softmaxSharedUb = vecSharedTBuf.template Get<uint8_t>();
        AscendC::LocalTensor<DTYPE_IN> rightMatrix = cubeSharedTBuf.template Get<DTYPE_IN>();
        AscendC::LocalTensor<DTYPE_IN> trans = cubeSharedTBuf.template Get<DTYPE_IN>()[vSize];

        AscendC::DataCopy(rightMatrix, valueGm[bOffset], s * h);

        AscendC::SoftMax<DTYPE_MASK, true, false, true>(pUbTensor, softmaxSumUb, softmaxMaxUb, pUbTensor,
                                                        softmaxSharedUb, tilingData->softmaxTilingData,
                                                        AscendC::SoftMaxShapeInfo{roundM, round16S, actualM, s});
        Stage2AscendQuantNz(pInt8, pUbTensor, scaleQuantUb, biasQuantUb);

        event_t pEventIDVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(pEventIDVToMte3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(pEventIDVToMte3);
        CopyNZ2NZ(pL1Tensor, pInt8, 0, 0, roundM, round32S, roundM);

        event_t eventIDMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIDMte2ToV);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIDMte2ToV);
        TransDataBMatrix(trans, rightMatrix, s, h);

        event_t vEventIDVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vEventIDVToMte3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vEventIDVToMte3);
        CopyNZ2NZ(vL1Tensor, trans, 0, 0, h, round32S, h);

        event_t eventIDMte3ToMte1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE1));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE1>(eventIDMte3ToMte1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE1>(eventIDMte3ToMte1);

        event_t eventIDMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIDMte3ToMte2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIDMte3ToMte2);
    }

    __aicore__ inline void Stage3BmmPV(uint32_t bIdx, uint32_t sIdx, uint32_t bOffset,
                                       AscendC::LocalTensor<DTYPE_IN> &pL1Tensor,
                                       AscendC::LocalTensor<DTYPE_IN> &vL1Tensor)
    {
        pvBmm.SetTail(actualM, h, s);
        pvBmm.SetOrgShape(roundM, h, round32S, round32S);
        pvBmm.SetTensorA(pL1Tensor);
        pvBmm.SetTensorB(vL1Tensor, true);
        pvBmm.IterateAll(attentionOutGm[bOffset + sIdx * singleCoreM * h]);
        pvBmm.End();
    }

    SwinAttentionScoreQuantTilingData *tilingData;
    AscendC::TPipe *pipe;
    AscendC::TQue<AscendC::TPosition::A1, SWIN_BUFFER_NUM> qL1TQue;
    AscendC::TQue<AscendC::TPosition::A1, SWIN_BUFFER_NUM> kL1TQue;
    AscendC::TQue<AscendC::TPosition::B1, SWIN_BUFFER_NUM> pL1TQue;
    AscendC::TQue<AscendC::TPosition::B1, SWIN_BUFFER_NUM> vL1TQue;
    AscendC::TBuf<> pUbTBuf;
    AscendC::TBuf<> softMaxTBuf;
    AscendC::TBuf<> softSumTBuf;
    AscendC::TBuf<> cubeSharedTBuf;
    AscendC::TBuf<> vecSharedTBuf;
    AscendC::TBuf<> scaleQuantTBuf;
    AscendC::TBuf<> biasQuantTBuf;

    AscendC::GlobalTensor<DTYPE_IN> queryGm;
    AscendC::GlobalTensor<DTYPE_IN> keyGm;
    AscendC::GlobalTensor<DTYPE_IN> valueGm;
    AscendC::GlobalTensor<DTYPE_MASK> scaleQuantGm;
    AscendC::GlobalTensor<DTYPE_SCALE> scaleDequant1Gm;
    AscendC::GlobalTensor<DTYPE_SCALE> scaleDequant2Gm;
    AscendC::GlobalTensor<DTYPE_MASK> biasQuantGm;
    AscendC::GlobalTensor<DTYPE_BIAS> biasDequant1Gm;
    AscendC::GlobalTensor<DTYPE_BIAS> biasDequant2Gm;
    AscendC::GlobalTensor<DTYPE_OUT> attentionOutGm;

    uint32_t coreIdx{0};
    uint32_t blockDim{0};
    uint32_t coreLoops{0};
    uint32_t b{0};
    uint32_t n{0};
    uint32_t s{0};
    uint32_t h{0};
    uint32_t round16S{0};
    uint32_t round32S{0};
    uint32_t roundM{0};
    uint32_t singleCoreM{0};
    uint32_t actualM{0};
    uint32_t singleMTail{0};
    uint32_t numPerS{0};
    uint32_t qSize{0};
    uint32_t kSize{0};
    uint32_t pSize{0};
    uint32_t vSize{0};
};

#endif
