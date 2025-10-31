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
 * \file swin_attention_ffn.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;
using namespace matmul;

#define BUFFER_NUM 1
#define DATA_COPY_UNIT 32U

template<class X1_TYPE, class X2_TYPE, class X3_TYPE, class BIAS_TYPE, class Y_TYPE>
class KernelBatchMatmul {
public:
    __aicore__ inline KernelBatchMatmul() {};
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR x3, GM_ADDR y, GM_ADDR workspace,
        const SwinAttentionFFNTilingData* tiling, TPipe* tPipe);
    __aicore__ inline void Process();
    using x1Type = MatmulType<TPosition::GM, CubeFormat::ND, X1_TYPE>;
    using x2Type = MatmulType<TPosition::GM, CubeFormat::ND, X2_TYPE>;
    using x3Type = MatmulType<TPosition::VECCALC, CubeFormat::ND, X3_TYPE>;
    using biasType = MatmulType<TPosition::GM, CubeFormat::ND, BIAS_TYPE>;
    Matmul<x1Type, x2Type, x3Type, biasType> mm;

protected:
    const SwinAttentionFFNTilingData* tilingData;
    TPipe* pipe;
    TCubeTiling bmmTiling;
    GlobalTensor<X1_TYPE> aGlobal;
    GlobalTensor<X2_TYPE> bGlobal;
    GlobalTensor<X3_TYPE> cGlobal;
    GlobalTensor<BIAS_TYPE> biasGlobal;
    GlobalTensor<Y_TYPE> dGlobal;
    GlobalTensor<Y_TYPE> workspaceGm;
    LocalTensor<X3_TYPE> xLocal;
    LocalTensor<Y_TYPE> zLocal;
    LocalTensor<Y_TYPE> tmpLocal;
    TQue<QuePosition::VECIN, 1> inQueueTmp;
    TQue<QuePosition::VECIN, 1> inQueueX;
    TQue<QuePosition::VECOUT, 1> outQueueZ;

    uint32_t tmp_block_idx;

    uint32_t offsetA        = 0;
    uint32_t offsetBmm      = 0;
    uint32_t offsetC        = 0;
    uint32_t offsetD        = 0;

    uint32_t dataNumPerBatchA;
    uint32_t dataNumPerBatchD;
    uint32_t dataNumPerLoop;
    uint32_t aivNum;
    uint32_t shift1;
    uint32_t shift2;

    uint32_t tpBlockSize;
    uint32_t tpSpaceCnt;
    uint32_t tpSpaceH;
    uint32_t tpSpaceW;
    uint32_t blockInSpace;
    uint32_t tpSpaceSize;
    uint32_t tpSpaceWTransposed;
    uint32_t tpSpaceHTransposed;

    uint32_t bmmSpaceID, bmmBlockID, bmmRowID, bmmColID;
    uint32_t cSpaceID, cBlockID, cRowID, cColID;

    DataCopyParams inDCParams, outDCParams, inDCParams2, inDCParams3, inDCParams4, outDCParams2, outDCParams3, outDCParams4;

    __aicore__ inline void BmmCompute1(uint32_t coreIdx);
    __aicore__ inline void BmmCompute2(uint32_t coreIdx);
    __aicore__ inline void CopyIn1();
    __aicore__ inline void CopyIn2();
    __aicore__ inline void CopyIn3();
    __aicore__ inline void CopyIn4();
    __aicore__ inline void Compute();
    __aicore__ inline void CopyOut1();
    __aicore__ inline void CopyOut2();
    __aicore__ inline void CopyOut3();
    __aicore__ inline void CopyOut4();
};

template<class X1_TYPE, class X2_TYPE, class X3_TYPE, class BIAS_TYPE, class Y_TYPE>
__aicore__ inline void KernelBatchMatmul<X1_TYPE, X2_TYPE, X3_TYPE, BIAS_TYPE, Y_TYPE>::Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias,
    GM_ADDR x3, GM_ADDR y, GM_ADDR workspace, const SwinAttentionFFNTilingData* tiling, TPipe* tPipe)
{
    tmp_block_idx = GetBlockIdx();
    // init global buffer
    aGlobal.SetGlobalBuffer((__gm__ X1_TYPE*)x1);
    bGlobal.SetGlobalBuffer((__gm__ X2_TYPE*)x2);
    cGlobal.SetGlobalBuffer((__gm__ X3_TYPE*)x3);
    biasGlobal.SetGlobalBuffer((__gm__ BIAS_TYPE*)bias);
    dGlobal.SetGlobalBuffer((__gm__ Y_TYPE*)y);
    workspaceGm.SetGlobalBuffer((__gm__ Y_TYPE*)workspace);
    offsetA = 0;
    offsetD = 0;
    tilingData = tiling;
    pipe = tPipe;

    tpBlockSize = tilingData->tpBlockSize;
    tpSpaceCnt = tilingData->tpSpaceCnt;
    tpSpaceH = tilingData->tpSpaceH;
    tpSpaceW = tilingData->tpSpaceW;
    blockInSpace = tilingData->blockInSpace;
    tpSpaceSize = tilingData->tpSpaceSize;
    tpSpaceWTransposed = tilingData->tpSpaceWTransposed;
    tpSpaceHTransposed = tilingData->tpSpaceHTransposed;

    aivNum = tilingData->aivNum;
    dataNumPerBatchA = tilingData->dataNumPerBatchA;
    dataNumPerBatchD = tilingData->dataNumPerBatchD;
    dataNumPerLoop = tilingData->dataNumPerLoop;
    bmmTiling = tilingData->bmmTilingData;
    shift1 = tilingData->shift1;
    shift2 = tilingData->shift2;
    
    mm.Init(&bmmTiling);

    pipe->InitBuffer(inQueueTmp, BUFFER_NUM, dataNumPerLoop * sizeof(X3_TYPE));
    pipe->InitBuffer(inQueueX, BUFFER_NUM, dataNumPerLoop * sizeof(X3_TYPE));
    pipe->InitBuffer(outQueueZ, BUFFER_NUM, dataNumPerLoop * sizeof(Y_TYPE));

    inDCParams.blockCount = 8;
    inDCParams.blockLen = tpBlockSize * sizeof(X3_TYPE) / DATA_COPY_UNIT;
    inDCParams.srcStride = (tpSpaceH - 1) * inDCParams.blockLen;
    inDCParams.dstStride = 0;

    outDCParams.blockCount = 8;
    outDCParams.blockLen = tpBlockSize * sizeof(X3_TYPE) / DATA_COPY_UNIT;
    outDCParams.srcStride = 0;
    outDCParams.dstStride = (tpSpaceH - 1) * outDCParams.blockLen;

    if (shift1 == 0 && shift2 == 0)
    {
        return;
    }

    inDCParams2.blockCount = 4;
    inDCParams2.blockLen = tpBlockSize * sizeof(X3_TYPE) / DATA_COPY_UNIT;
    inDCParams2.srcStride = (tpSpaceH - 1) * inDCParams2.blockLen;
    inDCParams2.dstStride = 0;

    inDCParams3.blockCount = 8;
    inDCParams3.blockLen = tpBlockSize * sizeof(X3_TYPE) / 2 / DATA_COPY_UNIT;
    inDCParams3.srcStride = (tpSpaceH * 2 - 1) * inDCParams3.blockLen;
    inDCParams3.dstStride = 1 * inDCParams3.blockLen;
    
    inDCParams4.blockCount = 4;
    inDCParams4.blockLen = tpBlockSize * sizeof(X3_TYPE) / 2 / DATA_COPY_UNIT;
    inDCParams4.srcStride = (tpSpaceH * 2 - 1) * inDCParams4.blockLen;
    inDCParams4.dstStride = 1 * inDCParams4.blockLen;

    outDCParams2.blockCount = 4;
    outDCParams2.blockLen = tpBlockSize * sizeof(X3_TYPE) / DATA_COPY_UNIT;
    outDCParams2.srcStride = 0;
    outDCParams2.dstStride = (tpSpaceH - 1) * outDCParams2.blockLen;

    outDCParams3.blockCount = 8;
    outDCParams3.blockLen = tpBlockSize * sizeof(X3_TYPE) / 2 / DATA_COPY_UNIT;
    outDCParams3.srcStride = 1 * outDCParams3.blockLen;
    outDCParams3.dstStride = (tpSpaceH * 2 - 1) * outDCParams3.blockLen;

    outDCParams4.blockCount = 4;
    outDCParams4.blockLen = tpBlockSize * sizeof(X3_TYPE) / 2 / DATA_COPY_UNIT;
    outDCParams4.srcStride = 1 * outDCParams4.blockLen;
    outDCParams4.dstStride = (tpSpaceH * 2 - 1) * outDCParams4.blockLen;
}

template<class X1_TYPE, class X2_TYPE, class X3_TYPE, class BIAS_TYPE, class Y_TYPE>
__aicore__ inline void KernelBatchMatmul<X1_TYPE, X2_TYPE, X3_TYPE, BIAS_TYPE, Y_TYPE>::Process()
{
    if (shift1 == 0 && shift2 == 0)
    {
        BmmCompute1(tmp_block_idx);
    }
    else
    {
        BmmCompute2(tmp_block_idx);
    }
}

template<class X1_TYPE, class X2_TYPE, class X3_TYPE, class BIAS_TYPE, class Y_TYPE>
__aicore__ inline void KernelBatchMatmul<X1_TYPE, X2_TYPE, X3_TYPE, BIAS_TYPE, Y_TYPE>::BmmCompute1(uint32_t coreIdx)
{
    if (coreIdx >= aivNum)
    {
        return;
    }
    int loopNum;
    if (coreIdx < tilingData->bmmFormerNum) {
        loopNum = tilingData->bmmFormerBatchNum;
    } else {
        loopNum = tilingData->bmmTailBatchNum;
    }

    mm.SetWorkspace((__gm__ uint8_t*)workspaceGm[coreIdx * dataNumPerBatchD].GetPhyAddr(), dataNumPerBatchD * sizeof(Y_TYPE));
    for (int i = 0; i < loopNum; i++) {
        offsetA = (aivNum * i + coreIdx) * dataNumPerBatchA;
        offsetBmm = (aivNum * i + coreIdx) * dataNumPerBatchD;
        mm.SetTensorA(aGlobal[offsetA]);
        mm.SetTensorB(bGlobal);
        mm.SetBias(biasGlobal);

        mm.template Iterate<false>();
        for (int i = 0; i < bmmTiling.singleCoreM / bmmTiling.baseM; i++) {
            cSpaceID = offsetBmm / tpSpaceSize;
            cBlockID = (offsetBmm - cSpaceID * tpSpaceSize) / tpBlockSize;
            cRowID = cBlockID % tpSpaceW;
            cColID = cBlockID / tpSpaceW;
            cBlockID = cRowID * tpSpaceWTransposed + cColID;
            offsetD = offsetC = cSpaceID * tpSpaceSize + cBlockID * tpBlockSize;

            CopyIn1();
            Compute();
            CopyOut1();
            offsetBmm += dataNumPerLoop;
        }
    }
}

template<class X1_TYPE, class X2_TYPE, class X3_TYPE, class BIAS_TYPE, class Y_TYPE>
__aicore__ inline void KernelBatchMatmul<X1_TYPE, X2_TYPE, X3_TYPE, BIAS_TYPE, Y_TYPE>::BmmCompute2(uint32_t coreIdx)
{
    if (coreIdx >= aivNum)
    {
        return;
    }
    int loopNum;
    if (coreIdx < tilingData->bmmFormerNum) {
        loopNum = tilingData->bmmFormerBatchNum;
    } else {
        loopNum = tilingData->bmmTailBatchNum;
    }

    // 使用 MatMul 异步 Iterate 接口时必须使用
    mm.SetWorkspace((__gm__ uint8_t*)workspaceGm[coreIdx * dataNumPerBatchD].GetPhyAddr(), dataNumPerBatchD * sizeof(Y_TYPE));

    for (int i = 0; i < loopNum; i++) {
        offsetA = (aivNum * i + coreIdx) * dataNumPerBatchA;
        offsetBmm = (aivNum * i + coreIdx) * dataNumPerBatchD;
        mm.SetTensorA(aGlobal[offsetA]);
        mm.SetTensorB(bGlobal);
        mm.SetBias(biasGlobal);

        mm.template Iterate<false>();
        for (int i = 0; i < bmmTiling.singleCoreM / bmmTiling.baseM; i++) {
            cSpaceID = offsetBmm / tpSpaceSize;
            cBlockID = (offsetBmm - cSpaceID * tpSpaceSize) / tpBlockSize;
            cRowID = cBlockID % tpSpaceW + 4;
            cColID = cBlockID / tpSpaceW;
            cBlockID = cRowID * tpSpaceWTransposed + cColID;
            offsetD = offsetC = cSpaceID * tpSpaceSize + cBlockID * tpBlockSize + tpBlockSize / 2;

            if (likely(cSpaceID != tpSpaceCnt - 1)) {
                if (likely(cColID != tpSpaceWTransposed - 2)) {
                    CopyIn1();
                    Compute();
                    CopyOut1();
                }
                else
                {
                    CopyIn2();
                    Compute();
                    CopyOut2();
                }
            }
            else
            {
                if (likely(cColID != tpSpaceWTransposed - 2)) {
                    CopyIn3();
                    Compute();
                    CopyOut3();
                }
                else
                {
                    CopyIn4();
                    Compute();
                    CopyOut4();
                }
            }
            offsetBmm += dataNumPerLoop;
        }
    }
}

template<class X1_TYPE, class X2_TYPE, class X3_TYPE, class BIAS_TYPE, class Y_TYPE>
__aicore__ inline void KernelBatchMatmul<X1_TYPE, X2_TYPE, X3_TYPE, BIAS_TYPE, Y_TYPE>::CopyIn1()
{
    xLocal = inQueueX.AllocTensor<X3_TYPE>();
    DataCopy(xLocal, cGlobal[offsetC], inDCParams);
    DataCopy(xLocal[dataNumPerLoop >> 1], cGlobal[offsetC + tpBlockSize], inDCParams);
    inQueueX.EnQue(xLocal);
}

template<class X1_TYPE, class X2_TYPE, class X3_TYPE, class BIAS_TYPE, class Y_TYPE>
__aicore__ inline void KernelBatchMatmul<X1_TYPE, X2_TYPE, X3_TYPE, BIAS_TYPE, Y_TYPE>::CopyIn2()
{
    xLocal = inQueueX.AllocTensor<X3_TYPE>();
    DataCopy(xLocal, cGlobal[offsetC], inDCParams);
    DataCopy(xLocal[dataNumPerLoop / 2], cGlobal[offsetC + tpBlockSize], inDCParams3);
    offsetC = cSpaceID * tpSpaceSize + cRowID * tpSpaceWTransposed * tpBlockSize;
    DataCopy(xLocal[dataNumPerLoop / 2 + tpBlockSize / 2], cGlobal[offsetC], inDCParams3);
    inQueueX.EnQue(xLocal);
}

template<class X1_TYPE, class X2_TYPE, class X3_TYPE, class BIAS_TYPE, class Y_TYPE>
__aicore__ inline void KernelBatchMatmul<X1_TYPE, X2_TYPE, X3_TYPE, BIAS_TYPE, Y_TYPE>::CopyIn3()
{
    xLocal = inQueueX.AllocTensor<X3_TYPE>();
    DataCopy(xLocal, cGlobal[offsetC], inDCParams2);
    DataCopy(xLocal[dataNumPerLoop / 2], cGlobal[offsetC + tpBlockSize], inDCParams2);

    offsetC = cColID * tpBlockSize + tpBlockSize / 2;  // 重新计算 offsetC 的值, cSpaceID 为 0, cRowID 为 0
    DataCopy(xLocal[dataNumPerLoop / 4], cGlobal[offsetC], inDCParams2);
    DataCopy(xLocal[dataNumPerLoop * 3 / 4], cGlobal[offsetC + tpBlockSize], inDCParams2);
    inQueueX.EnQue(xLocal);
}

template<class X1_TYPE, class X2_TYPE, class X3_TYPE, class BIAS_TYPE, class Y_TYPE>
__aicore__ inline void KernelBatchMatmul<X1_TYPE, X2_TYPE, X3_TYPE, BIAS_TYPE, Y_TYPE>::CopyIn4()
{
    xLocal = inQueueX.AllocTensor<X3_TYPE>();
    DataCopy(xLocal, cGlobal[offsetC], inDCParams2);
    DataCopy(xLocal[dataNumPerLoop / 2], cGlobal[offsetC + tpBlockSize], inDCParams4);
    offsetC = cSpaceID * tpSpaceSize + cRowID * tpSpaceWTransposed * tpBlockSize;
    DataCopy(xLocal[dataNumPerLoop / 2 + tpBlockSize / 2], cGlobal[offsetC], inDCParams4);

    offsetC = cColID * tpBlockSize + tpBlockSize / 2;
    DataCopy(xLocal[dataNumPerLoop / 4], cGlobal[offsetC], inDCParams2);
    DataCopy(xLocal[dataNumPerLoop * 3 / 4], cGlobal[offsetC + tpBlockSize], inDCParams4);

    offsetC = 0;
    DataCopy(xLocal[dataNumPerLoop * 3 / 4 + tpBlockSize / 2], cGlobal[offsetC], inDCParams4);
    inQueueX.EnQue(xLocal);
}

template<class X1_TYPE, class X2_TYPE, class X3_TYPE, class BIAS_TYPE, class Y_TYPE>
__aicore__ inline void KernelBatchMatmul<X1_TYPE, X2_TYPE, X3_TYPE, BIAS_TYPE, Y_TYPE>::Compute()
{
    xLocal = inQueueX.DeQue<X3_TYPE>();
    tmpLocal = inQueueTmp.AllocTensor<Y_TYPE>();
    zLocal = outQueueZ.AllocTensor<Y_TYPE>();
    mm.template GetTensorC<false>(tmpLocal, false, true);
    mm.End();

    Add(zLocal, xLocal, tmpLocal, dataNumPerLoop);
    outQueueZ.EnQue<Y_TYPE>(zLocal);
    inQueueX.FreeTensor(xLocal);
    inQueueTmp.FreeTensor(tmpLocal);
}

template<class X1_TYPE, class X2_TYPE, class X3_TYPE, class BIAS_TYPE, class Y_TYPE>
__aicore__ inline void KernelBatchMatmul<X1_TYPE, X2_TYPE, X3_TYPE, BIAS_TYPE, Y_TYPE>::CopyOut1()
{
    zLocal = outQueueZ.DeQue<Y_TYPE>();
    DataCopy(dGlobal[offsetD], zLocal, outDCParams);
    DataCopy(dGlobal[offsetD + tpBlockSize], zLocal[dataNumPerLoop >> 1], outDCParams);
    outQueueZ.FreeTensor(zLocal);
}

template<class X1_TYPE, class X2_TYPE, class X3_TYPE, class BIAS_TYPE, class Y_TYPE>
__aicore__ inline void KernelBatchMatmul<X1_TYPE, X2_TYPE, X3_TYPE, BIAS_TYPE, Y_TYPE>::CopyOut2()
{
    zLocal = outQueueZ.DeQue<Y_TYPE>();
    DataCopy(dGlobal[offsetD], zLocal, outDCParams);
    DataCopy(dGlobal[offsetD + tpBlockSize], zLocal[dataNumPerLoop / 2], outDCParams3);

    offsetD = cSpaceID * tpSpaceSize + cRowID * tpSpaceWTransposed * tpBlockSize;
    DataCopy(dGlobal[offsetD], zLocal[dataNumPerLoop / 2 + tpBlockSize / 2], outDCParams3);
    outQueueZ.FreeTensor(zLocal);
}

template<class X1_TYPE, class X2_TYPE, class X3_TYPE, class BIAS_TYPE, class Y_TYPE>
__aicore__ inline void KernelBatchMatmul<X1_TYPE, X2_TYPE, X3_TYPE, BIAS_TYPE, Y_TYPE>::CopyOut3()
{
    zLocal = outQueueZ.DeQue<Y_TYPE>();
    DataCopy(dGlobal[offsetD], zLocal, outDCParams2);
    DataCopy(dGlobal[offsetD + tpBlockSize], zLocal[dataNumPerLoop / 2], outDCParams2);

    offsetD = cColID * tpBlockSize + tpBlockSize / 2;
    DataCopy(dGlobal[offsetD], zLocal[dataNumPerLoop / 4], outDCParams2);
    DataCopy(dGlobal[offsetD + tpBlockSize], zLocal[dataNumPerLoop * 3 / 4], outDCParams2);
    outQueueZ.FreeTensor(zLocal);
}

template<class X1_TYPE, class X2_TYPE, class X3_TYPE, class BIAS_TYPE, class Y_TYPE>
__aicore__ inline void KernelBatchMatmul<X1_TYPE, X2_TYPE, X3_TYPE, BIAS_TYPE, Y_TYPE>::CopyOut4()
{
    zLocal = outQueueZ.DeQue<Y_TYPE>();
    DataCopy(dGlobal[offsetD], zLocal, outDCParams2);
    DataCopy(dGlobal[offsetD + tpBlockSize], zLocal[dataNumPerLoop / 2], outDCParams4);
    offsetD = cSpaceID * tpSpaceSize + cRowID * tpSpaceWTransposed * tpBlockSize;
    DataCopy(dGlobal[offsetD], zLocal[dataNumPerLoop / 2 + tpBlockSize / 2], outDCParams4);

    offsetD = cColID * tpBlockSize + tpBlockSize / 2;
    DataCopy(dGlobal[offsetD], zLocal[dataNumPerLoop / 4], outDCParams2);
    DataCopy(dGlobal[offsetD + tpBlockSize], zLocal[dataNumPerLoop * 3 / 4], outDCParams4);

    offsetD = 0;
    DataCopy(dGlobal[offsetD], zLocal[dataNumPerLoop * 3 / 4 + tpBlockSize / 2], outDCParams4);
    outQueueZ.FreeTensor(zLocal);
}
#ifdef __CCE_KT_TEST__
    REGISTER_TILING_DEFAULT(SwinAttentionFFNTilingData);
#endif
extern "C" __global__ __aicore__ void swin_attention_ffn(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR x3, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe tPipe;
    GET_TILING_DATA(tiling_data, tiling);
    GM_ADDR user = GetUserWorkspace(workspace);

    KernelBatchMatmul<half, half, half, half, half> op;
    REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm);
    op.Init(x1, x2, bias, x3, y, user, &tiling_data, &tPipe);
    op.Process();
}
