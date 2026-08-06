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
 * \file quant_block_sparse_attn_block_cube.h
 * \brief
 */
#ifndef QUANT_BLOCK_SPARSE_ATTN_BLOCK_CUBE_H_
#define QUANT_BLOCK_SPARSE_ATTN_BLOCK_CUBE_H_
#include "common/util_regbase.h"

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "common/offset_calculator.h"

#include "common/matmul.h"
#include "common/FixpipeOut.h"
#include "common/CopyInL1.h"
#include "quant_block_sparse_attn_common_arch35.h"
#include "kernel_operator_list_tensor_intf.h"
#include "quant_block_sparse_attn_attenmask.h"
using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace fa_base_matmul;
namespace BaseApi {
namespace QuantBlockSparseAttnCube {
template <QBSALayout LAYOUT>
__aicore__ inline constexpr GmFormat GetQueryGmFormat()
{
    if constexpr (LAYOUT == QBSALayout::TND)
        return GmFormat::TNGD;
    if constexpr (LAYOUT == QBSALayout::NTD)
        return GmFormat::NGTD;
    return GmFormat::TNGD;
}

template <QBSALayout LAYOUT>
__aicore__ inline constexpr GmFormat GetKVGmFormat()
{
    return GmFormat::TND;
}

/* ============确定Query的L1类型============= */
template <typename INPUT_T, uint32_t dBaseSize>
struct QL1BuffSel {
    using Type = std::conditional_t<std::is_same_v<INPUT_T, float> || (!(std::is_same_v<INPUT_T, fp8_e4m3fn_t> ||
                                                                         std::is_same_v<INPUT_T, fp8_e5m2_t> ||
                                                                         std::is_same_v<INPUT_T, hifloat8_t>) &&
                                                                       dBaseSize > 256),
                                    BuffersPolicySingleBuffer<BufferType::L1>, BuffersPolicyDB<BufferType::L1>>;
};

/* ============确定Key的L1类型============= */
template <typename INPUT_T, uint32_t s2BaseSize, uint32_t dBaseSize>
struct KVL1BuffSel {
    constexpr static bool isFP8DType = std::is_same_v<INPUT_T, fp8_e4m3fn_t> || std::is_same_v<INPUT_T, fp8_e5m2_t> ||
                                       std::is_same_v<INPUT_T, hifloat8_t>;
    using Type = std::conditional_t<
        (isFP8DType && s2BaseSize == 128 && dBaseSize == 576), BuffersPolicy4buff<BufferType::L1>,
        std::conditional_t<(!(isFP8DType) && s2BaseSize == 256 && dBaseSize > 128),
                           BuffersPolicySingleBuffer<BufferType::L1>, BuffersPolicyDB<BufferType::L1>>>;
};

/* ============确定L0A的类型============= */
template <typename INPUT_T>
struct L0ABuffSel {
    using Type = std::conditional_t<std::is_same_v<INPUT_T, float>, BuffersPolicySingleBuffer<BufferType::L0A>,
                                    BuffersPolicyDB<BufferType::L0A>>;
};
/* ============确定L0B的类型============= */
template <typename INPUT_T, uint32_t s2BaseSize, uint32_t dBaseSize>
struct L0BBuffSel {
    using Type = std::conditional_t<std::is_same_v<INPUT_T, float> ||
                                        (s2BaseSize == 256 && dBaseSize > 128 &&
                                         !(std::is_same_v<INPUT_T, fp8_e4m3fn_t> ||
                                           std::is_same_v<INPUT_T, fp8_e5m2_t> || std::is_same_v<INPUT_T, hifloat8_t>)),
                                    BuffersPolicySingleBuffer<BufferType::L0B>, BuffersPolicyDB<BufferType::L0B>>;
};
/* ============确定L0C的类型============= */
template <typename INPUT_T, uint32_t s1BaseSize, uint32_t s2BaseSize, uint32_t dVBaseSize>
struct L0CBuffSel {
    using Type = std::conditional_t<(s1BaseSize * s2BaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4 &&
                                     s1BaseSize * dVBaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4),
                                    BuffersPolicy4buff<BufferType::L0C>, BuffersPolicyDB<BufferType::L0C>>;
};
} // namespace QuantBlockSparseAttnCube

TEMPLATES_DEF
class QBSABlockCube {
public:
    /* =================编译期常量的基本块信息================= */
    static constexpr uint32_t s1BaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t dBaseSize = (uint32_t)dTemplateType;
    static constexpr uint32_t dVBaseSize = (uint32_t)dVTemplateType;
    static constexpr bool isFp8 = IsSameType<INPUT_T, fp8_e5m2_t>::value || IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
                                  IsSameType<INPUT_T, hifloat8_t>::value;
    static constexpr TPosition bmm2OutPos = GetC2Position();
    static constexpr bool bmm2Write2Ub = bmm2OutPos == TPosition::VECCALC;
    static constexpr FixpipeConfig BMM2_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, bmm2Write2Ub};
    using mm2ResPos = typename std::conditional<bmm2Write2Ub, Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>,
                                                Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_FORWARD>>::type;

    __aicore__ inline QBSABlockCube(){};
    __aicore__ inline void InitCubeBlock(TPipe *pipe, BufferManager<BufferType::L1> *l1BufferManagerPtr,
                                         __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                         __gm__ uint8_t *blockTable);
    __aicore__ inline void InitCubeInput(__gm__ uint8_t *key, __gm__ uint8_t *value, CVSharedParams<isPa> *sharedParams,
                                         AttenMaskInfo *attenMaskInfo, __gm__ int32_t *actualSeqQlenAddr,
                                         __gm__ int32_t *actualSeqKvlenAddr);
    __aicore__ inline void IterateBmm1(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &output, RunInfo &runInfo,
                                       ConstInfo &constInfo);

    __aicore__ inline void IterateBmm2(mm2ResPos &outputBuf,
                                       BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputBuf,
                                       RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void InitDequantParams(__gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK,
                                             __gm__ uint8_t *deqScaleV);
    GlobalTensor<int32_t> blockTableGm; // pageattention需要
private:
    __aicore__ inline void InitLocalBuffer();
    __aicore__ inline void InitGmTensor(CVSharedParams<isPa> *sharedParams, __gm__ int32_t *actualSeqQlenAddr);
    __aicore__ inline void CalcS2Coord(RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void GetKvByTensorList(RunInfo &runInfo, const ConstInfo &constInfo,
                                             GlobalTensor<INPUT_T> &keyValueGm, GlobalTensor<INPUT_T> &tempKeyValueGm);
    __aicore__ inline GlobalTensor<INPUT_T> GetKeyGm(RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline GlobalTensor<INPUT_T> GetValueGm(RunInfo &runInfo, ConstInfo &constInfo);

    __aicore__ inline void IterateBmm1Nd(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
                                         RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void IterateBmm1Dn(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
                                         RunInfo &runInfo, ConstInfo &constInfo);

    TPipe *tPipe;
    /* =====================GM变量==================== */
    __gm__ uint8_t *currentKey;    // pageattention需要
    __gm__ uint8_t *currentValue;  // pageattention需要
    __gm__ uint8_t *blocktablePtr; // pageattention需要
    static constexpr GmFormat Q_FORMAT = QuantBlockSparseAttnCube::GetQueryGmFormat<layout>();
    static constexpr GmFormat KV_FORMAT = QuantBlockSparseAttnCube::GetKVGmFormat<kvLayout>();
    FaGmTensor<INPUT_T, Q_FORMAT, int32_t> queryGm;
    FaGmTensor<INPUT_T, KV_FORMAT, int32_t> keyGm;
    FaGmTensor<INPUT_T, KV_FORMAT, int32_t> valueGm;
    GlobalTensor<float> deScaleQGm;
    GlobalTensor<float> deScaleKGm;
    GlobalTensor<float> deScaleVGm;

    uint32_t maxBlockNumPerBatch = 0; // pageattention需要
    KVLAYOUT pa_kvLayout;             // pageattention需要

    /* =====================运行时变量==================== */
    CubeCoordInfo coordInfo[3];

    /* =====================LocalBuffer变量==================== */
    BufferManager<BufferType::L1> *l1BufferManagerPtr;
    BufferManager<BufferType::L0A> l0aBufferManager;
    BufferManager<BufferType::L0B> l0bBufferManager;
    BufferManager<BufferType::L0C> l0cBufferManager;

    // D小于等于256 mm1左矩阵Q，GS1循环内左矩阵复用,
    // GS1循环间开pingpong；D大于256使用单块Buffer，S1循环间驻留；fp32场景单块不驻留
    typename QuantBlockSparseAttnCube::QL1BuffSel<INPUT_T, dBaseSize>::Type l1QBuffers;
    // mm1右矩阵K
    typename QuantBlockSparseAttnCube::KVL1BuffSel<INPUT_T, s2BaseSize, dBaseSize>::Type l1KBuffers;

    // mm2右矩阵V
    typename QuantBlockSparseAttnCube::KVL1BuffSel<INPUT_T, s2BaseSize, dBaseSize>::Type l1VBuffers;
    // L0A
    using L0AType = typename QuantBlockSparseAttnCube::L0ABuffSel<INPUT_T>::Type;
    L0AType mmL0ABuffers;
    // L0B
    using L0BType = typename QuantBlockSparseAttnCube::L0BBuffSel<INPUT_T, s2BaseSize, dBaseSize>::Type;
    L0BType mmL0BBuffers;
    // L0C
    using L0CType = typename QuantBlockSparseAttnCube::L0CBuffSel<INPUT_T, s1BaseSize, s2BaseSize, dVBaseSize>::Type;
    L0CType mmL0CBuffers;
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void QBSABlockCube<TEMPLATE_ARGS>::InitCubeBlock(TPipe *pipe,
                                                                   BufferManager<BufferType::L1> *l1BuffMgr,
                                                                   __gm__ uint8_t *query, __gm__ uint8_t *key,
                                                                   __gm__ uint8_t *value, __gm__ uint8_t *blockTable)
{
    if ASCEND_IS_AIC {
        tPipe = pipe;
        l1BufferManagerPtr = l1BuffMgr;
        this->queryGm.gmTensor.SetGlobalBuffer((__gm__ INPUT_T *)query);
        if constexpr (isPa) {
            blocktablePtr = blockTable;
        }
        InitLocalBuffer();
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void QBSABlockCube<TEMPLATE_ARGS>::InitCubeInput(__gm__ uint8_t *key, __gm__ uint8_t *value,
                                                                  CVSharedParams<isPa> *sharedParams,
                                                                  AttenMaskInfo *attenMaskInfo,
                                                                  __gm__ int32_t *actualSeqQlenAddr,
                                                                  __gm__ int32_t *actualSeqKvlenAddr)
{
    if ASCEND_IS_AIC {
        if (sharedParams->fromFused) {
            ListTensorDesc keyListTensorDescInit((__gm__ void *)key);
            ListTensorDesc valueListTensorDescInit((__gm__ void *)value);
            currentKey = (__gm__ uint8_t *)keyListTensorDescInit.GetDataPtr<__gm__ uint8_t>(0);
            currentValue = (__gm__ uint8_t *)valueListTensorDescInit.GetDataPtr<__gm__ uint8_t>(0);
            if (sharedParams->isKvContinuous == 1) {
                this->keyGm.gmTensor.SetGlobalBuffer((__gm__ INPUT_T *)currentKey);
                this->valueGm.gmTensor.SetGlobalBuffer((__gm__ INPUT_T *)currentValue);
            } else {
                this->keyGm.gmTensor.SetGlobalBuffer((__gm__ INPUT_T *)key);
                this->valueGm.gmTensor.SetGlobalBuffer((__gm__ INPUT_T *)value);
            }
        } else {
            this->keyGm.gmTensor.SetGlobalBuffer((__gm__ INPUT_T *)key);
            this->valueGm.gmTensor.SetGlobalBuffer((__gm__ INPUT_T *)value);
        }
        attenMaskInfo->preTokens = sharedParams->preTokens;
        attenMaskInfo->nextTokens = sharedParams->nextTokens;
        attenMaskInfo->compressMode = sharedParams->compressMode;
        if constexpr (isPa) {
            this->blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blocktablePtr);
            this->maxBlockNumPerBatch = sharedParams->blockTableDim2;
            if (sharedParams->paLayoutType == 2) { // NZ下paLayoutType == 2
                pa_kvLayout = KVLAYOUT::NZ;
            } else {
                pa_kvLayout = sharedParams->paLayoutType == 1 ? KVLAYOUT::BBH : KVLAYOUT::BNBD;
            }
        }
        InitGmTensor(sharedParams, actualSeqQlenAddr);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void QBSABlockCube<TEMPLATE_ARGS>::InitLocalBuffer()
{
    constexpr uint32_t mm1LeftSize = s1BaseSize * dBaseSize * sizeof(INPUT_T);
    constexpr uint32_t mm1RightSize = dBaseSize * s2BaseSize * sizeof(INPUT_T);
    constexpr uint32_t mm2RightSize = (uint32_t)dVTemplateType * s2BaseSize * sizeof(INPUT_T);
    l1QBuffers.Init((*l1BufferManagerPtr), mm1LeftSize);
    l1KBuffers.Init((*l1BufferManagerPtr), mm1RightSize);
    l1VBuffers.Init((*l1BufferManagerPtr), mm2RightSize);

    // L0A B C 当前写死，能否通过基础api获取
    l0aBufferManager.Init(tPipe, 65536);  // 64 * 1024
    l0bBufferManager.Init(tPipe, 65536);  // 64 * 1024
    l0cBufferManager.Init(tPipe, 262144); // 256 * 1024
    // L0A B C当前写死，要改成通过计算获取
    if (IsSameType<INPUT_T, float>::value) {
        mmL0ABuffers.Init(l0aBufferManager, 64 * 1024);
        mmL0BBuffers.Init(l0bBufferManager, 64 * 1024);
    } else {
        mmL0ABuffers.Init(l0aBufferManager, 32 * 1024);
        mmL0BBuffers.Init(l0bBufferManager, 32 * 1024);
    }

    if constexpr (s1BaseSize * s2BaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4 &&
                  s1BaseSize * dVBaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4) {
        mmL0CBuffers.Init(l0cBufferManager, (L0C_SIZE / NUM_4) * KB_TO_BYTES);
    } else {
        mmL0CBuffers.Init(l0cBufferManager, (L0C_SIZE / NUM_2) * KB_TO_BYTES);
    }
}

/* 初始化GmTensor,设置shape信息并计算strides */
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void QBSABlockCube<TEMPLATE_ARGS>::InitGmTensor(CVSharedParams<isPa> *sharedParams,
                                                                 __gm__ int32_t *actualSeqQlenAddr)
{
    if constexpr (GmLayoutParams<Q_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_TND) {
        GlobalTensor<int32_t> actualSeqQLen;
        actualSeqQLen.SetGlobalBuffer((__gm__ int32_t *)actualSeqQlenAddr);
        this->queryGm.offsetCalculator.Init(sharedParams->n2Size, sharedParams->gSize, sharedParams->dSize,
                                            actualSeqQLen, sharedParams->seqUsedQlenSize);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void QBSABlockCube<TEMPLATE_ARGS>::CalcS2Coord(RunInfo &runInfo, ConstInfo &constInfo)
{
    // QBSA sparse: sparse block indices 已在 SetRunInfo 中预计算
    coordInfo[runInfo.taskIdMod3].sparseBlockIdx[0] = runInfo.sparseBlkIdx1;
    coordInfo[runInfo.taskIdMod3].sparseBlockIdx[1] = runInfo.sparseBlkIdx2;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void QBSABlockCube<TEMPLATE_ARGS>::IterateBmm1(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf, RunInfo &runInfo, ConstInfo &constInfo)
{
    CalcS2Coord(runInfo, constInfo);
    if constexpr (isFp8) {
        if constexpr (useDn) {
            IterateBmm1Dn(outputBuf, runInfo, constInfo);
        } else {
            IterateBmm1Nd(outputBuf, runInfo, constInfo);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void QBSABlockCube<TEMPLATE_ARGS>::IterateBmm2(
    mm2ResPos &outputBuf, BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputBuf,
    RunInfo &runInfo, ConstInfo &constInfo)
{
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> mm2A = inputBuf.Get();
    Buffer<BufferType::L1> mm2B = l1VBuffers.Get();
    // mm2A.WaitCrossCore();
    mm2B.Wait<HardEvent::MTE1_MTE2>(); // 占用L1B
    LocalTensor<INPUT_T> mm2BTensor = mm2B.GetTensor<INPUT_T>();
    uint32_t actualS2RealSize = (uint32_t)runInfo.s2SparseBlk1RealSize;
    if (likely(runInfo.sparseBlkIdx2 != -1)) {
        actualS2RealSize += (uint32_t)runInfo.s2SparseBlk2RealSize;
    }
    if constexpr (isPa) {
        Position startPos;
        startPos.bIdx = runInfo.boIdx;
        startPos.n2Idx = runInfo.n2oIdx;
        startPos.s2Offset = coordInfo[runInfo.taskIdMod3].sparseBlockIdx[0] * constInfo.kvSparseBlockSize;
        startPos.dIdx = 0;
        PAShape shape;
        shape.blockSize = constInfo.kvSparseBlockSize;
        shape.headNum = constInfo.n2Size;
        shape.headDim = constInfo.dSizeV;
        shape.actHeadDim = constInfo.dSizeV;
        shape.maxblockNumPerBatch = maxBlockNumPerBatch;
        shape.copyRowNum = runInfo.s2SparseBlk1RealSize;
        shape.pageStride = constInfo.paBlockStride;
        shape.rowStride = constInfo.dSizeV;
        if constexpr (isFp8) {
            shape.copyRowNumAlign = (actualS2RealSize + 31) >> 5 << 5;
        } else {
            shape.copyRowNumAlign = (actualS2RealSize + 15) >> 4 << 4;
        }
        GlobalTensor<INPUT_T> mm2BGmTensor = GetValueGm(runInfo, constInfo);
        GmCopyInToL1PA<INPUT_T>(mm2BTensor, mm2BGmTensor, blockTableGm, pa_kvLayout, shape, startPos);

        if (likely(runInfo.sparseBlkIdx2 != -1)) {
            startPos.s2Offset = coordInfo[runInfo.taskIdMod3].sparseBlockIdx[1] * constInfo.kvSparseBlockSize;
            shape.copyRowNum = runInfo.s2SparseBlk2RealSize;
            uint32_t blockElementCnt = 32U / sizeof(INPUT_T);
            if constexpr (IsSameType<INPUT_T, int4b_t>::value) {
                blockElementCnt = 64U;
            }
            uint32_t halfOffset = runInfo.s2SparseBlk1RealSize * blockElementCnt;
            LocalTensor<INPUT_T> mm2BTensor2 = mm2BTensor[halfOffset];
            GmCopyInToL1PA<INPUT_T>(mm2BTensor2, mm2BGmTensor, blockTableGm, pa_kvLayout, shape, startPos);
        }
    } else {
        uint64_t gmOffset = runInfo.keyOffset;
        if (constInfo.dSize != constInfo.dSizeV) {
            gmOffset = this->valueGm.offsetCalculator.GetOffset(coordInfo[runInfo.taskIdMod3].curBIdx, runInfo.n2oIdx,
                                                                coordInfo[runInfo.taskIdMod3].s2Coord, 0);
        }
        CopyToL1Nd2Nz<INPUT_T>(mm2BTensor, GetValueGm(runInfo, constInfo)[gmOffset], runInfo.s2RealSize,
                               constInfo.dSizeV, constInfo.mm2Kb);
    }
    mm2B.Set<HardEvent::MTE2_MTE1>(); // 通知

    mm2A.WaitCrossCore();
    Buffer<BufferType::L0C> mm2ResL0C = mmL0CBuffers.Get();
    mm2ResL0C.Wait<HardEvent::FIX_M>(); // 占用
    MMParam param = {
        (uint32_t)s1BaseSize,       // singleM 128
        (uint32_t)constInfo.dSizeV, // singleN 128
        (uint32_t)actualS2RealSize, // singleK
        useDn,                      // isLeftTranspose
        false                       // isRightTranspose
    };
    if constexpr (!useDn) {
        param.realM = (uint32_t)runInfo.s1RealSize;
    }
    mm2B.Wait<HardEvent::MTE2_MTE1>(); // 等待
    if constexpr (isFp8) {
        MatmulFull<INPUT_T, INPUT_T, T, 128, (uint32_t)dVTemplateType, 128, ABLayout::MK, ABLayout::KN>(
            mm2A.GetTensor<INPUT_T>(), mm2BTensor, mmL0ABuffers, mmL0BBuffers, mm2ResL0C.GetTensor<T>(), param);
    } else {
        if constexpr ((uint32_t)dVTemplateType > 128) {
            MatmulN<INPUT_T, INPUT_T, T, (uint32_t)s1TemplateType, 128, s2BaseSize, ABLayout::MK, ABLayout::KN>(
                mm2A.GetTensor<INPUT_T>(), mm2BTensor, mmL0ABuffers, mmL0BBuffers, mm2ResL0C.GetTensor<T>(), param);
        } else {
            if constexpr (s2BaseSize == 128) {
                MatmulFull<INPUT_T, INPUT_T, T, 128, (uint32_t)dVTemplateType, 128, ABLayout::MK, ABLayout::KN>(
                    mm2A.GetTensor<INPUT_T>(), mm2BTensor, mmL0ABuffers, mmL0BBuffers, mm2ResL0C.GetTensor<T>(), param);
            } else {
                MatmulBase<INPUT_T, INPUT_T, T, 128, (uint32_t)dVTemplateType, 128, ABLayout::MK, ABLayout::KN>(
                    mm2A.GetTensor<INPUT_T>(), mm2BTensor, mmL0ABuffers, mmL0BBuffers, mm2ResL0C.GetTensor<T>(), param);
            }
        }
    }

    mm2B.Set<HardEvent::MTE1_MTE2>(); // 释放L1B

    mm2ResL0C.Set<HardEvent::M_FIX>();  // 通知
    mm2ResL0C.Wait<HardEvent::M_FIX>(); // 等待

    if constexpr (bmm2Write2Ub) {
        outputBuf.WaitCrossCore();
    }

    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams; // L0C→UB;FixpipeParamsM300:L0C→UB
    if constexpr (bmm2Write2Ub) {
        fixpipeParams.nSize = (constInfo.dSizeV + 7) >> 3 << 3; // L0C上的bmm1结果矩阵N方向的size大小
    } else {
        fixpipeParams.nSize = constInfo.dSizeV; // L0C上的bmm1结果矩阵N方向的size大小
    }

    if constexpr (useDn) {
        fixpipeParams.mSize = s1BaseSize;
        fixpipeParams.srcStride = ((s1BaseSize + 15) / 16) * 16;
    } else {
        fixpipeParams.mSize =
            (runInfo.s1RealSize + 1) >>
            1 << 1; // 有效数据不足16行，只需输出部分行即可;L0C上的bmm1结果矩阵M方向的size大小必须是偶数
        fixpipeParams.srcStride = ((fixpipeParams.mSize + 15) / 16) *
                                  16; // L0C上bmm1结果相邻连续数据片段间隔（前面一个数据块的头与后面数据块的头的间隔）
    }
    if constexpr (bmm2Write2Ub) {
        fixpipeParams.dstStride = ((uint32_t)dVTemplateType + 15) >> 4 << 4;
    } else {
        fixpipeParams.dstStride = (uint32_t)constInfo.dSizeV; // dstGm 两行之间的间隔
    }
    fixpipeParams.dualDstCtl = 1;
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;
    Fixpipe<T, T, BMM2_FIXPIPE_CONFIG>(outputBuf.template GetTensor<T>(), mm2ResL0C.GetTensor<T>(),
                                       fixpipeParams); // 将matmul结果从L0C搬运到UB
    mm2ResL0C.Set<HardEvent::FIX_M>();                 // 释放
    outputBuf.SetCrossCore();
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void QBSABlockCube<TEMPLATE_ARGS>::IterateBmm1Dn(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf, RunInfo &runInfo, ConstInfo &constInfo)
{
    Buffer<BufferType::L1> mm1A;
    Buffer<BufferType::L1> mm1B;
    if (unlikely(runInfo.s2LoopCount == 0)) {
        mm1B = l1QBuffers.Get();
        mm1B.Wait<HardEvent::MTE1_MTE2>();
        LocalTensor<INPUT_T> mm1BTensor = mm1B.GetTensor<INPUT_T>();

        CopyToL1Nd2Nz<INPUT_T>(mm1BTensor, this->queryGm.gmTensor[runInfo.queryOffset], runInfo.s1RealSize,
                               constInfo.dSize, constInfo.mm1Ka);
        mm1B.Set<HardEvent::MTE2_MTE1>();
    } else {
        mm1B = l1QBuffers.GetPre();
        mm1B.Set<HardEvent::MTE2_MTE1>();
    }
    mm1A = l1KBuffers.Get();
    mm1A.Wait<HardEvent::MTE1_MTE2>();
    LocalTensor<INPUT_T> mm1ATensor = mm1A.GetTensor<INPUT_T>();
    uint32_t actualS2RealSize = (uint32_t)runInfo.s2SparseBlk1RealSize;
    if (likely(runInfo.sparseBlkIdx2 != -1)) {
        actualS2RealSize += (uint32_t)runInfo.s2SparseBlk2RealSize;
    }
    if constexpr (isPa) {
        Position startPos;
        startPos.bIdx = runInfo.boIdx;
        startPos.n2Idx = runInfo.n2oIdx;
        startPos.s2Offset = coordInfo[runInfo.taskIdMod3].sparseBlockIdx[0] * constInfo.kvSparseBlockSize;
        startPos.dIdx = 0;
        PAShape shape;
        shape.blockSize = constInfo.kvSparseBlockSize;
        shape.headNum = constInfo.n2Size;
        shape.headDim = constInfo.dSize;
        shape.actHeadDim = constInfo.dSize;
        shape.maxblockNumPerBatch = maxBlockNumPerBatch;
        shape.copyRowNum = runInfo.s2SparseBlk1RealSize;
        shape.pageStride = constInfo.paBlockStride;
        shape.rowStride = constInfo.dSize;
        if constexpr (isFp8) {
            shape.copyRowNumAlign = (actualS2RealSize + 31) >> 5 << 5;
        } else {
            shape.copyRowNumAlign = (actualS2RealSize + 15) >> 4 << 4;
        }
        GlobalTensor<INPUT_T> mm1AGmTensor = GetKeyGm(runInfo, constInfo);
        GmCopyInToL1PA<INPUT_T>(mm1ATensor, mm1AGmTensor, blockTableGm, pa_kvLayout, shape, startPos);

        if (likely(runInfo.sparseBlkIdx2 != -1)) {
            startPos.s2Offset = coordInfo[runInfo.taskIdMod3].sparseBlockIdx[1] * constInfo.kvSparseBlockSize;
            shape.copyRowNum = runInfo.s2SparseBlk2RealSize;
            uint32_t blockElementCnt = 32U / sizeof(INPUT_T);
            if constexpr (IsSameType<INPUT_T, int4b_t>::value) {
                blockElementCnt = 64U;
            }
            uint32_t halfOffset = runInfo.s2SparseBlk1RealSize * blockElementCnt;
            LocalTensor<INPUT_T> mm1ATensor2 = mm1ATensor[halfOffset];
            GmCopyInToL1PA<INPUT_T>(mm1ATensor2, mm1AGmTensor, blockTableGm, pa_kvLayout, shape, startPos);
        }
    } else {
        runInfo.keyOffset = this->keyGm.offsetCalculator.GetOffset(
            coordInfo[runInfo.taskIdMod3].curBIdx, runInfo.n2oIdx, coordInfo[runInfo.taskIdMod3].s2Coord, 0);
        CopyToL1Nd2Nz<INPUT_T>(mm1ATensor, GetKeyGm(runInfo, constInfo)[runInfo.keyOffset], runInfo.s2RealSize,
                               constInfo.dSize, constInfo.mm1Kb);
    }
    mm1A.Set<HardEvent::MTE2_MTE1>();

    mm1A.Wait<HardEvent::MTE2_MTE1>();
    mm1B.Wait<HardEvent::MTE2_MTE1>();

    Buffer<BufferType::L0C> mm1ResL0C = mmL0CBuffers.Get();
    mm1ResL0C.Wait<HardEvent::FIX_M>();
    MMParam param = {actualS2RealSize, (uint32_t)runInfo.s1RealSize, (uint32_t)(constInfo.dSize), 0, 1};
    MatmulBase<INPUT_T, INPUT_T, T, 256, 128, dBaseSize, ABLayout::MK, ABLayout::KN>(
        mm1A.GetTensor<INPUT_T>(), mm1B.GetTensor<INPUT_T>(), mmL0ABuffers, mmL0BBuffers, mm1ResL0C.GetTensor<T>(),
        param);

    if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopLimit)) {
        mm1B.Set<HardEvent::MTE1_MTE2>();
    }
    mm1A.Set<HardEvent::MTE1_MTE2>();

    mm1ResL0C.Set<HardEvent::M_FIX>();
    mm1ResL0C.Wait<HardEvent::M_FIX>();

    outputBuf.WaitCrossCore();

    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
    fixpipeParams.nSize = (runInfo.s1RealSize + 31) >> 5 << 5;
    fixpipeParams.mSize = actualS2RealSize;
    fixpipeParams.srcStride = ((fixpipeParams.mSize + 15) / 16) * 16;
    if constexpr (useDn && isFp8) {
        fixpipeParams.dstStride = 64;
    } else {
        fixpipeParams.dstStride = fixpipeParams.nSize / 2;
    }
    fixpipeParams.dualDstCtl = 2;
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;
    Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(outputBuf.template GetTensor<T>(), mm1ResL0C.GetTensor<T>(), fixpipeParams);
    mm1ResL0C.Set<HardEvent::FIX_M>();
    outputBuf.SetCrossCore();
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void QBSABlockCube<TEMPLATE_ARGS>::InitDequantParams(__gm__ uint8_t *deqScaleQ,
                                                                      __gm__ uint8_t *deqScaleK,
                                                                      __gm__ uint8_t *deqScaleV)
{
    if constexpr (isFp8) {
        deScaleQGm.SetGlobalBuffer((__gm__ float *)deqScaleQ);
        deScaleKGm.SetGlobalBuffer((__gm__ float *)deqScaleK);
        deScaleVGm.SetGlobalBuffer((__gm__ float *)deqScaleV);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void QBSABlockCube<TEMPLATE_ARGS>::GetKvByTensorList(RunInfo &runInfo, const ConstInfo &constInfo,
                                                                       GlobalTensor<INPUT_T> &keyValueGm,
                                                                       GlobalTensor<INPUT_T> &tempKeyValueGm)
{
    if (constInfo.isKvContinuous != 0) {
        return;
    }
    ListTensorDesc keyValueListTensorDesc((__gm__ void *)keyValueGm.GetPhyAddr());
    __gm__ uint8_t *tempKeyValueGmPtr =
        (__gm__ uint8_t *)keyValueListTensorDesc.GetDataPtr<__gm__ uint8_t>(runInfo.boIdx);
    tempKeyValueGm.SetGlobalBuffer((__gm__ INPUT_T *)tempKeyValueGmPtr);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline GlobalTensor<INPUT_T> QBSABlockCube<TEMPLATE_ARGS>::GetKeyGm(RunInfo &runInfo, ConstInfo &constInfo)
{
    GlobalTensor<INPUT_T> tempKeyGm = this->keyGm.gmTensor;
    GetKvByTensorList(runInfo, constInfo, this->keyGm.gmTensor, tempKeyGm);
    return tempKeyGm;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline GlobalTensor<INPUT_T> QBSABlockCube<TEMPLATE_ARGS>::GetValueGm(RunInfo &runInfo, ConstInfo &constInfo)
{
    GlobalTensor<INPUT_T> tempValueGm = this->valueGm.gmTensor;
    GetKvByTensorList(runInfo, constInfo, this->valueGm.gmTensor, tempValueGm);
    return tempValueGm;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void QBSABlockCube<TEMPLATE_ARGS>::IterateBmm1Nd(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf, RunInfo &runInfo, ConstInfo &constInfo)
{
    // 计算key的offset
    Buffer<BufferType::L1> mm1A;
    Buffer<BufferType::L1> mm1B;
    // 左矩阵复用，S2的第一次循环加载左矩阵
    // 加载左矩阵到L1 ,当前使用全载方式
    if (unlikely(runInfo.s2LoopCount == 0)) { // sOuter循环第一个基本块：搬运Q
        mm1A = l1QBuffers.Get();
        mm1A.Wait<HardEvent::MTE1_MTE2>(); // 占用L1A
        LocalTensor<INPUT_T> mm1ATensor = mm1A.GetTensor<INPUT_T>();

        CopyToL1Nd2Nz<INPUT_T>(mm1ATensor, this->queryGm.gmTensor[runInfo.queryOffset], runInfo.s1RealSize,
                               constInfo.dSize, constInfo.mm1Ka);

        mm1A.Set<HardEvent::MTE2_MTE1>(); // 通知
    } else {                              // 非S2的第一次循环直接复用Q
        mm1A = l1QBuffers.GetPre();
        // 左矩阵复用时，sinner循环内不需要MTE2同步等待
        mm1A.Set<HardEvent::MTE2_MTE1>(); // 通知
    }

    // 加载当前轮的右矩阵到L1
    mm1B = l1KBuffers.Get();
    mm1B.Wait<HardEvent::MTE1_MTE2>(); // 占用L1B
    LocalTensor<INPUT_T> mm1BTensor = mm1B.GetTensor<INPUT_T>();
    if constexpr (isPa) {
        Position startPos;
        startPos.bIdx = runInfo.boIdx;
        startPos.n2Idx = runInfo.n2oIdx;
        startPos.s2Offset = runInfo.s2StartIdx + runInfo.s2LoopCount * s2BaseSize; // 非FD场景
        startPos.dIdx = 0;
        PAShape shape;
        shape.blockSize = constInfo.kvSparseBlockSize;
        shape.headNum = constInfo.n2Size;
        shape.headDim = constInfo.dSize;
        shape.actHeadDim = constInfo.dSize;
        shape.maxblockNumPerBatch = maxBlockNumPerBatch;
        shape.copyRowNum = runInfo.s2RealSize;
        shape.pageStride = constInfo.paBlockStride;
        shape.rowStride = constInfo.dSize;
        if constexpr (isFp8) {
            shape.copyRowNumAlign = (runInfo.s2RealSize + 31) >> 5 << 5;
        } else {
            shape.copyRowNumAlign = (runInfo.s2RealSize + 15) >> 4 << 4;
        }
        GlobalTensor<INPUT_T> mm1BGmTensor = GetKeyGm(runInfo, constInfo);
        GmCopyInToL1PA<INPUT_T>(mm1BTensor, mm1BGmTensor, blockTableGm, pa_kvLayout, shape, startPos);
    } else {
        runInfo.keyOffset = this->keyGm.offsetCalculator.GetOffset(
            coordInfo[runInfo.taskIdMod3].curBIdx, runInfo.n2oIdx, coordInfo[runInfo.taskIdMod3].s2Coord, 0);
        CopyToL1Nd2Nz<INPUT_T>(mm1BTensor, GetKeyGm(runInfo, constInfo)[runInfo.keyOffset], runInfo.s2RealSize,
                               constInfo.dSize, constInfo.mm1Kb);
    }
    mm1B.Set<HardEvent::MTE2_MTE1>(); // 通知

    mm1A.Wait<HardEvent::MTE2_MTE1>(); // 等待L1A
    mm1B.Wait<HardEvent::MTE2_MTE1>(); // 等待L1B

    Buffer<BufferType::L0C> mm1ResL0C = mmL0CBuffers.Get();
    mm1ResL0C.Wait<HardEvent::FIX_M>(); // 占用
    MMParam param = {
        (uint32_t)runInfo.s1RealSize, // singleM
        (uint32_t)runInfo.s2RealSize, // singleN
        (uint32_t)(constInfo.dSize),  // singleK
        0,                            // isLeftTranspose
        1                             // isRightTranspose
    };
    MatmulBase<INPUT_T, INPUT_T, T, 128, 128, dBaseSize, ABLayout::MK, ABLayout::KN>(
        mm1A.GetTensor<INPUT_T>(), mm1B.GetTensor<INPUT_T>(), mmL0ABuffers, mmL0BBuffers, mm1ResL0C.GetTensor<T>(),
        param);
    if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopLimit)) {
        mm1A.Set<HardEvent::MTE1_MTE2>(); // 释放L1A
    }

    mm1B.Set<HardEvent::MTE1_MTE2>(); // 释放L1B

    mm1ResL0C.Set<HardEvent::M_FIX>();  // 通知
    mm1ResL0C.Wait<HardEvent::M_FIX>(); // 等待L0C

    outputBuf.WaitCrossCore();

    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams; // L0C→UB
    fixpipeParams.nSize =
        (runInfo.s2RealSize + 7) >>
        3 << 3; // L0C上的bmm1结果矩阵N方向的size大小; 同mmadParams.n; 为什么要8个元素对齐(32B对齐) // 128
    fixpipeParams.mSize =
        (runInfo.s1RealSize + 1) >>
        1 << 1; // 有效数据不足16行，只需要输出部分行即可; L0C上的bmm1结果矩阵M方向的size大小(必须为偶数) // 128
    fixpipeParams.srcStride = ((fixpipeParams.mSize + 15) / 16) *
                              16; // L0C上bmm1结果相邻连续数据片段间隔(前面一个数据块的头与后面数据块的头的间隔),
                                  // 单位为16*sizeof(T) // 源Nz矩阵中相邻大Z排布的起始地址偏移
    fixpipeParams.dstStride = s2BaseSize; // mmResUb上两行之间的间隔，单位：element。 // 128:根据比对dump文件得到,
                                          // ND方案(S1*S2)时脏数据用mask剔除
    fixpipeParams.dualDstCtl = 1; // 双目标模式，按M维度拆分，M / 2 * N写入每个UB, M必须为2的倍数
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;

    Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(outputBuf.template GetTensor<T>(), mm1ResL0C.GetTensor<T>(),
                                        fixpipeParams); // 将matmul结果从L0C搬运到UB
    mm1ResL0C.Set<HardEvent::FIX_M>();                  // 释放L0C
    outputBuf.SetCrossCore();
}

TEMPLATES_DEF
class QBSABlockCubeDummy {
public:
    GlobalTensor<int32_t> blockTableGm; // pageattention需要
    static constexpr bool isFp8 = QBSABlockCube<TEMPLATE_ARGS>::isFp8;
    static constexpr TPosition bmm2OutPos = QBSABlockCube<TEMPLATE_ARGS>::bmm2OutPos;
    static constexpr bool bmm2Write2Ub = QBSABlockCube<TEMPLATE_ARGS>::bmm2Write2Ub;
    __aicore__ inline QBSABlockCubeDummy(){};
    __aicore__ inline void InitCubeBlock(TPipe *pipe, BufferManager<BufferType::L1> *l1BufferManagerPtr,
                                         __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                         __gm__ uint8_t *blockTable)
    {}
    __aicore__ inline void InitCubeInput(__gm__ uint8_t *key, __gm__ uint8_t *value, CVSharedParams<isPa> *sharedParams,
                                         AttenMaskInfo *attenMaskInfo, __gm__ int32_t *actualSeqQlenAddr,
                                         __gm__ int32_t *actualSeqKvlenAddr)
    {}
    __aicore__ inline void IterateBmm1(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
                                       RunInfo &runInfo, ConstInfo &constInfo)
    {}

    using mm2ResPos = typename std::conditional<bmm2Write2Ub, Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>,
                                                Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_FORWARD>>::type;
    __aicore__ inline void IterateBmm2(mm2ResPos &outputBuf,
                                       BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputBuf,
                                       RunInfo &runInfo, ConstInfo &constInfo)
    {}
    __aicore__ inline void InitDequantParams(__gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK,
                                             __gm__ uint8_t *deqScaleV)
    {}
};

template <typename T>
struct CubeBlockTraits; // 声明

/* 生成CubeBlockTraits */
#define GEN_TRAIT_TYPE(name, ...) using name##_TRAITS = name;
#define GEN_TRAIT_CONST(name, type, ...) static constexpr type name##Traits = name;

#define DEFINE_CUBE_BLOCK_TRAITS(CUBE_BLOCK_CLASS) \
    TEMPLATES_DEF_NO_DEFAULT \
    struct CubeBlockTraits<CUBE_BLOCK_CLASS<TEMPLATE_ARGS>> { \
        CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TRAIT_TYPE) \
        CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_TRAIT_CONST) \
    };

DEFINE_CUBE_BLOCK_TRAITS(QBSABlockCube);
DEFINE_CUBE_BLOCK_TRAITS(QBSABlockCubeDummy);

// /* 生成Arg Traits, kernel中只需要调用ARGS_TRAITS就可以获取所有CubeBlock中的模板参数 */
#define GEN_ARGS_TYPE(name, ...) using name = typename CubeBlockTraits<CubeBlockType>::name##_TRAITS;
#define GEN_ARGS_CONST(name, type, ...) static constexpr type name = CubeBlockTraits<CubeBlockType>::name##Traits;
#define ARGS_TRAITS \
    CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_ARGS_TYPE) \
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_ARGS_CONST)
} // namespace BaseApi
#endif // QUANT_BLOCK_SPARSE_ATTN_BLOCK_CUBE_H_
