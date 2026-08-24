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
 * \file fia_block_cube_noquant_mla.h
 * \brief arch35 FIA block cube 非量化 MLA
 */
#ifndef FIA_BLOCK_CUBE_NOQUANT_MLA_H_
#define FIA_BLOCK_CUBE_NOQUANT_MLA_H_
#include "../../../common/op_kernel/offset_calculator.h"
#include "../../../common/op_kernel/matmul.h"
#include "../../../common/op_kernel/FixpipeOut.h"
#include "memory_copy_arch35_fused_infer.h"
#include "kernel_operator_list_tensor_intf.h"
using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace fa_base_matmul;
using namespace AttentionCommon;

namespace BaseApi {

template <typename INPUT_T, typename T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, uint8_t KvLayoutType = 0>
class FANoQuantMlaBlockCube {
public:
    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t dBaseSize = (uint32_t)dTemplateType;
    static constexpr uint32_t dVBaseSize = (uint32_t)dVTemplateType;
    static constexpr LayOutTypeEnum LAYOUT = layout;
    static constexpr bool PAGE_ATTENTION = (KvLayoutType > 0);

    static constexpr FixpipeConfig BMM2_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, true}; // true: bmm2Write2Ub
    static constexpr GmFormat Q_FORMAT = GetQueryGmFormat<layout>();
    static constexpr GmFormat KV_FORMAT = GetKVGmFormat<layout, KvLayoutType, PAGE_ATTENTION>();

    using ROPE_T = INPUT_T;
    using Q_T = INPUT_T;
    using KV_T = INPUT_T;
    using MM_T = T;
    using mm2ResPos = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    using MM1_DBUF_T = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    using KV_BUF_T = Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>;

    using ConstInfoX = ConstInfo_t<FiaKernelType::NO_QUANT>;

    TPipe *tPipe = nullptr;

    /* =====================GM变量(with layout)==================== */
    FaGmTensor<Q_T, Q_FORMAT> queryGm;
    FaGmTensor<KV_T, KV_FORMAT> keyGm;
    FaGmTensor<KV_T, KV_FORMAT> valueGm;
    FaGmTensor<ROPE_T, Q_FORMAT> queryRopeGm;
    FaGmTensor<ROPE_T, KV_FORMAT> keyRopeGm;
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<uint64_t> actualSeqLengthsGmQ;
    GlobalTensor<uint64_t> actualSeqLengthsGm;

    CopyQueryGmToL1<Q_T, Q_FORMAT> copyQueryGmToL1;
    CopyKvGmToL1<KV_T, KV_FORMAT> copyKvGmToL1;

    /* =====================LocalBuffer变量====================*/
    BufferManager<BufferType::L1> *l1BufferManagerPtr;
    BufferManager<BufferType::L0A> l0aBufferManager;
    BufferManager<BufferType::L0B> l0bBufferManager;
    BufferManager<BufferType::L0C> l0cBufferManager;

    BuffersPolicyDB<BufferType::L0A> mmL0ABuffers;
    BuffersPolicyDB<BufferType::L0B> mmL0BBuffers;
    BuffersPolicyDB<BufferType::L0C> mmL0CBuffers;

    BuffersPolicySingleBuffer<BufferType::L1> l1QBuffers;

    const ConstInfoX &constInfo;

    /*============================================================================== */
    __aicore__ inline FANoQuantMlaBlockCube(ConstInfoX &constInfo)
        : constInfo(constInfo){};

    __aicore__ inline void InitCubeBlock(TPipe *pipe, BufferManager<BufferType::L1> *l1BuffMgr, __gm__ uint8_t *query,
                                         __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *blockTable,
                                         __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
                                         __gm__ uint8_t *actualSeqQlenAddr, __gm__ uint8_t *actualSeqKvlenAddr)
    {
        tPipe = pipe;
        l1BufferManagerPtr = l1BuffMgr;
        InitCubeInput(query, key, value, blockTable, queryRope, keyRope, actualSeqQlenAddr, actualSeqKvlenAddr);
    }

    __aicore__ inline void InitBuffers()
    {
        static_assert(mBaseSize == 64 && s2BaseSize == 128, "mBaseSize != 64 or s2BaseSize != 128");
        l1QBuffers.Init(*l1BufferManagerPtr, (uint32_t)mBaseSize * 576 * sizeof(Q_T));

        // L0A B C 当前写死，能否通过基础api获取
        l0aBufferManager.Init(tPipe, 65536);  // 64 * 1024
        l0bBufferManager.Init(tPipe, 65536);  // 64 * 1024
        l0cBufferManager.Init(tPipe, 262144); // 256 * 1024
        // L0A B C当前写死，要改成通过计算获取
        mmL0ABuffers.Init(l0aBufferManager, 32 * 1024);
        mmL0BBuffers.Init(l0bBufferManager, 32 * 1024);
        mmL0CBuffers.Init(l0cBufferManager, 128 * 1024);
    }

    __aicore__ inline void InitCubeInput(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                         __gm__ uint8_t *blockTable, __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
                                         __gm__ uint8_t *actualSeqQlenAddr, __gm__ uint8_t *actualSeqKvlenAddr)
    {
        if (constInfo.actualSeqLenSize != 0) {
            actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ uint64_t *)actualSeqQlenAddr, constInfo.actualSeqLenSize);
        }
        if (constInfo.actualSeqLenKVSize != 0) {
            actualSeqLengthsGm.SetGlobalBuffer((__gm__ uint64_t *)actualSeqKvlenAddr, constInfo.actualSeqLenKVSize);
        }
        if constexpr (PAGE_ATTENTION) {
            blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        }

        InitQBuffer(constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size, constInfo.dSize,
                    actualSeqLengthsGmQ, constInfo.actualSeqLenSize, queryGm, query);

        ListTensorDesc keyListTensorDesc((__gm__ void *)(key));
        __gm__ uint8_t *key_ = (__gm__ uint8_t *)keyListTensorDesc.GetDataPtr<__gm__ uint8_t>(0);
        ListTensorDesc valueListTensorDesc((__gm__ void *)(value));
        __gm__ uint8_t *value_ = (__gm__ uint8_t *)valueListTensorDesc.GetDataPtr<__gm__ uint8_t>(0);

        InitKVBuffer(constInfo.bSize, constInfo.s2Size, actualSeqLengthsGm, constInfo.actualSeqLenKVSize,
                     constInfo.n2Size, constInfo.blockSize, constInfo.dSize, keyGm, key_, constInfo.keyStrides.bnStride,
                     constInfo.keyStrides.n2Stride);
        InitKVBuffer(constInfo.bSize, constInfo.s2Size, actualSeqLengthsGm, constInfo.actualSeqLenKVSize,
                     constInfo.n2Size, constInfo.blockSize, constInfo.dSizeV, valueGm, value_,
                     constInfo.valueStrides.bnStride, constInfo.valueStrides.n2Stride);

        // ROPE
        InitQBuffer(constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size, constInfo.dSizeRope,
                    actualSeqLengthsGmQ, constInfo.actualSeqLenSize, queryRopeGm, queryRope);
        InitKVBuffer(constInfo.bSize, constInfo.s2Size, actualSeqLengthsGm, constInfo.actualSeqLenKVSize,
                     constInfo.n2Size, constInfo.blockSize, constInfo.dSizeRope, keyRopeGm, keyRope,
                     constInfo.kRopeStrides.bnStride, constInfo.kRopeStrides.n2Stride);

        if (constInfo.l2CacheOffFlag) {
            // gSize*s1Size<=64单token场景: K/V数据量远超L2容量, 完全无法复用, 关闭L2 Cache避免无意义的缓存填充/驱逐开销
            // 注意: 必须等全部GM buffer(含keyRope) SetGlobalBuffer完成后再统一设置hint
#ifndef ASCENDC_OOM
            keyGm.gmTensor.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            valueGm.gmTensor.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            keyRopeGm.gmTensor.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
#endif
        }
    }

    __aicore__ inline void InitQBuffer(uint32_t batchSize, uint32_t n2Size, uint32_t gSize, uint32_t qSeqSize,
                                       uint32_t headDim, GlobalTensor<uint64_t> actualSeqLengthsGmQ,
                                       uint32_t actualLenQDims, FaGmTensor<Q_T, Q_FORMAT> &qGmTensor,
                                       __gm__ uint8_t *gm)
    {
        qGmTensor.gmTensor.SetGlobalBuffer((__gm__ Q_T *)gm);
        if constexpr (GmLayoutParams<Q_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_BNGSD) {
            qGmTensor.offsetCalculator.Init(batchSize, n2Size, gSize, qSeqSize, headDim, actualSeqLengthsGmQ,
                                            actualLenQDims);
        } else if constexpr (GmLayoutParams<Q_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_TND) {
            qGmTensor.offsetCalculator.Init(n2Size, gSize, headDim, actualSeqLengthsGmQ, actualLenQDims);
        }
    }

    __aicore__ inline void InitKVBuffer(uint32_t batchSize, uint32_t kvSeqSize,
                                        GlobalTensor<uint64_t> actualSeqLengthsGmQ, uint32_t actualLenDims,
                                        uint32_t n2Size, uint32_t kvCacheBlockSize, uint32_t headDim,
                                        FaGmTensor<KV_T, KV_FORMAT> &kvGmTensor, __gm__ uint8_t *gm, uint32_t bnStride,
                                        uint32_t n2Stride)
    {
        kvGmTensor.gmTensor.SetGlobalBuffer((__gm__ KV_T *)gm);

        if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_BNBD) {
            kvGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, headDim, blockTableGm,
                                             constInfo.maxBlockNumPerBatch, bnStride, n2Stride);
        } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_NZ) {
            constexpr uint32_t d0 = 32 / sizeof(KV_T);
            uint32_t d1 = headDim / d0;
            kvGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, d1, d0, blockTableGm,
                                             constInfo.maxBlockNumPerBatch, bnStride, n2Stride);
        } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_BNSD) {
            kvGmTensor.offsetCalculator.Init(batchSize, n2Size, kvSeqSize, headDim, actualSeqLengthsGm, actualLenDims);
        } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_TND) {
            kvGmTensor.offsetCalculator.Init(n2Size, headDim, actualSeqLengthsGm, actualLenDims);
        }
    }

    __aicore__ inline void AllocEventID()
    {
        // InitBuffers阶段已完成eventId申请和SetFlag，这里为空实现
    }

    __aicore__ inline void FreeEventID()
    {
        l1QBuffers.Uninit((*l1BufferManagerPtr));
        mmL0ABuffers.Uninit(l0aBufferManager);
        mmL0BBuffers.Uninit(l0bBufferManager);
        mmL0CBuffers.Uninit(l0cBufferManager);
    }

    // copy query and query_rope, query和query_rope在GM上分开存储, L1上合并存储, D方向全部拷入
    __aicore__ inline void CopyQueryAndRopeTile(const LocalTensor<Q_T> &dstTensor, RunInfoX &runInfo)
    {
        uint32_t dstStride = (runInfo.actMSize + 15) >> 4 << 4;
        {
            FaL1Tensor<Q_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};

            GmCoord gmCoord{.bIdx = runInfo.bIdx,
                            .n2Idx = runInfo.n2Idx,
                            .gS1Idx = runInfo.gS1Idx,
                            .dIdx = 0,
                            .gS1DealSize = runInfo.actMSize,
                            .dDealSize = (uint32_t)constInfo.dSize};
            copyQueryGmToL1(l1Tensor, queryGm, gmCoord);
        }

        {
            uint32_t queryRopeL1Offset = constInfo.dSize * dstStride;
            FaL1Tensor<Q_T, L1Format::NZ> l1Tensor{.tensor = dstTensor[queryRopeL1Offset], .rowCount = dstStride};

            GmCoord gmCoord{.bIdx = runInfo.bIdx,
                            .n2Idx = runInfo.n2Idx,
                            .gS1Idx = runInfo.gS1Idx,
                            .dIdx = 0,
                            .gS1DealSize = runInfo.actMSize,
                            .dDealSize = (uint32_t)constInfo.dSizeRope};
            copyQueryGmToL1(l1Tensor, queryRopeGm, gmCoord);
        }
    }

    // copy key/value and key_rope
    // MLA场景下, key and value是同一份数据, rope在GM上与key/value分开存储, L1上合并存储, D方向全部拷入
    __aicore__ inline void CopyKeyAndRopeTile(const LocalTensor<KV_T> &dstTensor, RunInfoX &runInfo)
    {
        uint32_t dstStride = (runInfo.actSingleLoopS2Size + 15) >> 4 << 4;
        {
            FaL1Tensor<KV_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};

            GmKvCoord gmCoord{.bIdx = runInfo.bIdx,
                              .n2Idx = runInfo.n2Idx,
                              .s2Idx = runInfo.s2Idx,
                              .dIdx = 0,
                              .s2DealSize = runInfo.actSingleLoopS2Size,
                              .dDealSize = (uint32_t)constInfo.dSize};
            copyKvGmToL1(l1Tensor, keyGm, gmCoord);
        }

        {
            uint32_t keyRopeL1Offset = constInfo.dSize * dstStride;
            FaL1Tensor<KV_T, L1Format::NZ> l1Tensor{.tensor = dstTensor[keyRopeL1Offset], .rowCount = dstStride};

            GmKvCoord gmCoord{.bIdx = runInfo.bIdx,
                              .n2Idx = runInfo.n2Idx,
                              .s2Idx = runInfo.s2Idx,
                              .dIdx = 0,
                              .s2DealSize = runInfo.actSingleLoopS2Size,
                              .dDealSize = (uint32_t)constInfo.dSizeRope};
            copyKvGmToL1(l1Tensor, keyRopeGm, gmCoord);
        }
    }

    __aicore__ inline void FixpipeMm1(const LocalTensor<T> &dstTensor, const LocalTensor<T> &l0C, RunInfoX &runInfo)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        // L0C上的bmm1结果矩阵N方向的size大小, 使能NZ2ND, nSize*sizeof(T) 必须是32B的倍数
        fixpipeParams.nSize = (runInfo.actSingleLoopS2Size + 7) >> 3 << 3;
        // 有效数据不足16行，只需输出部分行即可;L0C上的bmm1结果矩阵M方向的size大小必须是偶数
        fixpipeParams.mSize = (runInfo.actMSize + 1) >> 1 << 1;
        // L0C上matmul结果相邻连续数据片断间隔（前面一个数据块的头与后面数据块的头的间隔），单位为16 *sizeof(T)
        // 源NZ矩阵中相邻Z排布的起始地址偏移
        fixpipeParams.srcStride = (fixpipeParams.mSize + 15) >> 4 << 4;
        fixpipeParams.dstStride = s2BaseSize; // mmResUb上两行之间的间隔，单位：element
        fixpipeParams.dualDstCtl = 1; // 双目标模式，按M维度拆分， M / 2 * N写入每个UB，M必须为2的倍数
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;

        Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(dstTensor, l0C, fixpipeParams);
    }

    __aicore__ inline void IterateBmm1(MM1_DBUF_T &outputBuf, KV_BUF_T &kvpSharedBuf, RunInfoX &runInfo)
    {
        outputBuf.WaitCrossCore();

        Buffer<BufferType::L1> mm1A;
        if (unlikely(runInfo.isFirstS2Loop)) {
            mm1A = l1QBuffers.Get();
            mm1A.Wait<HardEvent::MTE1_MTE2>();
            LocalTensor<Q_T> mm1ATensor = mm1A.GetTensor<Q_T>();
            CopyQueryAndRopeTile(mm1ATensor, runInfo);
        } else {
            mm1A = l1QBuffers.GetPre();
        }
        mm1A.Set<HardEvent::MTE2_MTE1>();

        WaitFlag<HardEvent::MTE1_MTE2>(kvpSharedBuf.GetEventID<HardEvent::MTE1_MTE2>());
        LocalTensor<KV_T> mm1BTensor = kvpSharedBuf.GetTensor<KV_T>();
        CopyKeyAndRopeTile(mm1BTensor, runInfo);
        SetFlag<HardEvent::MTE2_MTE1>(kvpSharedBuf.GetEventID<HardEvent::MTE2_MTE1>());
        mm1A.Wait<HardEvent::MTE2_MTE1>();
        WaitFlag<HardEvent::MTE2_MTE1>(kvpSharedBuf.GetEventID<HardEvent::MTE2_MTE1>());
        {
            Buffer<BufferType::L0C> mm1ResL0C = mmL0CBuffers.Get();
            mm1ResL0C.Wait<HardEvent::FIX_M>();

            MMParam mmParam = MakeMMParam((uint32_t)runInfo.actMSize, (uint32_t)runInfo.actSingleLoopS2Size,
                                          (uint32_t)(constInfo.dSize + constInfo.dSizeRope), false, true);

            MatmulK<Q_T, KV_T, T, mBaseSize, s2BaseSize, 128, ABLayout::MK, ABLayout::KN>(
                mm1A.GetTensor<Q_T>(), kvpSharedBuf.GetTensor<KV_T>(), mmL0ABuffers, mmL0BBuffers,
                mm1ResL0C.GetTensor<T>(), mmParam);

            mm1ResL0C.Set<HardEvent::M_FIX>();  // 通知
            mm1ResL0C.Wait<HardEvent::M_FIX>(); // 等待L0C
            FixpipeMm1(outputBuf.template GetTensor<T>(), mm1ResL0C.GetTensor<T>(), runInfo);
            mm1ResL0C.Set<HardEvent::FIX_M>();
        }

        if (unlikely(runInfo.isLastS2Loop)) {
            mm1A.Set<HardEvent::MTE1_MTE2>();
        }

        outputBuf.SetCrossCore();
    }

    template <typename DST_TENSOR_T>
    __aicore__ inline void FixpipeMm2PartialN(const DST_TENSOR_T &dstTensor, const LocalTensor<T> &l0C, uint32_t realN,
                                              RunInfoX &runInfo)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams; // L0C→UB;FixpipeParamsM300:L0C→UB
        fixpipeParams.nSize = (realN + 7) >> 3 << 3;
        fixpipeParams.mSize = (runInfo.actMSize + 1) >> 1 << 1;
        fixpipeParams.srcStride = (fixpipeParams.mSize + 15) >> 4 << 4;
        fixpipeParams.dstStride = ((uint32_t)dVTemplateType + 15) >> 4 << 4;
        fixpipeParams.dualDstCtl = 1;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        Fixpipe<T, T, BMM2_FIXPIPE_CONFIG>(dstTensor, l0C, fixpipeParams);
    }

    __aicore__ inline void IterateBmm2(mm2ResPos &outputBuf, KV_BUF_T &kvpSharedBuf, RunInfoX &runInfo)
    {
        kvpSharedBuf.WaitCrossCore();
        outputBuf.WaitCrossCore();
        {
            uint64_t VDsize = (uint32_t)dVTemplateType;
            Buffer<BufferType::L0C> mm2ResL0C = mmL0CBuffers.Get();
            mm2ResL0C.Wait<HardEvent::FIX_M>(); // 占用
            MMParam param =
                MakeMMParam((uint32_t)mBaseSize, (uint32_t)constInfo.dSizeV, (uint32_t)runInfo.actSingleLoopS2Size,
                            false, false, true, true, 0, (uint32_t)runInfo.actMSize);

            MatmulN<Q_T, KV_T, T, mBaseSize, 128, s2BaseSize, ABLayout::MK, ABLayout::KN>(
                kvpSharedBuf.GetTensor<INPUT_T>(s2BaseSize * VDsize), kvpSharedBuf.GetTensor<INPUT_T>(), mmL0ABuffers,
                mmL0BBuffers, mm2ResL0C.GetTensor<T>(), param);

            mm2ResL0C.Set<HardEvent::M_FIX>();  // 通知
            mm2ResL0C.Wait<HardEvent::M_FIX>(); // 等待
            FixpipeMm2PartialN(outputBuf.template GetTensor<T>(), mm2ResL0C.GetTensor<T>(), constInfo.dSizeV, runInfo);
            mm2ResL0C.Set<HardEvent::FIX_M>(); // 释放
        }
        SetFlag<HardEvent::MTE1_MTE2>(kvpSharedBuf.GetEventID<HardEvent::MTE1_MTE2>());

        outputBuf.SetCrossCore();
    }
}; // FANoQuantMlaBlockCube

template <typename INPUT_T, typename T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, uint8_t KvLayoutType = 0>
class FANoQuantMlaBlockCubeDummy {
public:
    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t dBaseSize = (uint32_t)dTemplateType;
    static constexpr uint32_t dVBaseSize = (uint32_t)dVTemplateType;
    static constexpr LayOutTypeEnum LAYOUT = layout;
    static constexpr bool PAGE_ATTENTION = (KvLayoutType > 0);

    using ROPE_T = INPUT_T;
    using Q_T = INPUT_T;
    using KV_T = INPUT_T;
    using MM_T = T;
    using mm2ResPos = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    using MM1_DBUF_T = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;

    using ConstInfoX = ConstInfo_t<FiaKernelType::NO_QUANT>;
    __aicore__ inline FANoQuantMlaBlockCubeDummy(ConstInfoX &constInfo){};
};

} // namespace BaseApi

#endif // FIA_BLOCK_CUBE_NOQUANT_MLA_H_
