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
 * \file quant_flash_attn_block_cube_hif8.h
 * \brief
 */
#ifndef QUANT_FLASH_ATTN_BLOCK_CUBE_HIF8_H_
#define QUANT_FLASH_ATTN_BLOCK_CUBE_HIF8_H_
#if __has_include("../../../common/op_kernel/offset_calculator.h")
#include "../../../common/op_kernel/offset_calculator.h"
#include "../../../common/op_kernel/matmul.h"
#include "../../../common/op_kernel/FixpipeOut.h"
#include "../../../common/op_kernel/arch35/infer_flash_attention_comm_arch35.h"
#include "../../../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h"
#include "../../../common/op_kernel/arch35/util_regbase.h"
#else
#include "../../common/op_kernel/offset_calculator.h"
#include "../../common/op_kernel/matmul.h"
#include "../../common/op_kernel/FixpipeOut.h"
#include "../../common/op_kernel/arch35/infer_flash_attention_comm_arch35.h"
#include "../../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h"
#include "../../common/op_kernel/arch35/util_regbase.h"
#endif
#include "kernel_operator_list_tensor_intf.h"
#include "memory_copy_arch35_quant_flash_attn.h"
using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace fa_base_matmul;
using namespace AttentionCommon;

namespace BaseApi {

template <LayOutTypeEnum LAYOUT, uint8_t kvLayoutType = 0, bool isPa = false>
__aicore__ inline constexpr GmFormat GetKVGmFormatHif8()
{
    if constexpr (kvLayoutType == 0) {
        if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BSH) {
            return GmFormat::BSND;
        } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_SBH) {
            return GmFormat::SBND;
        } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BNSD) {
            return GmFormat::BNSD;
        } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_TND) {
            return GmFormat::TND;
        } else {
            return GmFormat::NTD;
        }
    } else if constexpr (kvLayoutType == 1) {
        return GmFormat::PA_BnBsND;
    } else if constexpr (kvLayoutType == 2) {
        return GmFormat::PA_BnNBsD;
    } else {
        return GmFormat::PA_NZ;
    }
}

template <typename INPUT_T, typename T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, uint8_t KvLayoutType = 0, bool useDn = false>
class QuantFlashAttnBlockCubeHif8 {
public:
    /* ============确定L0A的类型============= */
    struct L0ABuffSel {
        using Type = BuffersPolicyDB<BufferType::L0A>;
    };
    /* ============确定L0B的类型============= */
    template <uint32_t s2BaseSize, uint32_t dBaseSize>
    struct L0BBuffSel {
        using Type = std::conditional_t<(s2BaseSize == 256 && dBaseSize > 128),
                                        BuffersPolicySingleBuffer<BufferType::L0B>, BuffersPolicyDB<BufferType::L0B>>;
    };

    /* ============确定L0C的类型============= */
    template <uint32_t mBaseSize, uint32_t s2BaseSize, uint32_t dVBaseSize>
    struct L0CBuffSel {
        using Type = std::conditional_t<(mBaseSize * s2BaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4 &&
                                         mBaseSize * dVBaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4),
                                        BuffersPolicy4buff<BufferType::L0C>, BuffersPolicyDB<BufferType::L0C>>;
    };

    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t dBaseSize = (uint32_t)dTemplateType;
    static constexpr uint32_t dVBaseSize = (uint32_t)dVTemplateType;
    static constexpr uint32_t l1BaseD = 128;
    static constexpr uint32_t MXFP_DIVISOR_SIZE = 64U;
    static constexpr LayOutTypeEnum LAYOUT = layout;
    static constexpr bool PAGE_ATTENTION = (KvLayoutType > 0);
    static constexpr bool USE_DN = useDn;

    static constexpr FixpipeConfig BMM2_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, true};
    static constexpr GmFormat Q_FORMAT = GetQueryGmFormat<layout>();
    static constexpr GmFormat KV_FORMAT = GetKVGmFormatHif8<layout, KvLayoutType, PAGE_ATTENTION>();

    using Q_T = INPUT_T;
    using KV_T = INPUT_T;
    using MM_T = T;
    using MM1_DBUF_T = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    using MM2_ABUF_POLICY_T = BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>;
    using MM2_ABUF_T = Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>;
    using MM2_DBUF_T = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;

    using L1KvType = BuffersPolicy4buff<BufferType::L1>;
    using L0AType = typename L0ABuffSel::Type;
    using L0BType = typename L0BBuffSel<s2BaseSize, dBaseSize>::Type;
    using L0CType = typename L0CBuffSel<mBaseSize, s2BaseSize, dVBaseSize>::Type;

    static constexpr bool IS_TND_LAYOUT =
        (LAYOUT == LayOutTypeEnum::LAYOUT_TND || LAYOUT == LayOutTypeEnum::LAYOUT_NTD);
    static constexpr bool Q_NEEDS_WZH = (GmLayoutParams<Q_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_TND);
    static constexpr bool KV_NEEDS_WZH = (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_TND);

    static constexpr ActualSeqLensMode Q_PARSER_MODE = GetQActSeqMode<LAYOUT>();
    static constexpr ActualSeqLensMode KV_PARSER_MODE = GetKvActSeqMode<LAYOUT, PAGE_ATTENTION>();

    using QSeqParserType = typename std::conditional<IS_TND_LAYOUT, ActualSeqLensParser<Q_PARSER_MODE, int32_t, true>,
                                                     ActualSeqLensParser<Q_PARSER_MODE, int32_t>>::type;

    using KvSeqParserType = typename std::conditional<(!PAGE_ATTENTION && IS_TND_LAYOUT),
                                                      ActualSeqLensParser<KV_PARSER_MODE, int32_t, true>,
                                                      ActualSeqLensParser<KV_PARSER_MODE, int32_t>>::type;

    using FaGmTensorQ = FaGmTensor<Q_T, Q_FORMAT, int32_t, Q_NEEDS_WZH>;
    using FaGmTensorKV = FaGmTensor<KV_T, KV_FORMAT, int32_t, KV_NEEDS_WZH>;

    using ConstInfoX = ConstInfo_t;
    TPipe *tPipe_ = nullptr;
    /* =====================GM变量(with layout)==================== */
    FaGmTensorQ queryGm_;
    FaGmTensorKV keyGm_;
    FaGmTensorKV valueGm_;
    GlobalTensor<int32_t> blockTableGm_;

    QSeqParserType *qSeqParserPtr_ = nullptr;
    KvSeqParserType *kvSeqParserPtr_ = nullptr;

    CopyQueryGmToL1<Q_T, Q_FORMAT> copyQueryGmToL1_;
    CopyKvGmToL1<KV_T, KV_FORMAT> copyKvGmToL1_;

    /* =====================LocalBuffer变量====================*/
    BufferManager<BufferType::L1> *l1BufferManagerPtr_;
    BufferManager<BufferType::L0A> l0aBufferManager_;
    BufferManager<BufferType::L0B> l0bBufferManager_;
    BufferManager<BufferType::L0C> l0cBufferManager_;

    BuffersPolicyDB<BufferType::L1> l1QBuffers_;
    L1KvType l1KVBuffers_;

    L0AType mmL0ABuffers_;
    L0BType mmL0BBuffers_;
    L0CType mmL0CBuffers_;

    __gm__ uint8_t *keyPtr_ = nullptr;
    __gm__ uint8_t *valuePtr_ = nullptr;

    const ConstInfoX &constInfo_;
    LocalTensor<float> dummyScale_;

    /*============================================================================== */
    __aicore__ inline QuantFlashAttnBlockCubeHif8(ConstInfoX &constInfo)
        : constInfo_(constInfo){};

    __aicore__ inline void InitCubeBlock(TPipe *pipe, BufferManager<BufferType::L1> *l1BuffMgr, __gm__ uint8_t *query,
                                         __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *blockTable,
                                         __gm__ uint8_t *dequantScaleQuery, __gm__ uint8_t *dequantScaleKey,
                                         __gm__ uint8_t *dequantScaleValue, QSeqParserType &qParser,
                                         KvSeqParserType &kvParser)
    {
        tPipe_ = pipe;
        l1BufferManagerPtr_ = l1BuffMgr;
        this->qSeqParserPtr_ = &qParser;
        this->kvSeqParserPtr_ = &kvParser;
        InitCubeInput(query, key, value, blockTable, dequantScaleQuery, dequantScaleKey, dequantScaleValue);
    }

    __aicore__ inline void InitBuffers()
    {
        constexpr uint32_t s2BaseSizeCur = s2BaseSize;
        constexpr uint32_t mm1QSize = mBaseSize * dBaseSize * sizeof(INPUT_T);
        constexpr uint32_t mmKVSize = dBaseSize * s2BaseSizeCur * sizeof(INPUT_T);

        l1QBuffers_.Init((*l1BufferManagerPtr_), mm1QSize);
        l1KVBuffers_.Init((*l1BufferManagerPtr_), mmKVSize);

        // L0A B C 当前写死，能否通过基础api获取
        l0aBufferManager_.Init(tPipe_, 65536);  // 64 * 1024
        l0bBufferManager_.Init(tPipe_, 65536);  // 64 * 1024
        l0cBufferManager_.Init(tPipe_, 262144); // 256 * 1024
        // L0A B C当前写死，要改成通过计算获取
        mmL0ABuffers_.Init(l0aBufferManager_, 32 * 1024);
        mmL0BBuffers_.Init(l0bBufferManager_, 32 * 1024);

        if constexpr (mBaseSize * s2BaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4 &&
                      mBaseSize * dVBaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4) {
            mmL0CBuffers_.Init(l0cBufferManager_, (L0C_SIZE / NUM_4) * KB_TO_BYTES);
        } else {
            mmL0CBuffers_.Init(l0cBufferManager_, (L0C_SIZE / NUM_2) * KB_TO_BYTES);
        }
    }

    __aicore__ inline void InitCubeInput(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                         __gm__ uint8_t *blockTable, __gm__ uint8_t *dequantScaleQuery,
                                         __gm__ uint8_t *dequantScaleKey, __gm__ uint8_t *dequantScaleValue)
    {
        if constexpr (PAGE_ATTENTION) {
            blockTableGm_.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        }

        InitQBuffer(constInfo_.bSize, constInfo_.realN2Size, constInfo_.realGSize, constInfo_.s1Size, constInfo_.dSize,
                    queryGm_, query);

        keyPtr_ = key;
        valuePtr_ = value;
        InitKVBuffer(constInfo_.bSize, constInfo_.s2Size, constInfo_.n2Size, constInfo_.blockSize, constInfo_.dSize,
                     keyGm_, key);
        InitKVBuffer(constInfo_.bSize, constInfo_.s2Size, constInfo_.n2Size, constInfo_.blockSize, constInfo_.dSizeV,
                     valueGm_, value);
    }

    __aicore__ inline void InitQBuffer(uint32_t batchSize, uint32_t n2Size, uint32_t gSize, uint32_t qSeqSize,
                                       uint32_t headDim, FaGmTensorQ &qGmTensor, __gm__ uint8_t *gm)
    {
        qGmTensor.gmTensor.SetGlobalBuffer((__gm__ Q_T *)gm);
        if constexpr (Q_NEEDS_WZH) {
            qGmTensor.offsetCalculator.Init(n2Size, gSize, headDim, *this->qSeqParserPtr_);
        } else {
            qGmTensor.offsetCalculator.Init(batchSize, n2Size, gSize, qSeqSize, headDim, *this->qSeqParserPtr_);
        }
    }

    __aicore__ inline void InitKVBuffer(uint32_t batchSize, uint32_t kvSeqSize, uint32_t n2Size,
                                        uint32_t kvCacheBlockSize, uint32_t headDim, FaGmTensorKV &kvGmTensor,
                                        __gm__ uint8_t *gm)
    {
        kvGmTensor.gmTensor.SetGlobalBuffer((__gm__ KV_T *)gm);
        if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_BNBD) {
            kvGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, headDim, blockTableGm_,
                                             constInfo_.maxBlockNumPerBatch);
        } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_NZ) {
            constexpr uint32_t d0 = 32 / sizeof(KV_T);
            uint32_t d1 = headDim / d0;
            kvGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, d1, d0, blockTableGm_,
                                             constInfo_.maxBlockNumPerBatch);
        } else if constexpr (KV_NEEDS_WZH) {
            kvGmTensor.offsetCalculator.Init(n2Size, headDim, *this->kvSeqParserPtr_);
        } else {
            kvGmTensor.offsetCalculator.Init(batchSize, n2Size, kvSeqSize, headDim, *this->kvSeqParserPtr_);
        }
    }

    __aicore__ inline void AllocEventID()
    {
        // InitBuffers阶段已完成eventId申请和SetFlag，这里为空实现
    }

    __aicore__ inline void FreeEventID()
    {
        l1QBuffers_.Uninit((*l1BufferManagerPtr_));
        l1KVBuffers_.Uninit((*l1BufferManagerPtr_));
        mmL0ABuffers_.Uninit(l0aBufferManager_);
        mmL0BBuffers_.Uninit(l0bBufferManager_);
        mmL0CBuffers_.Uninit(l0cBufferManager_);
    }

    // copy query with full s1g
    __aicore__ inline void CopyQuerySlice(const LocalTensor<Q_T> &dstTensor, uint32_t dOffset, uint32_t dRealSize,
                                          RunInfoX &runInfo)
    {
        constexpr uint32_t blockNumDtype = 32 / sizeof(Q_T);
        uint32_t nopeDealSize = dRealSize;
        uint32_t dstStride = (runInfo.actMSize + 31) >> 5 << 5;
        FaL1Tensor<Q_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};

        GmCoord gmCoord{.bIdx = runInfo.bIdx,
                        .n2Idx = runInfo.realN2Idx,
                        .gS1Idx = runInfo.gS1Idx,
                        .dIdx = dOffset,
                        .gS1DealSize = runInfo.actMSize,
                        .dDealSize = nopeDealSize};
        copyQueryGmToL1_(l1Tensor, queryGm_, gmCoord);
    }

    __aicore__ inline void CopyQueryTile(const LocalTensor<Q_T> &dstTensor, RunInfoX &runInfo)
    {
        CopyQuerySlice(dstTensor, 0, constInfo_.dSize, runInfo);
    }

    // copy key with full s2
    __aicore__ inline void CopyKeySlice(const LocalTensor<KV_T> &dstTensor, uint32_t s2Offset, uint32_t s2RealSize,
                                        uint32_t dOffset, uint32_t dRealSize, RunInfoX &runInfo)
    {
        constexpr uint32_t blockNumDtype = 32 / sizeof(KV_T);
        uint32_t dstStride = (s2RealSize + 31) >> 5 << 5;

        uint32_t nopeDealSize = dRealSize;
        FaL1Tensor<KV_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};
        // CopyKey的控制参数
        GmKvCoord gmCoord{.bIdx = runInfo.bIdx,
                          .n2Idx = runInfo.n2Idx,
                          .s2Idx = s2Offset,
                          .dIdx = dOffset,
                          .s2DealSize = s2RealSize,
                          .dDealSize = nopeDealSize};
        copyKvGmToL1_(l1Tensor, keyGm_, gmCoord);
    }

    __aicore__ inline void CopyKeyTile(const LocalTensor<KV_T> &dstTensor, RunInfoX &runInfo, uint32_t s2RealSize)
    {
        CopyKeySlice(dstTensor, runInfo.s2Idx, s2RealSize, 0, constInfo_.dSize, runInfo);
    }

    // copy key with full s2
    __aicore__ inline void CopyValueSlice(const LocalTensor<KV_T> &dstTensor, uint32_t s2Offset, uint32_t s2RealSize,
                                          uint32_t dOffset, uint32_t dRealSize, RunInfoX &runInfo)
    {
        uint32_t dstStride = (s2RealSize + 63) >> 6 << 6;
        FaL1Tensor<KV_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};

        GmKvCoord gmCoord{.bIdx = runInfo.bIdx,
                          .n2Idx = runInfo.n2Idx,
                          .s2Idx = s2Offset,
                          .dIdx = dOffset,
                          .s2DealSize = s2RealSize,
                          .dDealSize = dRealSize};
        copyKvGmToL1_(l1Tensor, valueGm_, gmCoord);
    }

    __aicore__ inline void CopyValueTile(const LocalTensor<KV_T> &dstTensor, RunInfoX &runInfo)
    {
        CopyValueSlice(dstTensor, runInfo.s2Idx, runInfo.actSingleLoopS2Size, 0, constInfo_.dSizeV, runInfo);
    }

    __aicore__ inline void IterateBmm1(MM1_DBUF_T &outputBuf, RunInfoX &runInfo)
    {
        IterateBmm1Dn(outputBuf, runInfo);
    }

    __aicore__ inline void FixpipeMm1(const LocalTensor<T> &dstTensor, const LocalTensor<T> &l0C, RunInfoX &runInfo,
                                      uint32_t s2RealSize)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        // L0C上的bmm1结果矩阵N方向的size大小, 使能NZ2ND, nSize*sizeof(T) 必须是32B的倍数
        fixpipeParams.nSize = (s2RealSize + 7) >> 3 << 3;
        // 有效数据不足16行，只需输出部分行即可;L0C上的bmm1结果矩阵M方向的size大小必须是偶数
        fixpipeParams.mSize = (runInfo.actMSize + 31) >> 5 << 5;
        // L0C上matmul结果相邻连续数据片断间隔（前面一个数据块的头与后面数据块的头的间隔），单位为16 *sizeof(T)
        // 源NZ矩阵中相邻Z排布的起始地址偏移
        fixpipeParams.srcStride = ((runInfo.actMSize + 15) / 16) * 16;
        fixpipeParams.dstStride = s2BaseSize; // mmResUb上两行之间的间隔，单位：element
        fixpipeParams.dualDstCtl = 1; // 双目标模式，按M维度拆分， M / 2 * N写入每个UB，M必须为2的倍数
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(dstTensor, l0C, fixpipeParams);
    }

    __aicore__ inline void FixpipeMm1Dn(const LocalTensor<T> &dstTensor, const LocalTensor<T> &l0C, RunInfoX &runInfo,
                                        uint32_t s2RealSize)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        // L0C上的bmm1结果矩阵N方向的size大小, 使能NZ2ND, nSize*sizeof(T) 必须是32B的倍数
        fixpipeParams.nSize = (runInfo.actMSize + 31) >> 5 << 5;
        // 有效数据不足16行，只需输出部分行即可;L0C上的bmm1结果矩阵M方向的size大小必须是偶数
        fixpipeParams.mSize = (s2RealSize + 7) >> 3 << 3;
        // L0C上matmul结果相邻连续数据片断间隔（前面一个数据块的头与后面数据块的头的间隔），单位为16 *sizeof(T)
        // 源NZ矩阵中相邻Z排布的起始地址偏移
        fixpipeParams.srcStride = ((fixpipeParams.mSize + 15) / 16) * 16;
        fixpipeParams.dstStride = 64; // mmResUb上两行之间的间隔，单位：element
        fixpipeParams.dualDstCtl = 2; // 双目标模式，按M维度拆分， M * N / 2写入每个UB，M必须为2的倍数
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(dstTensor, l0C, fixpipeParams);
    }

    /* 针对S1Base=128, S2Base = 256, D = 128场景, L1全载/L0全载, K * Q^T, Q矩阵驻留. GS1>80, S=S1*S2 */
    __aicore__ inline void IterateBmm1Dn(MM1_DBUF_T &outputBuf, RunInfoX &runInfo)
    {
        uint32_t s2CalcSize = runInfo.actSingleLoopS2Size;
        Buffer<BufferType::L1> mm1B;
        if (unlikely(runInfo.isFirstS2Loop)) {
            mm1B = l1QBuffers_.Get();
            mm1B.Wait<HardEvent::MTE1_MTE2>();
            LocalTensor<Q_T> mm1BTensor = mm1B.GetTensor<Q_T>();
            CopyQueryTile(mm1BTensor, runInfo);
            mm1B.Set<HardEvent::MTE2_MTE1>();
        } else {
            mm1B = l1QBuffers_.GetPre();
            mm1B.Set<HardEvent::MTE2_MTE1>();
        }

        Buffer<BufferType::L1> mm1A = l1KVBuffers_.Get();
        mm1A.Wait<HardEvent::MTE1_MTE2>();
        LocalTensor<KV_T> mm1ATensor = mm1A.GetTensor<KV_T>();
        CopyKeyTile(mm1ATensor, runInfo, s2CalcSize);
        mm1A.Set<HardEvent::MTE2_MTE1>();
        mm1A.Wait<HardEvent::MTE2_MTE1>();
        mm1B.Wait<HardEvent::MTE2_MTE1>();

        Buffer<BufferType::L0C> mm1ResL0C = mmL0CBuffers_.Get();
        mm1ResL0C.Wait<HardEvent::FIX_M>();
        MMParam param =
            MakeMMParam((uint32_t)s2CalcSize, (uint32_t)runInfo.actMSize, (uint32_t)constInfo_.dSize, false, true);
        MatmulFullMX<Q_T, KV_T, T, 256, 128, dBaseSize, ABLayout::MK, ABLayout::KN, L0AType, L0BType, float, float,
                     INPUT_T, INPUT_T>(mm1A.GetTensor<INPUT_T>(), mm1B.GetTensor<INPUT_T>(), mmL0ABuffers_,
                                       mmL0BBuffers_, mm1ResL0C.GetTensor<T>(), param);

        if (unlikely(runInfo.isLastS2Loop)) {
            mm1B.Set<HardEvent::MTE1_MTE2>();
        }
        mm1A.Set<HardEvent::MTE1_MTE2>();
        mm1ResL0C.Set<HardEvent::M_FIX>();
        mm1ResL0C.Wait<HardEvent::M_FIX>();

        outputBuf.WaitCrossCore();
        FixpipeMm1Dn(outputBuf.template GetTensor<T>(), mm1ResL0C.GetTensor<T>(), runInfo, s2CalcSize);
        mm1ResL0C.Set<HardEvent::FIX_M>();
        outputBuf.SetCrossCore();
    }

    __aicore__ inline void IterateBmm2(MM2_DBUF_T &outputBuf, MM2_ABUF_POLICY_T &inputBuf, RunInfoX &runInfo)
    {
        IterateBmm2Dn(outputBuf, inputBuf, runInfo);
    }

    template <typename DST_TENSOR_T>
    __aicore__ inline void FixpipeMm2(const DST_TENSOR_T &dstTensor, const LocalTensor<T> &l0C, RunInfoX &runInfo)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = (constInfo_.dSizeV + 7) >> 3 << 3;

        fixpipeParams.mSize = mBaseSize;
        fixpipeParams.srcStride = ((mBaseSize + 15) / 16) * 16;

        fixpipeParams.dstStride = ((uint32_t)dVTemplateType + 15) >> 4 << 4;
        fixpipeParams.dualDstCtl = 1;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        Fixpipe<T, T, BMM2_FIXPIPE_CONFIG>(dstTensor, l0C, fixpipeParams);
    }

    __aicore__ inline void IterateBmm2Dn(MM2_DBUF_T &outputBuf, MM2_ABUF_POLICY_T &inputBuf, RunInfoX &runInfo)
    {
        MM2_ABUF_T mm2A = inputBuf.Get();
        mm2A.WaitCrossCore();

        Buffer<BufferType::L0C> mm2ResL0C = mmL0CBuffers_.Get();
        mm2ResL0C.Wait<HardEvent::FIX_M>();

        constexpr uint32_t baseK = s2BaseSize;
        uint32_t s2RealSizeAlign = ((runInfo.actSingleLoopS2Size + 63) >> 6) << 6;
        uint64_t l1BaseKOffset = s2RealSizeAlign * mBaseSize;
        uint32_t kLoops = (runInfo.actSingleLoopS2Size + baseK - 1) / baseK;
        uint32_t realK = baseK;
        for (uint32_t k = 0; k < kLoops; k++) {
            Buffer<BufferType::L1> mm2B = l1KVBuffers_.Get();
            if (k == kLoops - 1) {
                realK = runInfo.actSingleLoopS2Size - k * baseK;
            }
            mm2B.Wait<HardEvent::MTE1_MTE2>();
            LocalTensor<KV_T> mm2BTensor = mm2B.GetTensor<KV_T>();
            if (unlikely(runInfo.isLastS2Loop)) {
                InitValueL1BufferNoTrans(mm2BTensor, realK, (uint32_t)constInfo_.dSizeV);
            }
            CopyValueSlice(mm2BTensor, runInfo.s2Idx + k * baseK, realK, 0, constInfo_.dSizeV, runInfo);
            mm2B.Set<HardEvent::MTE2_MTE1>();

            mm2B.Wait<HardEvent::MTE2_MTE1>();
            MMParam param =
                MakeMMParam((uint32_t)mBaseSize, (uint32_t)constInfo_.dSizeV,
                            (realK + 63) / MXFP_DIVISOR_SIZE * MXFP_DIVISOR_SIZE, USE_DN, false, k == 0, k == 0);
            MatmulFullMX<INPUT_T, KV_T, T, 128, dVBaseSize, baseK, ABLayout::MK, ABLayout::KN, L0AType, L0BType, float,
                         float, INPUT_T, INPUT_T>(mm2A.GetTensor<INPUT_T>()[k * l1BaseKOffset], mm2BTensor,
                                                  mmL0ABuffers_, mmL0BBuffers_, mm2ResL0C.GetTensor<T>(), param,
                                                  dummyScale_, dummyScale_);
            mm2B.Set<HardEvent::MTE1_MTE2>();
        }

        mm2ResL0C.Set<HardEvent::M_FIX>();
        mm2ResL0C.Wait<HardEvent::M_FIX>();

        outputBuf.WaitCrossCore();
        FixpipeMm2(outputBuf.template GetTensor<T>(), mm2ResL0C.GetTensor<T>(), runInfo);
        mm2ResL0C.Set<HardEvent::FIX_M>();
        outputBuf.SetCrossCore();
    }

    __aicore__ inline void InitValueL1BufferNoTrans(const LocalTensor<KV_T> &valueL1, const uint32_t realK,
                                                    const uint32_t curN)
    {
        InitConstValueParams<half> initConstValueParams;
        uint64_t offset = 0;
        uint32_t curPadK = (realK + 63) >> 6 << 6;
        uint32_t movK = realK;
        if (curPadK == movK) {
            return;
        }
        offset = movK * 16U;
        initConstValueParams.repeatTimes = AttentionCommon::Align(curN, 16U) / 32U;
        initConstValueParams.blockNum = curPadK - movK;
        initConstValueParams.dstGap = movK;
        initConstValueParams.initValue = 0;
        InitConstValue(valueL1.template ReinterpretCast<half>()[offset], initConstValueParams);
    }

}; // QuantFlashAttnBlockCubeHif8

template <typename INPUT_T, typename T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, uint8_t KvLayoutType = 0, bool useDn = false>
class QuantFlashAttnBlockCubeHif8Dummy {
public:
    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t dBaseSize = (uint32_t)dTemplateType;
    static constexpr uint32_t dVBaseSize = (uint32_t)dVTemplateType;
    static constexpr LayOutTypeEnum LAYOUT = layout;
    static constexpr bool PAGE_ATTENTION = (KvLayoutType > 0);
    static constexpr bool USE_DN = useDn;

    using Q_T = INPUT_T;
    using KV_T = INPUT_T;
    using MM_T = T;
    using MM1_DBUF_T = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    using MM2_ABUF_POLICY_T = BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>;
    using MM2_ABUF_T = Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>;
    using MM2_DBUF_T = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;

    using ConstInfoX = ConstInfo_t;
    __aicore__ inline QuantFlashAttnBlockCubeHif8Dummy(ConstInfoX &constInfo){};
};
} // namespace BaseApi

#endif // QUANT_FLASH_ATTN_BLOCK_CUBE_HIF8_H_
