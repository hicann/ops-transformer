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
 * \file quant_flash_attn_block_cube_mxfp8_softmax_fp16.h
 * \brief quant_mode=3 (QFA_MXFP8_SOFTMAX_FP16) cube 层:
 *        基本块 128x512, MM1 FIXPIPE 随路 FP32->FP16 并乘 softmaxscale;
 *        MM2 ND2NZ 方案 (V经ND2NZ搬入L1不转置, L1→L0转置为V^T, 尾行全1随路得出 row_sum)
 */
#ifndef QUANT_FLASH_ATTN_BLOCK_CUBE_MXFP8_SOFTMAX_FP16_H_
#define QUANT_FLASH_ATTN_BLOCK_CUBE_MXFP8_SOFTMAX_FP16_H_
#include "../../common/op_kernel/matmul.h"
#include "../../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h"
#include "../../common/op_kernel/arch35/util_regbase.h"
#include "kernel_operator_list_tensor_intf.h"
#include "utils/memory_copy_arch35.h"
using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace fa_base_matmul;
using namespace AttentionCommon;

// ROW_MAJOR: 使能NZ2ND，输出数据格式为ND格式; true: 用于用户指定目的地址的位置是否是UB
constexpr FixpipeConfig PFA_CFG_ROW_MAJOR_UB = {CO2Layout::ROW_MAJOR, true};

namespace BaseApi {

template <typename INPUT_T, typename T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, uint8_t KvLayoutType = 0>
class QuantFlashAttnBlockCubeMxfp8SoftmaxFp16 {
public:
    struct L0ABuffSel {
        using Type = BuffersPolicyDB<BufferType::L0A>;
    };
    template <uint32_t s2BaseSize, uint32_t dBaseSize>
    struct L0BBuffSel {
        using Type = BuffersPolicyDB<BufferType::L0B>;
    };
    template <uint32_t mBaseSize, uint32_t s2BaseSize, uint32_t dVBaseSize>
    struct L0CBuffSel {
        using Type = BuffersPolicyDB<BufferType::L0C>;
    };

    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t dBaseSize = (uint32_t)dTemplateType;
    static constexpr uint32_t dVBaseSize = (uint32_t)dVTemplateType;
    static constexpr uint32_t s2SplitSize = 256U;
    static constexpr uint32_t MXFP_GROUP_SIZE = 32U;
    static constexpr uint32_t MXFP_DIVISOR_SIZE = 64U;
    static constexpr uint32_t MXFP_MULTI_BASE_SIZE = 2U;
    /* ============MM2 (FP16 softmax) DN方案参数============= */
    // 基本块128x512拆4次进mm2, 每次算(d+32)x128
    static constexpr uint32_t mm2BaseK = 128U;
    // V在L1未转置: s2行×(d+32)列含全1带, L1→L0时转置为(d+32)×K
    static constexpr uint32_t mm2VtMSize = dVBaseSize + 32U;
    static constexpr uint32_t mm2VtSegDataSize = mm2VtMSize * mm2BaseK;
    static constexpr uint32_t mm2VtSegScaleSize = dVBaseSize * mm2BaseK;
    static constexpr LayOutTypeEnum LAYOUT = layout;
    // 仅DN方案(prefill): Q按S1_ONLY布局, realG=1, gIdx恒为0
    static constexpr InnerMLayout Q_M_LAYOUT = InnerMLayout::S1_ONLY_LAYOUT;

    static constexpr FixpipeConfig BMM2_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, true};
    static constexpr GmFormat Q_FORMAT = GetQueryGmFormat<layout>();
    static constexpr GmFormat KV_FORMAT = GetKVGmFormat<layout, KvLayoutType>();
    static constexpr GmFormat Q_SCALE_FORMAT = GetQueryScaleGmFormat<layout, true>();
    static constexpr GmFormat K_SCALE_FORMAT = GetKeyScaleGmFormat<layout, KvLayoutType>();
    static constexpr GmFormat V_SCALE_FORMAT = GetValueScaleGmFormat<layout, KvLayoutType>();

    using Q_T = INPUT_T;
    using KV_T = INPUT_T;
    using SCALE_T = fp8_e8m0_t;
    using MM_T = T;
    using MM1_DBUF_T = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    using MM2_PBUF_POLICY_T = BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>;
    using MM2_PBUF_T = Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>;
    using MM2_DBUF_T = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;

    // K与V^T的L1 tile大小不同(V^T含全1带), 由原KV共享4buff改为各自独立2buff池
    using L1KType = BuffersPolicyDB<BufferType::L1>;
    using L1VtType = BuffersPolicyDB<BufferType::L1>;
    using L0AType = typename L0ABuffSel::Type;
    using L0BType = typename L0BBuffSel<s2BaseSize, dBaseSize>::Type;
    using L0CType = typename L0CBuffSel<mBaseSize, s2BaseSize, dVBaseSize>::Type;

    using KeyScaleGmToL1Type = CopyKeyScaleGmToL1<SCALE_T, K_SCALE_FORMAT, L1Format::NZ, ScaleTrans::DN2NZ>;
    using ValueScaleGmToL1Type = CopyValueScaleGmToL1<SCALE_T, V_SCALE_FORMAT, L1Format::NZ, ScaleTrans::ND2NZ>;

    static constexpr ActualSeqLensMode Q_PARSER_MODE = GetQActSeqMode<LAYOUT>();
    static constexpr ActualSeqLensMode KV_PARSER_MODE = GetKvActSeqMode<LAYOUT, false>();

    using QSeqParserType = ActualSeqLensParser<Q_PARSER_MODE, int32_t>;

    using KvSeqParserType = ActualSeqLensParser<KV_PARSER_MODE, int32_t>;

    using FaGmTensorQ = FaGmTensor<Q_T, Q_FORMAT, int32_t>;
    using FaGmTensorKV = FaGmTensor<KV_T, KV_FORMAT, int32_t>;
    using FaGmTensorQScale = FaGmTensor<SCALE_T, Q_SCALE_FORMAT, int32_t>;
    using FaGmTensorKScale = FaGmTensor<SCALE_T, K_SCALE_FORMAT, int32_t>;
    using FaGmTensorVScale = FaGmTensor<SCALE_T, V_SCALE_FORMAT, int32_t>;

    using ConstInfoX = ConstInfo_t;
    TPipe* tPipe_ = nullptr;
    FaGmTensorQ queryGm_;
    FaGmTensorKV keyGm_;
    FaGmTensorKV valueGm_;
    FaGmTensorQScale queryScaleGm_;
    FaGmTensorKScale keyScaleGm_;
    FaGmTensorVScale valueScaleGm_;

    QSeqParserType* qSeqParserPtr_ = nullptr;
    KvSeqParserType* kvSeqParserPtr_ = nullptr;

    CopyQueryGmToL1<Q_T, Q_FORMAT, L1Format::NZ, Q_M_LAYOUT> copyQueryGmToL1_;
    CopyKvGmToL1<KV_T, KV_FORMAT> copyKvGmToL1_;
    CopyQueryScaleGmToL1<SCALE_T, Q_SCALE_FORMAT> copyQueryScaleGmToL1_;
    KeyScaleGmToL1Type copyKeyScaleGmToL1_;
    ValueScaleGmToL1Type copyValueScaleGmToL1_;

    BufferManager<BufferType::L1>* l1BufferManagerPtr_;
    BufferManager<BufferType::L0A> l0aBufferManager_;
    BufferManager<BufferType::L0B> l0bBufferManager_;
    BufferManager<BufferType::L0C> l0cBufferManager_;

    BuffersPolicyDB<BufferType::L1> l1QBuffers_;
    L1KType l1KBuffers_;
    L1VtType l1VtBuffers_;

    L0AType mmL0ABuffers_;
    L0BType mmL0BBuffers_;
    L0CType mmL0CBuffers_;

    __gm__ uint8_t* keyPtr_ = nullptr;
    __gm__ uint8_t* valuePtr_ = nullptr;

    const ConstInfoX& constInfo_;

    __aicore__ inline QuantFlashAttnBlockCubeMxfp8SoftmaxFp16(ConstInfoX& constInfo)
        : constInfo_(constInfo){};

    __aicore__ inline void InitCubeBlock(TPipe* pipe, BufferManager<BufferType::L1>* l1BuffMgr, __gm__ uint8_t* query,
                                         __gm__ uint8_t* key, __gm__ uint8_t* value, __gm__ uint8_t* blockTable,
                                         __gm__ uint8_t* dequantScaleQuery, __gm__ uint8_t* dequantScaleKey,
                                         __gm__ uint8_t* dequantScaleValue, QSeqParserType& qParser,
                                         KvSeqParserType& kvParser)
    {
        tPipe_ = pipe;
        l1BufferManagerPtr_ = l1BuffMgr;
        this->qSeqParserPtr_ = &qParser;
        this->kvSeqParserPtr_ = &kvParser;
        InitCubeInput(query, key, value, blockTable, dequantScaleQuery, dequantScaleKey, dequantScaleValue);
    }

    __aicore__ inline void InitBuffers()
    {
        constexpr uint32_t s2BaseSizeCur = s2BaseSize >> 1;
        constexpr uint32_t mm1QSize = mBaseSize * dBaseSize * sizeof(INPUT_T);
        constexpr uint32_t mm1QScaleSize = mBaseSize * dBaseSize / MXFP_GROUP_SIZE * sizeof(SCALE_T);
        constexpr uint32_t mm1KSize = dBaseSize * s2BaseSizeCur * sizeof(INPUT_T);
        constexpr uint32_t mm1KScaleSize = s2BaseSizeCur * dBaseSize / MXFP_GROUP_SIZE * sizeof(SCALE_T);
        constexpr uint32_t mm2VtBufSize =
            (mm2VtSegDataSize + mm2VtSegScaleSize) * sizeof(INPUT_T); // per-bank: 20480+16384=36864B

        l1QBuffers_.Init((*l1BufferManagerPtr_), mm1QSize + mm1QScaleSize);
        l1KBuffers_.Init((*l1BufferManagerPtr_), mm1KSize + mm1KScaleSize);
        l1VtBuffers_.Init((*l1BufferManagerPtr_), mm2VtBufSize);
        // A(V^T)两个bank均在kernel启动时全置1(数据0x38/scale 0x7f), 此后各k段搬运只写真实d通道
        InitVtL1AllOne();
        AscendC::PipeBarrier<PIPE_MTE2>();

        l0aBufferManager_.Init(tPipe_, 65536);
        l0bBufferManager_.Init(tPipe_, 65536);
        l0cBufferManager_.Init(tPipe_, 262144);
        mmL0ABuffers_.Init(l0aBufferManager_, 32 * 1024);
        mmL0BBuffers_.Init(l0bBufferManager_, 32 * 1024);
        mmL0CBuffers_.Init(l0cBufferManager_, (L0C_SIZE / NUM_2) * KB_TO_BYTES);
    }

    __aicore__ inline void InitCubeInput(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
                                         __gm__ uint8_t* blockTable, __gm__ uint8_t* dequantScaleQuery,
                                         __gm__ uint8_t* dequantScaleKey, __gm__ uint8_t* dequantScaleValue)
    {
        InitQBuffer(constInfo_.bSize, constInfo_.realN2Size, constInfo_.realGSize, constInfo_.s1Size, constInfo_.dSize,
                    queryGm_, query);

        InitQScaleBuffer(constInfo_.bSize, constInfo_.realN2Size, constInfo_.realGSize, constInfo_.s1Size,
                         dBaseSize / MXFP_GROUP_SIZE, queryScaleGm_, dequantScaleQuery);

        keyPtr_ = key;
        valuePtr_ = value;
        InitKVBuffer(constInfo_.bSize, constInfo_.s2Size, constInfo_.n2Size, constInfo_.blockSize, constInfo_.dSize,
                     keyGm_, key, constInfo_.keyStrides.bnStride, constInfo_.keyStrides.n2Stride);
        InitKVBuffer(constInfo_.bSize, constInfo_.s2Size, constInfo_.n2Size, constInfo_.blockSize, constInfo_.dSizeV,
                     valueGm_, value, constInfo_.valueStrides.bnStride, constInfo_.valueStrides.n2Stride);
        InitKScaleBuffer(constInfo_.bSize, constInfo_.s2Size, constInfo_.n2Size, constInfo_.blockSize,
                         dBaseSize / MXFP_GROUP_SIZE, keyScaleGm_, dequantScaleKey);
        InitVScaleBuffer(constInfo_.bSize, (constInfo_.s2Size + 63) / MXFP_DIVISOR_SIZE, constInfo_.n2Size,
                         constInfo_.blockSize / MXFP_DIVISOR_SIZE, constInfo_.dSizeV * MXFP_MULTI_BASE_SIZE,
                         valueScaleGm_, dequantScaleValue);
    }

    __aicore__ inline void InitQBuffer(uint32_t batchSize, uint32_t n2Size, uint32_t gSize, uint32_t qSeqSize,
                                       uint32_t headDim, FaGmTensorQ& qGmTensor, __gm__ uint8_t* gm)
    {
        qGmTensor.gmTensor.SetGlobalBuffer((__gm__ Q_T*)gm);
        qGmTensor.offsetCalculator.Init(batchSize, n2Size, gSize, qSeqSize, headDim, *this->qSeqParserPtr_);
    }

    __aicore__ inline void InitQScaleBuffer(uint32_t batchSize, uint32_t n2Size, uint32_t gSize, uint32_t qSeqSize,
                                            uint32_t headDim, FaGmTensorQScale& qScaleGmTensor, __gm__ uint8_t* gm)
    {
        qScaleGmTensor.gmTensor.SetGlobalBuffer((__gm__ SCALE_T*)gm);
        qScaleGmTensor.offsetCalculator.Init(batchSize, n2Size, gSize, qSeqSize, headDim, *this->qSeqParserPtr_);
    }

    __aicore__ inline void InitKVBuffer(uint32_t batchSize, uint32_t kvSeqSize, uint32_t n2Size,
                                        uint32_t kvCacheBlockSize, uint32_t headDim, FaGmTensorKV& kvGmTensor,
                                        __gm__ uint8_t* gm, uint64_t bnStride, uint64_t n2Stride)
    {
        kvGmTensor.gmTensor.SetGlobalBuffer((__gm__ KV_T*)gm);
        kvGmTensor.offsetCalculator.Init(batchSize, n2Size, kvSeqSize, headDim);
        kvGmTensor.offsetCalculator.Init(*this->kvSeqParserPtr_);
    }

    __aicore__ inline void InitKScaleBuffer(uint32_t batchSize, uint32_t kvSeqSize, uint32_t n2Size,
                                            uint32_t kvCacheBlockSize, uint32_t headDim,
                                            FaGmTensorKScale& kScaleGmTensor, __gm__ uint8_t* gm)
    {
        kScaleGmTensor.gmTensor.SetGlobalBuffer((__gm__ SCALE_T*)gm);
        kScaleGmTensor.offsetCalculator.Init(batchSize, n2Size, kvSeqSize, headDim);
        kScaleGmTensor.offsetCalculator.Init(*this->kvSeqParserPtr_);
    }

    __aicore__ inline void InitVScaleBuffer(uint32_t batchSize, uint32_t kvSeqSize, uint32_t n2Size,
                                            uint32_t kvCacheBlockSize, uint32_t headDim,
                                            FaGmTensorVScale& vScaleGmTensor, __gm__ uint8_t* gm)
    {
        vScaleGmTensor.gmTensor.SetGlobalBuffer((__gm__ SCALE_T*)gm);
        vScaleGmTensor.offsetCalculator.Init(batchSize, n2Size, kvSeqSize, headDim);
        vScaleGmTensor.offsetCalculator.Init(*this->kvSeqParserPtr_);
    }

    __aicore__ inline void AllocEventID()
    {
        // InitBuffers阶段已完成eventId申请和SetFlag，这里为空实现
    }

    __aicore__ inline void FreeEventID()
    {
        l1QBuffers_.Uninit((*l1BufferManagerPtr_));
        l1KBuffers_.Uninit((*l1BufferManagerPtr_));
        l1VtBuffers_.Uninit((*l1BufferManagerPtr_));
        mmL0ABuffers_.Uninit(l0aBufferManager_);
        mmL0BBuffers_.Uninit(l0bBufferManager_);
        mmL0CBuffers_.Uninit(l0cBufferManager_);
    }

    // GM→L1 单层搬运: Q数据/Q scale/K数据/K scale按整块搬, K按subLoop取前后半s2;
    // V^T数据/V scale按k段(realK)搬(见CopyValueAsATile/CopyValueScaleTile)
    __aicore__ inline void CopyQueryTile(const LocalTensor<Q_T>& dstTensor, RunInfoX& runInfo)
    {
        uint32_t dstStride = (runInfo.actMSize + 31) >> 5 << 5;
        FaL1Tensor<Q_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};
        GmCoordS1Only gmCoord{.bIdx = runInfo.bIdx,
                              .n2Idx = runInfo.realN2Idx,
                              .gIdx = 0,
                              .s1Idx = runInfo.gS1Idx,
                              .dIdx = 0,
                              .s1DealSize = runInfo.actMSize,
                              .dDealSize = constInfo_.dSize};
        copyQueryGmToL1_(l1Tensor, queryGm_, gmCoord);
    }

    __aicore__ inline void CopyQueryScaleTile(const LocalTensor<SCALE_T>& dstTensor, RunInfoX& runInfo)
    {
        uint32_t dstStride = (runInfo.actMSize + 31) >> 5 << 5;
        FaL1Tensor<SCALE_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};
        GmCoordGs1Merge gmCoord{.bIdx = runInfo.bIdx,
                                .n2Idx = runInfo.realN2Idx,
                                .gS1Idx = runInfo.gS1Idx,
                                .dIdx = 0,
                                .gS1DealSize = runInfo.actMSize,
                                .dDealSize = dBaseSize / MXFP_GROUP_SIZE};
        copyQueryScaleGmToL1_(l1Tensor, queryScaleGm_, gmCoord);
    }

    __aicore__ inline void CopyKeyTile(const LocalTensor<KV_T>& dstTensor, RunInfoX& runInfo, uint32_t s2RealSize,
                                       uint32_t subLoop)
    {
        uint32_t s2Offset = (subLoop % 2 == 0) ? runInfo.s2Idx : runInfo.s2Idx + s2BaseSize / 2;
        uint32_t dstStride = (s2RealSize + 31) >> 5 << 5;
        FaL1Tensor<KV_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};
        GmKvCoord gmCoord{.bIdx = runInfo.bIdx,
                          .n2Idx = runInfo.n2Idx,
                          .s2Idx = s2Offset,
                          .dIdx = 0,
                          .s2DealSize = s2RealSize,
                          .dDealSize = constInfo_.dSize};
        copyKvGmToL1_(l1Tensor, keyGm_, gmCoord);
    }

    __aicore__ inline void CopyKeyScaleTile(const LocalTensor<SCALE_T>& dstTensor, RunInfoX& runInfo,
                                            uint32_t s2RealSize, uint32_t subLoop)
    {
        uint32_t s2Offset = subLoop % 2 == 0 ? runInfo.s2Idx : runInfo.s2Idx + s2BaseSize / 2;
        uint32_t dstStride = (s2RealSize + 31) >> 5 << 5;
        FaL1Tensor<SCALE_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};
        GmKvCoord gmCoord{.bIdx = runInfo.bIdx,
                          .n2Idx = runInfo.n2Idx,
                          .s2Idx = s2Offset,
                          .dIdx = 0,
                          .s2DealSize = s2RealSize,
                          .dDealSize = dBaseSize / MXFP_GROUP_SIZE};
        copyKeyScaleGmToL1_(l1Tensor, keyScaleGm_, gmCoord);
    }

    __aicore__ inline void CopyValueAsATile(LocalTensor<KV_T>& dstTensor, uint32_t s2Offset, uint32_t realK,
                                            RunInfoX& runInfo)
    {
        uint32_t dstStride = mm2BaseK;
        FaL1Tensor<KV_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};
        GmKvCoord gmCoord{.bIdx = runInfo.bIdx,
                          .n2Idx = runInfo.n2Idx,
                          .s2Idx = s2Offset,
                          .dIdx = 0U,
                          .s2DealSize = realK,
                          .dDealSize = constInfo_.dSizeV};
        copyKvGmToL1_(l1Tensor, valueGm_, gmCoord);
    }

    __aicore__ inline void CopyValueScaleTile(const LocalTensor<SCALE_T>& dstTensor, uint32_t s2Offset,
                                              uint32_t s2ScaleUnits, RunInfoX& runInfo)
    {
        uint32_t rowCount = mm2BaseK / MXFP_DIVISOR_SIZE;
        FaL1Tensor<SCALE_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = rowCount};
        GmKvCoord gmCoord{.bIdx = runInfo.bIdx,
                          .n2Idx = runInfo.n2Idx,
                          .s2Idx = s2Offset,
                          .dIdx = 0,
                          .s2DealSize = s2ScaleUnits,
                          .dDealSize = constInfo_.dSizeV * MXFP_MULTI_BASE_SIZE};
        copyValueScaleGmToL1_(l1Tensor, valueScaleGm_, gmCoord);
    }

    // A位一次性初始化: 两个bank的整块V^T L1(数据+scale)开机全置1(fp8数据0x38, scale 0x7f, 均为1.0);
    __aicore__ inline void InitVtL1AllOne()
    {
        for (uint32_t bank = 0; bank < 2; ++bank) {
            Buffer<BufferType::L1>& vtBuf = l1VtBuffers_.Get();
            // V^T数据区(d+32)×128=160×128=20480B, 全置0x38
            InitConstValueParams<uint16_t> vtDataInitParams(1, static_cast<uint16_t>(mm2VtSegDataSize / 32), 0, 0x3838);
            Fill(vtBuf.template GetTensor<KV_T>().template ReinterpretCast<uint16_t>(), vtDataInitParams);

            AscendC::PipeBarrier<PIPE_MTE2>();

            // VScale区 d×128=128×128=16384B, 全置0x7f
            InitConstValueParams<uint16_t> vtScaleInitParams(1, static_cast<uint16_t>(mm2VtSegScaleSize / 32), 0,
                                                             0x7f7f);
            Fill(vtBuf.template GetTensor<SCALE_T>(mm2VtSegDataSize).template ReinterpretCast<uint16_t>(),
                 vtScaleInitParams);
        }
    }

    __aicore__ inline void UpdateKey(uint32_t bIdx)
    {
        ListTensorDesc keyListTensorDesc((__gm__ void*)(this->keyPtr_));
        __gm__ uint8_t* key_ = (__gm__ uint8_t*)keyListTensorDesc.GetDataPtr<__gm__ uint8_t>(bIdx);

        uint64_t s2Size = SeqLenFromTensorList<LAYOUT>(this->keyPtr_, bIdx);
        keyGm_.gmTensor.SetGlobalBuffer((__gm__ KV_T*)key_);
        keyGm_.offsetCalculator.Init(0, constInfo_.n2Size, s2Size, constInfo_.dSize);
        keyGm_.offsetCalculator.Init(*this->kvSeqParserPtr_);
    }

    __aicore__ inline void UpdateValue(uint32_t bIdx)
    {
        ListTensorDesc valueListTensorDesc((__gm__ void*)(this->valuePtr_));
        __gm__ uint8_t* value_ = (__gm__ uint8_t*)valueListTensorDesc.GetDataPtr<__gm__ uint8_t>(bIdx);
        uint64_t s2Size = SeqLenFromTensorList<LAYOUT>(valuePtr_, bIdx);
        valueGm_.gmTensor.SetGlobalBuffer((__gm__ KV_T*)value_);
        valueGm_.offsetCalculator.Init(0, constInfo_.n2Size, s2Size, constInfo_.dSizeV);
        valueGm_.offsetCalculator.Init(*this->kvSeqParserPtr_);
    }

    /* MM1: K * Q^T, Q驻留L1(首个S2 loop搬入, isLastS2Loop释放), K按subLoop搬入 */
    __aicore__ inline void IterateBmm1(MM1_DBUF_T& outputBuf, RunInfoX& runInfo, uint32_t subLoop)
    {
        uint32_t s2CalcSize = s2SplitSize;
        if (unlikely(runInfo.actSingleLoopS2Size < s2BaseSize)) {
            if (subLoop == 0) {
                s2CalcSize = AttentionCommon::Min((int32_t)s2SplitSize, (int32_t)runInfo.actSingleLoopS2Size);
            } else {
                s2CalcSize = runInfo.actSingleLoopS2Size - s2SplitSize;
            }
        }
        Buffer<BufferType::L1> mm1B;
        uint32_t qScaleOffset = ((runInfo.actMSize + 31) >> 5 << 5) * dBaseSize;
        if (unlikely(runInfo.isFirstS2Loop && subLoop == 0)) {
            mm1B = l1QBuffers_.Get();
            mm1B.Wait<HardEvent::MTE1_MTE2>(); // B矩阵常驻，两次Sub中，只有第一次搬B
            LocalTensor<Q_T> mm1BTensor = mm1B.GetTensor<Q_T>();
            CopyQueryTile(mm1BTensor, runInfo);

            LocalTensor<SCALE_T> mm1BScaleTensor = mm1B.GetTensor<SCALE_T>(qScaleOffset);
            CopyQueryScaleTile(mm1BScaleTensor, runInfo);
            mm1B.Set<HardEvent::MTE2_MTE1>();
        } else {
            mm1B = l1QBuffers_.GetPre();
            mm1B.Set<HardEvent::MTE2_MTE1>();
        }

        Buffer<BufferType::L1> mm1A = l1KBuffers_.Get();
        mm1A.Wait<HardEvent::MTE1_MTE2>();
        LocalTensor<KV_T> mm1ATensor = mm1A.GetTensor<KV_T>();
        CopyKeyTile(mm1ATensor, runInfo, s2CalcSize, subLoop);

        uint32_t kScaleOffset = ((s2CalcSize + 31) >> 5 << 5) * dBaseSize;
        LocalTensor<SCALE_T> mm1AScaleTensor = mm1A.GetTensor<SCALE_T>(kScaleOffset);
        CopyKeyScaleTile(mm1AScaleTensor, runInfo, s2CalcSize, subLoop);
        mm1A.Set<HardEvent::MTE2_MTE1>();
        mm1A.Wait<HardEvent::MTE2_MTE1>();
        mm1B.Wait<HardEvent::MTE2_MTE1>();

        Buffer<BufferType::L0C> mm1ResL0C = mmL0CBuffers_.Get();
        mm1ResL0C.Wait<HardEvent::FIX_M>();
        MMParam param = MakeMMParam((uint32_t)s2CalcSize, (uint32_t)runInfo.actMSize, dBaseSize, false, true);
        MatmulFullMX<Q_T, KV_T, T, (s2BaseSize >> 1), 128, dBaseSize, ABLayout::MK, ABLayout::KN, L0AType, L0BType,
                     SCALE_T, SCALE_T, mx_fp8_e4m3_t, mx_fp8_e4m3_t>(
            mm1A.GetTensor<INPUT_T>(), mm1B.GetTensor<INPUT_T>(), mmL0ABuffers_, mmL0BBuffers_,
            mm1ResL0C.GetTensor<T>(), param, mm1A.GetTensor<SCALE_T>(kScaleOffset),
            mm1B.GetTensor<SCALE_T>(qScaleOffset));

        if (unlikely(runInfo.isLastS2Loop && (((runInfo.actSingleLoopS2Size > s2SplitSize) && (subLoop % 2 == 1)) ||
                                              (runInfo.actSingleLoopS2Size <= s2SplitSize)))) {
            mm1B.Set<HardEvent::MTE1_MTE2>();
        }
        mm1A.Set<HardEvent::MTE1_MTE2>();
        mm1ResL0C.Set<HardEvent::M_FIX>();
        mm1ResL0C.Wait<HardEvent::M_FIX>();

        outputBuf.WaitCrossCore();

        FixpipeMm1(outputBuf.template GetTensor<half>(), mm1ResL0C.GetTensor<T>(), runInfo, s2CalcSize);

        mm1ResL0C.Set<HardEvent::FIX_M>();
        outputBuf.SetCrossCore();
    }

    __aicore__ inline void FixpipeMm1(const LocalTensor<half>& dstTensor, const LocalTensor<T>& l0C, RunInfoX& runInfo,
                                      uint32_t s2RealSize)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        // L0C上的bmm1结果矩阵N方向的size大小, 使能NZ2ND, nSize*sizeof(T) 必须是32B的倍数
        fixpipeParams.nSize = (runInfo.actMSize + 15) >> 4 << 4;
        fixpipeParams.mSize = (s2RealSize + 15) >> 4 << 4;
        // L0C上matmul结果相邻连续数据片断间隔（前面一个数据块的头与后面数据块的头的间隔），单位为16 *sizeof(T)
        // 源NZ矩阵中相邻Z排布的起始地址偏移
        fixpipeParams.srcStride = ((fixpipeParams.mSize + 15) / 16) * 16;
        fixpipeParams.dstStride = mBaseSize; // mmResUb上两行之间的间隔，单位：element
        fixpipeParams.dualDstCtl = 0;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;

        // C1 FIXPIPE 随路: FP32 -> FP16 (QF322F16_PRE), 并乘以 softmaxscale (deqScalar)
        float softmaxScale = constInfo_.scaleValue;
        fixpipeParams.quantPre = QuantMode_t::QF322F16_PRE;
        fixpipeParams.deqScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&softmaxScale));

        Fixpipe<half, T, PFA_CFG_ROW_MAJOR_UB>(dstTensor, l0C, fixpipeParams);
    }

    /* ============ MM2: A=V(s2×(d+32)) 经ND2NZ搬入L1不转置 × B=P(Kx128) ============
     * V GM→L1为ND2NZ不转置(s2行×(d+32)列含全1带), L1→L0时转置为(d+32)×K(即V^T);
     * A的m方向尾32行为全1带 -> Mmad(m=d+32)后L0C尾32行自然为row_sum(全等);
     * V在L1为专用2buff乒乓, 两bank全1带均在kernel启动时初始化一次, k循环数据搬运只写真实d通道;
     * B(P)由vec层写入l1PBuffers_(3buff轮转), isRightTranspose=true走LoadData转置分支;
     * C输出(d+32)x128 float, fixpipe到UB后由V2除以row_sum并cast bf16。
     */
    __aicore__ inline void IterateBmm2(MM2_DBUF_T& outputBuf, MM2_PBUF_POLICY_T& inputBuf, RunInfoX& runInfo)
    {
        // 1. 等待P就绪
        MM2_PBUF_T mm2P = inputBuf.Get();
        mm2P.WaitCrossCore();

        // 2. P的scale区域初始化0x7f(E8M0 1.0): vec层仅在subLoop%2==1时写入P0修正,
        //    其余k段(前半/单subLoop尾块)的scale必须为1.0, 每个P buffer(3buff轮转)每次使用前初始化
        InitPScaleRegion(mm2P, runInfo);

        // 3. k循环: baseK=128, 每轮 搬V^T段 -> 搬P scale修正(仅subLoop奇数次) -> Mmad累加
        Buffer<BufferType::L0C> mm2ResL0C = mmL0CBuffers_.Get();
        mm2ResL0C.Wait<HardEvent::FIX_M>();
        IterateBmm2KLoop(mm2P, mm2ResL0C, runInfo);

        // 4. fixpipe搬出: (d+32)x128 float到UB, 供V2做row_sum除法
        mm2ResL0C.Set<HardEvent::M_FIX>();
        mm2ResL0C.Wait<HardEvent::M_FIX>();
        outputBuf.WaitCrossCore();
        FixpipeMm2(outputBuf.template GetTensor<T>(), mm2ResL0C.GetTensor<T>(), runInfo);
        mm2ResL0C.Set<HardEvent::FIX_M>();
        outputBuf.SetCrossCore();
    }

    // 双subLoop(>s2SplitSize): vec在subLoop1写入前半的P0修正(从scale区起点起), 此处初始化后半;
    // 单subLoop(<=s2SplitSize): vec不写修正, 此处初始化前半(即本loop实际占用的scale区域)
    __aicore__ inline void InitPScaleRegion(MM2_PBUF_T& mm2P, RunInfoX& runInfo)
    {
        constexpr uint32_t pScaleOffset = mBaseSize * s2BaseSize; // PScale在L1的偏移量(元素)
        constexpr uint32_t sizeOfInt16 = 2;
        // 前半scale个数(subLoop0的k列对应区域)
        constexpr uint32_t halfScaleNum = mBaseSize * s2SplitSize / MXFP_GROUP_SIZE;
        constexpr uint32_t initBlockNum = halfScaleNum * sizeof(SCALE_T) / blockBytes; // 1KB=32块
        uint32_t initOffsetInt16 = runInfo.actSingleLoopS2Size > s2SplitSize ?
                                       pScaleOffset / sizeOfInt16 + halfScaleNum / sizeOfInt16 : // 后半起点
                                       pScaleOffset / sizeOfInt16;                               // 前半起点
        LocalTensor<int16_t> mm2PScaleTensor = mm2P.GetTensor<int16_t>(initOffsetInt16);
        InitConstValue(mm2PScaleTensor, {1, initBlockNum, 0, 0x7f7f});
    }

    // MM2 k循环: 总K(本loopS2)拆baseK=128段, 每段V^T搬运(Dn2Nz直通)+P scale修正+Mmad累加;
    // V^T为2buff乒乓: k循环Get()按段轮转bank, CopyVtSegToL1/MmadBmm2Seg内部用GetPre()取同段bank,
    __aicore__ inline void IterateBmm2KLoop(MM2_PBUF_T& mm2P, Buffer<BufferType::L0C>& mm2ResL0C, RunInfoX& runInfo)
    {
        uint32_t kLoops = (runInfo.actSingleLoopS2Size + mm2BaseK - 1) / mm2BaseK;
        uint32_t realK = mm2BaseK;
        for (uint32_t k = 0; k < kLoops; k++) {
            if (k == kLoops - 1) {
                realK = runInfo.actSingleLoopS2Size - k * mm2BaseK;
            }
            Buffer<BufferType::L1>& vtBuf = l1VtBuffers_.Get();

            vtBuf.Wait<HardEvent::MTE1_MTE2>();

            CopyVtSegToL1(k * mm2BaseK, realK, runInfo);

            vtBuf.Set<HardEvent::MTE2_MTE1>();
            vtBuf.Wait<HardEvent::MTE2_MTE1>();

            MmadBmm2Seg(mm2P, mm2ResL0C, k, realK, runInfo);

            vtBuf.Set<HardEvent::MTE1_MTE2>();
        }
    }

    __aicore__ inline void CopyVtSegToL1(uint32_t kStart, uint32_t realK, RunInfoX& runInfo)
    {
        Buffer<BufferType::L1>& vtBuf = l1VtBuffers_.GetPre();
        // vtBuf: 每bank数据区=mm2VtSegDataSize=20480B, scale区紧随其后;
        // V数据固定写在bank数据区起点, kStart仅决定GM源地址
        LocalTensor<KV_T> vtL1 = vtBuf.template GetTensor<KV_T>();
        LocalTensor<SCALE_T> vtScaleL1 = vtBuf.template GetTensor<SCALE_T>(mm2VtSegDataSize);
        CopyValueAsATile(vtL1, runInfo.s2Idx + kStart, realK, runInfo);
        CopyValueScaleTile(vtScaleL1, (runInfo.s2Idx + kStart) / MXFP_DIVISOR_SIZE, (realK + 63) / MXFP_DIVISOR_SIZE,
                           runInfo);
    }

    // 单段Mmad: V在L1未转置(s2×(d+32)), L1→L0需转置为(d+32)×K(isLeftTranspose=true);
    __aicore__ inline void MmadBmm2Seg(MM2_PBUF_T& mm2P, Buffer<BufferType::L0C>& mm2ResL0C, uint32_t k, uint32_t realK,
                                       RunInfoX& runInfo)
    {
        Buffer<BufferType::L1>& vtBuf = l1VtBuffers_.GetPre();
        LocalTensor<KV_T> vtL1 = vtBuf.template GetTensor<KV_T>();
        LocalTensor<SCALE_T> vtScaleL1 = vtBuf.template GetTensor<SCALE_T>(mm2VtSegDataSize);

        // B位(P)的scale: P的E8M0按行scale(P0修正, vec层已写入L1), 按k段偏移寻址;
        // 双subLoop时vec在subLoop1写入的P0修正落在k段0的scale区, k=1的Mmad也按段0寻址
        constexpr uint32_t pScaleOffset = mBaseSize * s2BaseSize;
        constexpr uint32_t pScaleKSegOffset = mm2BaseK * mBaseSize / MXFP_GROUP_SIZE;
        uint32_t pScaleK = ((k == 1) ? 0 : k);
        LocalTensor<SCALE_T> pScaleFakeTensor = mm2P.GetTensor<SCALE_T>(pScaleOffset);

        MMParam param = MakeMMParam(mm2VtMSize, mBaseSize, (realK + 127) >> 7 << 7, true, false, k == 0, k == 0);
        MatmulFullMX<KV_T, INPUT_T, T, mm2VtMSize, mBaseSize, mm2BaseK, ABLayout::MK, ABLayout::KN, L0AType, L0BType,
                     SCALE_T, SCALE_T, mx_fp8_e4m3_t, mx_fp8_e4m3_t>(
            vtL1, mm2P.GetTensor<INPUT_T>()[k * mm2BaseK * mBaseSize], mmL0ABuffers_, mmL0BBuffers_,
            mm2ResL0C.GetTensor<T>(), param, vtScaleL1, pScaleFakeTensor[pScaleK * pScaleKSegOffset]);
    }

    // MM2 DN方案fixpipe: L0C (d+32)x128 float -> UB, 搬出前129行(V2取前128行O + 第128行row_sum)
    __aicore__ inline void FixpipeMm2(const LocalTensor<T>& dstTensor, const LocalTensor<T>& l0C, RunInfoX& runInfo)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.mSize = 129; // d真实行 + 1行row_sum(全1带首行, 其余31行冗余不搬)
        fixpipeParams.nSize = mBaseSize;
        fixpipeParams.srcStride = ((mm2VtMSize + 15) / 16 * 16); // 160, 匹配MMAD L0C stride
        fixpipeParams.dstStride = mBaseSize;
        fixpipeParams.dualDstCtl = 0;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        Fixpipe<T, T, BMM2_FIXPIPE_CONFIG>(dstTensor, l0C, fixpipeParams);
    }

}; // QuantFlashAttnBlockCubeMxfp8SoftmaxFp16

template <typename INPUT_T, typename T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, uint8_t KvLayoutType = 0>
class QuantFlashAttnBlockCubeMxfp8SoftmaxFp16Dummy {
public:
    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t dBaseSize = (uint32_t)dTemplateType;
    static constexpr uint32_t dVBaseSize = (uint32_t)dVTemplateType;
    static constexpr LayOutTypeEnum LAYOUT = layout;

    using Q_T = INPUT_T;
    using KV_T = INPUT_T;
    using MM_T = T;
    using MM1_DBUF_T = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    using MM2_PBUF_POLICY_T = BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>;
    using MM2_PBUF_T = Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>;
    using MM2_DBUF_T = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;

    using ConstInfoX = ConstInfo_t;
    __aicore__ inline QuantFlashAttnBlockCubeMxfp8SoftmaxFp16Dummy(ConstInfoX& constInfo){};
};
} // namespace BaseApi

#endif // QUANT_FLASH_ATTN_BLOCK_CUBE_MXFP8_SOFTMAX_FP16_H_
