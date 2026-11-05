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
 * \file quant_flash_attn_block_vec_flashdecode_arch92.h
 * \brief
 */
#ifndef QUANT_FLASH_ATTN_BLOCK_VEC_FLASHDECODE_ARCH92_H_
#define QUANT_FLASH_ATTN_BLOCK_VEC_FLASHDECODE_ARCH92_H_

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "../arch35/memory_copy_arch35_quant_flash_attn.h"
#include "lib/matrix/matmul/tiling.h"
#include "quant_flash_attn_common_def_arch92.h"
#include "../../../common/op_kernel/arch35/vf/vf_flash_decode_arch35.h"

namespace BaseApi {

// 与 FA 侧 UpdateMinCheckValue 语义一致的 min 判值计算（VF 形式）：
// DN 路径全 mask 行写入的 max 经过了 scaleValue 乘法 + ln2 量化 + ln(pScale) 偏移，
// 使用与 vec1 相同的硬件 Truncate(CAST_CEIL) 保证判值 bit 级一致；非 DN 路径保持原始 minValue
template <bool useDn>
__simd_vf__ inline void CalcMinCheckValueVF(__ubuf__ float* dstUb, const float minValue, const float scaleValue,
                                            const float pScale)
{
    RegTensor<float> vregMin;
    MaskReg pregAll = CreateMask<uint16_t, MaskPattern::ALL>();
    Duplicate(vregMin, minValue);
    if constexpr (useDn) {
        RegTensor<float> vregPScale;
        RegTensor<float> vregLnPScale;
        Duplicate(vregPScale, pScale);
        Ln(vregLnPScale, vregPScale, pregAll);
        Muls(vregMin, vregMin, scaleValue, pregAll);
        Muls(vregMin, vregMin, INV_LN2, pregAll);
        Truncate<float, RoundMode::CAST_CEIL>(vregMin, vregMin, pregAll);
        Muls(vregMin, vregMin, LN2, pregAll);
        Sub(vregMin, vregMin, vregLnPScale, pregAll);
    }
    StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ float*&)dstUb, vregMin, pregAll);
}

// 与 FD 的 LSE 共用已变换的 min 判值；条件选择直接覆盖 NaN，不能以乘零代替。
__simd_vf__ inline void ClearInvalidFdRowsVF(__ubuf__ float* outputUb, __ubuf__ float* maxUb,
                                             __ubuf__ float* minCheckUb, uint32_t rows, uint32_t columns)
{
    constexpr uint32_t FLOATS_PER_BLOCK = 32U / sizeof(float);   // max 每行占一个 32B 广播块。
    constexpr uint32_t FLOATS_PER_REPEAT = 256U / sizeof(float); // 输出 D 已按 64 个 FP32 对齐。
    Reg::RegTensor<float> vregMin;
    Reg::RegTensor<float> vregMax;
    Reg::RegTensor<float> vregOutput;
    Reg::RegTensor<float> vregZero;
    Reg::MaskReg pregInvalid;
    Reg::MaskReg pregAll = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(vregMin, minCheckUb);
    Reg::Duplicate(vregZero, 0.0f);
    for (uint16_t row = 0; row < rows; ++row) {
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(vregMax, maxUb + row * FLOATS_PER_BLOCK);
        Reg::Compare<float, CMPMODE::EQ>(pregInvalid, vregMax, vregMin, pregAll);
        for (uint16_t repeat = 0; repeat < columns / FLOATS_PER_REPEAT; ++repeat) {
            __ubuf__ float* outputRow = outputUb + row * columns + repeat * FLOATS_PER_REPEAT;
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(vregOutput, outputRow);
            Reg::Select<float>(vregOutput, vregZero, vregOutput, pregInvalid);
            Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(outputRow, vregOutput, pregAll);
        }
    }
}

template <LayOutTypeEnum LAYOUT>
__aicore__ inline constexpr fa_base_vector::UbInputFormat GeInputUbFormat()
{
    static_assert((LAYOUT == LayOutTypeEnum::LAYOUT_BSH) || (LAYOUT == LayOutTypeEnum::LAYOUT_BNSD) ||
                      (LAYOUT == LayOutTypeEnum::LAYOUT_TND) || (LAYOUT == LayOutTypeEnum::LAYOUT_NTD),
                  "Get Query GmFormat fail, LAYOUT_T is incorrect");
    if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BSH || LAYOUT == LayOutTypeEnum::LAYOUT_TND) {
        return fa_base_vector::UbInputFormat::S1G;
    } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BNSD || LAYOUT == LayOutTypeEnum::LAYOUT_NTD) {
        return fa_base_vector::UbInputFormat::GS1;
    }
}

template <typename INPUT_T, typename T, typename OUTPUT_T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          LayOutTypeEnum outLayout = LayOutTypeEnum::None, S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, bool hasAtten = false, uint8_t KvLayoutType = 0,
          bool useDn = false>
class QuantFlashAttnBlockVecFlashDecode {
public:
    // =================================类型定义区=================================
    struct TaskInfo {
        uint32_t bIdx;
        uint32_t n2Idx;
        uint32_t gS1Idx;
        uint32_t actualCombineLoopSize;
    };

    // FD 静态 Tensor 的总字节数，供 DN/ND 调用点检查各自允许复用的原有缓冲范围。
    static __aicore__ inline constexpr uint32_t GetFdTotalUbSize()
    {
        // 五份统计/权重、四份输出、四份单寄存器 LSE/判值区。
        return 5U * FD_LSE_BUF_BYTES + 4U * FD_MM2_BUF_BYTES + 4U * BUFFER_SIZE_BYTE_256B;
    }

private:
    // =================================常量区=================================
    static constexpr int64_t BYTE_BLOCK = 32UL;        // UB 广播与搬运块宽，单位字节。
    static constexpr int64_t REPEAT_BLOCK_BYTE = 256U; // 一次 VF 完整寄存器访存宽度，单位字节。
    // 事件号只在 FA 退役到下一 section 恢复之间使用，不与 FA 队列同时占用。
    static constexpr uint64_t SYNC_LSE_MAX_SUM_BUF1_FLAG = 0; // 第一组 max/sum 输入槽。
    static constexpr uint64_t SYNC_LSE_MAX_SUM_BUF2_FLAG = 1; // 第二组 max/sum 输入槽。
    static constexpr uint64_t SYNC_MM2RES_BUF1_FLAG = 2;      // 第一份 partial O 输入槽。
    static constexpr uint64_t SYNC_MM2RES_BUF2_FLAG = 3;      // 第二份 partial O 输入槽。
    static constexpr uint64_t SYNC_FDOUTPUT_BUF_FLAG = 4;     // 最终 O 输出槽。
    static constexpr uint64_t SYNC_LSEOUTPUT_BUF_FLAG = 5;    // 最终 LSE 输出槽。

    static constexpr uint32_t BUFFER_SIZE_BYTE_256B = 256; // 单个寄存器的字节数。

    static constexpr uint32_t BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(T);
    static constexpr uint32_t FP32_REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(float);

    // dV按64对齐后的元素数(与FA侧dTemplateAlign64推导一致, 不依赖头文件)
    static constexpr uint32_t FD_DV_ALIGN64 =
        ((static_cast<uint32_t>(dVTemplateType) + FP32_REPEAT_ELEMENT_NUM - 1U) / FP32_REPEAT_ELEMENT_NUM) *
        FP32_REPEAT_ELEMENT_NUM;
    // mm2系buffer实际最大行数(fdBalanceMBaseSize=8, 见FlashDecode主循环)
    static constexpr uint32_t FD_BALANCE_M_BASE = QFA_FD_MAX_ROWS;
    // lse系buffer(sum/max/exp)字节尺寸; mm2系buffer 8行*dAlign*fp32
    static constexpr uint32_t FD_LSE_BUF_BYTES = QFA_FD_LSE_BYTES;
    static constexpr uint32_t FD_MM2_BUF_BYTES = FD_BALANCE_M_BASE * FD_DV_ALIGN64 * sizeof(float);

    static constexpr float FLOAT_INF = 3e+99; // 超过 FP32 有限范围，表示正无穷。
    static constexpr bool HAS_MASK = hasAtten;
    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;

    uint32_t preLoadNum_ = 2U; // 最多预加载两个输入槽；任务开始时按实际分片数收缩。
    uint32_t dSizeV_Align_;
    using ConstInfoX = ConstInfo_t;

protected:
    GlobalTensor<float> lseSumFdGm_;
    GlobalTensor<float> lseMaxFdGm_;
    GlobalTensor<float> accumOutGm_;
    GlobalTensor<OUTPUT_T> attentionOutGm_;
    GlobalTensor<float> softmaxLseGm_;

    static constexpr UbFormat UB_FORMAT = GetOutUbFormat<layout>();
    static constexpr bool isPa = KvLayoutType > 0;

    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<layout>();
    static constexpr ActualSeqLensMode KV_MODE = GetKvActSeqMode<layout, isPa>();
    __gm__ uint8_t* keyPtr_ = nullptr;

    using QSeqParserType =
        typename std::conditional<(layout == LayOutTypeEnum::LAYOUT_TND || layout == LayOutTypeEnum::LAYOUT_NTD),
                                  ActualSeqLensParser<Q_MODE, int32_t, true>,
                                  ActualSeqLensParser<Q_MODE, int32_t>>::type;

    using KvSeqParserType = typename std::conditional<
        (!isPa && (layout == LayOutTypeEnum::LAYOUT_TND || layout == LayOutTypeEnum::LAYOUT_NTD)),
        ActualSeqLensParser<KV_MODE, int32_t, true>, ActualSeqLensParser<KV_MODE, int32_t>>::type;

    QSeqParserType* qActSeqLensParser_ = nullptr;
    KvSeqParserType* kvActSeqLensParser_ = nullptr;

    int64_t preTokensPerBatch_ = 0;
    int64_t nextTokensPerBatch_ = 0;

    float pScaleValue_{1.0f}; // 未提供 P 系数时使用乘法单位值。

    uint64_t actSeqLensKv_ = 0;
    uint64_t actSeqLensQ_ = 0;
    // ================================类成员变量====================================
    // 结构体
    const ConstInfoX& constInfo_;
    TaskInfo taskInfo_{};

private:
    // ================================FD Local Buffer区(静态Tensor)====================================
    LocalTensor<T> fdSumBuf1_;
    LocalTensor<T> fdSumBuf2_;
    LocalTensor<T> fdMaxBuf1_;
    LocalTensor<T> fdMaxBuf2_;
    LocalTensor<T> fdLseExpBuf_;
    LocalTensor<T> fdMm2ResBuf1_;
    LocalTensor<T> fdMm2ResBuf2_;
    LocalTensor<T> fdReduceBuf_;
    LocalTensor<OUTPUT_T> fdOutputBuf_;

    LocalTensor<T> fdLseMaxUbBuf1_;
    LocalTensor<T> fdLseMaxUbBuf2_;
    LocalTensor<T> fdLseUbBuf_;
    LocalTensor<float> fdMinCheckBuf_;

public:
    __aicore__ inline QuantFlashAttnBlockVecFlashDecode(ConstInfoX& constInfo)
        : constInfo_(constInfo){};

    __aicore__ inline void InitGlobalTensor(GlobalTensor<float> lseMaxFdGm, GlobalTensor<float> lseSumFdGm,
                                            GlobalTensor<float> accumOutGm, GlobalTensor<OUTPUT_T> attentionOutGm,
                                            __gm__ uint8_t* key)
    {
        this->lseMaxFdGm_ = lseMaxFdGm;
        this->lseSumFdGm_ = lseSumFdGm;
        this->accumOutGm_ = accumOutGm;
        this->attentionOutGm_ = attentionOutGm;
        this->keyPtr_ = key;
    }

    __aicore__ inline void SetCuSeqLensParsers(QSeqParserType& qParser, KvSeqParserType& kvParser)
    {
        this->qActSeqLensParser_ = &qParser;
        this->kvActSeqLensParser_ = &kvParser;
    }

    __aicore__ inline void InitSoftmaxLseGm(GlobalTensor<float> softmaxLseGm)
    {
        this->softmaxLseGm_ = softmaxLseGm;
    }

    __aicore__ inline void InitPScaleValue(float pScaleValue)
    {
        this->pScaleValue_ = pScaleValue;
    }

    __aicore__ inline void InitParams()
    {
        this->dSizeV_Align_ = AttentionCommon::Align(constInfo_.dSizeV, FP32_REPEAT_ELEMENT_NUM);
    }

    template <uint32_t UB_FD_BASE_OFFSET>
    __aicore__ inline void InitBuffers()
    {
        if ASCEND_IS_AIV {
            // 调用点按原有分配顺序选择复用起点：DN 为 stage2，ND 为 stage1。
            // 两条路线均避开跨核 BMM、pScale 和固定索引表；不重排 FA 缓冲或重置 TPipe。
            constexpr uint32_t OFF_SUM1 = UB_FD_BASE_OFFSET;
            constexpr uint32_t OFF_SUM2 = OFF_SUM1 + FD_LSE_BUF_BYTES;
            constexpr uint32_t OFF_MAX1 = OFF_SUM2 + FD_LSE_BUF_BYTES;
            constexpr uint32_t OFF_MAX2 = OFF_MAX1 + FD_LSE_BUF_BYTES;
            constexpr uint32_t OFF_LSEEXP = OFF_MAX2 + FD_LSE_BUF_BYTES;
            constexpr uint32_t OFF_MM2RES1 = OFF_LSEEXP + FD_LSE_BUF_BYTES;
            constexpr uint32_t OFF_MM2RES2 = OFF_MM2RES1 + FD_MM2_BUF_BYTES;
            constexpr uint32_t OFF_REDUCE = OFF_MM2RES2 + FD_MM2_BUF_BYTES;
            constexpr uint32_t OFF_OUTPUT = OFF_REDUCE + FD_MM2_BUF_BYTES;
            constexpr uint32_t OFF_LSEMAX1 = OFF_OUTPUT + FD_MM2_BUF_BYTES;
            constexpr uint32_t OFF_LSEMAX2 = OFF_LSEMAX1 + BUFFER_SIZE_BYTE_256B;
            constexpr uint32_t OFF_LSEUB = OFF_LSEMAX2 + BUFFER_SIZE_BYTE_256B;
            constexpr uint32_t OFF_MINCHECK = OFF_LSEUB + BUFFER_SIZE_BYTE_256B;
            static_assert(OFF_MINCHECK + BUFFER_SIZE_BYTE_256B == UB_FD_BASE_OFFSET + GetFdTotalUbSize(),
                          "FD 静态槽末地址与总容量公式不一致");

            fdSumBuf1_ =
                LocalTensor<uint8_t>(TPosition::VECIN, OFF_SUM1, FD_LSE_BUF_BYTES).template ReinterpretCast<T>();
            fdSumBuf2_ =
                LocalTensor<uint8_t>(TPosition::VECIN, OFF_SUM2, FD_LSE_BUF_BYTES).template ReinterpretCast<T>();
            fdMaxBuf1_ =
                LocalTensor<uint8_t>(TPosition::VECIN, OFF_MAX1, FD_LSE_BUF_BYTES).template ReinterpretCast<T>();
            fdMaxBuf2_ =
                LocalTensor<uint8_t>(TPosition::VECIN, OFF_MAX2, FD_LSE_BUF_BYTES).template ReinterpretCast<T>();
            fdLseExpBuf_ =
                LocalTensor<uint8_t>(TPosition::VECIN, OFF_LSEEXP, FD_LSE_BUF_BYTES).template ReinterpretCast<T>();
            fdMm2ResBuf1_ =
                LocalTensor<uint8_t>(TPosition::VECIN, OFF_MM2RES1, FD_MM2_BUF_BYTES).template ReinterpretCast<T>();
            fdMm2ResBuf2_ =
                LocalTensor<uint8_t>(TPosition::VECIN, OFF_MM2RES2, FD_MM2_BUF_BYTES).template ReinterpretCast<T>();
            fdReduceBuf_ =
                LocalTensor<uint8_t>(TPosition::VECIN, OFF_REDUCE, FD_MM2_BUF_BYTES).template ReinterpretCast<T>();
            fdOutputBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, OFF_OUTPUT, FD_MM2_BUF_BYTES)
                               .template ReinterpretCast<OUTPUT_T>();
            fdLseMaxUbBuf1_ = LocalTensor<uint8_t>(TPosition::VECIN, OFF_LSEMAX1, BUFFER_SIZE_BYTE_256B)
                                  .template ReinterpretCast<T>();
            fdLseMaxUbBuf2_ = LocalTensor<uint8_t>(TPosition::VECIN, OFF_LSEMAX2, BUFFER_SIZE_BYTE_256B)
                                  .template ReinterpretCast<T>();
            fdLseUbBuf_ =
                LocalTensor<uint8_t>(TPosition::VECIN, OFF_LSEUB, BUFFER_SIZE_BYTE_256B).template ReinterpretCast<T>();
            fdMinCheckBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, OFF_MINCHECK, BUFFER_SIZE_BYTE_256B)
                                 .template ReinterpretCast<float>();
        }
    }

    __aicore__ inline void AllocEventID()
    {
        // 输入双槽分别使用 0/1 和 2/3，空闲态表示向量流水已释放给 MTE2。
        SetFlag<HardEvent::V_MTE2>(SYNC_LSE_MAX_SUM_BUF1_FLAG);
        SetFlag<HardEvent::V_MTE2>(SYNC_LSE_MAX_SUM_BUF2_FLAG);
        SetFlag<HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG);
        SetFlag<HardEvent::V_MTE2>(SYNC_MM2RES_BUF2_FLAG);
        // 输出 4/5 号的空闲态表示 MTE3 已释放输出槽，LSE 关闭时也成对消费初始令牌。
        SetFlag<HardEvent::MTE3_V>(SYNC_FDOUTPUT_BUF_FLAG);
        SetFlag<HardEvent::MTE3_V>(SYNC_LSEOUTPUT_BUF_FLAG);
    }

    __aicore__ inline void FreeEventID()
    {
        WaitFlag<HardEvent::V_MTE2>(SYNC_LSE_MAX_SUM_BUF1_FLAG);
        WaitFlag<HardEvent::V_MTE2>(SYNC_LSE_MAX_SUM_BUF2_FLAG);
        WaitFlag<HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG);
        WaitFlag<HardEvent::V_MTE2>(SYNC_MM2RES_BUF2_FLAG);
        WaitFlag<HardEvent::MTE3_V>(SYNC_FDOUTPUT_BUF_FLAG);
        WaitFlag<HardEvent::MTE3_V>(SYNC_LSEOUTPUT_BUF_FLAG);
    }

protected:
    __aicore__ inline void CopyAccumOutIn(LocalTensor<T>& accumOutLocal, uint32_t splitKVIndex, uint32_t startRow,
                                          uint32_t dealRowCount)
    {
        DataCopyExtParams copyInParams{};
        DataCopyPadExtParams<T> copyInPadParams{};
        copyInParams.blockCount = dealRowCount;
        copyInParams.blockLen = constInfo_.dSizeV * sizeof(T);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = (this->dSizeV_Align_ - constInfo_.dSizeV) / BLOCK_ELEMENT_NUM;

        copyInPadParams.isPad = true;
        copyInPadParams.leftPadding = 0;
        copyInPadParams.rightPadding = (this->dSizeV_Align_ - constInfo_.dSizeV) % BLOCK_ELEMENT_NUM;
        copyInPadParams.paddingValue = 0;
        uint64_t combineAccumOutOffset =
            static_cast<uint64_t>(startRow) * constInfo_.dSizeV +                // taskoffset + g轴offset
            static_cast<uint64_t>(splitKVIndex) * mBaseSize * constInfo_.dSizeV; // 份数offset

        DataCopyPad(accumOutLocal, accumOutGm_[combineAccumOutOffset], copyInParams, copyInPadParams);
    }
    __aicore__ inline void CopyLseIn(uint32_t startRow, uint32_t dealRowCount, uint64_t baseOffset, uint32_t cntM)
    {
        LocalTensor<T> lseSum = (cntM & 1) == 0 ? fdSumBuf1_ : fdSumBuf2_;
        LocalTensor<T> lseMax = (cntM & 1) == 0 ? fdMaxBuf1_ : fdMaxBuf2_;

        uint64_t combineLseOffset = (baseOffset + startRow) * BLOCK_ELEMENT_NUM;
        uint64_t combineLoopOffset = mBaseSize * BLOCK_ELEMENT_NUM;
        uint64_t dealRowCountAlign = dealRowCount * BLOCK_ELEMENT_NUM;

        for (uint32_t i = 0; i < taskInfo_.actualCombineLoopSize; ++i) {
            DataCopy(lseSum[i * dealRowCountAlign], lseSumFdGm_[combineLseOffset + i * combineLoopOffset],
                     dealRowCountAlign); // 份数offset

            DataCopy(lseMax[i * dealRowCountAlign], lseMaxFdGm_[combineLseOffset + i * combineLoopOffset],
                     dealRowCountAlign);
        }
    }
    __aicore__ inline void CalcMinCheckValue()
    {
        // 与 FA 侧 UpdateMinCheckValue 保持一致：DN 路径全 mask 行写入的 max 经过了
        // scaleValue 乘法 + ln2 量化 + ln(pScale) 偏移；非 DN 路径保持原始 minValue
        uint32_t minBits = NEGATIVE_MIN_VALUE_FP32_LN2;
        float minValue = *((float*)&minBits);
        LocalTensor<float> minCheckUb = fdMinCheckBuf_;
        CalcMinCheckValueVF<useDn>((__ubuf__ float*)minCheckUb.GetPhyAddr(), minValue, constInfo_.scaleValue,
                                   pScaleValue_);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeScaleValue(LocalTensor<T>& lseExp, uint32_t dealRowCount,
                                             uint32_t actualCombineLoopSize, uint32_t cntM, uint32_t startRow)
    {
        LocalTensor<T> lseSum = (cntM & 1) == 0 ? fdSumBuf1_ : fdSumBuf2_;
        LocalTensor<T> lseMax = (cntM & 1) == 0 ? fdMaxBuf1_ : fdMaxBuf2_;
        LocalTensor<T> lseMaxUb = (cntM & 1) == 0 ? fdLseMaxUbBuf1_ : fdLseMaxUbBuf2_;

        LocalTensor<T> sinkExpBuf;
        LocalTensor<T> maxLseUb = fdLseUbBuf_;
        // minCheckValue 经 UB 传入 VF（__simd_vf__ 末位运行时标量在 AIV 首次调用会丢失，
        // 会导致 LSE 在合法的 ln2 量化边界 max=0.0 行误输出 inf）
        CalcMinCheckValue();
        ComputeScaleValue_VF_FD_MinUb(sinkExpBuf, lseMax, lseSum, lseExp, maxLseUb, lseMaxUb, fdMinCheckBuf_,
                                      dealRowCount, actualCombineLoopSize, constInfo_.isSoftmaxLseEnable, false);
    }

    __aicore__ inline void Bmm2DataCopyOutTrans(LocalTensor<OUTPUT_T>& attenOutUb, uint32_t startRow,
                                                uint32_t dealRowCount, uint32_t columnCount)
    {
        // 统一采用支持非对齐 D 的搬运描述，覆盖 D72；对齐 D 仍由相同 helper 写回。
        FaUbTensor<OUTPUT_T, true> ubTensor{
            .tensor = attenOutUb,
            .rowCount = dealRowCount,
            .colCount = columnCount,
        };
        GmCoordGs1Merge gmCoord{.bIdx = taskInfo_.bIdx,
                                .n2Idx = taskInfo_.n2Idx,
                                .gS1Idx = taskInfo_.gS1Idx + startRow,
                                .dIdx = 0,
                                .gS1DealSize = dealRowCount,
                                .dDealSize = (uint32_t)constInfo_.dSizeV};

        if constexpr (outLayout == LayOutTypeEnum::LAYOUT_BSH) {
            constexpr GmFormat OUT_FORMAT = GmFormat::BSNGD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm_;
            outGmTensor.offsetCalculator.Init(constInfo_.bSize, constInfo_.realN2Size, constInfo_.realGSize,
                                              constInfo_.s1Size, constInfo_.dSizeV, *qActSeqLensParser_);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if constexpr (outLayout == LayOutTypeEnum::LAYOUT_BNSD) {
            constexpr GmFormat OUT_FORMAT = GmFormat::BNGSD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm_;
            outGmTensor.offsetCalculator.Init(constInfo_.bSize, constInfo_.realN2Size, constInfo_.realGSize,
                                              constInfo_.s1Size, constInfo_.dSizeV, *qActSeqLensParser_);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if constexpr (outLayout == LayOutTypeEnum::LAYOUT_TND) {
            constexpr GmFormat OUT_FORMAT = GmFormat::TNGD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t, true> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm_;
            outGmTensor.offsetCalculator.Init(constInfo_.realN2Size, constInfo_.realGSize, constInfo_.dSizeV,
                                              *qActSeqLensParser_);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if constexpr (outLayout == LayOutTypeEnum::LAYOUT_NTD) {
            constexpr GmFormat OUT_FORMAT = GmFormat::NGTD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t, true> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm_;
            outGmTensor.offsetCalculator.Init(constInfo_.realN2Size, constInfo_.realGSize, constInfo_.dSizeV,
                                              *qActSeqLensParser_);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        }
    }
    __aicore__ inline void ReduceFinalRes(LocalTensor<T>& reduceOut, LocalTensor<T>& mm2Res, LocalTensor<T>& lseLocal,
                                          uint32_t cntKV, uint32_t dealRowCount)
    {
        uint64_t dSizeV_Align = (uint64_t)this->dSizeV_Align_;
        ReduceFinalRes_VF<T>(reduceOut, lseLocal, mm2Res, dealRowCount, dSizeV_Align, cntKV);
    }
    __aicore__ inline void CopyFinalResOut(LocalTensor<T>& accumOutLocal, uint32_t startRow, uint32_t dealRowCount,
                                           uint32_t cntM)
    {
        LocalTensor<OUTPUT_T> tmpBmm2ResCastTensor = fdOutputBuf_;
        AscendC::PipeBarrier<PIPE_V>();
        DealInvalidRows(accumOutLocal, startRow, dealRowCount, this->dSizeV_Align_);
        DealInvalidMaskRows(accumOutLocal, startRow, dealRowCount, this->dSizeV_Align_, cntM);
        WaitFlag<HardEvent::MTE3_V>(SYNC_FDOUTPUT_BUF_FLAG);
        // 输出为二维的行×D 描述，形状只使用真实 D。
        uint32_t shapeArray[] = {dealRowCount, (uint32_t)constInfo_.dSizeV};
        tmpBmm2ResCastTensor.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
        if constexpr (IsSameType<OUTPUT_T, bfloat16_t>::value) {
            Cast(tmpBmm2ResCastTensor, accumOutLocal, AscendC::RoundMode::CAST_RINT,
                 dealRowCount * this->dSizeV_Align_);
        } else {
            Cast(tmpBmm2ResCastTensor, accumOutLocal, AscendC::RoundMode::CAST_ROUND,
                 dealRowCount * this->dSizeV_Align_);
        }
        SetFlag<HardEvent::V_MTE3>(SYNC_FDOUTPUT_BUF_FLAG);
        WaitFlag<HardEvent::V_MTE3>(SYNC_FDOUTPUT_BUF_FLAG);
        Bmm2DataCopyOutTrans(tmpBmm2ResCastTensor, startRow, dealRowCount, this->dSizeV_Align_);
        SetFlag<HardEvent::MTE3_V>(SYNC_FDOUTPUT_BUF_FLAG);
    }
    __aicore__ inline void CalcPreNextTokens()
    {
        actSeqLensQ_ = qActSeqLensParser_->GetActualSeqLength(taskInfo_.bIdx);
        actSeqLensKv_ = kvActSeqLensParser_->GetActualSeqLength(taskInfo_.bIdx);
        int64_t safePreToken = constInfo_.preTokens;
        int64_t safeNextToken = constInfo_.nextTokens;

        fa_base_vector::GetSafeActToken(actSeqLensQ_, actSeqLensKv_, safePreToken, safeNextToken,
                                        constInfo_.sparseMode);

        if (constInfo_.sparseMode == BAND) {
            preTokensPerBatch_ = safePreToken;
            nextTokensPerBatch_ = actSeqLensKv_ - actSeqLensQ_ + safeNextToken;
        } else if ((constInfo_.sparseMode == DEFAULT_MASK) && HAS_MASK) {
            nextTokensPerBatch_ = safeNextToken;
            preTokensPerBatch_ = actSeqLensKv_ - actSeqLensQ_ + safePreToken;
        } else {
            nextTokensPerBatch_ = actSeqLensKv_ - actSeqLensQ_;
            preTokensPerBatch_ = 0;
        }
    }

    template <typename UBOUT_T>
    __aicore__ inline void DealInvalidRows(LocalTensor<UBOUT_T>& attenOutUb, uint32_t startRow, uint32_t dealRowCount,
                                           uint32_t columnCount)
    {
        if constexpr (!HAS_MASK) {
            return;
        }

        if (constInfo_.sparseMode == ALL_MASK || constInfo_.sparseMode == LEFT_UP_CAUSAL) {
            return;
        }

        fa_base_vector::InvalidRowParams params{
            .actS1Size = actSeqLensQ_,
            .gSize = static_cast<uint64_t>(constInfo_.realGSize),
            .gS1Idx = taskInfo_.gS1Idx + startRow,
            .dealRowCount = dealRowCount,
            .columnCount = columnCount,
            .preTokensPerBatch = preTokensPerBatch_,
            .nextTokensPerBatch = nextTokensPerBatch_,
        };

        fa_base_vector::InvalidRows<UBOUT_T, GeInputUbFormat<layout>()> invalidRows;
        invalidRows(attenOutUb, params);
    }

    template <typename UBOUT_T>
    __aicore__ inline void DealInvalidMaskRows(LocalTensor<UBOUT_T>& attenOutUb, uint32_t startRow,
                                               uint32_t dealRowCount, uint32_t columnCount, uint32_t cntM)
    {
        if constexpr (!HAS_MASK) {
            return;
        }
        if (constInfo_.sparseMode != DEFAULT_MASK && constInfo_.sparseMode != ALL_MASK) {
            return;
        }
        LocalTensor<T> lseMaxUb = (cntM & 1) == 0 ? fdLseMaxUbBuf1_ : fdLseMaxUbBuf2_;

        // max 与输出均已定位到当前行块；判值直接复用 ComputeScaleValue 生成的 UB，
        // DN/ND 与 LSE 使用相同 FP32 EQ 语义，合法 max=0 不会被独立常量误判。
        static_assert(IsSameType<UBOUT_T, float>::value, "FD invalid-mask epilogue expects FP32 accumulators");
        AscendC::PipeBarrier<PIPE_V>();
        ClearInvalidFdRowsVF((__ubuf__ float*)attenOutUb.GetPhyAddr(), (__ubuf__ float*)lseMaxUb.GetPhyAddr(),
                             (__ubuf__ float*)fdMinCheckBuf_.GetPhyAddr(), dealRowCount, columnCount);
        AscendC::PipeBarrier<PIPE_V>();
    }

public:
    __aicore__ inline void FlashDecode(FDparamsX& fd)
    {
        if (!fd.fdCoreEnable || fd.mLen == 0U) {
            return;
        }
        // 在减一、除法、初始化事件与访问 GM 之前校验 metadata；外层所有 AIV 仍参与双屏障。
        const uint64_t splitCount = fd.fdS2SplitNum;
        const uint64_t outputSlotElements = static_cast<uint64_t>(mBaseSize) * constInfo_.dSizeV;
        const uint64_t lseSlotElements = static_cast<uint64_t>(mBaseSize) * BLOCK_ELEMENT_NUM;
        const uint64_t workspaceEnd = static_cast<uint64_t>(fd.fdWorkspaceIdx) + splitCount;
        if (splitCount == 0U || splitCount > constInfo_.coreNum || outputSlotElements == 0U ||
            constInfo_.realN2Size == 0U ||
            static_cast<uint64_t>(fd.fdBN2Idx) >= static_cast<uint64_t>(constInfo_.bSize) * constInfo_.realN2Size ||
            fd.mStart >= mBaseSize || fd.mLen > mBaseSize - fd.mStart ||
            workspaceEnd > constInfo_.accumOutSize / outputSlotElements ||
            workspaceEnd > constInfo_.logSumExpSize / lseSlotElements) {
            return;
        }
        const uint64_t batch = fd.fdBN2Idx / constInfo_.realN2Size;
        const uint64_t taskEnd = static_cast<uint64_t>(fd.fdMIdx) * mBaseSize + fd.mStart + fd.mLen;
        if (taskEnd > qActSeqLensParser_->GetActualSeqLength(batch) * constInfo_.realGSize) {
            return;
        }
        // VF 最后一次无掩码加载仍读取 256 字节，不能只按有效行数计算末地址。
        const uint64_t safeRows = splitCount == 1U ?
                                      FD_BALANCE_M_BASE :
                                      (FD_LSE_BUF_BYTES - REPEAT_BLOCK_BYTE) / ((splitCount - 1U) * BYTE_BLOCK);
        uint32_t fdBalanceMBaseSize = safeRows < FD_BALANCE_M_BASE ? safeRows : FD_BALANCE_M_BASE;
        if (fdBalanceMBaseSize == 0U || splitCount * fdBalanceMBaseSize * BYTE_BLOCK > FD_LSE_BUF_BYTES) {
            return;
        }
        preLoadNum_ = splitCount < 2U ? splitCount : 2U; // 两个 partial 输入槽最多预加载两份。
        AllocEventID();
        uint32_t fdBalanceMSplitNum = (fd.mLen + fdBalanceMBaseSize - 1) / fdBalanceMBaseSize;
        uint32_t fdBalanceMTailSize =
            (fd.mLen % fdBalanceMBaseSize == 0) ? fdBalanceMBaseSize : fd.mLen % fdBalanceMBaseSize;

        uint32_t reduceGlobaLoop = 0;
        uint32_t reduceMLoop = 0;

        uint32_t tmpFdS1gOuterMStart = 0;
        uint32_t tmpFdS1gOuterMEnd = fdBalanceMSplitNum - 1;
        taskInfo_.bIdx = fd.fdBN2Idx / constInfo_.realN2Size;
        taskInfo_.n2Idx = fd.fdBN2Idx % constInfo_.realN2Size;
        taskInfo_.gS1Idx = fd.fdMIdx * mBaseSize;
        taskInfo_.actualCombineLoopSize = fd.fdS2SplitNum; // 当前规约任务kv方向有几份
        uint64_t combineTaskPrefixSum = fd.fdWorkspaceIdx;
        uint64_t taskOffset = combineTaskPrefixSum * mBaseSize;

        for (uint32_t fdS1gOuterMIdx = tmpFdS1gOuterMStart; fdS1gOuterMIdx <= tmpFdS1gOuterMEnd;
             ++fdS1gOuterMIdx) { // 左闭右闭
            uint32_t actualGSplitSize = fdBalanceMBaseSize;
            if (fdS1gOuterMIdx == fdBalanceMSplitNum - 1) {
                actualGSplitSize = fdBalanceMTailSize;
            }
            uint32_t startRow = fd.mStart + fdS1gOuterMIdx * fdBalanceMBaseSize;

            LocalTensor<T> lseExp = fdLseExpBuf_;
            LocalTensor<T> reduceOut = fdReduceBuf_;
            WaitFlag<HardEvent::V_MTE2>(SYNC_LSE_MAX_SUM_BUF1_FLAG + (reduceMLoop & 1));
            CopyLseIn(startRow, actualGSplitSize, taskOffset, reduceMLoop);
            SetFlag<HardEvent::MTE2_V>(SYNC_LSE_MAX_SUM_BUF1_FLAG + (reduceMLoop & 1));
            for (uint32_t preLoadIdx = 0; preLoadIdx < preLoadNum_; ++preLoadIdx) {
                LocalTensor<T> mm2Res = ((reduceGlobaLoop + preLoadIdx) & 1) == 0 ? fdMm2ResBuf1_ : fdMm2ResBuf2_;
                WaitFlag<HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG + ((reduceGlobaLoop + preLoadIdx) & 1));
                CopyAccumOutIn(mm2Res, preLoadIdx, taskOffset + startRow, actualGSplitSize);
                SetFlag<HardEvent::MTE2_V>(SYNC_MM2RES_BUF1_FLAG + ((reduceGlobaLoop + preLoadIdx) & 1));
            }
            WaitFlag<HardEvent::MTE2_V>(SYNC_LSE_MAX_SUM_BUF1_FLAG + (reduceMLoop & 1));
            if (constInfo_.isSoftmaxLseEnable) {
                // 等待上一行块搬出完成后，才允许 ComputeScaleValue 覆盖同一个 LSE 槽。
                WaitFlag<HardEvent::MTE3_V>(SYNC_LSEOUTPUT_BUF_FLAG);
            }
            ComputeScaleValue(lseExp, actualGSplitSize, taskInfo_.actualCombineLoopSize, reduceMLoop, startRow);
            if (constInfo_.isSoftmaxLseEnable) {
                SetFlag<HardEvent::V_MTE3>(SYNC_LSEOUTPUT_BUF_FLAG);
            }
            SetFlag<HardEvent::V_MTE2>(SYNC_LSE_MAX_SUM_BUF1_FLAG + (reduceMLoop & 1));
            CalcPreNextTokens();
            if (constInfo_.isSoftmaxLseEnable) {
                // lse行无效在ComputeScaleValue的VF计算时已经进行了赋值inf处理
                LocalTensor<T> maxLseUb = fdLseUbBuf_;
                WaitFlag<HardEvent::V_MTE3>(SYNC_LSEOUTPUT_BUF_FLAG);
                uint32_t mOffset = taskInfo_.gS1Idx + startRow;
                if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
                    // LSE 输出改为 N-major 排布 [N2*G, T]: N 在外, T 在内
                    uint32_t prefixBS1 = qActSeqLensParser_->GetTBase(taskInfo_.bIdx);
                    uint64_t bN2Offset =
                        static_cast<uint64_t>(taskInfo_.n2Idx) * constInfo_.realGSize * constInfo_.t1Size + prefixBS1;
                    if constexpr (useDn) {
                        DataCopySoftmaxLseTNDtoNTArch35NoGS1Merge<T, ConstInfoX>(softmaxLseGm_, maxLseUb, bN2Offset,
                                                                                 mOffset, actualGSplitSize, constInfo_);
                    } else {
                        DataCopySoftmaxLseTNDtoNTArch35<T, ConstInfoX>(softmaxLseGm_, maxLseUb, bN2Offset, mOffset,
                                                                       actualGSplitSize, constInfo_);
                    }
                } else if constexpr (layout == LayOutTypeEnum::LAYOUT_NTD) {
                    uint32_t prefixBS1 = qActSeqLensParser_->GetTBase(taskInfo_.bIdx);
                    uint32_t s1Size = qActSeqLensParser_->GetActualSeqLength(taskInfo_.bIdx);
                    uint64_t bN2Offset =
                        static_cast<uint64_t>(prefixBS1) * constInfo_.realGSize * constInfo_.realN2Size +
                        static_cast<uint64_t>(taskInfo_.n2Idx) * constInfo_.realGSize;
                    DataCopySoftmaxLseNTDArch35<T, ConstInfoX>(softmaxLseGm_, maxLseUb, bN2Offset, mOffset,
                                                               actualGSplitSize, constInfo_, s1Size);
                } else if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH) {
                    uint64_t bN2Offset =
                        static_cast<uint64_t>(taskInfo_.bIdx) * constInfo_.realGSize * constInfo_.realN2Size *
                            constInfo_.s1Size +
                        static_cast<uint64_t>(taskInfo_.n2Idx) * constInfo_.realGSize * constInfo_.s1Size;
                    uint64_t qActSeqLens = qActSeqLensParser_->GetActualSeqLength(taskInfo_.bIdx);
                    uint64_t s1LeftPaddingSize = 0;
                    DataCopySoftmaxLseBSNDArch35<T, ConstInfoX>(softmaxLseGm_, maxLseUb, bN2Offset, mOffset,
                                                                actualGSplitSize, constInfo_, s1LeftPaddingSize);
                } else {
                    uint64_t bN2Offset =
                        static_cast<uint64_t>(taskInfo_.bIdx) * constInfo_.realGSize * constInfo_.realN2Size *
                            constInfo_.s1Size +
                        static_cast<uint64_t>(taskInfo_.n2Idx) * constInfo_.realGSize * constInfo_.s1Size;
                    uint64_t qActSeqLens = qActSeqLensParser_->GetActualSeqLength(taskInfo_.bIdx);
                    uint64_t s1LeftPaddingSize = 0;
                    DataCopySoftmaxLseBNSDArch35<T, ConstInfoX>(softmaxLseGm_, maxLseUb, bN2Offset, mOffset,
                                                                actualGSplitSize, constInfo_, qActSeqLens,
                                                                s1LeftPaddingSize);
                }
                SetFlag<HardEvent::MTE3_V>(SYNC_LSEOUTPUT_BUF_FLAG);
            }

            for (uint32_t i = 0; i < taskInfo_.actualCombineLoopSize; ++i) {
                LocalTensor<T> mm2Res = (reduceGlobaLoop & 1) == 0 ? fdMm2ResBuf1_ : fdMm2ResBuf2_;
                if (i >= preLoadNum_) {
                    WaitFlag<HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG + (reduceGlobaLoop & 1));
                    CopyAccumOutIn(mm2Res, i, taskOffset + startRow, actualGSplitSize);
                    SetFlag<HardEvent::MTE2_V>(SYNC_MM2RES_BUF1_FLAG + (reduceGlobaLoop & 1));
                }
                WaitFlag<HardEvent::MTE2_V>(SYNC_MM2RES_BUF1_FLAG + (reduceGlobaLoop & 1));
                ReduceFinalRes(reduceOut, mm2Res, lseExp, i, actualGSplitSize);
                SetFlag<HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG + (reduceGlobaLoop & 1));
                reduceGlobaLoop += 1;
            }
            CopyFinalResOut(reduceOut, startRow, actualGSplitSize, reduceMLoop);
            reduceMLoop += 1;
        }
        FreeEventID();
    }
};

template <typename INPUT_T, typename T, typename OUTPUT_T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          LayOutTypeEnum outLayout = LayOutTypeEnum::None, S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, bool hasAtten = false, uint8_t KvLayoutType = 0,
          bool useDn = false>
class QuantFlashAttnBlockVecFlashDecodeDummy {
public:
    using ConstInfoX = ConstInfo_t;
    __aicore__ inline QuantFlashAttnBlockVecFlashDecodeDummy(ConstInfoX& constInfo){};
};

} // namespace BaseApi
#endif
