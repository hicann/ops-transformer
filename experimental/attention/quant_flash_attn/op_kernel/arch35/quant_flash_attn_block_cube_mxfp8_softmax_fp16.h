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
 * \brief MxFP8 Softmax FP16 Cube Block（AIC 侧 L1/L0 静态 Tensor 分配 + C1/F1/C2/F2 计算接口）
 */

#ifndef QUANT_FLASH_ATTN_BLOCK_CUBE_MXFP8_SOFTMAX_FP16_H_
#define QUANT_FLASH_ATTN_BLOCK_CUBE_MXFP8_SOFTMAX_FP16_H_

#include "kernel_operator.h"
#include "quant_flash_attn_common_def_mxfp8_softmax_fp16.h"
#include "../../common/op_kernel/memcopy/attn/fa_gm_tensor.h"
#include "../../common/op_kernel/memcopy/attn/gm_coord.h"
#include "../../common/op_kernel/memcopy/attn/parser.h"

using namespace AscendC;

#include "../../common/op_kernel/memcopy/attn/attn_copy_gm_to_l1.h"
#include "../../common/op_kernel/memcopy/attn/fa_l1_tensor.h"

namespace QFA_KERNEL {

static constexpr AscendC::FixpipeConfig CFG_ROW_MAJOR_UB{AscendC::CO2Layout::ROW_MAJOR, true};

template <typename QUANT_T, typename SCALE_T>
class QuantFlashAttnBlockCubeMxfp8SoftmaxFp16 {
public:
    using MM_OUT_T = float;
    using SOFTMAX_T = half;

    static constexpr uint32_t mBaseSize = S1_BASE_SIZE;  // 256
    static constexpr uint32_t s2BaseSize = S2_BASE_SIZE; // 256
    static constexpr uint32_t dBaseSize = D_BASE_SIZE;   // 128
    static constexpr uint32_t dVBaseSize = DV_BASE_SIZE; // 128

private:
    static constexpr uint32_t S2_SPLIT_SIZE = S2_SUB_LOOP_SIZE; // 128, C1 subLoop M 轴
    static constexpr uint32_t MM2_SUB_K_SIZE = 256U;            // 256, one PV K tile
    static constexpr uint32_t MXFP_GROUP_SIZE = 32U;            // MxFP8 量化组大小（每 32 元素共享 scale）

    // ===== L1 私有区尺寸常量 =====
    static constexpr uint32_t L1_Q_SINGLE_SLOT_SIZE = 128 * 256; // 32KB
    static constexpr uint32_t L1_Q_BUFCNT = 2;
    static constexpr uint32_t L1_QSCALE_SINGLE_SLOT_SIZE = 1024; // 1KB
    static constexpr uint32_t L1_QSCALE_BUFCNT = 2;
    // Three shared K/V slots; each copy prefetches two consecutive S2 tiles.
    // Each slot is [512,128] fp8; both halves remain live until their consumers finish.
    static constexpr uint32_t L1_KV_SINGLE_SLOT_SIZE = 512 * 128; // 64KB
    static constexpr uint32_t L1_KV_BUFCNT = 3;
    static constexpr uint32_t L1_KVSCALE_SINGLE_SLOT_SIZE = 2048; // 2KB per pair
    static constexpr uint32_t L1_KVSCALE_BUFCNT = 3;
    static constexpr uint32_t L1_ONEFILL_V_SIZE = 144 * 256;                          // 36KB
    static constexpr uint32_t L1_ONEFILL_VSCALE_SIZE = 1152;                          // 1152B
    static constexpr uint16_t ONE_FILL_BLOCK_CNT = L1_ONEFILL_V_SIZE / 32;            // 1152
    static constexpr uint16_t ONE_FILL_SCALE_BLOCK_CNT = L1_ONEFILL_VSCALE_SIZE / 32; // 36
    static constexpr uint16_t FP8_ONE_U16 = 0x3838U;                                  // fp8 e4m3 1.0, byte pair
    static constexpr uint16_t E8M0_ONE_U16 = 0x7F7FU;                                 // e8m0 2^0, byte pair

    // ===== L1 私有区 BASE（从共享区末尾链式推导）=====
    static constexpr uint32_t L1_Q_BASE = L1_SHARED_REGION_SIZE;
    static constexpr uint32_t L1_QSCALE_BASE = L1_Q_BASE + L1_Q_BUFCNT * L1_Q_SINGLE_SLOT_SIZE;
    static constexpr uint32_t L1_KV_BASE = L1_QSCALE_BASE + L1_QSCALE_BUFCNT * L1_QSCALE_SINGLE_SLOT_SIZE;
    static constexpr uint32_t L1_KVSCALE_BASE = L1_KV_BASE + L1_KV_BUFCNT * L1_KV_SINGLE_SLOT_SIZE;
    static constexpr uint32_t L1_ONEFILL_V_BASE = L1_KVSCALE_BASE + L1_KVSCALE_BUFCNT * L1_KVSCALE_SINGLE_SLOT_SIZE;
    static constexpr uint32_t L1_ONEFILL_VSCALE_BASE = L1_ONEFILL_V_BASE + L1_ONEFILL_V_SIZE;

    // ===== L0 尺寸常量 =====
    static constexpr uint32_t L0A_MM1_SIZE = 128 * 128;             // 16KB, single K slot
    static constexpr uint32_t MM2_L0A_M = dVBaseSize + ROWSUM_ROWS; // 128+16 = 144
    static constexpr uint32_t MM2_L0A_K = MM2_SUB_K_SIZE;           // 256
    static constexpr uint32_t L0A_MM2_SLOT_SIZE = MM2_L0A_M * MM2_L0A_K;
    static constexpr uint32_t L0A_MM2_SIZE = L0A_MM2_SLOT_SIZE; // 36KB, single V_ext slot
    static constexpr uint32_t L0B_Q_SIZE = 128 * 256;           // 32KB, Q 常驻
    static constexpr uint32_t L0B_P_SLOT_SIZE = 128 * 256; // fp8 [S1半=128, S2=256]，与 L1_P_SINGLE_SLOT_SIZE 同构
    static constexpr uint32_t L0B_P_SIZE = L0B_P_SLOT_SIZE; // 32KB, single P slot
    static constexpr uint32_t L0C_SIZE = 2 * 128 * 256;     // 256KB, C1+C2 ring

    // ===== L0A BASE =====
    static constexpr uint32_t L0A_MM1_BASE = 0;
    static constexpr uint32_t L0A_MM2_BASE = L0A_MM1_BASE + L0A_MM1_SIZE;

    // ===== L0B BASE =====
    static constexpr uint32_t L0B_Q_BASE = 0;
    static constexpr uint32_t L0B_P_BASE = L0B_Q_BASE + L0B_Q_SIZE;

    using QSeqParser = ActualSeqLensParser<ActualSeqLensMode::BY_BATCH, int32_t>;
    using KvSeqParser = ActualSeqLensParser<ActualSeqLensMode::BY_BATCH, int32_t>;

    const ConstInfo& constInfo_;
    bool fullTileUnitReuse_ = false;

    CopyQueryGmToL1<QUANT_T, GmFormat::BNGSD, L1Format::NZ> copyQueryGmToL1_;
    CopyQueryScaleGmToL1<SCALE_T, GmFormat::BNGSD> copyQueryScaleGmToL1_;
    CopyKvGmToL1<QUANT_T, GmFormat::BNSD> copyKvGmToL1_;
    CopyKeyScaleGmToL1<SCALE_T, GmFormat::BNSD> copyKeyScaleGmToL1_;
    CopyValueScaleGmToL1<SCALE_T, GmFormat::BNSD> copyValueScaleGmToL1_;

    // ===== GM 输入（BNSD 单模板，直接 GlobalTensor）=====
    FaGmTensor<QUANT_T, GmFormat::BNGSD, int32_t> queryGm_;
    FaGmTensor<SCALE_T, GmFormat::BNGSD, int32_t> queryScaleGm_;
    FaGmTensor<QUANT_T, GmFormat::BNSD, int32_t> keyGm_;
    FaGmTensor<SCALE_T, GmFormat::BNSD, int32_t> keyScaleGm_;
    FaGmTensor<QUANT_T, GmFormat::BNSD, int32_t> valueGm_;
    FaGmTensor<SCALE_T, GmFormat::BNSD, int32_t> valueScaleGm_;

    // ===== L1 Buffer =====
    // --- 共享区（P/pScale，AIC/AIV 同偏移，常量来源 common_def.h L1 v3）---
    // P 槽核间交错布局下双核槽不连续——统一走宽视图 + L1_PSLOT(core, ring) 计算基址
    LocalTensor<QUANT_T> l1SharedTensor_;   // L1 共享区整域视图（P 数据，交错槽）
    LocalTensor<SCALE_T> pScaleL1V0Tensor_; // V0 pscale task grids (3 x 1280B).
    LocalTensor<SCALE_T> pScaleL1V1Tensor_; // V1 pscale task grids (3 x 1280B).

    // --- 私有区（Q 及 scale / KV 合用 ring）---
    LocalTensor<QUANT_T> qL1Tensor_;
    LocalTensor<SCALE_T> qScaleL1Tensor_;
    LocalTensor<QUANT_T> kvL1Tensor_; // Three shared pair-prefetch slots
    LocalTensor<SCALE_T> kvScaleL1Tensor_;

    // --- 刷 1 常驻（独立 L1 buffer，一次 Fill 全程只读）---
    LocalTensor<QUANT_T> oneFillVL1Tensor_;
    LocalTensor<SCALE_T> oneFillVScaleL1Tensor_;

    // ===== L0 Buffer =====
    LocalTensor<mx_fp8_e4m3_t> mm1L0ATensor_; // L0A, K 左矩阵双缓冲 [128,128]×2
    LocalTensor<mx_fp8_e4m3_t> mm2L0ATensor_; // L0A, V_ext [144,128]（前 128 行 V + 后 16 行全 1）
    LocalTensor<mx_fp8_e4m3_t> qL0BTensor_;   // L0B, Q 常驻槽 [128,256]
    LocalTensor<mx_fp8_e4m3_t> pL0BTensor_;   // L0B, P 单块 [128,128]（C2 串行无 buffering）
    LocalTensor<MM_OUT_T> mmL0CTensor_;       // L0C, C1[128,256]+C2[144,128] ring 双槽 [128,256]×2

    // ===== UB（Fixpipe 目标，TOTAL ≤ 256KB）=====
    // UB 常量统一引用 common_def 跨核契约（UB_MM1RES_*/UB_MM2RES_*，单一来源，VectorBlock 同源）
    LocalTensor<SOFTMAX_T> mm1ResUB_;
    LocalTensor<MM_OUT_T> mm2ResUB_;

    // ===== Event ID（内部流水同步，SetFlag/WaitFlag 显式编排）=====
    // mte2 ↔ mte1：Q（独立 2 槽）/ KV 合用 ring（每槽一事件，DN 同款 KV_EVENT0~3）
    static constexpr uint32_t Q_EVENT0 = EVENT_ID2;
    static constexpr uint32_t Q_EVENT1 = EVENT_ID3;
    uint32_t qBufId_ = 0;
    static constexpr uint32_t KV_EVENT0 = EVENT_ID4;
    static constexpr uint32_t KV_EVENT1 = EVENT_ID5;
    static constexpr uint32_t KV_EVENT2 = EVENT_ID6;
    static constexpr uint32_t KV_EVENT3 = EVENT_ID7;
    uint32_t kvBufId_ = 0;
    uint32_t kDataBufId_ = 0;
    uint32_t vDataBufId_ = 0;
    uint32_t mm2VSlotBase_ = 0; // K（C1）与 V（C2）按使用顺序轮转，% 4

    // mte1 ↔ m：Q→L0B / K→L0A（使用 EVENT_ID2~5，Q_L0B=2/3, K_L0A=4/5）
    static constexpr uint32_t Q_L0B_EVENT0 = EVENT_ID2;
    static constexpr uint32_t K_L0A_EVENT0 = EVENT_ID4;
    static constexpr uint32_t K_L0A_EVENT1 = EVENT_ID5;
    uint32_t kL0aBufId_ = 0;

    // C2 L0AB（P L0B / V L0A 单块，事件同步用固定 bufId=0）
    static constexpr uint32_t MM2_L0AB_EVENT0 = EVENT_ID6;
    uint32_t mm2L0abBufId_ = 0;

    // m ↔ fix：L0C（使用 EVENT_ID2~3，C1 ring 双缓冲）
    static constexpr uint32_t L0C_EVENT0 = EVENT_ID2;
    static constexpr uint32_t L0C_EVENT1 = EVENT_ID3;
    uint32_t l0cBufId_ = 0;

    // ===== Q/K 搬运 helper（BNSD 布局，GM→L1）=====
    __aicore__ inline void InitQBuffer(uint32_t b, uint32_t n2, uint32_t g, uint32_t s1, uint32_t d,
                                       FaGmTensor<QUANT_T, GmFormat::BNGSD, int32_t>& qGmTensor, __gm__ uint8_t* gm,
                                       QSeqParser& qSeqParser)
    {
        qGmTensor.gmTensor.SetGlobalBuffer((__gm__ QUANT_T*)gm);
        qGmTensor.offsetCalculator.Init(b, n2, g, s1, d, qSeqParser);
    }

    __aicore__ inline void InitQScaleBuffer(uint32_t b, uint32_t n2, uint32_t g, uint32_t s1, uint32_t d,
                                            FaGmTensor<SCALE_T, GmFormat::BNGSD, int32_t>& qScaleGmTensor,
                                            __gm__ uint8_t* gm, QSeqParser& qSeqParser)
    {
        qScaleGmTensor.gmTensor.SetGlobalBuffer((__gm__ SCALE_T*)gm);
        qScaleGmTensor.offsetCalculator.Init(b, n2, g, s1, d, qSeqParser);
    }

    __aicore__ inline void InitKBuffer(uint32_t b, uint32_t n2, uint32_t s2, uint32_t d,
                                       FaGmTensor<QUANT_T, GmFormat::BNSD, int32_t>& kVGmTensor, __gm__ uint8_t* gm,
                                       KvSeqParser& kvSeqParser)
    {
        kVGmTensor.gmTensor.SetGlobalBuffer((__gm__ QUANT_T*)gm);
        kVGmTensor.offsetCalculator.Init(b, n2, s2, d, kvSeqParser);
    }

    __aicore__ inline void InitKScaleBuffer(uint32_t b, uint32_t n2, uint32_t s2, uint32_t d,
                                            FaGmTensor<SCALE_T, GmFormat::BNSD, int32_t>& kScaleGmTensor,
                                            __gm__ uint8_t* gm, KvSeqParser& kvSeqParser)
    {
        kScaleGmTensor.gmTensor.SetGlobalBuffer((__gm__ SCALE_T*)gm);
        kScaleGmTensor.offsetCalculator.Init(b, n2, s2, d, kvSeqParser);
    }

    __aicore__ inline void InitVBuffer(uint32_t b, uint32_t n2, uint32_t s2, uint32_t d,
                                       FaGmTensor<QUANT_T, GmFormat::BNSD, int32_t>& vGmTensor, __gm__ uint8_t* gm,
                                       KvSeqParser& kvSeqParser)
    {
        vGmTensor.gmTensor.SetGlobalBuffer((__gm__ QUANT_T*)gm);
        vGmTensor.offsetCalculator.Init(b, n2, s2, d, kvSeqParser);
    }

    __aicore__ inline void InitVScaleBuffer(uint32_t b, uint32_t n2, uint32_t s2, uint32_t d,
                                            FaGmTensor<SCALE_T, GmFormat::BNSD, int32_t>& vScaleGmTensor,
                                            __gm__ uint8_t* gm, KvSeqParser& kvSeqParser)
    {
        vScaleGmTensor.gmTensor.SetGlobalBuffer((__gm__ SCALE_T*)gm);
        vScaleGmTensor.offsetCalculator.Init(b, n2, s2, d, kvSeqParser);
    }

    __aicore__ inline void AllocEventID()
    {
        SetFlag<HardEvent::MTE1_MTE2>(Q_EVENT0);
        SetFlag<HardEvent::MTE1_MTE2>(Q_EVENT1);
        SetFlag<HardEvent::MTE1_MTE2>(KV_EVENT0);
        SetFlag<HardEvent::MTE1_MTE2>(KV_EVENT1);
        SetFlag<HardEvent::MTE1_MTE2>(KV_EVENT2);
        SetFlag<HardEvent::MTE1_MTE2>(KV_EVENT3);

        SetFlag<HardEvent::M_MTE1>(Q_L0B_EVENT0);
        SetFlag<HardEvent::M_MTE1>(K_L0A_EVENT0);
        SetFlag<HardEvent::M_MTE1>(K_L0A_EVENT1);

        SetFlag<HardEvent::M_MTE1>(MM2_L0AB_EVENT0);
        SetFlag<HardEvent::M_MTE1>(MM2_L0AB_EVENT0 + 1U);

        if (fullTileUnitReuse_) {
            LocalTensor<uint64_t> empty;
            SetFixPipeConfig(empty, empty, true);
            PipeBarrier<PIPE_ALL>();
        } else {
            SetFlag<HardEvent::FIX_M>(L0C_EVENT0);
            SetFlag<HardEvent::FIX_M>(L0C_EVENT1);
        }
    }

    __aicore__ inline void FreeEventID()
    {
        WaitFlag<HardEvent::MTE1_MTE2>(Q_EVENT0);
        WaitFlag<HardEvent::MTE1_MTE2>(Q_EVENT1);
        WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT0);
        WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT1);
        WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT2);
        WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT3);

        WaitFlag<HardEvent::M_MTE1>(Q_L0B_EVENT0);
        WaitFlag<HardEvent::M_MTE1>(K_L0A_EVENT0);
        WaitFlag<HardEvent::M_MTE1>(K_L0A_EVENT1);

        WaitFlag<HardEvent::M_MTE1>(MM2_L0AB_EVENT0);
        WaitFlag<HardEvent::M_MTE1>(MM2_L0AB_EVENT0 + 1U);

        if (fullTileUnitReuse_) {
            // Drain all consumers before clearing hardware unit state at kernel exit.
            PipeBarrier<PIPE_ALL>();
            LocalTensor<uint64_t> empty;
            SetFixPipeConfig(empty, empty, true);
            PipeBarrier<PIPE_ALL>();
        } else {
            WaitFlag<HardEvent::FIX_M>(L0C_EVENT0);
            WaitFlag<HardEvent::FIX_M>(L0C_EVENT1);
        }
    }

public:
    __aicore__ inline explicit QuantFlashAttnBlockCubeMxfp8SoftmaxFp16(ConstInfo& constInfo)
        : constInfo_(constInfo)
    {}

    __aicore__ inline void SetFullTileUnitReuse(bool enable)
    {
        fullTileUnitReuse_ = enable;
    }

    __aicore__ inline void InitInput(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
                                     __gm__ uint8_t* dequantScaleQuery, __gm__ uint8_t* dequantScaleKey,
                                     __gm__ uint8_t* dequantScaleValue, QSeqParser& qSeqParser,
                                     KvSeqParser& kvSeqParser)
    {
        // Q/Qscale 用折叠头维（realN2 = n2×G, realG = 1，与 kernel 不合轴调度一致）：
        // 拷贝坐标传 realN2Idx（扁平头索引），gS1Idx 为纯 s1 索引（g 恒为 0）——
        // 若用分立 (n2Size, gSize) 初始化 + n2Idx 寻址，GQA 时 g>0 的头全部错读 head 0
        InitQBuffer(constInfo_.bSize, constInfo_.realN2Size, constInfo_.realGSize, constInfo_.s1Size, constInfo_.dSize,
                    queryGm_, query, qSeqParser);
        InitQScaleBuffer(constInfo_.bSize, constInfo_.realN2Size, constInfo_.realGSize, constInfo_.s1Size,
                         constInfo_.dSize / MXFP_GROUP_SIZE, queryScaleGm_, dequantScaleQuery, qSeqParser);
        InitKBuffer(constInfo_.bSize, constInfo_.n2Size, constInfo_.s2Size, constInfo_.dSize, keyGm_, key, kvSeqParser);
        InitKScaleBuffer(constInfo_.bSize, constInfo_.n2Size, constInfo_.s2Size, constInfo_.dSize / MXFP_GROUP_SIZE,
                         keyScaleGm_, dequantScaleKey, kvSeqParser);
        InitVBuffer(constInfo_.bSize, constInfo_.n2Size, constInfo_.s2Size, constInfo_.dSize, valueGm_, value,
                    kvSeqParser);
        InitVScaleBuffer(constInfo_.bSize, constInfo_.n2Size,
                         ((constInfo_.s2Size + 63U) / 64U * 64U) / (2U * MXFP_GROUP_SIZE), // S2/64 组行
                         2U * constInfo_.dSize,                                            // 2×D 字节/行
                         valueScaleGm_, dequantScaleValue, kvSeqParser);
        (void)kvSeqParser;
    }

    __aicore__ inline void InitTensors()
    {
        AllocEventID();

        // ===== L1 共享区 =====
        // P 槽核间交错布局（common_def L1_PSLOT）：整域宽视图，P 基址由 LoadPToL0B 按槽计算
        l1SharedTensor_ = LocalTensor<QUANT_T>(TPosition::A1, 0, L1_SHARED_REGION_SIZE);
        pScaleL1V0Tensor_ =
            LocalTensor<SCALE_T>(TPosition::A1, L1_PSCALE_V0_BASE, L1_P_BUFCNT * L1_PSCALE_SINGLE_SLOT_SIZE);
        pScaleL1V1Tensor_ =
            LocalTensor<SCALE_T>(TPosition::A1, L1_PSCALE_V1_BASE, L1_P_BUFCNT * L1_PSCALE_SINGLE_SLOT_SIZE);

        // ===== L1 私有区 =====
        qL1Tensor_ = LocalTensor<QUANT_T>(TPosition::A1, L1_Q_BASE, L1_Q_BUFCNT * L1_Q_SINGLE_SLOT_SIZE);
        qScaleL1Tensor_ =
            LocalTensor<SCALE_T>(TPosition::A1, L1_QSCALE_BASE, L1_QSCALE_BUFCNT * L1_QSCALE_SINGLE_SLOT_SIZE);
        kvL1Tensor_ = LocalTensor<QUANT_T>(TPosition::A1, L1_KV_BASE, L1_KV_BUFCNT * L1_KV_SINGLE_SLOT_SIZE);
        kvScaleL1Tensor_ =
            LocalTensor<SCALE_T>(TPosition::A1, L1_KVSCALE_BASE, L1_KVSCALE_BUFCNT * L1_KVSCALE_SINGLE_SLOT_SIZE);
        oneFillVL1Tensor_ = LocalTensor<QUANT_T>(TPosition::A1, L1_ONEFILL_V_BASE, L1_ONEFILL_V_SIZE);
        oneFillVScaleL1Tensor_ = LocalTensor<SCALE_T>(TPosition::A1, L1_ONEFILL_VSCALE_BASE, L1_ONEFILL_VSCALE_SIZE);

        // ===== L0A =====
        mm1L0ATensor_ = LocalTensor<mx_fp8_e4m3_t>(TPosition::A2, L0A_MM1_BASE, L0A_MM1_SIZE);
        mm2L0ATensor_ = LocalTensor<mx_fp8_e4m3_t>(TPosition::A2, L0A_MM2_BASE, L0A_MM2_SIZE);

        // ===== L0B =====
        qL0BTensor_ = LocalTensor<mx_fp8_e4m3_t>(TPosition::B2, L0B_Q_BASE, L0B_Q_SIZE);
        pL0BTensor_ = LocalTensor<mx_fp8_e4m3_t>(TPosition::B2, L0B_P_BASE, L0B_P_SIZE);

        // ===== L0C =====
        mmL0CTensor_ = LocalTensor<MM_OUT_T>(TPosition::CO1, 0U, L0C_SIZE);

        static_assert(UB_MM2RES_BASE == 64U * 1024U, "common_def contract: mm2ResUB follows mm1ResUB 64KB");
        mm1ResUB_ = LocalTensor<SOFTMAX_T>(TPosition::VECCALC, UB_MM1RES_BASE, UB_MM1RES_SLOT * UB_MM1RES_BUFCNT);
        mm2ResUB_ = LocalTensor<MM_OUT_T>(TPosition::VECCALC, UB_MM2RES_BASE, UB_MM2RES_SIZE);

        static_assert(L1_ONEFILL_VSCALE_BASE + L1_ONEFILL_VSCALE_SIZE <= 512U * 1024U, "L1 overflow");
        static_assert(L0A_MM2_BASE + L0A_MM2_SIZE <= 64U * 1024U, "L0A overflow");
        static_assert(L0B_P_BASE + L0B_P_SIZE <= 64U * 1024U, "L0B overflow");
        InitOneFillBuffer();
        mm2L0abBufId_ = 0U;
        InitL0BufferForReduceSum();
    }

    __aicore__ inline void InitOneFillBuffer()
    {
        // The second 128-row P half always uses the final exponent (scale=1).
        InitConstValueParams<uint16_t> pScaleOnes(1U, L1_P_BUFCNT * L1_PSCALE_SINGLE_SLOT_SIZE / 32U, 0U, E8M0_ONE_U16);
        Fill(pScaleL1V0Tensor_.template ReinterpretCast<uint16_t>(), pScaleOnes);
        Fill(pScaleL1V1Tensor_.template ReinterpretCast<uint16_t>(), pScaleOnes);
        PipeBarrier<PIPE_MTE2>();
        InitConstValueParams<uint16_t> dataParams(1U, ONE_FILL_BLOCK_CNT, 0U, FP8_ONE_U16);
        Fill(oneFillVL1Tensor_.template ReinterpretCast<uint16_t>(), dataParams);
        PipeBarrier<PIPE_MTE2>();

        InitConstValueParams<uint16_t> scaleParams(1U, ONE_FILL_SCALE_BLOCK_CNT, 0U, E8M0_ONE_U16);
        Fill(oneFillVScaleL1Tensor_.template ReinterpretCast<uint16_t>(), scaleParams);
        PipeBarrier<PIPE_MTE2>();
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
    }

    __aicore__ inline void InitL0BufferForReduceSum()
    {
        LoadData2DParamsV2 dataParams;
        dataParams.mStartPosition = 0U;
        dataParams.kStartPosition = 0U;
        dataParams.mStep = MM2_L0A_M / 16U;
        dataParams.kStep = MM2_L0A_K / 32U;
        dataParams.srcStride = dataParams.mStep;
        dataParams.dstStride = dataParams.mStep;
        dataParams.ifTranspose = false;

        LoadData2DMxParams scaleParams;
        scaleParams.xStartPosition = 0U;
        scaleParams.yStartPosition = 0U;
        scaleParams.xStep = MM2_L0A_M / 16U;
        scaleParams.yStep = MM2_L0A_K / 64U; // y 方向上分形的个数
        scaleParams.srcStride = scaleParams.yStep;
        scaleParams.dstStride = scaleParams.yStep;

        LoadData(mm2L0ATensor_[mm2L0abBufId_ * L0A_MM2_SLOT_SIZE],
                 oneFillVL1Tensor_.template ReinterpretCast<QUANT_T>(), oneFillVScaleL1Tensor_, dataParams,
                 scaleParams);
    }

    __aicore__ inline void ReleaseTensors()
    {
        FreeEventID();
    }

    __aicore__ inline void ComputeMm1(RunInfoMxfp8SoftmaxFp16& runInfo, uint32_t subLoopIdx, bool isLastSubLoop)
    {
        // ———— Q 常驻载入（行首一次，同一行所有 K subLoop 复用）————
        if (unlikely(runInfo.isFirstS2Loop && subLoopIdx == 0)) {
            WaitFlag<HardEvent::MTE1_MTE2>(Q_EVENT0 + qBufId_);
            CopyQGmToL1(runInfo);
            CopyQScaleGmToL1(runInfo);
            SetFlag<HardEvent::MTE2_MTE1>(Q_EVENT0 + qBufId_);
            WaitFlag<HardEvent::MTE2_MTE1>(Q_EVENT0 + qBufId_);

            WaitFlag<HardEvent::M_MTE1>(Q_L0B_EVENT0);
            LoadQToL0B(runInfo);
            SetFlag<HardEvent::MTE1_M>(Q_L0B_EVENT0);
            WaitFlag<HardEvent::MTE1_M>(Q_L0B_EVENT0);
        }

        // ———— K 整块搬运（每 task 一次，仅 subLoopIdx==0；槽跨整个 C1 subLoop 循环持有，
        //     subLoop 切片由 LoadKToL0A 的 mStartPosition 在槽内寻址）————
        if (subLoopIdx == 0U && !runInfo.pairSecond) {
            WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT0 + kvBufId_);
            CopyKGmToL1(runInfo);
            CopyKScaleGmToL1(runInfo);
            SetFlag<HardEvent::MTE2_MTE1>(KV_EVENT0 + kvBufId_);
            WaitFlag<HardEvent::MTE2_MTE1>(KV_EVENT0 + kvBufId_);
            kDataBufId_ = kvBufId_;
            kvBufId_ = (kvBufId_ + 1U) % L1_KV_BUFCNT;
        }

        {
            if (!fullTileUnitReuse_) {
                WaitFlag<HardEvent::FIX_M>(L0C_EVENT0 + l0cBufId_);
            }
            WaitFlag<HardEvent::M_MTE1>(K_L0A_EVENT0 + kL0aBufId_);
            LoadKToL0A(runInfo, subLoopIdx);
            SetFlag<HardEvent::MTE1_M>(K_L0A_EVENT0 + kL0aBufId_);
            WaitFlag<HardEvent::MTE1_M>(K_L0A_EVENT0 + kL0aBufId_);

            uint32_t actKSize =
                (subLoopIdx + 1 == AttentionCommon::CeilDiv(runInfo.actSingleLoopS2Size, S2_SPLIT_SIZE)) ?
                    runInfo.actSingleLoopS2Size - subLoopIdx * S2_SPLIT_SIZE :
                    S2_SPLIT_SIZE;
            MatmulKQ(runInfo, actKSize);
            SetFlag<HardEvent::M_MTE1>(K_L0A_EVENT0 + kL0aBufId_);
            kL0aBufId_ = 0U; // K uses one slot; the remaining L0A holds the 256-wide PV operand.
            if (!(runInfo.actMSize == 256U && actKSize == 128U)) {
                SetFlag<HardEvent::M_FIX>(L0C_EVENT0 + l0cBufId_);
            }
        }

        // ———— K 槽在 C1 末尾释放（所有 subLoop 的 L0A 切片加载完成后）————
        if (isLastSubLoop) {
            if (runInfo.pairSecond || runInfo.isLastS2Loop) {
                SetFlag<HardEvent::MTE1_MTE2>(KV_EVENT0 + kDataBufId_);
            }
            if (unlikely(runInfo.isLastS2Loop)) {
                SetFlag<HardEvent::MTE1_MTE2>(Q_EVENT0 + qBufId_);
                SetFlag<HardEvent::M_MTE1>(Q_L0B_EVENT0);
                qBufId_ = (qBufId_ + 1U) % 2U;
            }
        }
    }

    __aicore__ inline void FixpipeMm1(RunInfoMxfp8SoftmaxFp16& runInfo, uint32_t subLoopIdx, uint32_t subBlockId)
    {
        // Tail MM1 blocks use M_FIX; complete blocks synchronize through unit flags.
        uint32_t actKSize = (subLoopIdx + 1 == AttentionCommon::CeilDiv(runInfo.actSingleLoopS2Size, S2_SPLIT_SIZE)) ?
                                runInfo.actSingleLoopS2Size - subLoopIdx * S2_SPLIT_SIZE :
                                S2_SPLIT_SIZE;

        const bool unitFlag = runInfo.actMSize == 256U && actKSize == 128U;
        if (subBlockId == 0U && !unitFlag) {
            WaitFlag<HardEvent::M_FIX>(L0C_EVENT0 + l0cBufId_);
        }
        uint32_t actVecS1Size = (subBlockId == 0U) ?
                                    AttentionCommon::Min(runInfo.actMSize, S1_SUB_BLOCK_SIZE) :
                                    (runInfo.actMSize > S1_SUB_BLOCK_SIZE ? runInfo.actMSize - S1_SUB_BLOCK_SIZE : 0U);

        FixpipeParamsArch3510<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = (actVecS1Size + 7U) >> 3 << 3;
        fixpipeParams.mSize = (actKSize + 1U) >> 1 << 1;
        fixpipeParams.srcStride = ((actKSize + 15U) / 16U) * 16U;
        fixpipeParams.dstStride = S1_SUB_BLOCK_SIZE; // 固定 pitch 128，nSize 恒 ≤ pitch，杜绝行覆盖
        fixpipeParams.quantPre = QuantMode_t::QF322F16_PRE;
        fixpipeParams.unitFlag = unitFlag ? 3 : 0;
        fixpipeParams.deqScalar = static_cast<uint64_t>(*reinterpret_cast<const int32_t*>(&constInfo_.scaleValue));
        fixpipeParams.dualDstCtl = 0; // 单目标模式（CANN PTO GetDualDstCtl 权威定义：0=Single/1=SplitM/2=SplitN）
        fixpipeParams.subBlockId = static_cast<bool>(subBlockId); // 函数参数：0→V0(S1前半)、1→V1(S1后半)

        fixpipeParams.params.ndNum = 1U;
        fixpipeParams.params.srcNdStride = fixpipeParams.mSize;
        fixpipeParams.params.dstNdStride = fixpipeParams.dstStride;

        uint32_t l0cSlotOff = l0cBufId_ * 128U * 256U; // L0C ring 槽基址（每槽 [128,256] fp32 = 128KB）
        // S1 半块起点偏移（#4 修正）：L0C NZ 布局为列组(16列) N-major 排布，
        // 列组 ng 起点 = ng × align16(M) × 16 元素（组大小 = M行×16列；与 srcStride 同基准，
        // srcStride 单位 = 16 元素）。列 128 = 列组 8 → 偏移 = subBlockId × 8 × align16(M) × 16
        //   = subBlockId × S1_SUB_BLOCK_SIZE × align16(actKSize)。
        // 满块 actKSize=128 → 16384 元素（64KB 半槽边界）；尾块 actKSize=96 → 12288（[96,256] 精确半分）。
        // 布局模型由 srcStride 语义（DN/mxfp8 参考互证）+ 槽尺寸双重验算，待 S1=256 数值对拍终审
        uint32_t l0cHalfOff = subBlockId * S1_SUB_BLOCK_SIZE * ((actKSize + 15U) / 16U) * 16U;
        uint32_t ubRingSlot = subLoopIdx % UB_MM1RES_BUFCNT;
        uint32_t ubBase = ubRingSlot * UB_MM1RES_SLOT;

        Fixpipe<SOFTMAX_T, MM_OUT_T, CFG_ROW_MAJOR_UB>(mm1ResUB_[ubBase], mmL0CTensor_[l0cSlotOff + l0cHalfOff],
                                                       fixpipeParams);

        // subBlockId=1 做完最后一次 Fixpipe 后释放 L0C 并翻转槽
        if (subBlockId == 1U) {
            if (!fullTileUnitReuse_) {
                SetFlag<HardEvent::FIX_M>(L0C_EVENT0 + l0cBufId_);
            }
            l0cBufId_ = (l0cBufId_ + 1U) % 2U;
        }
    }

    __aicore__ inline void ComputeMm2(RunInfoMxfp8SoftmaxFp16& runInfo, uint32_t vCore)
    {
        // S1 尾行（actMSize ≤ 128）时 vCore=1 无有效列：跳过计算与 L0C 搬运（不赌 Mmad n=0
        // 的硬件行为）；V 拷贝（vCore=0 恒有效）与 V 槽归还（vCore=1 分支）保持无条件执行
        uint32_t actVecS1Size = (vCore == 0U) ?
                                    AttentionCommon::Min(runInfo.actMSize, S1_SUB_BLOCK_SIZE) :
                                    (runInfo.actMSize > S1_SUB_BLOCK_SIZE ? runInfo.actMSize - S1_SUB_BLOCK_SIZE : 0U);

        // ———— V 整行搬运（每 task 仅 vCore=0 一次，槽跨两个 vCore 的 C2 持有）————
        // 配对依赖：ExecuteTask Phase2 的 vCore 循环必须无条件 0→1 顺序执行
        // （vCore=1 的 LoadVToL0A 无需额外同步——MTE1 管道序保证在 copy 之后；
        //  释放的 Set<MTE1_MTE2> 挂 MTE1 管道，两个 vCore 的全部 LoadV 完成后才触发）
        if (vCore == 0U) {
            mm2VSlotBase_ = mm2L0abBufId_;
        } else {
            mm2L0abBufId_ = mm2VSlotBase_;
        }
        if (vCore == 0U && !runInfo.pairSecond) {
            WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT0 + kvBufId_);
            CopyVGmToL1(runInfo);
            CopyVScaleGmToL1(runInfo);
            SetFlag<HardEvent::MTE2_MTE1>(KV_EVENT0 + kvBufId_);
            WaitFlag<HardEvent::MTE2_MTE1>(KV_EVENT0 + kvBufId_);
            vDataBufId_ = kvBufId_;
            kvBufId_ = (kvBufId_ + 1U) % L1_KV_BUFCNT;
        }

        if (actVecS1Size > 0U) {
            if (!fullTileUnitReuse_) {
                WaitFlag<HardEvent::FIX_M>(L0C_EVENT0 + l0cBufId_);
            }

            // ———— V/P 分 subK 加载 + MatmulVP 累加 ————
            uint32_t subKLoop = AttentionCommon::CeilDiv(runInfo.actSingleLoopS2Size, MM2_SUB_K_SIZE);
            bool isFirstSubK = true;
            for (uint32_t subK = 0U; subK < subKLoop; ++subK) {
                uint32_t actSubKSize =
                    (subK + 1U == subKLoop) ? runInfo.actSingleLoopS2Size - subK * MM2_SUB_K_SIZE : MM2_SUB_K_SIZE;

                WaitFlag<HardEvent::M_MTE1>(MM2_L0AB_EVENT0 + mm2L0abBufId_);
                if (unlikely(vCore == 0U && runInfo.isLastS2Loop && actSubKSize < MM2_SUB_K_SIZE)) {
                    InitL0BufferForReduceSum();
                }
                LoadPToL0B(runInfo, vCore, subK, actSubKSize);
                if (vCore == 0U) {
                    LoadVToL0A(runInfo, subK, actSubKSize);
                }
                SetFlag<HardEvent::MTE1_M>(MM2_L0AB_EVENT0 + mm2L0abBufId_);
                WaitFlag<HardEvent::MTE1_M>(MM2_L0AB_EVENT0 + mm2L0abBufId_);
                MatmulVP(runInfo, vCore, actSubKSize, isFirstSubK);
                SetFlag<HardEvent::M_MTE1>(MM2_L0AB_EVENT0 + mm2L0abBufId_);
                mm2L0abBufId_ = 0U;
                isFirstSubK = false;
            }

            if (!(runInfo.actMSize == 256U && runInfo.actSingleLoopS2Size == 256U)) {
                SetFlag<HardEvent::M_FIX>(L0C_EVENT0 + l0cBufId_);
            } // 与 FixpipeMm2 的 Wait<M_FIX> 配对（同守卫内）
        }

        // ———— V 槽释放移至 vCore=1 末尾（两个 vCore 的 LoadV 都完成后；无条件执行，
        //     即使 vCore=1 闲置也必须归还 V 单拷共享的槽，否则 ring 死锁）————
        if (vCore == 1U && (runInfo.pairSecond || runInfo.isLastS2Loop)) {
            SetFlag<HardEvent::MTE1_MTE2>(KV_EVENT0 + vDataBufId_);
        }
    }

    __aicore__ inline void FixpipeMm2(RunInfoMxfp8SoftmaxFp16& runInfo, uint32_t vCore)
    {
        // 与 ComputeMm2 同守卫：S1 尾行时 vCore=1 跳过（其 Wait<M_FIX>/Set<FIX_M> 与
        // ComputeMm2 的 Set<M_FIX>/Wait<FIX_M> 在守卫内成对跳过，flag 保持配平；
        // l0cBufId_ 翻转也随之跳过，与活跃 vCore 的翻转节奏一致）
        uint32_t actVecS1Size = (vCore == 0U) ?
                                    AttentionCommon::Min(runInfo.actMSize, S1_SUB_BLOCK_SIZE) :
                                    (runInfo.actMSize > S1_SUB_BLOCK_SIZE ? runInfo.actMSize - S1_SUB_BLOCK_SIZE : 0U);
        if (actVecS1Size == 0U) {
            return;
        }

        const bool unitFlag = runInfo.actMSize == 256U && runInfo.actSingleLoopS2Size == 256U;
        if (!unitFlag) {
            WaitFlag<HardEvent::M_FIX>(L0C_EVENT0 + l0cBufId_);
        }

        uint32_t l0cOffset = l0cBufId_ * 128U * 256U;

        FixpipeParamsArch3510<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = (actVecS1Size + 7U) >> 3 << 3;
        fixpipeParams.mSize = dBaseSize + 1U;
        fixpipeParams.srcStride = ((dBaseSize + ROWSUM_ROWS + 15U) / 16U) * 16U;
        fixpipeParams.dstStride = S1_SUB_BLOCK_SIZE; // 固定 pitch 128，nSize 恒 ≤ pitch，杜绝行覆盖
        fixpipeParams.quantPre = QuantMode_t::NoQuant;
        fixpipeParams.unitFlag = unitFlag ? 3 : 0;
        fixpipeParams.dualDstCtl = 0; // 单目标模式（CANN PTO 权威定义：0=Single/1=SplitM/2=SplitN）
        fixpipeParams.subBlockId = static_cast<bool>(vCore); // 整块投递给对应 V 核
        fixpipeParams.params.ndNum = 1U;
        fixpipeParams.params.srcNdStride = fixpipeParams.mSize;
        fixpipeParams.params.dstNdStride = fixpipeParams.dstStride;

        Fixpipe<MM_OUT_T, MM_OUT_T, CFG_ROW_MAJOR_UB>(mm2ResUB_, mmL0CTensor_[l0cOffset], fixpipeParams);

        if (!fullTileUnitReuse_) {
            SetFlag<HardEvent::FIX_M>(L0C_EVENT0 + l0cBufId_);
        }
        l0cBufId_ = (l0cBufId_ + 1U) % 2U;
    }

    __aicore__ inline void CopyQGmToL1(const RunInfoMxfp8SoftmaxFp16& runInfo)
    {
        uint32_t l1BaseOffset = qBufId_ * L1_Q_SINGLE_SLOT_SIZE;
        uint32_t dstStride = (runInfo.actMSize + 31U) / 32U * 32U;
        FaL1Tensor<QUANT_T, L1Format::NZ> l1Tensor{.tensor = qL1Tensor_[l1BaseOffset], .rowCount = dstStride};

        // n2Idx 传 realN2Idx（折叠头索引，GQA 时 = n2*G+g；GM 布局按 realN2×1 折叠初始化）
        GmCoordGs1Merge gmCoord{.bIdx = runInfo.bIdx,
                                .n2Idx = runInfo.realN2Idx,
                                .gS1Idx = runInfo.gS1Idx,
                                .dIdx = 0U,
                                .gS1DealSize = runInfo.actMSize,
                                .dDealSize = dBaseSize};
        copyQueryGmToL1_(l1Tensor, queryGm_, gmCoord);
    }

    __aicore__ inline void CopyQScaleGmToL1(const RunInfoMxfp8SoftmaxFp16& runInfo)
    {
        uint32_t l1BaseOffset = qBufId_ * L1_QSCALE_SINGLE_SLOT_SIZE;
        uint32_t dstStride = (runInfo.actMSize + 31U) / 32U * 32U;
        uint32_t dDealSize = dBaseSize / MXFP_GROUP_SIZE;
        FaL1Tensor<SCALE_T, L1Format::NZ> l1Tensor{.tensor = qScaleL1Tensor_[l1BaseOffset], .rowCount = dstStride};

        // 同 CopyQGmToL1：n2Idx 传 realN2Idx（折叠头索引）
        GmCoordGs1Merge gmCoord{.bIdx = runInfo.bIdx,
                                .n2Idx = runInfo.realN2Idx,
                                .gS1Idx = runInfo.gS1Idx,
                                .dIdx = 0U,
                                .gS1DealSize = runInfo.actMSize,
                                .dDealSize = dDealSize};
        copyQueryScaleGmToL1_(l1Tensor, queryScaleGm_, gmCoord);
    }

    // K 整块拷贝（每 task 一次）：[actSingleLoopS2Size, dBaseSize] → kvL1 槽；
    // rowCount 与 LoadKToL0A 的 srcStride 同基准（actSingleLoopS2SizeAlign，DN 同款），
    // subLoop 切片由 LoadKToL0A 的 mStartPosition 在槽内寻址
    __aicore__ inline void CopyKGmToL1(const RunInfoMxfp8SoftmaxFp16& runInfo)
    {
        // BNSD D=128 path: keep invariant geometry out of generic helpers.
        Nd2NzParams params;
        params.ndNum = 1U;
        params.nValue = runInfo.pairCopyS2Size;
        params.dValue = dBaseSize;
        params.srcDValue = dBaseSize;
        params.dstNzC0Stride = 512U;
        params.dstNzNStride = 1U;
        params.srcNdMatrixStride = 0U;
        params.dstNzMatrixStride = 0U;
        uint64_t offset =
            ((static_cast<uint64_t>(runInfo.bIdx) * constInfo_.n2Size + runInfo.n2Idx) * constInfo_.s2Size +
             runInfo.s2Idx) *
            dBaseSize;
        DataCopy(kvL1Tensor_[kvBufId_ * L1_KV_SINGLE_SLOT_SIZE], keyGm_.gmTensor[offset], params);
    }

    __aicore__ inline void CopyKScaleGmToL1(const RunInfoMxfp8SoftmaxFp16& runInfo)
    {
        Dn2NzParams params;
        params.dnNum = 1U;
        params.nValue = 2U;
        params.dValue = runInfo.pairCopyS2Size;
        params.srcDnMatrixStride = 0U;
        params.srcDValue = 2U;
        params.dstNzC0Stride = 2U;
        params.dstNzNStride = 1U;
        params.dstNzMatrixStride = 2U;
        uint64_t offset =
            ((static_cast<uint64_t>(runInfo.bIdx) * constInfo_.n2Size + runInfo.n2Idx) * constInfo_.s2Size +
             runInfo.s2Idx) *
            4U;
        GlobalTensor<bfloat16_t> source;
        source.SetGlobalBuffer((__gm__ bfloat16_t*)keyScaleGm_.gmTensor[offset].GetPhyAddr());
        DataCopy(kvScaleL1Tensor_[kvBufId_ * L1_KVSCALE_SINGLE_SLOT_SIZE].template ReinterpretCast<bfloat16_t>(),
                 source, params);
    }

    __aicore__ inline void CopyVGmToL1(const RunInfoMxfp8SoftmaxFp16& runInfo)
    {
        // ND2NZ pads only to 16 rows, but PV reads a multiple of 64.
        // Clear the tail slot before copying so padded V cannot contain NaNs.
        if ((runInfo.pairCopyS2Size % 64U) != 0U) {
            InitConstValueParams<uint16_t> zeros(1U, L1_KV_SINGLE_SLOT_SIZE / 32U, 0U, 0U);
            Fill(kvL1Tensor_[kvBufId_ * L1_KV_SINGLE_SLOT_SIZE].template ReinterpretCast<uint16_t>(), zeros);
            PipeBarrier<PIPE_MTE2>();
        }
        Nd2NzParams params;
        params.ndNum = 1U;
        params.nValue = runInfo.pairCopyS2Size;
        params.dValue = dBaseSize;
        params.srcDValue = dBaseSize;
        params.dstNzC0Stride = 512U;
        params.dstNzNStride = 1U;
        params.srcNdMatrixStride = 0U;
        params.dstNzMatrixStride = 0U;
        uint64_t offset =
            ((static_cast<uint64_t>(runInfo.bIdx) * constInfo_.n2Size + runInfo.n2Idx) * constInfo_.s2Size +
             runInfo.s2Idx) *
            dBaseSize;
        DataCopy(kvL1Tensor_[kvBufId_ * L1_KV_SINGLE_SLOT_SIZE], valueGm_.gmTensor[offset], params);
    }

    __aicore__ inline void CopyVScaleGmToL1(const RunInfoMxfp8SoftmaxFp16& runInfo)
    {
        uint32_t groups = (runInfo.pairCopyS2Size + 63U) / 64U;
        Nd2NzParams params;
        params.ndNum = 1U;
        params.nValue = groups;
        params.dValue = dBaseSize;
        params.srcDValue = dBaseSize;
        params.dstNzC0Stride = 8U;
        params.dstNzNStride = 1U;
        params.srcNdMatrixStride = 0U;
        params.dstNzMatrixStride = groups;
        uint64_t offset = ((static_cast<uint64_t>(runInfo.bIdx) * constInfo_.n2Size + runInfo.n2Idx) *
                               ((constInfo_.s2Size + 63U) / 64U) +
                           runInfo.s2Idx / 64U) *
                          (2U * dBaseSize);
        GlobalTensor<bfloat16_t> source;
        source.SetGlobalBuffer((__gm__ bfloat16_t*)valueScaleGm_.gmTensor[offset].GetPhyAddr());
        DataCopy(kvScaleL1Tensor_[kvBufId_ * L1_KVSCALE_SINGLE_SLOT_SIZE].template ReinterpretCast<bfloat16_t>(),
                 source, params);
    }

    __aicore__ inline void LoadQToL0B(const RunInfoMxfp8SoftmaxFp16& runInfo)
    {
        LoadData2DParamsV2 dataParams;
        dataParams.mStartPosition = 0U;
        dataParams.kStartPosition = 0U;
        dataParams.mStep = ((runInfo.actMSize + 31U) / 32U * 32U) / 16U;
        dataParams.kStep = dBaseSize / 32U;
        dataParams.srcStride = dataParams.mStep;
        dataParams.dstStride = dataParams.mStep;
        dataParams.ifTranspose = false;

        LoadData2DMxParams scaleParams;
        scaleParams.xStartPosition = 0U;
        scaleParams.yStartPosition = 0U;
        scaleParams.xStep = dataParams.mStep;
        scaleParams.yStep = 2U;
        scaleParams.srcStride = scaleParams.yStep;
        scaleParams.dstStride = scaleParams.yStep;

        uint32_t qL1Offset = qBufId_ * L1_Q_SINGLE_SLOT_SIZE;
        uint32_t qScaleL1Offset = qBufId_ * L1_QSCALE_SINGLE_SLOT_SIZE;
        LoadData(qL0BTensor_, qL1Tensor_[qL1Offset], qScaleL1Tensor_[qScaleL1Offset], dataParams, scaleParams);
    }

    __aicore__ inline void LoadKToL0A(const RunInfoMxfp8SoftmaxFp16& runInfo, uint32_t subLoopIdx)
    {
        uint32_t actKSize = (subLoopIdx + 1 == AttentionCommon::CeilDiv(runInfo.actSingleLoopS2Size, S2_SPLIT_SIZE)) ?
                                runInfo.actSingleLoopS2Size - subLoopIdx * S2_SPLIT_SIZE :
                                S2_SPLIT_SIZE;
        uint32_t actKSizeAlign = (actKSize + 15U) / 16U * 16U;

        LoadData2DParamsV2 dataParams;
        dataParams.mStartPosition = (runInfo.pairSecond ? 16U : 0U) + subLoopIdx * (S2_SPLIT_SIZE / 16U);
        dataParams.kStartPosition = 0U;
        dataParams.mStep = actKSizeAlign / 16U;
        dataParams.kStep = dBaseSize / 32U;
        dataParams.srcStride = 32U; // 恒 32（L1 KV 对槽 512 行 / 16）
        dataParams.dstStride = actKSizeAlign / 16U;
        dataParams.ifTranspose = false;

        LoadData2DMxParams scaleParams;
        scaleParams.xStartPosition = (runInfo.pairSecond ? 16U : 0U) + subLoopIdx * (S2_SPLIT_SIZE / 16U);
        scaleParams.yStartPosition = 0U;
        scaleParams.xStep = actKSizeAlign / 16U;
        scaleParams.yStep = 2U;
        scaleParams.srcStride = scaleParams.yStep;
        scaleParams.dstStride = scaleParams.yStep;

        // 槽内整块寻址：srcStride = actSingleLoopS2SizeAlign/16（与 CopyKGmToL1 的 rowCount 同基准），
        // subLoop 切片靠 mStartPosition —— 整块拷贝后本参数组自洽（修复原 subLoop 紧凑拷贝与此的不一致）
        uint32_t kvL1Offset = kDataBufId_ * L1_KV_SINGLE_SLOT_SIZE;
        uint32_t kvScaleL1Offset = kDataBufId_ * L1_KVSCALE_SINGLE_SLOT_SIZE;
        LoadData(mm1L0ATensor_[kL0aBufId_ * 128U * 128U], kvL1Tensor_[kvL1Offset], kvScaleL1Tensor_[kvScaleL1Offset],
                 dataParams, scaleParams);
    }

    __aicore__ inline void MatmulKQ(const RunInfoMxfp8SoftmaxFp16& runInfo, uint32_t actKSize)
    {
        SetMMRowMajor();
        MmadParams mmadParams;
        mmadParams.unitFlag = (runInfo.actMSize == 256U && actKSize == 128U) ? 3 : 0;
        mmadParams.m = actKSize;
        mmadParams.n = runInfo.actMSizeAlign32;
        mmadParams.k = dBaseSize;
        mmadParams.cmatrixInitVal = true;
        mmadParams.cmatrixSource = false;
        mmadParams.disableGemv = true;

        uint32_t l0aOffset = kL0aBufId_ * 128U * 128U;
        uint32_t l0cOffset = l0cBufId_ * 128U * 256U;
        Mmad(mmL0CTensor_[l0cOffset], mm1L0ATensor_[l0aOffset], qL0BTensor_, mmadParams);
    }

    __aicore__ inline void LoadPToL0B(const RunInfoMxfp8SoftmaxFp16& runInfo, uint32_t vCore, uint32_t subKIdx,
                                      uint32_t actSubKSize)
    {
        // m/k 方向对齐 DN LoadPToL0（量产）：m 沿 S2（subK 切片）、k 沿 S1、ifTranspose=true。
        // subLoop 粒度槽（L1 v3）：每 subK 一槽 [128,128]，mStartPosition 恒 0、srcStride 恒 8
        // （槽内行数固定 128），subK → 槽选择（ring = loop×2+subK，核间交错 L1_PSLOT）
        uint32_t actSubKSizeAlign64 = (actSubKSize + 63U) / 64U * 64U; // S2 方向 64 对齐（DN 同款）

        uint32_t actVecS1Size = (vCore == 0U) ?
                                    AttentionCommon::Min(runInfo.actMSize, S1_SUB_BLOCK_SIZE) :
                                    (runInfo.actMSize > S1_SUB_BLOCK_SIZE ? runInfo.actMSize - S1_SUB_BLOCK_SIZE : 0U);
        uint32_t actVecS1SizeAlign32 = (actVecS1Size + 31U) / 32U * 32U; // 与 MatmulVP 的 n 同源

        uint32_t pRing = L1_PRING(runInfo.loop, subKIdx);
        uint32_t pL1Offset = L1_PSLOT(vCore, pRing);
        uint32_t pScaleL1Offset = (pRing / 2U) * L1_PSCALE_SINGLE_SLOT_SIZE; // pscale 同 ring（本 subK 专属槽）

        LoadData2DParamsV2 dataParams;
        dataParams.mStartPosition = 0U; // 恒 0（subK → 槽选择）
        dataParams.kStartPosition = 0U;
        dataParams.mStep = actSubKSizeAlign64 / 16U;      // m 沿 S2：本 subK 有效行
        dataParams.kStep = actVecS1SizeAlign32 / 32U;     // k 沿 S1：本核呈现宽（128→4，64→2）
        dataParams.srcStride = 256U / 16U;                // 恒 16（L1 P 槽整程 S2=256 行）
        dataParams.dstStride = actVecS1SizeAlign32 / 16U; // 与 MatmulVP n 匹配（128→8，64→4）
        dataParams.ifTranspose = true;                    // L1 NZ [S2,S1] → L0B [K=S2, N=S1]

        LoadData2DMxParams scaleParams;
        scaleParams.xStartPosition = 0U;                      // x 沿 S1（与 data k 同向）
        scaleParams.yStartPosition = 0U;                      // 恒 0（槽即本 subK 专属）
        scaleParams.xStep = actVecS1SizeAlign32 / 16U;        // 本核呈现宽（128→8，64→4）
        scaleParams.yStep = (actSubKSizeAlign64 + 63U) / 64U; // 本 subK 的 y 单元数（128→2，64→1）
        scaleParams.srcStride = 5U;                           // x 行距 = 2 y + 1 bank pad（DN 款）
        scaleParams.dstStride = scaleParams.yStep;

        uint32_t l0bOffset = mm2L0abBufId_ * L0B_P_SLOT_SIZE;
        LocalTensor<SCALE_T>& pScaleL1Tensor = (vCore == 0U) ? pScaleL1V0Tensor_ : pScaleL1V1Tensor_;
        LoadData(pL0BTensor_[l0bOffset], l1SharedTensor_[pL1Offset], pScaleL1Tensor[pScaleL1Offset], dataParams,
                 scaleParams);
    }

    __aicore__ inline void LoadVToL0A(const RunInfoMxfp8SoftmaxFp16& runInfo, uint32_t subKIdx, uint32_t actSubKSize)
    {
        uint32_t actSubKSizeAlign = (actSubKSize + 63U) / 64U * 64U;

        LoadData2DParamsV2 dataParams;
        dataParams.mStartPosition = (runInfo.pairSecond ? 16U : 0U) + subKIdx * (MM2_SUB_K_SIZE / 16U);
        dataParams.kStartPosition = 0U;
        dataParams.mStep = actSubKSizeAlign / 16U;
        dataParams.kStep = dBaseSize / 32U;
        dataParams.srcStride = 32U; // 恒 32（L1 KV 对槽 512 行 / 16）
        dataParams.dstStride = (dBaseSize + 15U) / 16U + 1U;
        dataParams.ifTranspose = true;

        LoadData2DMxParams scaleParams;
        scaleParams.xStartPosition = 0U;
        scaleParams.yStartPosition = (runInfo.pairSecond ? 4U : 0U) + subKIdx * (MM2_SUB_K_SIZE / 64U);
        scaleParams.xStep = (dBaseSize + 15U) / 16U;
        scaleParams.yStep = (actSubKSizeAlign + 63U) / 64U;
        scaleParams.srcStride = 8U; // 恒 8（对槽 8 个 64 行 scale 组）
        scaleParams.dstStride = scaleParams.yStep;

        uint32_t kvL1Offset = vDataBufId_ * L1_KV_SINGLE_SLOT_SIZE;
        uint32_t kvScaleL1Offset = vDataBufId_ * L1_KVSCALE_SINGLE_SLOT_SIZE;
        // dst mx 直传（InitL0BufferForReduceSum 同款类型契约：plain-plain 触发静态断言）
        LoadData(mm2L0ATensor_[mm2L0abBufId_ * L0A_MM2_SLOT_SIZE], kvL1Tensor_[kvL1Offset],
                 kvScaleL1Tensor_[kvScaleL1Offset], dataParams, scaleParams);
    }

    __aicore__ inline void MatmulVP(const RunInfoMxfp8SoftmaxFp16& runInfo, uint32_t vCore, uint32_t subKSize,
                                    bool isFirstSubK)
    {
        const bool unitFlag = runInfo.actMSize == 256U && runInfo.actSingleLoopS2Size == 256U;
        SetMMRowMajor();
        MmadParams mmadParams;
        mmadParams.unitFlag = unitFlag ? 3 : 0;
        // L0A/L0C retain a physical row pitch of 144 for the output and rowsum fragment.
        // Full MM2 computes 128 output rows plus one rowsum row, matching the UnitFlag read extent.
        mmadParams.m = unitFlag ? dVBaseSize + 1U : dVBaseSize + ROWSUM_ROWS;
        // 本 V 核的 S1 半（与 LoadPToL0B 固定满宽 kStep=4 的加载范围一致，杜绝读 L0B 越界；
        // L0B 垃圾列 ≥ actVecS1Size 不参与计算）
        uint32_t actVecS1Size = (vCore == 0U) ?
                                    AttentionCommon::Min(runInfo.actMSize, S1_SUB_BLOCK_SIZE) :
                                    (runInfo.actMSize > S1_SUB_BLOCK_SIZE ? runInfo.actMSize - S1_SUB_BLOCK_SIZE : 0U);
        uint32_t actVecS1SizeAlign32 = (actVecS1Size + 31U) / 32U * 32U;
        mmadParams.n = actVecS1SizeAlign32;
        mmadParams.k = (subKSize + 63U) / 64U * 64U;
        mmadParams.cmatrixInitVal = isFirstSubK;
        mmadParams.cmatrixSource = false;
        mmadParams.disableGemv = true;

        uint32_t l0cOffset = l0cBufId_ * 128U * 256U;
        uint32_t l0bOffset = mm2L0abBufId_ * L0B_P_SLOT_SIZE;
        Mmad(mmL0CTensor_[l0cOffset], mm2L0ATensor_[mm2L0abBufId_ * L0A_MM2_SLOT_SIZE], pL0BTensor_[l0bOffset],
             mmadParams);
    }
};
} // namespace QFA_KERNEL

#endif // QUANT_FLASH_ATTN_BLOCK_CUBE_MXFP8_SOFTMAX_FP16_H_
