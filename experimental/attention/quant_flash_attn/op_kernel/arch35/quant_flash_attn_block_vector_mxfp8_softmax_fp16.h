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
 * \file quant_flash_attn_block_vector_mxfp8_softmax_fp16.h
 * \brief MxFP8 Softmax FP16 Vector Block：V1（softmax→fp8）+ V2（累积→除→转置→GM 输出）
 */

#ifndef QUANT_FLASH_ATTN_BLOCK_VECTOR_MXFP8_SOFTMAX_FP16_H_
#define QUANT_FLASH_ATTN_BLOCK_VECTOR_MXFP8_SOFTMAX_FP16_H_

#include "vf/vf_common_def_mxfp8_softmax_fp16.h"
#include "vf/vf_calc_rescale_factor.h"
#include "vf/vf_calc_pscale.h"
#include "vf/vf_flashupdate_acc.h"
#include "vf/vf_div_cast_bf16.h"
#include "vf/vf_softmax_fp16_to_fp8.h"
#include "vf/vf_calc_lse.h"
#include "../../common/op_kernel/memcopy/attn/copy_ub_to_gm.h"
#include "../../common/op_kernel/memcopy/attn/fa_gm_tensor.h"
#include "../../common/op_kernel/memcopy/attn/parser.h"

namespace QFA_KERNEL {

template <typename QUANT_T, typename SCALE_T, typename OUT_T>
class QuantFlashAttnBlockVectorMxfp8SoftmaxFp16 {
public:
    // ===== 类型别名 =====
    using MM_OUT_T = float; // mm2Res fp32
    using SOFTMAX_T = half; // 状态 half（log2 整数域）

    static constexpr uint32_t s1BaseSize = S1_BASE_SIZE; // 256（S1 总宽，V0/V1 各持一半）
    static constexpr uint32_t s2SubLoopSize =
        S2_SUB_LOOP_SIZE; // 128（每 subLoop 的 mm1Res/P 行数，即 S2 方向每次处理的行数）
    static constexpr uint32_t dVBaseSize = DV_BASE_SIZE;                 // 128（V2 输出的 DV 方向行数）
    static constexpr half MIN_HALF_VALUE = static_cast<half>(-65504.0f); // fp16 最小有限负值

private:
    // ===== UB 尺寸常量（vector 侧自有；mm1Res/mm2Res 直接用 common_def 跨核契约）=====
    static constexpr uint32_t UB_VEC1P_SIZE =
        2 * QfaVectorApi::QFA_UB_P_SLOT; // 2 padded slots: 4 * (4096 + 32) bytes each
    static constexpr uint32_t UB_STAGE2OUT_SIZE = dVBaseSize * (s1BaseSize / 2); // [128,128] fp32
    static constexpr uint32_t UB_ACCROWSUM_SIZE = s1BaseSize / 2;                // [128] fp32
    static constexpr uint32_t UB_SMAX_SLOT = s1BaseSize / 2;                     // [128] half
    static constexpr uint32_t UB_SMAX_BUFCNT = 3;                                // mloop%3
    static constexpr uint32_t UB_SEXP_BUFCNT = 3;                                // loop%3
    static constexpr uint32_t UB_PSCALE_SUBLOOPS = 2;                            // subLoopUsedMaxUB 槽数
    // pscale 网格影像（共享常量同源 vf_common_def）：[ring][768B]，ring 周期 6
    static constexpr uint32_t UB_PSCALE_SLOT = QfaVectorApi::QFA_UB_PSCALE_SLOT;   // 768B/槽
    static constexpr uint32_t UB_PSCALE_RINGS = QfaVectorApi::QFA_UB_PSCALE_RINGS; // 6

    // ===== UB BASE 常量链（mm1Res/mm2Res 的 BASE 用 common_def 值，不自引用）=====
    static constexpr uint32_t UB_VEC1P_BASE = UB_MM2RES_BASE + UB_MM2RES_SIZE * sizeof(MM_OUT_T);
    static constexpr uint32_t UB_STAGE2OUT_BASE = UB_VEC1P_BASE + UB_VEC1P_SIZE * sizeof(QUANT_T);
    static constexpr uint32_t UB_ACCROWSUM_BASE = UB_STAGE2OUT_BASE + UB_STAGE2OUT_SIZE * sizeof(MM_OUT_T);
    static constexpr uint32_t UB_SMAX_BASE = UB_ACCROWSUM_BASE + UB_ACCROWSUM_SIZE * sizeof(MM_OUT_T);
    static constexpr uint32_t UB_SEXP_BASE = UB_SMAX_BASE + UB_SMAX_SLOT * UB_SMAX_BUFCNT * sizeof(SOFTMAX_T);
    static constexpr uint32_t UB_BSMAX_BASE = UB_SEXP_BASE + UB_SMAX_SLOT * UB_SEXP_BUFCNT * sizeof(MM_OUT_T);
    static constexpr uint32_t UB_SUBMAX_BASE = UB_BSMAX_BASE + UB_SMAX_SLOT * sizeof(SOFTMAX_T);
    static constexpr uint32_t UB_PSCALE_BASE = UB_SUBMAX_BASE + UB_SMAX_SLOT * UB_PSCALE_SUBLOOPS * sizeof(SOFTMAX_T);
    static constexpr uint32_t UB_NZIDX_BASE = UB_PSCALE_BASE + UB_PSCALE_SLOT * UB_PSCALE_RINGS;
    static constexpr uint32_t UB_ZERO_OUT_SIZE = 64 * dVBaseSize; // OUT_T 元素
    static constexpr uint32_t UB_ZERO_OUT_BASE = UB_NZIDX_BASE + 256;
    // LSE 暂存槽：[128] fp32，VfCalcLse 输出 → GM
    static constexpr uint32_t UB_LSE_SIZE = dVBaseSize; // fp32 元素
    static constexpr uint32_t UB_LSE_BASE = UB_ZERO_OUT_BASE + UB_ZERO_OUT_SIZE * sizeof(OUT_T);
    static constexpr uint32_t UB_LSE_INF_BASE = UB_LSE_BASE + UB_LSE_SIZE * sizeof(float);
    // pNzIdxUB 索引表 256B 计入总量
    static constexpr uint32_t UB_TOTAL = UB_LSE_INF_BASE + UB_LSE_SIZE * sizeof(float);
    static_assert(UB_TOTAL <= 256 * 1024, "V-core UB budget (256KB) exceeded");

    // ===== UB Tensor 成员 =====
    LocalTensor<SOFTMAX_T> mm1ResUB_;     // [2][128,128] half，C1 Fixpipe 目标（loop%2）
    LocalTensor<MM_OUT_T> mm2ResUB_;      // [129,128] fp32，C2 Fixpipe 目标（V2 输入 + divOut/outputT 别名）
    LocalTensor<QUANT_T> vec1PUb_;        // [2][128,128] fp8，V1 P 的 NZ 源布局暂存
    LocalTensor<MM_OUT_T> stage2OutUB_;   // [128,128] fp32，V2 跨块累积分子 [DV,S1]
    LocalTensor<MM_OUT_T> accRowsumUB_;   // [128] fp32，V2 跨块累积分母 per S1
    LocalTensor<SOFTMAX_T> softmaxMaxUB_; // [3][128] half，accMax log2 整数 k（mloop%3）
    LocalTensor<MM_OUT_T> softmaxExpUB_;  // [3][128] fp32，跨块因子 fp32 位模式（loop%3）
    LocalTensor<SOFTMAX_T> blockStartMaxUB_;  // [128] half，块首 K^{blkStart} 快照
    LocalTensor<SOFTMAX_T> subLoopUsedMaxUB_; // [2][128] half，各 subLoop k_i（供末 subLoop 算 pscale）
    LocalTensor<SCALE_T> pscaleGridUB_; // [6][768B] pscale L1 网格影像（[ring]，VF 散布 + 连续整槽发 L1）
    LocalTensor<uint8_t> pNzIdxUB_; // P NZ 打包 Gather 索引表（pscale 改字节对后唯一消费者是 P 存储）
    LocalTensor<OUT_T> zeroOutUB_; // 全量预清零零源 [64,128]（ClearOutput 分片 memset GM）
    LocalTensor<float> lseUb_;     // softmaxLse 暂存 [128] fp32（VfCalcLse → GM）
    LocalTensor<float> lseInfUb_;  // LSE -inf 源 [128] fp32（DEAL_ZERO kv=0 批行）

    // ===== L1 共享区视图成员（AIC L1，V1 经 MTE3 写入 P/pscale 的目标地址）=====
    // P 槽核间交错布局（common_def L1_PSLOT 宏）下双核槽不连续——统一走宽视图 + 计算基址
    LocalTensor<QUANT_T> l1SharedTensor_;   // L1 共享区整域视图（P 数据，交错槽）
    LocalTensor<SCALE_T> pScaleL1V0Tensor_; // V0 pscale task grids (3 x 1280B).
    LocalTensor<SCALE_T> pScaleL1V1Tensor_; // V1 pscale task grids (3 x 1280B).

    // ===== 依赖注入 =====
    const ConstInfo& constInfo_; // 核信息（subBlockIdx、scaleValue 等）

    // ===== GM 输出（BNSD 布局 + S1_ONLY 行拷贝——realGSize=1 语义下 GS1 退化为 S1）=====
    GlobalTensor<OUT_T> attentionOutGm_;                      // GM 输出 tensor
    GlobalTensor<float> softmaxLseGm_;                        // softmaxLse GM 输出（[B,N,S1] fp32）
    FaGmTensor<OUT_T, GmFormat::BNGSD, int32_t> outGmTensor_; // + 偏移计算器
    CopyAttenOutUbToGm<OUT_T, GmFormat::BNGSD, UbFormat::S1_ONLY> attenOutUbToGm_; // UB→GM 拷贝仿函数

    // ===== 外部 pScale（静态 headroom 因子，InitTensorsVec 时 cast half 存成员）=====
    half pScaleHalf_ = static_cast<half>(1.0f);

    // ===== 核内事件 ID =====
    static constexpr uint32_t EVT_P_COPY = 0;   // V↔MTE3：P/pscale MTE3 前后 + 顶部等待
    static constexpr uint32_t EVT_GM_OUT = 1;   // V→MTE3：GM 输出前
    static constexpr uint32_t EVT_ZERO_OUT = 2; // V↔MTE3：ClearOutput 零源就绪 + 清零排空
    static constexpr uint32_t EVT_LSE_OUT = 3;  // V→MTE3：LSE 计算就绪后放行 GM 拷贝

public:
    __aicore__ inline explicit QuantFlashAttnBlockVectorMxfp8SoftmaxFp16(ConstInfo& constInfo)
        : constInfo_(constInfo)
    {}

    /*
     * GM 输出初始化（kernel InitInput 阶段调用）
     * 注意：parser 必须已 Init 完成后再调用（OffsetCalculator 内部为拷贝语义，早调会拷到空 parser）
     */
    __aicore__ inline void InitInput(__gm__ uint8_t* attentionOut, __gm__ uint8_t* softmaxLse, __gm__ uint8_t* pScale,
                                     const ActualSeqLensParser<ActualSeqLensMode::BY_BATCH, int32_t>& parser)
    {
        attentionOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ OUT_T*>(attentionOut));
        if (constInfo_.isSoftmaxLseEnable) {
            softmaxLseGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(softmaxLse));
        }
        outGmTensor_.gmTensor = attentionOutGm_;
        outGmTensor_.offsetCalculator.Init(constInfo_.bSize, constInfo_.realN2Size, constInfo_.realGSize,
                                           constInfo_.s1Size, constInfo_.dSizeV, parser);
        // 外部 pScale（headroom 因子，外部动态输入）：GM fp32 标量 → half 存成员
        if (pScale != nullptr) {
            pScaleHalf_ = static_cast<half>(*reinterpret_cast<__gm__ float*>(pScale));
        }
    }

    __aicore__ inline void InitTensorsVec()
    {
        // ① UB 静态分配（编译期 BASE 常量链）
        //    mm1Res/mm2Res 用 common_def 跨核契约；其余按 BASE 链推导
        //    addr = 字节地址，size = 元素数
        mm1ResUB_ = LocalTensor<SOFTMAX_T>(TPosition::VECCALC, UB_MM1RES_BASE, UB_MM1RES_SLOT * UB_MM1RES_BUFCNT);
        mm2ResUB_ = LocalTensor<MM_OUT_T>(TPosition::VECCALC, UB_MM2RES_BASE, UB_MM2RES_SIZE);
        vec1PUb_ = LocalTensor<QUANT_T>(TPosition::VECCALC, UB_VEC1P_BASE, UB_VEC1P_SIZE);
        stage2OutUB_ = LocalTensor<MM_OUT_T>(TPosition::VECCALC, UB_STAGE2OUT_BASE, UB_STAGE2OUT_SIZE);
        accRowsumUB_ = LocalTensor<MM_OUT_T>(TPosition::VECCALC, UB_ACCROWSUM_BASE, UB_ACCROWSUM_SIZE);
        softmaxMaxUB_ = LocalTensor<SOFTMAX_T>(TPosition::VECCALC, UB_SMAX_BASE, UB_SMAX_SLOT * UB_SMAX_BUFCNT);
        softmaxExpUB_ = LocalTensor<MM_OUT_T>(TPosition::VECCALC, UB_SEXP_BASE, UB_SMAX_SLOT * UB_SEXP_BUFCNT);
        blockStartMaxUB_ = LocalTensor<SOFTMAX_T>(TPosition::VECCALC, UB_BSMAX_BASE, UB_SMAX_SLOT);
        subLoopUsedMaxUB_ =
            LocalTensor<SOFTMAX_T>(TPosition::VECCALC, UB_SUBMAX_BASE, UB_SMAX_SLOT * UB_PSCALE_SUBLOOPS);
        pscaleGridUB_ = LocalTensor<SCALE_T>(TPosition::VECCALC, UB_PSCALE_BASE, UB_PSCALE_SLOT * UB_PSCALE_RINGS);
        pNzIdxUB_ = LocalTensor<uint8_t>(TPosition::VECCALC, UB_NZIDX_BASE,
                                         256); // 索引表 256B（实现期可调）
        zeroOutUB_ = LocalTensor<OUT_T>(TPosition::VECCALC, UB_ZERO_OUT_BASE, UB_ZERO_OUT_SIZE);
        lseUb_ = LocalTensor<float>(TPosition::VECCALC, UB_LSE_BASE, UB_LSE_SIZE);
        lseInfUb_ = LocalTensor<float>(TPosition::VECCALC, UB_LSE_INF_BASE, UB_LSE_SIZE);

        // ② L1 共享区视图（按 common_def 常量构造，不分配、不初始化）。
        //    P 槽核间交错布局下双核槽不连续——整域宽视图 + L1_PSLOT(core, ring) 计算基址
        l1SharedTensor_ = LocalTensor<QUANT_T>(TPosition::A1, 0, L1_SHARED_REGION_SIZE);
        pScaleL1V0Tensor_ =
            LocalTensor<SCALE_T>(TPosition::A1, L1_PSCALE_V0_BASE, L1_P_BUFCNT * L1_PSCALE_SINGLE_SLOT_SIZE);
        pScaleL1V1Tensor_ =
            LocalTensor<SCALE_T>(TPosition::A1, L1_PSCALE_V1_BASE, L1_P_BUFCNT * L1_PSCALE_SINGLE_SLOT_SIZE);

        //    公式 idx[i*128+j] = i + 2j（i∈{0,1}，j∈[0,128)）——一张表两个消费者：
        //    P 存储：Cast+Or 交织产物 [a0,b0,a1,b1,…] → Gather → [行i | 行i+1]（解交织）
        //    pscale：单 Cast 产物（偶位=值，奇位=零）→ Gather → [顺序128字节 | 零]（压缩）
        for (int32_t i = 0; i < 2; i++) {
            for (int32_t j = 0; j < 128; j++) {
                pNzIdxUB_.SetValue(i * 128 + j, static_cast<uint8_t>(i + 2 * j));
            }
        }

        SetFlag<HardEvent::MTE3_V>(EVT_P_COPY);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID4);
    }

    __aicore__ inline void ComputeVec1(RunInfoMxfp8SoftmaxFp16& runInfo, uint32_t subLoopIdx)
    {
        // S1 尾行（actMSize ≤ 128）时本核（V1）没有自己的 S1 列，直接返回：
        // S1 按 V0=前半 / V1=后半分给两个核，行尾块的 S1 总数不足 256 时 V1 分不到列。
        // 此时若继续执行，会在上一轮的残留数据（stale mm1Res）上空算一遍，
        // 产出的 P/pscale 虽无消费者但浪费流水。外层 ExecuteTask 的跨核 flag
        // 是无条件 Set 的，这里直接 return 不会引起死锁。
        if (runInfo.actVecS1Size == 0U) {
            return;
        }

        uint32_t c1v1Loop = AttentionCommon::CeilDiv(runInfo.actSingleLoopS2Size, s2SubLoopSize); // ≤2
        bool isLastSub = (subLoopIdx + 1U == c1v1Loop);
        uint32_t validRows = isLastSub ? (runInfo.actSingleLoopS2Size - subLoopIdx * s2SubLoopSize) // 1..128
                                         :
                                         s2SubLoopSize;
        uint32_t mmSlot = subLoopIdx % 2U;             // 对齐 cube FixpipeMm1 ring
        uint32_t stateSlot = runInfo.softmaxStateSlot; // mloop % 3
        uint32_t taskSlot = runInfo.loop % 3U;         // exp/pscale/pSlot 共用

        // Reuse each P slot as soon as its own copy completes.
        uint32_t pCopyEvent = subLoopIdx == 0U ? EVT_P_COPY : EVENT_ID4;
        WaitFlag<HardEvent::MTE3_V>(pCopyEvent);

        // 行首复位 accMax：新 [bN2,gS1] 行的槽位含 3 行前旧值，必须复位为 MIN
        if (runInfo.isFirstS2Loop && subLoopIdx == 0U) {
            Duplicate(softmaxMaxUB_[stateSlot * UB_SMAX_SLOT], MIN_HALF_VALUE, UB_SMAX_SLOT);
        }

        // 块首快照 m_{k-1}：非首块的 subLoop 0 时拷贝 accMax → blockStartMaxUB
        if (subLoopIdx == 0U && !runInfo.isFirstS2Loop) {
            DataCopy(blockStartMaxUB_, softmaxMaxUB_[stateSlot * UB_SMAX_SLOT], UB_SMAX_SLOT);
        }

        if (validRows < s2SubLoopSize) {
            constexpr uint32_t pSlotSize = QfaVectorApi::QFA_UB_P_SLOT; // P slot includes four 32-byte padding blocks
            Duplicate(vec1PUb_.template ReinterpretCast<uint8_t>()[subLoopIdx * pSlotSize], static_cast<uint8_t>(0U),
                      pSlotSize);
        }
        QfaVectorApi::VfSoftmaxFp16ToFp8(vec1PUb_, softmaxMaxUB_, subLoopUsedMaxUB_, pNzIdxUB_, mm1ResUB_, pScaleHalf_,
                                         validRows, subLoopIdx, mmSlot, stateSlot);

        // P → AIC L1（MTE3；subLoop 粒度槽：ring = loop×2+subLoopIdx，核间交错 L1_PSLOT）
        SetFlag<HardEvent::V_MTE3>(EVT_P_COPY);
        WaitFlag<HardEvent::V_MTE3>(EVT_P_COPY);
        // Scatter each 16KB UB half into its half of the full 32KB L1 NZ tile.
        // UB groups have 4128-byte pitch; full-S2 L1 groups have 8192-byte pitch.
        // ——结构性消除旧跳写公式 subLoopIdx×4KB 越过 s2Align64=192 列组边界的 bug
        uint32_t pRing = L1_PRING(runInfo.loop, subLoopIdx);
        constexpr uint32_t pSlotSize = QfaVectorApi::QFA_UB_P_SLOT; // UB padding is skipped during the copy to L1
        // Interleave the two S2 halves into one [256,128] NZ tile.
        DataCopyParams pCopy;
        pCopy.blockCount = 4U;
        pCopy.blockLen = 128U;
        pCopy.srcStride = 1U;
        pCopy.dstStride = 128U;
        DataCopy(l1SharedTensor_[L1_PSLOT(constInfo_.subBlockIdx, pRing) + subLoopIdx * 4096U],
                 vec1PUb_[subLoopIdx * pSlotSize], pCopy);

        SetFlag<HardEvent::MTE3_V>(pCopyEvent);

        // ===== 末 subLoop epilogue：跨块因子 + pscale 网格影像直发 =====
        if (isLastSub) {
            // 跨块 rescale 因子（非首块才算）
            if (!runInfo.isFirstS2Loop) {
                QfaVectorApi::VfCalcRescaleFactor(softmaxExpUB_, blockStartMaxUB_, softmaxMaxUB_, taskSlot, stateSlot);
            }
            // pscale：所有 subLoop 统一计算 + 网格直落（末 subLoop 的 k_i = K_final——
            // accMax 单调累积 → Δk 恒 0 → 127，走通用路径无需特判；读侧每个被读的 y 行
            // 都由本任务的 V1 写过，无陈旧读取，不需要 cube 侧预填）
            LocalTensor<SCALE_T>& pScaleL1Target =
                (constInfo_.subBlockIdx == 0U) ? pScaleL1V0Tensor_ : pScaleL1V1Tensor_;
            LocalTensor<uint8_t> pscaleGridU8 = pscaleGridUB_.template ReinterpretCast<uint8_t>();
            for (uint32_t i = 0; i < 1U; i++) {
                uint32_t psRing = L1_PRING(runInfo.loop, i);
                QfaVectorApi::VfCalcPScale(pscaleGridU8, subLoopUsedMaxUB_, softmaxMaxUB_, psRing, i, stateSlot);
            }
            SetFlag<HardEvent::V_MTE3>(EVT_P_COPY);
            WaitFlag<HardEvent::V_MTE3>(EVT_P_COPY);
            for (uint32_t i = 0; i < 1U; i++) {
                uint32_t psRing = L1_PRING(runInfo.loop, i);
                DataCopyParams scaleCopy;
                scaleCopy.blockCount = 8U;
                scaleCopy.blockLen = 2U;
                scaleCopy.srcStride = 1U;
                scaleCopy.dstStride = 3U;
                DataCopy(pScaleL1Target
                             .template ReinterpretCast<uint8_t>()[(psRing / 2U) * L1_PSCALE_SINGLE_SLOT_SIZE + i * 64U],
                         pscaleGridU8[psRing * UB_PSCALE_SLOT], scaleCopy);
            }
        }
    }

    __aicore__ inline void ComputeVec2(RunInfoMxfp8SoftmaxFp16& runInfo)
    {
        // V1 半无有效列（actVecS1Size==0）：C2 由 cube 侧防御跳过，V2 同步早退
        // （外层 C2_V2 flag 无条件 Set，早退不破坏跨核配平）
        if (runInfo.actVecS1Size == 0U) {
            return;
        }

        uint32_t taskSlot = runInfo.loop % 3;
        if (runInfo.isFirstS2Loop) {
            QfaVectorApi::VfFlashUpdateAcc<false>(stage2OutUB_, accRowsumUB_, mm2ResUB_, softmaxExpUB_, taskSlot);
        } else {
            QfaVectorApi::VfFlashUpdateAcc<true>(stage2OutUB_, accRowsumUB_, mm2ResUB_, softmaxExpUB_, taskSlot);
        }

        // ===== (2) 末块：除 + cast + 转置写 + GM 输出 =====
        if (runInfo.isLastS2Loop) {
            // 踩踏闸：累积步骤读尽 mm2ResUB 后，才允许转置写以别名区覆写它
            PipeBarrier<PIPE_V>();
            // 转置带 pad 输出（T 后缀 = Transposed），别名落 mm2Res 区 [0, ~36KB)
            LocalTensor<bfloat16_t> outputT = mm2ResUB_.template ReinterpretCast<bfloat16_t>();
            QfaVectorApi::VfDivAndCast(outputT, stage2OutUB_, accRowsumUB_);

            // ===== GM 输出（V 管道写尽 outputT 后才允许 MTE3 搬运）=====
            SetFlag<HardEvent::V_MTE3>(EVT_GM_OUT);
            WaitFlag<HardEvent::V_MTE3>(EVT_GM_OUT);
            // colCount=144 = 128 数据 + 16 pad：助手 srcStride 公式算得 1 个 32B 单位，
            // 每行末尾恰好跳过填充块（无需压实拷贝）
            FaUbTensor<OUT_T> ubTensor{outputT, runInfo.actVecS1Size, 144U};
            GmCoordS1Only gmCoord{
                runInfo.bIdx,         runInfo.realN2Idx, 0U, runInfo.gS1Idx + runInfo.vecS1BaseIdx, 0U,
                runInfo.actVecS1Size, constInfo_.dSizeV};
            attenOutUbToGm_(outGmTensor_, ubTensor, gmCoord);

            if (constInfo_.isSoftmaxLseEnable) {
                QfaVectorApi::VfCalcLse(lseUb_, softmaxMaxUB_, accRowsumUB_, runInfo.softmaxStateSlot);
                SetFlag<HardEvent::V_MTE3>(EVT_LSE_OUT);
                WaitFlag<HardEvent::V_MTE3>(EVT_LSE_OUT);
                uint64_t lseOffset = (static_cast<uint64_t>(runInfo.bIdx) * constInfo_.realN2Size + runInfo.realN2Idx) *
                                         constInfo_.s1Size +
                                     runInfo.gS1Idx + runInfo.vecS1BaseIdx;
                SafeStrideCopy<float>(softmaxLseGm_[lseOffset], lseUb_, 1U, runInfo.actVecS1Size * sizeof(float), 0U,
                                      0U);
            }
        }
    }

    __aicore__ inline void ClearOutput()
    {
        if (!constInfo_.needInitOutput) {
            return;
        }
        int64_t totalSize = static_cast<int64_t>(constInfo_.bSize) * constInfo_.realN2Size * constInfo_.realGSize *
                            constInfo_.s1Size * constInfo_.dSizeV;
        // AIV 数 = 2×coreNum（MIX_AIC_1_2，coreNum 为 AIC 数——主线同款公式）
        uint32_t aivNum = 2U * constInfo_.coreNum;
        int64_t sliceSize = (totalSize + aivNum - 1) / aivNum;
        // 切片界 32B 对齐（D=128 ⇒ totalSize 恒 16 倍数，尾块天然对齐，块界 8192 亦 16 倍数）
        sliceSize = (sliceSize + 15) / 16 * 16;
        int64_t offset = static_cast<int64_t>(constInfo_.aivIdx) * sliceSize;
        // 零源整备（V 管道，无条件——LSE 切片可独立于 attention 切片非空）
        Duplicate(zeroOutUB_, static_cast<OUT_T>(0.0f), UB_ZERO_OUT_SIZE);
        SetFlag<HardEvent::V_MTE3>(EVT_ZERO_OUT);
        WaitFlag<HardEvent::V_MTE3>(EVT_ZERO_OUT);
        if (offset < totalSize) {
            int64_t count = totalSize - offset;
            if (count > sliceSize) {
                count = sliceSize;
            }
            // 分块清零（16KB/块）
            while (count > 0) {
                uint32_t chunk =
                    static_cast<uint32_t>(AttentionCommon::Min(count, static_cast<int64_t>(UB_ZERO_OUT_SIZE)));
                DataCopy(attentionOutGm_[offset], zeroOutUB_, chunk);
                offset += chunk;
                count -= chunk;
            }
        }
        // LSE 张量同清（isSoftmaxLseEnable 时；[B, realN2, S1] fp32——golden padding 行为 0）。
        // total 可为奇数（s1Size 任意）→ 块长不保证 32B 对齐，走 DataCopyPad 路径
        if (constInfo_.isSoftmaxLseEnable) {
            int64_t totalLse = static_cast<int64_t>(constInfo_.bSize) * constInfo_.realN2Size * constInfo_.s1Size;
            int64_t lseSlice = (totalLse + aivNum - 1) / aivNum;
            int64_t lseOffset = static_cast<int64_t>(constInfo_.aivIdx) * lseSlice;
            LocalTensor<float> zeroF32 = zeroOutUB_.template ReinterpretCast<float>();
            while (lseOffset < totalLse) {
                int64_t cnt = totalLse - lseOffset;
                if (cnt > lseSlice) {
                    cnt = lseSlice;
                }
                uint32_t bytes = static_cast<uint32_t>(AttentionCommon::Min(
                    cnt * static_cast<int64_t>(sizeof(float)), static_cast<int64_t>(UB_ZERO_OUT_SIZE * sizeof(OUT_T))));
                SafeStrideCopy<float>(softmaxLseGm_[lseOffset], zeroF32, 1U, bytes, 0U, 0U);
                lseOffset += bytes / sizeof(float);
            }
        }
        // 本核全部清零 MTE3 排空后才过 SyncAll（主线末对 Set/Wait<MTE3_V> 同款；
        // 无拷贝的核 Set 立即触发，Wait 无碍——每核 Set/Wait 恒配平）
        SetFlag<HardEvent::MTE3_V>(EVT_ZERO_OUT);
        WaitFlag<HardEvent::MTE3_V>(EVT_ZERO_OUT);
        SyncAll(); // 切片为空的 AIV 也须到屏障（全 AIV 参与配平）
    }

    __aicore__ inline void WriteLseNegInf(uint32_t bIdx, uint32_t realN2Idx, uint32_t rowCount)
    {
        static constexpr int32_t NEG_INF_BITS = static_cast<int32_t>(0xFF800000U); // fp32 -inf 位模式
        const float negInfValue = *reinterpret_cast<const float*>(&NEG_INF_BITS);
        Duplicate(lseInfUb_, negInfValue, UB_LSE_SIZE);
        SetFlag<HardEvent::V_MTE3>(EVT_LSE_OUT);
        WaitFlag<HardEvent::V_MTE3>(EVT_LSE_OUT);

        uint64_t base = (static_cast<uint64_t>(bIdx) * constInfo_.realN2Size + realN2Idx) * constInfo_.s1Size;
        uint32_t offset = 0U;
        while (offset < rowCount) {
            uint32_t chunk = AttentionCommon::Min(rowCount - offset, UB_LSE_SIZE);
            SafeStrideCopy<float>(softmaxLseGm_[base + offset], lseInfUb_, 1U, chunk * sizeof(float), 0U, 0U);
            offset += chunk;
        }
    }

    __aicore__ inline void ReleaseTensorsVec()
    {
        WaitFlag<HardEvent::MTE3_V>(EVT_P_COPY);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID4);
    }
};

} // namespace QFA_KERNEL

#endif // QUANT_FLASH_ATTN_BLOCK_VECTOR_MXFP8_SOFTMAX_FP16_H_
