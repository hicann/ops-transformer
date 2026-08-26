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
 * \file mixed_quant_sparse_flash_mla_common_arch35.h
 * \brief
 */
#ifndef MIXED_QUANT_SPARSE_FLASH_MLA_COMMON_ARCH35_H
#define MIXED_QUANT_SPARSE_FLASH_MLA_COMMON_ARCH35_H
#include <type_traits>
#include "kernel_tiling/kernel_tiling.h"
#include "../mixed_quant_sparse_flash_mla_common.h"
#if __has_include("../../../sparse_flash_mla/op_kernel/arch35/common/static_buffer.h")
#include "../../../sparse_flash_mla/op_kernel/arch35/common/static_buffer.h"
#else
#include "../../sparse_flash_mla/arch35/common/static_buffer.h"
#endif

constexpr uint64_t BLOCK_BYTE = 32;
constexpr uint32_t NEGATIVE_MIN_VALUE_FP32 = 0xFF7FFFFF;

constexpr uint32_t L0AB_SHARED_SIZE_64K = 65536;  // 65536表示64*1024
constexpr uint32_t L0C_SHARED_SIZE_256K = 262144; // 262144表示256 * 1024

constexpr uint32_t BUFFER_SIZE_16K = 16384;   // 16384表示16 * 1024
constexpr uint32_t BUFFER_SIZE_32K = 32768;   // 32768表示32 * 1024
constexpr uint32_t BUFFER_SIZE_96K = 98304;   // 98304表示96 * 1024
constexpr uint32_t BUFFER_SIZE_256K = 262144; // 262144表示256 * 1024

constexpr uint32_t CV_RATIO = 2;
constexpr uint64_t SYNC_MODE = 4;
constexpr uint32_t BATCH_CONSISTENCY_MAX_REDUCE_BLOCK_NUM = 33U; // 同token最大规约块数

// ===== C 侧 buffer 元素个数 (用于 tensor 偏移及地址递增) =====
constexpr uint32_t L1Q_ELEM_PER_BUF = 16384;        // Q_T 元素, 每个 L1Q buffer (与 BUFFER_SIZE_16K 对齐)
constexpr uint32_t L1_RIGHT_ELEM_PER_BLOCK = 65536; // Q_T 元素, 每个 L1Right 块 (= s2BaseSize * dBaseSize)
constexpr uint32_t L0A_ELEM_PER_BUF = 8192;         // Q_T 元素, 每个 L0A buffer (16KB)
constexpr uint32_t L0B_ELEM_PER_BUF = 16384;        // Q_T 元素, 每个 L0B buffer (32KB)
constexpr uint32_t L0C_ELEM_PER_BUF = 32768;        // T   元素, 每个 L0C buffer

// ===== C 侧核内 flag id (HardEvent 各自独立命名空间) =====
#define INNERCORE_L0AB(s) (s)       // 0,1: L0A+L0B 共用, M_MTE1 / MTE1_M
#define INNERCORE_L0C(s) (s)        // 0,1: L0C, FIX_M / M_FIX
#define INNERCORE_L1Q(s) (s)        // 0,1,2: L1Q, MTE2_MTE1 / MTE1_MTE2
#define INNERCORE_L1KV(s) (3 + (s)) // 3,4,5: L1Right, MTE2_MTE1 / MTE1_MTE2

// ===== V 侧主流程 buffer slot id (MTE2_V/V_MTE2/MTE3_V/V_MTE3 各自独立) =====
#define INNERCORE_STAGE0_IN(s) (s)        // 0,1: stage0In, MTE2_V / V_MTE2
#define INNERCORE_STAGE0_OUT(s) (2 + (s)) // 2,3: stage0Out, MTE3_V / V_MTE3
#define INNERCORE_STAGE1(s) (4 + (s))     // 4,5: stage1, MTE3_V / V_MTE3
#define INNERCORE_STAGE2 (6)              // 6:   stage2, MTE3_V / V_MTE3
#define INNERCORE_SINKS_SYNC (7)          // 7:   sinks,  MTE2_V / V_MTE2

// ===== GetKVPhyAddr 独立阶段, 保留原 flag 值 =====
#define INNERCORE_PHYADDR_BLKTABLE_FREE (3)   // V_MTE2, blkTable 空闲
#define INNERCORE_PHYADDR_BLKTABLE_READY (8)  // MTE2_V, blkTable 就绪
#define INNERCORE_PHYADDR_SPARSEIDX_FREE (4)  // V_MTE2, sparseIdx 空闲
#define INNERCORE_PHYADDR_SPARSEIDX_READY (6) // MTE2_V, sparseIdx 就绪
#define INNERCORE_PHYADDR_KVADDR_READY (5)    // V_MTE3, kvPhyAddr 就绪
#define INNERCORE_PHYADDR_KVADDR_FREE (7)     // MTE3_V, kvPhyAddr 空闲

// ===== V 侧其余核内 flag id (FD / batch-consistency / LSE / init) =====
// 各 HardEvent 命名空间独立; 主流程 stage 槽位 id 已占 0~6/7, 其余事件在其命名空间内取不冲突值。
// V_MTE2: STAGE0_IN 0,1
#define INNERCORE_REDUCE_MAXSUM_V_MTE2 (2) // batch-consistency reduce max/sum
#define INNERCORE_INTRAPARTIALO_V_MTE2 (3) // batch-consistency partial O
#define INNERCORE_FD_V_MTE2(s) (4 + (s))   // 4,5: flash decode

// MTE2_V: STAGE0_IN 0,1; SINKS 7
#define INNERCORE_REDUCE_MTE2_V (2) // batch-consistency reduce
#define INNERCORE_FD_MTE2_V (3)     // flash decode

// V_MTE3: STAGE0_OUT 2,3; STAGE1 4,5; STAGE2 6
#define INNERCORE_LSE_V_MTE3 (1) // LSE out (条件阶段)

// MTE3_V: STAGE0_OUT 2,3; STAGE1 4,5; STAGE2 6
#define INNERCORE_STAGE_FD_MTE3_V (7) // FD/BC staging (StageVec1Lse)
#define INNERCORE_LSE_MTE3_V (0)      // LSE out (条件阶段)
#define INNERCORE_FD_MTE3_V (1)       // FD mte3ToV (SyncAll 之后阶段)
#define INNERCORE_INITOUT_MTE3_V (0)  // init 阶段 (CleanOutput)

// MTE3_MTE2 (batch consistency + fd)
#define INNERCORE_INTRALSE_MTE3_MTE2(s) (s)        // 0,1
#define INNERCORE_INTRAATTN_MTE3_MTE2(s) (2 + (s)) // 2,3
#define INNERCORE_FD_MTE3_MTE2 (4)

// ===== 跨核 flag id (mode 4, 保留现有数值分配) =====
#define CROSSCORE_L1P(s) (s) // 0,1
#define CROSSCORE_BMM2 (2)
#define CROSSCORE_BMM1(s) (3 + (s))  // 3,4
#define CROSSCORE_V0RES(s) (5 + (s)) // 5,6,7 (GM backward)

namespace BaseApi {
__aicore__ constexpr uint64_t Align2Func(uint64_t data)
{
    return (data + 1UL) >> 1UL << 1UL; // 向上2对齐, +1移位2
}

__aicore__ constexpr uint64_t Align8Func(uint64_t data)
{
    return (data + 7UL) >> 3UL << 3UL; // 向上8对齐, +7移位3
}

__aicore__ constexpr uint64_t Align16Func(uint64_t data)
{
    return (data + 15UL) >> 4UL << 4UL; // 向上16对齐, +15移位4
}

__aicore__ constexpr uint64_t Align64Func(uint64_t data)
{
    return (data + 63UL) >> 6UL << 6UL; // 向上64对齐, +63移位6
}
} // namespace BaseApi

#define TEMPLATE_INTF \
    template <typename Q_T, typename KV_T, typename T, typename OUTPUT_T, bool isFd, bool isPa, QSMLA_LAYOUT LAYOUT_T, \
              QSMLA_LAYOUT KV_LAYOUT_T, QSMLATemplateMode TEMPLATE_MODE, bool IS_SPLIT_G, \
              SCALE_CONTIGUOUS_MODE QUANT_MODE, bool IS_BATCH_CONSISTENCY, bool IS_VEC_S2PHYADDR, bool HIGH_PERF>

#define TEMPLATE_INTF_ARGS \
    Q_T, KV_T, T, OUTPUT_T, isFd, isPa, LAYOUT_T, KV_LAYOUT_T, TEMPLATE_MODE, IS_SPLIT_G, QUANT_MODE, \
        IS_BATCH_CONSISTENCY, IS_VEC_S2PHYADDR, HIGH_PERF

#define CUBE_BLOCK_TRAITS_TYPE_FIELDS(X) \
    X(Q_T) \
    X(KV_T) \
    X(T) \
    X(OUTPUT_T)

#define CUBE_BLOCK_TRAITS_CONST_FIELDS(X) \
    X(isFd, bool, false) \
    X(isPa, bool, true) \
    X(LAYOUT_T, QSMLA_LAYOUT, QSMLA_LAYOUT::BSND) \
    X(KV_LAYOUT_T, QSMLA_LAYOUT, QSMLA_LAYOUT::PA_BBND) \
    X(TEMPLATE_MODE, QSMLATemplateMode, QSMLATemplateMode::CSA_TEMPLATE_MODE) \
    X(IS_SPLIT_G, bool, false) \
    X(QUANT_MODE, SCALE_CONTIGUOUS_MODE, SCALE_CONTIGUOUS_MODE::CONTIGUOUS) \
    X(IS_BATCH_CONSISTENCY, bool, false) \
    X(IS_VEC_S2PHYADDR, bool, false) \
    X(HIGH_PERF, bool, false)

/* 1. 生成带默认值的模版Template */
#define GEN_TYPE_PARAM(name) typename name,
#define GEN_CONST_PARAM(name, type, default_val) type name = default_val,

#define TEMPLATES_DEF \
    template <CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TYPE_PARAM) CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_CONST_PARAM) bool end = \
                  true>

/* 2. 生成不带带默认值的模版Template */
#define GEN_TEMPLATE_TYPE_NODEF(name) typename name,
#define GEN_TEMPLATE_CONST_NODEF(name, type, default_val) type name,
#define TEMPLATES_DEF_NO_DEFAULT \
    template <CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TEMPLATE_TYPE_NODEF) \
                  CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_TEMPLATE_CONST_NODEF) bool end>

/* 3. 生成有默认值的Args */
#define GEN_ARG_NAME(name, ...) name,
#define TEMPLATE_ARGS \
    CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_ARG_NAME) \
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_ARG_NAME) \
    end

#endif
