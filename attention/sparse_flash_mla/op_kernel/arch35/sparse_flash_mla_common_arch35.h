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
 * \file sparse_flash_mla_common_arch35.h
 * \brief
 */
#ifndef SPARSE_FLASH_MLA_COMMON_ARCH35_H
#define SPARSE_FLASH_MLA_COMMON_ARCH35_H
#include <type_traits>
#include "kernel_tiling/kernel_tiling.h"
#include "../sparse_flash_mla_common.h"
#include "common/static_buffer.h"
#include "common/smla_common_defs.h"

// ===== V 侧主流程 =====
#define INNERCORE_STAGE1(s) (4 + (s)) // 4,5  V_MTE3 / MTE3_V (stage1->L1)
#define INNERCORE_STAGE2 (6)          // 6    V_MTE3 / MTE3_V (vec2结果+attentionOut拷出+staging, 串行复用)
#define INNERCORE_STAGE0OUT_MTE3_MTE2(s) (s) // 0,1  Vec0 stage0OutBuf (原 mte3ToMte2)
#define INNERCORE_STAGE0OUT_MTE2_MTE3(s) (s) // 0,1  Vec0 stage0OutBuf (原 mte2ToMte3)
#define INNERCORE_SINKS_SYNC (7)             // 7    MTE2_V / V_MTE2

// ===== GetKVPhyAddr 独立相位 (保留原值; 与主流程相位分离, 可复用) =====
#define INNERCORE_PHYADDR_BLKTABLE_FREE (3)   // V_MTE2
#define INNERCORE_PHYADDR_BLKTABLE_READY (8)  // MTE2_V
#define INNERCORE_PHYADDR_SPARSEIDX_FREE (4)  // V_MTE2
#define INNERCORE_PHYADDR_SPARSEIDX_READY (6) // MTE2_V
#define INNERCORE_PHYADDR_KVADDR_READY (5)    // V_MTE3
#define INNERCORE_PHYADDR_KVADDR_FREE (7)     // MTE3_V

// ===== batch-consistency / LSE / FD / init =====
#define INNERCORE_REDUCE_MAXSUM_V_MTE2 (2)         // V_MTE2
#define INNERCORE_INTRAPARTIALO_V_MTE2 (3)         // V_MTE2
#define INNERCORE_FD_V_MTE2(s) (4 + (s))           // 4,5 V_MTE2
#define INNERCORE_REDUCE_MTE2_V (2)                // MTE2_V
#define INNERCORE_FD_MTE2_V (3)                    // MTE2_V
#define INNERCORE_LSE_V_MTE3 (1)                   // V_MTE3
#define INNERCORE_STAGE_FD_MTE3_V (7)              // MTE3_V (Stage* staging)
#define INNERCORE_LSE_MTE3_V (0)                   // MTE3_V
#define INNERCORE_FD_MTE3_V (1)                    // MTE3_V
#define INNERCORE_INITOUT_MTE3_V (0)               // MTE3_V (init, 与 LSE 相位分离)
#define INNERCORE_INTRALSE_MTE3_MTE2(s) (2 + (s))  // 2,3
#define INNERCORE_INTRAATTN_MTE3_MTE2(s) (4 + (s)) // 4,5
#define INNERCORE_FD_MTE3_MTE2 (6)

namespace SMLAKernel {
using AttentionCommon::Align2Func;
using AttentionCommon::Align8Func;
using AttentionCommon::Align16Func;
using AttentionCommon::Align64Func;
} // namespace SMLAKernel

#define TEMPLATE_INTF \
    template <typename Q_T, typename KV_T, typename T, typename OUTPUT_T, bool IS_FD, SMLA_LAYOUT LAYOUT_T, \
              SMLA_LAYOUT KV_LAYOUT_T, SMLATemplateMode TEMPLATE_MODE, bool IS_SPLIT_G, bool IS_BATCH_CONSISTENCY, \
              bool IS_VEC_S2PHYADDR, TopkValueMode TOPK_VALUE_MODE>

#define TEMPLATE_INTF_ARGS \
    Q_T, KV_T, T, OUTPUT_T, IS_FD, LAYOUT_T, KV_LAYOUT_T, TEMPLATE_MODE, IS_SPLIT_G, IS_BATCH_CONSISTENCY, \
        IS_VEC_S2PHYADDR, TOPK_VALUE_MODE

#define SMLA_CUBE_TRAIT_TYPES(X) \
    X(Q_T) \
    X(KV_T) \
    X(T) \
    X(OUTPUT_T)

#define SMLA_CUBE_TRAIT_CONSTANTS(X) \
    X(IS_FD, bool, false) \
    X(LAYOUT_T, SMLA_LAYOUT, SMLA_LAYOUT::BSND) \
    X(KV_LAYOUT_T, SMLA_LAYOUT, SMLA_LAYOUT::PA_BBND) \
    X(TEMPLATE_MODE, SMLATemplateMode, SMLATemplateMode::CSA_TEMPLATE_MODE) \
    X(IS_SPLIT_G, bool, false) \
    X(IS_BATCH_CONSISTENCY, bool, false) \
    X(IS_VEC_S2PHYADDR, bool, false) \
    X(TOPK_VALUE_MODE, TopkValueMode, TopkValueMode::TOPK_INDEX_MODE)

/* 1. 生成带默认值的模版Template */
#define SMLA_TEMPLATE_TYPE(name) typename name,
#define SMLA_TEMPLATE_CONSTANT(name, type, default_val) type name = default_val,

#define TEMPLATES_DEF \
    template <SMLA_CUBE_TRAIT_TYPES(SMLA_TEMPLATE_TYPE) SMLA_CUBE_TRAIT_CONSTANTS(SMLA_TEMPLATE_CONSTANT) bool end = \
                  true>

/* 2. 生成不带带默认值的模版Template */
#define SMLA_REQUIRED_TYPE(name) typename name,
#define SMLA_REQUIRED_CONSTANT(name, type, default_val) type name,
#define TEMPLATES_DEF_NO_DEFAULT \
    template <SMLA_CUBE_TRAIT_TYPES(SMLA_REQUIRED_TYPE) SMLA_CUBE_TRAIT_CONSTANTS(SMLA_REQUIRED_CONSTANT) bool end>

/* 3. 生成有默认值的Args */
#define SMLA_TEMPLATE_ARGUMENT(name, ...) name,
#define TEMPLATE_ARGS \
    SMLA_CUBE_TRAIT_TYPES(SMLA_TEMPLATE_ARGUMENT) \
    SMLA_CUBE_TRAIT_CONSTANTS(SMLA_TEMPLATE_ARGUMENT) \
    end

#endif
