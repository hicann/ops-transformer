/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_flash_attn_template_tiling_key.h
 * \brief QuantFlashAttn 统一模板 tiling key 参数表（各 quant_mode 共用一张表，以 QuantMode 位区分）：
 *        mxfp4 DN（quant_mode=5）与 MxFP8 Softmax FP16（quant_mode=3）。
 *        注意：本文件不 include tiling_data——QuantFlashAttnTilingData 按 quant_mode 分布局实现，
 *        使用方须先 include 本算子对应模式的 tiling_data，再 include 本文件。
 */

#ifndef TEMPLATE_TILING_KEY_QUANT_FLASH_ATTN_H_
#define TEMPLATE_TILING_KEY_QUANT_FLASH_ATTN_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define ASCENDC_TPL_5_BW 5
#define ASCENDC_TPL_10_BW 10

// ============================== InOutLayoutType（8bit）：q/out 布局组合 ==============================
#define InOutLayoutType_BSND 0
#define InOutLayoutType_BNSD 1
#define InOutLayoutType_TND 2
#define InOutLayoutType_NTD 3
#define InOutLayoutType_BNSD_BSND_COMBO 4 // q=BNSD, out=BSND（mxfp4 DN 专用组合）
#define InOutLayoutType_BSND_BSND InOutLayoutType_BSND
#define InOutLayoutType_BNSD_BNSD InOutLayoutType_BNSD
#define InOutLayoutType_TND_TND InOutLayoutType_TND
#define InOutLayoutType_NTD_TND InOutLayoutType_NTD

// mxfp4 DN 侧布局/存储宏名（对齐统一取值域）
#define LAYOUT_ENUM_BSND InOutLayoutType_BSND
#define LAYOUT_ENUM_BNSD InOutLayoutType_BNSD
#define LAYOUT_ENUM_TND InOutLayoutType_TND
#define LAYOUT_ENUM_BNSD_BSND InOutLayoutType_BNSD_BSND_COMBO

// ============================== KvLayoutType（2bit） ==============================
#define KvLayoutType_NO_PA 0
#define KvLayoutType_PA_BSND 1
#define KvLayoutType_PA_BNSD 2
#define KvLayoutType_PA_NZ 3
#define KvLayoutType_PA_BBND KvLayoutType_PA_BSND
#define KvLayoutType_PA_BNBD KvLayoutType_PA_BNSD

// mxfp4 DN 侧 KV 存储模式宏名（对齐统一取值域）
#define KV_STORAGE_MODE_CONTINUE KvLayoutType_NO_PA
#define KV_STORAGE_MODE_PA_BSND KvLayoutType_PA_BSND
#define KV_STORAGE_MODE_PA_BNSD KvLayoutType_PA_BNSD

// ============================== Config（10bit）：切分配置编码 ==============================
#define Config_DN_FIXED 0 // mxfp4 DN 不使用切分配置，固定 0
#define Config_S1Aligned128_S2Aligned512_DAligned64_DVAligned64 0
#define Config_S1Aligned128_S2Aligned512_DAligned128_DVAligned128 1
#define Config_S1Aligned128_S2Aligned256_DAligned128_DVAligned128 2
#define Config_S1Aligned128_S2Aligned256_DAligned256_DVAligned256 3
#define Config_S1Aligned128_S2Aligned512_DAligned72_DVAligned72 4
#define Config_S1Aligned256_S2Aligned256_DAligned128_DVAligned128 5

// ============================== QuantMode（5bit） ==============================
#define QFA_HIF8_FP32 0
#define QFA_MXFP8_FP32_PREFILL 1
#define QFA_MXFP8_FP32_DECODE 2
#define QFA_MXFP8_SOFTMAX_FP16 3
#define QFA_MXFP4_DN 5
#define QFA_GQA_FP8_FULLQUANT 6

// ============================== mxfp4 DN kernel 布局枚举与类型萃取 ==============================
enum class QFA_LAYOUT : uint32_t {
    BSND = LAYOUT_ENUM_BSND,
    BNSD = LAYOUT_ENUM_BNSD,
    TND = LAYOUT_ENUM_TND,
};

template <typename QUANT_T, typename SCALE_T, typename OUT_T, const bool PAGE_ATTENTION = false,
          QFA_LAYOUT LAYOUT_T = QFA_LAYOUT::BSND, QFA_LAYOUT KV_LAYOUT_T = QFA_LAYOUT::BSND,
          QFA_LAYOUT OUT_LAYOUT_T = QFA_LAYOUT::BSND, const bool HAS_MASK = false, typename... Args>
struct QFAType {
    using quantType = QUANT_T;
    using scaleType = SCALE_T;
    using outputType = OUT_T;
    static constexpr bool pageAttention = PAGE_ATTENTION;
    static constexpr QFA_LAYOUT qLayout = LAYOUT_T;
    static constexpr QFA_LAYOUT kvLayout = KV_LAYOUT_T;
    static constexpr QFA_LAYOUT outLayout = KV_LAYOUT_T;
    static constexpr bool hasMask = HAS_MASK;
};

// ============================== 统一参数表（6 参，bit 布局自低到高） ==============================
ASCENDC_TPL_ARGS_DECL(QuantFlashAttn,
                      //    bit 0-7 InOutLayoutType
                      ASCENDC_TPL_UINT_DECL(InOutLayoutType, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 255),
                      //    bit 8-17 Config
                      ASCENDC_TPL_UINT_DECL(Config, ASCENDC_TPL_10_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 1023),
                      //    bit 18-22 QuantMode
                      ASCENDC_TPL_UINT_DECL(QuantMode, ASCENDC_TPL_5_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 31),
                      //    bit 23 HasAttenMask
                      ASCENDC_TPL_BOOL_DECL(HasAttenMask, 0, 1),
                      //    bit 24-25 KvLayoutType
                      ASCENDC_TPL_UINT_DECL(KvLayoutType, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 3),
                      //    bit 26 IsFd
                      ASCENDC_TPL_BOOL_DECL(IsFd, 0, 1));

// ============================== 统一特化选择：DN 笛卡尔积 + mode-3 单组合 ==============================
ASCENDC_TPL_SEL(
    // mxfp4 DN（quant_mode=5）：layout × kv × mask 笛卡尔，Config/IsFd 固定
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, LAYOUT_ENUM_BSND, LAYOUT_ENUM_BNSD,
                                              LAYOUT_ENUM_TND, LAYOUT_ENUM_BNSD_BSND),
                         ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_DN_FIXED),
                         ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, QFA_MXFP4_DN),
                         ASCENDC_TPL_BOOL_SEL(HasAttenMask, 0, 1),
                         ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KV_STORAGE_MODE_CONTINUE,
                                              KV_STORAGE_MODE_PA_BSND, KV_STORAGE_MODE_PA_BNSD),
                         ASCENDC_TPL_BOOL_SEL(IsFd, 0), ASCENDC_TPL_TILING_STRUCT_SEL(QuantFlashAttnTilingData)),
    // MxFP8 Softmax FP16（quant_mode=3）：单模板组合
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD),
                         ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST,
                                              Config_S1Aligned256_S2Aligned256_DAligned128_DVAligned128),
                         ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, QFA_MXFP8_SOFTMAX_FP16),
                         ASCENDC_TPL_BOOL_SEL(HasAttenMask, 0),
                         ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA),
                         ASCENDC_TPL_BOOL_SEL(IsFd, 0), ASCENDC_TPL_TILING_STRUCT_SEL(QuantFlashAttnTilingData)));

#endif // TEMPLATE_TILING_KEY_QUANT_FLASH_ATTN_H_
