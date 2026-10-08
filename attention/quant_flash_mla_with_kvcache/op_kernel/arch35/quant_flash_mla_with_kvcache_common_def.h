/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_flash_mla_with_kvcache_common_def.h
 * \brief QuantFlashMlaWithKvcache 公共枚举与常量定义（tiling key取值等, 供tiling/kernel/autogen共用）
 *        注意: 常量须位于全局作用域, template_tiling_key的ASCENDC_TPL宏在全局作用域展开
 */

#ifndef QUANT_FLASH_MLA_WITH_KVCACHE_COMMON_DEF_H_
#define QUANT_FLASH_MLA_WITH_KVCACHE_COMMON_DEF_H_

// ASCENDC_TPL位宽宏（CANN头文件仅提供1/2/4/8, 与QFA一致在此补充）
#ifndef ASCENDC_TPL_5_BW
#define ASCENDC_TPL_5_BW 5
#endif
#ifndef ASCENDC_TPL_10_BW
#define ASCENDC_TPL_10_BW 10
#endif

// QuantMode（quant_mode属性/tiling key取值）
// 1: Q/K均为FP8_E4M3全量化, Q per-token-head动态量化, K per-tensor静态量化
#define QMLA_MLA_FP8_E4M3_FULLQUANT 1

// InOutLayoutType: 输入layout固定TND, 输出layout由layout_out属性决定
#define InOutLayoutType_TND_TND 0  // 输出TND
#define InOutLayoutType_TND_BSND 1 // 输出BSND
#define InOutLayoutType_TND_BNSD 2 // 输出BNSD
#define InOutLayoutType_TND_NTD 3  // 输出NTD

// Config: 分块对齐（M=64（cube视角, SOuter=32*cvRatio）, S2Inner=256, D=576, DV=512）
// 与FIA MLA实现及metadata分核mBaseSize=32*cvRatio=64保持一致
#define Config_S1Aligned64_S2Aligned256_DAligned576_DVAligned512 0

// KvLayoutType
#define KvLayoutType_PA_BBND 0
#define KvLayoutType_PA_BNBD 1
#define KvLayoutType_PA_NZ 2

// 与QFA一致: 编译工具链解析ASCENDC_TPL_BOOL_DECL时通过extract_num提取数字,
// 需将false/true宏替换为0/1, 否则模板tiling参数解析报"values is empty"
#define false 0
#define true 1

// 与kernel侧LayOutTypeEnum（util.h）取值一致, kernel入口通过static_cast转换
enum class QmlaInferLayOutType {
    None = 0,
    LAYOUT_BSH = 1,
    LAYOUT_SBH = 2,
    LAYOUT_BNSD = 3,
    LAYOUT_TND = 4,
    LAYOUT_NTD_TND = 5,
    LAYOUT_NTD = 6
};

// inOutLayoutType → (输入layout, 输出layout), BSND输出与BSH模板等价
static constexpr QmlaInferLayOutType InOutLayoutTypeValue[4][2] = {
    {QmlaInferLayOutType::LAYOUT_TND, QmlaInferLayOutType::LAYOUT_TND},  // InOutLayoutType_TND_TND
    {QmlaInferLayOutType::LAYOUT_TND, QmlaInferLayOutType::LAYOUT_BSH},  // InOutLayoutType_TND_BSND
    {QmlaInferLayOutType::LAYOUT_TND, QmlaInferLayOutType::LAYOUT_BNSD}, // InOutLayoutType_TND_BNSD
    {QmlaInferLayOutType::LAYOUT_TND, QmlaInferLayOutType::LAYOUT_NTD},  // InOutLayoutType_TND_NTD
};

// 与kernel侧S1TemplateType/S2TemplateType/DTemplateType（util_regbase.h）取值一致
enum class QmlaS1TemplateType {
    Aligned16 = 16,
    Aligned64 = 64,
    Aligned128 = 128,
    Aligned256 = 256,
    NotAligned,
};

enum class QmlaS2TemplateType {
    Aligned16 = 16,
    Aligned32 = 32,
    Aligned64 = 64,
    Aligned128 = 128,
    Aligned256 = 256,
    Aligned512 = 512,
    NotAligned,
};

enum class QmlaDTemplateType {
    Aligned16 = 16,
    Aligned32 = 32,
    Aligned64 = 64,
    Aligned128 = 128,
    Aligned256 = 256,
    Aligned512 = 512,
    Aligned576 = 576,
    NotAligned,
};

struct QmlaConfigParams {
    QmlaS1TemplateType s1;
    QmlaS2TemplateType s2;
    QmlaDTemplateType d;
    QmlaDTemplateType dv;
};

// config → 分块模板参数
static constexpr QmlaConfigParams ConfigValue[] = {
    {QmlaS1TemplateType::Aligned64, QmlaS2TemplateType::Aligned256, QmlaDTemplateType::Aligned576,
     QmlaDTemplateType::Aligned512},
};

#endif // QUANT_FLASH_MLA_WITH_KVCACHE_COMMON_DEF_H_
