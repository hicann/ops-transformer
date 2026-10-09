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
 * \file mixed_quant_flash_attn_template_tiling_key.h
 * \brief MixedQuantFlashAttn TilingKey定义（伪量化，Q为BF16/FP16，K/V为mxFP4）
 */

#ifndef TEMPLATE_TILING_KEY_MIXED_QUANT_FLASH_ATTN_H_
#define TEMPLATE_TILING_KEY_MIXED_QUANT_FLASH_ATTN_H_

#include "ascendc/host_api/tiling/template_argument.h"
#include "utils/mixed_quant_flash_attn_common_def.h"
#include "mixed_quant_flash_attn_tiling_data.h"

#ifndef ORIG_DTYPE_QUERY
#define ORIG_DTYPE_QUERY (DT_BF16)
#endif

#ifndef ORIG_DTYPE_KEY
#define ORIG_DTYPE_KEY (DT_BF16)
#endif

#ifndef ORIG_DTYPE_ATTENTION_OUT
#define ORIG_DTYPE_ATTENTION_OUT (DT_BF16)
#endif

ASCENDC_TPL_ARGS_DECL(MixedQuantFlashAttn,
                      // q_out layout (8-bit)   TODO bit分配问题
                      //    0: BSND
                      //    1: BNSD
                      //    2: TND
                      ASCENDC_TPL_UINT_DECL(InOutLayoutType, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 2),

                      // kv存储类型 (8-bit)
                      //    0: 连续（不开pageattention）
                      //    1: PA_BBND
                      //    2: PA_BNBD
                      //    3: PA_NZ
                      ASCENDC_TPL_UINT_DECL(KvLayoutType, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 3),
                      // hasAttenMask (1-bit)
                      //    0: false
                      //    1: true
                      ASCENDC_TPL_BOOL_DECL(HasAttenMask, 0, 1),
                      // config (4-bit), support D=64/128/256
                      //    config=0: sOuter=64, sInner=128 → D=64,  DV=64
                      //    config=1: sOuter=32, sInner=256 → D=64,  DV=64
                      //    config=2: sOuter=64, sInner=128 → D=128, DV=128
                      //    config=3: sOuter=32, sInner=256 → D=128, DV=128
                      //    config=4: sOuter=32, sInner=256 → D=256, DV=256
                      //    config=5: sOuter=32, sInner=256 → D=128, DV=128
                      //    config=6: sOuter=32, sInner=512 → D=128, DV=128
                      //    config=7: sOuter=64, sInner=512 → D=128, DV=128
                      //    config=8: sOuter=48, sInner=512 → D=128, DV=128
                      ASCENDC_TPL_UINT_DECL(Config, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 8),
                      // quantComputeMode
                      ASCENDC_TPL_UINT_DECL(QuantComputeMode, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 2), );

ASCENDC_TPL_SEL(
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201))
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BSND, InOutLayoutType_BNSD),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA, KvLayoutType_PA_NZ,
                             KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned32_S2Aligned512_DAligned128_DVAligned128,
                             Config_S1Aligned48_S2Aligned512_DAligned128_DVAligned128,
                             Config_S1Aligned64_S2Aligned512_DAligned128_DVAligned128),
        ASCENDC_TPL_UINT_SEL(QuantComputeMode, ASCENDC_TPL_UI_LIST, A16C4_KV_MXFP4_SOFTMAX_FP32,
                             A16C4_KV_HIF4_SOFTMAX_FP32),
        ASCENDC_TPL_TILING_STRUCT_SEL(MixedQuantFlashAttnTilingData)),
#else
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BSND, InOutLayoutType_BNSD,
                             InOutLayoutType_TND),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned128),
        ASCENDC_TPL_UINT_SEL(QuantComputeMode, ASCENDC_TPL_UI_LIST, A16C4_KV_MXFP4_SOFTMAX_FP32),
        ASCENDC_TPL_TILING_STRUCT_SEL(MixedQuantFlashAttnTilingData)),
#endif
);
#endif // TEMPLATE_TILING_KEY_MIXED_QUANT_FLASH_ATTN_H_
