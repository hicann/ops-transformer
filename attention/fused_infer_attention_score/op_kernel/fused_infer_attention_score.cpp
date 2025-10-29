/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_infer_attention_score.cpp
 * \brief
 */

#define FIA_IFA
#define FIA_PFA
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "fused_infer_attention_score_tilingkey.h"
// ifa must include before pfa
#define FIA_ENABLE_MLA
#include "../../incre_flash_attention/op_kernel/incre_flash_attention.cpp"
#include "../../prompt_flash_attention/op_kernel/prompt_flash_attention.cpp"

#include "fused_infer_attention_score_v3.cpp"

template<uint8_t Q_T, uint8_t KV_T, uint8_t OUT_T, uint8_t PAGE_ATTENTIOND, uint8_t LAYOUT_T, uint16_t KV_LAYOUT_T, uint16_t FLASH_DECODE, uint8_t ENABLE_PREFIX,
            uint8_t M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE, uint8_t M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T, uint8_t M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA, uint8_t M_V_QUANTMODE_P_PRECISION_MODE_I_BALANCE,
            uint8_t M_FIAFLAG_P_MMTYPETMP_I_MODEVAL, uint8_t P_CVDIFF_BASE_FLAG, uint8_t P_CVDIFF_MLA_FLAG, uint8_t P_TEMPLATE_VERSION, uint8_t TEMPLATE_MODE>
__global__ __aicore__ void fused_infer_attention_score(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
                                                             __gm__ uint8_t* pse_shift, __gm__ uint8_t* attenMask,
                                                             __gm__ uint8_t* actualSeqLengths, __gm__ uint8_t* actualSeqLengthsKV,
                                                             __gm__ uint8_t* deq_scale1, __gm__ uint8_t* quant_scale1,
                                                             __gm__ uint8_t* deq_scale2, __gm__ uint8_t* quant_scale2,
                                                             __gm__ uint8_t* quant_offset2, __gm__ uint8_t* antiquantScale,
                                                             __gm__ uint8_t* antiquantOffset, __gm__ uint8_t* blocktable,
                                                             __gm__ uint8_t* queryPaddingSize, __gm__ uint8_t* kvPaddingSize,
                                                             __gm__ uint8_t* keyAntiquantScale, __gm__ uint8_t* keyAntiquantOffset,
                                                             __gm__ uint8_t* valueAntiquantScale, __gm__ uint8_t* valueAntiquantOffset,
                                                             __gm__ uint8_t* keySharedPrefix, __gm__ uint8_t* valueSharedPrefix, __gm__ uint8_t* actualSharedPrefixLen,
                                                             __gm__ uint8_t* queryRope, __gm__ uint8_t* keyRope, __gm__ uint8_t* keyRopeAntiquantScale, __gm__ uint8_t* dequantScaleQuery,
                                                             __gm__ uint8_t* learnableSink, __gm__ uint8_t* qStartIdx, __gm__ uint8_t* kvStartIdx,
                                                             __gm__ uint8_t* attentionOut, __gm__ uint8_t* softmaxLse, __gm__ uint8_t* workspace,
                                                             __gm__ uint8_t* tiling) {
  // judge ifa or pfa or fia by range of tilingKey
  if constexpr (TEMPLATE_MODE == 2) { // 10^18
      prompt_flash_attention_FIAS<Q_T, KV_T, OUT_T, PAGE_ATTENTIOND, LAYOUT_T, KV_LAYOUT_T, FLASH_DECODE, ENABLE_PREFIX, M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE,
                    M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T, M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA, M_V_QUANTMODE_P_PRECISION_MODE_I_BALANCE,
                    M_FIAFLAG_P_MMTYPETMP_I_MODEVAL, P_CVDIFF_BASE_FLAG, P_CVDIFF_MLA_FLAG, P_TEMPLATE_VERSION, TEMPLATE_MODE>
                                  (query, key, value, pse_shift, attenMask, actualSeqLengths, 
                                  actualSeqLengthsKV, deq_scale1, quant_scale1,
                                  deq_scale2, quant_scale2, quant_offset2, antiquantScale, 
                                  antiquantOffset, blocktable, queryPaddingSize, kvPaddingSize, 
                                  keyAntiquantScale, keyAntiquantOffset, valueAntiquantScale, 
                                  valueAntiquantOffset, keySharedPrefix, valueSharedPrefix, 
                                  actualSharedPrefixLen, queryRope, keyRope, learnableSink, 
                                  attentionOut, softmaxLse, workspace, tiling);
  } else if (TEMPLATE_MODE == 0) { // 10^17
    fused_infer_attention<Q_T, KV_T, OUT_T, PAGE_ATTENTIOND, LAYOUT_T, KV_LAYOUT_T, FLASH_DECODE, ENABLE_PREFIX, M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE,
                    M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T, M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA, M_V_QUANTMODE_P_PRECISION_MODE_I_BALANCE,
                    M_FIAFLAG_P_MMTYPETMP_I_MODEVAL, P_CVDIFF_BASE_FLAG, P_CVDIFF_MLA_FLAG, P_TEMPLATE_VERSION, TEMPLATE_MODE>
                                  (query, key, value, pse_shift, attenMask, actualSeqLengths,
                                  actualSeqLengthsKV, deq_scale1, quant_scale1, deq_scale2, quant_scale2,
                                  quant_offset2, antiquantScale, antiquantOffset, blocktable, queryPaddingSize, kvPaddingSize,
                                  keyAntiquantScale, keyAntiquantOffset, valueAntiquantScale, valueAntiquantOffset,
                                  keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen, queryRope, keyRope, keyRopeAntiquantScale,
                                  attentionOut, softmaxLse, workspace, tiling);
  } else if (TEMPLATE_MODE == 1){
    incre_flash_attention_FIAS<Q_T, KV_T, OUT_T, PAGE_ATTENTIOND, LAYOUT_T, KV_LAYOUT_T, FLASH_DECODE, ENABLE_PREFIX, M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE,
                    M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T, M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA, M_V_QUANTMODE_P_PRECISION_MODE_I_BALANCE,
                    M_FIAFLAG_P_MMTYPETMP_I_MODEVAL, P_CVDIFF_BASE_FLAG, P_CVDIFF_MLA_FLAG, P_TEMPLATE_VERSION, TEMPLATE_MODE>
                                  (query, key, value, pse_shift, attenMask, actualSeqLengths,
                                  actualSeqLengthsKV, deq_scale1, quant_scale1, deq_scale2, quant_scale2,
                                  quant_offset2, antiquantScale, antiquantOffset, blocktable, queryPaddingSize, kvPaddingSize,
                                  keyAntiquantScale, keyAntiquantOffset, valueAntiquantScale, valueAntiquantOffset,
                                  keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen, queryRope, keyRope, keyRopeAntiquantScale, dequantScaleQuery,
                                  attentionOut, softmaxLse, workspace, tiling);
  }
}
