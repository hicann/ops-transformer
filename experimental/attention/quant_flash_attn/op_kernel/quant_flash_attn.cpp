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
 * \file quant_flash_attn.cpp
 * \brief
 */

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif

#if (ORIG_DTYPE_Q == DT_FLOAT4_E2M1) && (ORIG_DTYPE_Q_DESCALE == DT_FLOAT8_E8M0) && (ORIG_DTYPE_ATTN_OUT == DT_BF16)
// ===================== MxFP4 DN (quant_mode=5, A4C4_QKV_MXFP4_P_MXFP4_SOFTMAX_FP16) =====================
#include "arch35/quant_flash_attn_tiling_data.h"
#include "arch35/quant_flash_attn_template_tiling_key.h"
#include "arch35/quant_flash_attn_kernel_dn.h"
#include "arch35/quant_flash_attn_block_cube_dn.h"
#include "arch35/quant_flash_attn_block_vector_dn.h"

using namespace optiling;
using namespace AscendC;
using namespace QFA_KERNEL;

template <uint8_t Q_OUT_LAYOUT_T>
__aicore__ inline constexpr QFA_LAYOUT GetQueryLayout()
{
    static_assert((Q_OUT_LAYOUT_T == LAYOUT_ENUM_BSND) || (Q_OUT_LAYOUT_T == LAYOUT_ENUM_BNSD) ||
                      (Q_OUT_LAYOUT_T == LAYOUT_ENUM_BNSD_BSND) || (Q_OUT_LAYOUT_T == LAYOUT_ENUM_TND),
                  "Get Query Layout fail, Q_OUT_LAYOUT_T is incorrect");
    if constexpr (Q_OUT_LAYOUT_T == LAYOUT_ENUM_BSND) {
        return QFA_LAYOUT::BSND;
    } else if constexpr (Q_OUT_LAYOUT_T == LAYOUT_ENUM_BNSD || Q_OUT_LAYOUT_T == LAYOUT_ENUM_BNSD_BSND) {
        return QFA_LAYOUT::BNSD;
    } else if constexpr (Q_OUT_LAYOUT_T == LAYOUT_ENUM_TND) {
        return QFA_LAYOUT::TND;
    }
}

template <uint8_t Q_OUT_LAYOUT_T>
__aicore__ inline constexpr QFA_LAYOUT GetOutLayout()
{
    static_assert((Q_OUT_LAYOUT_T == LAYOUT_ENUM_BSND) || (Q_OUT_LAYOUT_T == LAYOUT_ENUM_BNSD) ||
                      (Q_OUT_LAYOUT_T == LAYOUT_ENUM_BNSD_BSND) || (Q_OUT_LAYOUT_T == LAYOUT_ENUM_TND),
                  "Get AttnOut Layout fail, Q_OUT_LAYOUT_T is incorrect");
    if constexpr (Q_OUT_LAYOUT_T == LAYOUT_ENUM_BSND || Q_OUT_LAYOUT_T == LAYOUT_ENUM_BNSD_BSND) {
        return QFA_LAYOUT::BSND;
    } else if constexpr (Q_OUT_LAYOUT_T == LAYOUT_ENUM_BNSD) {
        return QFA_LAYOUT::BNSD;
    } else if constexpr (Q_OUT_LAYOUT_T == LAYOUT_ENUM_TND) {
        return QFA_LAYOUT::TND;
    }
}

template <uint8_t Q_OUT_LAYOUT_T, uint8_t KV_STORAGE_MODE>
__aicore__ inline constexpr QFA_LAYOUT GetKvLayout()
{
    static_assert((KV_STORAGE_MODE == KV_STORAGE_MODE_CONTINUE) || (KV_STORAGE_MODE == KV_STORAGE_MODE_PA_BSND) ||
                      (KV_STORAGE_MODE == KV_STORAGE_MODE_PA_BNSD),
                  "Get Key/Value Layout fail, KV_STORAGE_MODE is incorrect");
    if constexpr (KV_STORAGE_MODE == KV_STORAGE_MODE_CONTINUE) {
        return GetQueryLayout<Q_OUT_LAYOUT_T>();
    } else if constexpr (KV_STORAGE_MODE == KV_STORAGE_MODE_PA_BSND) {
        return QFA_LAYOUT::BSND; // block内的格式类似于BSND
    } else if constexpr (KV_STORAGE_MODE == KV_STORAGE_MODE_PA_BNSD) {
        return QFA_LAYOUT::BNSD; // block内的格式类似于BNSD
    }
}

template <uint8_t KV_STORAGE_MODE>
__aicore__ inline constexpr bool IsPageAttention()
{
    static_assert((KV_STORAGE_MODE == KV_STORAGE_MODE_CONTINUE) || (KV_STORAGE_MODE == KV_STORAGE_MODE_PA_BSND) ||
                      (KV_STORAGE_MODE == KV_STORAGE_MODE_PA_BNSD),
                  "Get PAGE_ATTENTION flag fail, KV_STORAGE_MODE is incorrect");
    return (KV_STORAGE_MODE != KV_STORAGE_MODE_CONTINUE);
}

template <uint8_t Q_OUT_LAYOUT_T, uint16_t CONFIG_T, uint8_t QUANT_MODE_T, bool HAS_MASK, uint8_t KV_STORAGE_MODE,
          bool IS_FD_T>
__global__ __aicore__ void quant_flash_attn(
    __gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value, __gm__ uint8_t* q_descale,
    __gm__ uint8_t* k_descale, __gm__ uint8_t* v_descale, __gm__ uint8_t* blockTable, __gm__ uint8_t* pScale,
    __gm__ uint8_t* cuSeqlensQ, __gm__ uint8_t* cuSeqlensKv, __gm__ uint8_t* sequsedQ, __gm__ uint8_t* sequsedKv,
    __gm__ uint8_t* learnableSink, __gm__ uint8_t* attnMask, __gm__ uint8_t* metadata, __gm__ uint8_t* attnOut,
    __gm__ uint8_t* softmaxLse, __gm__ uint8_t* workspace, __gm__ uint8_t* tiling)
{
    REGISTER_TILING_DEFAULT(QuantFlashAttnTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    (void)pScale;

#if (ORIG_DTYPE_Q == DT_FLOAT4_E2M1) && (ORIG_DTYPE_Q_DESCALE == DT_FLOAT8_E8M0) && (ORIG_DTYPE_ATTN_OUT == DT_BF16)
    if constexpr (QUANT_MODE_T == QFA_MXFP4_DN) {
        InitSocState();
        GET_TILING_DATA_MEMBER(QuantFlashAttnTilingData, baseTiling, baseTilingIn, tiling);
        const FlashAttnTilingData* __restrict tilingData = &baseTilingIn;

        using QFA_T = QFAType<fp4x2_e2m1_t, fp8_e8m0_t, bfloat16_t, IsPageAttention<KV_STORAGE_MODE>(),
                              GetQueryLayout<Q_OUT_LAYOUT_T>(), GetKvLayout<Q_OUT_LAYOUT_T, KV_STORAGE_MODE>(),
                              GetOutLayout<Q_OUT_LAYOUT_T>(), HAS_MASK>;
        using CubBlock = QuantFlashAttnBlockCubeDn<QFA_T>;
        using VectorBlock = QuantFlashAttnBlockVectorDn<QFA_T>;
        QuantFlashAttnKernelDn<QFA_T, CubBlock, VectorBlock> op;
        op.Init(query, key, value, q_descale, k_descale, v_descale, blockTable, cuSeqlensQ, cuSeqlensKv, sequsedQ,
                sequsedKv, attnMask, learnableSink, softmaxLse, attnOut, workspace, metadata, tilingData);
        op.Process();
        PipeBarrier<PIPE_ALL>();
    }
#endif
}
#elif (ORIG_DTYPE_Q == DT_FLOAT8_E4M3FN)
// ===================== MxFP8 (quant_mode=3, softmax FP16) =====================
#include "kernel_operator.h"

/* ===== 平台相关部分：按 arch 整体分流 ===== */
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201)

/* --- arch92 (v121 porting) --- */
#include "arch92/quant_flash_attn_common_def.h"
#include "arch92/quant_flash_attn_tiling_data.h"
#include "arch92/quant_flash_attn_template_tiling_key.h"
#include "arch92/quant_flash_attn_kernel_mxfp8_softmax_fp16.h"
#include "util.h"

using namespace AscendC;
using namespace optiling;

template <uint8_t inOutLayoutType, uint16_t config, uint8_t quantMode, bool hasAttenMask, uint8_t KvLayoutType,
          bool isFd>
__aicore__ inline void quant_flash_attn_mxfp8_softmax_fp16(
    __gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value, __gm__ uint8_t* dequantScaleQuery,
    __gm__ uint8_t* dequantScaleKey, __gm__ uint8_t* dequantScaleValue, __gm__ uint8_t* blockTable,
    __gm__ uint8_t* pScale, __gm__ uint8_t* cuSeqLensQ, __gm__ uint8_t* cuSeqLensKv, __gm__ uint8_t* sequsedQ,
    __gm__ uint8_t* sequsedKv, __gm__ uint8_t* sinks, __gm__ uint8_t* attnMask, __gm__ uint8_t* metadata,
    __gm__ uint8_t* attnOut, __gm__ uint8_t* softmaxLse, __gm__ uint8_t* workspace, __gm__ uint8_t* tiling)
{
    using INPUT_T = fp8_e4m3fn_t;
    using OUT_T = bfloat16_t;

    fa_base_matmul::ResetIdCounter();

    constexpr LayOutTypeEnum inputLayoutType = static_cast<LayOutTypeEnum>(InOutLayoutTypeValue[inOutLayoutType][0]);
    constexpr LayOutTypeEnum outputLayoutType = static_cast<LayOutTypeEnum>(InOutLayoutTypeValue[inOutLayoutType][1]);

    constexpr S1TemplateType s1TemplateType = static_cast<S1TemplateType>(ConfigValue[config].s1);
    constexpr S2TemplateType s2TemplateType = static_cast<S2TemplateType>(ConfigValue[config].s2);
    constexpr DTemplateType dTemplateType = static_cast<DTemplateType>(
        (static_cast<std::underlying_type_t<inferDTemplateType> >(ConfigValue[config].d) + 63) >> 6 << 6);
    constexpr DTemplateType dVTemplateType = static_cast<DTemplateType>(
        (static_cast<std::underlying_type_t<inferDTemplateType> >(ConfigValue[config].dv) + 63) >> 6 << 6);
    constexpr bool isDAligned = ConfigValue[config].d != inferDTemplateType::NotAligned &&
                                static_cast<int32_t>(ConfigValue[config].d) % 64 == 0;

    constexpr bool isFdConst = false;
    constexpr bool useDn = true;

    using CubeBlock =
        BaseApi::QuantFlashAttnBlockCubeMxfp8SoftmaxFp16<INPUT_T, float, inputLayoutType, s1TemplateType,
                                                         s2TemplateType, dTemplateType, dVTemplateType, KvLayoutType>;
    using VecFaBlock =
        BaseApi::QuantFlashAttnBlockVecMxfp8SoftmaxFp16<INPUT_T, float, OUT_T, inputLayoutType, outputLayoutType,
                                                        s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
                                                        hasAttenMask, KvLayoutType, isFdConst, useDn, isDAligned>;

    using CubeBlockDummy = BaseApi::QuantFlashAttnBlockCubeMxfp8SoftmaxFp16Dummy<
        INPUT_T, float, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType, KvLayoutType>;
    using VecFaBlockDummy = BaseApi::QuantFlashAttnBlockVecMxfp8SoftmaxFp16Dummy<
        INPUT_T, float, OUT_T, inputLayoutType, outputLayoutType, s1TemplateType, s2TemplateType, dTemplateType,
        dVTemplateType, hasAttenMask, KvLayoutType, isFdConst, useDn>;

#ifdef __DAV_CUBE__
    using Kernel = BaseApi::QuantFlashAttnKernelMxfp8SoftmaxFp16<CubeBlock, VecFaBlockDummy>;
#else
    using Kernel = BaseApi::QuantFlashAttnKernelMxfp8SoftmaxFp16<CubeBlockDummy, VecFaBlock>;
#endif

    const __gm__ QuantFlashAttnTilingData* __restrict tilingData =
        (const __gm__ QuantFlashAttnTilingData* __restrict)tiling;

    TPipe tPipe;
    Kernel op;
    op.Init(query, key, value, sinks, attnMask, cuSeqLensQ, cuSeqLensKv, blockTable, dequantScaleQuery, dequantScaleKey,
            dequantScaleValue, pScale, softmaxLse, attnOut, workspace, metadata, sequsedQ, sequsedKv, tilingData,
            &tPipe);
    op.Process();
}

template <uint8_t inOutLayoutType, uint16_t config, uint8_t quantMode, bool hasAttenMask, uint8_t KvLayoutType,
          bool isFd>
__global__ __aicore__ void quant_flash_attn(
    __gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value, __gm__ uint8_t* dequantScaleQuery,
    __gm__ uint8_t* dequantScaleKey, __gm__ uint8_t* dequantScaleValue, __gm__ uint8_t* blockTable,
    __gm__ uint8_t* pScale, __gm__ uint8_t* cuSeqLensQ, __gm__ uint8_t* cuSeqLensKv, __gm__ uint8_t* sequsedQ,
    __gm__ uint8_t* sequsedKv, __gm__ uint8_t* sinks, __gm__ uint8_t* attnMask, __gm__ uint8_t* metadata,
    __gm__ uint8_t* attnOut, __gm__ uint8_t* softmaxLse, __gm__ uint8_t* workspace, __gm__ uint8_t* tiling)
{
    REGISTER_TILING_DEFAULT(QuantFlashAttnTilingData);
    __gm__ uint8_t* user = GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);

#if (ORIG_DTYPE_Q == DT_FLOAT8_E4M3FN)
    if constexpr (quantMode == QFA_MXFP8_SOFTMAX_FP16) {
        quant_flash_attn_mxfp8_softmax_fp16<inOutLayoutType, config, quantMode, hasAttenMask, KvLayoutType, isFd>(
            query, key, value, dequantScaleQuery, dequantScaleKey, dequantScaleValue, blockTable, pScale, cuSeqLensQ,
            cuSeqLensKv, sequsedQ, sequsedKv, sinks, attnMask, metadata, attnOut, softmaxLse, workspace, tiling);
        return;
    }
#endif
}

#else /* arch35 */

/* --- arch35 --- */
#include "arch35/quant_flash_attn_tiling_data.h"
#include "arch35/quant_flash_attn_template_tiling_key.h"
#include "arch35/quant_flash_attn_kernel_mxfp8_softmax_fp16.h"

using namespace optiling;

// ─────────────────────────────────────────────────────────
// quant_flash_attn_mxfp8_softmax_fp16: MxFP8 Softmax FP16（S1/S2=256，V0/V1 切 S1 半）
// ─────────────────────────────────────────────────────────
template <uint8_t inOutLayoutType, uint16_t config, uint8_t quantMode, bool hasAttenMask, uint8_t KvLayoutType,
          bool isFd>
__aicore__ inline void quant_flash_attn_mxfp8_softmax_fp16(
    __gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value, __gm__ uint8_t* dequantScaleQuery,
    __gm__ uint8_t* dequantScaleKey, __gm__ uint8_t* dequantScaleValue, __gm__ uint8_t* blockTable,
    __gm__ uint8_t* pScale, __gm__ uint8_t* cuSeqLensQ, __gm__ uint8_t* cuSeqLensKv, __gm__ uint8_t* sequsedQ,
    __gm__ uint8_t* sequsedKv, __gm__ uint8_t* sinks, __gm__ uint8_t* attnMask, __gm__ uint8_t* metadata,
    __gm__ uint8_t* attnOut, __gm__ uint8_t* softmaxLse, __gm__ uint8_t* workspace, __gm__ uint8_t* tiling)
{
    InitSocState();
    const __gm__ QuantFlashAttnTilingData* __restrict tilingData =
        (const __gm__ QuantFlashAttnTilingData* __restrict)tiling;
    using CubeBlock = QFA_KERNEL::QuantFlashAttnBlockCubeMxfp8SoftmaxFp16<QFA_KERNEL::QUANT_T, QFA_KERNEL::SCALE_T>;
    using VectorBlock = QFA_KERNEL::QuantFlashAttnBlockVectorMxfp8SoftmaxFp16<QFA_KERNEL::QUANT_T, QFA_KERNEL::SCALE_T,
                                                                              QFA_KERNEL::OUT_T>;
    QFA_KERNEL::QuantFlashAttnKernelMxfp8SoftmaxFp16<CubeBlock, VectorBlock> op;
    op.Init(query, key, value, dequantScaleQuery, dequantScaleKey, dequantScaleValue, blockTable, pScale, cuSeqLensQ,
            cuSeqLensKv, sequsedQ, sequsedKv, sinks, attnMask, metadata, attnOut, softmaxLse, workspace, tilingData);
    op.Process();
    PipeBarrier<PIPE_ALL>();
}

template <uint8_t inOutLayoutType, uint16_t config, uint8_t quantMode, bool hasAttenMask, uint8_t KvLayoutType,
          bool isFd>
__global__ __aicore__ void quant_flash_attn(
    __gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value, __gm__ uint8_t* dequantScaleQuery,
    __gm__ uint8_t* dequantScaleKey, __gm__ uint8_t* dequantScaleValue, __gm__ uint8_t* blockTable,
    __gm__ uint8_t* pScale, __gm__ uint8_t* cuSeqLensQ, __gm__ uint8_t* cuSeqLensKv, __gm__ uint8_t* sequsedQ,
    __gm__ uint8_t* sequsedKv, __gm__ uint8_t* sinks, __gm__ uint8_t* attnMask, __gm__ uint8_t* metadata,
    __gm__ uint8_t* attnOut, __gm__ uint8_t* softmaxLse, __gm__ uint8_t* workspace, __gm__ uint8_t* tiling)
{
    REGISTER_TILING_DEFAULT(QuantFlashAttnTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

#if (ORIG_DTYPE_Q == DT_FLOAT8_E4M3FN)
    if constexpr (quantMode == QFA_MXFP8_SOFTMAX_FP16) {
        quant_flash_attn_mxfp8_softmax_fp16<inOutLayoutType, config, quantMode, hasAttenMask, KvLayoutType, isFd>(
            query, key, value, dequantScaleQuery, dequantScaleKey, dequantScaleValue, blockTable, pScale, cuSeqLensQ,
            cuSeqLensKv, sequsedQ, sequsedKv, sinks, attnMask, metadata, attnOut, softmaxLse, workspace, tiling);
    }
#endif
}

#endif /* arch35 vs arch92 */
#endif
