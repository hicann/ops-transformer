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
 * \file stem_indexer.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#if (defined(__CCE_AICORE__) && (__CCE_AICORE__ == 310)) || defined(__DAV_310R6__)
#define SI_ENABLE_ARCH35 1
#include "arch35/stem_indexer_kernel.h"
#else
#define SI_ENABLE_ARCH35 0
#endif
#include "stem_indexer_template_tiling_key.h"

#if SI_ENABLE_ARCH35
#define INVOKE_LI_NO_KFC_OP_IMPL(templateClass, ...)                                                                   \
    do {                                                                                                               \
        templateClass<SICommon::SIType<__VA_ARGS__>> op;                                                               \
        GET_TILING_DATA_WITH_STRUCT(StemIndexerTilingData, tiling_data_in, tiling);                                    \
        const StemIndexerTilingData *__restrict tiling_data = &tiling_data_in;                                         \
        op.Init(qflat, kflat, vbias, qSeqLens, kvSeqLens, numPromptTokens, metadata, sparseIndices, sparseSeqLen,      \
                user, tiling_data, &tPipe);                                                                            \
        op.Process();                                                                                                  \
    } while (0)
#endif

template <int DT_Q, int DT_K, int DT_OUT, int CAUSAL, int TOPK_SCORE_PRECISION>
__global__ __aicore__ void stem_indexer(__gm__ uint8_t *qflat, __gm__ uint8_t *kflat, __gm__ uint8_t *vbias,
                                        __gm__ uint8_t *qSeqLens, __gm__ uint8_t *kvSeqLens,
                                        __gm__ uint8_t *numPromptTokens, __gm__ uint8_t *metadata,
                                        __gm__ uint8_t *sparseIndices, __gm__ uint8_t *sparseSeqLen,
                                        __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
#if SI_ENABLE_ARCH35
    AscendC::TPipe tPipe;
    __gm__ uint8_t *user = AscendC::GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    INVOKE_LI_NO_KFC_OP_IMPL(SIKernel::SIPreload, bfloat16_t, bfloat16_t, int32_t, CAUSAL, TOPK_SCORE_PRECISION);
#else
    (void)qflat;
    (void)kflat;
    (void)vbias;
    (void)qSeqLens;
    (void)kvSeqLens;
    (void)numPromptTokens;
    (void)metadata;
    (void)sparseIndices;
    (void)sparseSeqLen;
    (void)workspace;
    (void)tiling;
#endif
}
