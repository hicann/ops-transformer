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
 * \file aclnn_quant_flash_attn.h
 * \brief QuantFlashAttn aclnn level-2 interface
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_QUANT_FLASH_ATTN_H_
#define OP_API_INC_LEVEL2_ACLNN_QUANT_FLASH_ATTN_H_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief First phase of aclnnQuantFlashAttn: calculate workspace size.
 * @domain aclnn_ops_train_infer
 *
 * @param q                   [IN]  query tensor. Dtype FP8_E4M3 (quant_mode=1).
 *                                  Layout determined by layoutQ: BSND/BNSD/TND.
 * @param k                   [IN]  key tensor. Same dtype as q. Layout determined by layoutKv.
 * @param v                   [IN]  value tensor. Same dtype as q. Layout determined by layoutKv.
 * @param qDescale            [IN]  query descale tensor. Dtype E8M0 (quant_mode=1).
 *                                  Shape depends on quantization granularity.
 * @param kDescale            [IN]  key descale tensor. Dtype E8M0 (quant_mode=1).
 * @param vDescale            [IN]  value descale tensor. Dtype E8M0 (quant_mode=1).
 * @param blockTableOptional  [IN]  optional. Block index table for paged attention, INT32.
 *                                  shape=(B, max_num_blocks_per_seq). Required when layout_kv is PA_*.
 * @param pScaleOptional      [IN]  optional. P scale tensor for per-token scaling.
 * @param cuSeqlensQOptional  [IN]  optional. Query cumulative sequence lengths, INT32.
 *                                  shape=(B+1,). Effective in TND variable-length scenario.
 * @param cuSeqlensKvOptional [IN]  optional. KV cumulative sequence lengths, INT32.
 *                                  shape=(B+1,).
 * @param sequsedQOptional    [IN]  optional. Actual query sequence lengths per batch, INT32.
 *                                  shape=(B,). Effective in padded batch mode.
 * @param sequsedKvOptional   [IN]  optional. Actual KV sequence lengths per batch, INT32.
 *                                  shape=(B,).
 * @param sinksOptional       [IN]  optional. Learnable sink attention weights, FLOAT32.
 * @param attnMaskOptional    [IN]  optional. Attention mask, INT8.
 * @param metadataOptional    [IN]  optional. Pre-computed tiling metadata, INT32.
 *                                  When non-null, tiling side skips split calculation.
 * @param quantMode           [IN]  ATTR. Quantization mode. INT.
 *                                  1: MXFP8 softmax FP32; 2: MXFP8 softmax BF16 (reserved).
 * @param softmaxScale        [IN]  ATTR. Softmax scaling factor. DOUBLE. 0.0 means 1/sqrt(D).
 * @param maskMode            [IN]  ATTR. Mask mode. INT.
 *                                  0: no mask; 1: causal; 2: anti-causal;
 *                                  3: prefix/band; 4: sliding window (uses winLeft/winRight).
 * @param winLeft             [IN]  ATTR. Left window size (maskMode=4). INT.
 * @param winRight            [IN]  ATTR. Right window size (maskMode=4). INT.
 * @param maxSeqlenQ          [IN]  ATTR. Max query sequence length. INT. -1 for auto.
 * @param maxSeqlenKV         [IN]  ATTR. Max KV sequence length. INT. -1 for auto.
 * @param layoutQ             [IN]  ATTR. Query layout: "BSND"/"BNSD"/"TND". STRING.
 * @param layoutQDescale      [IN]  ATTR. Q descale layout: "BSND"/"BNSD"/"TND"/"N2TGD". STRING.
 * @param layoutKv            [IN]  ATTR. KV layout: "BSND"/"TND"/"PA_ND"/"PA_NZ". STRING.
 * @param layoutOut           [IN]  ATTR. Output layout: "BSND"/"BNSD"/"TND". STRING.
 * @param returnSoftmaxLse    [IN]  ATTR. Whether to output softmax_lse. BOOL. True=output, False=no.
 *                                  Training forward pass set True, inference set False.
 * @param attnOut             [OUT] Required output. Attention result, BF16, layout by layoutOut.
 * @param softmaxLseOptional  [OUT] Optional output. Softmax log-sum-exp, FLOAT32.
 *                                  Valid when returnSoftmaxLse=True.
 * @param workspaceSize       [OUT] Workspace size in bytes.
 * @param executor            [OUT] Op executor handle for second phase.
 * @return aclnnStatus. ACLNN_SUCCESS on success.
 */
aclnnStatus aclnnQuantFlashAttnGetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *qDescale, const aclTensor *kDescale, const aclTensor *vDescale,
    const aclTensor *blockTableOptional,
    const aclTensor *pScaleOptional,
    const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensKvOptional,
    const aclTensor *sequsedQOptional, const aclTensor *sequsedKvOptional,
    const aclTensor *sinksOptional, const aclTensor *attnMaskOptional,
    const aclTensor *metadataOptional,
    int64_t quantMode,
    double softmaxScale, int64_t maskMode, int64_t winLeft, int64_t winRight,
    int64_t maxSeqlenQ, int64_t maxSeqlenKV,
    const char *layoutQ, const char *layoutQDescale, const char *layoutKv, const char *layoutOut,
    bool returnSoftmaxLse,
    const aclTensor *attnOut, const aclTensor *softmaxLseOptional,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief Second phase of aclnnQuantFlashAttn: execute computation.
 * @param workspace       [IN] Workspace device memory pointer from first phase.
 * @param workspaceSize   [IN] Workspace size in bytes.
 * @param executor        [IN] Op executor handle from first phase.
 * @param stream          [IN] ACL stream for execution.
 * @return aclnnStatus.
 */
aclnnStatus aclnnQuantFlashAttn(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_QUANT_FLASH_ATTN_H_
