/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_QUANT_FLASH_ATTN_H_
#define OP_API_INC_LEVEL2_ACLNN_QUANT_FLASH_ATTN_H_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnQuantFlashAttn的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train_infer
 *
 * @param q                   [IN]  query tensor。数据类型FLOAT16/BFLOAT16/FP8_E4M3(quant_mode=1)，ND格式。
 *                                  layout由layoutQ决定，支持BSND/BNSD/TND。
 * @param k                   [IN]  key tensor。数据类型与q一致，layout由layoutKv决定。
 * @param v                   [IN]  value tensor。数据类型与q一致，layout由layoutKv决定，shape同k。
 * @param qDescale            [IN]  query descale tensor。数据类型E8M0(quant_mode=1)。
 * @param kDescale            [IN]  key descale tensor。数据类型E8M0(quant_mode=1)。
 * @param vDescale            [IN]  value descale tensor。数据类型E8M0(quant_mode=1)。
 * @param blockTableOptional  [IN]  可选。分页KV缓存的块映射表，数据类型INT32。
 * @param pScaleOptional      [IN]  可选。P scale tensor，用于per-token缩放。
 * @param cuSeqlensQOptional  [IN]  可选。query的累积序列长度（前缀和），数据类型INT32。
 * @param cuSeqlensKvOptional [IN]  可选。kv的累积序列长度（前缀和），数据类型INT32。
 * @param sequsedQOptional    [IN]  可选。各batch中query的实际序列长度，数据类型INT32。
 * @param sequsedKvOptional   [IN]  可选。各batch中kv的实际序列长度，数据类型INT32。
 * @param sinksOptional       [IN]  可选。可学习的sink注意力权重，数据类型FLOAT32。
 * @param attnMaskOptional    [IN]  可选。attnmask的参数，数据类型INT8。
 * @param metadataOptional    [IN]  可选。预计算的tiling切分方案，数据类型INT32，由上游算子传入。
 * @param quantMode           [IN]  ATTR。量化模式。INT。1:MXFP8 softmax FP32; 2:MXFP8 softmax BF16(预留)。
 * @param softmaxScale        [IN]  ATTR可选。softmax缩放系数。DOUBLE。默认值0.0表示使用1/sqrt(D)。
 * @param maskMode            [IN]  ATTR可选。掩码模式。INT。
 *                                  0: 不使用掩码；1: 因果掩码；2: 非因果掩码；
 *                                  3: prefix/band掩码；4: 滑动窗口掩码（使用winLeft/winRight）。
 * @param winLeft             [IN]  ATTR可选。左侧注意力窗口大小（maskMode=4时）。INT。
 * @param winRight            [IN]  ATTR可选。右侧注意力窗口大小（maskMode=4时）。INT。
 * @param maxSeqlenQ          [IN]  ATTR可选。query最大序列长度。INT。-1表示自动推断。
 * @param maxSeqlenKV         [IN]  ATTR可选。kv最大序列长度。INT。-1表示自动推断。
 * @param layoutQ             [IN]  ATTR可选。query的数据布局，支持"BSND"/"BNSD"/"TND"。
 * @param layoutQDescale      [IN]  ATTR可选。q
 * descale的数据布局，支持"BSND"/"BNSD"/"TND"/"N2TGD"。mxfp4不使用，透传保持接口一致。
 * @param layoutKv            [IN]  ATTR可选。key/value的数据布局，支持"BSND"/"TND"/"PA_ND"/"PA_NZ"。
 * @param layoutOut           [IN]  ATTR可选。输出的数据布局，支持"BSND"/"BNSD"/"TND"。
 * @param returnSoftmaxLse    [IN]  ATTR可选。是否输出softmax_lse。BOOL。True输出，False不输出。
 *                                  训练正向传播时置True，推理时置False。
 * @param attnOut             [OUT] 必选输出。attention计算结果，数据类型BF16，layout由layoutOut决定。
 * @param softmaxLseOptional  [OUT] 可选输出。softmax的log-sum-exp值，FLOAT32类型。
 *                                  returnSoftmaxLse=True时有效。
 * @param workspaceSize       [OUT] workspace大小（字节数）。
 * @param executor            [OUT] op执行器句柄，供第二段接口使用。
 * @return aclnnStatus 执行状态。ACLNN_SUCCESS表示成功。
 */
aclnnStatus aclnnQuantFlashAttnGetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *qDescale, const aclTensor *kDescale,
    const aclTensor *vDescale, const aclTensor *blockTableOptional, const aclTensor *pScaleOptional,
    const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensKvOptional, const aclTensor *sequsedQOptional,
    const aclTensor *sequsedKvOptional, const aclTensor *sinksOptional, const aclTensor *attnMaskOptional,
    const aclTensor *metadataOptional, int64_t quantMode, double softmaxScale, int64_t maskMode, int64_t winLeft,
    int64_t winRight, int64_t maxSeqlenQ, int64_t maxSeqlenKV, const char *layoutQ, const char *layoutQDescale,
    const char *layoutKv, const char *layoutOut, bool returnSoftmaxLse, const aclTensor *attnOut,
    const aclTensor *softmaxLseOptional, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnQuantFlashAttn的第二段接口，用于执行计算。
 * @param workspace       [IN] 由第一段接口计算得到的workspace设备内存指针。
 * @param workspaceSize   [IN] workspace大小（字节数）。
 * @param executor        [IN] 第一段接口输出的op执行器句柄。
 * @param stream          [IN] 用于执行计算的acl stream。
 * @return aclnnStatus 执行状态。
 */
aclnnStatus aclnnQuantFlashAttn(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_QUANT_FLASH_ATTN_H_
