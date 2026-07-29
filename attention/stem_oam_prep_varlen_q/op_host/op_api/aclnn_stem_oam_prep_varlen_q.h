/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_STEM_OAM_PREP_VARLEN_Q_H_
#define ACLNN_STEM_OAM_PREP_VARLEN_Q_H_
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnStemOamPrepVarlenQ的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * @param q              [IN]  变长Q tensor，所有batch的token拼接存储。
 *                             数据类型FLOAT8_E4M3FN，ND格式，shape=(totalTokens, numHeadQ, dimQk)。
 * @param qSeqLens       [IN]  每个batch的Q序列长度。INT32数组，长度等于batch。
 * @param cuSeqLensQ     [IN]  Q的累积序列长度偏移量。INT32数组，长度等于batch+1，首元素必须为0。
 * @param qScale         [IN]  可选。Q的per-token scale factor。FLOAT，ND格式，shape=(totalTokens, numHeadQ)。
 *                             FP8输入时必填；BF16/FP16输入时传nullptr。
 * @param stemBlockSize  [IN]  stem block大小，默认128。必须是32的倍数，最大256。
 * @param stemStride     [IN]  stem stride大小，默认16。必须是16的倍数，最大64，且<=stemBlockSize。
 * @param qFlat          [OUT] flattened Q输出。BF16，ND格式，shape=(batch, numHeadQ, maxQb, stemStride*dimQk)。
 * @param workspaceSize  [OUT] workspace大小（字节数）。
 * @param executor       [OUT] op执行器句柄，供第二段接口使用。
 * @return aclnnStatus 执行状态。ACLNN_SUCCESS表示成功。
 */
aclnnStatus aclnnStemOamPrepVarlenQGetWorkspaceSize(const aclTensor *q, const aclIntArray *qSeqLens,
                                                    const aclIntArray *cuSeqLensQ, const aclTensor *qScale,
                                                    int64_t stemBlockSize, int64_t stemStride, aclTensor *qFlat,
                                                    uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnStemOamPrepVarlenQ的第二段接口，用于执行计算。
 * @param workspace       [IN] 由第一段接口计算得到的workspace设备内存指针。
 * @param workspaceSize   [IN] workspace大小（字节数）。
 * @param executor        [IN] 第一段接口输出的op执行器句柄。
 * @param stream          [IN] 用于执行计算的acl stream。
 * @return aclnnStatus 执行状态。
 */
aclnnStatus aclnnStemOamPrepVarlenQ(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
