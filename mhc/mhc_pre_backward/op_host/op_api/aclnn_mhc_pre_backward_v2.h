/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_MHC_PRE_BACKWARD_V2_H
#define OP_API_INC_MHC_PRE_BACKWARD_V2_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnMhcPreBackwardV2GetWorkspaceSize 的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * 算子功能：MhcPre反向算子
 * @param [in] x: 输入tensor，shape为[B,S,N,D]或[T,N,D]，数据类型支持：BF16/FP16。
 * @param [in] phi: 输入tensor，shape为[n^2+2n, nD]，数据类型支持：FP32。
 * @param [in] alpha: 输入tensor，shape为[3]，数据类型支持：FP32。
 * @param [in] gradHIn: 输入梯度，shape为[B,S,D]或[T,D]，数据类型支持：BF16/FP16。
 * @param [in] gradHPost: 输入梯度，shape为[B,S,N]或[T,N]，数据类型支持：FP32。
 * @param [in] gradHRes: 输入梯度，shape为[B,S,N,N]或[T,N,N]，数据类型支持：FP32。
 * @param [in] invRms: 输入tensor，shape为[B,S]或[T]，数据类型：FP32。
 * @param [in] hMix: 输入tensor，shape为[B,S,N^2+2N]或[T,N^2+2N]，数据类型：FP32。
 * @param [in] hPre: 输入tensor，shape为[B,S,N]或[T,N]，数据类型：FP32。
 * @param [in] hPost: 输入tensor，shape为[B,S,N]或[T,N]，数据类型：FP32。
 * @param [in] gammaOptional: 可选输入tensor，shape为[N,D]，数据类型：FP32。
 * @param [in] gradXPostOptional: 可选输入tensor，shape为[B,S,N,D]或[T,N,D]，数据类型：BF16/FP16。
 * @param [in] hcEps: 可选属性，默认值为1e-6。
 * @param [in] opImplMode: Cube计算模式，0表示使用FP32模式，1表示使用HF32模式。
 * @param [out] gradX: 输出梯度，数据类型支持：BF16/FP16。
 * @param [out] gradPhi: 输出梯度，数据类型支持：FP32。
 * @param [out] gradAlpha: 输出梯度，数据类型支持：FP32。
 * @param [out] gradBias: 输出梯度，数据类型支持：FP32。
 * @param [out] gradGamma: 可选输出梯度，数据类型支持：FP32。
 * @param [out] workspaceSize: 返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnMhcPreBackwardV2GetWorkspaceSize(
    const aclTensor *x, const aclTensor *phi, const aclTensor *alpha, const aclTensor *gradHIn,
    const aclTensor *gradHPost, const aclTensor *gradHRes, const aclTensor *invRms, const aclTensor *hMix,
    const aclTensor *hPre, const aclTensor *hPost, const aclTensor *gammaOptional, const aclTensor *gradXPostOptional,
    float hcEps, int64_t opImplMode, const aclTensor *gradX, const aclTensor *gradPhi, const aclTensor *gradAlpha,
    const aclTensor *gradBias, const aclTensor *gradGamma, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnMhcPreBackwardV2 的第二段接口。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口
 * aclnnMhcPreBackwardV2GetWorkspaceSize 获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnMhcPreBackwardV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                            aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_MHC_PRE_BACKWARD_V2_H
