/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_MHC_PRE_SINKHORN_PREMIX_BACKWARD_H
#define OP_API_INC_MHC_PRE_SINKHORN_PREMIX_BACKWARD_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnMhcPreSinkhornPremixBackwardGetWorkspaceSize 的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * 算子功能：MhcPreSinkhornPremix反向算子
 * @param [in] gradHin: 输入梯度，shape为[B,S,C]，数据类型支持：BF16/FP16。
 * @param [in] gradHPost: 输入梯度，shape为[B,S,N]，数据类型支持：FP32。
 * @param [in] gradHRes: 输入梯度，shape为[B,S,N,N]或[B,S,N*N]，数据类型支持：FP32。
 * @param [in] x: 输入tensor，shape为[B,S,N,C]，数据类型支持：BF16/FP16。
 * @param [in] phi: 输入tensor，shape为[2N+N^2, N*C]，数据类型支持：FP32。
 * @param [in] alpha: 输入tensor，shape为[3]，数据类型支持：FP32。
 * @param [in] bias: 输入tensor，shape为[2N+N^2]，数据类型支持：FP32。
 * @param [in] hPre: 输入tensor，shape为[B,S,N]，数据类型：FP32。
 * @param [in] hcBeforeNorm: 输入tensor，shape为[B,S,2N+N^2]，数据类型：FP32。
 * @param [in] invRms: 输入tensor，shape为[B,S,1]，数据类型：FP32。
 * @param [in] sumOut: 输入tensor，shape为[2*sk_iter_count,B,S,N]，数据类型：FP32。
 * @param [in] normOut: 输入tensor，shape为[2*sk_iter_count,B,S,N,N]，数据类型：FP32。
 * @param [in] preMix: 可选输入tensor，shape为[B,S,N]，数据类型：FP32。传入时y折叠项使用pre_mix加权，
 *                     并输出gradPreMix；不传可为nullptr。
 * @param [in] gradPre: 可选输入tensor，shape为[B,S,N]，数据类型：FP32。下游回传的pre梯度，
 *                      合并进d_pre；不传可为nullptr。注意：preMix与gradPre必须同时传入或同时
 *                      缺省，只传其一将报参数校验错误。
 * @param [in] hcEps: 可选属性，默认值为1e-6。
 * @param [out] gradX: 输出梯度，数据类型支持：BF16/FP16。
 * @param [out] gradPhi: 输出梯度，数据类型支持：FP32。
 * @param [out] gradAlpha: 输出梯度，数据类型支持：FP32。
 * @param [out] gradBias: 输出梯度，数据类型支持：FP32。
 * @param [out] gradPreMix: 可选输出梯度，shape为[B,S,N]，数据类型：FP32。仅传入preMix时有效，
 *                          否则须为nullptr。
 * @param [out] workspaceSize: 返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
aclnnStatus aclnnMhcPreSinkhornPremixBackwardGetWorkspaceSize(
    const aclTensor *gradHin, const aclTensor *gradHPost, const aclTensor *gradHRes, const aclTensor *x,
    const aclTensor *phi, const aclTensor *alpha, const aclTensor *bias, const aclTensor *hPre,
    const aclTensor *hcBeforeNorm, const aclTensor *invRms, const aclTensor *sumOut, const aclTensor *normOut,
    const aclTensor *preMix, const aclTensor *gradPre, double hcEps, const aclTensor *gradX, const aclTensor *gradPhi,
    const aclTensor *gradAlpha, const aclTensor *gradBias, const aclTensor *gradPreMix, uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief aclnnMhcPreSinkhornPremixBackward 的第二段接口。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口
 * aclnnMhcPreSinkhornPremixBackwardGetWorkspaceSize 获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
aclnnStatus aclnnMhcPreSinkhornPremixBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                              aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_MHC_PRE_SINKHORN_PREMIX_BACKWARD_H
