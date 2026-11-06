/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_ACLNN_TEXT_GATED_RESIDUAL_H
#define OP_API_INC_ACLNN_TEXT_GATED_RESIDUAL_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 计算TextGatedResidual所需的工作空间大小。
 *
 * @param x [输入] 形状为[B, S, D]的ND张量，FLOAT16/BFLOAT16，D=C*H。
 * @param wDown [输入] 形状为[R, D]的ND张量，dtype须与x一致。
 * @param wUp [输入] 形状为[D, R]的ND张量，dtype须与x一致。
 * @param wGateOptional [输入] 可选的[C, D] ND张量，dtype须与x一致；为空时不计算inject。
 * @param gammaOptional [输入] 可选的[D] FLOAT32 ND张量；内部使用1+gamma缩放，为空时缩放因子为1。
 * @param channels [输入] 通道数C，须大于0且D可被C整除。
 * @param eps [输入] RMS归一化数值稳定性参数，须为非负有限DOUBLE且可表示为FLOAT，通常取1e-6。
 * @param needBackward [输入] 是否保存反向缓存；为true时invRmsOptional、downPreActOptional和upPreActOptional不能为空。
 * @param mixed [输出] 形状为[B, S, H]的ND张量，dtype须与x一致。
 * @param injectOptional [输出] 形状为[B, S, C]的ND张量，dtype须与x一致；wGateOptional非空时必须提供。
 * @param invRmsOptional [输出] 形状为[B, S, C]的FLOAT32 ND张量；needBackward为false时可传入nullptr。
 * @param downPreActOptional [输出] 形状为[B, S,
 * R]的ND张量，dtype须与x一致，保存未除以C的down投影；needBackward为false时可传入nullptr。
 * @param upPreActOptional [输出] 形状为[B, S,
 * D]的ND张量，dtype须与x一致，保存sigmoid之前的up投影；needBackward为false时可传入nullptr。
 * @param workspaceSize [输出] 设备侧所需的工作空间大小，单位为字节。
 * @param executor [输出] 第二段接口使用的执行器。
 * @return 成功时返回ACLNN_SUCCESS。
 */
ACLNN_API aclnnStatus aclnnTextGatedResidualGetWorkspaceSize(
    const aclTensor* x, const aclTensor* wDown, const aclTensor* wUp, const aclTensor* wGateOptional,
    const aclTensor* gammaOptional, int64_t channels, double eps, bool needBackward, const aclTensor* mixed,
    const aclTensor* injectOptional, const aclTensor* invRmsOptional, const aclTensor* downPreActOptional,
    const aclTensor* upPreActOptional, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief 使用第一段接口生成的执行器运行TextGatedResidual。
 *
 * @param workspace [输入] Device侧workspace地址；workspaceSize为0时可传入nullptr。
 * @param workspaceSize [输入] 第一段接口返回的workspace大小，单位为字节。
 * @param executor [输入] 第一段接口返回的op执行器。
 * @param stream [输入] AscendCL Stream。
 * @return ACLNN_SUCCESS或对应错误码。
 */
ACLNN_API aclnnStatus aclnnTextGatedResidual(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                             aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
