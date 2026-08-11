/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_MLA_PREPROCESS_V2_H
#define ACLNN_MLA_PREPROCESS_V2_H

#include "aclnn/acl_meta.h"
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 计算MlaPreprocessV2所需的workspace大小并创建算子执行器。
 *
 * @param input [IN] 用于计算Query和Key的隐藏状态Tensor。
 * @param gamma0 [IN] 第一次RMSNorm计算的gamma参数。
 * @param beta0 [IN] 第一次RMSNorm计算的beta参数。
 * @param quantScale0 [IN] 第一次RMSNormQuant计算的量化缩放参数。
 * @param quantOffset0 [IN] 第一次RMSNormQuant计算的量化偏移参数。
 * @param wdqkv [IN] 作用于input的降维权重。
 * @param descale0 [IN] wdqkv矩阵乘的反量化缩放参数。
 * @param bias0 [IN] wdqkv矩阵乘的偏置参数。
 * @param gamma1 [IN] 第二次RMSNorm计算的gamma参数。
 * @param beta1 [IN] 第二次RMSNorm计算的beta参数。
 * @param quantScale1 [IN] 第二次RMSNormQuant计算的量化缩放参数。
 * @param quantOffset1 [IN] 第二次RMSNormQuant计算的量化偏移参数。
 * @param wuq [IN] Query升维权重。
 * @param descale1 [IN] wuq矩阵乘的反量化缩放参数。
 * @param bias1 [IN] wuq矩阵乘的偏置参数。
 * @param gamma2 [IN] 对压缩后的Key/Value执行归一化时使用的gamma参数。
 * @param cos [IN] RoPE计算的余弦Tensor。与sin同时传入nullptr时关闭RoPE，仅一个为nullptr时输入非法。
 * @param sin [IN] RoPE计算的正弦Tensor。与cos同时传入nullptr时关闭RoPE，仅一个为nullptr时输入非法。
 * @param wuk [IN] 计算Q-nope输出时使用的Key升维权重。
 * @param kvCache [IN] 根据slotMapping更新的Key/Value Cache Tensor。
 * @param kvCacheRope [IN] RoPE Cache Tensor，是否需要传入及其布局由cacheMode决定。
 * @param slotMapping [IN] 每个输入Token对应的Cache槽位索引。
 * @param ctkvScale [IN] cacheMode为2时，压缩Key/Value Cache的量化缩放参数。
 * @param qNopeScale [IN] cacheMode为2时，Q-nope输出的量化缩放参数。
 * @param wdqDim [IN] 从wdqkv矩阵乘结果中拆分出的Query LoRA维度。
 * @param qRopeDim [IN] Query中参与RoPE计算的维度。
 * @param kRopeDim [IN] Key中参与RoPE计算的维度。
 * @param epsilon [IN] 为防止除零而加到RMSNorm分母上的常数。
 * @param qRotaryCoeff [IN] Query的旋转系数。
 * @param kRotaryCoeff [IN] Key的旋转系数。
 * @param transeposeWdq [IN] 表示wdqkv在矩阵乘计算中是否转置。
 * @param transeposeWuq [IN] 表示wuq在矩阵乘计算中是否转置。
 * @param transeposeWuk [IN] 表示wuk在矩阵乘计算中是否转置。
 * @param cacheMode [IN] Cache布局和输出模式，支持0、1、2、3。
 * @param quantMode [IN] RMSNorm量化模式。
 * @param doRmsNorm [IN] 表示第一次量化前是否执行RMSNorm。
 * @param wdkvSplitCount [IN] wdqkv降维权重的拆分段数。
 * @param qDownOutFlag [IN] 表示是否输出Query降维结果。
 * @param qOutOut [OUT] Query输出Tensor。
 * @param kvCacheOutOut [OUT] 更新后的Key/Value Cache输出Tensor。
 * @param qRopeOutOut [OUT] Cache拆分模式下的Query RoPE分量输出Tensor。
 * @param krCacheOutOut [OUT] Cache拆分模式下更新后的Key RoPE Cache输出Tensor。
 * @param qDownOutOut [OUT] Query降维结果输出Tensor。
 * @param workspaceSize [OUT] 返回Device侧所需的workspace大小，单位为字节。
 * @param executor [OUT] 返回包含算子执行流程的执行器。
 * @return aclnnStatus 成功时返回ACLNN_SUCCESS，否则返回对应错误码。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnMlaPreprocessV2GetWorkspaceSize(
    const aclTensor *input, const aclTensor *gamma0, const aclTensor *beta0, const aclTensor *quantScale0,
    const aclTensor *quantOffset0, const aclTensor *wdqkv, const aclTensor *descale0, const aclTensor *bias0,
    const aclTensor *gamma1, const aclTensor *beta1, const aclTensor *quantScale1, const aclTensor *quantOffset1,
    const aclTensor *wuq, const aclTensor *descale1, const aclTensor *bias1, const aclTensor *gamma2,
    const aclTensor *cos, const aclTensor *sin, const aclTensor *wuk, const aclTensor *kvCache,
    const aclTensor *kvCacheRope, const aclTensor *slotMapping, const aclTensor *ctkvScale,
    const aclTensor *qNopeScale, int64_t wdqDim, int64_t qRopeDim, int64_t kRopeDim, double epsilon,
    int64_t qRotaryCoeff, int64_t kRotaryCoeff, bool transeposeWdq, bool transeposeWuq, bool transeposeWuk,
    int64_t cacheMode, int64_t quantMode, bool doRmsNorm, int64_t wdkvSplitCount, bool qDownOutFlag,
    const aclTensor *qOutOut, const aclTensor *kvCacheOutOut, const aclTensor *qRopeOutOut,
    const aclTensor *krCacheOutOut, const aclTensor *qDownOutOut, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief 使用aclnnMlaPreprocessV2GetWorkspaceSize创建的执行器异步执行MlaPreprocessV2。
 *
 * @param workspace [IN] Device侧workspace地址。workspaceSize为0时可以传入nullptr。
 * @param workspaceSize [IN] 第一阶段接口返回的Device侧workspace大小，单位为字节。
 * @param executor [IN] aclnnMlaPreprocessV2GetWorkspaceSize返回的算子执行器。
 * @param stream [IN] 执行算子使用的ACL运行时Stream。
 * @return aclnnStatus 下发成功时返回ACLNN_SUCCESS，否则返回对应错误码。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnMlaPreprocessV2(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // ACLNN_MLA_PREPROCESS_V2_H
