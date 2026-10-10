/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_CAUSAL_CONV1D_UPDATE_H_
#define OP_API_INC_CAUSAL_CONV1D_UPDATE_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnCausalConv1dUpdate (decode / update) 的第一段接口，计算workspace大小。
 * @param [in] x 输入张量，shape为[B, 1, D]（3D），数据类型为FLOAT16或BFLOAT16。
 *               2D [N, D]仅在提供numAcceptedTokens（投机解码）时支持
 * @param [in] weight 卷积权重张量，shape为[W, D]，数据类型与x相同
 * @param [in, out] convStatesRef 卷积状态张量，shape为[numCacheLines, stateLen, D]，数据类型与x相同
 * @param [in] biasOptional 可选偏置张量，shape为[D]，数据类型与x相同
 * @param [in] queryStartLocOptional 可选的变长序列起始位置，shape为[batch+1]，数据类型为INT32
 * @param [in] cacheIndicesOptional 可选的缓存索引，shape为[batch]，数据类型为INT32。
 *            将每个序列映射到convStatesRef中对应的缓存槽位，取值范围为[0, numCacheLines)。
 *            cacheIndices[i]等于nullBlockId的序列将被跳过，nullBlockId可被多个序列重复使用。
 *            除nullBlockId外的值不应重复，否则多个序列读写同一缓存槽位会导致历史状态错误和写入冲突。
 *            不提供时，序列i直接使用缓存槽位i（恒等映射）。
 * @param [in] numAcceptedTokensOptional 可选的已接受token数，shape为[batch]，数据类型为INT32，用于投机解码
 * @param [in] blockIdxLastScheduledTokenOptional 可选的最后调度token块索引，shape为[batch]，数据类型为INT32
 * @param [in] initialStateIdxOptional 可选的初始状态索引，shape为[batch]，数据类型为INT32
 * @param [in] activation 激活模式，"silu"表示SiLU，"none"表示无激活
 * @param [in] nullBlockId 空块ID，用于跳过不需要计算的序列
 * @param [in] maxQueryLen 最大查询长度参数
 * @param [out] y 输出张量，shape与x相同，数据类型与x相同
 * @param [out] workspaceSize workspace大小（字节）
 * @param [out] executor 执行器指针
 * @return ACLNN_SUCCESS 成功
 * @return ACLNN_ERR_PARAM_NULLPTR 参数为空指针
 * @return ACLNN_ERR_INNER_TILING_ERROR 内部tiling错误
 * @domain aclnn_ops_infer
 */
ACLNN_API aclnnStatus aclnnCausalConv1dUpdateGetWorkspaceSize(
    const aclTensor* x, const aclTensor* weight, aclTensor* convStatesRef, const aclTensor* biasOptional,
    const aclTensor* queryStartLocOptional, const aclTensor* cacheIndicesOptional,
    const aclTensor* numAcceptedTokensOptional, const aclTensor* blockIdxLastScheduledTokenOptional,
    const aclTensor* initialStateIdxOptional, const char* activation, int64_t nullBlockId, int64_t maxQueryLen,
    aclTensor* y, uint64_t* workspaceSize, aclOpExecutor** executor);

/* @brief aclnnCausalConv1dUpdate 的第二段接口，用于执行计算。 */
ACLNN_API aclnnStatus aclnnCausalConv1dUpdate(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                              aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
