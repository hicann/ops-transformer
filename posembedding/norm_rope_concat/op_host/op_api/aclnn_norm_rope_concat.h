/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_norm_rope_concat.h
 * \brief
 */
#ifndef ACLNN_NORM_ROPE_CONCAT_H_
#define ACLNN_NORM_ROPE_CONCAT_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnNormRopeConcat的第一段接口，根据具体的计算流程，计算workspace大小。
 *
 * @param [in] query: npu
 * device侧的aclTensor, 表示注意力机制中的Query, 数据类型支持FLOAT32,FLOAT16,BFLOAT16且需与key、value一致,
 * shape为(B, S, N, D), 支持非连续的Tensor, 数据格式支持ND。
 * @param [in] key: npu
 * device侧的aclTensor, 表示注意力机制中的Key, 数据类型支持FLOAT32,FLOAT16,BFLOAT16且需与query、value一致,
 * shape为(B, S, N, D), 支持非连续的Tensor, 数据格式支持ND。
 * @param [in] value: npu
 * device侧的aclTensor, 表示注意力机制中的Value, 数据类型支持FLOAT32,FLOAT16,BFLOAT16且需与query、key一致,
 * shape为(B, S, N, D), 支持非连续的Tensor, 数据格式支持ND。
 * @param [in] encoderQuery: npu
 * device侧的aclTensor, 表示来自EncoderHiddenState的Query, 数据类型支持FLOAT32,FLOAT16,BFLOAT16且需与主输入一致,
 * shape为(B, S, N, D), 支持非连续的Tensor, 数据格式支持ND, 可以为空指针。
 * @param [in] encoderKey: npu
 * device侧的aclTensor, 表示来自EncoderHiddenState的Key, 数据类型支持FLOAT32,FLOAT16,BFLOAT16且需与主输入一致,
 * shape为(B, S, N, D), 支持非连续的Tensor, 数据格式支持ND, 可以为空指针。
 * @param [in] encoderValue: npu
 * device侧的aclTensor, 表示来自EncoderHiddenState的Value, 数据类型支持FLOAT32,FLOAT16,BFLOAT16且需与主输入一致,
 * shape为(B, S, N, D), 支持非连续的Tensor, 数据格式支持ND, 可以为空指针。
 * @param [in] normQueryWeight: npu
 * device侧的aclTensor, 表示LayerNorm的仿射变换参数, 作用在Query上, 数据类型支持FLOAT32,FLOAT16,BFLOAT16,
 * shape为(D), 支持非连续的Tensor, 数据格式支持ND, 可选, normType=2或4时需要提供。
 * @param [in] normQueryBias: npu
 * device侧的aclTensor, 表示LayerNorm的仿射变换参数, 作用在Query上, 数据类型支持FLOAT32,FLOAT16,BFLOAT16,
 * shape为(D), 支持非连续的Tensor, 数据格式支持ND, 可选, normType=2时需要提供。
 * @param [in] normKeyWeight: npu
 * device侧的aclTensor, 表示LayerNorm的仿射变换参数, 作用在Key上, 数据类型支持FLOAT32,FLOAT16,BFLOAT16,
 * shape为(D), 支持非连续的Tensor, 数据格式支持ND, 可选, normType=2或4时需要提供。
 * @param [in] normKeyBias: npu
 * device侧的aclTensor, 表示LayerNorm的仿射变换参数, 作用在Key上, 数据类型支持FLOAT32,FLOAT16,BFLOAT16,
 * shape为(D), 支持非连续的Tensor, 数据格式支持ND, 可选, normType=2时需要提供。
 * @param [in] normAddedQueryWeight: npu
 * device侧的aclTensor, 表示LayerNorm的仿射变换参数, 作用在encoderQuery上, 数据类型支持FLOAT32,FLOAT16,BFLOAT16,
 * shape为(D), 支持非连续的Tensor, 数据格式支持ND, 可选, normAddedType=2或4时需要提供。
 * @param [in] normAddedQueryBias: npu
 * device侧的aclTensor, 表示LayerNorm的仿射变换参数, 作用在encoderQuery上, 数据类型支持FLOAT32,FLOAT16,BFLOAT16,
 * shape为(D), 支持非连续的Tensor, 数据格式支持ND, 可选, normAddedType=2时需要提供。
 * @param [in] normAddedKeyWeight: npu
 * device侧的aclTensor, 表示LayerNorm的仿射变换参数, 作用在encoderKey上, 数据类型支持FLOAT32,FLOAT16,BFLOAT16,
 * shape为(D), 支持非连续的Tensor, 数据格式支持ND, 可选, normAddedType=2或4时需要提供。
 * @param [in] normAddedKeyBias: npu
 * device侧的aclTensor, 表示LayerNorm的仿射变换参数, 作用在encoderKey上, 数据类型支持FLOAT32,FLOAT16,BFLOAT16,
 * shape为(D), 支持非连续的Tensor, 数据格式支持ND, 可选, normAddedType=2时需要提供。
 * @param [in] ropeSin: npu
 * device侧的aclTensor, 表示RoPE的正弦编码, 数据类型支持FLOAT32,FLOAT16,BFLOAT16, shape为(S, D),
 * 支持非连续的Tensor, 数据格式支持ND, ropeType!=0时需要提供。
 * @param [in] ropeCos: npu
 * device侧的aclTensor, 表示RoPE的余弦编码, 数据类型支持FLOAT32,FLOAT16,BFLOAT16, shape为(S, D),
 * 支持非连续的Tensor, 数据格式支持ND, ropeType!=0时需要提供。
 * @param [in] normType: 作用在query、key上的归一化类型, 取值范围[0, 4],
 * 0表示NONE(不做归一化), 1表示LayerNorm, 2表示LayerNormAffine, 3表示RmsNorm, 4表示RmsNormAffine。
 * @param [in] normAddedType: 作用在encoderQuery、encoderKey上的归一化类型, 取值范围[0, 4],
 * 0表示NONE(不做归一化), 1表示LayerNorm, 2表示LayerNormAffine, 3表示RmsNorm, 4表示RmsNormAffine。
 * @param [in] ropeType: 旋转位置编码类型, 取值范围[0, 2], 0表示NONE(不做RoPE), 1表示Interleave, 2表示Half。
 * @param [in] concatOrder: 拼接顺序, 取值范围[0, 1], 0表示query在前, 1表示query在后。
 * @param [in] eps: 归一化时的epsilon值。
 * @param [in] isTraining: 是否为训练阶段, 决定是否输出反向使用的值。
 * @param [out] queryOutput: npu
 * device侧的aclTensor, 表示注意力机制中的query输出, 数据类型与query一致, 数据格式支持ND,
 * shape为(B, N, seqQuery + seqEncoderQuery, D), 支持非连续的Tensor。
 * @param [out] keyOutput: npu
 * device侧的aclTensor, 表示注意力机制中的key输出, 数据类型与key一致, 数据格式支持ND,
 * shape为(B, N, seqKey + seqEncoderKey, D), 支持非连续的Tensor。
 * @param [out] valueOutput: npu
 * device侧的aclTensor, 表示注意力机制中的value输出, 数据类型与value一致, 数据格式支持ND,
 * shape为(B, N, seqValue + seqEncoderValue, D), 支持非连续的Tensor。
 * @param [out] normQueryMean: npu
 * device侧的aclTensor, 表示LayerNorm中query的均值输出, 供反向使用, 数据类型支持FLOAT32,
 * shape为(B, seqQuery, N, 1), 支持非连续的Tensor, 数据格式支持ND, (normType=1或2)且isTraining=true时有效。
 * @param [out] normQueryRstd: npu
 * device侧的aclTensor, 表示LayerNorm中query的标准差倒数输出, 供反向使用, 数据类型支持FLOAT32,
 * shape为(B, seqQuery, N, 1), 支持非连续的Tensor, 数据格式支持ND, (normType=1或2)且isTraining=true时有效。
 * @param [out] normKeyMean: npu
 * device侧的aclTensor, 表示LayerNorm中key的均值输出, 供反向使用, 数据类型支持FLOAT32,
 * shape为(B, seqKey, N, 1), 支持非连续的Tensor, 数据格式支持ND, (normType=1或2)且isTraining=true时有效。
 * @param [out] normKeyRstd: npu
 * device侧的aclTensor, 表示LayerNorm中key的标准差倒数输出, 供反向使用, 数据类型支持FLOAT32,
 * shape为(B, seqKey, N, 1), 支持非连续的Tensor, 数据格式支持ND, (normType=1或2)且isTraining=true时有效。
 * @param [out] normAddedQueryMean: npu
 * device侧的aclTensor, 表示LayerNorm中encoderQuery的均值输出, 供反向使用,
 * 数据类型支持FLOAT32, shape为(B, seqEncoderQuery, N, 1), 支持非连续的Tensor, 数据格式支持ND,
 * (normAddedType=1或2)且isTraining=true时有效。
 * @param [out] normAddedQueryRstd: npu
 * device侧的aclTensor, 表示LayerNorm中encoderQuery的标准差倒数输出, 供反向使用,
 * 数据类型支持FLOAT32, shape为(B, seqEncoderQuery, N, 1), 支持非连续的Tensor, 数据格式支持ND,
 * (normAddedType=1或2)且isTraining=true时有效。
 * @param [out] normAddedKeyMean: npu
 * device侧的aclTensor, 表示LayerNorm中encoderKey的均值输出, 供反向使用,
 * 数据类型支持FLOAT32, shape为(B, seqEncoderKey, N, 1), 支持非连续的Tensor, 数据格式支持ND,
 * (normAddedType=1或2)且isTraining=true时有效。
 * @param [out] normAddedKeyRstd: npu
 * device侧的aclTensor, 表示LayerNorm中encoderKey的标准差倒数输出, 供反向使用,
 * 数据类型支持FLOAT32, shape为(B, seqEncoderKey, N, 1), 支持非连续的Tensor, 数据格式支持ND,
 * (normAddedType=1或2)且isTraining=true时有效。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnNormRopeConcatGetWorkspaceSize(
    const aclTensor* query, const aclTensor* key, const aclTensor* value, const aclTensor* encoderQuery,
    const aclTensor* encoderKey, const aclTensor* encoderValue, const aclTensor* normQueryWeight,
    const aclTensor* normQueryBias, const aclTensor* normKeyWeight, const aclTensor* normKeyBias,
    const aclTensor* normAddedQueryWeight, const aclTensor* normAddedQueryBias, const aclTensor* normAddedKeyWeight,
    const aclTensor* normAddedKeyBias, const aclTensor* ropeSin, const aclTensor* ropeCos, int64_t normType,
    int64_t normAddedType, int64_t ropeType, int64_t concatOrder, double eps, bool isTraining,
    const aclTensor* queryOutput, const aclTensor* keyOutput, const aclTensor* valueOutput,
    const aclTensor* normQueryMean, const aclTensor* normQueryRstd, const aclTensor* normKeyMean,
    const aclTensor* normKeyRstd, const aclTensor* normAddedQueryMean, const aclTensor* normAddedQueryRstd,
    const aclTensor* normAddedKeyMean, const aclTensor* normAddedKeyRstd, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnNormRopeConcat的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnNormRopeConcatGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnNormRopeConcat(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                          aclrtStream stream);
#ifdef __cplusplus
}
#endif

#endif
