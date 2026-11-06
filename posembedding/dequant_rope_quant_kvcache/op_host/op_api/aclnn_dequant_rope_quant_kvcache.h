/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_ACLNN_DEQUANT_ROPE_QUANT_KVCACHE_H_
#define OP_API_ACLNN_DEQUANT_ROPE_QUANT_KVCACHE_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnDequantRopeQuantKvcache的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：对x进行可选反量化后切分为q、k、v，对q、k进行旋转位置编码，
 * 再对k、v进行量化，并按indices原地更新KV Cache。
 * B表示批大小，S表示序列长度，Nq和Nkv分别表示q和k/v的头数，D表示每个头的维度，H=(Nq+2*Nkv)*D。
 * Tensor参数均为npu device侧的aclTensor，数据格式支持ND；输入支持非连续Tensor，qOut、kOut、vOut须连续。
 *
 * @param [in] x: 计算输入张量，npu device侧的aclTensor，数据类型支持FLOAT16、BFLOAT16、INT32，
 * shape为[B, S, H]或[B, H]，H小于等于4096且为64的整数倍。
 * @param [in] cos: 旋转位置编码的余弦张量，npu device侧的aclTensor，数据类型支持FLOAT16、BFLOAT16，
 * x为3维时shape为[B, S, 1, D]，x为2维时shape为[B, D]；x非INT32时数据类型需与x一致。
 * @param [in] sin: 旋转位置编码的正弦张量，数据类型、shape与cos一致。
 * @param [in/out] kCacheRef: 缓存量化后的k，按indices原地更新，npu device侧的aclTensor，
 * 数据类型支持INT8，shape为[C_1, C_2, Nkv, D]，D为16的正整数倍。
 * @param [in/out] vCacheRef: 缓存量化后的v，按indices原地更新，数据类型、shape与kCacheRef一致。
 * @param [in] indices: KV Cache的更新位置信息，npu device侧的aclTensor，数据类型支持INT32，
 * cacheModeOptional为page且x为3维时shape为[B*S]，否则为[B]。
 * @param [in] scaleK: 量化k的缩放因子，npu device侧的aclTensor，数据类型支持FLOAT32，
 * 元素个数为Nkv*D，推荐shape为[Nkv, D]，兼容[Nkv*D]。
 * @param [in] scaleV: 量化v的缩放因子，数据类型、元素个数及推荐shape与scaleK一致。
 * @param [in] offsetKOptional: 可选输入，量化k的偏移因子，npu device侧的aclTensor，可以为空指针，
 * 数据类型、元素个数及推荐shape与scaleK一致。
 * @param [in] offsetVOptional: 可选输入，量化v的偏移因子，可以为空指针，
 * 数据类型、元素个数及推荐shape与scaleV一致。
 * @param [in] weightScaleOptional: 反量化的权重缩放因子，npu device侧的aclTensor，数据类型支持FLOAT32，
 * shape为[H]。x为INT32时必选，x为浮点类型时不生效。
 * @param [in] activationScaleOptional: 可选输入，反量化的激活缩放因子，npu device侧的aclTensor，
 * 可以为空指针，数据类型支持FLOAT32，x为3维时shape为[B*S]，x为2维时为[B]，仅在x为INT32时生效。
 * @param [in] biasOptional: 可选输入，反量化的偏置，npu device侧的aclTensor，可以为空指针，
 * 数据类型支持FLOAT32、FLOAT16、INT32、BFLOAT16，shape为[H]，仅在x为INT32时生效。
 * @param [in] sizeSplits: 切分q、k、v的长度，host侧的aclIntArray，长度为3，
 * 值为[Nq*D, Nkv*D, Nkv*D]，各元素非负且总和等于H。
 * @param [in] quantModeOptional: 量化模式，host侧的char*，目前仅支持"static"。
 * @param [in] layoutOptional: 数据布局，host侧的char*，目前仅支持"BSND"。
 * @param [in] kvOutput: host侧的bool，表示是否输出kOut和vOut。
 * 为false时kOut和vOut均须传入元素个数为0的空Tensor，不能传入空指针。
 * @param [in] cacheModeOptional: KV Cache的更新模式，host侧的char*，支持"page"和"contiguous"，
 * 默认为"contiguous"。
 * @param [out] qOut: 旋转位置编码后的q，npu device侧的aclTensor，数据类型与cos一致，
 * x为3维时shape为[B, S, Nq, D]，x为2维时shape为[B, Nq, D]。
 * @param [out] kOut: 旋转位置编码后的k，npu device侧的aclTensor，数据类型与cos一致，
 * kvOutput为true时，x为3维时shape为[B, S, Nkv, D]，x为2维时为[B, Nkv, D]。
 * @param [out] vOut: 切分得到的v，数据类型、shape与kOut一致。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnDequantRopeQuantKvcacheGetWorkspaceSize(
    const aclTensor* x, const aclTensor* cos, const aclTensor* sin, aclTensor* kCacheRef, aclTensor* vCacheRef,
    const aclTensor* indices, const aclTensor* scaleK, const aclTensor* scaleV, const aclTensor* offsetKOptional,
    const aclTensor* offsetVOptional, const aclTensor* weightScaleOptional, const aclTensor* activationScaleOptional,
    const aclTensor* biasOptional, const aclIntArray* sizeSplits, char* quantModeOptional, char* layoutOptional,
    bool kvOutput, char* cacheModeOptional, const aclTensor* qOut, const aclTensor* kOut, const aclTensor* vOut,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnDequantRopeQuantKvcache的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，
 * 由第一段接口aclnnDequantRopeQuantKvcacheGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: 指定执行任务的acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnDequantRopeQuantKvcache(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                   aclrtStream stream);

#ifdef __cplusplus
}
#endif
#endif
