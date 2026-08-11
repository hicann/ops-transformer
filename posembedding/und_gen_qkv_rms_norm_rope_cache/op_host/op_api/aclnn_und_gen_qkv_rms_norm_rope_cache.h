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
 * \file aclnn_und_gen_qkv_rms_norm_rope_cache.h
 * \brief
 */

#ifndef OP_API_ACLNN_UND_GEN_QKV_RMS_NORM_ROPE_CACHE_H_
#define OP_API_ACLNN_UND_GEN_QKV_RMS_NORM_ROPE_CACHE_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnUndGenQkvRmsNormRopeCache的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：把 und/gen 两段 QKV 拼接后按 catIndicesOptional 间接寻址，逐 token 做 Q/K 的 RMSNorm + MRoPE，
 * Q 作为独立输出，K/V 经 Cast float32->bf16 后按 slotMapping 写入分页 KV Cache。
 *
 * @param [in] undQkv: 计算输入张量，npu device侧的aclTensor，数据类型支持BF16，shape为[und_len, N, D]，
 * N = numHeadsQ + numHeadsK + numHeadsV，数据格式支持ND。
 * @param [in] undWeightsQ: 计算输入张量，npu device侧的aclTensor，数据类型支持BF16，shape为[D]，数据格式支持ND。
 * @param [in] undWeightsK: 计算输入张量，npu device侧的aclTensor，数据类型支持BF16，shape为[D]，数据格式支持ND。
 * @param [in] cosSinCache: 计算输入张量，npu device侧的aclTensor，数据类型支持FLOAT32，shape为[max_pos, D]，
 * 前半为cos、后半为sin，数据格式支持ND。
 * @param [in/out] kCacheRef: 计算输入输出张量（原地更新），npu device侧的aclTensor，数据类型支持BF16，
 * shape为[Bn, Bs, Hk, D]（连续BBND布局），数据格式支持ND。
 * @param [in/out] vCacheRef: 计算输入输出张量（原地更新），npu device侧的aclTensor，数据类型支持BF16，
 * shape为[Bn, Bs, Hv, D]（连续BBND布局），数据格式支持ND。
 * @param [in] slotMapping: 计算输入张量，npu device侧的aclTensor，数据类型支持INT64，shape为[T]，
 * 值为 block_num * Bs + row_idx，由调用方预计算，数据格式支持ND。
 * @param [in] positions: 计算输入张量，npu device侧的aclTensor，数据类型支持INT64，shape为[3, T]，
 * 分别为MRoPE的时间/高度/宽度三轴位置，数据格式支持ND。
 * @param [in] genQkvOptional: 计算输入张量，可选，npu device侧的aclTensor，数据类型支持BF16，shape为[gen_len, N, D]；
 * 纯prefill场景传nullptr，数据格式支持ND。
 * @param [in] genWeightsQOptional: 计算输入张量，可选，npu device侧的aclTensor，数据类型支持BF16，shape为[D]；
 * genQkv非空时必须提供，数据格式支持ND。
 * @param [in] genWeightsKOptional: 计算输入张量，可选，npu device侧的aclTensor，数据类型支持BF16，shape为[D]；
 * genQkv非空时必须提供，数据格式支持ND。
 * @param [in] catIndicesOptional: 计算输入张量，可选，npu device侧的aclTensor，数据类型支持INT64，shape为[T]，
 * 表示out_t到src_t的映射；传nullptr时退化为恒等映射，数据格式支持ND。
 * @param [in] numHeadsQ: 计算属性，数据类型支持int64_t，Q的头数。
 * @param [in] numHeadsK: 计算属性，数据类型支持int64_t，K的头数。
 * @param [in] numHeadsV: 计算属性，数据类型支持int64_t，V的头数。
 * @param [in] normEps: 计算属性，数据类型支持double（下发前转 float），RMSNorm防除零epsilon，默认值1e-6。
 * @param [in] mropeSection: 计算属性，可选，数据类型支持aclIntArray（长度3），MRoPE三轴分段；
 * 传nullptr或空数组时退化为[D/2, 0, 0]，即标准RoPE。
 * @param [out] qOut: 计算输出张量，npu device侧的aclTensor，数据类型支持BF16，shape为[T, Hq, D]，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnUndGenQkvRmsNormRopeCacheGetWorkspaceSize(
    const aclTensor* undQkv, const aclTensor* undWeightsQ, const aclTensor* undWeightsK, const aclTensor* cosSinCache,
    aclTensor* kCacheRef, aclTensor* vCacheRef, const aclTensor* slotMapping, const aclTensor* positions,
    const aclTensor* genQkvOptional, const aclTensor* genWeightsQOptional,
    const aclTensor* genWeightsKOptional, const aclTensor* catIndicesOptional,
    int64_t numHeadsQ, int64_t numHeadsK, int64_t numHeadsV, double normEps, const aclIntArray* mropeSection,
    aclTensor* qOut, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnUndGenQkvRmsNormRopeCache的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnUndGenQkvRmsNormRopeCache(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_ACLNN_UND_GEN_QKV_RMS_NORM_ROPE_CACHE_H_
