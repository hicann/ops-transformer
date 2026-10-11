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
 * \file aclnn_moe_fused_topk.h
 * \brief
 */

#ifndef OP_API_INC_MOE_FUSED_TOPK_H_
#define OP_API_INC_MOE_FUSED_TOPK_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
 * @brief aclnnMoeFusedTopk的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * @param [in] x: npu
 * device侧的aclTensor，数据类型支持FLOAT16、BFLOAT16和FLOAT，数据格式支持ND。
 * @param [in] addNum: npu
 * device侧的aclTensor，数据类型与x相同，数据格式支持ND。
 * @param [in] mappingNum: npu
 * device侧的aclTensor，数据类型支持INT32，数据格式支持ND。
 * @param [in] mappingTable: npu
 * device侧的aclTensor，数据类型支持INT32，数据格式支持ND。
 * @param [in] groupNum：表示分组数量。
 * uint32_t，数值必须大于0，专家分组数量。
 * @param [in] groupTopk：表示选择的分组数量。
 * uint32_t，数值必须大于0，选择的专家分组数量。
 * @param [in] topN：表示组内选取用于计算各组得分的专家数量。
 * uint32_t，数值必须大于0，组内选取专家的数量，用于计算各组得分。
 * @param [in] topK：最终选取的专家数量。
 * uint32_t，数值必须大于0。
 * @param [in] activateType：uint32_t，激活类型，当前只支持0(ACTIVATION_SIGMOID)。
 * uint32_t，激活类型，当前只支持0(ACTIVATION_SIGMOID)。
 * @param [in] isNorm：bool，是否对输出进行归一化。
 * bool，true 表示对输出进行归一化。
 * @param [in] scale：归一化后的系数乘。
 * float，isNorm为true时，对输出进行归一化后的系数乘。
 * @param [in] enableExpertMapping：是否使能物理专家到逻辑专家的映射。
 * bool，true 表示使能物理专家到逻辑专家的映射。
 * @param [out] y: npu
 * device侧的aclTensor，数据类型支持FLOAT32，数据格式支持ND。
 * @param [out] indices: npu
 * device侧的aclTensor，数据类型支持INT32，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小.
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnMoeFusedTopkGetWorkspaceSize(const aclTensor* x, const aclTensor* addNum,
                                                        const aclTensor* mappingNum, const aclTensor* mappingTable,
                                                        uint32_t groupNum, uint32_t groupTopk, uint32_t topN,
                                                        uint32_t topK, uint32_t activateType, bool isNorm, float scale,
                                                        bool enableExpertMapping, aclTensor* y, aclTensor* indices,
                                                        uint64_t* workspaceSize, aclOpExecutor** executor);

/* @brief aclnnMoeFusedTopk的第二段接口，用于执行计算.
 * @param [in] workspace: 在npu device侧申请的workspace内存起址
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnMoeFusedTopkGetWorkspaceSize获取.
 * @param [in] executor: op执行器，包含了算子计算流程.
 * @param [in] stream: acl stream流.
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnMoeFusedTopk(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                        aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_MOE_FUSED_TOPK_H_
