/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_MOE_DISTRIBUTE_COMBINE_V3_
#define OP_API_INC_MOE_DISTRIBUTE_COMBINE_V3_

#include <string>

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 算子功能：实现MoeDistributeCombineV2功能。
 * @brief aclnnMoeDistributeCombineV3的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * @param [in] expandX: 计算输入，Tensor，数据类型float16，bfloat16，必须为2维，数据格式支持ND。
 * @param [in] expertIds: 计算输入，Tensor，数据类型int32，必须为2维，数据格式支持ND。
 * @param [in] assistInfoForCombine: 计算输入，Tensor，数据类型int32，必须为1维，数据格式支持ND。
 * @param [in] epSendCounts: 计算输入，Tensor，数据类型int32，必须为1维，数据格式支持ND。
 * @param [in] expertScales: 计算输入，Tensor，数据类型float32，必须为2维，数据格式支持ND。
 * @param [in] tpSendCountsOptional: 计算输入，Tensor，数据类型int32，必须为1维，数据格式支持ND。若有TP域通信需要传参，若无TP域通信，传空指针即可。
 * @param [in] xActiveMaskOptional: 计算输入，Tensor，数据类型bool，必须为1维，数据格式支持ND。
 * @param [in] activationScaleOptional: 计算输入，Tensor，数据类型float32，必须为1维，数据格式支持ND。预留参数，当前版本不支持，传空指针即可。
 * @param [in] weightScaleOptional: 计算输入，Tensor，数据类型float32，必须为2维，数据格式支持ND。预留参数，暂未使用，传空即可。
 * @param [in] groupListOptional: 计算输入，Tensor，数据类型int64，必须为1维，数据格式支持ND。预留参数，暂未使用，传空即可。
 * @param [in] expandScalesOptional: 计算输入，Tensor，数据类型float32，必须为1维，数据格式支持ND。
 * @param [in] sharedExpertXOptional: 计算可选输入，Tensor，数据类型float16，bfloat16，必须为2维，数据格式支持ND。
 * @param [in] elastic_infoOptional: 计算可选输入，Tensor，数据类型int32，必须为1维，数据格式支持ND。
 * @param [in] ori_xOptional: 计算可选输入，Tensor，数据类型float16，bfloat16，必须为2维，数据格式支持ND。
 * @param [in] const_expert_alpha_1Optional: 计算可选输入，Tensor，数据类型float16，bfloat16，必须为2维，数据格式支持ND。
 * @param [in] const_expert_alpha_2Optional: 计算可选输入，Tensor，数据类型float16，bfloat16，必须为2维，数据格式支持ND。
 * @param [in] const_expert_vOptional: 计算可选输入，Tensor，数据类型float16，bfloat16，必须为2维，数据格式支持ND。
 * @param [in] groupEp: 计算输入，str。ep通信域名称，专家并行的通信域。不能和groupTp相同。
 * @param [in] epWorldSize: 计算输入，int。ep通信域size。
 * @param [in] epRankId: 计算输入，int。ep本卡Id。同一个EP通信域中各卡的epRankId不重复。
 * @param [in] moeExpertNum: 计算输入，int。MOE专家数量。
 * @param [in] groupTp: 计算可选输入，str。tp通信域名称，数据并行的通信域。
 * @param [in] tpWorldSize: 计算可选输入，int。tp通信域size。
 * @param [in] tpRankId: 计算可选输入，int。tp本卡Id。同一个TP通信域中各卡的tpRankId不能重复。
 * @param [in] expertShardType: 计算可选输入，int。专家共享类型。当前仅支持传0。
 * @param [in] sharedExpertNum: 计算可选输入，int。共享专家数量。
 * @param [in] sharedExpertRankNum: 计算可选输入，int。共享专家卡数量。
 * @param [in] globalBs: 计算可选输入，int。
 * @param [in] outDtype: 计算可选输入，int。输出数据类型。预留参数，暂未使用，传0即可。
 * @param [in] commQuantMode: 计算可选输入，int。通信量化类型。
 * @param [in] groupListType: 计算可选输入，int。groupList格式。预留参数，暂未使用，传0即可。
 * @param [in] commAlg: 计算可选输入，str。 通信算法类型。预留参数，暂未使用。
 * @param [in] zero_expert_num: 计算可选输入，int。
 * @param [in] copy_expert_num: 计算可选输入，int。
 * @param [in] const_expert_num: 计算可选输入，int。
 * @param [out] xOut: 计算输出，Tensor，必选输出，数据类型支持float16, bfloat16，仅支持2维，数据格式支持ND。
 * @param [out] workspaceSize: 出参，返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 出参，返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回值，返回状态码
 *
 */
ACLNN_API aclnnStatus aclnnMoeDistributeCombineV3GetWorkspaceSize(const aclTensor* expandX, const aclTensor* expertIds,
    const aclTensor* assistInfoForCombine, const aclTensor* epSendCounts,
    const aclTensor* expertScales, const aclTensor* tpSendCountsOptional,
    const aclTensor* xActiveMaskOptional, const aclTensor* activationScaleOptional,
    const aclTensor* weightScaleOptional, const aclTensor* groupListOptional, const aclTensor* expandScalesOptional,
    const aclTensor* sharedExpertXOptional, const aclTensor* elasticInfoOptional, const aclTensor* oriXOptional,
    const aclTensor* constExpertAlpha1Optional, const aclTensor* constExpertAlpha2Optional, 
    const aclTensor* constExpertVOptional,
    const char* groupEp, int64_t epWorldSize, 
    int64_t epRankId, int64_t moeExpertNum,
    const char* groupTp, int64_t tpWorldSize, int64_t tpRankId,
    int64_t expertShardType, int64_t sharedExpertNum, int64_t sharedExpertRankNum, 
    int64_t globalBs, int64_t outDtype, int64_t commQuantMode,
    int64_t groupListType, const char* commAlg, int64_t zeroExpertNum,
    int64_t copyExpertNum, int64_t constExpertNum,
    aclTensor* xOut, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnMoeDistributeCombineV3的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnMoeDistributeCombineV3GetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnMoeDistributeCombineV3(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_MOE_DISTRIBUTE_COMBINE_V3_