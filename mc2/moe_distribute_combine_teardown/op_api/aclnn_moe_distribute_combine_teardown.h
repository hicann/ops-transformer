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
 * \file aclnn_moe_distribute_combine_teardown.h
 * \brief
 */
#ifndef OP_API_INC_MOE_DISTRIBUTE_COMBINE_TEARDOWN_
#define OP_API_INC_MOE_DISTRIBUTE_COMBINE_TEARDOWN_

#include <string>

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 算子功能：实现MoeDistributeCombineTeardown功能
 * @brief aclnnMoeDistributeCombineTeardown的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * @param [in] expandX: 计算输入，Tensor，数据类型float16, bfloat16，必须为2维，数据格式支持ND。
 * @param [in] quantExpandX: 计算输入，Tensor，数据类型int8，必须为2维，数据格式支持ND。
 * @param [in] expertIds: 计算输入，Tensor，数据类型int32，必须为2维，数据格式支持ND。
 * @param [in] expandIdx: 计算输入，Tensor，数据类型int32，必须为2维，数据格式支持ND。
 * @param [in] expertScales: 计算输入，Tensor，数据类型float32，必须为2维，数据格式支持ND。
 * @param [in] commCmdInfo:
 * 计算输入，Tensor，数据类型int32，必须为1维，数据格式支持ND，aclnnMoeDistributeCombineSetUp的输出。
 * @param [in] xActiveMaskOptional:
 * 计算输入，Tensor，数据类型bool，必须为1维，数据格式支持ND。预留参数，暂未使用，传空即可。
 * @param [in] sharedExpertXOptional:
 * 计算输入，Tensor，数据类型与expandX保持一致，必须为2维，数据格式支持ND。预留参数，暂未使用，传空即可。
 * @param [in] groupEp: 计算输入，str。ep通信域名称，专家并行的通信域。不能和groupTp相同。
 * @param [in] epWorldSize: 计算输入，int。ep通信域size。
 * @param [in] epRankId: 计算输入，int。ep本卡Id。同一个EP通信域中各卡的epRankId不能重复。
 * @param [in] moeExpertNum: 计算输入，int。MOE专家数量。
 * @param [in] expertShardType: 计算可选输入，int。共享专家卡分布类型。当前仅支持传0。
 * @param [in] sharedExpertNum: 计算可选输入，int。共享专家数量。当前仅支持传0。
 * @param [in] sharedExpertRankNum: 计算可选输入，int。共享专家卡数量。当前仅支持传0。
 * @param [in] globalBs: 计算可选输入，int。
 * @param [in] commQuantMode: 计算可选输入，int。通信量化类型。预留参数，暂未使用，传0即可。
 * @param [in] commType: 计算可选输入，int。通信方案选择。预留参数，暂未使用，传0即可。
 * @param [in] commAlg: 计算可选输入，str。通信算法类型。预留参数，暂未使用。
 * @param [out] xOut: 计算输出，Tensor，必选输出，数据类型支持float16，bfloat16，仅支持2维，数据格式支持ND。
 * @param [out] workspaceSize: 出参，返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 出参，返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回值，返回状态码。
 */
ACLNN_API aclnnStatus aclnnMoeDistributeCombineTeardownGetWorkspaceSize(
    const aclTensor* expandX, const aclTensor* quantExpandX, const aclTensor* expertIds, const aclTensor* expandIdx,
    const aclTensor* expertScales, const aclTensor* commCmdInfo, const aclTensor* xActiveMaskOptional,
    const aclTensor* sharedExpertXOptional, const char* groupEp, int64_t epWorldSize, int64_t epRankId,
    int64_t moeExpertNum, int64_t expertShardType, int64_t sharedExpertNum, int64_t sharedExpertRankNum,
    int64_t globalBs, int64_t commQuantMode, int64_t commType, const char* commAlg, aclTensor* xOut,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnMoeDistributeCombineTeardown的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存地址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，
 * 由第一段接口aclnnMoeDistributeCombineTeardownGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnMoeDistributeCombineTeardown(void* workspace, uint64_t workspaceSize,
                                                        aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_MOE_DISTRIBUTE_COMBINE_TEARDOWN_
