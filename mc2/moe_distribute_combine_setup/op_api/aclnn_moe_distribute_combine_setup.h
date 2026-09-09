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
 * \file aclnn_moe_distribute_combine_setup.h
 * \brief
 */
#ifndef OP_API_INC_MOE_DISTRIBUTE_COMBINE_SETUP
#define OP_API_INC_MOE_DISTRIBUTE_COMBINE_SETUP

#include <string>

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 算子功能：进行AlltoAllv通信，将数据写入对端GM。
 * @brief aclnnMoeDistributeCombineSetup的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * @param [in] expandX: 计算输入，Tensor，数据类型float16，bfloat16，必须为2维，数据格式支持ND。
 * @param [in] expertIds: 计算输入，Tensor，数据类型int32，必须为2维，数据格式支持ND。
 * @param [in] assistInfoForCombine: 计算输入，Tensor，数据类型int32，必须为1维，数据格式支持ND。
 * @param [in] groupEp: 计算输入，str。ep通信域名称，专家并行的通信域。不能和groupTp相同。
 * @param [in] epWorldSize: 计算输入，int。ep通信域size。
 * @param [in] epRankId: 计算输入，int。ep本卡Id。同一个EP通信域中各卡的epRankId不重复。
 * @param [in] moeExpertNum: 计算输入，int。MOE专家数量。
 * @param [in] expertShardType: 计算可选输入，int。共享专家卡分布类型。当前仅支持传0。
 * @param [in] sharedExpertNum: 计算可选输入，int。共享专家数量。
 * @param [in] sharedExpertRankNum: 计算可选输入，int。共享专家卡数量。
 * @param [in] globalBs: 计算可选输入，int。
 * @param [in] commQuantMode: 计算可选输入，int。通信量化类型。预留参数，暂未使用，传0即可。
 * @param [in] commType: 计算可选输入，int。通信方案选择。预留参数，暂未使用，传0即可。
 * @param [in] commAlg: 计算可选输入，str。通信算法选择。预留参数，暂未使用。
 * @param [out] quantExpandXOut: 计算输出，Tensor，数据类型支持int8，仅支持2维，数据格式支持ND。表示量化处理后的token。
 * @param [out] commCmdInfoOut: 计算输出，Tensor，数据类型支持int32，仅支持1维，数据格式支持ND。表示通信的cmd信息。
 * @param [out] workspaceSize: 出参，返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 出参，返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回值，返回状态码.
 *
 */
ACLNN_API aclnnStatus aclnnMoeDistributeCombineSetupGetWorkspaceSize(
    const aclTensor *expandX, const aclTensor *expertIds, const aclTensor *assistInfoForCombine, const char *groupEp,
    int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum, int64_t expertShardType, int64_t sharedExpertNum,
    int64_t sharedExpertRankNum, int64_t globalBs, int64_t commQuantMode, int64_t commType, const char *commAlg,
    aclTensor *quantExpandXOut, aclTensor *commCmdInfoOut, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * 算子功能：进行AlltoAllv通信，将数据写入对端GM。
 * @brief 计算aclnnMoeDistributeCombineSetup部分输出的size大小
 * @domain aclnn_ops_infer
 * @param [in] expandX: 计算输入，Tensor，数据类型float16，bfloat16，必须为2维，数据格式支持ND。
 * @param [in] expertIds: 计算输入，Tensor，数据类型int32，必须为2维，数据格式支持ND。
 * @param [in] assistInfoForCombine: 计算输入，Tensor，数据类型int32，必须为1维，数据格式支持ND。
 * @param [in] groupEp: 计算输入，str。ep通信域名称，专家并行的通信域。不能和groupTp相同。
 * @param [in] epWorldSize: 计算输入，int。ep通信域size。
 * @param [in] epRankId: 计算输入，int。ep本卡Id。同一个EP通信域中各卡的epRankId不重复。
 * @param [in] moeExpertNum: 计算输入，int。MOE专家数量。
 * @param [in] expertShardType: 计算可选输入，int。共享专家卡分布类型。当前仅支持传0。
 * @param [in] sharedExpertNum: 计算可选输入，int。共享专家数量。
 * @param [in] sharedExpertRankNum: 计算可选输入，int。共享专家卡数量。
 * @param [in] globalBs: 计算可选输入，int。
 * @param [in] commQuantMode: 计算可选输入，int。通信量化类型。预留参数，暂未使用，传0即可。
 * @param [in] commType: 计算可选输入，int。通信方案选择。预留参数，暂未使用，传0即可。
 * @param [in] commAlg: 计算可选输入，str。通信算法选择。预留参数，暂未使用。
 * @param [out] tokenMsgSize: 计算输出，int。aclnnMoeDistributeCombineSetup接口quantExpandXOut第二维的大小。
 * @param [out] commCmdInfoOutSize: 计算输出，int。aclnnMoeDistributeCombineSetup接口commCmdInfoOut的大小。
 * @return aclnnStatus: 返回值，返回状态码
 *
 */
ACLNN_API aclnnStatus aclnnMoeDistributeCombineSetupTeardownCalcOutputSize(
    const aclTensor *expandX, const aclTensor *expertIds, const aclTensor *assistInfoForCombine, const char *groupEp,
    int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum, int64_t expertShardType, int64_t sharedExpertNum,
    int64_t sharedExpertRankNum, int64_t globalBs, int64_t commQuantMode, int64_t commType, const char *commAlg,
    uint64_t &tokenMsgSize, uint64_t &commCmdInfoOutSize);

/**
 * @brief aclnnMoeDistributeCombineSetup的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，
 * 由第一段接口aclnnMoeDistributeCombineSetupGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnMoeDistributeCombineSetup(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                     aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_MOE_DISTRIBUTE_COMBINE_SETUP
