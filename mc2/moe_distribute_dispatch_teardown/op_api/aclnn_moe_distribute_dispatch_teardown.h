/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_MOE_DISTRIBUTE_DISPATCH_TEARDOWN_
#define OP_API_INC_MOE_DISTRIBUTE_DISPATCH_TEARDOWN_

#include <string>
#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 算子功能：实现MoeDistributeDispatchTeardown功能，接收EP域的alltoallv通信结果，并整理结果输出
 * @brief aclnnMoeDistributeDispatchTeardown的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * @param [in] x: 计算输入，Tensor，数据类型float16，bfloat16，必须为2维，数据格式支持ND。输入的token数据。
 * @param [in] y: 计算输入，Tensor，数据类型float16，bfloat16，必须为2维，数据格式支持ND。本卡待发送的通信数据。
 * @param [in] expertIds:
    计算输入，Tensor，数据类型int32，必须为1维，数据格式支持ND。每个token的topK个专家索引。
 * @param [in] commCmdInfo: 计算输入，Tensor，数据类型int32，必须为2维，数据格式支持ND，通信的cmd信息。
 * @param [in] groupEp: 计算输入，str。ep通信域名称，专家并行的通信域。
 * @param [in] epWorldSize: 计算输入，数据类型int64。ep通信域size。
 * @param [in] epRankId: 计算输入，数据类型int64。ep域本卡Id。同一个EP通信域中各卡的epRankId不能重复。
 * @param [in] moeExpertNum: 计算输入，int64。MOE专家数量。
 * @param [in] expertShardType: 计算可选输入，int64。专家卡分布类型。
 * @param [in] sharedExpertNum: 计算可选输入，int64。共享专家数量。
 * @param [in] sharedExpertRankNum: 计算可选输入，int64。共享专家卡数量。
 * @param [in] quantMode: 计算可选输入，int64。量化模式。
 * @param [in] globalBs: 计算可选输入，int64。ep通信域全局的batch size大小。
 * @param [in] expertTokenNumsType: 计算可选输入，int64。输出expertTokenNums中的值语义类型。
 * @param [in] comm_type: 计算输入，int。通信方案选择。
 * @param [in] commAlg: 计算输入，char*。通信亲和内存布局算法。当前版本不支持，传空指针即可。
 * @param [out] expandXOut: 计算输出，Tensor，必选输出，数据类型支持float16，bfloat16，
    int8，仅支持2维，数据格式支持ND。根据expertIdx进行扩展过的token特征。
 * @param [out] dynamicScalesOut:
    计算输出，Tensor，必选输出，数据类型float32，仅支持1维，数据格式支持ND。quantMode为0时输出为空。
 * @param [out] expertTokenNumsOut:
    计算输出，Tensor，必选输出，数据类型int64，仅支持1维，数据格式支持ND。每个专家收到的token个数。
 * @param [out] epRecvCountsOut:
    计算输出，Tensor，必选输出，数据类型int32，仅支持1维，数据格式支持ND。从各卡接收的token数。
 * @param [out] workspaceSize: 出参，返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 出参，返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回值，返回状态码。
 *
 */
ACLNN_API aclnnStatus aclnnMoeDistributeDispatchTeardownGetWorkspaceSize(
    const aclTensor* x, const aclTensor* y, const aclTensor* expertIds, const aclTensor* commCmdInfo,
    const char* groupEp, int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum, int64_t expertShardType,
    int64_t sharedExpertNum, int64_t sharedExpertRankNum, int64_t quantMode, int64_t globalBs,
    int64_t expertTokenNumsType, int64_t commType, char* commAlg, aclTensor* expandXOut, aclTensor* dynamicScalesOut,
    aclTensor* assistInfoForCombineOut, aclTensor* expertTokenNumsOut, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnMoeDistributeDispatchTeardown的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，
    由第一段接口aclnnMoeDistributeDispatchTeardownGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnMoeDistributeDispatchTeardown(void* workspace, uint64_t workspaceSize,
                                                         aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_MOE_DISTRIBUTE_DISPATCH_TEARDOWN_
