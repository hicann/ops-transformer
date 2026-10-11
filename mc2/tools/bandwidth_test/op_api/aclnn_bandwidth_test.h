/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_BANDWIDTH_TEST_
#define OP_API_INC_BANDWIDTH_TEST_

#include <string>

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 算子功能：实现BandwidthTest功能，测试带宽性能的调度算子。
 * @brief aclnnBandwidthTest的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * @param [in] x: 计算输入，Tensor，数据类型float16，bfloat16，float32，必须为2维，数据格式支持ND。输入的token数据。
 * @param [in] dstRankId: 计算输入，Tensor，数据类型int32，必须为1维，数据格式支持ND。每个token的目标rank索引。
 * @param [in] group: 计算输入，str。通信域名称。
 * @param [in] worldSize: 计算输入，int。通信域size。
 * @param [in] maxBs: 计算输入，int。最大batch size大小。
 * @param [in] mode: 计算输入，int。模式，0表示精度模式，1表示纯发模式。
 * @param [out] y: 计算输出，Tensor，必选输出，数据类型float16, bfloat16,
 * float32，仅支持2维，数据格式支持ND。输出的token数据。
 * @param [out] receiveCnt:
 * 计算输出，Tensor，必选输出，数据类型int32，仅支持1维，数据格式支持ND。从各rank接收的token数。
 * @param [out] workspaceSize: 出参，返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 出参，返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回值，返回状态码
 *
 */
ACLNN_API aclnnStatus aclnnBandwidthTestGetWorkspaceSize(const aclTensor* x, const aclTensor* dstRankId, char* group,
                                                         int64_t worldSize, int64_t maxBs, int64_t mode, char* commAlg,
                                                         int64_t aivNum, aclTensor* y, aclTensor* receiveCnt,
                                                         uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnBandwidthTest的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnBandwidthTestGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnBandwidthTest(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                         aclrtStream stream);
#ifdef __cplusplus
}
#endif
#endif // OP_API_INC_BANDWIDTH_TEST_
