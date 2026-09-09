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
 * \file aclnn_quant_reduce_scatter.h
 * \brief
 */
#ifndef OP_API_INC_QUANT_REDUCE_SCATTER_
#define OP_API_INC_QUANT_REDUCE_SCATTER_

#include <string>

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"
#include "hccl/hccl_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 算子功能：实现quant + reduceScatter融合计算
 * @brief aclnnQuantReduceScatter的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * @param [in] context: 通信上下文信息，由框架侧通信上下文管理模块创建，数据类型支持：INT32，数据格式支持ND。
 * @param [in] x: 公式中的输入x，不支持空Tensor，支持的维度为2-3维，shape为(BS, H)或者(B, S, H)，
 * 当量化方式为mx量化时，数据类型支持：FLOAT8_E4M3FN、FLOAT8_E5M2，数据格式支持ND，
 * 当量化方式为pertoken-pergroup量化时，数据类型支持：INT8, HIFLOAT8, FLOAT8_E4M3FN, FLOAT8_E5M2，数据格式支持ND。
 * @param [in] scales: 公式中的输入scales，不支持空Tensor，
 * 当量化方式为mx量化时，支持的维度为3-4维，shape必须对应x的shape为(BS, H/64, 2)或者(B, S, H/64,
 * 2)，数据类型支持：FLOAT8_E8M0，数据格式支持ND，
 * 当量化方式为pertoken-pergroup量化时，支持的维度为2-3维，shape必须对应x的shape为(BS, H/128)或者(B, S, H/
 * 128)，数据类型支持：FLOAT，数据格式支持ND。
 * @param [in] hcclBufferSize: hccl通信buffer大小，单位为字节。
 * @param [in] worldSize: 通信域卡数，取值支持：2、4、8。
 * @param [in] reduceOp: 公式中的reduce操作类型，默认值：sum，当前版本只支持sum，数据类型支持：string。
 * @param [out] output:
 * 公式中的输出output，不支持空Tensor，支持的维度为2维，shape为(BS/rankNum,H)，数据类型支持：FLOAT16, BFLOAT16,
 * FLOAT，数据格式为ND。
 * @param [out] workspaceSize: 返回需要在Device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnQuantReduceScatterGetWorkspaceSize(const aclTensor *context, const aclTensor *x,
                                                              const aclTensor *scales, const int64_t hcclBufferSize,
                                                              const int64_t worldSize, const char *reduceOp,
                                                              aclTensor *output, uint64_t *workspaceSize,
                                                              aclOpExecutor **executor);

/**
 * @brief aclnnQuantReduceScatter的第二段接口，用于执行计算。
 * @param [in] workspace: 在Device侧申请的workspace内存地址。
 * @param [in] workspacesize: 在Device侧申请的workspace大小，由第一段接口aclnnQuantReduceScatterGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: 指定执行任务的Stream。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnQuantReduceScatter(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                              const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_QUANT_REDUCE_SCATTER_
