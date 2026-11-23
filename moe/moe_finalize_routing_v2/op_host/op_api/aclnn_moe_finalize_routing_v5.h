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
 * \file aclnn_moe_finalize_routing_v5.h
 * \brief
 */

#ifndef OP_API_INC_MOE_FINALIZE_ROUTING_V5_H_
#define OP_API_INC_MOE_FINALIZE_ROUTING_V5_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnMoeFinalizeRoutingV5的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：合并MoE FFN的路由输出，支持在合并时对MXFP4输入进行反量化。
 * @param [in] expandedX: MoE FFN输出，支持FLOAT16、BFLOAT16、FLOAT32和FLOAT4_E2M1。
 * Drop Less场景使用二维Tensor，Drop Pad场景使用三维Tensor。FLOAT4_E2M1输入必须为连续ND Tensor，逻辑宽度H为正偶数。
 * @param [in] expandedRowIdx: 路由到expandedX物理行的索引，为一维INT32 Tensor，长度为NUM_ROWS * K。
 * @param [in] x1Optional: 第一个共享专家输出，可传nullptr。传入时shape与out一致，MXFP4场景使用BFLOAT16。
 * @param [in] x2Optional: 第二个共享专家输出，可传nullptr。传入时要求同时传入x1Optional，shape与out一致，
 * MXFP4场景使用BFLOAT16。
 * @param [in] biasOptional: 专家偏置，可传nullptr。传入时要求同时传入expertIdxOptional，shape为(E_bias, H)，
 * MXFP4场景使用BFLOAT16。
 * @param [in] scalesOptional: 路由权重，可传nullptr，shape为(NUM_ROWS, K)。支持FLOAT16、BFLOAT16、FLOAT32，
 * 具体组合依平台而定。MXFP4场景仅支持FLOAT32。
 * @param [in] expertIdxOptional: 专家索引，可传nullptr，类型为INT32，shape为(NUM_ROWS, K)。提供biasOptional时必传。
 * @param [in] xOptional: 拷贝专家和常量专家使用的原始token输入，可传nullptr，shape为(NUM_ROWS, H)。
 * MXFP4场景必须传nullptr。
 * @param [in] alpha1Optional: 常量专家对原始token输入的缩放系数，可传nullptr。
 * 传入时shape为(constantExpertEnd - constantExpertStart, H)，MXFP4场景必须传nullptr。
 * @param [in] alpha2Optional: 常量专家对偏移向量的缩放系数，可传nullptr。
 * 传入时shape为(constantExpertEnd - constantExpertStart, H)，MXFP4场景必须传nullptr。
 * @param [in] vOptional: 常量专家的偏移向量，可传nullptr。
 * 传入时shape为(constantExpertEnd - constantExpertStart, H)，MXFP4场景必须传nullptr。
 * @param [in] expandedXScaleOptional: MXFP4块反量化系数。当expandedX为FLOAT4_E2M1时必传，为连续的FLOAT8_E8M0
 * ND Tensor，shape为(R, ceil(H / 64), 2)，R为expandedX的物理行数，每32个元素共享一个系数。
 * 其他受支持的输入类型会忽略此参数。
 * @param [in] dropPadMode: 路由索引的布局和填充模式。0和2为Drop Less，1和3为Drop Pad。
 * 0和1按列读取expandedRowIdx，2和3按行读取。
 * @param [in] zeroExpertRange: 零计算专家的左闭右开区间，可传nullptr。MXFP4场景必须传nullptr或空数组。
 * @param [in] copyExpertRange: 拷贝专家的左闭右开区间，可传nullptr。MXFP4场景必须传nullptr或空数组。
 * @param [in] constantExpertRange: 常量专家的左闭右开区间，可传nullptr。MXFP4场景必须传nullptr或空数组。
 * @param [in] k: 每个token的Top-K路由数。MXFP4场景要求k大于0，且expandedRowIdx的长度可被k整除，
 * NUM_ROWS为expandedRowIdx的长度除以k。
 * @param [in] out: 调用方提供的输出Tensor，shape为(NUM_ROWS, H)。MXFP4场景为连续的BFLOAT16 ND Tensor。
 * @param [out] workspaceSize: 返回需要在NPU Device侧申请的workspace大小，单位为字节。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnMoeFinalizeRoutingV5GetWorkspaceSize(
    const aclTensor* expandedX, const aclTensor* expandedRowIdx, const aclTensor* x1Optional,
    const aclTensor* x2Optional, const aclTensor* biasOptional, const aclTensor* scalesOptional,
    const aclTensor* expertIdxOptional, const aclTensor* xOptional, const aclTensor* alpha1Optional,
    const aclTensor* alpha2Optional, const aclTensor* vOptional, const aclTensor* expandedXScaleOptional,
    int64_t dropPadMode, const aclIntArray* zeroExpertRange, const aclIntArray* copyExpertRange,
    const aclIntArray* constantExpertRange, int64_t k, const aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnMoeFinalizeRoutingV5的第二段接口，用于执行计算。
 *
 * 算子功能：合并MoE FFN的路由输出，支持在合并时对MXFP4输入进行反量化。
 * @param [in] workspace: 在NPU Device侧申请的workspace起始地址。workspaceSize为0时可传nullptr。
 * @param [in] workspaceSize: workspace大小，由第一段接口aclnnMoeFinalizeRoutingV5GetWorkspaceSize获取。
 * @param [in] executor: 第一段接口返回的op执行器。
 * @param [in] stream: 执行算子所用的ACL Stream。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnMoeFinalizeRoutingV5(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
