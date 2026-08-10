/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_MOE_INIT_ROUTING_V4_H_
#define OP_API_INC_MOE_INIT_ROUTING_V4_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnMoeInitRoutingV4的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * @param x，token特征输入，必选参数
 * @param expertIdx，每一行特征对应的K个处理专家，必选参数
 * @param scaleOptional，用于计算quant结果的scale，可选参数，传nullptr表示不使用scale
 * @param offsetOptional，用于计算quant结果的偏移值，可选参数，传nullptr表示不输入
 * @param activeNumOptional，表示总的最大处理row数，可选参数，传nullptr表示默认为NUM_ROWS*K
 * @param topkWeightOptional，topk专家的路由权重，可选参数，传nullptr表示不输出expandedTopkWeightOut。
 * @param expertCapacity，每个专家能够处理的tokens数
 * @param expertNum，专家数
 * @param dropPadMode，0表示Dropless场景，1表示DropPad场景
 * @param expertTokensNumType，0为cumsum模式，1为count模式，2为key_value模式
 * @param expertTokensNumFlag，是否输出expertTokensCountOrCumsumOut
 * @param quantMode，量化模式
 * @param activeExpertRangeOptional，活跃的expert范围[expertStart, expertEnd]，左闭右开
 * @param rowIdxType，0为gather索引，1为scatter索引
 * @param expandedXOut，根据expertIdx进行扩展过的特征，必选参数
 * @param expandedRowIdxOut，expandedXOut和x的索引映射关系，必选参数
 * @param expertTokensCountOrCumsumOut，每个专家处理的token数量的统计结果，必选参数
 * @param expandedScaleOut，量化计算过程中scale的中间值，必选参数
 * @param expandedTopkWeightOut，按排序索引重排后的路由权重，可选参数。
 * @param workspaceSize，返回需要在Device侧申请的workspace大小
 * @param executor，返回op执行器，包含了算子计算流程
 * @return ACLNN_SUCCESS表示成功，其他值表示失败
 */
ACLNN_API aclnnStatus aclnnMoeInitRoutingV4GetWorkspaceSize(
    const aclTensor *x, const aclTensor *expertIdx,
    const aclTensor *scaleOptional, const aclTensor *offsetOptional,
    const aclTensor *activeNumOptional, const aclTensor *topkWeightOptional,
    int64_t expertCapacity, int64_t expertNum, int64_t dropPadMode,
    int64_t expertTokensNumType, bool expertTokensNumFlag,
    int64_t quantMode, const aclIntArray *activeExpertRangeOptional,
    int64_t rowIdxType,
    const aclTensor *expandedXOut, const aclTensor *expandedRowIdxOut,
    const aclTensor *expertTokensCountOrCumsumOut,
    const aclTensor *expandedScaleOut, const aclTensor *expandedTopkWeightOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/* @brief aclnnMoeInitRoutingV4的第二段接口，用于执行计算。 */
ACLNN_API aclnnStatus aclnnMoeInitRoutingV4(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                            aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
