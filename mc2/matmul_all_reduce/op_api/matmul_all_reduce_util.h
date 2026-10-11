/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_SRC_LEVEL2_MATMUL_ALL_REDUCE_UTIL_H_
#define OP_API_SRC_LEVEL2_MATMUL_ALL_REDUCE_UTIL_H_

#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/platform.h"
#include "acl/acl.h"
#include "common/utils/hccl_util.h"

#include "securec.h"

#include "common/utils/op_mc2.h"
#include "common/utils/op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "aclnn_kernels/contiguous.h"

#ifdef __cplusplus
extern "C" {
#endif

enum class NnopbaseHcclServerType : uint32_t {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_CCU,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

constexpr size_t FOUR_DIMS = 4;
constexpr size_t THREE_DIMS = 3;
constexpr size_t TWO_DIMS = 2;
constexpr size_t ONE_DIM = 1;
constexpr int64_t NUM_ACL_STOP_ON_FAILURE = 1;
constexpr uint8_t MC2_DEBUG_ONLY_CUBE = 1;
constexpr char DEBUG_MODE_ENV[] = "ASCEND_MC2_DEBUG_MODE";
constexpr size_t DIM_LEN_ONE = 1;
constexpr size_t DIM_LEN_TWO = 2;
constexpr char COMM_MODE_AICPU[] = "ai_cpu";
constexpr char COMM_MODE_CCU[] = "ccu";

struct NnopbaseDfxId {
    uint32_t id;
    const char* funcName;
    bool hasReg;
};

inline bool IsTransposeLastTwoDims(const aclTensor* tensor)
{
    // 当输入tensor的shape小于2或者大于6的时候，返回错误
    if (tensor->GetViewShape().GetDimNum() < 2 || tensor->GetViewShape().GetDimNum() > 6) {
        return false;
    }
    int64_t dim1 = tensor->GetViewShape().GetDimNum() - 1;
    int64_t dim2 = tensor->GetViewShape().GetDimNum() - 2;
    // BMM 场景下，Batch维度的stride需要等于 N, D 的乘积
    if (tensor->GetViewStrides()[dim2] == 1 && tensor->GetViewStrides()[dim1] == tensor->GetViewShape().GetDim(dim2)) {
        int64_t tmpNxD = tensor->GetViewShape().GetDim(dim1) * tensor->GetViewShape().GetDim(dim2);
        // 多batch连续，3是batch索引
        for (int64_t batchDim = tensor->GetViewShape().GetDimNum() - 3; batchDim >= 0; batchDim--) {
            if (tensor->GetViewStrides()[batchDim] != tmpNxD) {
                return false;
            }
            tmpNxD *= tensor->GetViewShape().GetDim(batchDim);
        }
        if (tensor->GetViewShape().GetDim(dim1) == 1 && tensor->GetViewShape().GetDim(dim2) == 1) {
            return false;
        }
        return true;
    }
    return false;
}

aclnnStatus MatmulAllReduceCheckParams(const aclTensor* x1, const aclTensor* x2, const aclTensor* x3,
                                       const aclTensor* bias, const char* reduceOp, int64_t streamMode,
                                       const aclTensor* output);
aclnnStatus QuantMatmulAllReduceCheckParams(const aclTensor* x1, const aclTensor* x2, const aclTensor* bias,
                                            const aclTensor* dequantScale, const aclTensor* pertokenScale,
                                            const aclTensor* x3, const char* reduceOp, int64_t streamMode,
                                            const aclTensor* output);
bool MatmulAllReduceCheckNotNull(const aclTensor* x1, const aclTensor* x2, const aclTensor* output);
bool MatmulAllReduceCheckAttr(const char* reduceOp, int64_t streamMode);
bool MatmulAllReduceCheckFormat(const aclTensor* x2);

bool MatmulAllReduceCheckDtypeValid(const aclTensor* x1, const aclTensor* x2, const aclTensor* x3,
                                    const aclTensor* bias, const aclTensor* output, bool is310P);
bool QuantMatmulAllReduceCheckDtypeValid(const aclTensor* x1, const aclTensor* x2, const aclTensor* bias,
                                         const aclTensor* dequantScale, const aclTensor* pertokenScale,
                                         const aclTensor* x3, const aclTensor* output);
bool MatmulAllReduceCheckShape(const aclTensor* x1, const aclTensor* x2, const aclTensor* x3, const aclTensor* bias,
                               const aclTensor* output);
bool QuantMatmulAllReduceCheckShape(const aclTensor* x1, const aclTensor* x2, const aclTensor* bias,
                                    const aclTensor* dequantScale, const aclTensor* pertokenScale, const aclTensor* x3,
                                    const aclTensor* output);
bool QuantMatmulAllReduceCheckDequantScale(const aclTensor* dequantScale, const op::Shape& outShape, size_t x1Len);
bool MatmulAllReduceIsWeightNZFormat(const aclTensor* x2);
bool QuantMatmulAllReduceIsWeightNZFormat(const aclTensor* x2);

bool MatmulAllReduceCheckValidContiguous(const aclTensor* tensor, const char* tensorName);
bool MatmulAllReduceCheckValidEmptyTensor(const aclTensor* x1, const aclTensor* x2);

// 全量化
bool QuantMatmulAllReduceIsAclnnPreTransposed(const aclTensor* x2);
void QuantMatmulAllReduceProcessTransposedX2(const aclTensor* x2, uint64_t& x2Dim0, uint64_t& x2Dim1,
                                             ge::AscendString& x2ShapeStr);
bool QuantMatmulAllReduceCheckPertokenScaleShape(const aclTensor* pertokenScale, const aclTensor* x1,
                                                 const size_t x1Len);

aclTensor* QuantMatmulAllReduceCopyTensor(const aclTensor* x2);
const aclTensor* QuantMatmulAllReduceTransTensor(const aclTensor* x2);

aclnnStatus InnerQuantMatmulAllReduceGetWorkspaceSize(const aclTensor* x1, const aclTensor* x2,
                                                      const aclTensor* biasOptional, const aclTensor* x3Optional,
                                                      const aclTensor* dequant, const aclTensor* pertokenScaleOptional,
                                                      const char* group, const char* reduceOp, int64_t commTurn,
                                                      const aclTensor* output, uint64_t* workspaceSize,
                                                      aclOpExecutor** executor);
bool IsCommModeValid(const char* commModePtr);
#ifdef __cplusplus
}
#endif

#endif // OP_API_SRC_LEVEL2_MATMUL_ALL_REDUCE_UTIL_H_
