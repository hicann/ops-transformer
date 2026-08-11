/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstring>
#include "graph/types.h"
#include "aclnn_mla_prolog_v2_weight_nz.h"
#include "opdev/platform.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

extern aclnnStatus aclnnInnerMlaPrologV2GetWorkspaceSize(
    const aclTensor *tokenX, const aclTensor *weightDq, const aclTensor *weightUqQr, const aclTensor *weightUk,
    const aclTensor *weightDkvKr, const aclTensor *rmsnormGammaCq, const aclTensor *rmsnormGammaCkv,
    const aclTensor *ropeSin, const aclTensor *ropeCos, const aclTensor *cacheIndex, aclTensor *kvCacheRef,
    aclTensor *krCacheRef, const aclTensor *dequantScaleXOptional, const aclTensor *dequantScaleWDqOptional,
    const aclTensor *dequantScaleWUqQrOptional, const aclTensor *dequantScaleWDkvKrOptional,
    const aclTensor *quantScaleCkvOptional, const aclTensor *quantScaleCkrOptional,
    const aclTensor *smoothScalesCqOptional, double rmsnormEpsilonCq, double rmsnormEpsilonCkv, char *cacheModeOptional,
    const aclTensor *queryOut, const aclTensor *queryRopeOut, const aclTensor *dequantScaleQNopeOutOptional,
    uint64_t *workspaceSize, aclOpExecutor **executor);

extern aclnnStatus aclnnInnerMlaPrologV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                         const aclrtStream stream);

class TensorHolder {
public:
    TensorHolder(const aclTensor *&output, aclDataType dataType, std::string varName)
    {
        inner_ = nullptr;
        name_ = varName;
        if (output == nullptr) {
            std::vector<int64_t> shape = {0};
            int64_t addr = 0xff;
            inner_ = aclCreateTensor(shape.data(), shape.size(), dataType, shape.data(), 0, ACL_FORMAT_ND, shape.data(),
                                     shape.size(), static_cast<void *>(&addr));
            output = inner_;
        }
    }

    ~TensorHolder()
    {
        if (inner_) {
            aclDestroyTensor(inner_);
            inner_ = nullptr;
        }
    }

    void CheckTensorConditionalNotNull(bool conditional) const
    {
        if (inner_ && conditional) {
            OP_LOGW("Check %s != nullptr failed!", name_.c_str());
        } else if (!inner_ && !conditional) {
            OP_LOGW("Check %s == nullptr failed!", name_.c_str());
        }
    }

    bool IsTensorNotNull() const
    {
        return inner_ == nullptr;
    }

private:
    const aclTensor *inner_;
    std::string name_;
};

aclnnStatus aclnnMlaPrologV2WeightNzGetWorkspaceSize(
    const aclTensor *tokenX, const aclTensor *weightDq, const aclTensor *weightUqQr, const aclTensor *weightUk,
    const aclTensor *weightDkvKr, const aclTensor *rmsnormGammaCq, const aclTensor *rmsnormGammaCkv,
    const aclTensor *ropeSin, const aclTensor *ropeCos, const aclTensor *cacheIndex, aclTensor *kvCacheRef,
    aclTensor *krCacheRef, const aclTensor *dequantScaleXOptional, const aclTensor *dequantScaleWDqOptional,
    const aclTensor *dequantScaleWUqQrOptional, const aclTensor *dequantScaleWDkvKrOptional,
    const aclTensor *quantScaleCkvOptional, const aclTensor *quantScaleCkrOptional,
    const aclTensor *smoothScalesCqOptional, double rmsnormEpsilonCq, double rmsnormEpsilonCkv, char *cacheModeOptional,
    const aclTensor *queryOut, const aclTensor *queryRopeOut, const aclTensor *dequantScaleQNopeOutOptional,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "Interface aclnnMlaPrologV2WeightNzGetWorkspaceSize "
                                         "are no longer supported on Ascend950.");
        return ACLNN_ERR_RUNTIME_ERROR;
    }
    static bool isFirstCall = true;
    if (isFirstCall) {
        OP_LOGW("aclnnMlaPrologV2WeightNzGetWorkspaceSize is scheduled to be deprecated in December 2026, "
                "and will be replaced by the aclnnMlaPrologV3WeightNzGetWorkspaceSize. "
                "We apologize for any inconvenience caused and appreciate your timely migration to the new interface.");
        isFirstCall = false;
    }
    auto dequantScaleQNopeHolder =
        TensorHolder(dequantScaleQNopeOutOptional, aclDataType::ACL_FLOAT, std::string("dequantScaleQNopeOut"));
    if (dequantScaleQNopeOutOptional == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Failed to create the holder of tensor dequantScaleQNopeOut!");
        return ge::GRAPH_FAILED;
    }
    if (tokenX == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "tokenX cannot be null!");
        return ge::GRAPH_FAILED;
    }
    if (kvCacheRef == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "kvCacheRef cannot be null!");
        return ge::GRAPH_FAILED;
    }
    dequantScaleQNopeHolder.CheckTensorConditionalNotNull(tokenX->GetDataType() == ge::DT_INT8 &&
                                                          kvCacheRef->GetDataType() == ge::DT_INT8);

    return aclnnInnerMlaPrologV2GetWorkspaceSize(
        tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin, ropeCos,
        cacheIndex, kvCacheRef, krCacheRef, dequantScaleXOptional, dequantScaleWDqOptional, dequantScaleWUqQrOptional,
        dequantScaleWDkvKrOptional, quantScaleCkvOptional, quantScaleCkrOptional, smoothScalesCqOptional,
        rmsnormEpsilonCq, rmsnormEpsilonCkv, cacheModeOptional, queryOut, queryRopeOut, dequantScaleQNopeOutOptional,
        workspaceSize, executor);
}

aclnnStatus aclnnMlaPrologV2WeightNz(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                     const aclrtStream stream)
{
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "Interface aclnnMlaPrologV2WeightNz are no longer supported on Ascend950.");
        return ACLNN_ERR_RUNTIME_ERROR;
    }
    static bool isFirstCall = true;
    if (isFirstCall) {
        OP_LOGW("aclnnMlaPrologV2WeightNz is scheduled to be deprecated in December 2026, "
                "and will be replaced by the aclnnMlaPrologV3WeightNz. "
                "We apologize for any inconvenience caused and appreciate your timely migration to the new interface.");
        isFirstCall = false;
    }
    return aclnnInnerMlaPrologV2(workspace, workspaceSize, executor, stream);
}

} // namespace

#ifdef __cplusplus
}
#endif
