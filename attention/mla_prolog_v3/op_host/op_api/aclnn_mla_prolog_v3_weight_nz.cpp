/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstring>
#include "graph/types.h"
#include "aclnn_mla_prolog_v3_weight_nz.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_log.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

extern aclnnStatus aclnnInnerMlaPrologV3GetWorkspaceSize(
    const aclTensor *tokenX, const aclTensor *weightDq, const aclTensor *weightUqQr, const aclTensor *weightUk, const aclTensor *weightDkvKr,
    const aclTensor *rmsnormGammaCq, const aclTensor *rmsnormGammaCkv, const aclTensor *ropeSin, const aclTensor *ropeCos,
    aclTensor *kvCacheRef, aclTensor *krCacheRef, const aclTensor *cacheIndexOptional, const aclTensor *dequantScaleXOptional,
    const aclTensor *dequantScaleWDqOptional, const aclTensor *dequantScaleWUqQrOptional, const aclTensor *dequantScaleWDkvKrOptional,
    const aclTensor *quantScaleCkvOptional, const aclTensor *quantScaleCkrOptional, const aclTensor *smoothScalesCqOptional,
    const aclTensor *actualSeqLenOptional, double rmsnormEpsilonCq, double rmsnormEpsilonCkv, char *cacheModeOptional,
    bool queryNormFlag, int64_t weightQuantMode, int64_t kvCacheQuantMode, int64_t queryQuantMode, int64_t ckvkrRepoMode,
    int64_t quantScaleRepoMode, int64_t tileSize, double kNopeClipAlpha, double qcQrScale, double kcScale, const aclTensor *queryOut,
    const aclTensor *queryRopeOut, const aclTensor *dequantScaleQNopeOut, const aclTensor *queryNormOut, const aclTensor *dequantScaleQNormOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

extern aclnnStatus aclnnInnerMlaPrologV3(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                         const aclrtStream stream);


class TensorHolder {
public:
    explicit TensorHolder(const aclTensor *&output, aclDataType dataType) {
        inner = nullptr;
        if (output == nullptr) {
            std::vector<int64_t> shape = {0};
            int64_t addr = 0xff;
            inner = aclCreateTensor(shape.data(), shape.size(),
                dataType, shape.data(), 0, ACL_FORMAT_ND,
                shape.data(), shape.size(), static_cast<void *>(&addr));
            output = inner;
        }
    }
    ~TensorHolder() {
        if (inner) {
            aclDestroyTensor(inner);
            inner = nullptr;
        }
    }
private:
    const aclTensor *inner;
};

aclnnStatus aclnnMlaPrologV3WeightNzGetWorkspaceSize(
    const aclTensor *tokenX,
    const aclTensor *weightDq,
    const aclTensor *weightUqQr,
    const aclTensor *weightUk,
    const aclTensor *weightDkvKr,
    const aclTensor *rmsnormGammaCq,
    const aclTensor *rmsnormGammaCkv,
    const aclTensor *ropeSin,
    const aclTensor *ropeCos,
    aclTensor *kvCacheRef,
    aclTensor *krCacheRef,
    const aclTensor *cacheIndexOptional,
    const aclTensor *dequantScaleXOptional,
    const aclTensor *dequantScaleWDqOptional,
    const aclTensor *dequantScaleWUqQrOptional,
    const aclTensor *dequantScaleWDkvKrOptional,
    const aclTensor *quantScaleCkvOptional,
    const aclTensor *quantScaleCkrOptional,
    const aclTensor *smoothScalesCqOptional,
    const aclTensor *actualSeqLenOptional,
    double rmsnormEpsilonCq,
    double rmsnormEpsilonCkv,
    char *cacheModeOptional,
    bool queryNormFlag,
    int64_t weightQuantMode,
    int64_t kvCacheQuantMode,
    int64_t queryQuantMode,
    int64_t ckvkrRepoMode,
    int64_t quantScaleRepoMode,
    int64_t tileSize,
    double kNopeClipAlpha,
    double qcQrScale,
    double kcScale,
    const aclTensor *queryOut,
    const aclTensor *queryRopeOut,
    const aclTensor *dequantScaleQNopeOutOptional,
    const aclTensor *queryNormOutOptional,
    const aclTensor *dequantScaleQNormOutOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    if (tokenX ->GetDataType() == ge::DT_INT8 && kvCacheRef ->GetDataType() == ge::DT_INT8 && dequantScaleQNopeOutOptional == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Check dequantScaleQNopeOut != nullptr failed!");
    }
    auto holder1 = TensorHolder(dequantScaleQNopeOutOptional, aclDataType::ACL_FLOAT);
    auto holder2 = TensorHolder(queryNormOutOptional, aclDataType::ACL_BF16);
    auto holder3 = TensorHolder(dequantScaleQNormOutOptional, aclDataType::ACL_FLOAT);
    return aclnnInnerMlaPrologV3GetWorkspaceSize(
        tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin, ropeCos, kvCacheRef, krCacheRef,
        cacheIndexOptional, dequantScaleXOptional, dequantScaleWDqOptional, dequantScaleWUqQrOptional,
        dequantScaleWDkvKrOptional, quantScaleCkvOptional, quantScaleCkrOptional, smoothScalesCqOptional, actualSeqLenOptional,
        rmsnormEpsilonCq, rmsnormEpsilonCkv, cacheModeOptional,
        queryNormFlag, weightQuantMode, kvCacheQuantMode, queryQuantMode, ckvkrRepoMode, quantScaleRepoMode, tileSize,
        kNopeClipAlpha, qcQrScale, kcScale, queryOut, queryRopeOut,
        dequantScaleQNopeOutOptional, queryNormOutOptional, dequantScaleQNormOutOptional,
        workspaceSize, executor);
}

aclnnStatus aclnnMlaPrologV3WeightNz(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                     const aclrtStream stream)
{
    return aclnnInnerMlaPrologV3(workspace, workspaceSize, executor, stream);
}

} // namespace

#ifdef __cplusplus
}
#endif