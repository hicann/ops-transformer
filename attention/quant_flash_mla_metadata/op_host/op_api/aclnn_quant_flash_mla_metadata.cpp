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
 * \file aclnn_quant_flash_mla_metadata.cpp
 * \brief QuantFlashMlaMetadata公共aclnn接口: 参数校验后经l0层发射AICPU分核任务
 */

#include "l0_quant_flash_mla_metadata.h"
#include <cstring>
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/contiguous.h"
#include "acl/acl_rt.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/tensor_view_utils.h"

#include "../quant_flash_mla_metadata_check.h"

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnQuantFlashMlaMetadataGetWorkspaceSize(const aclTensor* cacheSeqlens, int64_t numHeadsQ,
                                                       int64_t numHeadsKv, int64_t quantMode,
                                                       const aclTensor* cuSeqlensQOptional,
                                                       const aclTensor* sequsedQOptional, int64_t maxSeqlenQ,
                                                       int64_t maxSeqlenKv, int64_t headDimQk, int64_t headDimV,
                                                       int64_t maskMode, const char* layoutQ, const aclTensor* metaData,
                                                       uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnQuantFlashMlaMetadata,
                   DFX_IN(cacheSeqlens, numHeadsQ, numHeadsKv, quantMode, cuSeqlensQOptional, sequsedQOptional,
                          maxSeqlenQ, maxSeqlenKv, headDimQk, headDimV, maskMode, layoutQ),
                   DFX_OUT(metaData));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // Torch/schema 层已经对可选属性应用默认值。这里不能把 0/负数静默改成合法值，
    // 否则异常 case 会绕过 checker。仅为空 layout 按当前唯一支持布局 TND 应用默认值。
    layoutQ = (layoutQ != nullptr && strlen(layoutQ) > 0) ? layoutQ : "TND";

    auto ret = QuantFlashMlaMetadataCheck::ParamsCheck(cacheSeqlens, cuSeqlensQOptional, sequsedQOptional, maxSeqlenQ,
                                                       maxSeqlenKv, numHeadsQ, numHeadsKv, headDimQk, headDimV,
                                                       quantMode, maskMode, layoutQ, metaData);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    const aclTensor* cacheSeqlensContiguous = nullptr;
    if (cacheSeqlens != nullptr) {
        cacheSeqlensContiguous = l0op::Contiguous(cacheSeqlens, uniqueExecutor.get());
        if (cacheSeqlensContiguous == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "cacheSeqlens contiguous is null");
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    const aclTensor* cuSeqlensQOptionalContiguous = nullptr;
    if (cuSeqlensQOptional != nullptr) {
        cuSeqlensQOptionalContiguous = l0op::Contiguous(cuSeqlensQOptional, uniqueExecutor.get());
        if (cuSeqlensQOptionalContiguous == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "cuSeqlensQOptional contiguous is null");
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    const aclTensor* sequsedQOptionalContiguous = nullptr;
    if (sequsedQOptional != nullptr) {
        sequsedQOptionalContiguous = l0op::Contiguous(sequsedQOptional, uniqueExecutor.get());
        if (sequsedQOptionalContiguous == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "sequsedQOptional contiguous is null");
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    // 与主算子 ACLNN tiling 保持一致：优先获取当前线程的有效核数（含 stream 控核）。
    // 查询失败时，与 opbase InitL2Phase1Context 一样回退到硬件平台核数。
    const op::PlatformInfo& npuInfo = op::GetCurrentPlatformInfo();
    uint32_t aicCoreNum = 0;
    uint32_t aivCoreNum = 0;
    if (aclrtGetResInCurrentThread(ACL_RT_DEV_RES_CUBE_CORE, &aicCoreNum) != ACL_SUCCESS) {
        aicCoreNum = npuInfo.GetCubeCoreNum();
    }
    if (aclrtGetResInCurrentThread(ACL_RT_DEV_RES_VECTOR_CORE, &aivCoreNum) != ACL_SUCCESS) {
        aivCoreNum = npuInfo.GetVectorCoreNum();
    }
    const char* socVersion = npuInfo.GetSocLongVersion().c_str();

    constexpr int64_t DIM_ONE = 1;
    constexpr int64_t DIM_TWO = 2;
    constexpr int64_t DIM_IDX_0 = 0;
    constexpr int64_t DIM_IDX_1 = 1;
    const int64_t metadataDimNum = metaData->GetViewShape().GetDimNum();
    int64_t metadataRowSize = 0;
    if (metadataDimNum >= DIM_TWO) {
        metadataRowSize = metaData->GetViewShape().GetDim(DIM_IDX_1);
    } else if (metadataDimNum == DIM_ONE) {
        metadataRowSize = metaData->GetViewShape().GetDim(DIM_IDX_0);
    }

    auto output = l0op::QuantFlashMlaMetadata(
        cacheSeqlensContiguous, cuSeqlensQOptionalContiguous, sequsedQOptionalContiguous, maxSeqlenQ, maxSeqlenKv,
        numHeadsQ, numHeadsKv, headDimQk, headDimV, quantMode, maskMode, layoutQ, socVersion, aicCoreNum, aivCoreNum,
        metadataDimNum, metadataRowSize, metaData, uniqueExecutor.get());
    CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

__attribute__((visibility("default"))) aclnnStatus aclnnQuantFlashMlaMetadata(void* workspace, uint64_t workspaceSize,
                                                                              aclOpExecutor* executor,
                                                                              aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnQuantFlashMlaMetadata);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
