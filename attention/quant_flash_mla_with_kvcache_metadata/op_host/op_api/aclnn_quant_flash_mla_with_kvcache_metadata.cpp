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
 * \file aclnn_quant_flash_mla_with_kvcache_metadata.cpp
 * \brief QuantFlashMlaWithKvcacheMetadata公共aclnn接口: 参数校验后经l0层发射AICPU分核任务
 */

#include "l0_quant_flash_mla_with_kvcache_metadata.h"
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

aclnnStatus aclnnQuantFlashMlaWithKvcacheMetadataGetWorkspaceSize(
    const aclTensor* cacheSeqlens, int64_t numHeadsQ, int64_t numHeadsKv, int64_t quantMode,
    const aclTensor* cuSeqlensQOptional, const aclTensor* sequsedQOptional, int64_t maxSeqlenQ, int64_t maxSeqlenKv,
    int64_t headDimQk, int64_t headDimV, int64_t maskMode, const char* layoutQ, const aclTensor* metaData,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnQuantFlashMlaWithKvcacheMetadata,
                   DFX_IN(cacheSeqlens, numHeadsQ, numHeadsKv, quantMode, cuSeqlensQOptional, sequsedQOptional,
                          maxSeqlenQ, maxSeqlenKv, headDimQk, headDimV, maskMode, layoutQ),
                   DFX_OUT(metaData));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 文档默认值: head_dim_qk默认576, head_dim_v默认512, mask_mode默认NO_MASK(0), layout_q默认BSND
    constexpr int64_t MLA_HEAD_DIM_QK_DEFAULT = 576;
    constexpr int64_t MLA_HEAD_DIM_V_DEFAULT = 512;
    constexpr int64_t NO_MASK = 0;
    headDimQk = (headDimQk > 0) ? headDimQk : MLA_HEAD_DIM_QK_DEFAULT;
    headDimV = (headDimV > 0) ? headDimV : MLA_HEAD_DIM_V_DEFAULT;
    maskMode = (maskMode >= 0) ? maskMode : NO_MASK;
    layoutQ = (layoutQ != nullptr && strlen(layoutQ) > 0) ? layoutQ : "BSND";

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

    auto output = l0op::QuantFlashMlaWithKvcacheMetadata(
        cacheSeqlensContiguous, cuSeqlensQOptionalContiguous, sequsedQOptionalContiguous, maxSeqlenQ, maxSeqlenKv,
        numHeadsQ, numHeadsKv, headDimQk, headDimV, quantMode, maskMode, layoutQ, socVersion, aicCoreNum, aivCoreNum,
        metaData, uniqueExecutor.get());
    CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

__attribute__((visibility("default"))) aclnnStatus aclnnQuantFlashMlaWithKvcacheMetadata(void* workspace,
                                                                                         uint64_t workspaceSize,
                                                                                         aclOpExecutor* executor,
                                                                                         aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnQuantFlashMlaWithKvcacheMetadata);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
