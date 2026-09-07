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
 * \file aclnn_quant_flash_attn_metadata.cpp
 * \brief
 */

#include "l0_quant_flash_attn_metadata.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"

#include "../quant_flash_attn_metadata_check.h"

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnQuantFlashAttnMetadataGetWorkspaceSize(
    const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensKvOptional, const aclTensor *sequsedQOptional,
    const aclTensor *sequsedKvOptional, int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenKv, int64_t numHeadsQ,
    int64_t numHeadsKv, int64_t headDim, int64_t headDimV, int64_t quantMode, int64_t maskMode, int64_t winLeft,
    int64_t winRight, const char *layoutQ, const char *layoutQDescale, const char *layoutKv, const char *layoutOut,
    bool isGradEnabled, const aclTensor *metaData, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnQuantFlashAttnMetadata,
                   DFX_IN(cuSeqlensQOptional, cuSeqlensKvOptional, sequsedQOptional, sequsedKvOptional, batchSize,
                          maxSeqlenQ, maxSeqlenKv, numHeadsQ, numHeadsKv, headDim, headDimV, quantMode, maskMode,
                          winLeft, winRight, layoutQ, layoutQDescale, layoutKv, layoutOut, isGradEnabled),
                   DFX_OUT(metaData));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = QuantFlashAttnMetadataCheck::ParamsCheck(
        cuSeqlensQOptional, cuSeqlensKvOptional, sequsedQOptional, sequsedKvOptional, batchSize, maxSeqlenQ,
        maxSeqlenKv, numHeadsQ, numHeadsKv, headDim, headDimV, quantMode, maskMode, winLeft, winRight, layoutQ,
        layoutQDescale, layoutKv, layoutOut, isGradEnabled, metaData);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    const op::PlatformInfo &npuInfo = op::GetCurrentPlatformInfo();
    uint32_t aicCoreNum = npuInfo.GetCubeCoreNum();
    uint32_t aivCoreNum = npuInfo.GetVectorCoreNum();
    const char *socVersion = npuInfo.GetSocLongVersion().c_str();

    // 宿主侧读取输出 tensor 的 shape 并经 attr 下发: AICPU 侧输出 TensorShape 可能未填充,
    // FAG 偏移与容量校验依赖该值, 必须使用宿主侧可靠 shape。
    // 必须取 ViewShape: torch extension 链路下 aclTensor 的 StorageShape 是一维展平的
    // 总元素数(aclnn_common.h ConvertType: storage.nbytes()/itemsize), 二维行距只有
    // ViewShape(DimNum=2, Dim(1)=行长度) 才是正确语义
    constexpr int64_t DIM_ONE = 1;
    constexpr int64_t DIM_TWO = 2;
    constexpr int64_t DIM_IDX_0 = 0;
    constexpr int64_t DIM_IDX_1 = 1;
    int64_t metadataDimNum = metaData->GetViewShape().GetDimNum();
    int64_t metadataRowSize = 0;
    if (metadataDimNum >= DIM_TWO) {
        metadataRowSize = metaData->GetViewShape().GetDim(DIM_IDX_1);
    } else if (metadataDimNum == DIM_ONE) {
        metadataRowSize = metaData->GetViewShape().GetDim(DIM_IDX_0);
    }

    auto output = l0op::QuantFlashAttnMetadata(
        cuSeqlensQOptional, cuSeqlensKvOptional, sequsedQOptional, sequsedKvOptional, batchSize, maxSeqlenQ,
        maxSeqlenKv, numHeadsQ, numHeadsKv, headDim, headDimV, quantMode, maskMode, winLeft, winRight, layoutQ,
        layoutQDescale, layoutKv, layoutOut, isGradEnabled, socVersion, aicCoreNum, aivCoreNum, metadataDimNum,
        metadataRowSize, metaData, uniqueExecutor.get());
    CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

__attribute__((visibility("default"))) aclnnStatus aclnnQuantFlashAttnMetadata(void *workspace, uint64_t workspaceSize,
                                                                               aclOpExecutor *executor,
                                                                               aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnQuantFlashAttnMetadata);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
