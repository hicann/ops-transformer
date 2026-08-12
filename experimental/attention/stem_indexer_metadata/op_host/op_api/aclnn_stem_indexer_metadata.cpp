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
 * \file aclnn_stem_indexer_metadata.cpp
 * \brief
 */

#include "aclnn_stem_indexer_metadata.h"
#include "l0_stem_indexer_metadata.h"
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

#include "../stem_indexer_metadata_checker.h"

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnStemIndexerMetadataGetWorkspaceSize(
    const aclTensor *qSeqLens, const aclTensor *kvSeqLens,
    int64_t qHeads, int64_t kvHeads, bool causal, int64_t stemBlockSize, int64_t dimQkflat, int64_t windowSize,
    const aclTensor *metadata, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnStemIndexerMetadata,
                   DFX_IN(qSeqLens, kvSeqLens,
                          qHeads, kvHeads, causal, stemBlockSize, dimQkflat, windowSize),
                   DFX_OUT(metadata));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = ParamsCheck(qSeqLens, kvSeqLens, qHeads, kvHeads, stemBlockSize, dimQkflat, windowSize, metadata);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    const op::PlatformInfo &npuInfo = op::GetCurrentPlatformInfo();
    uint32_t aicCoreNum = npuInfo.GetCubeCoreNum();
    uint32_t aivCoreNum = npuInfo.GetVectorCoreNum();
    const char *socVersion = npuInfo.GetSocLongVersion().c_str();

    auto output = l0op::StemIndexerMetadata(qSeqLens, kvSeqLens,
                                            qHeads, kvHeads, causal, stemBlockSize, dimQkflat, windowSize,
                                            socVersion, aicCoreNum, aivCoreNum, metadata, uniqueExecutor.get());
    CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

__attribute__((visibility("default")))
aclnnStatus
aclnnStemIndexerMetadata(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                         aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnStemIndexerMetadata);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
