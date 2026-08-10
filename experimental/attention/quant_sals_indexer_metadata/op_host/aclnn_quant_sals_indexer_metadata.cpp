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
 * \file aclnn_quant_sals_indexer_metadata.cpp
 * \brief
 */

#include "aclnn_quant_sals_indexer_metadata.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "l0_quant_sals_indexer_metadata.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"

#ifdef __cplusplus
extern "C" {
#endif

static aclnnStatus ParamsCheck(const aclTensor* actualSeqLengthsKvOptional,
                               int64_t batchSize,
                               int64_t keySeqSize,
                               int64_t keyHeadNum,
                               int64_t headDim,
                               int64_t sparseBlockSize,
                               double sparseRatio,
                               int64_t fixedTailCount,
                               char* layoutKeyOptional,
                               const aclTensor* metaData) {
  if (headDim != 128) {
    return ACLNN_ERR_PARAM_INVALID;
  }

  if (sparseBlockSize != 16) {
    return ACLNN_ERR_PARAM_INVALID;
  }
  if (batchSize < 0 || keySeqSize < 0 || keyHeadNum < 0 || sparseBlockSize < 0 || fixedTailCount < 0) {
    return ACLNN_ERR_PARAM_INVALID;
  }
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnQuantSalsIndexerMetadataGetWorkspaceSize(
    const aclTensor* actualSeqLengthsKvOptional,
    int64_t batchSize,
    int64_t keySeqSize,
    int64_t keyHeadNum,
    int64_t headDim,
    int64_t sparseBlockSize,
    double sparseRatio,
    int64_t fixedTailCount,
    char* layoutKeyOptional,
    const aclTensor* metaData,
    uint64_t* workspaceSize,
    aclOpExecutor** executor) {
  L2_DFX_PHASE_1(
      aclnnQuantSalsIndexerMetadata,
      DFX_IN(actualSeqLengthsKvOptional, batchSize, keySeqSize, keyHeadNum, headDim,
             sparseBlockSize, sparseRatio, fixedTailCount, layoutKeyOptional),
      DFX_OUT(metaData));

  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  auto ret = ParamsCheck(actualSeqLengthsKvOptional, batchSize, keySeqSize, keyHeadNum, headDim,
             sparseBlockSize, sparseRatio, fixedTailCount, layoutKeyOptional, metaData);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  const op::PlatformInfo &npuInfo = op::GetCurrentPlatformInfo();
  uint32_t aicCoreNum = npuInfo.GetCubeCoreNum();
  uint32_t aivCoreNum = npuInfo.GetVectorCoreNum();
  std::string socVersionStr = npuInfo.GetSocLongVersion();
  const char* socVersionOptional = socVersionStr.c_str();

  auto output = l0op::QuantSalsIndexerMetadata(
      actualSeqLengthsKvOptional, batchSize, keySeqSize, keyHeadNum, headDim, aicCoreNum, aivCoreNum,
      sparseBlockSize, sparseRatio, fixedTailCount, layoutKeyOptional, socVersionOptional, metaData,
      uniqueExecutor.get());
  CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);

  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

__attribute__((visibility("default"))) aclnnStatus
aclnnQuantSalsIndexerMetadata(void* workspace,
                              uint64_t workspaceSize,
                              aclOpExecutor* executor,
                              aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnQuantSalsIndexerMetadata);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
