/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/**
 * @file chunk_local_cumsum.cpp
 */
#include "chunk_local_cumsum.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

#include "acl/acl.h"
#include "aclnn_chunk_local_cumsum.h"
#include "common.h"

void ChunkLocalCumsum(aclrtStream& stream, void* workspace, Tensor& input, Tensor& cu_seqlens, Tensor& output,
                      int64_t head_num, int64_t chunk_size, const std::vector<int64_t>& cu_seqlen)
{
    // std::vector<int64_t> ioShape{token_num, qk_head_num, head_dim};
    // std::vector<int64_t> ioShape{token_num, vo_head_num, head_dim};
    // std::vector<int64_t> stateShape{chunk_num, vo_head_num, head_dim, head_dim};
    // std::vector<int64_t> gShape{token_num, vo_head_num};
    // std::vector<int64_t> cuSeqLenShape{cu_seqlen.size()};

    int64_t token_num = cu_seqlen.back();
    std::vector<int64_t> ioShape{1, token_num, head_num};
    std::vector<int64_t> cuSeqLenShape{static_cast<int64_t>(cu_seqlen.size())};

    aclIntArray* cuSeqlenDict = aclCreateIntArray(cu_seqlen.data(), cu_seqlen.size());
    std::vector<aclTensor*> inputTensor, outputTensor;
    aclFormat format = ACL_FORMAT_ND;
    inputTensor.emplace_back(aclCreateTensor(ioShape.data(), ioShape.size(), ACL_FLOAT, nullptr, 0, format,
                                             ioShape.data(), ioShape.size(), input.device()));

    inputTensor.emplace_back(aclCreateTensor(cuSeqLenShape.data(), cuSeqLenShape.size(), ACL_INT64, nullptr, 0, format,
                                             cuSeqLenShape.data(), cuSeqLenShape.size(), cu_seqlens.device()));
    outputTensor.emplace_back(aclCreateTensor(ioShape.data(), ioShape.size(), ACL_FLOAT, nullptr, 0, format,
                                              ioShape.data(), ioShape.size(), output.device()));

    // 3. 输入打包的aclTensor，调用GetWorkspaceSize
    size_t workspaceSize;
    aclOpExecutor* handle;
    auto ret = aclnnChunkLocalCumsumGetWorkspaceSize(inputTensor[0], inputTensor[1], chunk_size, cuSeqlenDict,
                                                     outputTensor[0], &workspaceSize, &handle);

    if (ret != ACL_SUCCESS) {
        (void)aclrtDestroyStream(stream);
        const char* tmp_err_msg = NULL;
        tmp_err_msg = aclGetRecentErrMsg();
        if (tmp_err_msg != NULL) {
            printf(" ERROR Message : %s \n ", tmp_err_msg);
        }
        ERROR_LOG("Get Operator aclnnChunkLocalCumsumGetWorkspaceSize Workspace failed. "
                  "error code is %d",
                  static_cast<int32_t>(ret));
    }
    DEBUG_LOG("Execute aclnnChunkLocalCumsumGetWorkspaceSize success, workspace size "
              "%lu",
              workspaceSize);

    // 4. 执行计算
    ret = aclnnChunkLocalCumsum(workspace, workspaceSize, handle, stream);
    if (ret != ACL_SUCCESS) {
        const char* tmp_err_msg = NULL;
        tmp_err_msg = aclGetRecentErrMsg();
        if (tmp_err_msg != NULL) {
            printf(" ERROR Message : %s \n ", tmp_err_msg);
        }
        (void)aclrtDestroyStream(stream);
        ERROR_LOG("Execute Operator failed. error code is %d", static_cast<int32_t>(ret));
    }
    DEBUG_LOG("Execute aclnnChunkLocalCumsum success");
}
