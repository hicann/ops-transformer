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
 * @file chunk_local_cumsum.h
 */
#ifndef MI_QKV_UNIFY_H
#define MI_QKV_UNIFY_H

#include <vector>
#include "acl/acl.h"
#include "tensor.h"
// #include "op_runner.h"

void ChunkLocalCumsum(aclrtStream& stream, void* workspace, Tensor& input, Tensor& cu_seqlens, Tensor& output,
                      int64_t head_num, int64_t chunk_size, const std::vector<int64_t>& cu_seqlen);

#endif
