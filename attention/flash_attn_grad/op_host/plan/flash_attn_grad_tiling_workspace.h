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
 * \file flash_attn_grad_tiling_workspace.h
 * \brief L4 策略层：workspace 具名分区。
 *
 * 把分区写成带 offset 的结构体而不是只算一个总和，是为了让"host 的分区"与
 * "kernel make_ptr 的切分"能逐段对照，而不是只对一个总字节数。
 */

#ifndef FLASH_ATTN_GRAD_TILING_WORKSPACE_H_
#define FLASH_ATTN_GRAD_TILING_WORKSPACE_H_

#include "../info/flash_attn_grad_tiling_info.h"

namespace optiling {

FagWorkspacePlan BuildWorkspacePlan(const FagParsedInfo &info);

} // namespace optiling

#endif // FLASH_ATTN_GRAD_TILING_WORKSPACE_H_
