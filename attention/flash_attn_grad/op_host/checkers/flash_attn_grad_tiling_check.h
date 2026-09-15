/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_ATTN_GRAD_CHECK_H_
#define FLASH_ATTN_GRAD_CHECK_H_

#include "exe_graph/runtime/tiling_context.h"

namespace optiling {

// 校验层门面。具体校验按关注点拆在 op_host/checkers/ 下，顺序由
// checkers/fag_checker.cpp 的 RegisterCheckers 显式表达；输入/属性索引常量统一
// 定义在 info/flash_attn_grad_tiling_info.h。
class FlashAttnGradCheck {
public:
    static ge::graphStatus CheckParams(gert::TilingContext *context);
};

} // namespace optiling

#endif // FLASH_ATTN_GRAD_CHECK_H_
