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
 * \file flash_attn_grad_tiling_info_parser.h
 * \brief L2 解析层：全算子唯一读取 gert::TilingContext 取轴与属性的地方。
 */

#ifndef FLASH_ATTN_GRAD_TILING_INFO_PARSER_H_
#define FLASH_ATTN_GRAD_TILING_INFO_PARSER_H_

#include "exe_graph/runtime/tiling_context.h"
#include "flash_attn_grad_tiling_info.h"

namespace optiling {

// 平台能力与前置资源校验（核数、SOC、workspace 槽位）。
ge::graphStatus ParsePlatform(gert::TilingContext *context, FagParsedInfo &info);

// 属性 + shape + 可选输入存在性 + 布局视图参数。
ge::graphStatus ParseFlashAttnGradInfo(gert::TilingContext *context, FagParsedInfo &info);

// 空 shape 检测。**预留钩子**：当前只用于日志，不改变分核与 workspace，
// 以保持与重构前逐位一致。待空 tensor 分支落地后由编排层消费。
bool IsEmptyShape(const FagParsedInfo &info);

} // namespace optiling

#endif // FLASH_ATTN_GRAD_TILING_INFO_PARSER_H_
