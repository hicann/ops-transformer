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
 * \file engram_fetch_tiling_key.h
 * \brief kernel侧tiling key
 */

#ifndef ENGRAM_FETCH_TILING_KEY_H
#define ENGRAM_FETCH_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

#define ENGRAM_FETCH_DEFAULT_MODE 0
#define ENGRAM_FETCH_TRAIN_MODE 1
// 模板参数
ASCENDC_TPL_ARGS_DECL(EngramFetch, ASCENDC_TPL_UINT_DECL(EngramFetchMode, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST,
                                                         ENGRAM_FETCH_DEFAULT_MODE, ENGRAM_FETCH_TRAIN_MODE));

// 模板参数组合
// 用于调用GET_TPL_TILING_KEY获取TilingKey时，接口内部校验TilingKey是否合法
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(EngramFetchMode, ASCENDC_TPL_UI_LIST,
                                                          ENGRAM_FETCH_DEFAULT_MODE, ENGRAM_FETCH_TRAIN_MODE)));

#endif // ENGRAM_FETCH_TILING_KEY_H
