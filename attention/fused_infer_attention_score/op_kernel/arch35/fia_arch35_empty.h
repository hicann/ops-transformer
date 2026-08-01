/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fia_arch35_empty.h
 * \brief arch35 FIA 空tensor路由(emptyTensor: kv为空,直接输出零)
 */

#ifndef FIA_ARCH35_EMPTY_H_
#define FIA_ARCH35_EMPTY_H_

#include "fia_arch35_common.h"
#include "fia_arch35_zero_output.h"

inline __aicore__ void fia_empty_regbase(__gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse,
                                         __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    TPipe tPipe;
    FIA_REGBASE_COPY_TILING_DATA(tiling);
#if (ORIG_DTYPE_ATTENTION_OUT != DT_FLOAT16 && ORIG_DTYPE_ATTENTION_OUT != DT_BF16)
    FiaZeroOutput<fp8_e4m3fn_t> op;
#else
    FiaZeroOutput<half> op;
#endif
    op.Init(attentionOut, softmaxLse, tilingData);
    op.Process();
}

#endif // FIA_ARCH35_EMPTY_H_
