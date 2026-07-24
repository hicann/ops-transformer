/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef A2AV_COMM_GUARD_H
#define A2AV_COMM_GUARD_H

#define A2AV_AIV_ONLY()                     \
    if ASCEND_IS_AIC { return; }            \
    if ASCEND_IS_AIV {                      \
        if (GetBlockIdx() != 0) { return; } \
    }

#define A2AV_AIV_ALL()                      \
    if ASCEND_IS_AIC { return; }

#endif // A2AV_COMM_GUARD_H