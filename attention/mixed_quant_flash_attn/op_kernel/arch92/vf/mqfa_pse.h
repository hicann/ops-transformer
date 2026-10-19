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
 * \file mqfa_pse.h
 * \brief MQFA-local PSE type definitions, isolated from regbaseutil to avoid
 * pulling in the arch35 PSE header (which introduces other-regbase include paths).
 */

#ifndef MQFA_PSE_H
#define MQFA_PSE_H

namespace mqfaregbaseutil {
enum class PseTypeEnum {
    PSE_OUTER_MUL_ADD_TYPE = 0, // default
    PSE_OUTER_ADD_MUL_TYPE,
    PSE_INNER_MUL_ADD_TYPE,
    PSE_INNER_MUL_ADD_SQRT_TYPE,
    PSE_INVALID_TYPE,
    PSE_NONE_TYPE = 9
};
} // namespace mqfaregbaseutil
#endif
