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
 * \file fag_feature_checker.h
 * \brief 特性开关类校验：不是参数非法，而是"这个特性本轮没实现"。
 *
 * 与其他 checker 的区别：这里拒绝的输入本身是合法的，只是 FAGrad 还没实现。
 * 所以要拒绝得明确，不能静默放行让用户以为拿到了想要的语义。
 */

#ifndef FLASH_ATTN_GRAD_CHECKERS_FAG_FEATURE_CHECKER_H_
#define FLASH_ATTN_GRAD_CHECKERS_FAG_FEATURE_CHECKER_H_

#include "fag_base_checker.h"

namespace optiling {

// D2：确定性计算明确拒绝。
class FagDeterministicChecker : public FagBaseChecker {
public:
    ge::graphStatus Check(FagCheckCtx &ctx) override;
    const char *Name() const override
    {
        return "Deterministic";
    }
};

} // namespace optiling

#endif // FLASH_ATTN_GRAD_CHECKERS_FAG_FEATURE_CHECKER_H_
