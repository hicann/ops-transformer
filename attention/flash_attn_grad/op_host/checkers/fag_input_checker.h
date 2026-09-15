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
 * \file fag_input_checker.h
 * \brief 张量清单自身的校验：必选输入是否齐、dtype 是否一致。
 *
 * 形态参照正向 attention/flash_attn/op_host/checkers/common_checker_flash_attn.h。
 */

#ifndef FLASH_ATTN_GRAD_CHECKERS_FAG_INPUT_CHECKER_H_
#define FLASH_ATTN_GRAD_CHECKERS_FAG_INPUT_CHECKER_H_

#include "fag_base_checker.h"

namespace optiling {

// 必选输入 q/k/v/dout/attn_out/softmax_lse 的 shape 必须都拿得到。
// 放在最前面：后面所有 checker 都直接解引用这些指针。
class FagInputExistenceChecker : public FagBaseChecker {
public:
    ge::graphStatus Check(FagCheckCtx &ctx) override;
    const char *Name() const override
    {
        return "InputExistence";
    }
};

// q/k/v/dout/attn_out 与 dq/dk/dv 必须同 dtype。FAGrad 只支持 fp16/bf16，
// 具体取值范围由 flash_attn_grad_def.cpp 的注册信息挡在 tiling 之前。
class FagDtypeChecker : public FagBaseChecker {
public:
    ge::graphStatus Check(FagCheckCtx &ctx) override;
    const char *Name() const override
    {
        return "Dtype";
    }
};

} // namespace optiling

#endif // FLASH_ATTN_GRAD_CHECKERS_FAG_INPUT_CHECKER_H_
