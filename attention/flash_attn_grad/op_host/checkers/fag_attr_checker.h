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
 * \file fag_attr_checker.h
 * \brief 属性驱动的校验：属性取值范围、mask_mode 与 attn_mask/window 的搭配、layout。
 *
 * 形态参照正向 attention/flash_attn/op_host/checkers/mask_checker_flash_attn.h。
 *
 * 易漂移口径：
 *   - mask_mode 只收 0/3/4，其余 sparse 模式本轮不接；
 *   - attn_mask 必须是 [2048, 2048]；
 *   - layout_q / layout_kv / layout_out 必须三者相同。
 */

#ifndef FLASH_ATTN_GRAD_CHECKERS_FAG_ATTR_CHECKER_H_
#define FLASH_ATTN_GRAD_CHECKERS_FAG_ATTR_CHECKER_H_

#include "fag_base_checker.h"

namespace optiling {

// 唯一读属性的 checker：把值读进 FagCheckCtx 并做单参数范围校验。
// 必须注册在 Mask / Layout / Shape 之前，它们都依赖这里填好的字段。
class FagAttrRangeChecker : public FagBaseChecker {
public:
    ge::graphStatus Check(FagCheckCtx &ctx) override;
    const char *Name() const override
    {
        return "AttrRange";
    }
};

// mask_mode 与 attn_mask 是否存在、win_left/win_right 取值必须自洽。
class FagMaskChecker : public FagBaseChecker {
public:
    ge::graphStatus Check(FagCheckCtx &ctx) override;
    const char *Name() const override
    {
        return "Mask";
    }
};

class FagLayoutChecker : public FagBaseChecker {
public:
    ge::graphStatus Check(FagCheckCtx &ctx) override;
    const char *Name() const override
    {
        return "Layout";
    }
};

} // namespace optiling

#endif // FLASH_ATTN_GRAD_CHECKERS_FAG_ATTR_CHECKER_H_
