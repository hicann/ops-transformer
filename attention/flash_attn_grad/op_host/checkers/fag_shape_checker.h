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
 * \file fag_shape_checker.h
 * \brief 跨张量的 shape 一致性：维度数、B/S/N/D 对齐、softmax_lse、cu_seqlens。
 *
 * 形态参照正向 attention/flash_attn/op_host/checkers/seq_len_checker_flash_attn.h 与
 * softmax_lse_checker_flash_attn.h。
 *
 * 易漂移口径：
 *   - TND 下 cu_seqlens_q / cu_seqlens_kv 必传且为 1D，非 TND 下必须不传；
 *   - D 与 Dv 可以不等，但要求 Dv <= D，且两者都落在 (0, MAX_HEAD_DIM]。
 */

#ifndef FLASH_ATTN_GRAD_CHECKERS_FAG_SHAPE_CHECKER_H_
#define FLASH_ATTN_GRAD_CHECKERS_FAG_SHAPE_CHECKER_H_

#include "fag_base_checker.h"

namespace optiling {

// 依赖 FagAttrRangeChecker 已把 layout 填进 ctx。
class FagShapeChecker : public FagBaseChecker {
public:
    ge::graphStatus Check(FagCheckCtx &ctx) override;
    const char *Name() const override
    {
        return "Shape";
    }
};

} // namespace optiling

#endif // FLASH_ATTN_GRAD_CHECKERS_FAG_SHAPE_CHECKER_H_
