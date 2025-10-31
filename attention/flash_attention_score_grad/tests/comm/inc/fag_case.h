/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fag_case.h
 * \brief
 */

#include "fa_case.h"
#include "../../../op_kernel/flash_attention_score_grad_tiling.h"

namespace ops::adv::tests::fag {

class FagCase : public ops::adv::tests::fa::FaCase {
public:
    /* 注册到 Tiling 公共框架的各模板优先级 */
    static constexpr int32_t kTemplatePriority_Ubngs1s2_Bb = 10000;     // Ubngs1s2Bb 模板
    static constexpr int32_t kTemplatePriority_Ungs1s2_Bbn = 11000;     // Ungs1s2Bbn 模板
    static constexpr int32_t kTemplatePriority_Us1s2_Bbn2 = 15000;      // S1s2Bn2 模板
    static constexpr int32_t kTemplatePriority_Us1s2_Bbn2gs1s2_sab = 15500;             // sameAB 模板
    static constexpr int32_t kTemplatePriority_Tnd_Mla = 15600;             // sameAB 模板
    static constexpr int32_t kTemplatePriority_Us1s2_Bbn2gs1s2 = 16000; // S1s2Bn2gs1s2 模板

public:
    FagCase();
    FagCase(const char *name, bool enable, const char *dbgInfo, OpInfoWithSocversion reverse, FaParam param,
            int32_t tilingTemplatePriority = kTilingTemplatePriority_Invalid);
    FagCase(const char *name, bool enable, const char *dbgInfo, 
            const std::function<void(FAG_INPUT_DTYPE)>& templatekeyKernelFunc, OpInfoWithSocversion reverse, 
            FaParam param, int32_t tilingTemplatePriority);

    bool Run() override;
};

} // namespace ops::adv::tests::fag
