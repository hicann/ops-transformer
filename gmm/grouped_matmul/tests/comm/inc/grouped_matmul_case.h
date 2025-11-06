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
 * \file grouped_matmul_case.h
 * \brief GroupedMatmul 测试用例.
 */

#ifndef UTEST_GROUPED_MATMUL_CASE_H
#define UTEST_GROUPED_MATMUL_CASE_H

#include <exe_graph/runtime/tiling_context.h>
#include <register/op_impl_registry.h>

#include "tests/utils/case.h"
#include "tests/utils/tensor.h"
#include "tests/utils/context.h"
#include "tests/utils/context_with_template_tilingkey.h"
#include "tests/utils/op_info.h"
#include "grouped_matmul_param.h"

namespace ops::adv::tests::grouped_matmul {

using ops::adv::tests::grouped_matmul::Param;
using ops::adv::tests::utils::Context;
using ops::adv::tests::utils::OpInfo;

#define GMM_INPUT_DTYPE                                                \
    uint8_t *, uint8_t *, uint8_t *, uint8_t *,                        \
    uint8_t *, uint8_t *, uint8_t *, uint8_t *,                        \
    uint8_t *, uint8_t *, uint8_t *, uint8_t *

class GroupedMatmulCase : public ops::adv::tests::utils::Case {
public:
    using ContextWithTemplateTilingKey = ops::adv::tests::utils::ContextWithTemplateTilingKey<GMM_INPUT_DTYPE>;
    OpInfo mOpInfo;
    // Context mCtx;
    ContextWithTemplateTilingKey mCtx;
    Param mParam;

public:
    GroupedMatmulCase();
    GroupedMatmulCase(const char *name, bool enable, const char *dbgInfo, OpInfo opInfo, Param param,
                      int32_t tilingTemplatePriority);

    bool Run() override;

protected:
    bool InitParam() override;
    bool InitOpInfo() override;
    bool InitCurrentCasePtr() override;
};

} // namespace ops::adv::tests::grouped_matmul
#endif // UTEST_GROUPED_MATMUL_CASE_H