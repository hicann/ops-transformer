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
 * \file quant_grouped_matmul_inplace_add_case.h
 * \brief QuantGroupedMatmulInplaceAdd 测试用例.
 */

#ifndef UTEST_QUANT_QUANT_GROUPED_MATMUL_INPLACE_ADD_CASE_H
#define UTEST_QUANT_QUANT_GROUPED_MATMUL_INPLACE_ADD_CASE_H

#include <exe_graph/runtime/tiling_context.h>
#include <register/op_impl_registry.h>

#include "tests/utils/case.h"
#include "tests/utils/tensor.h"
#include "tests/utils/context.h"
#include "tests/utils/op_info.h"
#include "quant_grouped_matmul_inplace_add_param.h"

namespace ops::adv::tests::quant_grouped_matmul_inplace_add {

using ops::adv::tests::quant_grouped_matmul_inplace_add::Param;
using ops::adv::tests::utils::Context;
using ops::adv::tests::utils::OpInfo;

class QuantGroupedMatmulInplaceAddCase : public ops::adv::tests::utils::Case {
public:
    OpInfo mOpInfo;
    Context mCtx;
    Param mParam;

public:
    QuantGroupedMatmulInplaceAddCase();
    QuantGroupedMatmulInplaceAddCase(const char *name, bool enable, const char *dbgInfo, OpInfo opInfo, Param param,
                                     int32_t tilingTemplatePriority);

    bool Run() override;

protected:
    bool InitParam() override;
    bool InitOpInfo() override;
    bool InitCurrentCasePtr() override;
};

} // namespace ops::adv::tests::quant_grouped_matmul_inplace_add
#endif // UTEST_QUANT_QUANT_GROUPED_MATMUL_INPLACE_ADD_CASE_H
