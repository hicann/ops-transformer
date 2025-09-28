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
 * \file nsa_selected_attention_case.h
 * \brief NsaSelectedAttention 测试用例.
 */

#ifndef UTEST_NSA_SELECTED_ATTENTION_CASE_H
#define UTEST_NSA_SELECTED_ATTENTION_CASE_H

#include <exe_graph/runtime/tiling_context.h>
#include <register/op_impl_registry.h>
#include "tests/utils/case.h"
#include "tests/utils/tensor.h"
#include "tests/utils/context.h"
#include "tests/utils/op_info.h"
#include "nsa_selected_attention_param.h"


namespace ops::adv::tests::nsa_selected_attention_ns {

using ops::adv::tests::nsa_selected_attention_ns::Param;
using ops::adv::tests::utils::Context;
using ops::adv::tests::utils::OpInfo;

class NsaSelectedAttentionCase : public ops::adv::tests::utils::Case {
public:
    OpInfo mOpInfo;
    Context mCtx;
    Param mParam;
    gert::OpImplRegisterV2::TilingKernelFunc tilingKernelFunc = nullptr;

    class DoTilingParam {
    public:
        gert::TilingContext *ctx = nullptr;
        ge::graphStatus ret = ge::GRAPH_SUCCESS;
    };

public:
    NsaSelectedAttentionCase();
    NsaSelectedAttentionCase(const char *name, bool enable, const char *dbgInfo, OpInfo opInfo, Param param);

    bool Run() override;
    bool DoOpTiling(DoTilingParam &tilingParam);

protected:
    bool InitParam() override;
    bool InitOpInfo() override;
    bool InitCurrentCasePtr() override;
};
} // namespace ops::adv::tests::nsa_selected_attention_ns
#endif // UTEST_NSA_SELECTED_ATTENTION_CASE_H