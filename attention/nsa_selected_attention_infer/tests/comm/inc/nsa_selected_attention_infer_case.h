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
 * \file nsa_selected_attention_infer_case.h
 * \brief NsaSelectedAttentionInfer 测试用例.
 */

#ifndef UTEST_NSA_SELECT_ATTENTION_INFER_CASE_H
#define UTEST_NSA_SELECT_ATTENTION_INFER_CASE_H

#include <vector>
#include <exe_graph/runtime/tiling_context.h>
#include <register/op_impl_registry.h>
#include "tests/utils/case.h"
#include "tests/utils/op_info.h"
#include "tests/utils/context.h"
#include "nsa_selected_attention_infer_param.h"

#define NSA_SELECT_ATTENTION_INFER_KERNEL_PARAM                                                                           \
    (uint8_t *query, uint8_t *key, uint8_t *value, uint8_t *topkIndices, uint8_t *attenMask,             \
    uint8_t *blockTable, uint8_t *actualQSeqLengths, uint8_t *actualKVSeqLengths, uint8_t *attentionOut, uint8_t *workspace, uint8_t *tiling)

namespace ops::adv::tests::NsaSelectedAttentionInfer {
class NsaSelectAttentionInferCase : public ops::adv::tests::utils::Case {
public:
    using OpInfo = ops::adv::tests::utils::OpInfo;
    using Context = ops::adv::tests::utils::Context;
    using Tensor = ops::adv::tests::utils::Tensor;

    typedef void(*NsaSelectAttentionInferKernelFunc) NSA_SELECT_ATTENTION_INFER_KERNEL_PARAM;

    class DoTilingParam {
    public:
        gert::TilingContext *ctx = nullptr;
        ge::graphStatus ret = ge::GRAPH_SUCCESS;
        gert::Tensor *actSeqSelQLenTensor = nullptr;
        gert::Tensor *actSeqSelKVLenTensor = nullptr;
        gert::Tensor *blocktableTensor = nullptr;
        gert::Tensor *topkIndicesTensor = nullptr;
    };
    /**
     * 执行 Tiling 前, 修改 TilingContext 回调函数.
     *
     * \attention: 一般用于异常用例.
     */
    typedef void (*PreTilingRunCbf)(DoTilingParam &tilingParam);

public:
    /* 算子控制信息 */
    OpInfo mOpInfo;
    Context mCtx;

    /* 输入/输出 参数 */
    NsaSelectAttentionInferParam mParam;

    gert::OpImplRegisterV2::TilingKernelFunc nsaSelectAttentionInferTilingFunc = nullptr;
    PreTilingRunCbf mPreTilingRunCbf = nullptr;

    NsaSelectAttentionInferCase();
    NsaSelectAttentionInferCase(const char *name, bool enable, const char *dbgInfo, OpInfo incre,
                             NsaSelectAttentionInferParam param,
                             int32_t tilingTemplatePriority = kTilingTemplatePriority_Invalid);
    bool Run() override;
    bool DoOpTiling(DoTilingParam &tilingParam);

    static void PreTilingRunCbf_SetPlatformInfoNull(NsaSelectAttentionInferCase::DoTilingParam &tilingParam);

protected:
    std::string mNsaSelectAttentionInferOriginTilingFuncName;
    void *mNsaSelectAttentionInferKernelFunc = nullptr;

    bool InitParam() override;
    bool InitOpInfo() override;
    bool InitOpInfoCtx();
    bool InitOriginTilingFunc();
    bool InitCurrentCasePtr() override;
};

}
#endif // UTEST_NSA_SELECT_ATTENTION_INFER_CASE_H
