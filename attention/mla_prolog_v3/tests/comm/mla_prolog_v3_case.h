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
 * \file mla_prolog_v3_case.h
 * \brief MlaPrologV3 测试用例.
 */

#ifndef MLA_PROLOG_V3_CASE_H
#define MLA_PROLOG_V3_CASE_H

#include <vector>
#include <utility>
#include <exe_graph/runtime/tiling_context.h>
#include <register/op_impl_registry.h>
#include "tests/utils/case.h"
#include "tests/utils/op_info.h"
#include "tests/utils/context.h"
#include "mla_prolog_v3_param.h"


namespace ops::adv::tests::MlaPrologV3 {

/**
 * 以下参数宏声明要与 MlaPrologV3 Kernel 入口函数保持一致.
 */
#define MLA_PROLOG_V3_KERNEL_PARAM                                                                                               \
    (uint8_t *tokenX, uint8_t *weightDq,    \
        uint8_t *weightUqQr, uint8_t *weightUk,    \
        uint8_t *weightDkvKr, uint8_t *rmsnormGammaCq,    \
        uint8_t *rmsnormGammaCkv, uint8_t *ropeSin,    \
        uint8_t *ropeCos, uint8_t *cacheIndex,    \
        uint8_t *kvCache, uint8_t *krCache,    \
        uint8_t *dequantScaleX, uint8_t *dequantScaleWDq,    \
        uint8_t *dequantScaleWUqQr, uint8_t *dequantScaleWDkvKr,    \
        uint8_t *quantScaleCkv, uint8_t *quantScaleCkr,    \
        uint8_t *smoothScalesCq,   uint8_t *actualSeqLen, \
        uint8_t *query, uint8_t *queryRope,    \
        uint8_t *kvCacheOut, uint8_t *krCacheOut, uint8_t *dequantScaleQNopeOut,    \
        uint8_t *queryNormOut, uint8_t *dequantScaleQNormOut,    \
        uint8_t *workspace, uint8_t *tiling)

/**
 * 算子 MlaPrologV3 参数
 */
class MlaPrologV3Case : public ops::adv::tests::utils::Case {
public:
    using OpInfo = ops::adv::tests::utils::OpInfo;
    using Context = ops::adv::tests::utils::Context;
    using MlaPrologV3Param = ops::adv::tests::MlaPrologV3::MlaPrologV3Param;

    typedef void(*MlaPrologV3KernelFunc) MLA_PROLOG_V3_KERNEL_PARAM;

    class DoTilingParam {
    public:
        gert::TilingContext *ctx = nullptr;
        ge::graphStatus ret = ge::GRAPH_SUCCESS;
        gert::Tensor *prefixTensor = nullptr;
        gert::Tensor *actSeqQLenTensor = nullptr;
        gert::Tensor *actSeqKVLenTensor = nullptr;
        gert::Tensor *qStartIdxTensor = nullptr;
        gert::Tensor *kvStartIdxTensor = nullptr;
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
    MlaPrologV3Param mParam;

    gert::OpImplRegisterV2::TilingKernelFunc mMlaPrologV3OriginTilingFunc = nullptr;
    PreTilingRunCbf mPreTilingRunCbf = nullptr;

public:
    MlaPrologV3Case();
    MlaPrologV3Case(const char *name, bool enable, const char *dbgInfo, OpInfo prompt, MlaPrologV3Param param,
           int32_t tilingTemplatePriority = kTilingTemplatePriority_Invalid);

    bool Run() override;
    bool DoOpTiling(DoTilingParam &tilingParam);

    static void PreTilingRunCbf_SetPlatformInfoNull(MlaPrologV3Case::DoTilingParam &tilingParam);

protected:
    std::string mMlaPrologV3OriginTilingFuncName;
    void *mMlaPrologV3KernelFunc = nullptr;

protected:
    bool InitParam() override;
    bool InitOpInfo() override;
    bool InitOpInfoCtx();
    bool InitOriginTilingFunc();
    bool InitCurrentCasePtr() override;
};

} // namespace ops::adv::tests::MlaPrologV3

#endif