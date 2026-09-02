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
 * \file nsa_compress_with_cache_case.h
 * \brief NsaCompressWithCache 测试用例.
 */

#ifndef UTEST_NSA_COMPRESS_WITH_CACHE_CASE_H
#define UTEST_NSA_COMPRESS_WITH_CACHE_CASE_H

#include <vector>
#include <exe_graph/runtime/tiling_context.h>
#include <register/op_impl_registry.h>
#include "tests/utils/case.h"
#include "tests/utils/op_info.h"
#include "tests/utils/context.h"
#include "nsa_compress_with_cache_param.h"

#define NSA_COMPRESS_WITH_CACHE_KERNEL_PARAM \
    (uint8_t * input, uint8_t * weight, uint8_t * slotMapping, uint8_t * output_cache, uint8_t * act_seq_len, \
     uint8_t * block_table, uint8_t * output_cache_ref, uint8_t * workspace, uint8_t * tiling)

namespace ops::adv::tests::NsaCompressWithCache {
class NsaCompressWithCacheCase : public ops::adv::tests::utils::Case {
public:
    using OpInfo = ops::adv::tests::utils::OpInfo;
    using Context = ops::adv::tests::utils::Context;
    using Tensor = ops::adv::tests::utils::Tensor;

    typedef void(*NsaCompressWithCacheKernelFunc) NSA_COMPRESS_WITH_CACHE_KERNEL_PARAM;

    class DoTilingParam {
    public:
        gert::TilingContext *ctx = nullptr;
        ge::graphStatus ret = ge::GRAPH_SUCCESS;
        gert::Tensor *slotMappingTensor = nullptr;
        gert::Tensor *actSeqLenTensor = nullptr;
        gert::Tensor *blockTableTensor = nullptr;
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
    NsaCompressWithCacheParam mParam;

    gert::OpImplRegisterV2::TilingKernelFunc mNsaCompressWithCacheOriginTilingFunc = nullptr;
    PreTilingRunCbf mPreTilingRunCbf = nullptr;

    NsaCompressWithCacheCase();
    NsaCompressWithCacheCase(const char *name, bool enable, const char *dbgInfo, OpInfo incre,
                             NsaCompressWithCacheParam param,
                             int32_t tilingTemplatePriority = kTilingTemplatePriority_Invalid);
    bool Run() override;
    bool DoOpTiling(DoTilingParam &tilingParam);

    static void PreTilingRunCbf_SetPlatformInfoNull(NsaCompressWithCacheCase::DoTilingParam &tilingParam);

protected:
    std::string mNsaCompressWithCacheOriginTilingFuncName;
    void *mNsaCompressWithCacheKernelFunc = nullptr;

    bool InitParam() override;
    bool InitOpInfo() override;
    bool InitOpInfoCtx();
    bool InitOriginTilingFunc();
    bool InitCurrentCasePtr() override;
};

} // namespace ops::adv::tests::NsaCompressWithCache
#endif // UTEST_NSA_COMPRESS_WITH_CACHE_CASE_H
