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
 * \file fa_case.h
 * \brief FlashAttentionScore / FlashAttentionScoreGrad 测试用例.
 */

#pragma once

#include <vector>
#include <functional>
#include <utility>
#include <tikicpulib.h>
#include <graph/utils/type_utils.h>
#include <exe_graph/runtime/tiling_context.h>
#include <register/op_impl_registry.h>
#include "tests/utils/case_with_socversion.h"
#include "tests/utils/op_info_with_socversion.h"
#include "tests/utils/context.h"
#include "tests/utils/context_with_template_tilingkey.h"
#include "tests/utils/log.h"
#include "tests/utils/platform.h"
#include "tiling/fa/tiling_data.h"
#include "tiling/fa/tiling_stub.h"
#include "fa_param.h"
#define __NPU_HOST__

/**
 * 以下参数宏声明要与 FA/FAG Kernel 入口函数保持一致.
 */
#define FAS_KERNEL_PARAM                                                                                               \
    (uint8_t * query, uint8_t * key, uint8_t * value, uint8_t * pse, uint8_t * dropMask, uint8_t * paddingMask,        \
     uint8_t * attenMask, uint8_t * prefix, uint8_t * actualSeqLengths, uint8_t * actualSeqLengthsKv,                  \
     uint8_t * qStartIdx, uint8_t * kvStartIdx, uint8_t * deqScaleQ, uint8_t * deqScaleK, uint8_t * deqScaleV,         \
     uint8_t * queryRope, uint8_t * keyRope,                                                                           \
     uint8_t * softmaxMax, uint8_t * softmaxSum, uint8_t * softmaxOut, uint8_t * attentionOut, uint8_t * workspace,    \
     uint8_t * tiling)

#define FAG_KERNEL_PARAM                                                                                               \
    (uint8_t * query, uint8_t * key, uint8_t * value, uint8_t * dy, uint8_t * pse_shift, uint8_t * drop_mask,          \
     uint8_t * padding_mask, uint8_t * atten_mask, uint8_t * softmax_max, uint8_t * softmax_sum, uint8_t * softmax_in, \
     uint8_t * attention_in, uint8_t * prefix, uint8_t * actual_seq_qlen, uint8_t * actual_seq_kvlen,                  \
     uint8_t * q_start_idx, uint8_t * kv_start_idx,                                                                    \
     uint8_t * deqScaleQ, uint8_t * deqScaleK, uint8_t * deqScaleV, uint8_t * deqScaleDy, uint8_t * deqScaleO,         \
     uint8_t * queryRope, uint8_t * keyRope,                                                                           \
     uint8_t * dq, uint8_t * dk, uint8_t * dv, uint8_t * dpse, uint8_t * dqRope, uint8_t * dkRope,                     \
     uint8_t * workspace, uint8_t * tiling_data)

#define FAG_KERNEL_PARAM_                                                                                              \
     uint8_t * query, uint8_t * key, uint8_t * value, uint8_t * dy, uint8_t * pse_shift, uint8_t * drop_mask,          \
     uint8_t * padding_mask, uint8_t * atten_mask, uint8_t * softmax_max, uint8_t * softmax_sum, uint8_t * softmax_in, \
     uint8_t * attention_in, uint8_t * prefix, uint8_t * actual_seq_qlen, uint8_t * actual_seq_kvlen,                  \
     uint8_t * q_start_idx, uint8_t * kv_start_idx,                                                                    \
     uint8_t * deqScaleQ, uint8_t * deqScaleK, uint8_t * deqScaleV, uint8_t * deqScaleDy, uint8_t * deqScaleO,         \
     uint8_t * queryRope, uint8_t * keyRope,                                                                           \
     uint8_t * dq, uint8_t * dk, uint8_t * dv, uint8_t * dpse, uint8_t * dqRope, uint8_t * dkRope,                     \
     uint8_t * workspace, uint8_t * tiling_data

#define FAG_INPUT_DTYPE                                                                                               \
    uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *,     \
    uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *,     \
    uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *,     \
    uint8_t *, uint8_t *

#define FAG_INPUT_PARAMS                                                                                               \
     query, key, value, dy, pse_shift, drop_mask, padding_mask, atten_mask, softmax_max, softmax_sum, softmax_in,      \
     attention_in, prefix, actual_seq_qlen, actual_seq_kvlen, q_start_idx, kv_start_idx, deqScaleQ, deqScaleK,         \
     deqScaleV, deqScaleDy, deqScaleO, queryRope, keyRope, dq, dk, dv, dpse, dqRope, dkRope, workspace, tiling_data

namespace ops::adv::tests::fa {

/**
 * 算子 FlashAttentionScore / FlashAttentionScoreGrad 参数
 */
class FaCase : public ops::adv::tests::utils::CaseWithSocversion {
public:
    using OpInfoWithSocversion = ops::adv::tests::utils::OpInfoWithSocversion;
    using Context = ops::adv::tests::utils::Context;
    using ContextWithTemplateTilingKey = ops::adv::tests::utils::ContextWithTemplateTilingKey<FAG_INPUT_DTYPE>;
    using FaParam = ops::adv::tests::fa::FaParam;

    typedef void(*FasKernelFunc) FAS_KERNEL_PARAM;
    typedef void(*FagKernelFunc) FAG_KERNEL_PARAM;

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
    OpInfoWithSocversion mForward;
    OpInfoWithSocversion mReverse;
    Context mForwardCtx;
    ContextWithTemplateTilingKey mReverseCtx;

    /* 输入/输出 参数 */
    FaParam mParam;

    gert::OpImplRegisterV2::TilingKernelFunc mFasOriginTilingFunc = nullptr;
    gert::OpImplRegisterV2::TilingKernelFunc mFagOriginTilingFunc = nullptr;
    PreTilingRunCbf mPreTilingRunCbf = nullptr;

public:
    FaCase();
    FaCase(const char *name, bool enable, const char *dbgInfo, OpInfoWithSocversion forward, OpInfoWithSocversion reverse, FaParam param,
           int32_t tilingTemplatePriority = kTilingTemplatePriority_Invalid);
    FaCase(const char *name, bool enable, const char *dbgInfo,
           const std::function<void(FAG_INPUT_DTYPE)>& templatekeyKernelFunc,
           OpInfoWithSocversion forward, OpInfoWithSocversion reverse, FaParam param,
           int32_t tilingTemplatePriority = kTilingTemplatePriority_Invalid);

    bool Run() override;
    bool DoOpTiling(DoTilingParam &tilingParam);

    static void PreTilingRunCbf_SetPlatformInfoNull(FaCase::DoTilingParam &tilingParam);

protected:
    std::string mFasOriginTilingFuncName;
    std::string mFagOriginTilingFuncName;
    void *mFasKernelFunc = nullptr;
    void *mFagKernelFunc = nullptr;
    std::function<void(FAG_INPUT_DTYPE)> mFagKernelTemplateFunc;

protected:
    bool InitParam() override;
    bool InitOpInfo() override;
    bool InitOpInfoCtx();
    bool InitOriginTilingFunc();
    bool InitCurrentCasePtr() override;
};

} // namespace ops::adv::tests::fa
