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
 * \file mc2_moe_context_holder.h
 * \brief 走 Mc2Aclnn::Mc2MoeContext 的 EP 上下文持有器。
 */
#ifndef MC2_MOE_CONTEXT_HOLDER_H
#define MC2_MOE_CONTEXT_HOLDER_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "moe_distribute_v2_constant.h"
#if __has_include("../common/mc2_moe_context.h")
#include "../common/mc2_moe_context.h"
#else
#include "../../common/op_kernel/mc2_moe_context.h"
#endif

namespace Mc2Kernel {

using namespace AscendC;

class Mc2MoeContextHolder {
public:
    __aicore__ inline Mc2MoeContextHolder(){};

    // mc2 ctx 路径无需 winsize 校验，Check 仅作为接口对齐的 no-op。
    // Mc2MoeContext 不携带 epWorldSize，由调用方从 tilingData 传入并暂存。
    __aicore__ inline void InitAndCheck(GM_ADDR ctxPtr, uint32_t epWorldSize, uint64_t /* totalWinSize */,
                                        TPipe * /* pipe */, GM_ADDR /* exceptionAddr */)
    {
        mc2Context_ = (__gm__ Mc2Aclnn::Mc2MoeContext *)ctxPtr;
        epWorldSize_ = epWorldSize;
    }

    __aicore__ inline void Check(uint64_t /* totalWinSize */, TPipe * /* pipe */, GM_ADDR /* exceptionAddr */) {}

    __aicore__ inline GM_ADDR GetStatusDataSpaceGm()
    {
        return (GM_ADDR)(mc2Context_->epHcclBuffer_[mc2Context_->epRankId]);
    }

    __aicore__ inline uint32_t GetEpRankId()
    {
        return mc2Context_->epRankId;
    }

    __aicore__ inline uint32_t GetEpWorldSize()
    {
        return epWorldSize_;
    }

    // 数据区基址在 epHcclBuffer 的 STATE_SIZE 之后；与 hccl helper 同语义。
    __aicore__ inline GM_ADDR GetWindAddrByRankId(int32_t rankId, int32_t /* curRankId */)
    {
        return (GM_ADDR)mc2Context_->epHcclBuffer_[rankId] + Mc2Kernel::STATE_SIZE;
    }

    // 状态区基址即 epHcclBuffer 起始地址。
    __aicore__ inline GM_ADDR GetWindStateAddrByRankId(int32_t rankId, int32_t /* curRankId */)
    {
        return (GM_ADDR)mc2Context_->epHcclBuffer_[rankId];
    }

private:
    __gm__ Mc2Aclnn::Mc2MoeContext *mc2Context_{nullptr};
    uint32_t epWorldSize_{0};
};

} // namespace Mc2Kernel

#endif // MC2_MOE_CONTEXT_HOLDER_H
