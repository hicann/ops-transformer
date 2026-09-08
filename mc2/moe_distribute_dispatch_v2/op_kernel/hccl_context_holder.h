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
 * \file hccl_context_holder.h
 * \brief 走 Mc2Kernel::HcclOpParam 的 EP 上下文持有器。
 */
#ifndef HCCL_CONTEXT_HOLDER_H
#define HCCL_CONTEXT_HOLDER_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "check_winsize.h"
#include "moe_distribute_v2_constant.h"
#include "../../common/op_kernel/moe_distribute_base.h"

namespace Mc2Kernel {

using namespace AscendC;

class HcclContextHolder {
public:
    __aicore__ inline HcclContextHolder(){};

    // 一次完成：保存 EP context + WinSize 越界校验。epWorldSize 仅供 mc2 路径暂存使用，hccl 路径忽略。
    __aicore__ inline void InitAndCheck(GM_ADDR ctxPtr, uint32_t /* epWorldSize */, uint64_t totalWinSize, TPipe *pipe,
                                        GM_ADDR exceptionAddr)
    {
        epContext_ = (__gm__ Mc2Kernel::HcclOpParam *)ctxPtr;
        Check(totalWinSize, pipe, exceptionAddr);
    }

    // 独立 check，便于调用方在某些场景仅做校验。
    __aicore__ inline void Check(uint64_t totalWinSize, TPipe *pipe, GM_ADDR exceptionAddr)
    {
        CheckWindowSize(totalWinSize, Mc2Kernel::GetWinSize(epContext_), pipe, exceptionAddr);
    }

    __aicore__ inline GM_ADDR GetStatusDataSpaceGm()
    {
        return Mc2Kernel::GetStatusDataSpaceGm(epContext_);
    }

    __aicore__ inline uint32_t GetEpRankId()
    {
        return Mc2Kernel::GetRankId(epContext_);
    }

    __aicore__ inline uint32_t GetEpWorldSize()
    {
        return Mc2Kernel::GetRankDim(epContext_);
    }

    __aicore__ inline GM_ADDR GetWindAddrByRankId(int32_t rankId, int32_t curRankId)
    {
        return Mc2Kernel::GetBaseWindAddrByRankId(epContext_, rankId, curRankId);
    }

    __aicore__ inline GM_ADDR GetWindStateAddrByRankId(int32_t rankId, int32_t curRankId)
    {
        return Mc2Kernel::GetBaseWindStateAddrByRankId(epContext_, rankId, curRankId);
    }

private:
    __gm__ Mc2Kernel::HcclOpParam *epContext_{nullptr};
};

} // namespace Mc2Kernel

#endif // HCCL_CONTEXT_HOLDER_H
