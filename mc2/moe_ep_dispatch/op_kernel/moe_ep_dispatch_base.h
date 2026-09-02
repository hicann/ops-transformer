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
 * \file moe_ep_dispatch_base.h
 * \brief
 */

#ifndef MOE_EP_DISPATCH_BASE_H
#define MOE_EP_DISPATCH_BASE_H

#if __has_include("version/asc_devkit_version.h") && __has_include("version/hcomm_version.h")
#include "version/asc_devkit_version.h"
#include "version/hcomm_version.h"

#if (ASC_DEVKIT_MAJOR > 9 || (ASC_DEVKIT_MAJOR == 9 && ASC_DEVKIT_MINOR > 0)) && \
    (HCOMM_MAJOR > 9 || (HCOMM_MAJOR == 9 && HCOMM_MINOR > 0))
#define ENABLE_MOE_EP_KERNEL
#endif

#endif

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#if __has_include("../common/mc2_moe_context.h")
#include "../common/mc2_moe_context.h"
#include "../common/mc2_kernel_utils.h"
#else
#include "../../common/op_kernel/mc2_moe_context.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"
#endif

namespace MoeEpDispatchBase {

#if defined(ENABLE_MOE_EP_KERNEL)

using namespace AscendC;

// PerExpertCnt Calculation
template <Reg::HistogramsType htype, typename T, typename U>
__simd_vf__ __aicore__ inline void HistogramsVf(__ubuf__ U *dst, __ubuf__ T *src, uint16_t repeatElm,
                                                uint16_t halfRepeat, uint32_t totalElm, uint16_t repeatTimes)
{
    Reg::RegTensor<T> srcReg;
    Reg::RegTensor<U> dst0Reg;
    Reg::RegTensor<U> dst1Reg;
    Reg::MaskReg pregOut = Reg::CreateMask<U>();
    Reg::Duplicate(dst0Reg, 0);
    Reg::Duplicate(dst1Reg, 0);
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        uint32_t remaining = totalElm - static_cast<uint32_t>(i) * repeatElm;
        uint32_t valid = (remaining > repeatElm) ? repeatElm : remaining;
        Reg::MaskReg preg = Reg::UpdateMask<T>(valid);
        Reg::LoadAlign(srcReg, src + repeatElm * i);
        Reg::Histograms<T, U, Reg::HistogramsBinType::BIN0, htype>(dst0Reg, srcReg, preg);
        Reg::Histograms<T, U, Reg::HistogramsBinType::BIN1, htype>(dst1Reg, srcReg, preg);
    }
    Reg::StoreAlign(dst, dst0Reg, pregOut);
    Reg::StoreAlign(dst + halfRepeat, dst1Reg, pregOut);
}

__aicore__ inline void GetExpertFreq(LocalTensor<uint16_t> &dstLocal, LocalTensor<uint8_t> &srcLocal, uint32_t totalElm)
{
    uint32_t repeatElm = GetVecLen();
    uint16_t repeatTimes = Ceil(totalElm, repeatElm);
    __ubuf__ uint8_t *src = (__ubuf__ uint8_t *)srcLocal.GetPhyAddr();
    __ubuf__ uint16_t *dst = (__ubuf__ uint16_t *)dstLocal.GetPhyAddr();
    asc_vf_call<HistogramsVf<Reg::HistogramsType::FREQUENCY, uint8_t, uint16_t>>(dst, src, repeatElm, repeatElm >> 1,
                                                                                 totalElm, repeatTimes);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline GM_ADDR GetWindowAddrByRankId(__gm__ Mc2Aclnn::MoeCommContext *context, uint32_t rankId,
                                                uint64_t offset)
{
    return (GM_ADDR)context->epHcclBuffer[rankId] + offset;
}

__aicore__ inline uint64_t GetCommHandle(__gm__ Mc2Aclnn::MoeCommContext *context, uint32_t rankId,
                                         uint32_t channelIndex = 0U)
{
    uint32_t channelsPerRank = context->channelsPerRank == 0U ? 1U : context->channelsPerRank;
    return context->hcommHandle[rankId * channelsPerRank + channelIndex];
}

// For Hybrid
__aicore__ inline uint32_t GetCurrentServerIndex(uint32_t epRankId, uint32_t rankNumPerServer)
{
    return rankNumPerServer == 0U ? 0U : epRankId / rankNumPerServer;
}

__aicore__ inline uint32_t GetServerStartRank(uint32_t epRankId, uint32_t rankNumPerServer)
{
    return GetCurrentServerIndex(epRankId, rankNumPerServer) * rankNumPerServer;
}

__aicore__ inline uint32_t GetServerEndRank(uint32_t epRankId, uint32_t rankNumPerServer, uint32_t epWorldSize)
{
    uint32_t serverEndRank = GetServerStartRank(epRankId, rankNumPerServer) + rankNumPerServer;
    return serverEndRank > epWorldSize ? epWorldSize : serverEndRank;
}

#endif

} // namespace MoeEpDispatchBase

namespace Mc2Kernel {

#if defined(ENABLE_MOE_EP_KERNEL)

using MoeEpDispatchBase::GetCommHandle;
using MoeEpDispatchBase::GetExpertFreq;

__aicore__ inline GM_ADDR GetWinAddrByRankId(__gm__ Mc2Aclnn::MoeCommContext *context, uint32_t rankId, uint64_t offset)
{
    return MoeEpDispatchBase::GetWindowAddrByRankId(context, rankId, offset);
}

#endif

} // namespace Mc2Kernel

#endif // MOE_EP_DISPATCH_BASE_H
