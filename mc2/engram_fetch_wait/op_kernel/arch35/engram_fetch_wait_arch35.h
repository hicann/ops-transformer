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
 * \file engram_fetch_wait_arch35.h
 * \brief engram_fetch_wait算子arch35 kernel实现
 */

#ifndef ENGRAM_FETCH_WAIT_ARCH35_H
#define ENGRAM_FETCH_WAIT_ARCH35_H

#if __has_include("version/asc_devkit_version.h") && __has_include("version/hcomm_version.h")
#include "version/asc_devkit_version.h"
#include "version/hcomm_version.h"

#if (ASC_DEVKIT_MAJOR > 9 || (ASC_DEVKIT_MAJOR == 9 && ASC_DEVKIT_MINOR > 0)) && \
    (HCOMM_MAJOR > 9 || (HCOMM_MAJOR == 9 && HCOMM_MINOR > 0))
#define ENABLE_ENGRAM_FETCH_WAIT_KERNEL
#endif

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#endif
#include "kernel_tiling/kernel_tiling.h"
#include "../engram_fetch_wait_tiling_data.h"
#if __has_include("../../engram_fetch/engram_fetch_utils.h")
#include "../../engram_fetch/engram_fetch_utils.h"
#else
#include "../../../engram_fetch/op_kernel/engram_fetch_utils.h"
#endif
#include "adv_api/hccl/hccl.h"
#if __has_include("adv_api/hcomm/hcomm.h")
#include "adv_api/hcomm/hcomm.h"
#endif

namespace Mc2Kernel {

#if defined(ENABLE_ENGRAM_FETCH_WAIT_KERNEL)

class EngramFetchWaitArch35 {
public:
    __aicore__ inline EngramFetchWaitArch35() = default;

    __aicore__ inline void Init(GM_ADDR commContext, GM_ADDR workspaceGM, AscendC::TPipe *pipe);

    __aicore__ inline void Process();

private:
    AscendC::TPipe *tpipe_{nullptr};
    __gm__ EngramCommContext *ctxPtr_{nullptr};
    uint32_t aivId_{0};
    uint32_t rankId_{0};
    uint32_t numRanks_{0};
    uint32_t channelsPerRank_{1};

    AscendC::TBuf<> hcommBuf_;
    AscendC::Hcomm<AscendC::COMM_PROTOCOL_UBC_CTP> hcomm_;
};

__aicore__ inline void EngramFetchWaitArch35::Init(GM_ADDR commContext, GM_ADDR workspaceGM, AscendC::TPipe *pipe)
{
    tpipe_ = pipe;
    aivId_ = AscendC::GetBlockIdx();
    (void)workspaceGM;

    ctxPtr_ = (__gm__ EngramCommContext *)commContext;
    rankId_ = ctxPtr_->rankId;
    numRanks_ = ctxPtr_->rankSize;
    channelsPerRank_ = (Mc2Kernel::HANDLE_ARRAY_SIZE + numRanks_ - 1U) / numRanks_;
    if (channelsPerRank_ == 0U) {
        channelsPerRank_ = 1U;
    }

    tpipe_->InitBuffer(hcommBuf_, Mc2Kernel::HCOMM_INIT_SIZE);
    AscendC::LocalTensor<uint8_t> hcommTensor = hcommBuf_.Get<uint8_t>();
    hcomm_.Init(hcommTensor, Mc2Kernel::HCOMM_INIT_SIZE);
}

__aicore__ inline void EngramFetchWaitArch35::Process()
{
    if ASCEND_IS_AIV {
        uint32_t totalBlocks = AscendC::GetBlockNum();
        if (totalBlocks >= numRanks_) {
            auto coreAssign = GetCoreAssignment(totalBlocks, aivId_, numRanks_);
            if (coreAssign.assignedRank != rankId_ && coreAssign.idxInRankGroup < channelsPerRank_) {
                uint32_t globalChannelIdx = coreAssign.assignedRank * channelsPerRank_ + coreAssign.idxInRankGroup;
                int32_t ret = hcomm_.Drain(ctxPtr_->hcommHandle[globalChannelIdx]);
                ascendc_assert(ret == 0, "Urma drain failed, ret=%d, globalChannelIdx=%u", ret, globalChannelIdx);
            }
        } else {
            uint32_t startRank = (aivId_ < numRanks_) ? aivId_ : rankId_;
            for (uint32_t ownerRank = startRank; ownerRank < numRanks_; ownerRank += totalBlocks) {
                if (ownerRank != rankId_) {
                    uint32_t globalChannelIdx = ownerRank * channelsPerRank_;
                    int32_t ret = hcomm_.Drain(ctxPtr_->hcommHandle[globalChannelIdx]);
                    ascendc_assert(ret == 0, "Urma drain failed, ret=%d, globalChannelIdx=%u", ret, globalChannelIdx);
                }
            }
        }
    }
}

#endif

} // namespace Mc2Kernel

#endif
