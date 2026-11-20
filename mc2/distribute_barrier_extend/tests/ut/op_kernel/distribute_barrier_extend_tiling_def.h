/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DISTRIBUTE_BARRIER_EXTEND_TILING_DEF_H
#define DISTRIBUTE_BARRIER_EXTEND_TILING_DEF_H

#include "securec.h"
#include "version/asc_devkit_version.h"
#include "basic_api/kernel_basic_intf.h"
#include "../../../../common/op_kernel/mc2_moe_context.h"
#if ASC_DEVKIT_VERSION_NUM >= MC2_BARRIER_HCOMM_MIN_VERSION
#include "adv_api/hcomm/hcomm.h"
#endif
#include "kernel_tiling/kernel_tiling.h"
#include "../../../../../tests/ut/framework_normal/common/hccl_stub.h"
#include "../../../../distribute_barrier/op_kernel/distribute_barrier_tiling.h"

namespace Mc2BarrierUt {
constexpr uint32_t FLAG_BYTES = 32U;
constexpr uint32_t HCOMM_BUFFER_BYTES = 512U;

enum class Transport {
    CTP,
    RTP,
    ROCE
};

enum class Failure {
    NONE,
    INIT,
    LOCK,
    WRITE,
    DRAIN,
    UNLOCK,
    STALL_DRAIN
};

struct CommState {
    uint32_t initCalls[2]{};
    uint32_t invalidCalls = 0;
    uint32_t drainEntered = 0;
    uint32_t trapCalls = 0;
    uint32_t trapCore = 0;
    uint32_t completed[2]{};
    Failure failure = Failure::NONE;
};

struct Channel {
    Transport protocol;
    uint8_t* source;
    uint8_t* destination[2];
    uint8_t* pending = nullptr;
    uint8_t payloads[2][FLAG_BYTES]{};
    uint32_t bank = 0;
    uint32_t writes[2]{};
    uint32_t drains[2]{};
    uint32_t locked = 0;
    uint32_t owner = 0;
    uint32_t locks = 0;
    uint32_t unlocks = 0;
};

extern CommState* g_commState;
int32_t Init(Transport protocol, uint32_t bytes);
int32_t Lock(Transport protocol, uint64_t handle);
int32_t Unlock(Transport protocol, uint64_t handle);
int32_t Write(Transport protocol, uint64_t handle, uint8_t* dst, uint8_t* src, uint64_t bytes);
int32_t Drain(Transport protocol, uint64_t handle);

#if ASC_DEVKIT_VERSION_NUM >= MC2_BARRIER_HCOMM_MIN_VERSION
template <Transport protocol>
class HcommStub {
public:
    template <typename T>
    int32_t Init(const AscendC::LocalTensor<T>&, uint32_t bytes)
    {
        const int32_t ret = Mc2BarrierUt::Init(protocol, bytes);
        initialized_ = ret == 0;
        return ret;
    }

    template <bool commit, auto commitPipe, auto reqPipe, auto const& config = AscendC::URMA_DEFAULT_CFG>
    int32_t WriteNbi(uint64_t handle, uint8_t* dst, uint8_t* src, uint64_t bytes)
    {
        return initialized_ ? Mc2BarrierUt::Write(protocol, handle, dst, src, bytes) : -1;
    }

    int32_t Drain(uint64_t handle)
    {
        return initialized_ ? Mc2BarrierUt::Drain(protocol, handle) : -1;
    }

    int32_t Lock(uint64_t handle)
    {
        static_assert(protocol == Transport::CTP, "Only URMA channels use Lock");
        return initialized_ ? Mc2BarrierUt::Lock(protocol, handle) : -1;
    }

    int32_t Unlock(uint64_t handle)
    {
        static_assert(protocol == Transport::CTP, "Only URMA channels use Unlock");
        return initialized_ ? Mc2BarrierUt::Unlock(protocol, handle) : -1;
    }

private:
    bool initialized_ = false;
};
#endif
} // namespace Mc2BarrierUt

#if ASC_DEVKIT_VERSION_NUM >= MC2_BARRIER_HCOMM_MIN_VERSION
namespace AscendC {
template <>
class Hcomm<COMM_PROTOCOL_UBC_CTP> : public Mc2BarrierUt::HcommStub<Mc2BarrierUt::Transport::CTP> {};
template <>
class Hcomm<COMM_PROTOCOL_ROCE> : public Mc2BarrierUt::HcommStub<Mc2BarrierUt::Transport::ROCE> {};
} // namespace AscendC
#endif

inline void InitDistributeBarrierTilingData(uint8_t* tiling, DistributeBarrierTilingData* constData)
{
    memcpy_s(constData, sizeof(DistributeBarrierTilingData), tiling, sizeof(DistributeBarrierTilingData));
}

#define GET_TILING_DATA_WITH_STRUCT(DistributeBarrierTilingData, tilingData, tilingArg) \
    DistributeBarrierTilingData tilingData; \
    InitDistributeBarrierTilingData(tilingArg, &tilingData)

#endif
