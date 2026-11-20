/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_COMM_PACKET_ARCH92_H
#define MEGA_MOE_COMM_PACKET_ARCH92_H

#include <type_traits>

#include "../../arch35_92/common/mega_moe_copy.h"
#include "mega_moe_layout_arch92.h"

namespace MegaMoeImpl {
namespace CommPacketArch92 {

#if MEGA_MOE_ENABLE_ARCH92_4K_PACKET
// 调用方只选择远端整行；接口前不检查地址或长度，不改变真实 payload。
template <typename Element>
__aicore__ inline void CopyRemoteCombine4K(__gm__ Element* dst, __ubuf__ Element* src, uint32_t elementCount)
{
    const uint32_t payloadBytes = elementCount * sizeof(Element);
    auto local = Arch92Copy::UbView(src, payloadBytes);
    auto global = Arch92Copy::GmView(dst, static_cast<uint8_t>(GmStoreCacheMode::NORMAL_FIRST_VICTIM),
                                     CopyRows{1U, payloadBytes, payloadBytes, payloadBytes});
    AscendC::RemoteDataCopyParams copyConfig;
    copyConfig.enableRSW = false;
    copyConfig.writeSize = AscendC::WritePacketSize::SIZE_4K_BYTES;
    copyConfig.enableRetry = false;
    AscendC::RemoteDataCopy(global, local, elementCount, copyConfig);
}
#endif

template <typename Element, typename Params>
__aicore__ inline void SendCombineRow(__gm__ Element* dst, __ubuf__ Element* src, uint32_t count, const Params& params,
                                      bool isRemote)
{
#if MEGA_MOE_ENABLE_ARCH92_4K_PACKET
    if constexpr (std::is_same_v<Element, bfloat16_t>) {
        if (params.tilingData->comm.combineRowAlignmentBytes != 0U && isRemote) {
            CopyRemoteCombine4K(dst, src, count);
            return;
        }
    }
#else
    (void)params;
    (void)isRemote;
#endif
    CopyUbToGm(dst, src, static_cast<uint32_t>(count * sizeof(Element)));
}

} // namespace CommPacketArch92
} // namespace MegaMoeImpl

#endif // MEGA_MOE_COMM_PACKET_ARCH92_H
