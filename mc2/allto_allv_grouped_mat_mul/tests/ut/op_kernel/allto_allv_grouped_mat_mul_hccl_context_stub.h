/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#ifndef ALLTO_ALLV_GROUPED_MAT_MUL_HCCL_CONTEXT_STUB_H
#define ALLTO_ALLV_GROUPED_MAT_MUL_HCCL_CONTEXT_STUB_H

#include <cstdint>

// framework_normal replaces the public HCCL header with hccl_stub.h. Keep the
// context ABI fields used by PeerWindowContext available to the CPU kernel UT.
namespace AscendC {

struct HcclCombineOpParam {
    uint64_t workSpace;
    uint64_t workSpaceSize;
    uint32_t rankId;
    uint32_t rankNum;
    uint64_t winSize;
    uint64_t windowsIn[HCCL_MAX_RANK_NUM];
    uint64_t windowsOut[HCCL_MAX_RANK_NUM];
    // Mirror the arch22 tail from CANN 9.2 hccl_inner_def.h. The production
    // kernel must be compilable against both fixed and runtime dynamic modes.
    uint8_t res[8328];
    uint8_t multiFlag;
    IbVerbsData *data;
};

namespace HcclContextDef {

struct HcclRankRelationResV2 {
    uint32_t remoteUsrRankId;
    uint32_t remoteWorldRank;
    uint64_t windowsIn;
    uint64_t windowsOut;
    uint64_t windowsExp;
};

struct RemoteResPtr {
    HcclRankRelationResV2 *nextHostPtr;
    HcclRankRelationResV2 *nextDevicePtr;
};

struct HcclOpResParam {
    uint64_t workSpace;
    uint64_t workSpaceSize;
    uint32_t rankId;
    uint32_t rankNum;
    uint64_t winSize;
    uint64_t localWindowsIn;
    uint64_t localWindowsOut;
    char hcomId[128];
    uint64_t winExpSize;
    uint64_t localWindowsExp;
    uint32_t rWinStart;
};

} // namespace HcclContextDef

inline HcclContextDef::HcclRankRelationResV2 *GetRemoteRankAddrs(HcclContextDef::HcclOpResParam *context,
                                                                 uint32_t rankId)
{
    auto *remoteRes =
        reinterpret_cast<HcclContextDef::RemoteResPtr *>(reinterpret_cast<uintptr_t>(context) + context->rWinStart);
    return remoteRes[rankId].nextDevicePtr;
}

} // namespace AscendC

#endif // ALLTO_ALLV_GROUPED_MAT_MUL_HCCL_CONTEXT_STUB_H
