/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MOE_EP_COMM_ALG_H
#define MOE_EP_COMM_ALG_H

#include <cstdint>
#include <string>

#include "../op_kernel/moe_ep_comm_alg_def.h"

namespace Mc2Aclnn {

constexpr char MOE_EP_COMM_ALG_AUTO_NAME[] = "auto";
constexpr char MOE_EP_COMM_ALG_CLOS_ONLY_NAME[] = "clos_only";
constexpr int64_t MOE_EP_CLOS_ONLY_MIN_RANK_SIZE = 64;
constexpr int64_t MOE_EP_SHARED_JETTY_MIN_RANK_SIZE = MOE_EP_CLOS_ONLY_MIN_RANK_SIZE;

inline bool MoeEpCommAlgEquals(const char* lhs, const char* rhs)
{
    if (lhs == nullptr || rhs == nullptr) {
        return false;
    }
    return std::string(lhs) == std::string(rhs);
}

inline bool IsValidMoeEpCommAlg(const char* commAlg)
{
    return commAlg == nullptr || *commAlg == '\0' || MoeEpCommAlgEquals(commAlg, MOE_EP_COMM_ALG_AUTO_NAME) ||
           MoeEpCommAlgEquals(commAlg, MOE_EP_COMM_ALG_CLOS_ONLY_NAME);
}

inline bool IsMoeEpClosOnly(const char* commAlg)
{
    return MoeEpCommAlgEquals(commAlg, MOE_EP_COMM_ALG_CLOS_ONLY_NAME);
}

inline uint32_t ParseMoeEpCommAlg(const char* commAlg)
{
    return IsMoeEpClosOnly(commAlg) ? static_cast<uint32_t>(Mc2Tiling::MOE_EP_COMM_ALG_CLOS_ONLY) :
                                      static_cast<uint32_t>(Mc2Tiling::MOE_EP_COMM_ALG_AUTO);
}

} // namespace Mc2Aclnn

#endif // MOE_EP_COMM_ALG_H
