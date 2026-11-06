/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MC2_MOE_SHARED_JETTY_ADAPTER_H
#define MC2_MOE_SHARED_JETTY_ADAPTER_H

#include <cstdint>
#include <string>

#include "hccl/hccl_res.h"

namespace at {
class Tensor;
}

namespace Mc2Api {
constexpr char MOE_COMM_ALG_DEFAULT[] = "";
constexpr char MOE_COMM_ALG_AUTO[] = "auto";
constexpr char MOE_COMM_ALG_CLOS_ONLY[] = "clos_only";
constexpr int64_t MOE_CLOS_ONLY_MIN_RANK_SIZE = 64;
// Internal compatibility aliases. The public comm_alg names are auto and clos_only.
constexpr const char* MOE_COMM_ALG_FULLMESH = MOE_COMM_ALG_AUTO;
constexpr const char* MOE_COMM_ALG_SHARED_JETTY = MOE_COMM_ALG_CLOS_ONLY;
constexpr int64_t MOE_SHARED_JETTY_MIN_RANK_SIZE = MOE_CLOS_ONLY_MIN_RANK_SIZE;

at::Tensor BuildMoeSharedJettyContext(const std::string& groupName, uint64_t requiredBufferSize);

HcclResult CreateMoeMultiChannelHandle(HcclComm comm, const char* queueTag, const HcclChannelDesc* descs,
                                       uint32_t channelNum, uint64_t* handle);
} // namespace Mc2Api

#endif // MC2_MOE_SHARED_JETTY_ADAPTER_H
