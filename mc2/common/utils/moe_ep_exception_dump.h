/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MOE_EP_EXCEPTION_DUMP_H
#define MOE_EP_EXCEPTION_DUMP_H

#include <array>
#include <cstring>
#include <string>
#include <vector>

#include "mc2_exception_dump.h"
#include "../op_kernel/moe_ep_exception_dump_defs.h"

#if RUNTIME_VERSION_NUM >= EXCEPTION_DUMP_SUPPORT_VERSION && METADEF_VERSION_NUM >= EXCEPTION_DUMP_SUPPORT_VERSION
namespace Mc2Exception {

inline void MoeEpExceptionImpl(aclrtExceptionInfo *args, void *userdata, const char *op)
{
    (void)userdata;
    if (args == nullptr || op == nullptr) {
        OP_LOGE(OP_NAME, "Moe EP exception args or op is null.");
        return;
    }

    auto getArgsFunc = GetAclrtGetArgsFromExceptionInfoFunc();
    if (getArgsFunc == nullptr) {
        OP_LOGE(OP_NAME, "Failed to load aclrtGetArgsFromExceptionInfo function.");
        return;
    }

    void *devArgsPtr = nullptr;
    uint32_t devArgsLen = 0U;
    aclError ret = getArgsFunc(args, &devArgsPtr, &devArgsLen);
    if (ret != ACL_SUCCESS) {
        OP_LOGE(OP_NAME, "aclrtGetArgsFromExceptionInfo failed for %s. ret=%d", op, ret);
        return;
    }

    uint64_t contextAddr = 0UL;
    ret = aclrtMemcpy(&contextAddr, sizeof(contextAddr), devArgsPtr, sizeof(contextAddr), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        OP_LOGE(OP_NAME, "aclrtMemcpy Moe EP context address failed for %s. ret=%d", op, ret);
        return;
    }

    Mc2Aclnn::MoeCommContext context{};
    ret = aclrtMemcpy(&context, sizeof(context), reinterpret_cast<void *>(contextAddr), sizeof(context),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        OP_LOGE(OP_NAME, "aclrtMemcpy MoeCommContext failed for %s. ret=%d", op, ret);
        return;
    }

    const uint32_t epRankId = context.epRankId;
    const uint64_t winBase = context.epHcclBuffer[epRankId];
    std::vector<uint8_t> metadataBuffer(MOE_EP_DUMP_METADATA_BYTES, 0U);
    ret = aclrtMemcpy(metadataBuffer.data(), metadataBuffer.size(), reinterpret_cast<void *>(winBase),
                      metadataBuffer.size(), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        OP_LOGE(OP_NAME, "aclrtMemcpy Moe EP metadata failed for %s rank %u. ret=%d", op, epRankId, ret);
        return;
    }

    MoeEpDumpMetadata metadata{};
    std::memcpy(&metadata, metadataBuffer.data(), sizeof(metadata));
    auto getDumpPathFunc = GetAcldumpGetPathFunc();
    if (getDumpPathFunc == nullptr) {
        OP_LOGE(OP_NAME, "Failed to load acldumpGetPath function for %s.", op);
        return;
    }
    const char *dumpPath = getDumpPathFunc(acldumpType::AIC_ERR_BRIEF_DUMP);
    if (dumpPath == nullptr) {
        OP_LOGE(OP_NAME, "acldumpGetPath returned null for %s.", op);
        return;
    }

    const uint32_t deviceId = aclrtGetDeviceIdFromExceptionInfo(args);
    const std::string filePrefix = GenDumpFileName(args, op) + "." + std::to_string(epRankId);
    if (DumpToFile(dumpPath, filePrefix + ".metadata.bin", deviceId, metadataBuffer.data(), metadataBuffer.size()) !=
        0) {
        OP_LOGE(OP_NAME, "Failed to write Moe EP metadata for %s rank %u.", op, epRankId);
        return;
    }

    static constexpr std::array<const char *, MOE_EP_DUMP_REGION_COUNT> REGION_NAMES = {
        "per_core_diag",       "count_notify",        "expert_count",
        "dispatch_slot_state", "combine_token_state", "hybrid_scaleout_status"};
    for (uint32_t regionIndex = 0U; regionIndex < MOE_EP_DUMP_REGION_COUNT; ++regionIndex) {
        const MoeEpDumpRegion &region = metadata.regions[regionIndex];
        if (region.size == 0UL) {
            continue;
        }
        std::vector<uint8_t> regionBuffer(region.size, 0U);
        ret = aclrtMemcpy(regionBuffer.data(), regionBuffer.size(), reinterpret_cast<void *>(winBase + region.offset),
                          regionBuffer.size(), ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            OP_LOGE(OP_NAME, "aclrtMemcpy Moe EP region %s failed for %s rank %u. ret=%d", REGION_NAMES[regionIndex],
                    op, epRankId, ret);
            return;
        }
        if (DumpToFile(dumpPath, filePrefix + "." + REGION_NAMES[regionIndex] + ".bin", deviceId, regionBuffer.data(),
                       regionBuffer.size()) != 0) {
            OP_LOGE(OP_NAME, "Failed to write Moe EP region %s for %s rank %u.", REGION_NAMES[regionIndex], op,
                    epRankId);
            return;
        }
    }
}

} // namespace Mc2Exception
#endif

#endif // MOE_EP_EXCEPTION_DUMP_H
