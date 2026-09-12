/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mc2_comm_algo_selector.h"
#include "mc2_log_compat.h"
#include <algorithm>
#include <vector>

namespace Mc2Hcom {

uint32_t Mc2CommAlgoSelector::GetL0TopoType(const char *group)
{
    uint32_t topoType = INVALID_TOPO_TYPE;
    HcclResult ret = MC2HcomTopology::TryGetGroupTopoType(group, &topoType);
    if (ret != HCCL_SUCCESS) {
        OP_LOGW("", "[CommAlgoSelector] TryGetGroupTopoType failed, ret=%d", static_cast<int>(ret));
        return INVALID_TOPO_TYPE;
    }
    return topoType;
}

uint32_t Mc2CommAlgoSelector::GetLayerNum(const char *group)
{
    std::vector<uint32_t> layers;
    HcclResult ret = MC2HcomTopology::CommGetNetLayersByGroup(group, layers);
    if (ret != HCCL_SUCCESS) {
        OP_LOGW("", "[CommAlgoSelector] CommGetNetLayersByGroup failed, ret=%d", static_cast<int>(ret));
        return 0;
    }
    return static_cast<uint32_t>(layers.size());
}

std::string Mc2CommAlgoSelector::SelectAlgoName(const std::string &opName, const char *group, uint8_t commEngine,
                                                uint64_t commDataBytes, uint32_t rankSize, const CommAlgoEntry *entries,
                                                uint32_t entryCount, const std::string &defaultAlgoName)
{
    if (group == nullptr || entries == nullptr || entryCount == 0) {
        OP_LOGW(opName.c_str(), "[CommAlgoSelector] invalid input, fallback to default: %s", defaultAlgoName.c_str());
        return defaultAlgoName;
    }

    uint32_t n = entryCount;
    if (n > MAX_COMM_ALGO_CANDIDATES) {
        OP_LOGW(opName.c_str(), "[CommAlgoSelector] entries size %u exceeds max %u, truncated", n,
                MAX_COMM_ALGO_CANDIDATES);
        n = MAX_COMM_ALGO_CANDIDATES;
    }

    uint32_t topoType = GetL0TopoType(group);
    uint32_t layerNum = GetLayerNum(group);
    OP_LOGI(opName.c_str(),
            "[CommAlgoSelector] group=%s, topoType=%u(0x%x), layerNum=%u, commEngine=%u, "
            "commDataBytes=%llu, rankSize=%u, entries=%u",
            group, topoType, topoType, layerNum, commEngine, commDataBytes, rankSize, n);

    const CommAlgoEntry *buf[2][MAX_COMM_ALGO_CANDIDATES];
    uint32_t cnt[2] = {n, 0};
    uint32_t src = 0;
    uint32_t dst = 1;
    for (uint32_t i = 0; i < n; i++) {
        buf[0][i] = &entries[i];
    }

    auto applyFilter = [&](const char *name, auto &&pred) {
        if (cnt[src] == 0) {
            return;
        }
        cnt[dst] = 0;
        for (uint32_t i = 0; i < cnt[src]; i++) {
            if (pred(buf[src][i])) {
                buf[dst][cnt[dst]++] = buf[src][i];
            }
        }
        if (cnt[dst] == 0) {
            OP_LOGI(opName.c_str(), "[CommAlgoSelector] filter '%s' removed all %u candidates, skipping this filter",
                    name, cnt[src]);
            return;
        }
        std::swap(src, dst);
        OP_LOGD(opName.c_str(), "[CommAlgoSelector] after '%s': %u candidates remain", name, cnt[src]);
    };

    applyFilter("commEngine",
                [&](const CommAlgoEntry *e) { return e->commEngine == 0 || e->commEngine == commEngine; });
    applyFilter("rankSize",
                [&](const CommAlgoEntry *e) { return rankSize >= e->minRankSize && rankSize <= e->maxRankSize; });
    applyFilter("layers", [&](const CommAlgoEntry *e) {
        return layerNum == 0 || (e->minLayers <= layerNum && e->maxLayers >= layerNum);
    });
    applyFilter("topoType", [&](const CommAlgoEntry *e) {
        return topoType == INVALID_TOPO_TYPE || e->topoType == WILDCARD_TOPO_TYPE || e->topoType == topoType;
    });
    applyFilter("commDataBytes",
                [&](const CommAlgoEntry *e) { return commDataBytes >= e->minBytes && commDataBytes <= e->maxBytes; });

    if (cnt[src] == 0) {
        OP_LOGW(opName.c_str(), "[CommAlgoSelector] no candidate survived all filters, fallback to default: %s",
                defaultAlgoName.c_str());
        return defaultAlgoName;
    }

    const CommAlgoEntry *best = buf[src][0];
    for (uint32_t i = 1; i < cnt[src]; i++) {
        if (buf[src][i]->priority > best->priority) {
            best = buf[src][i];
        }
    }

    OP_LOGI(opName.c_str(), "[CommAlgoSelector] selected algo: %s (priority=%d, candidates=%u)", best->algoName,
            best->priority, cnt[src]);
    return best->algoName;
}

} // namespace Mc2Hcom
