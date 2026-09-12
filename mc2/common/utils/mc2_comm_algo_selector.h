/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MC2_COMM_ALGO_SELECTOR_H
#define MC2_COMM_ALGO_SELECTOR_H

#include <cstdint>
#include <string>
#include "mc2_hcom_topo_info.h"

namespace Mc2Hcom {

constexpr uint32_t MAX_COMM_ALGO_CANDIDATES = 32;
constexpr uint32_t INVALID_TOPO_TYPE = 0xFFFFFFFFU;
constexpr uint32_t WILDCARD_TOPO_TYPE = 0xFFFFFFFEU;

struct CommAlgoEntry {
    uint8_t commEngine;
    uint32_t topoType;
    uint32_t minLayers;
    uint32_t maxLayers;
    uint64_t minBytes;
    uint64_t maxBytes;
    uint32_t minRankSize;
    uint32_t maxRankSize;
    int32_t priority;
    const char *algoName;
};

class Mc2CommAlgoSelector {
public:
    static std::string SelectAlgoName(const std::string &opName, const char *group, uint8_t commEngine,
                                      uint64_t commDataBytes, uint32_t rankSize, const CommAlgoEntry *entries,
                                      uint32_t entryCount, const std::string &defaultAlgoName);

private:
    static uint32_t GetL0TopoType(const char *group);
    static uint32_t GetLayerNum(const char *group);
};

} // namespace Mc2Hcom

#endif // MC2_COMM_ALGO_SELECTOR_H
