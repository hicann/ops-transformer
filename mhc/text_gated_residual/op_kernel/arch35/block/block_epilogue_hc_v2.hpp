/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or
 * modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS
 * SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT
 * NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of
 * the software repository for the full text of the License.
 */
#pragma once
#include "v2_native_experiment.hpp"
namespace Catlass::Epilogue {
// Native VF epilogue shared by all A5 large-shape keys.
template <class Element>
class BlockEpilogueHcV2 {
public:
    template <class Resource>
    __aicore__ inline void operator()(Resource& resource, GM_ADDR up, GM_ADDR xn, GM_ADDR mixed, uint32_t actualM,
                                      uint32_t channels, uint32_t hidden, uint32_t taskRows = 0)
    {
        const uint32_t subNum = AscendC::GetSubBlockNum();
        uint32_t grain = 1u;
        while ((static_cast<uint64_t>(hidden) * sizeof(Element) * grain) % 32u != 0u)
            grain *= 2u;
        const uint32_t splitRows = taskRows == 0 ? actualM : taskRows;
        const uint32_t groups = splitRows / grain + (splitRows % grain != 0u);
        const uint32_t subIdx = AscendC::GetSubBlockIdx();
        const uint32_t begin = (static_cast<uint64_t>(groups) * subIdx / subNum) * grain;
        const uint32_t limit = (static_cast<uint64_t>(groups) * (subIdx + 1u) / subNum) * grain;
        const uint32_t end = limit < actualM ? limit : actualM;
        if (begin < end) {
            V2NativeExperiment::Run<Element>(resource, up, xn, mixed, begin, end, channels, hidden);
        }
    }
};
} // namespace Catlass::Epilogue
