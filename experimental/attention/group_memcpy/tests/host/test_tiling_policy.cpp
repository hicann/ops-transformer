/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "../../op_host/group_memcpy_tiling_policy.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

static void Require(bool condition)
{
    if (!condition) {
        throw std::runtime_error("group_memcpy tiling invariant violated");
    }
}
int main()
{
    using group_memcpy::TileBytes;
    Require(TileBytes(0, 184 * 1024) == 0);
    Require(TileBytes(1, 127) == 0);
    Require(TileBytes(UINT64_MAX, 184 * 1024) == 0);
    for (uint64_t ub : {uint64_t(128), uint64_t(1024), uint64_t(184 * 1024), uint64_t(192 * 1024)}) {
        const auto budget = std::min(ub, group_memcpy::UB_BUDGET);
        for (uint64_t width = 1; width < 600000; ++width) {
            const auto tile = TileBytes(width, ub);
            Require(tile > 0 && tile % 64 == 0 && tile * 2 <= budget);
            const auto tiles = width / tile + (width % tile != 0);
            const auto last = width - (tiles - 1) * tile;
            Require(last > 0 && last <= tile);
            Require((last + 63) / 64 * 64 <= tile);
        }
    }
    // Every real task is covered exactly once, including uneven tail cores.
    for (uint64_t tasks : {1, 39, 40, 41, 81, 257}) {
        for (uint64_t cores : {1, 8, 40, 48}) {
            const auto used = std::min(tasks, cores);
            std::vector<unsigned> visits(tasks, 0);
            for (uint64_t core = 0; core < used; ++core) {
                for (uint64_t task = core; task < tasks; task += used) {
                    ++visits.at(task);
                }
            }
            Require(std::all_of(visits.begin(), visits.end(), [](unsigned n) { return n == 1; }));
        }
    }
    std::cout << "tiling boundaries and task coverage passed\n";
}
