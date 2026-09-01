/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include "acl/acl.h"
#include "aclnnop/aclnn_gen_position_ids_from_mask.h"

#define CHECK(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "[FATAL] " << msg << std::endl; \
            return false; \
        } \
    } while (0)

namespace {

int g_passed = 0;
int g_failed = 0;

// ---- host 侧 golden ----
std::vector<int64_t> ComputeGolden(const std::vector<int32_t> &mask, int64_t B, int64_t S, int64_t fill)
{
    std::vector<int64_t> golden(static_cast<size_t>(B * S));
    for (int64_t r = 0; r < B; ++r) {
        int32_t running = 0;
        for (int64_t i = 0; i < S; ++i) {
            int32_t m = mask[static_cast<size_t>(r * S + i)];
            running += m;
            golden[static_cast<size_t>(r * S + i)] = (m == 0) ? fill : static_cast<int64_t>(running - 1);
        }
    }
    return golden;
}

// ---- 把 int32 mask 转成目标 dtype 的字节流 ----
// 返回设备侧输入字节数, 并填充 hostBytes
size_t MakeInputBytes(const std::vector<int32_t> &mask, aclDataType dtype, std::vector<uint8_t> &hostBytes)
{
    size_t n = mask.size();
    if (dtype == ACL_INT32) {
        hostBytes.resize(n * sizeof(int32_t));
        auto *p = reinterpret_cast<int32_t *>(hostBytes.data());
        for (size_t i = 0; i < n; ++i)
            p[i] = mask[i];
        return hostBytes.size();
    } else if (dtype == ACL_INT64) {
        hostBytes.resize(n * sizeof(int64_t));
        auto *p = reinterpret_cast<int64_t *>(hostBytes.data());
        for (size_t i = 0; i < n; ++i)
            p[i] = mask[i];
        return hostBytes.size();
    } else { // ACL_BOOL, 以 1 字节存
        hostBytes.resize(n * sizeof(uint8_t));
        auto *p = reinterpret_cast<uint8_t *>(hostBytes.data());
        for (size_t i = 0; i < n; ++i)
            p[i] = static_cast<uint8_t>(mask[i]);
        return hostBytes.size();
    }
}

const char *DtypeName(aclDataType dt)
{
    switch (dt) {
        case ACL_INT32:
            return "int32";
        case ACL_INT64:
            return "int64";
        case ACL_BOOL:
            return "bool ";
        default:
            return "?????";
    }
}

// ---- 单个用例 ----
// 仅 ACL/运行时失败时返回 false；数值不一致通过 g_failed 计数。
bool RunCase(const std::string &name, const std::vector<int32_t> &mask, int64_t B, int64_t S, int64_t fill,
             aclDataType inDtype, aclrtStream stream)
{
    std::vector<int64_t> shape = {B, S};
    std::vector<int64_t> golden = ComputeGolden(mask, B, S, fill);

    // 输入字节流
    std::vector<uint8_t> hostIn;
    size_t inBytes = MakeInputBytes(mask, inDtype, hostIn);
    size_t outBytes = static_cast<size_t>(B * S) * sizeof(int64_t);

    void *maskDev = nullptr;
    void *posDev = nullptr;
    CHECK(aclrtMalloc(&maskDev, inBytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, name + ": malloc mask failed");
    CHECK(aclrtMalloc(&posDev, outBytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, name + ": malloc pos failed");
    CHECK(aclrtMemcpy(maskDev, inBytes, hostIn.data(), inBytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS,
          name + ": memcpy H2D failed");

    aclTensor *maskT = aclCreateTensor(shape.data(), shape.size(), inDtype, nullptr, 0, ACL_FORMAT_ND, shape.data(),
                                       shape.size(), maskDev);
    aclTensor *posT = aclCreateTensor(shape.data(), shape.size(), ACL_INT64, nullptr, 0, ACL_FORMAT_ND, shape.data(),
                                      shape.size(), posDev);
    CHECK(maskT != nullptr && posT != nullptr, name + ": create tensor failed");

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus st = aclnnGenPositionIdsFromMaskGetWorkspaceSize(maskT, fill, posT, &workspaceSize, &executor);
    CHECK(st == ACL_SUCCESS, name + ": GetWorkspaceSize failed, ret=" + std::to_string(st));

    void *workspace = nullptr;
    if (workspaceSize > 0) {
        CHECK(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS,
              name + ": malloc workspace failed");
    }
    st = aclnnGenPositionIdsFromMask(workspace, workspaceSize, executor, stream);
    CHECK(st == ACL_SUCCESS, name + ": aclnnGenPositionIdsFromMask failed, ret=" + std::to_string(st));
    CHECK(aclrtSynchronizeStream(stream) == ACL_SUCCESS, name + ": sync failed");

    std::vector<int64_t> hostOut(static_cast<size_t>(B * S));
    CHECK(aclrtMemcpy(hostOut.data(), outBytes, posDev, outBytes, ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS,
          name + ": memcpy D2H failed");

    bool ok = (hostOut == golden);
    std::cout << "[" << (ok ? "PASS" : "FAIL") << "] " << DtypeName(inDtype) << " | " << name << " B=" << B
              << " S=" << S << " fill=" << fill;
    if (!ok) {
        size_t firstBad = 0;
        while (firstBad < hostOut.size() && hostOut[firstBad] == golden[firstBad]) {
            ++firstBad;
        }

        if (firstBad < hostOut.size()) {
            const int64_t row = static_cast<int64_t>(firstBad) / S;
            const int64_t col = static_cast<int64_t>(firstBad) % S;

            std::cout << "\n        first mismatch: flat=" << firstBad << " row=" << row << " col=" << col
                      << " got=" << hostOut[firstBad] << " expect=" << golden[firstBad];

            const size_t begin = (firstBad > 4) ? firstBad - 4 : 0;
            const size_t end = (firstBad + 5 < hostOut.size()) ? firstBad + 5 : hostOut.size();

            std::cout << "\n        context:";
            for (size_t i = begin; i < end; ++i) {
                std::cout << "\n          [" << i << "] got=" << hostOut[i] << " expect=" << golden[i];
                if (i == firstBad) {
                    std::cout << "  <--";
                }
            }
        }
    }
    std::cout << std::endl;

    if (ok)
        ++g_passed;
    else
        ++g_failed;

    if (workspace)
        aclrtFree(workspace);
    aclDestroyTensor(maskT);
    aclDestroyTensor(posT);
    aclrtFree(maskDev);
    aclrtFree(posDev);
    return true;
}

} // namespace

int main()
{
    if (aclInit(nullptr) != ACL_SUCCESS) {
        std::cerr << "aclInit failed" << std::endl;
        return -1;
    }
    aclrtSetDevice(0);
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    // ---- 用例集: 每个模式在三种 dtype 下各跑一遍 ----
    struct Case {
        std::string name;
        std::vector<int32_t> mask;
        int64_t B, S, fill;
    };
    std::vector<Case> cases = {
        {"left_pad", {0, 0, 1, 1, 1, 1}, 1, 6, 1},
        {"right_pad", {1, 1, 1, 1, 0, 0}, 1, 6, 1},
        {"mid_pad", {1, 1, 0, 0, 1, 1}, 1, 6, 1},
        {"all_zeros", {0, 0, 0, 0}, 1, 4, 1},
        {"all_ones", {1, 1, 1, 1}, 1, 4, 1},
        {"s_one", {1}, 1, 1, 1},
        {"s_one_zero", {0}, 1, 1, 1},
        {"fill_zero", {0, 1, 1, 0, 1}, 1, 5, 0},
        {"fill_neg", {0, 1, 1, 0, 1}, 1, 5, -1},
        // 多行: 验证按 B 切核
        {"multi_row", {0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1}, 3, 4, 1},
    };

    std::vector<aclDataType> dtypes = {ACL_INT32, ACL_INT64, ACL_BOOL};

    bool fatal = false;
    for (auto dt : dtypes) {
        for (auto &c : cases) {
            if (!RunCase(c.name, c.mask, c.B, c.S, c.fill, dt, stream)) {
                fatal = true;
                break;
            }
        }
        if (fatal)
            break;
    }

    std::cout << "\n==== summary: " << g_passed << " passed, " << g_failed << " failed ====" << std::endl;

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return (g_failed == 0 && !fatal) ? 0 : 1;
}
