/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnn/opdev/op_errno.h"
#include "aclnnop/aclnn_text_gated_residual_grad.h"

#include "../text_gated_residual_grad_test_constants.h"

namespace {
using namespace HcGradTest;
constexpr size_t STORAGE_BYTES = 4096;
constexpr int INITIALIZATION_FAILURE = 2;
constexpr int GATE_PRESENT = 1;
constexpr int GAMMA_PRESENT = 2;
constexpr int OPTIONAL_COMBINATIONS = 4;
enum class MissingResult {
    NONE,
    WORKSPACE,
    EXECUTOR
};

struct QueryChecks {
    int failures = 0;
    int checks = 0;

    void CheckFirstError(std::string_view expectedDetail, std::string_view duplicateDetail)
    {
        ++checks;
        const char* report = aclGetRecentErrMsg();
        const std::string_view message = report == nullptr ? "" : report;
        const auto first = message.find(expectedDetail);
        if (message.find("EZ1001") == std::string_view::npos || message.find("EZ9999") != std::string_view::npos ||
            first == std::string_view::npos ||
            message.find(expectedDetail, first + expectedDetail.size()) != std::string_view::npos ||
            message.find(duplicateDetail) != std::string_view::npos) {
            std::cerr << "expected one EZ1001 with " << expectedDetail << ", got: " << message << '\n';
            ++failures;
        }
    }

    void CheckMissingLaunchExecutor()
    {
        aclGetRecentErrMsg();
        ++checks;
        const auto status = aclnnTextGatedResidualGrad(nullptr, 0, nullptr, nullptr);
        if (status != ACLNN_ERR_INNER) {
            std::cerr << "null launch executor: expected " << ACLNN_ERR_INNER << ", got " << status << '\n';
            ++failures;
        }
        CheckFirstError("executor is nullptr", "CommonOpExecutorRun failed");
    }

    void Query(const std::vector<aclTensor*>& a, int64_t expected, MissingResult missing = MissingResult::NONE,
               int64_t channels = CHANNELS)
    {
        uint64_t workspace = 0;
        aclOpExecutor* executor = nullptr;
        const auto status = aclnnTextGatedResidualGradGetWorkspaceSize(
            a[GRAD_MIXED], a[X], a[W_DOWN], a[W_UP], a[INV_RMS], a[DOWN_PRE], a[UP], a[GRAD_INJECT], a[W_GATE],
            a[GAMMA], a[INJECT], channels, a[DX], a[DW_DOWN], a[DW_UP], a[DW_GATE], a[DGAMMA],
            missing == MissingResult::WORKSPACE ? nullptr : &workspace,
            missing == MissingResult::EXECUTOR ? nullptr : &executor);
        ++checks;
        if (status != expected) {
            std::cerr << "check " << checks << ": expected " << expected << ", got " << status << '\n';
            ++failures;
        }
        if (executor != nullptr && (aclSetAclOpExecutorRepeatable(executor) || aclDestroyAclOpExecutor(executor))) {
            throw std::runtime_error("executor cleanup failed");
        }
    }

    void CheckCombinations(const std::vector<aclTensor*>& tensors)
    {
        // Query-only tests: storage aliases intentionally; no kernel reads/writes this buffer.
        for (int mask = 0; mask < OPTIONAL_COMBINATIONS; ++mask) {
            auto a = tensors;
            if (!(mask & GATE_PRESENT)) {
                a[GRAD_INJECT] = a[W_GATE] = a[INJECT] = a[DW_GATE] = nullptr;
            }
            if (!(mask & GAMMA_PRESENT)) {
                a[GAMMA] = a[DGAMMA] = nullptr;
            }
            Query(a, ACLNN_SUCCESS);
        }
        for (size_t index : {W_GATE, GAMMA, DW_GATE, DGAMMA}) {
            auto a = tensors;
            a[index] = nullptr;
            Query(a, ACLNN_ERR_PARAM_NULLPTR); // Both directions of each optional input/output pair.
        }
        Query(tensors, ACLNN_ERR_PARAM_NULLPTR, MissingResult::WORKSPACE);
        Query(tensors, ACLNN_ERR_PARAM_NULLPTR, MissingResult::EXECUTOR);

        auto missingX = tensors;
        missingX[X] = nullptr;
        aclGetRecentErrMsg();
        Query(missingX, ACLNN_ERR_PARAM_NULLPTR);
        CheckFirstError("x must not be nullptr.", "required param is nullptr.");
    }
};

std::vector<aclTensor*> CreateTensors(aclDataType dtype, void* storage, int64_t channels = CHANNELS,
                                      int64_t projection = PROJECTION, int64_t hidden = HIDDEN)
{
    const int64_t features = channels * hidden;
    const std::vector<std::vector<int64_t>> shapes = {
        {1, 1, hidden},         {1, 1, channels},       {1, 1, features},     {projection, features},
        {features, projection}, {channels, features},   {features},           {1, 1, channels},
        {1, 1, channels},       {1, 1, projection},     {1, 1, features},     {1, 1, features},
        {projection, features}, {features, projection}, {channels, features}, {features}};
    std::vector<aclTensor*> tensors;
    for (size_t i = 0; i < shapes.size(); ++i) {
        const auto& shape = shapes[i];
        std::vector<int64_t> stride(shape.size(), 1);
        for (size_t j = shape.size() - 1; j > 0; --j) {
            stride[j - 1] = stride[j] * shape[j];
        }
        const auto type = (i == GAMMA || i == INV_RMS || i == DGAMMA) ? ACL_FLOAT : dtype;
        auto tensor = aclCreateTensor(shape.data(), shape.size(), type, stride.data(), 0, ACL_FORMAT_ND, shape.data(),
                                      shape.size(), storage);
        if (tensor == nullptr) {
            for (auto created : tensors) {
                aclDestroyTensor(created);
            }
            throw std::runtime_error("aclCreateTensor failed");
        }
        tensors.push_back(tensor);
    }
    return tensors;
}
} // namespace

int main(int argc, char** argv)
{
    const int device = argc > 1 ? std::stoi(argv[1]) : 0;
    if (aclInit(nullptr) || aclrtSetDevice(device)) {
        return INITIALIZATION_FAILURE;
    }
    void* storage = nullptr;
    if (aclrtMalloc(&storage, STORAGE_BYTES, ACL_MEM_MALLOC_NORMAL_ONLY)) {
        return INITIALIZATION_FAILURE;
    }
    QueryChecks results;
    results.CheckMissingLaunchExecutor();
    for (auto dtype : {ACL_FLOAT16, ACL_BF16}) {
        const auto tensors = CreateTensors(dtype, storage);
        results.CheckCombinations(tensors);
        // Keep every tensor shape consistent so only the public dimension limit can reject the call.
        const std::array<std::array<int64_t, 3>, 4> invalidShapes{{{21, PROJECTION, HIDDEN},
                                                                   {CHANNELS, 656, HIDDEN},
                                                                   {CHANNELS, PROJECTION + 1, HIDDEN},
                                                                   {CHANNELS, PROJECTION, HIDDEN + 1}}};
        for (const auto& [channels, projection, hidden] : invalidShapes) {
            auto invalid = CreateTensors(dtype, storage, channels, projection, hidden);
            results.Query(invalid, ACLNN_ERR_PARAM_INVALID, MissingResult::NONE, channels);
            for (auto tensor : invalid) {
                aclDestroyTensor(tensor);
            }
        }
        for (auto invalidType : {ACL_FLOAT, dtype == ACL_BF16 ? ACL_FLOAT16 : ACL_BF16}) {
            const auto invalidTensors = CreateTensors(invalidType, storage);
            if (invalidType == ACL_FLOAT) {
                auto invalidX = tensors;
                invalidX[X] = invalidTensors[X];
                aclGetRecentErrMsg();
                results.Query(invalidX, ACLNN_ERR_PARAM_INVALID);
                results.CheckFirstError("Expected BF16 or FP16.", "invalid dtype.");
            }
            for (size_t index : {DW_DOWN, DW_UP, DW_GATE}) {
                auto mismatched = tensors;
                mismatched[index] = invalidTensors[index];
                results.Query(mismatched, ACLNN_ERR_PARAM_INVALID);
            }
            for (auto tensor : invalidTensors) {
                aclDestroyTensor(tensor);
            }
        }
        for (auto tensor : tensors) {
            aclDestroyTensor(tensor);
        }
    }
    aclrtFree(storage);
    aclrtResetDevice(device);
    aclFinalize();
    std::cout << "optional output checks=" << results.checks << " failures=" << results.failures << '\n';
    return results.failures ? 1 : 0;
}
