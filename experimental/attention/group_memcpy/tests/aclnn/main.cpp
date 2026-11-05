/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "acl/acl.h"
#include "aclnn_group_memcpy.h"
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using DevicePtr = std::unique_ptr<void, aclError (*)(void*)>;
using TensorPtr = std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)>;
using ListPtr = std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList*)>;

static void Check(int status, const char* operation)
{
    if (status != 0) {
        const char* detail = aclGetRecentErrMsg();
        throw std::runtime_error(std::string(operation) + ": status=" + std::to_string(status) +
                                 (detail == nullptr ? "" : std::string("; ") + detail));
    }
}
static void Require(bool value, const std::string& message)
{
    if (!value) {
        throw std::runtime_error(message);
    }
}
struct Runtime {
    bool initialized = false;
    bool deviceSet = false;
    aclrtStream stream = nullptr;
    void Init()
    {
        Check(aclInit(nullptr), "aclInit");
        initialized = true;
        Check(aclrtSetDevice(0), "aclrtSetDevice");
        deviceSet = true;
        Check(aclrtCreateStream(&stream), "aclrtCreateStream");
    }
    ~Runtime()
    {
        if (stream != nullptr) {
            aclrtSynchronizeStream(stream);
            aclrtDestroyStream(stream);
        }
        if (deviceSet) {
            aclrtResetDevice(0);
        }
        if (initialized) {
            aclFinalize();
        }
    }
};
static DevicePtr Allocate(size_t bytes)
{
    void* ptr = nullptr;
    Check(aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_NORMAL_ONLY), "aclrtMalloc");
    return DevicePtr(ptr, aclrtFree);
}
static std::vector<uint8_t> ReadFile(const fs::path& path, size_t bytes)
{
    Require(fs::file_size(path) == bytes, "unexpected binary size: " + path.string());
    std::ifstream file(path, std::ios::binary);
    std::vector<uint8_t> data(bytes);
    if (bytes != 0) {
        file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(bytes));
    }
    Require(bool(file), "cannot read " + path.string());
    return data;
}
static void WriteFile(const fs::path& path, const std::vector<uint8_t>& bytes)
{
    Require(!fs::exists(path), "refusing to overwrite existing replay result: " + path.string());
    std::ofstream file(path, std::ios::binary);
    if (!bytes.empty()) {
        file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    Require(bool(file), "cannot write " + path.string());
}

// aclDestroyTensorList owns/destroys member descriptors; allocations are separate.
// Declare list last so it is destroyed before the device allocations.
struct Group {
    size_t width;
    size_t prefix;
    std::vector<DevicePtr> allocations;
    std::vector<std::vector<uint8_t>> host;
    ListPtr list{nullptr, aclDestroyTensorList};

    Group(size_t count, size_t bytes, aclDataType dtype, size_t itemBytes, bool source, int64_t stride = 1)
        : width(bytes),
          prefix(64 + 3 * itemBytes)
    {
        Require(count > 0 && count <= static_cast<size_t>(INT32_MAX), "invalid pair count");
        Require(bytes % itemBytes == 0 && bytes <= (static_cast<size_t>(INT64_MAX) - 256) / 2, "invalid byte count");
        const size_t storageBytes = prefix + bytes * static_cast<size_t>(stride) + 64;
        std::vector<TensorPtr> tensors;
        std::vector<const aclTensor*> pointers;
        for (size_t i = 0; i < count; ++i) {
            allocations.push_back(Allocate(storageBytes));
            host.emplace_back(storageBytes, uint8_t(0xA5));
            for (size_t j = 0; source && j < bytes; ++j) {
                host.back()[prefix + j] = dtype == ACL_BOOL ? uint8_t((i + j) % 2) : uint8_t((i * 19 + j * 73) % 256);
            }
            Upload(i);
            const int64_t dims[] = {static_cast<int64_t>(bytes / itemBytes)};
            const int64_t strides[] = {stride};
            const int64_t storageDims[] = {static_cast<int64_t>(storageBytes / itemBytes)};
            aclTensor* tensor = aclCreateTensor(dims, 1, dtype, strides, prefix / itemBytes, ACL_FORMAT_ND, storageDims,
                                                1, allocations.back().get());
            Require(tensor != nullptr, "aclCreateTensor failed");
            tensors.emplace_back(tensor, aclDestroyTensor);
            pointers.push_back(tensor);
        }
        list.reset(aclCreateTensorList(pointers.data(), pointers.size()));
        Require(list != nullptr, "aclCreateTensorList failed");
        for (auto& tensor : tensors) {
            tensor.release(); // ownership transferred to list
        }
    }
    void Upload(size_t i)
    {
        Check(aclrtMemcpy(allocations[i].get(), host[i].size(), host[i].data(), host[i].size(),
                          ACL_MEMCPY_HOST_TO_DEVICE),
              "upload");
    }
    std::vector<uint8_t> Download(size_t i) const
    {
        std::vector<uint8_t> result(host[i].size());
        Check(aclrtMemcpy(result.data(), result.size(), allocations[i].get(), result.size(), ACL_MEMCPY_DEVICE_TO_HOST),
              "download");
        return result;
    }
};

static void Launch(const Group& src, const Group& dst, aclrtStream stream)
{
    uint64_t size = 0;
    aclOpExecutor* executor = nullptr;
    Check(aclnnGroupMemcpyGetWorkspaceSize(src.list.get(), dst.list.get(), &size, &executor),
          "aclnnGroupMemcpyGetWorkspaceSize");
    Require(executor != nullptr, "null executor after successful phase 1");
    DevicePtr workspace(nullptr, aclrtFree);
    if (size != 0) {
        workspace = Allocate(size);
    }
    Check(aclnnGroupMemcpy(workspace.get(), size, executor, stream), "aclnnGroupMemcpy");
    Check(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");
}

static void Verify(const Group& src, const Group& dst)
{
    for (size_t i = 0; i < src.host.size(); ++i) {
        auto expected = dst.host[i];
        std::copy_n(src.host[i].begin() + src.prefix, src.width, expected.begin() + dst.prefix);
        Require(dst.Download(i) == expected, "output/guard mismatch at pair " + std::to_string(i));
        Require(src.Download(i) == src.host[i], "input was modified at pair " + std::to_string(i));
    }
}
static void Negative(const aclTensorList* src, const aclTensorList* dst)
{
    uint64_t size = UINT64_MAX;
    aclOpExecutor* executor = nullptr;
    Require(aclnnGroupMemcpyGetWorkspaceSize(src, dst, &size, &executor) != 0,
            "invalid arguments unexpectedly accepted");
    Require(size == 0 && executor == nullptr, "phase 1 did not clear outputs on rejection");
}
static void TestInvalidArguments()
{
    Group one(1, 64, ACL_UINT8, 1, true);
    Group two(2, 64, ACL_UINT8, 1, false);
    Group wrongWidth(1, 32, ACL_UINT8, 1, false);
    Group wrongType(1, 64, ACL_INT8, 1, false);
    Group strided(1, 64, ACL_UINT8, 1, false, 2);
    Group unsupported(1, 64, ACL_DOUBLE, 8, false);
    Negative(nullptr, one.list.get());
    Negative(one.list.get(), nullptr);
    Negative(one.list.get(), two.list.get());
    Negative(one.list.get(), wrongWidth.list.get());
    Negative(one.list.get(), wrongType.list.get());
    Negative(one.list.get(), strided.list.get());
    Negative(unsupported.list.get(), unsupported.list.get());
    uint64_t size = 0;
    aclOpExecutor* executor = nullptr;
    Require(aclnnGroupMemcpyGetWorkspaceSize(one.list.get(), one.list.get(), nullptr, &executor) != 0,
            "null workspaceSize accepted");
    Require(aclnnGroupMemcpyGetWorkspaceSize(one.list.get(), one.list.get(), &size, nullptr) != 0,
            "null executor output accepted");
    Require(aclnnGroupMemcpy(nullptr, 0, nullptr, nullptr) != 0, "null phase-2 executor accepted");
}

static void SelfTest(aclrtStream stream)
{
    struct Type {
        aclDataType dtype;
        size_t bytes;
    };
    const Type types[] = {{ACL_INT8, 1},    {ACL_UINT8, 1}, {ACL_INT16, 2}, {ACL_INT32, 4}, {ACL_INT64, 8},
                          {ACL_FLOAT16, 2}, {ACL_BF16, 2},  {ACL_FLOAT, 4}, {ACL_BOOL, 1}};
    size_t cases = 0;
    for (const auto& type : types) {
        for (size_t elements : {0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 94207, 94208, 94209, 188417}) {
            Group src(3, elements * type.bytes, type.dtype, type.bytes, true);
            Group dst(3, elements * type.bytes, type.dtype, type.bytes, false);
            Launch(src, dst, stream);
            Verify(src, dst);
            ++cases;
        }
    }
    for (size_t count : {1, 39, 40, 41, 81, 257}) {
        Group src(count, 65, ACL_UINT8, 1, true);
        Group dst(count, 65, ACL_UINT8, 1, false);
        Launch(src, dst, stream);
        Verify(src, dst);
        ++cases;
    }
    TestInvalidArguments();
    std::cout << cases << " bytewise cases plus invalid-argument cases passed\n";
}

// Replay ORIGINAL direct-call bytes; never generate a substitute golden and
// describe it as an original-direct-vs-aclnn comparison.
static void Replay(char** argv, aclrtStream stream)
{
    const fs::path input = fs::weakly_canonical(argv[2]);
    const fs::path golden = fs::weakly_canonical(argv[3]);
    const fs::path output = fs::weakly_canonical(argv[4]);
    Require(output != input && output != golden, "output directory must differ from input/golden directories");
    const auto count = static_cast<size_t>(std::stoull(argv[5]));
    const auto width = static_cast<size_t>(std::stoull(argv[6]));
    Group src(count, width, ACL_UINT8, 1, true);
    Group dst(count, width, ACL_UINT8, 1, false);
    std::vector<std::vector<uint8_t>> expected;
    for (size_t i = 0; i < count; ++i) {
        auto bytes = ReadFile(input / ("src_" + std::to_string(i) + ".bin"), width);
        std::copy(bytes.begin(), bytes.end(), src.host[i].begin() + src.prefix);
        src.Upload(i);
        expected.push_back(ReadFile(golden / ("dst_" + std::to_string(i) + ".bin"), width));
    }
    Launch(src, dst, stream);
    Verify(src, dst);
    fs::create_directories(output);
    for (size_t i = 0; i < count; ++i) {
        auto frame = dst.Download(i);
        std::vector<uint8_t> actual(frame.begin() + dst.prefix, frame.begin() + dst.prefix + width);
        WriteFile(output / ("dst_" + std::to_string(i) + ".bin"), actual);
        Require(actual == expected[i], "original direct-call binary mismatch at pair " + std::to_string(i));
    }
    std::cout << count << " original direct-call binary comparisons passed\n";
}

int main(int argc, char** argv)
{
    try {
        Require(argc == 1 || (argc == 7 && std::string(argv[1]) == "--replay"),
                "usage: opapi_test [--replay INPUT_DIR GOLDEN_DIR OUTPUT_DIR PAIRS BYTES_PER_COPY]");
        Runtime runtime;
        runtime.Init();
        if (argc == 1) {
            SelfTest(runtime.stream);
        } else {
            Replay(argv, runtime.stream);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
