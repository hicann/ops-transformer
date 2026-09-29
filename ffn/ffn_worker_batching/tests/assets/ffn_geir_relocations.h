/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// FFN test-only auxiliary HBM relocation for the TTK GEIR executable.
#pragma once
#include "acl/acl.h"

// Auxiliary allocations belong to this executable, never to the Python parent.
static JsonValue input_relocations;
static vector<void*> auxiliary_buffers;
static string relocation_input_prefix;

static bool RelocateInput(const string& input_path, uint8_t* bytes, size_t size)
{
    for (const auto& fixup : input_relocations.getArray()) {
        const int64_t index = fixup.at("input_index").getInt();
        if (input_path != relocation_input_prefix + "_" + to_string(index) + ".bin")
            continue;
        const int64_t offset = fixup.at("offset").getInt();
        const int64_t length = fixup.at("size").getInt();
        if (offset < 0 || size < sizeof(uint64_t) || static_cast<uint64_t>(offset) > size - sizeof(uint64_t) ||
            length <= 0)
            return false;
        ifstream file(fixup.at("path").getString(), ios::binary | ios::ate);
        if (!file || file.tellg() != length)
            return false;
        file.seekg(0);
        vector<uint8_t> payload(static_cast<size_t>(length));
        if (!file.read(reinterpret_cast<char*>(payload.data()), length))
            return false;
        void* pointer = nullptr;
        if (aclrtMalloc(&pointer, length, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS)
            return false;
        auxiliary_buffers.push_back(pointer);
        if (aclrtMemcpy(pointer, length, payload.data(), length, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS)
            return false;
        uint64_t address = reinterpret_cast<uint64_t>(pointer);
        memcpy(bytes + offset, &address, sizeof(address));
    }
    return true;
}
