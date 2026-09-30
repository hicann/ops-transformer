/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/**
 * @file tensor.h
 */
#ifndef MI_TENSOR_H
#define MI_TENSOR_H
#include <string>
#include <cstdint>

class Tensor {
public:
    Tensor(size_t capacity);
    ~Tensor();
    inline uint8_t* device()
    {
        return device_;
    }
    inline uint8_t* host()
    {
        return host_;
    }
    inline size_t size()
    {
        return size_;
    }
    inline bool SetSize(size_t size)
    {
        size_ = size;
        return true;
    };

    bool FromFile(const std::string& fileName = "NONE");
    bool ToFile(const std::string& fileName = "NONE");

    // void SetValue(int32_t value);
    // void SetValue(float value);
    // void SetValue(half value);

    // void SetVector(const std::vector<int64_t> &data);
    // void SetVector(const std::vector<float> &data);

private:
    size_t capacity_, size_;
    uint8_t *host_, *device_;
};

#endif // MI_TENSOR
