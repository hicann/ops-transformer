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
 * @file common.h
 */
#ifndef COMMON_H
#define COMMON_H

#include <cstdio>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "tensor.h"

#define SUCCESS 0
#define FAILED 1

#define LOG_LEVEL_DEBUG 0
#define LOG_LEVEL_INFO 1
#define LOG_LEVEL_WARN 2
#define LOG_LEVEL_ERROR 3

#ifndef LOG_LEVEL
#define LOG_LEVEL LOG_LEVEL_ERROR // 默认日志级别为 LOG_LEVEL_INFO
#endif

#if LOG_LEVEL <= LOG_LEVEL_DEBUG
#define DEBUG_LOG(fmt, args...) fprintf(stdout, "[DEBUG]  " fmt "\n", ##args)
#else
#define DEBUG_LOG(fmt, args...)
#endif

#if LOG_LEVEL <= LOG_LEVEL_INFO
#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#else
#define INFO_LOG(fmt, args...)
#endif

#if LOG_LEVEL <= LOG_LEVEL_WARN
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
#else
#define WARN_LOG(fmt, args...)
#endif

#if LOG_LEVEL <= LOG_LEVEL_ERROR
#define ERROR_LOG(fmt, args...) fprintf(stderr, "[ERROR] " fmt "\n", ##args)
#else
#define ERROR_LOG(fmt, args...)
#endif

// #define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
// #define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
// #define ERROR_LOG(fmt, args...) fprintf(stderr, "[ERROR]  " fmt "\n", ##args)

template <typename T>
inline void AllocTensor(Tensor*& x, size_t N)
{
    if (x != nullptr) {
        std::cout << "===========================" << std::endl;
        throw std::runtime_error("Class member variables are not initialized.");
    } else {
        x = new Tensor(N * sizeof(T));
    }
}

template <typename T>
inline Tensor* CreateTensor(size_t N)
{
    return new Tensor(N * sizeof(T));
}

template <typename T>
inline void FreeTensor(T* x)
{
    if (x != nullptr) {
        delete x;
        x = nullptr;
    }
}

/**
 * @brief Read data from file
 * @param [in] filePath: file path
 * @param [out] fileSize: file size
 * @return read result
 */
bool ReadFile(const std::string& filePath, size_t& fileSize, void* buffer, size_t bufferSize);

/**
 * @brief Write data to file
 * @param [in] filePath: file path
 * @param [in] buffer: data to write to file
 * @param [in] size: size to write
 * @return write result
 */
bool WriteFile(const std::string& filePath, const void* buffer, size_t size);

#endif // COMMON_H
