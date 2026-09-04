/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef APACE_ST_UTILS_H
#define APACE_ST_UTILS_H

#include <cstring>
#include <algorithm>
#include <acl/acl.h>
#include <climits>
#include <iostream>
#include <fstream>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#define INFO_LOG(fmt, args...) \
    do { \
        fprintf(stdout, "[INFO] " fmt "\n", ##args); \
        fprintf(stdout, "[INFO] %s:%d " fmt "\n", __FILE__, __LINE__, ##args); \
        fflush(stdout); \
    } while (0)
#define WARNING_LOG(fmt, args...) \
    do { \
        fprintf(stdout, "[WARNING] " fmt "\n", ##args); \
        fflush(stdout); \
    } while (0)
#define ERROR_LOG(fmt, args...) \
    do { \
        fprintf(stdout, "[ERROR] " fmt "\n", ##args); \
        fflush(stdout); \
    } while (0)

// Macro function for unwinding acl errors.
#define ACL_CHECK(status) \
    do { \
        aclError error = status; \
        if (error != ACL_ERROR_NONE) { \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << error << std::endl; \
        } \
    } while (0)

// ACL错误检查宏
#define ACL_CHECK_WITH_RET(cond, log_func, return_expr) \
    do { \
        if ((cond) != ACL_ERROR_NONE) { \
            log_func; \
            return_expr; \
        } \
    } while (0)

// Macro function for unwinding rt errors.
#define RT_CHECK(status) \
    do { \
        rtError_t error = status; \
        if (error != RT_ERROR_NONE) { \
            std::cerr << __FILE__ << ":" << __LINE__ << " rtError:" << error << std::endl; \
        } \
    } while (0)

inline bool ReadFile(const std::string &filePath, void *buffer, size_t bufferSize)
{
    struct stat sBuf;
    int fileStatus = stat(filePath.data(), &sBuf);
    if (fileStatus == -1) {
        ERROR_LOG("Failed to get file");
        return false;
    }
    if (S_ISREG(sBuf.st_mode) == 0) {
        ERROR_LOG("%s is not a file, please enter a file.", filePath.c_str());
        return false;
    }

    std::ifstream file;
    file.open(filePath, std::ios::binary);
    if (!file.is_open()) {
        ERROR_LOG("Open file failed. path = %s.", filePath.c_str());
        return false;
    }

    std::filebuf *buf = file.rdbuf();
    size_t size = buf->pubseekoff(0, std::ios::end, std::ios::in);
    if (size == 0) {
        ERROR_LOG("File size is 0");
        file.close();
        return false;
    }
    if (size > bufferSize) {
        ERROR_LOG("File size is larger than buffer size.");
        file.close();
        return false;
    }
    buf->pubseekpos(0, std::ios::in);
    buf->sgetn(static_cast<char *>(buffer), size);
    file.close();
    return true;
}

inline bool WriteFile(const std::string &filePath, const void *buffer, size_t size, size_t offset = 0)
{
    if (buffer == nullptr) {
        ERROR_LOG("Write file failed. Buffer is nullptr.");
        return false;
    }

    int fd = open(filePath.c_str(), O_RDWR | O_CREAT, 0666);
    if (!fd) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    // lock
    if (flock(fd, LOCK_EX) == -1) {
        std::cerr << "Failed to acquire lock: " << strerror(errno) << std::endl;
        close(fd);
        return false;
    }

    // move ptr to specified offset
    if (lseek(fd, offset, SEEK_SET) == -1) {
        std::cerr << "Failed to seek in file: " << strerror(errno) << std::endl;
        close(fd);
        return false;
    }

    // write data
    if (write(fd, static_cast<const char *>(buffer), size) != static_cast<ssize_t>(size)) {
        std::cerr << "Failed to write to file: " << strerror(errno) << std::endl;
    }

    // unlock
    flock(fd, LOCK_UN);

    close(fd);
    return true;
}

inline uint64_t ParsePositiveUint64(const char *arg, const char *name)
{
    std::string value(arg);
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos) {
        throw std::invalid_argument(std::string("ERROR: ") + name + " must be a positive integer");
    }

    try {
        uint64_t parsed = std::stoull(value);
        if (parsed == 0UL) {
            throw std::invalid_argument(std::string("ERROR: ") + name + " must be greater than 0");
        }
        return parsed;
    } catch (const std::out_of_range &) {
        throw std::invalid_argument(std::string("ERROR: ") + name + " is out of range for uint64_t");
    }
}

inline void CheckUint32Shape(uint64_t value, const char *name)
{
    // QuantMatmulTilingData serializes public shape fields as uint32_t.
    constexpr uint64_t uint32Max = static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());
    if (value > uint32Max) {
        throw std::invalid_argument(std::string("ERROR: ") + name + " must not exceed UINT32_MAX");
    }
}

inline void PrintUsage(const std::string &programName)
{
    if (programName == "./matmul_a16w16_swat") {
        std::cerr << "Usage: " << programName << " m k n transA transB" << std::endl;
    } else {
        std::cerr << "Usage: " << programName << " m k n" << std::endl;
    }
    std::cerr << "Args: " << std::endl;
    std::cerr << "  m: row of matrix A" << std::endl;
    std::cerr << "  k: col of matrix A" << std::endl;
    std::cerr << "  n: col of matrix B" << std::endl;
    if (programName == "./matmul_a16w16_swat") {
        std::cerr << "  transA: transdata of matrix A" << std::endl;
        std::cerr << "  transB: transdata of matrix B" << std::endl;
    }
    std::cerr << "Example: " << programName << " 100 50 200" << std::endl;
    if (programName == "./matmul_a16w16_swat") {
        std::cerr << "Example: " << programName << " 100 50 200 false true" << std::endl;
    }
}

inline void ParseArguments(int argc, char *argv[], uint64_t &m, uint64_t &k, uint64_t &n)
{
    if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        PrintUsage(argv[0]);
        std::exit(1);
    }
    if (argc != 4) {
        throw std::invalid_argument("ERROR: Invalid number of arguments, expected exactly 3 arguments: m k n");
    }
    m = ParsePositiveUint64(argv[1], "m");
    k = ParsePositiveUint64(argv[2], "k");
    n = ParsePositiveUint64(argv[3], "n");
    CheckUint32Shape(m, "m");
    CheckUint32Shape(k, "k");
    CheckUint32Shape(n, "n");
}

inline void ParseArguments(int argc, char *argv[], uint64_t &m, uint64_t &k, uint64_t &n, bool &transA, bool &transB)
{
    if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        PrintUsage(argv[0]);
        std::exit(1);
    }
    if (argc != 4 && argc != 6) {
        throw std::invalid_argument(
            "ERROR: Invalid number of arguments, expected least 4 arguments: m k n, or 6 arguments: m k n transA "
            "transB");
    }
    m = ParsePositiveUint64(argv[1], "m");
    k = ParsePositiveUint64(argv[2], "k");
    n = ParsePositiveUint64(argv[3], "n");
    if (argc == 6) {
        transA = (std::string(argv[4]) == "true");
        transB = (std::string(argv[5]) == "true");
    } else {
        transA = false;
        transB = true;
    }
    CheckUint32Shape(m, "m");
    CheckUint32Shape(k, "k");
    CheckUint32Shape(n, "n");
}

#endif // UTILS_H
