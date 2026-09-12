/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MC2_WIN_DUMP_H
#define MC2_WIN_DUMP_H

#include "acl/acl.h"
#include "acl/acl_base.h"
#include "acl/acl_rt.h"
#include "acl/acl_dump.h"
#include "../op_kernel/moe_distribute_comm_ctx.h"
#include "../op_kernel/mc2_moe_context.h"
#include "urma_comm_ctx_dump.h"
#include "mc2_log.h"
#include "mc2_tiling_utils.h"
#include <chrono>
#include <iomanip>
#include <sstream>
#include <stdio.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <errno.h>
#include <vector>
#include <map>
#include "version/runtime_version.h"
#include "version/metadef_version.h"
#include <dlfcn.h>
#include <cstring>
#include <mutex>
#include "./mc2_hcom_topo_info.h"
#include "../op_kernel/mc2_exception_dump.h"

// 支持 9.0.0 版本（90000000）, 当版本带着 beta, alpha, rc 时, rts 接口获取到的版本号会在 90000000 基础上减去一个对应的
// weight 值, 其下限为 9.0.0-alpha, 对应版本号 90000000 - 300 = 89999700
#define EXCEPTION_DUMP_SUPPORT_VERSION 89999700

#if RUNTIME_VERSION_NUM >= EXCEPTION_DUMP_SUPPORT_VERSION && METADEF_VERSION_NUM >= EXCEPTION_DUMP_SUPPORT_VERSION
namespace Mc2Exception {

const std::string OP_NAME = "Mc2Exception";
const uint32_t WIN_SIZE = 1024U * 1024U;
const uint32_t MS_WIDTH = 3U;
const uint32_t MS_PER_S = 1000U;
const mode_t FILE_MODE = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH;
constexpr size_t MAX_GROUP_NAME_LENGTH = 128UL;

const std::map<std::string, std::string> MC2_OP_CONTEXT = {{"MoeDistributeDispatchV3", "Mc2MoeContext"},
                                                           {"MoeDistributeCombineV3", "Mc2MoeContext"},
                                                           {"MoeDistributeDispatchV2", "HcclCombinOpParam"},
                                                           {"MoeDistributeCombineV2", "HcclCombinOpParam"},
                                                           {"MegaMoe", "Mc2MoeContext"}};

class MC2GroupNameManager {
public:
    static MC2GroupNameManager &GetInstance()
    {
        static MC2GroupNameManager instance;
        return instance;
    }
    void SetGroupName(const char *groupName)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (groupName == nullptr || strnlen(groupName, MAX_GROUP_NAME_LENGTH) == 0 ||
            strnlen(groupName, MAX_GROUP_NAME_LENGTH) == MAX_GROUP_NAME_LENGTH) {
            OP_LOGE_FOR_INVALID_VALUE(OP_NAME, "groupName", "invalid", "valid group name");
            return;
        }
        groupName_ = std::string(groupName);
    }
    const char *GetGroupName()
    {
        return groupName_.empty() ? nullptr : groupName_.c_str();
    }

private:
    MC2GroupNameManager() = default;
    std::string groupName_;
    std::mutex mutex_;
};

// 函数指针类型定义
typedef aclError (*aclrtGetArgsFromExceptionInfo_t)(aclrtExceptionInfo *info, void **args, uint32_t *argsSize);
typedef const char *(*acldumpGetPath_t)(acldumpType dumpType);

// 动态加载clrtGetArgsFromExceptionInfo函数
inline aclrtGetArgsFromExceptionInfo_t GetAclrtGetArgsFromExceptionInfoFunc()
{
    static aclrtGetArgsFromExceptionInfo_t func = nullptr;
    static bool initialized = false;
    if (!initialized) {
        initialized = true;
        // 直接从 libacl_rt.so 加载符号
        void *handle = dlopen("libacl_rt.so", RTLD_NOLOAD | RTLD_LAZY);
        if (handle != nullptr) {
            func = (aclrtGetArgsFromExceptionInfo_t)dlsym(handle, "aclrtGetArgsFromExceptionInfo");
            if (func == nullptr) {
                OP_LOGE(OP_NAME, "dlsym aclrtGetArgsFromExceptionInfo failed: %s", dlerror());
            }
        } else {
            OP_LOGE(OP_NAME, "dlopen failed: %s", dlerror());
        }
    }
    return func;
}

// 动态加载acldumpGetPath函数
inline acldumpGetPath_t GetAcldumpGetPathFunc()
{
    static acldumpGetPath_t func = nullptr;
    static bool initialized = false;
    if (!initialized) {
        initialized = true;
        // 尝试加载ascend_dump.so
        void *handle = dlopen("libascend_dump.so", RTLD_NOLOAD | RTLD_LAZY);
        if (handle != nullptr) {
            func = (acldumpGetPath_t)dlsym(handle, "acldumpGetPath");
            if (func == nullptr) {
                OP_LOGE(OP_NAME, "dlsym acldumpGetPath failed: %s", dlerror());
            } else {
                OP_LOGD(OP_NAME, "Successfully loaded acldumpGetPath function");
            }
        } else {
            OP_LOGE(OP_NAME, "dlopen libascend_dump.so failed: %s", dlerror());
        }
    }
    return func;
}

inline std::string GetTimestampWithMilliseconds()
{
    auto now = std::chrono::system_clock::now();
    time_t now_time_t = std::chrono::system_clock::to_time_t(now);

    std::tm tm_utc;
    gmtime_r(&now_time_t, &tm_utc);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % MS_PER_S;

    std::stringstream ss;
    ss << std::put_time(&tm_utc, "%Y%m%d%H%M%S") << std::setfill('0') << std::setw(MS_WIDTH) << ms.count();
    return ss.str();
}

inline std::string GenDumpFileName(aclrtExceptionInfo *args, const char *op)
{
    std::stringstream ss;

    uint32_t streamId = aclrtGetStreamIdFromExceptionInfo(args);
    uint32_t taskId = aclrtGetTaskIdFromExceptionInfo(args);
    std::string ts = GetTimestampWithMilliseconds();

    ss << "mc2_exception_info_" << op << "." << streamId << "." << taskId << "." << ts;

    return ss.str();
}

inline bool IsStrEmpty(std::string str)
{
    // return if the string is null or empty or spaces
    return (str.empty() || std::all_of(str.begin(), str.end(), [](unsigned char c) { return std::isspace(c); }));
}

inline int DumpToFile(std::string dir, std::string name, uint32_t id, void *buf, size_t bufSize = WIN_SIZE)
{
    // check if dir and buf valid
    if (IsStrEmpty(dir) || IsStrEmpty(name)) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "dir or name");
        return -1;
    }

    std::string rankPath = dir + "/" + std::to_string(id) + "/";
    std::string path = rankPath + name;
    OP_LOGE(OP_NAME, "Start to dump file. The dump path is %s", path.c_str());

    // Open file
    int fd = open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, FILE_MODE);
    if (fd < 0) {
        int openErrno = errno;
        OP_LOGE(OP_NAME, "Failed to open a dump file. errno=%d(%s)", openErrno, strerror(openErrno));
        return -1;
    }

    // Write to file
    ssize_t ret = write(fd, buf, bufSize);
    if (ret < 0) {
        int writeErrno = errno;
        OP_LOGE(OP_NAME, "Failed to write a dump file. errno=%d(%s)", writeErrno, strerror(writeErrno));
        close(fd);
        return -1;
    }

    // Close file
    close(fd);
    OP_LOGE(OP_NAME, "Dump to file %s done.", path.c_str());
    return 0;
}

#if MC2_DFX_ENABLE
const uint32_t XN_ADDR_DUMP_SIZE = 64U * 64U * 8U;
const uint32_t CKE_ADDR_DUMP_SIZE = 64U * 2U * 8U;
const uint32_t MAX_PEERMEM_DUMP_SIZE = 64U * 1024U * 1024U;

inline std::string GenDumpFileNameWithPrefix(aclrtExceptionInfo *args, const char *op, const char *prefix)
{
    std::stringstream ss;

    uint32_t streamId = aclrtGetStreamIdFromExceptionInfo(args);
    uint32_t taskId = aclrtGetTaskIdFromExceptionInfo(args);
    std::string ts = GetTimestampWithMilliseconds();

    ss << prefix << op << "." << streamId << "." << taskId << "." << ts;

    return ss.str();
}

inline int DumpToFileWithSize(std::string dir, std::string name, uint32_t id, void *buf, uint32_t size)
{
    if (IsStrEmpty(dir) || IsStrEmpty(name)) {
        OP_LOGE(OP_NAME, "Dump path or buf is null.");
        return -1;
    }

    std::string rankPath = dir + "/" + std::to_string(id) + "/";
    std::string path = rankPath + name;
    OP_LOGE(OP_NAME, "Start to dump file. The dump path is %s", path.c_str());

    int fd = open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, FILE_MODE);
    if (fd < 0) {
        int openErrno = errno;
        OP_LOGE(OP_NAME, "Failed to open a dump file. errno=%d(%s)", openErrno, strerror(openErrno));
        return -1;
    }

    ssize_t ret = write(fd, buf, size);
    if (ret < 0) {
        int writeErrno = errno;
        OP_LOGE(OP_NAME, "Failed to write a dump file. errno=%d(%s)", writeErrno, strerror(writeErrno));
        close(fd);
        return -1;
    }

    close(fd);
    OP_LOGE(OP_NAME, "Dump to file %s done.", path.c_str());
    return 0;
}
#endif // MC2_DFX_ENABLE

inline int ProcessArgsForA5(uint64_t argsAddr, std::vector<uint8_t> &winBuf, const char *op)
{
    // Get hccl context from its addr
    if (op == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "op");
        return -1;
    }
    void *winAddr = nullptr;
    aclError ret = ACL_SUCCESS;
    auto is_support_op = MC2_OP_CONTEXT.find(op);
    if (is_support_op == MC2_OP_CONTEXT.end()) {
        // 不支持的算子
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "op", op, "The value of op must be within the supported range.");
        return -1;
    }
    const std::string &context_type = is_support_op->second;
    // V2分支
    if (context_type == "HcclCombinOpParam") {
        std::vector<uint8_t> hcclArgs(sizeof(HcclCombinOpParam), 0);
        ret = aclrtMemcpy(hcclArgs.data(), sizeof(HcclCombinOpParam), (void *)argsAddr, sizeof(HcclCombinOpParam),
                          ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            OP_LOGE(OP_NAME, "aclrtMemcpy HcclCombinOpParam from device to host failed. ret = %d", ret);
            return -1;
        }
        HcclCombinOpParam *winContext = reinterpret_cast<HcclCombinOpParam *>(hcclArgs.data());
        if (winContext == nullptr) {
            OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "winContext");
            return -1;
        }
        OP_LOGD(OP_NAME, "Get winContext from args. rankId=%u, rankDim=%u", winContext->rankId, winContext->rankDim);
        if (winContext->rankId >= HCCL_MTE_MAX_RANK_NUM) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "rankId", std::to_string(winContext->rankId).c_str(),
                                                  "The value of rankId must be less than HCCL_MTE_MAX_RANK_NUM");
            return -1;
        }
        winAddr = reinterpret_cast<void *>(winContext->windowsIn[winContext->rankId]);
    } else if (context_type == "Mc2MoeContext") {
        // V3分支
        std::vector<uint8_t> hcclArgs(sizeof(Mc2Aclnn::Mc2MoeContext), 0);
        ret = aclrtMemcpy(hcclArgs.data(), sizeof(Mc2Aclnn::Mc2MoeContext), (void *)argsAddr,
                          sizeof(Mc2Aclnn::Mc2MoeContext), ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            OP_LOGE(OP_NAME, "aclrtMemcpy Mc2MoeContext from device to host failed. ret = %d", ret);
            return -1;
        }
        Mc2Aclnn::Mc2MoeContext *winContext = reinterpret_cast<Mc2Aclnn::Mc2MoeContext *>(hcclArgs.data());
        if (winContext == nullptr) {
            OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "winContext");
            return -1;
        }
        OP_LOGD(OP_NAME, "Get winContext from args. rankId=%u", winContext->epRankId);
        if (winContext->epRankId >= Mc2Aclnn::HCCL_MAX_RANK_SIZE) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "epRankId", std::to_string(winContext->epRankId).c_str(),
                                                  "The value of epRankId must be less than HCCL_MAX_RANK_SIZE");
            return -1;
        }
        winAddr = reinterpret_cast<void *>(winContext->epHcclBuffer_[winContext->epRankId]);
    } else {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "context_type", context_type.c_str(),
                                              "The value of context_type must be within the supported range.");
        return -1;
    }
    if (winAddr == nullptr) {
        OP_LOGE(OP_NAME, "Get winaddr failed.");
        return -1;
    }
    // Get windowsIn of each rank from hccl context
    ret = aclrtMemcpy(winBuf.data(), WIN_SIZE, winAddr, WIN_SIZE, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        OP_LOGE(OP_NAME, "aclrtMemcpy windows from device to host failed. ret = %d", ret);
        return -1;
    }
    return 0;
}

inline int ProcessArgsForA3(uint64_t argsAddr, std::vector<uint8_t> &winBuf)
{
    // Get hccl context from its addr
    std::vector<uint8_t> hcclArgs(sizeof(CommContextForDump), 0);
    auto ret = aclrtMemcpy(hcclArgs.data(), sizeof(CommContextForDump), (void *)argsAddr, sizeof(CommContextForDump),
                           ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        OP_LOGE(OP_NAME, "aclrtMemcpy CommContextForDump from device to host failed. ret = %d", ret);
        return -1;
    }
    CommContextForDump *winContext = reinterpret_cast<CommContextForDump *>(hcclArgs.data());
    if (winContext == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "winContext");
        return -1;
    }
    OP_LOGD(OP_NAME, "Get winContext from args. rankId=%u", winContext->epRankId);

    void *winAddr = reinterpret_cast<void *>(winContext->epHcclBufffer_[winContext->epRankId]);
    if (winAddr == nullptr) {
        OP_LOGE(OP_NAME, "Get win addr failed.");
        return -1;
    }

    // Get windowsIn of each rank from hccl context
    ret = aclrtMemcpy(winBuf.data(), WIN_SIZE, winAddr, WIN_SIZE, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        OP_LOGE(OP_NAME, "aclrtMemcpy win from device to host failed. ret = %d", ret);
        return -1;
    }
    return 0;
}

inline int ProcessArgsForA2(const char *groupName, std::vector<uint8_t> &winBuf)
{
    if (groupName == nullptr || strnlen(groupName, MAX_GROUP_NAME_LENGTH) == 0 ||
        strnlen(groupName, MAX_GROUP_NAME_LENGTH) == MAX_GROUP_NAME_LENGTH) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "groupName", groupName,
                                              "The value of groupName must be within the supported range.");
        return -1;
    }
    uint64_t size;
    void *winAddr = nullptr;
    Mc2Hcom::MC2HcomTopology::CommGetHcclBufferByGroup(groupName, &winAddr, &size);
    if (winAddr == nullptr) {
        OP_LOGE(OP_NAME, "Get win addr failed.");
        return -1;
    }
    OP_LOGE(OP_NAME, "HCCL BUFFER SIZE=%luMB. WindowInAddr=%p.", size / (1 * 1024 * 1024) / 2, winAddr);
    auto ret = aclrtMemcpy(winBuf.data(), WIN_SIZE, winAddr, WIN_SIZE, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        OP_LOGE(OP_NAME, "aclrtMemcpy win from device to host failed. ret = %d", ret);
        return -1;
    }
    return 0;
}

// 获取 peermem 基址，用于 MegaMoe 结构化 dump
inline void *GetPeermemBase(const char *socName, uint64_t argsAddr)
{
    OP_LOGD(OP_NAME, "GetPeermemBase: argsAddr=0x%lx", argsAddr);
    if (std::strstr(socName, "Ascend950") != nullptr) {
        std::vector<uint8_t> hcclArgs(sizeof(Mc2Aclnn::Mc2MoeContext), 0);
        if (aclrtMemcpy(hcclArgs.data(), sizeof(Mc2Aclnn::Mc2MoeContext), reinterpret_cast<void *>(argsAddr),
                        sizeof(Mc2Aclnn::Mc2MoeContext), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            OP_LOGE(OP_NAME, "GetPeermemBase: aclrtMemcpy Mc2MoeContext failed");
            return nullptr;
        }
        auto *ctx = reinterpret_cast<Mc2Aclnn::Mc2MoeContext *>(hcclArgs.data());
        if (ctx->epRankId >= Mc2Aclnn::HCCL_MAX_RANK_SIZE) {
            OP_LOGE(OP_NAME, "GetPeermemBase: invalid epRankId=%u", ctx->epRankId);
            return nullptr;
        }
        void *winAddr = reinterpret_cast<void *>(ctx->epHcclBuffer_[ctx->epRankId]);
        OP_LOGD(OP_NAME, "GetPeermemBase: epRankId=%u peermemBase=%p", ctx->epRankId, winAddr);
        return winAddr;
    }

    std::vector<uint8_t> hcclArgs(sizeof(CommContextForDump), 0);
    if (aclrtMemcpy(hcclArgs.data(), sizeof(CommContextForDump), reinterpret_cast<void *>(argsAddr),
                    sizeof(CommContextForDump), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        OP_LOGE(OP_NAME, "GetPeermemBase: aclrtMemcpy CommContextForDump failed");
        return nullptr;
    }
    CommContextForDump *ctx = reinterpret_cast<CommContextForDump *>(hcclArgs.data());
    OP_LOGD(OP_NAME, "GetPeermemBase: epRankId=%u", ctx->epRankId);
    void *winAddr = reinterpret_cast<void *>(ctx->epHcclBufffer_[ctx->epRankId]);
    OP_LOGD(OP_NAME, "GetPeermemBase: peermemBase=%p", winAddr);
    return winAddr;
}

inline int ParseStructuredDump(uint64_t argsAddr, std::vector<uint8_t> &winBuf)
{
    // Phase 1: 从 device 拷贝 Header 到 host（读取固定区域大小的前 sizeof(Header) 字节）
    OP_LOGD(OP_NAME, "ParseStructuredDump Phase1: copy Header from device addr=0x%lx, regionSize=%zu", argsAddr,
            MC2ExceptionDump::DUMP_HEADER_REGION_SIZE);
    MC2ExceptionDump::Header hdr;
    aclError ret = aclrtMemcpy(&hdr, sizeof(MC2ExceptionDump::Header), reinterpret_cast<void *>(argsAddr),
                               sizeof(MC2ExceptionDump::Header), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        OP_LOGE(OP_NAME, "aclrtMemcpy Header failed. ret=%d", ret);
        return -1;
    }
    OP_LOGD(OP_NAME, "Phase1: headerMagic=0x%x opType=%d stageNum=%d tilingDataSize=%u numBlocks=%u dumpCount=%lu",
            hdr.headerMagic, hdr.opType, hdr.stageNum, hdr.tilingDataSize, hdr.numBlocks, hdr.dumpCount);
    if (hdr.headerMagic != MC2ExceptionDump::HEADER_MAGIC) {
        OP_LOGE(OP_NAME, "headerMagic mismatch: 0x%x != 0x%x, ExceptionDump not enabled", hdr.headerMagic,
                MC2ExceptionDump::HEADER_MAGIC);
        return -1;
    }

    uint64_t dumpCount = hdr.dumpCount;
    if (dumpCount > MC2ExceptionDump::MAX_DUMP_ENTRIES) {
        OP_LOGE(OP_NAME, "dumpCount=%lu > MAX_DUMP_ENTRIES=%d, truncating", dumpCount,
                MC2ExceptionDump::MAX_DUMP_ENTRIES);
        dumpCount = MC2ExceptionDump::MAX_DUMP_ENTRIES;
    }
    uint32_t tilingDataSize = hdr.tilingDataSize;
    if (tilingDataSize > MC2ExceptionDump::MAX_DUMP_TILING_SIZE) {
        OP_LOGE(OP_NAME, "tilingDataSize=%u > MAX=%zu, truncating", tilingDataSize,
                MC2ExceptionDump::MAX_DUMP_TILING_SIZE);
        tilingDataSize = MC2ExceptionDump::MAX_DUMP_TILING_SIZE;
    }
    uint32_t numBlocks = hdr.numBlocks;
    if (numBlocks == 0) {
        OP_LOGE(OP_NAME, "numBlocks=0, Header not initialized");
        return -1;
    }

    size_t headerSize = sizeof(MC2ExceptionDump::Header);
    size_t blockStageSize = sizeof(uint64_t) + hdr.stageNum * sizeof(uint64_t);
    size_t blockStageTotalSize = static_cast<size_t>(numBlocks) * blockStageSize;
    size_t dumpParamsSize = static_cast<size_t>(dumpCount) * sizeof(MC2ExceptionDump::DumpParams);
    OP_LOGD(OP_NAME, "Phase1: headerSize=%zu blockStageSize=%zu blockStageTotalSize=%zu dumpParamsSize=%zu", headerSize,
            blockStageSize, blockStageTotalSize, dumpParamsSize);

    // Device 内存偏移计算（使用固定区域大小）
    uint64_t tilingOffset = argsAddr + MC2ExceptionDump::DUMP_HEADER_REGION_SIZE;
    uint64_t blockStageOffset = tilingOffset + MC2ExceptionDump::DUMP_TILING_REGION_SIZE;
    uint64_t dumpParamsOffset = blockStageOffset + MC2ExceptionDump::DUMP_BLOCK_STAGE_REGION_SIZE;
    OP_LOGD(OP_NAME, "Phase1: tilingOffset=0x%lx blockStageOffset=0x%lx dumpParamsOffset=0x%lx", tilingOffset,
            blockStageOffset, dumpParamsOffset);

    // Phase 2: 从 device 拷贝 DumpParams 到临时 host buffer，计算 dump 数据总大小
    OP_LOGD(OP_NAME, "Phase2: copy DumpParams from device addr=0x%lx, size=%zu", dumpParamsOffset, dumpParamsSize);
    std::vector<uint8_t> dumpParamsBuf(dumpParamsSize, 0);
    ret = aclrtMemcpy(dumpParamsBuf.data(), dumpParamsSize, reinterpret_cast<void *>(dumpParamsOffset), dumpParamsSize,
                      ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        OP_LOGE(OP_NAME, "aclrtMemcpy DumpParams failed. ret=%d", ret);
        return -1;
    }

    size_t totalDumpDataSize = 0;
    for (uint64_t i = 0; i < dumpCount; i++) {
        const MC2ExceptionDump::DumpParams *dp = reinterpret_cast<const MC2ExceptionDump::DumpParams *>(
            dumpParamsBuf.data() + i * sizeof(MC2ExceptionDump::DumpParams));
        if (dp->dumpAddr != 0) {
            OP_LOGD(OP_NAME, "Phase2: dumpEntry[%lu] addr=0x%lx blockCount=%zu blockLen=%zu srcStride=%zu", i,
                    dp->dumpAddr, dp->blockCount, dp->blockLen, dp->srcStride);
            totalDumpDataSize += dp->blockCount * dp->blockLen;
        } else {
            OP_LOGD(OP_NAME, "Phase2: dumpEntry[%lu] addr=0, skipped", i);
        }
    }
    OP_LOGD(OP_NAME, "Phase2: totalDumpDataSize=%zu", totalDumpDataSize);

    // Phase 3: 一次性 resize winBuf，将所有数据连续拷入
    // 布局: [Header][Tiling][BlockStage[]][DumpParams[]][dump data...]
    size_t totalSize = headerSize + tilingDataSize + blockStageTotalSize + dumpParamsSize + totalDumpDataSize;
    OP_LOGD(OP_NAME, "Phase3: totalSize=%zu, resizing winBuf", totalSize);
    winBuf.resize(totalSize, 0);
    size_t offset = 0;

    // 3a. Header
    OP_LOGD(OP_NAME, "Phase3a: copy Header to winBuf offset=%zu, size=%zu", offset, headerSize);
    memcpy(winBuf.data() + offset, &hdr, headerSize);
    offset += headerSize;

    // 3b. Tiling（从固定区域偏移读取，按实际大小拷贝）
    OP_LOGD(OP_NAME, "Phase3b: copy Tiling from device addr=0x%lx, size=%zu, offset=%zu", tilingOffset, tilingDataSize,
            offset);
    if (tilingDataSize > 0) {
        ret = aclrtMemcpy(winBuf.data() + offset, tilingDataSize, reinterpret_cast<void *>(tilingOffset),
                          tilingDataSize, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            OP_LOGE(OP_NAME, "aclrtMemcpy Tiling failed. ret=%d", ret);
        }
    }
    offset += tilingDataSize;

    // 3c. BlockStage（从固定区域偏移读取，按实际大小拷贝）
    OP_LOGD(OP_NAME, "Phase3c: copy BlockStage from device addr=0x%lx, size=%zu, offset=%zu", blockStageOffset,
            blockStageTotalSize, offset);
    if (blockStageTotalSize > 0) {
        ret = aclrtMemcpy(winBuf.data() + offset, blockStageTotalSize, reinterpret_cast<void *>(blockStageOffset),
                          blockStageTotalSize, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            OP_LOGE(OP_NAME, "aclrtMemcpy BlockStage failed. ret=%d", ret);
        }
    }
    offset += blockStageTotalSize;

    // 3d. DumpParams（已在 host）
    OP_LOGD(OP_NAME, "Phase3d: copy DumpParams to winBuf offset=%zu, size=%zu", offset, dumpParamsSize);
    memcpy(winBuf.data() + offset, dumpParamsBuf.data(), dumpParamsSize);
    offset += dumpParamsSize;

    // 3e. 逐条搬运 dump 数据（从 device 各 dumpAddr 搬到 winBuf 末尾连续区域）
    for (uint64_t i = 0; i < dumpCount; i++) {
        const MC2ExceptionDump::DumpParams *dp = reinterpret_cast<const MC2ExceptionDump::DumpParams *>(
            dumpParamsBuf.data() + i * sizeof(MC2ExceptionDump::DumpParams));
        if (dp->dumpAddr == 0) {
            continue;
        }
        OP_LOGD(OP_NAME, "Phase3e: dumpEntry[%lu] addr=0x%lx blockCount=%zu blockLen=%zu srcStride=%zu offset=%zu", i,
                dp->dumpAddr, dp->blockCount, dp->blockLen, dp->srcStride, offset);
        for (size_t b = 0; b < dp->blockCount; b++) {
            void *srcAddr = reinterpret_cast<void *>(dp->dumpAddr + b * dp->srcStride);
            ret = aclrtMemcpy(winBuf.data() + offset, dp->blockLen, srcAddr, dp->blockLen, ACL_MEMCPY_DEVICE_TO_HOST);
            if (ret != ACL_SUCCESS) {
                OP_LOGE(OP_NAME, "aclrtMemcpy dump entry %lu block %zu failed. ret=%d", i, b, ret);
                break;
            }
            offset += dp->blockLen;
        }
    }

    OP_LOGE(OP_NAME, "Structured dump completed: totalSize=%zu dumpCount=%lu winBufSize=%zu", totalSize, dumpCount,
            winBuf.size());
    return 0;
}

inline void Mc2ExceptionImpl(aclrtExceptionInfo *args, void *userdata, const char *op)
{
    const char *socName = aclrtGetSocName();
    if ((std::strstr(socName, "Ascend950") == nullptr) && (std::strstr(socName, "Ascend910_93") == nullptr) &&
        (std::strstr(socName, "Ascend910B") == nullptr)) {
        OP_LOGE(OP_NAME, "The soc version is %s, skip dump process", socName);
        return;
    }

    OP_LOGD(OP_NAME, "Start to handle mc2 exception: soc=%s op=%s", socName, op);

    // Get addr of hccl context from ExceptionInfo
    void *devArgsPtr = nullptr;
    uint32_t devArgsLen = 0;

    auto aclrtGetArgsFromExceptionInfoFunc = GetAclrtGetArgsFromExceptionInfoFunc();
    if (aclrtGetArgsFromExceptionInfoFunc == nullptr) {
        OP_LOGE(OP_NAME, "Failed to load aclrtGetArgsFromExceptionInfo function.");
        return;
    }

    auto ret = aclrtGetArgsFromExceptionInfoFunc(args, &devArgsPtr, &devArgsLen);
    if (ret != ACL_SUCCESS) {
        OP_LOGE(OP_NAME, "aclrtGetArgsFromExceptionInfo failed. ret=%d", ret);
        return;
    }

    uint32_t deviceId = aclrtGetDeviceIdFromExceptionInfo(args);
    if (ret != ACL_SUCCESS) {
        OP_LOGE(OP_NAME, "aclrtGetDeviceIdFromExceptionInfo failed. ret=%d", ret);
        return;
    }
    OP_LOGD(OP_NAME, "Get context from args. deviceId=%u, devArgsAddr=%p, devArgsLen=%u", deviceId, devArgsPtr,
            devArgsLen);

    uint64_t argsAddr = 0;
    uint64_t argsOffset = 0;
    // 由于dispatchv3中包含syncacll，推断为mix算子，其args的首地址是ffts，第二个地址才是hcclcontext
    if ((std::strstr(socName, "Ascend910_93") != nullptr) && (std::strstr(op, "MoeDistributeDispatchV3") != nullptr)) {
        argsOffset = sizeof(uint64_t);
    }

    if ((std::strstr(socName, "Ascend950") == nullptr) && (std::strstr(op, "MegaMoe") != nullptr)) {
        argsOffset = sizeof(uint64_t);
    }
    ret =
        aclrtMemcpy(&argsAddr, sizeof(uint64_t), devArgsPtr + argsOffset, sizeof(uint64_t), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        OP_LOGE(OP_NAME, "aclrtMemcpy address of args failed. ret=%d", ret);
        return;
    }
    OP_LOGD(OP_NAME, "argsAddr=0x%lx (argsOffset=%lu)", argsAddr, argsOffset);

    // Get win content
    std::vector<uint8_t> winContent;
    bool isMegaMoe = (std::strstr(op, "MegaMoe") != nullptr);
    OP_LOGD(OP_NAME, "isMegaMoe=%d", isMegaMoe);

    if (isMegaMoe) {
        // MegaMoe: 结构化 dump（从 device 按 3 阶段搬运到 host）
        OP_LOGD(OP_NAME, "MegaMoe structured dump: getting peermem base");
        void *peermemBase = GetPeermemBase(socName, argsAddr);
        if (peermemBase == nullptr) {
            OP_LOGE(OP_NAME, "Failed to get peermem base.");
            return;
        }
        if (ParseStructuredDump(reinterpret_cast<uint64_t>(peermemBase), winContent) != 0) {
            OP_LOGE(OP_NAME, "Failed to parse structured dump.");
            return;
        }
    } else {
        // 其他场景: 原始 1MB raw dump
        OP_LOGD(OP_NAME, "Raw dump: winContent resize to WIN_SIZE=%u", WIN_SIZE);
        winContent.resize(WIN_SIZE, 0);
        if (std::strstr(socName, "Ascend910_93") != nullptr) {
            if (ProcessArgsForA3(argsAddr, winContent) != 0) {
                OP_LOGE(OP_NAME, "Failed to get win content.");
                return;
            }
        } else if (std::strstr(socName, "Ascend950") != nullptr) {
            if (ProcessArgsForA5(argsAddr, winContent, op) != 0) {
                OP_LOGE(OP_NAME, "Failed to get win content.");
                return;
            }
        } else if (std::strstr(socName, "Ascend910B") != nullptr) {
            const char *groupName = MC2GroupNameManager::GetInstance().GetGroupName();
            if (ProcessArgsForA2(groupName, winContent) != 0) {
                OP_LOGE(OP_NAME, "Failed to get win content.");
                return;
            }
        }
    }

    // 获取acldumpGetPath函数指针
    auto acldumpGetPathFunc = GetAcldumpGetPathFunc();
    if (acldumpGetPathFunc == nullptr) {
        OP_LOGE(OP_NAME, "Failed to load acldumpGetPath function, skip dump");
        return;
    }

    // 获取dump路径
    const char *dumpPath = acldumpGetPathFunc(acldumpType::AIC_ERR_BRIEF_DUMP);
    if (dumpPath == nullptr) {
        OP_LOGE(OP_NAME, "acldumpGetPath returned NULL");
        return;
    }
    OP_LOGD(OP_NAME, "dumpPath=%s, winContent.size()=%zu", dumpPath, winContent.size());

    // Write to bin file
    std::string fileName = GenDumpFileName(args, op);
    OP_LOGD(OP_NAME, "Writing dump file: deviceId=%u fileName=%s bufSize=%zu", deviceId, fileName.c_str(),
            winContent.size());
    if (DumpToFile(std::string(dumpPath), fileName, deviceId, winContent.data(), winContent.size()) != 0) {
        OP_LOGE(OP_NAME, "Failed to dump win content.");
    }
}

#if MC2_DFX_ENABLE
inline void Mc2ExceptionImplTmp(aclrtExceptionInfo *args, void *userdata, const char *op)
{
    const char *socName = aclrtGetSocName();
    if (std::strstr(socName, "Ascend950") == nullptr) {
        OP_LOGE(OP_NAME, "The soc version is %s, skip dump process", socName);
        return;
    }

    OP_LOGD(OP_NAME, "Start to handle mc2 exception and dump xn/cke info.");

    // 1. 动态加载 aclrtGetArgsFromExceptionInfo 函数
    auto aclrtGetArgsFromExceptionInfoFunc = GetAclrtGetArgsFromExceptionInfoFunc();
    if (aclrtGetArgsFromExceptionInfoFunc == nullptr) {
        OP_LOGE(OP_NAME, "Failed to load aclrtGetArgsFromExceptionInfo function.");
        return;
    }

    // 2. 调用获取函数，拿到 devArgsPtr 和 devArgsLen
    void *devArgsPtr = nullptr;
    uint32_t devArgsLen = 0;
    auto ret = aclrtGetArgsFromExceptionInfoFunc(args, &devArgsPtr, &devArgsLen);
    if (ret != ACL_SUCCESS) {
        OP_LOGE(OP_NAME, "aclrtGetArgsFromExceptionInfo failed. ret=%d", ret);
        return;
    }

    // 3. 获取报错的 deviceId
    uint32_t deviceId = aclrtGetDeviceIdFromExceptionInfo(args);
    OP_LOGD(OP_NAME, "Get context from args. deviceId=%u, devArgsAddr=%p, devArgsLen=%u", deviceId, devArgsPtr,
            devArgsLen);

    // 4. args 首地址中的内容即为 hccl context 的 device 地址，做一次 D2H copy（无 argsOffset）
    uint64_t argsAddr = 0;
    ret = aclrtMemcpy(&argsAddr, sizeof(uint64_t), devArgsPtr, sizeof(uint64_t), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        OP_LOGE(OP_NAME, "aclrtMemcpy address of args failed. ret=%d", ret);
        return;
    }

    // 5. 根据 hccl context 的 device 地址，做一次 D2H copy，大小为 HcclCombinOpParam
    std::vector<uint8_t> hcclArgs(sizeof(HcclCombinOpParam), 0);
    ret = aclrtMemcpy(hcclArgs.data(), sizeof(HcclCombinOpParam), (void *)argsAddr, sizeof(HcclCombinOpParam),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        OP_LOGE(OP_NAME, "aclrtMemcpy HcclCombinOpParam from device to host failed. ret = %d", ret);
        return;
    }
    HcclCombinOpParam *winContext = reinterpret_cast<HcclCombinOpParam *>(hcclArgs.data());
    if (winContext == nullptr) {
        OP_LOGE(OP_NAME, "Cast to winContext failed. HcclCombinOpParam is null.");
        return;
    }
    OP_LOGE(OP_NAME,
            "HcclCombinOpParam: workSpace=0x%lx, workSpaceSize=%lu, rankId=%u, rankDim=%u, "
            "xnAddr=0x%lx, ckeAddr=0x%lx",
            winContext->workSpace, winContext->workSpaceSize, winContext->rankId, winContext->rankDim,
            winContext->xnAddr, winContext->ckeAddr);

    // 6. 解析 HcclCombinOpParam，拿到 xnAddr 和 ckeAddr，分别做 D2H copy 并落盘
    auto acldumpGetPathFunc = GetAcldumpGetPathFunc();
    if (acldumpGetPathFunc == nullptr) {
        OP_LOGE(OP_NAME, "Failed to load acldumpGetPath function, skip dump");
        return;
    }
    const char *dumpPath = acldumpGetPathFunc(acldumpType::AIC_ERR_BRIEF_DUMP);
    if (dumpPath == nullptr) {
        OP_LOGE(OP_NAME, "acldumpGetPath returned NULL");
        return;
    }

    // xnAddr: 64*64*8 bytes
    if (winContext->xnAddr == 0) {
        OP_LOGE(OP_NAME, "xnAddr is 0, skip xn dump.");
    } else {
        std::vector<uint8_t> xnBuf(XN_ADDR_DUMP_SIZE, 0);
        ret = aclrtMemcpy(xnBuf.data(), XN_ADDR_DUMP_SIZE, (void *)winContext->xnAddr, XN_ADDR_DUMP_SIZE,
                          ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            OP_LOGE(OP_NAME, "aclrtMemcpy xnAddr from device to host failed. ret = %d", ret);
        } else if (DumpToFileWithSize(std::string(dumpPath), GenDumpFileNameWithPrefix(args, op, "xn_addr_info_"),
                                      deviceId, xnBuf.data(), XN_ADDR_DUMP_SIZE) != 0) {
            OP_LOGE(OP_NAME, "Failed to dump xn addr info.");
        }
    }

    // ckeAddr: 64*2*8 bytes
    if (winContext->ckeAddr == 0) {
        OP_LOGE(OP_NAME, "ckeAddr is 0, skip cke dump.");
    } else {
        std::vector<uint8_t> ckeBuf(CKE_ADDR_DUMP_SIZE, 0);
        ret = aclrtMemcpy(ckeBuf.data(), CKE_ADDR_DUMP_SIZE, (void *)winContext->ckeAddr, CKE_ADDR_DUMP_SIZE,
                          ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            OP_LOGE(OP_NAME, "aclrtMemcpy ckeAddr from device to host failed. ret = %d", ret);
        } else if (DumpToFileWithSize(std::string(dumpPath), GenDumpFileNameWithPrefix(args, op, "cke_addr_info_"),
                                      deviceId, ckeBuf.data(), CKE_ADDR_DUMP_SIZE) != 0) {
            OP_LOGE(OP_NAME, "Failed to dump cke addr info.");
        }
    }
}

inline void Mc2DumpTilingAndWorkspace(aclrtExceptionInfo *args, const char *op, uint32_t tilingGmArgIdx,
                                      uint32_t workspaceGmArgIdx = 0U, uint32_t dfxInfoOffset = 0U)
{
    if (args == nullptr || op == nullptr) {
        return;
    }

    auto getArgsFunc = GetAclrtGetArgsFromExceptionInfoFunc();
    if (getArgsFunc == nullptr) {
        OP_LOGE(OP_NAME, "Failed to load aclrtGetArgsFromExceptionInfo function.");
        return;
    }

    void *devArgsPtr = nullptr;
    uint32_t devArgsLen = 0U;
    if (getArgsFunc(args, &devArgsPtr, &devArgsLen) != ACL_SUCCESS) {
        OP_LOGE(OP_NAME, "aclrtGetArgsFromExceptionInfo failed.");
        return;
    }

    uint32_t deviceId = aclrtGetDeviceIdFromExceptionInfo(args);
    OP_LOGE(OP_NAME, "Tiling dump start: op=%s, devArgsPtr=%p, devArgsLen=%u, tilingGmArgIdx=%u", op, devArgsPtr,
            devArgsLen, tilingGmArgIdx);

    uint32_t tilingGmOffset = tilingGmArgIdx * sizeof(uint64_t);
    if (tilingGmOffset + sizeof(uint64_t) > devArgsLen) {
        OP_LOGE(OP_NAME, "tilingGmOffset=%u + 8 > devArgsLen=%u, args too short.", tilingGmOffset, devArgsLen);
        return;
    }

    constexpr uint32_t argsDumpSize = 256U;
    uint32_t actualArgsDumpSize = std::min(devArgsLen, argsDumpSize);
    std::vector<uint8_t> argsBuf(actualArgsDumpSize, 0);
    if (aclrtMemcpy(argsBuf.data(), actualArgsDumpSize, devArgsPtr, actualArgsDumpSize, ACL_MEMCPY_DEVICE_TO_HOST) !=
        ACL_SUCCESS) {
        OP_LOGE(OP_NAME, "aclrtMemcpy args buffer failed.");
        return;
    }

    auto dumpPathFunc = GetAcldumpGetPathFunc();
    if (dumpPathFunc == nullptr) {
        OP_LOGE(OP_NAME, "Failed to load acldumpGetPath function.");
        return;
    }
    const char *dumpPath = dumpPathFunc(acldumpType::AIC_ERR_BRIEF_DUMP);
    if (dumpPath == nullptr) {
        OP_LOGE(OP_NAME, "acldumpGetPath returned NULL.");
        return;
    }

    if (DumpToFileWithSize(std::string(dumpPath), GenDumpFileNameWithPrefix(args, op, "args_info_"), deviceId,
                           argsBuf.data(), actualArgsDumpSize) != 0) {
        OP_LOGE(OP_NAME, "Failed to dump args info.");
    }

    uint64_t tilingGM = 0U;
    if (aclrtMemcpy(&tilingGM, sizeof(uint64_t), reinterpret_cast<uint8_t *>(devArgsPtr) + tilingGmOffset,
                    sizeof(uint64_t), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        OP_LOGE(OP_NAME, "aclrtMemcpy tilingGM pointer failed, offset=%u.", tilingGmOffset);
        return;
    }
    OP_LOGE(OP_NAME, "tilingGM=0x%lx (offset=%u)", tilingGM, tilingGmOffset);
    if (tilingGM == 0U) {
        OP_LOGE(OP_NAME, "tilingGM is 0, skip tiling data dump.");
        return;
    }

    constexpr uint32_t tilingDumpSize = 2048U;
    std::vector<uint8_t> tilingBuf(tilingDumpSize, 0);
    if (aclrtMemcpy(tilingBuf.data(), tilingDumpSize, reinterpret_cast<void *>(tilingGM), tilingDumpSize,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        OP_LOGE(OP_NAME, "aclrtMemcpy tiling data failed.");
        return;
    }
    if (DumpToFileWithSize(std::string(dumpPath), GenDumpFileNameWithPrefix(args, op, "tiling_data_"), deviceId,
                           tilingBuf.data(), tilingDumpSize) != 0) {
        OP_LOGE(OP_NAME, "Failed to dump tiling data.");
    }
    OP_LOGE(OP_NAME, "Tiling dump done: op=%s, deviceId=%u.", op, deviceId);

    // ====== Workspace dump（workspaceGmArgIdx != 0 时执行）======
    if (workspaceGmArgIdx == 0U) {
        return;
    }

    OP_LOGE(OP_NAME, "Workspace dump start: op=%s, wsGmArgIdx=%u, dfxInfoOffset=%u", op, workspaceGmArgIdx,
            dfxInfoOffset);

    // 1. 解析 DfxDumpInfo（DFX 约定: dumpInfo 位于 tiling 结构 mc2InitTiling/mc2CcTiling 之后，
    //    调用方按 offsetof(..., dumpInfo) 传入偏移；dumpInfo 仍在 offset 0 的算子走默认值 0）
    constexpr uint32_t dfxDumpInfoSize = sizeof(Utils::DfxDumpInfo);
    static_assert(dfxDumpInfoSize <= tilingDumpSize, "DfxDumpInfo exceeds tiling dump size");
    if (dfxInfoOffset + dfxDumpInfoSize > tilingDumpSize) {
        OP_LOGE(OP_NAME, "dfxInfoOffset=%u + dfxDumpInfoSize=%u > tilingDumpSize=%u, skip workspace dump.",
                dfxInfoOffset, dfxDumpInfoSize, tilingDumpSize);
        return;
    }
    Utils::DfxDumpInfo dfxInfo;
    memcpy(&dfxInfo, tilingBuf.data() + dfxInfoOffset, dfxDumpInfoSize);
    const Utils::DfxWorkspaceLayoutInfo &wsLayout = dfxInfo.workspaceLayout;
    if (wsLayout.totalSize == 0U || wsLayout.segCount == 0U || wsLayout.segCount > Utils::MAX_WORKSPACE_SEGMENTS) {
        OP_LOGE(OP_NAME, "Invalid DfxWorkspaceLayoutInfo, skip workspace dump.");
        return;
    }
    OP_LOGE(OP_NAME, "wsLayout valid: totalSize=%lu, segCount=%u.", wsLayout.totalSize, wsLayout.segCount);

    // 2. 读取 workspaceGM 指针 (workspaceGmArgIdx = tilingGmArgIdx - 1)
    uint32_t wsGmOffset = workspaceGmArgIdx * sizeof(uint64_t);
    if (wsGmOffset + sizeof(uint64_t) > devArgsLen) {
        OP_LOGE(OP_NAME, "wsGmOffset=%u + 8 > devArgsLen=%u, args too short.", wsGmOffset, devArgsLen);
        return;
    }
    uint64_t workspaceGM = 0U;
    if (aclrtMemcpy(&workspaceGM, sizeof(uint64_t), reinterpret_cast<uint8_t *>(devArgsPtr) + wsGmOffset,
                    sizeof(uint64_t), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        OP_LOGE(OP_NAME, "aclrtMemcpy workspaceGM pointer failed, offset=%u.", wsGmOffset);
        return;
    }
    OP_LOGE(OP_NAME, "workspaceGM=0x%lx (offset=%u)", workspaceGM, wsGmOffset);
    if (workspaceGM == 0U) {
        OP_LOGE(OP_NAME, "workspaceGM is 0, skip workspace dump.");
        return;
    }

    // 3. 逐段 D2H 拷贝并落盘
    for (uint32_t i = 0; i < wsLayout.segCount && i < Utils::MAX_WORKSPACE_SEGMENTS; i++) {
        auto &seg = wsLayout.segments[i];
        if (seg.size == 0) {
            continue;
        }
        uint64_t dumpSize = seg.size;
        void *srcAddr = reinterpret_cast<void *>(workspaceGM + seg.offset);
        OP_LOGE(OP_NAME, "seg[%u]: type=%u, offset=%lu, size=%lu, dumpSize=%lu, srcAddr=0x%lx", i, seg.type, seg.offset,
                seg.size, dumpSize, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(srcAddr)));
        try {
            std::vector<uint8_t> segBuf(dumpSize, 0);
            if (aclrtMemcpy(segBuf.data(), dumpSize, srcAddr, dumpSize, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
                OP_LOGE(OP_NAME, "aclrtMemcpy workspace seg[%u] failed, offset=%lu, size=%lu.", i, seg.offset,
                        dumpSize);
                continue;
            }
            std::string segPrefix = "workspace_seg" + std::to_string(i) + "_type" + std::to_string(seg.type) + "_";
            std::string segName = GenDumpFileNameWithPrefix(args, op, segPrefix.c_str());
            if (DumpToFileWithSize(std::string(dumpPath), segName, deviceId, segBuf.data(), dumpSize) != 0) {
                OP_LOGE(OP_NAME, "Failed to dump workspace seg[%u].", i);
            }
        } catch (const std::bad_alloc &) {
            OP_LOGE(OP_NAME, "workspace seg[%u] host alloc %lu bytes failed, skip.", i, dumpSize);
            continue;
        } catch (...) {
            OP_LOGE(OP_NAME, "workspace seg[%u] dump exception, skip.", i);
            continue;
        }
    }
    OP_LOGE(OP_NAME, "Workspace dump done: op=%s, deviceId=%u.", op, deviceId);
}

inline void Mc2DumpUrmaContext(aclrtExceptionInfo *args, const char *op, uint32_t contextArgIdx,
                               uint32_t tilingGmArgIdx = 0U)
{
    if (args == nullptr || op == nullptr) {
        return;
    }

    auto getArgsFunc = GetAclrtGetArgsFromExceptionInfoFunc();
    if (getArgsFunc == nullptr) {
        OP_LOGE(OP_NAME, "Failed to load aclrtGetArgsFromExceptionInfo function.");
        return;
    }

    void *devArgsPtr = nullptr;
    uint32_t devArgsLen = 0U;
    if (getArgsFunc(args, &devArgsPtr, &devArgsLen) != ACL_SUCCESS) {
        OP_LOGE(OP_NAME, "aclrtGetArgsFromExceptionInfo failed.");
        return;
    }

    uint32_t deviceId = aclrtGetDeviceIdFromExceptionInfo(args);
    OP_LOGE(OP_NAME, "UrmaContext dump start: op=%s, contextArgIdx=%u", op, contextArgIdx);

    uint32_t contextOffset = contextArgIdx * sizeof(uint64_t);
    if (contextOffset + sizeof(uint64_t) > devArgsLen) {
        OP_LOGE(OP_NAME, "contextOffset=%u + 8 > devArgsLen=%u, args too short.", contextOffset, devArgsLen);
        return;
    }

    auto dumpPathFunc = GetAcldumpGetPathFunc();
    if (dumpPathFunc == nullptr) {
        OP_LOGE(OP_NAME, "Failed to load acldumpGetPath function.");
        return;
    }
    const char *dumpPath = dumpPathFunc(acldumpType::AIC_ERR_BRIEF_DUMP);
    if (dumpPath == nullptr) {
        OP_LOGE(OP_NAME, "acldumpGetPath returned NULL.");
        return;
    }

    // 1. 读取 context 指针 (arg 0)
    uint64_t contextAddr = 0U;
    if (aclrtMemcpy(&contextAddr, sizeof(uint64_t), reinterpret_cast<uint8_t *>(devArgsPtr) + contextOffset,
                    sizeof(uint64_t), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        OP_LOGE(OP_NAME, "aclrtMemcpy context pointer failed, offset=%u.", contextOffset);
        return;
    }
    OP_LOGE(OP_NAME, "contextAddr=0x%lx (offset=%u)", contextAddr, contextOffset);
    if (contextAddr == 0U) {
        OP_LOGE(OP_NAME, "contextAddr is 0, skip urma context dump.");
        return;
    }

    // 2. D2H 拷贝 CommContext (UrmaCommContextFullForDump, udmaCtx + ubmemCtx) -> comm_context
    constexpr uint32_t ctxDumpSize = sizeof(UrmaCommContextFullForDump);
    std::vector<uint8_t> ctxBuf(ctxDumpSize, 0);
    if (aclrtMemcpy(ctxBuf.data(), ctxDumpSize, reinterpret_cast<void *>(contextAddr), ctxDumpSize,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        OP_LOGE(OP_NAME, "aclrtMemcpy UrmaCommContext failed.");
        return;
    }
    if (DumpToFileWithSize(std::string(dumpPath), GenDumpFileNameWithPrefix(args, op, "comm_context_"), deviceId,
                           ctxBuf.data(), ctxDumpSize) != 0) {
        OP_LOGE(OP_NAME, "Failed to dump comm context.");
    }

    // 3. 解析 peermem 窗口地址
    UrmaCommContextFullForDump *urmaCtx = reinterpret_cast<UrmaCommContextFullForDump *>(ctxBuf.data());
    uint32_t rankId = urmaCtx->udmaCtx.rankId;
    OP_LOGE(OP_NAME, "urmaCtx: rankId=%u, rankSize=%u", rankId, urmaCtx->udmaCtx.rankSize);
    if (rankId >= URMA_MAX_RANK_NUM) {
        OP_LOGE(OP_NAME, "rankId=%u >= URMA_MAX_RANK_NUM=%u, skip peermem dump.", rankId, URMA_MAX_RANK_NUM);
        return;
    }
    uint64_t selfWinAddr = urmaCtx->udmaCtx.commBufferAddrs[rankId];
    OP_LOGE(OP_NAME, "selfWinAddr=0x%lx (commBufferAddrs[%u])", selfWinAddr, rankId);
    if (selfWinAddr == 0U) {
        OP_LOGE(OP_NAME, "selfWinAddr is 0, skip peermem dump.");
        return;
    }

    // 4. 从 tiling 头部解析 DfxDumpInfo（DFX 头部约定: dumpInfo 为 tiling 数据第一个成员），确定 peermem dump 大小
    uint32_t peermemDumpSize = WIN_SIZE;
    uint64_t tilingGM = 0U;
    Utils::DfxPeermemLayoutInfo peermemLayout;
    std::memset(&peermemLayout, 0, sizeof(peermemLayout));
    if (tilingGmArgIdx != 0U) {
        uint32_t tilingGmOffset = tilingGmArgIdx * sizeof(uint64_t);
        if (tilingGmOffset + sizeof(uint64_t) <= devArgsLen) {
            if (aclrtMemcpy(&tilingGM, sizeof(uint64_t), reinterpret_cast<uint8_t *>(devArgsPtr) + tilingGmOffset,
                            sizeof(uint64_t), ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS &&
                tilingGM != 0U) {
                Utils::DfxDumpInfo dfxInfo;
                std::memset(&dfxInfo, 0, sizeof(dfxInfo));
                if (aclrtMemcpy(&dfxInfo, sizeof(dfxInfo), reinterpret_cast<void *>(tilingGM), sizeof(dfxInfo),
                                ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS) {
                    peermemLayout = dfxInfo.peermemLayout;
                    if (dfxInfo.peermemDataSize > 0 && dfxInfo.peermemDataSize <= MAX_PEERMEM_DUMP_SIZE) {
                        peermemDumpSize = static_cast<uint32_t>(dfxInfo.peermemDataSize);
                    } else {
                        OP_LOGE(OP_NAME, "peermemDataSize=%lu, using fallback WIN_SIZE=%u", dfxInfo.peermemDataSize,
                                WIN_SIZE);
                    }
                }
            }
        }
    }
    OP_LOGE(OP_NAME, "peermemDumpSize=%u", peermemDumpSize);

    // 5. D2H 拷贝 peermem 窗口 -> 分段落盘或单文件落盘
    //    tiling 头部存在有效 peermemLayout 时逐段 D2H + 落盘，否则回退到原有单文件 peermem_window_ dump
    bool segDumped = false;
    if (tilingGM != 0U && peermemLayout.segCount > 0 && peermemLayout.segCount <= Utils::MAX_PEERMEM_SEGMENTS) {
        OP_LOGE(OP_NAME, "Peermem segmented dump: segCount=%u, totalSize=%lu", peermemLayout.segCount,
                peermemLayout.totalSize);
        for (uint32_t i = 0; i < peermemLayout.segCount; i++) {
            auto &seg = peermemLayout.segments[i];
            if (seg.size == 0) {
                continue;
            }
            uint64_t segDumpSize = seg.size;
            void *srcAddr = reinterpret_cast<void *>(selfWinAddr + seg.offset);
            OP_LOGE(OP_NAME, "peermem seg[%u]: type=%u, offset=%lu, size=%lu, dumpSize=%lu, srcAddr=0x%lx", i, seg.type,
                    seg.offset, seg.size, segDumpSize, static_cast<uint64_t>(selfWinAddr + seg.offset));
            try {
                std::vector<uint8_t> segBuf(segDumpSize, 0);
                if (aclrtMemcpy(segBuf.data(), segDumpSize, srcAddr, segDumpSize, ACL_MEMCPY_DEVICE_TO_HOST) !=
                    ACL_SUCCESS) {
                    OP_LOGE(OP_NAME, "aclrtMemcpy peermem seg[%u] failed, offset=%lu, size=%lu.", i, seg.offset,
                            segDumpSize);
                    continue;
                }
                std::string segPrefix = "peermem_seg" + std::to_string(i) + "_type" + std::to_string(seg.type) + "_";
                std::string segName = GenDumpFileNameWithPrefix(args, op, segPrefix.c_str());
                if (DumpToFileWithSize(std::string(dumpPath), segName, deviceId, segBuf.data(), segDumpSize) != 0) {
                    OP_LOGE(OP_NAME, "Failed to dump peermem seg[%u].", i);
                }
            } catch (const std::bad_alloc &) {
                OP_LOGE(OP_NAME, "peermem seg[%u] host alloc %lu bytes failed, skip.", i, segDumpSize);
                continue;
            } catch (...) {
                OP_LOGE(OP_NAME, "peermem seg[%u] dump exception, skip.", i);
                continue;
            }
        }
        segDumped = true;
    } else {
        OP_LOGE(OP_NAME, "DfxPeermemLayoutInfo invalid or tiling unavailable, falling back to single dump.");
    }

    if (!segDumped) {
        std::vector<uint8_t> winBuf(peermemDumpSize, 0);
        if (aclrtMemcpy(winBuf.data(), peermemDumpSize, reinterpret_cast<void *>(selfWinAddr), peermemDumpSize,
                        ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            OP_LOGE(OP_NAME, "aclrtMemcpy peermem window failed.");
            return;
        }
        if (DumpToFileWithSize(std::string(dumpPath), GenDumpFileNameWithPrefix(args, op, "peermem_window_"), deviceId,
                               winBuf.data(), peermemDumpSize) != 0) {
            OP_LOGE(OP_NAME, "Failed to dump peermem window.");
        }
    }
    OP_LOGE(OP_NAME, "UrmaContext dump done: op=%s, deviceId=%u.", op, deviceId);
}
#else  // !MC2_DFX_ENABLE
inline void Mc2ExceptionImplTmp(aclrtExceptionInfo *args, void *userdata, const char *op) {}
inline void Mc2DumpTilingAndWorkspace(aclrtExceptionInfo *args, const char *op, uint32_t tilingGmArgIdx,
                                      uint32_t workspaceGmArgIdx = 0U, uint32_t dfxInfoOffset = 0U)
{}
inline void Mc2DumpUrmaContext(aclrtExceptionInfo *args, const char *op, uint32_t contextArgIdx,
                               uint32_t tilingGmArgIdx = 0U)
{}
#endif // MC2_DFX_ENABLE

} // namespace Mc2Exception
#endif

#endif
