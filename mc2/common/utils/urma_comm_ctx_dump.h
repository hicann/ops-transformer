/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file urma_comm_ctx_dump.h
 * \brief URMA 通信上下文的 Host 侧 dump 镜像结构，供异常 dump (mc2_exception_dump.h) 做 D2H 拷贝解析。
 *        布局须与 Apace::AivComm::CommUdmaContext/CommUbmemContext 及各 fusion 的 CommContext 保持一致。
 */

#ifndef URMA_COMM_CTX_DUMP_H
#define URMA_COMM_CTX_DUMP_H

#include <cstdint>

constexpr uint32_t URMA_MAX_RANK_NUM = 64;

struct UrmaCommContextForDump {
    uint32_t rankId = 0;
    uint32_t rankSize = 0;
    uint64_t channelHandles[URMA_MAX_RANK_NUM] = {0};
    uint64_t commBufferAddrs[URMA_MAX_RANK_NUM] = {0};
};

// 镜像 Apace::AivComm::CommUbmemContext（同步面，无 channelHandles）
struct UrmaUbmemContextForDump {
    uint32_t rankId = 0;
    uint32_t rankSize = 0;
    uint64_t commBufferAddrs[URMA_MAX_RANK_NUM] = {0};
};

// 镜像 Apace::AivComm::CommContext（udmaCtx 1032B + ubmemCtx 520B，共 1552B）
struct UrmaCommContextFullForDump {
    UrmaCommContextForDump udmaCtx;
    UrmaUbmemContextForDump ubmemCtx;
};
#endif
