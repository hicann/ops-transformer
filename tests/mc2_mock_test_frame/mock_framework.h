/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * Mock HCCL Server Framework — Generic hcclClient Server-Side Simulator
 *
 * Simulates the CCU server (HCCL Server) on the host CPU side for single-card
 * testing. This framework is operator-agnostic — any MC2 operator that
 * communicates via the workspace message protocol can use it.
 *
 * Components:
 * 1. MockContextBuilder — constructs device-side HcclCombinOpParam (commContext)
 * 2. MockHcclServer — host polling thread that responds to kernel's
 *    communication requests with real collective semantics
 *
 * Communication simulation:
 *   The server accepts per-rank input tensors at construction. When the kernel
 *   issues a collective operation, the server reads these tensors as if they
 *   came from remote ranks, and performs the real collective semantics:
 *   - AllGather:      gather each rank's chunk into recvBuf
 *   - ReduceScatter:  element-wise reduce all ranks, scatter chunks
 *   - AllReduce:      element-wise reduce all ranks, full result to recvBuf
 *   - AlltoAll:       block redistribution
 */

#ifndef MOCK_FRAMEWORK_H_
#define MOCK_FRAMEWORK_H_

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <atomic>
#include <thread>
#include <chrono>
#include <functional>
#include <vector>
#include <unistd.h>
#include "acl/acl.h"

namespace mock_hccl {

// ============================================================
// HCCL Protocol Constants
// ============================================================
constexpr uint32_t COMMIT_VALID_MASK = 987654321U; // 0x3ADE68B1
constexpr uint64_t FINALIZE_FINISH_CNT = 1234567899999999999ULL;
constexpr uint32_t HCCL_MSG_VALID_MASK = 0x5CDF123AU;
constexpr uint32_t HCCL_MSG_CNT = 64;

// HCCL command types
constexpr uint32_t HCCL_CMD_ALLREDUCE = 2;
constexpr uint32_t HCCL_CMD_ALLGATHER = 6;
constexpr uint32_t HCCL_CMD_REDUCE_SCATTER = 7;
constexpr uint32_t HCCL_CMD_ALLTOALL = 12; // verify against actual HCCL
constexpr uint32_t HCCL_CMD_FINALIZE = 100;

// HCCL reduce operation types
constexpr uint32_t HCCL_REDUCE_SUM = 0;
constexpr uint32_t HCCL_REDUCE_PROD = 1;
constexpr uint32_t HCCL_REDUCE_MAX = 2;
constexpr uint32_t HCCL_REDUCE_MIN = 3;

// ============================================================
// Workspace Layout Offsets (from 512B-aligned base)
// ============================================================
constexpr size_t SEND_MSGS_OFFSET = 0x0000;      // sendMsgs[64], each 112 bytes
constexpr size_t RECV_MSGS_OFFSET = 0x1C00;      // recvMsgs[64]
constexpr size_t COMMIT_TURNCNT_OFFSET = 0x5800; // commitTurnCnt[64], each 64 bytes
constexpr size_t FINISH_TURNCNT_OFFSET = 0x6800; // finishedTurnCnt[64], each 64 bytes

constexpr size_t MSG_STRIDE = 112;    // bytes per HcclMsg
constexpr size_t TURNCNT_STRIDE = 64; // bytes per TurnCnt (cache-line aligned)

// Minimum workspace size (single queue mode)
constexpr size_t MIN_WORKSPACE_SIZE = 2 * 1024 * 1024; // 2MB

// ============================================================
// Workspace Structures (host-side mirrors)
// ============================================================

struct HcclMsg {
    uint32_t commType;     // +0x00  HCCL_CMD_ALLGATHER=6, etc.
    uint32_t opType;       // +0x04  reduce operation type (SUM=0, MAX=2, ...)
    uint64_t sendBuffer;   // +0x08  device source address
    uint64_t recvBuffer;   // +0x10  device destination address
    uint64_t dataCnt;      // +0x18  element count
    uint64_t strideCount;  // +0x20  stride between ranks in recvBuf (elements)
    uint32_t msgValid;     // +0x28  HCCL_MSG_VALID_MASK when valid
    uint32_t hcclDataType; // +0x2C  data type enum
    uint8_t rest[64];      // +0x30  remaining fields to fill 112 bytes total
};
static_assert(sizeof(HcclMsg) == MSG_STRIDE, "HcclMsg size mismatch");

struct TurnCnt {
    uint64_t valid;       // +0x00  COMMIT_VALID_MASK or 0
    uint64_t cnt;         // +0x08  commit/finish count
    uint64_t reserved[6]; // +0x10  padding to 64 bytes
};
static_assert(sizeof(TurnCnt) == TURNCNT_STRIDE, "TurnCnt size mismatch");

// ============================================================
// Context Structure (910B: HcclA2CombineOpParam)
// ============================================================
constexpr uint32_t MAX_RANK = 32;
constexpr int64_t FLAG_OFFSET_BYTES = 180LL * 1024 * 1024;
constexpr int64_t FLAG_AREA_SIZE = 4LL * 1024 * 1024;
constexpr int64_t WINDOW_TOTAL_SIZE = FLAG_OFFSET_BYTES + FLAG_AREA_SIZE;

struct MockHcclContext {
    uint64_t workSpace;            // +0
    uint64_t workSpaceSize;        // +8
    uint32_t rankId;               // +16
    uint32_t rankNum;              // +20
    uint64_t winSize;              // +24
    uint64_t windowsIn[MAX_RANK];  // +32
    uint64_t windowsOut[MAX_RANK]; // +288
    uint8_t res[8328];             // +544
    uint8_t multiFlag;             // +8872
    uint64_t data;                 // +8873
    uint64_t dataSize;             // +8881
    uint64_t sizeOfAiRMAInfo;      // +8889
    uint64_t aiRMAInfo;            // +8897
};

// ============================================================
// HCCL Data Type → byte size mapping
// ============================================================
inline size_t HcclDataTypeSize(uint32_t hcclType)
{
    switch (hcclType) {
        case 0:
            return 4; // FP32
        case 1:
            return 2; // FP16
        case 2:
            return 1; // INT8
        case 3:
            return 4; // INT32
        case 5:
            return 2; // BF16
        case 12:
            return 1; // FP8_E4M3
        case 13:
            return 1; // FP8_E5M2
        default:
            return 2;
    }
}

// ============================================================
// FP16 / BF16 ↔ float conversion (host-side computation)
// ============================================================
inline float Fp16ToFloat(uint16_t h)
{
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign << 31;
        } else {
            // denorm → norm
            exp = 1;
            while (!(mant & 0x400)) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3ff;
            f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | 0x7f800000 | (mant << 13); // inf/nan
    } else {
        f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    }
    float result;
    memcpy(&result, &f, 4);
    return result;
}

inline uint16_t FloatToFp16(float val)
{
    uint32_t u;
    memcpy(&u, &val, 4);
    uint32_t sign = (u >> 31) & 1;
    int32_t exp = ((u >> 23) & 0xff) - 127 + 15;
    uint32_t mant = (u >> 13) & 0x3ff;
    if (exp <= 0)
        return (uint16_t)(sign << 15);
    if (exp >= 31)
        return (uint16_t)((sign << 15) | 0x7c00);
    return (uint16_t)((sign << 15) | (exp << 10) | mant);
}

inline float Bf16ToFloat(uint16_t bf)
{
    uint32_t f = (uint32_t)bf << 16;
    float result;
    memcpy(&result, &f, 4);
    return result;
}

inline uint16_t FloatToBf16(float val)
{
    uint32_t u;
    memcpy(&u, &val, 4);
    return (uint16_t)(u >> 16);
}

// ============================================================
// Host-side element read/write by hcclDataType
// ============================================================
inline float ReadElement(const void *buf, size_t idx, uint32_t hcclType)
{
    switch (hcclType) {
        case 0:
            return ((const float *)buf)[idx]; // FP32
        case 1:
            return Fp16ToFloat(((const uint16_t *)buf)[idx]); // FP16
        case 3:
            return (float)((const int32_t *)buf)[idx]; // INT32
        case 5:
            return Bf16ToFloat(((const uint16_t *)buf)[idx]); // BF16
        default:
            return 0.0f;
    }
}

inline void WriteElement(void *buf, size_t idx, float val, uint32_t hcclType)
{
    switch (hcclType) {
        case 0:
            ((float *)buf)[idx] = val;
            break; // FP32
        case 1:
            ((uint16_t *)buf)[idx] = FloatToFp16(val);
            break; // FP16
        case 3:
            ((int32_t *)buf)[idx] = (int32_t)val;
            break; // INT32
        case 5:
            ((uint16_t *)buf)[idx] = FloatToBf16(val);
            break; // BF16
        default:
            break;
    }
}

// ============================================================
// Host-side reduce: dst = reduce(srcs[0..N-1])
// All buffers are host memory, count = number of elements.
// ============================================================
inline void HostReduce(void *dst, const std::vector<const void *> &srcs, size_t count, uint32_t hcclDataType,
                       uint32_t reduceOp)
{
    for (size_t i = 0; i < count; i++) {
        float acc = ReadElement(srcs[0], i, hcclDataType);
        for (size_t r = 1; r < srcs.size(); r++) {
            float val = ReadElement(srcs[r], i, hcclDataType);
            switch (reduceOp) {
                case HCCL_REDUCE_SUM:
                    acc += val;
                    break;
                case HCCL_REDUCE_PROD:
                    acc *= val;
                    break;
                case HCCL_REDUCE_MAX:
                    acc = std::max(acc, val);
                    break;
                case HCCL_REDUCE_MIN:
                    acc = std::min(acc, val);
                    break;
                default:
                    acc += val;
                    break; // default to sum
            }
        }
        WriteElement(dst, i, acc, hcclDataType);
    }
}

// ============================================================
// Device Memory Helper
// ============================================================
inline void *DevMalloc(size_t size)
{
    void *ptr = nullptr;
    if (aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_HUGE_FIRST) != 0)
        return nullptr;
    aclrtMemset(ptr, size, 0, size);
    return ptr;
}

inline void DevFree(void *ptr)
{
    if (ptr)
        aclrtFree(ptr);
}

inline void *AlignUp512(void *ptr)
{
    uint64_t addr = reinterpret_cast<uint64_t>(ptr);
    if (addr & 0x1ff) {
        addr = (addr & (~(uint64_t)0x1ff)) + 0x200;
    }
    return reinterpret_cast<void *>(addr);
}

inline void *OffsetPtr(void *base, size_t offset)
{
    return reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(base) + offset);
}

// ============================================================
// RankData — per-rank input tensor descriptor
// ============================================================
struct RankData {
    void *devicePtr; // device memory pointer
    size_t byteSize; // size in bytes
};

// ============================================================
// MockContextBuilder
// ============================================================
class MockContextBuilder {
public:
    ~MockContextBuilder()
    {
        Destroy();
    }

    bool Build(uint32_t rankNum = 1, uint32_t rankId = 0)
    {
        rankNum_ = rankNum;
        rankId_ = rankId;

        // Allocate window memory
        devWindowMem_ = DevMalloc(WINDOW_TOTAL_SIZE);
        if (!devWindowMem_) {
            printf("[MockContext] Window memory alloc failed (%lldMB)\n",
                   (long long)(WINDOW_TOTAL_SIZE / (1024 * 1024)));
            return false;
        }

        // Allocate workspace (extra 512B for alignment)
        size_t wsAllocSize = MIN_WORKSPACE_SIZE + 512;
        devWorkspaceRaw_ = DevMalloc(wsAllocSize);
        if (!devWorkspaceRaw_) {
            printf("[MockContext] Workspace alloc failed\n");
            return false;
        }
        devWorkspaceAligned_ = AlignUp512(devWorkspaceRaw_);

        // Construct context on host
        MockHcclContext hostCtx;
        memset(&hostCtx, 0, sizeof(hostCtx));
        hostCtx.rankId = rankId_;
        hostCtx.rankNum = rankNum_;
        hostCtx.winSize = WINDOW_TOTAL_SIZE;
        hostCtx.workSpace = reinterpret_cast<uint64_t>(devWorkspaceAligned_);
        hostCtx.workSpaceSize = MIN_WORKSPACE_SIZE;

        for (uint32_t i = 0; i < rankNum_; i++) {
            hostCtx.windowsIn[i] = reinterpret_cast<uint64_t>(devWindowMem_);
            hostCtx.windowsOut[i] = reinterpret_cast<uint64_t>(devWindowMem_);
        }

        // Copy to device
        devContext_ = DevMalloc(sizeof(MockHcclContext));
        if (!devContext_)
            return false;
        if (aclrtMemcpy(devContext_, sizeof(hostCtx), &hostCtx, sizeof(hostCtx), ACL_MEMCPY_HOST_TO_DEVICE) != 0) {
            printf("[MockContext] H2D copy failed\n");
            return false;
        }

        printf("[MockContext] Built: rankId=%u, rankNum=%u\n", rankId_, rankNum_);
        printf("  context   @ %p\n", devContext_);
        printf("  workspace @ %p (aligned: %p)\n", devWorkspaceRaw_, devWorkspaceAligned_);
        printf("  window    @ %p (%lldMB)\n", devWindowMem_, (long long)(WINDOW_TOTAL_SIZE / (1024 * 1024)));
        return true;
    }

    void Destroy()
    {
        DevFree(devContext_);
        devContext_ = nullptr;
        DevFree(devWorkspaceRaw_);
        devWorkspaceRaw_ = nullptr;
        DevFree(devWindowMem_);
        devWindowMem_ = nullptr;
        devWorkspaceAligned_ = nullptr;
    }

    void *GetContextAddr() const
    {
        return devContext_;
    }
    void *GetWorkspaceAligned() const
    {
        return devWorkspaceAligned_;
    }
    void *GetWindowMem() const
    {
        return devWindowMem_;
    }
    uint32_t GetRankNum() const
    {
        return rankNum_;
    }

private:
    void *devContext_{nullptr};
    void *devWorkspaceRaw_{nullptr};
    void *devWorkspaceAligned_{nullptr};
    void *devWindowMem_{nullptr};
    uint32_t rankNum_{1};
    uint32_t rankId_{0};
};

// ============================================================
// MultiRankMockContext — allocates N windows + N contexts
//
// Each rank gets its own window (184MB). All contexts share the
// same windowsIn[] array pointing to all ranks' windows.
// Combined with multi-stream + core binding, this enables real
// multi-rank AllGather on a single card.
//
// Usage:
//   MultiRankMockContext mctx;
//   mctx.Build(4);  // 4 ranks
//   void* ctx_r0 = mctx.GetContextAddr(0);  // rank 0's context
//   void* ctx_r1 = mctx.GetContextAddr(1);  // rank 1's context
//   // Launch V3 on stream[0] with ctx_r0, stream[1] with ctx_r1, ...
// ============================================================
class MultiRankMockContext {
public:
    ~MultiRankMockContext()
    {
        Destroy();
    }

    bool Build(uint32_t rankNum)
    {
        rankNum_ = rankNum;

        // Allocate per-rank windows
        devWindows_.resize(rankNum, nullptr);
        for (uint32_t r = 0; r < rankNum; r++) {
            devWindows_[r] = DevMalloc(WINDOW_TOTAL_SIZE);
            if (!devWindows_[r]) {
                printf("[MultiRankMock] Window alloc failed for rank %u\n", r);
                return false;
            }
        }

        // Allocate per-rank workspaces
        devWorkspaceRaw_.resize(rankNum, nullptr);
        devWorkspaceAligned_.resize(rankNum, nullptr);
        for (uint32_t r = 0; r < rankNum; r++) {
            devWorkspaceRaw_[r] = DevMalloc(MIN_WORKSPACE_SIZE + 512);
            if (!devWorkspaceRaw_[r]) {
                printf("[MultiRankMock] Workspace alloc failed for rank %u\n", r);
                return false;
            }
            devWorkspaceAligned_[r] = AlignUp512(devWorkspaceRaw_[r]);
        }

        // Build per-rank contexts with cross-referenced windows
        devContexts_.resize(rankNum, nullptr);
        for (uint32_t r = 0; r < rankNum; r++) {
            MockHcclContext hostCtx;
            memset(&hostCtx, 0, sizeof(hostCtx));
            hostCtx.rankId = r;
            hostCtx.rankNum = rankNum;
            hostCtx.winSize = WINDOW_TOTAL_SIZE;
            hostCtx.workSpace = reinterpret_cast<uint64_t>(devWorkspaceAligned_[r]);
            hostCtx.workSpaceSize = MIN_WORKSPACE_SIZE;

            // All ranks see all windows
            for (uint32_t j = 0; j < rankNum; j++) {
                hostCtx.windowsIn[j] = reinterpret_cast<uint64_t>(devWindows_[j]);
                hostCtx.windowsOut[j] = reinterpret_cast<uint64_t>(devWindows_[j]);
            }

            devContexts_[r] = DevMalloc(sizeof(MockHcclContext));
            if (!devContexts_[r])
                return false;
            if (aclrtMemcpy(devContexts_[r], sizeof(hostCtx), &hostCtx, sizeof(hostCtx), ACL_MEMCPY_HOST_TO_DEVICE) !=
                0) {
                printf("[MultiRankMock] H2D copy failed for rank %u\n", r);
                return false;
            }
        }

        printf("[MultiRankMock] Built %u ranks, each window %lldMB\n", rankNum,
               (long long)(WINDOW_TOTAL_SIZE / (1024 * 1024)));
        for (uint32_t r = 0; r < rankNum; r++) {
            printf("  rank %u: context=%p window=%p workspace=%p\n", r, devContexts_[r], devWindows_[r],
                   devWorkspaceAligned_[r]);
        }
        return true;
    }

    void ClearAllFlags()
    {
        for (uint32_t r = 0; r < rankNum_; r++) {
            if (devWindows_[r]) {
                void *flagArea = OffsetPtr(devWindows_[r], FLAG_OFFSET_BYTES);
                aclrtMemset(flagArea, FLAG_AREA_SIZE, 0, FLAG_AREA_SIZE);
            }
        }
    }

    void Destroy()
    {
        for (auto p : devContexts_)
            DevFree(p);
        for (auto p : devWorkspaceRaw_)
            DevFree(p);
        for (auto p : devWindows_)
            DevFree(p);
        devContexts_.clear();
        devWorkspaceRaw_.clear();
        devWorkspaceAligned_.clear();
        devWindows_.clear();
    }

    void *GetContextAddr(uint32_t rank) const
    {
        return devContexts_[rank];
    }
    void *GetWindowMem(uint32_t rank) const
    {
        return devWindows_[rank];
    }
    void *GetWorkspaceAligned(uint32_t rank) const
    {
        return devWorkspaceAligned_[rank];
    }
    uint32_t GetRankNum() const
    {
        return rankNum_;
    }

private:
    uint32_t rankNum_{0};
    std::vector<void *> devWindows_;
    std::vector<void *> devWorkspaceRaw_;
    std::vector<void *> devWorkspaceAligned_;
    std::vector<void *> devContexts_;
};

// ============================================================
// MockHcclServer — host thread simulating CCU server
//
// Accepts per-rank input tensors (device pointers) and uses them
// to simulate real collective communication semantics.
// ============================================================
class MockHcclServer {
public:
    /**
     * @param workspaceAligned  512B-aligned workspace device pointer
     * @param rankInputs        per-rank input data (rankInputs[i] = rank i's tensor)
     * @param localRankId       the rank ID of the kernel under test
     * @param deviceId          NPU device ID
     */
    explicit MockHcclServer(void *workspaceAligned, std::vector<RankData> rankInputs = {}, uint32_t localRankId = 0,
                            int deviceId = 0)
        : wsBase_(workspaceAligned),
          rankInputs_(std::move(rankInputs)),
          localRankId_(localRankId),
          rankNum_(rankInputs_.empty() ? 1u : static_cast<uint32_t>(rankInputs_.size())),
          deviceId_(deviceId)
    {}

    ~MockHcclServer()
    {
        Stop();
    }

    void Start()
    {
        running_ = true;
        serverThread_ = std::thread(&MockHcclServer::ServerLoop, this);
        printf("[MockServer] Started: rankNum=%u, localRankId=%u, workspace @ %p\n", rankNum_, localRankId_, wsBase_);
    }

    void Stop()
    {
        running_ = false;
        if (serverThread_.joinable()) {
            serverThread_.join();
        }
        printf("[MockServer] Stopped. Processed %u messages.\n", msgCount_.load());
    }

    bool IsFinalized() const
    {
        return finalized_.load();
    }
    uint32_t GetMsgCount() const
    {
        return msgCount_.load();
    }

    // Block until the kernel signals Finalize, then respond.
    bool WaitForFinalize(uint32_t slot = 0, uint32_t timeoutMs = 5000)
    {
        auto start = std::chrono::steady_clock::now();
        while (true) {
            auto elapsed = std::chrono::steady_clock::now() - start;
            if (std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() >= timeoutMs) {
                printf("[MockServer] WaitForFinalize: timeout after %ums\n", timeoutMs);
                return false;
            }

            TurnCnt commitCnt;
            void *commitAddr = OffsetPtr(wsBase_, COMMIT_TURNCNT_OFFSET + slot * TURNCNT_STRIDE);
            if (aclrtMemcpy(&commitCnt, sizeof(TurnCnt), commitAddr, sizeof(TurnCnt), ACL_MEMCPY_DEVICE_TO_HOST) != 0) {
                usleep(100);
                continue;
            }

            if (commitCnt.valid != COMMIT_VALID_MASK || commitCnt.cnt <= lastProcessedCnt_[slot]) {
                usleep(100);
                continue;
            }

            lastProcessedCnt_[slot] = commitCnt.cnt;

            // Respond with FINALIZE_FINISH_CNT
            TurnCnt finishCnt;
            memset(&finishCnt, 0, sizeof(finishCnt));
            finishCnt.cnt = FINALIZE_FINISH_CNT;
            void *finishAddr = OffsetPtr(wsBase_, FINISH_TURNCNT_OFFSET + slot * TURNCNT_STRIDE);
            aclrtMemcpy(finishAddr, sizeof(TurnCnt), &finishCnt, sizeof(TurnCnt), ACL_MEMCPY_HOST_TO_DEVICE);

            // Clear commit
            TurnCnt clearCnt;
            memset(&clearCnt, 0, sizeof(clearCnt));
            clearCnt.cnt = commitCnt.cnt;
            aclrtMemcpy(commitAddr, sizeof(TurnCnt), &clearCnt, sizeof(TurnCnt), ACL_MEMCPY_HOST_TO_DEVICE);

            printf("[MockServer] Finalize response sent on slot %u "
                   "(cnt=%lu → FINALIZE_FINISH_CNT)\n",
                   slot, commitCnt.cnt);
            finalized_ = true;
            return true;
        }
    }

private:
    void ServerLoop()
    {
        aclrtSetDevice(deviceId_);
        while (running_) {
            for (uint32_t i = 0; i < HCCL_MSG_CNT && running_; i++) {
                PollSlot(i);
            }
            usleep(50); // 50us polling interval
        }
    }

    void PollSlot(uint32_t slot)
    {
        // Read commitTurnCnt[slot] from device
        TurnCnt commitCnt;
        void *commitAddr = OffsetPtr(wsBase_, COMMIT_TURNCNT_OFFSET + slot * TURNCNT_STRIDE);
        if (aclrtMemcpy(&commitCnt, sizeof(TurnCnt), commitAddr, sizeof(TurnCnt), ACL_MEMCPY_DEVICE_TO_HOST) != 0) {
            return;
        }

        if (commitCnt.valid != COMMIT_VALID_MASK)
            return;
        if (commitCnt.cnt <= lastProcessedCnt_[slot])
            return;

        // Read the message
        HcclMsg msg;
        void *msgAddr = OffsetPtr(wsBase_, SEND_MSGS_OFFSET + slot * MSG_STRIDE);
        if (aclrtMemcpy(&msg, sizeof(HcclMsg), msgAddr, sizeof(HcclMsg), ACL_MEMCPY_DEVICE_TO_HOST) != 0) {
            return;
        }

        printf("[MockServer] slot=%u cnt=%lu commType=%u dataCnt=%lu opType=%u\n", slot, commitCnt.cnt, msg.commType,
               msg.dataCnt, msg.opType);

        // Dispatch by commType
        HandleMessage(msg);

        // Record processed count
        lastProcessedCnt_[slot] = commitCnt.cnt;

        // Write finishedTurnCnt[slot]
        TurnCnt finishCnt;
        memset(&finishCnt, 0, sizeof(finishCnt));
        finishCnt.cnt = commitCnt.cnt;
        void *finishAddr = OffsetPtr(wsBase_, FINISH_TURNCNT_OFFSET + slot * TURNCNT_STRIDE);
        aclrtMemcpy(finishAddr, sizeof(TurnCnt), &finishCnt, sizeof(TurnCnt), ACL_MEMCPY_HOST_TO_DEVICE);

        // Clear commit valid flag
        TurnCnt clearCnt;
        memset(&clearCnt, 0, sizeof(clearCnt));
        clearCnt.cnt = commitCnt.cnt;
        aclrtMemcpy(commitAddr, sizeof(TurnCnt), &clearCnt, sizeof(TurnCnt), ACL_MEMCPY_HOST_TO_DEVICE);

        msgCount_++;
    }

    // --------------------------------------------------------
    // Message dispatch
    // --------------------------------------------------------
    void HandleMessage(const HcclMsg &msg)
    {
        switch (msg.commType) {
            case HCCL_CMD_ALLGATHER:
                HandleAllGather(msg);
                break;
            case HCCL_CMD_REDUCE_SCATTER:
                HandleReduceScatter(msg);
                break;
            case HCCL_CMD_ALLREDUCE:
                HandleAllReduce(msg);
                break;
            case HCCL_CMD_ALLTOALL:
                HandleAlltoAll(msg);
                break;
            default:
                printf("[MockServer]   Unknown commType=%u, no-op\n", msg.commType);
                break;
        }
    }

    // --------------------------------------------------------
    // AllGather: each rank contributes dataCnt elements
    // recvBuf = [rank0_chunk | rank1_chunk | ... | rankN-1_chunk]
    //            ^stride       ^stride
    // --------------------------------------------------------
    void HandleAllGather(const HcclMsg &msg)
    {
        size_t elemSize = HcclDataTypeSize(msg.hcclDataType);
        size_t chunkBytes = msg.dataCnt * elemSize;
        size_t strideBytes = msg.strideCount * elemSize;
        if (strideBytes == 0)
            strideBytes = chunkBytes; // default: tightly packed

        void *recvBuf = reinterpret_cast<void *>(msg.recvBuffer);
        void *sendBuf = reinterpret_cast<void *>(msg.sendBuffer);

        for (uint32_t r = 0; r < rankNum_; r++) {
            void *dst = OffsetPtr(recvBuf, r * strideBytes);
            if (r == localRankId_) {
                // Local rank: copy from kernel's sendBuf
                if (sendBuf && sendBuf != dst) {
                    aclrtMemcpy(dst, chunkBytes, sendBuf, chunkBytes, ACL_MEMCPY_DEVICE_TO_DEVICE);
                }
            } else {
                // Remote rank: copy from rankInputs_[r]
                if (r < rankInputs_.size() && rankInputs_[r].devicePtr) {
                    size_t copyBytes = std::min(chunkBytes, rankInputs_[r].byteSize);
                    aclrtMemcpy(dst, copyBytes, rankInputs_[r].devicePtr, copyBytes, ACL_MEMCPY_DEVICE_TO_DEVICE);
                }
            }
        }

        printf("[MockServer]   AllGather: %u ranks × %zu bytes → recvBuf %p\n", rankNum_, chunkBytes, recvBuf);
    }

    // --------------------------------------------------------
    // ReduceScatter: each rank has dataCnt elements (total input),
    // element-wise reduce across all ranks, then rank i gets
    // chunk i of the result. Output size = dataCnt / rankNum.
    // --------------------------------------------------------
    void HandleReduceScatter(const HcclMsg &msg)
    {
        size_t elemSize = HcclDataTypeSize(msg.hcclDataType);
        size_t totalElems = msg.dataCnt;
        size_t totalBytes = totalElems * elemSize;
        size_t chunkElems = totalElems / rankNum_;
        size_t chunkBytes = chunkElems * elemSize;

        // D2H read all ranks' data
        std::vector<std::vector<uint8_t>> hostBufs(rankNum_);
        for (uint32_t r = 0; r < rankNum_; r++) {
            hostBufs[r].resize(totalBytes, 0);
            if (r == localRankId_) {
                // Local rank: read from kernel's sendBuf
                aclrtMemcpy(hostBufs[r].data(), totalBytes, reinterpret_cast<void *>(msg.sendBuffer), totalBytes,
                            ACL_MEMCPY_DEVICE_TO_HOST);
            } else if (r < rankInputs_.size() && rankInputs_[r].devicePtr) {
                size_t readBytes = std::min(totalBytes, rankInputs_[r].byteSize);
                aclrtMemcpy(hostBufs[r].data(), readBytes, rankInputs_[r].devicePtr, readBytes,
                            ACL_MEMCPY_DEVICE_TO_HOST);
            }
        }

        // Host-side element-wise reduce
        std::vector<uint8_t> reducedBuf(totalBytes);
        std::vector<const void *> srcs(rankNum_);
        for (uint32_t r = 0; r < rankNum_; r++)
            srcs[r] = hostBufs[r].data();
        HostReduce(reducedBuf.data(), srcs, totalElems, msg.hcclDataType, msg.opType);

        // Extract local rank's chunk and H2D to recvBuf
        aclrtMemcpy(reinterpret_cast<void *>(msg.recvBuffer), chunkBytes, reducedBuf.data() + localRankId_ * chunkBytes,
                    chunkBytes, ACL_MEMCPY_HOST_TO_DEVICE);

        printf("[MockServer]   ReduceScatter: %u ranks × %zu elems → reduce → chunk[%u] %zu elems\n", rankNum_,
               totalElems, localRankId_, chunkElems);
    }

    // --------------------------------------------------------
    // AllReduce: each rank has dataCnt elements,
    // element-wise reduce across all ranks, full result to recvBuf.
    // --------------------------------------------------------
    void HandleAllReduce(const HcclMsg &msg)
    {
        size_t elemSize = HcclDataTypeSize(msg.hcclDataType);
        size_t totalElems = msg.dataCnt;
        size_t totalBytes = totalElems * elemSize;

        // D2H read all ranks' data
        std::vector<std::vector<uint8_t>> hostBufs(rankNum_);
        for (uint32_t r = 0; r < rankNum_; r++) {
            hostBufs[r].resize(totalBytes, 0);
            if (r == localRankId_) {
                aclrtMemcpy(hostBufs[r].data(), totalBytes, reinterpret_cast<void *>(msg.sendBuffer), totalBytes,
                            ACL_MEMCPY_DEVICE_TO_HOST);
            } else if (r < rankInputs_.size() && rankInputs_[r].devicePtr) {
                size_t readBytes = std::min(totalBytes, rankInputs_[r].byteSize);
                aclrtMemcpy(hostBufs[r].data(), readBytes, rankInputs_[r].devicePtr, readBytes,
                            ACL_MEMCPY_DEVICE_TO_HOST);
            }
        }

        // Host-side element-wise reduce
        std::vector<uint8_t> reducedBuf(totalBytes);
        std::vector<const void *> srcs(rankNum_);
        for (uint32_t r = 0; r < rankNum_; r++)
            srcs[r] = hostBufs[r].data();
        HostReduce(reducedBuf.data(), srcs, totalElems, msg.hcclDataType, msg.opType);

        // H2D full result to recvBuf
        aclrtMemcpy(reinterpret_cast<void *>(msg.recvBuffer), totalBytes, reducedBuf.data(), totalBytes,
                    ACL_MEMCPY_HOST_TO_DEVICE);

        printf("[MockServer]   AllReduce: %u ranks × %zu elems → reduce → full result\n", rankNum_, totalElems);
    }

    // --------------------------------------------------------
    // AlltoAll: rank i's block j → rank j's block i
    // Each rank has dataCnt elements total (rankNum blocks of
    // dataCnt/rankNum elements each).
    // For local rank: collect block[localRankId] from each rank.
    // --------------------------------------------------------
    void HandleAlltoAll(const HcclMsg &msg)
    {
        size_t elemSize = HcclDataTypeSize(msg.hcclDataType);
        size_t totalElems = msg.dataCnt;
        size_t totalBytes = totalElems * elemSize;
        size_t blockElems = totalElems / rankNum_;
        size_t blockBytes = blockElems * elemSize;

        // D2H read all ranks' data
        std::vector<std::vector<uint8_t>> hostBufs(rankNum_);
        for (uint32_t r = 0; r < rankNum_; r++) {
            hostBufs[r].resize(totalBytes, 0);
            if (r == localRankId_) {
                aclrtMemcpy(hostBufs[r].data(), totalBytes, reinterpret_cast<void *>(msg.sendBuffer), totalBytes,
                            ACL_MEMCPY_DEVICE_TO_HOST);
            } else if (r < rankInputs_.size() && rankInputs_[r].devicePtr) {
                size_t readBytes = std::min(totalBytes, rankInputs_[r].byteSize);
                aclrtMemcpy(hostBufs[r].data(), readBytes, rankInputs_[r].devicePtr, readBytes,
                            ACL_MEMCPY_DEVICE_TO_HOST);
            }
        }

        // Reassemble: recvBuf[j] = rank_j's block[localRankId]
        // i.e., collect what each rank would send to localRankId
        std::vector<uint8_t> resultBuf(totalBytes);
        for (uint32_t r = 0; r < rankNum_; r++) {
            // rank r sends its block[localRankId_] to us
            const uint8_t *srcBlock = hostBufs[r].data() + localRankId_ * blockBytes;
            uint8_t *dstBlock = resultBuf.data() + r * blockBytes;
            memcpy(dstBlock, srcBlock, blockBytes);
        }

        // H2D to recvBuf
        aclrtMemcpy(reinterpret_cast<void *>(msg.recvBuffer), totalBytes, resultBuf.data(), totalBytes,
                    ACL_MEMCPY_HOST_TO_DEVICE);

        printf("[MockServer]   AlltoAll: %u ranks × %zu blocks → reassemble for rank %u\n", rankNum_, blockElems,
               localRankId_);
    }

    // --------------------------------------------------------
    // Member data
    // --------------------------------------------------------
    void *wsBase_{nullptr};
    std::vector<RankData> rankInputs_;
    uint32_t localRankId_{0};
    uint32_t rankNum_{1};
    int deviceId_{0};
    std::atomic<bool> running_{false};
    std::thread serverThread_;
    std::atomic<uint32_t> msgCount_{0};
    std::atomic<bool> finalized_{false};
    uint64_t lastProcessedCnt_[HCCL_MSG_CNT] = {};
};

} // namespace mock_hccl

#endif // MOCK_FRAMEWORK_H_
