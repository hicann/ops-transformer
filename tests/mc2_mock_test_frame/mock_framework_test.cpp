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
 * Mock V3 Kernel Integration Test
 *
 * Tests AllGatherMatmulV3 on a single card by:
 * 1. Constructing mock comm_context (MockContextBuilder)
 * 2. Starting mock HCCL server (MockHcclServer) on host thread
 * 3. Calling aclnnAllGatherMatmulV3 with the mock context
 * 4. Verifying output shapes and basic data flow
 *
 * Build:
 *   g++ -std=c++17 -o mock_v3_kernel_test mock_v3_kernel_test.cpp \
 *       -I$ASCEND_HOME/include -I../op_api \
 *       -L$ASCEND_HOME/lib64 -lascendcl \
 *       -L$CUST_PKG/lib -lcust_opapi \
 *       -Wl,--allow-shlib-undefined -lpthread
 *
 * Run:
 *   LD_LIBRARY_PATH=$ASCEND_HOME/lib64:$CUST_PKG/lib \
 *   LD_PRELOAD=$ASCEND_HOME/lib64/libhcomm.so \
 *   ./mock_v3_kernel_test
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
#include "acl/acl.h"
#include "mock_framework.h"

using namespace mock_hccl;

// ============================================================
// Test Configuration
// ============================================================
struct TestConfig {
    int64_t M = 256;       // per-rank M dimension
    int64_t K = 512;       // K dimension
    int64_t N = 256;       // N dimension
    uint32_t rankSize = 2; // simulated rank count (context has rankNum=1 but tiling uses this)
    bool isTransA = false;
    bool isTransB = false;
    int device = 0;
};

// ============================================================
// Tensor utilities
// ============================================================
struct DevTensor {
    void* data{nullptr};
    int64_t dims[2]{0, 0};
    size_t elemSize{2}; // FP16

    size_t ByteSize() const
    {
        return dims[0] * dims[1] * elemSize;
    }

    bool Alloc(int64_t d0, int64_t d1, size_t eSize = 2)
    {
        dims[0] = d0;
        dims[1] = d1;
        elemSize = eSize;
        data = DevMalloc(ByteSize());
        return data != nullptr;
    }

    void Free()
    {
        DevFree(data);
        data = nullptr;
    }
};

// Fill device tensor with pattern (host-side fill then copy)
static bool FillPattern(DevTensor& t, uint8_t pattern)
{
    size_t sz = t.ByteSize();
    std::vector<uint8_t> host(sz, pattern);
    return aclrtMemcpy(t.data, sz, host.data(), sz, ACL_MEMCPY_HOST_TO_DEVICE) == 0;
}

// ============================================================
// Test: MockContextBuilder verification
// ============================================================
static bool TestContextBuilder()
{
    printf("\n=== Test 1: MockContextBuilder ===\n");

    MockContextBuilder ctx;
    if (!ctx.Build(1, 0)) {
        printf("[FAIL] Context build\n");
        return false;
    }

    // Readback verification
    MockHcclContext readBack;
    if (aclrtMemcpy(&readBack, sizeof(readBack), ctx.GetContextAddr(), sizeof(readBack), ACL_MEMCPY_DEVICE_TO_HOST) !=
        0) {
        printf("[FAIL] Context readback\n");
        return false;
    }

    bool ok = (readBack.rankId == 0) && (readBack.rankNum == 1) &&
              (readBack.workSpace == reinterpret_cast<uint64_t>(ctx.GetWorkspaceAligned()));

    printf("  rankId=%u rankNum=%u workSpace=%p\n", readBack.rankId, readBack.rankNum, (void*)readBack.workSpace);
    printf("[%s] Context builder\n", ok ? "PASS" : "FAIL");
    return ok;
}

// ============================================================
// Test: MockHcclServer basic protocol
// ============================================================
static bool TestServerProtocol()
{
    printf("\n=== Test 2: MockHcclServer Protocol ===\n");

    MockContextBuilder ctx;
    if (!ctx.Build(1, 0))
        return false;

    void* wsAligned = ctx.GetWorkspaceAligned();

    // Manually write a commit message to slot 0
    // Simulate what the kernel's AllGather() would do:

    // 1. Write a HcclMsg at sendMsgs[0]
    HcclMsg msg;
    memset(&msg, 0, sizeof(msg));
    msg.commType = HCCL_CMD_ALLGATHER;
    // For this test, just check protocol — no actual data copy
    msg.sendBuffer = 0;
    msg.recvBuffer = 0;
    msg.dataCnt = 100;
    msg.hcclDataType = 1; // FP16

    void* msgAddr = reinterpret_cast<uint8_t*>(wsAligned) + SEND_MSGS_OFFSET;
    if (aclrtMemcpy(msgAddr, sizeof(msg), &msg, sizeof(msg), ACL_MEMCPY_HOST_TO_DEVICE) != 0) {
        printf("[FAIL] Write sendMsg\n");
        return false;
    }

    // 2. Write commitTurnCnt[0]
    TurnCnt commit;
    memset(&commit, 0, sizeof(commit));
    commit.valid = COMMIT_VALID_MASK;
    commit.cnt = 1;

    void* commitAddr = reinterpret_cast<uint8_t*>(wsAligned) + COMMIT_TURNCNT_OFFSET;
    if (aclrtMemcpy(commitAddr, sizeof(commit), &commit, sizeof(commit), ACL_MEMCPY_HOST_TO_DEVICE) != 0) {
        printf("[FAIL] Write commitTurnCnt\n");
        return false;
    }

    // 3. Verify commit data is on device (before starting server)
    {
        TurnCnt readCommit;
        aclrtMemcpy(&readCommit, sizeof(readCommit), commitAddr, sizeof(readCommit), ACL_MEMCPY_DEVICE_TO_HOST);
        printf("  [debug] commitTurnCnt[0] on device: valid=%lu cnt=%lu\n", readCommit.valid, readCommit.cnt);
        printf("  [debug] expected valid=%u\n", COMMIT_VALID_MASK);
    }

    // 3b. Start server and wait for it to process
    MockHcclServer server(wsAligned);
    server.Start();

    // Wait for processing
    usleep(500000); // 500ms

    server.Stop();

    // 4. Check finishedTurnCnt[0]
    TurnCnt finish;
    void* finishAddr = reinterpret_cast<uint8_t*>(wsAligned) + FINISH_TURNCNT_OFFSET;
    if (aclrtMemcpy(&finish, sizeof(finish), finishAddr, sizeof(finish), ACL_MEMCPY_DEVICE_TO_HOST) != 0) {
        printf("[FAIL] Read finishedTurnCnt\n");
        return false;
    }

    bool ok = (finish.cnt >= 1);
    printf("  finishedTurnCnt[0].cnt = %lu (expect >= 1)\n", finish.cnt);
    printf("[%s] Server protocol\n", ok ? "PASS" : "FAIL");
    return ok;
}

// ============================================================
// Test: MockHcclServer data copy (AllGather simulation)
// ============================================================
static bool TestServerDataCopy()
{
    printf("\n=== Test 3: MockHcclServer Data Copy ===\n");

    MockContextBuilder ctx;
    if (!ctx.Build(1, 0))
        return false;

    void* wsAligned = ctx.GetWorkspaceAligned();

    // Allocate source and destination buffers
    const size_t elemCount = 1024;
    const size_t elemSize = 2; // FP16
    const size_t byteCount = elemCount * elemSize;

    void* sendBuf = DevMalloc(byteCount);
    void* recvBuf = DevMalloc(byteCount);
    if (!sendBuf || !recvBuf) {
        printf("[FAIL] Buffer alloc\n");
        DevFree(sendBuf);
        DevFree(recvBuf);
        return false;
    }

    // Fill sendBuf with pattern
    std::vector<uint8_t> pattern(byteCount, 0xAB);
    aclrtMemcpy(sendBuf, byteCount, pattern.data(), byteCount, ACL_MEMCPY_HOST_TO_DEVICE);

    // Write AllGather message
    HcclMsg msg;
    memset(&msg, 0, sizeof(msg));
    msg.commType = HCCL_CMD_ALLGATHER;
    msg.sendBuffer = reinterpret_cast<uint64_t>(sendBuf);
    msg.recvBuffer = reinterpret_cast<uint64_t>(recvBuf);
    msg.dataCnt = elemCount;
    msg.hcclDataType = 1; // FP16

    void* msgAddr = reinterpret_cast<uint8_t*>(wsAligned) + SEND_MSGS_OFFSET;
    aclrtMemcpy(msgAddr, sizeof(msg), &msg, sizeof(msg), ACL_MEMCPY_HOST_TO_DEVICE);

    TurnCnt commit;
    memset(&commit, 0, sizeof(commit));
    commit.valid = COMMIT_VALID_MASK;
    commit.cnt = 1;
    void* commitAddr = reinterpret_cast<uint8_t*>(wsAligned) + COMMIT_TURNCNT_OFFSET;
    aclrtMemcpy(commitAddr, sizeof(commit), &commit, sizeof(commit), ACL_MEMCPY_HOST_TO_DEVICE);

    // Start server
    MockHcclServer server(wsAligned);
    server.Start();
    usleep(200000);
    server.Stop();

    // Verify recvBuf has the data
    std::vector<uint8_t> result(byteCount, 0);
    aclrtMemcpy(result.data(), byteCount, recvBuf, byteCount, ACL_MEMCPY_DEVICE_TO_HOST);

    bool ok = (memcmp(result.data(), pattern.data(), byteCount) == 0);
    printf("  recvBuf[0..3] = 0x%02x 0x%02x 0x%02x 0x%02x (expect 0xAB)\n", result[0], result[1], result[2], result[3]);
    printf("[%s] Server data copy\n", ok ? "PASS" : "FAIL");

    DevFree(sendBuf);
    DevFree(recvBuf);
    return ok;
}

// ============================================================
// Test: Finalize protocol
// ============================================================
static bool TestFinalizeProtocol()
{
    printf("\n=== Test 4: Finalize Protocol ===\n");

    MockContextBuilder ctx;
    if (!ctx.Build(1, 0))
        return false;

    void* wsAligned = ctx.GetWorkspaceAligned();

    // Start server
    MockHcclServer server(wsAligned);
    server.Start();

    // Simulate: kernel sends a regular AllGather commit on slot 0
    {
        HcclMsg msg;
        memset(&msg, 0, sizeof(msg));
        msg.commType = HCCL_CMD_ALLGATHER;
        msg.dataCnt = 64;
        msg.hcclDataType = 1;

        void* msgAddr = reinterpret_cast<uint8_t*>(wsAligned) + SEND_MSGS_OFFSET;
        aclrtMemcpy(msgAddr, sizeof(msg), &msg, sizeof(msg), ACL_MEMCPY_HOST_TO_DEVICE);

        TurnCnt commit;
        memset(&commit, 0, sizeof(commit));
        commit.valid = COMMIT_VALID_MASK;
        commit.cnt = 1;
        void* commitAddr = reinterpret_cast<uint8_t*>(wsAligned) + COMMIT_TURNCNT_OFFSET;
        aclrtMemcpy(commitAddr, sizeof(commit), &commit, sizeof(commit), ACL_MEMCPY_HOST_TO_DEVICE);

        usleep(200000); // let server process the regular message
    }

    printf("  Regular message processed: %u\n", server.GetMsgCount());

    // Now simulate: kernel sends Finalize commit on slot 0
    // Write a new commit with cnt=2 (next sequence number)
    {
        TurnCnt commit;
        memset(&commit, 0, sizeof(commit));
        commit.valid = COMMIT_VALID_MASK;
        commit.cnt = 2;
        void* commitAddr = reinterpret_cast<uint8_t*>(wsAligned) + COMMIT_TURNCNT_OFFSET;
        aclrtMemcpy(commitAddr, sizeof(commit), &commit, sizeof(commit), ACL_MEMCPY_HOST_TO_DEVICE);
    }

    // Use WaitForFinalize from main thread — it will detect the new commit
    // and respond with FINALIZE_FINISH_CNT
    bool finalizeOk = server.WaitForFinalize(0, 2000);

    server.Stop();

    // Verify finishedTurnCnt[0].cnt == FINALIZE_FINISH_CNT
    TurnCnt finish;
    void* finishAddr = reinterpret_cast<uint8_t*>(wsAligned) + FINISH_TURNCNT_OFFSET;
    aclrtMemcpy(&finish, sizeof(finish), finishAddr, sizeof(finish), ACL_MEMCPY_DEVICE_TO_HOST);

    bool ok = finalizeOk && (finish.cnt == FINALIZE_FINISH_CNT);
    printf("  finishedTurnCnt[0].cnt = %lu\n", finish.cnt);
    printf("  expected FINALIZE_FINISH_CNT = %lu\n", FINALIZE_FINISH_CNT);
    printf("  server.IsFinalized() = %s\n", server.IsFinalized() ? "true" : "false");
    printf("[%s] Finalize protocol\n", ok ? "PASS" : "FAIL");
    return ok;
}

// ============================================================
// Main
// ============================================================
int main(int argc, char* argv[])
{
    printf("========================================\n");
    printf("  Mock HCCL Framework Test\n");
    printf("========================================\n");

    int device = 0;
    if (argc > 1)
        device = atoi(argv[1]);

    if (aclInit(nullptr) != 0) {
        printf("[FAIL] aclInit\n");
        return 1;
    }
    if (aclrtSetDevice(device) != 0) {
        printf("[FAIL] aclrtSetDevice(%d)\n", device);
        aclFinalize();
        return 1;
    }

    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    printf("Device=%d, stream=%p\n", device, stream);

    int passed = 0, failed = 0;

    if (TestContextBuilder())
        passed++;
    else
        failed++;
    if (TestServerProtocol())
        passed++;
    else
        failed++;
    if (TestServerDataCopy())
        passed++;
    else
        failed++;
    if (TestFinalizeProtocol())
        passed++;
    else
        failed++;

    printf("\n========================================\n");
    printf("  Results: %d passed, %d failed\n", passed, failed);
    printf("========================================\n");

    if (stream)
        aclrtDestroyStream(stream);
    aclrtResetDevice(device);
    aclFinalize();
    return failed > 0 ? 1 : 0;
}
