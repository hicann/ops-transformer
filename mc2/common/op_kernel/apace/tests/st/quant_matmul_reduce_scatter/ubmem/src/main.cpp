/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file main.cpp
 * \brief ST host entry for QuantMatmul + ReduceScatter fusion (ubmem / HCCL variant)
 *
 * Multi-thread launch with HcclCommInitAll. The HCCL context device pointer is
 * obtained on the host side via HcclAllocComResourceByTiling (driven by
 * Mc2CcTilingConfig) and passed as the first kernel parameter.
 *
 * Operator semantics: MXFP8 MatMul + AlltoAll + ReduceAdd.
 *   - Each rank computes C_i = dequant(A_i) x dequant(B_i) -> [M, N]
 *   - AlltoAll scatters rows of C_i to other ranks
 *   - ReduceAdd sums all ranks' contributions for each rank's slice
 *   - Output: y_j = sum_i(C_i)[j*M/R : (j+1)*M/R, :] -> [M/R, N] (BF16)
 */

#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <iostream>
#include <limits.h>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>
#include <cstdlib>

#include "kernel_basic_intf.h"
#include "acl/acl.h"
#include "hccl/hccl.h"
#include "apace_st_utils.h"
#include "apace/utils/apace_constant.h"
#include "apace/utils/apace_common_utils.h"
#include "apace/tiling/quant_matmul_tiling_swat.h"
#include "apace/tiling/quant_matmul_tiling_data.h"
#include "tiling/hccl/hccl_tiling.h"
#include "apace_qmrs_kernel_launcher.h"

extern "C" HcclResult HcclCommInitAll(uint32_t commNum, int32_t *devIds, HcclComm *comm);
extern "C" HcclResult HcclCommDestroy(HcclComm comm);
extern "C" HcclResult HcclGetCommName(HcclComm comm, char *commName);
extern "C" HcclResult HcomGetCommHandleByGroup(const char *groupName, void **commHandle);
extern "C" HcclResult HcclAllocComResourceByTiling(HcclComm comm, void *stream, void *mc2Tiling, void **commContext);

static constexpr int32_t BENCHMARK_ITERATIONS = 20;
static constexpr int64_t MAX_RANK_NUM = 64;

static std::mutex gSetupMtx;
static std::condition_variable gSetupCv;
static int gSetupCount = 0;

static std::mutex gBarrierMtx;
static std::condition_variable gBarrierCv;
static int gBarrierCount = 0;

#define CHECK_RET(cond, return_expr) \
    do { \
        if (!(cond)) { \
            return_expr; \
        } \
    } while (0)

inline uint64_t CeilDivHost(uint64_t a, uint32_t b) { return (a + b - 1) / b; }

void ParseArgs(int argc, char *argv[], int *m, int *k, int *n, int *rankNum, std::string &mode)
{
    if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        std::cerr << "Usage: " << argv[0] << " m k n rankNum [mode]" << std::endl;
        std::cerr << "  m: row of matrix A (total M, must be divisible by rankNum)" << std::endl;
        std::cerr << "  k: col of matrix A (= row of matrix B, full K)" << std::endl;
        std::cerr << "  n: col of matrix B (total N)" << std::endl;
        std::cerr << "  rankNum: number of ranks" << std::endl;
        std::cerr << "  mode: optional, 'precision' (default) | 'perf'" << std::endl;
        exit(1);
    }
    if (argc < 5) {
        throw std::invalid_argument("ERROR: Lacks Arguments");
    }
    try {
        *m = std::stoi(argv[1]);
        *k = std::stoi(argv[2]);
        *n = std::stoi(argv[3]);
        *rankNum = std::stoi(argv[4]);
    } catch (const std::invalid_argument &) {
        throw std::invalid_argument("ERROR: m k n rankNum must be Integer");
    }
    if (*m <= 0 || *k <= 0 || *n <= 0 || *rankNum <= 0) {
        throw std::invalid_argument("ERROR: m k n rankNum must be positive");
    }
    if (static_cast<int64_t>(*rankNum) > MAX_RANK_NUM) {
        throw std::invalid_argument("ERROR: rankNum exceeds MAX_RANK_NUM");
    }
    if (*m % *rankNum != 0) {
        throw std::invalid_argument("ERROR: M must be divisible by rankNum for ReduceScatter");
    }
    if (*k % 32 != 0) {
        throw std::invalid_argument("ERROR: K must be divisible by 32 for MXFP8 quantization");
    }
    if (CeilDiv(*k, static_cast<int>(::MXFP_DIVISOR_SIZE)) % 2 != 0) {
        throw std::invalid_argument("ERROR: CeilDiv(K, 64) must be an even number");
    }
    mode = (argc >= 6) ? std::string(argv[5]) : std::string("precision");
    if (mode != "precision" && mode != "perf") {
        throw std::invalid_argument("ERROR: mode must be 'precision' or 'perf'");
    }
}

struct QuantMatmulReduceScatterTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    QuantMatmulTilingData mmTile;
};

int CreateTilingDataAndContext(const char *hcomName, aclrtStream stream, int m, int k, int n, int rankNum,
                               void **deviceTilingAddr, void **deviceContextAddr, uint32_t *usedCoreNum)
{
    QuantMatmulReduceScatterTilingData *tilingData = new QuantMatmulReduceScatterTilingData();
    if (tilingData == nullptr) {
        ERROR_LOG("tilingData alloc failed");
        return -1;
    }
    *tilingData = {};

    // Each rank computes the full M x N matmul (not M*rankNum).
    // The AlltoAll epilogue scatters the result, and ReduceAdd reduces.
    uint64_t totalM = static_cast<uint64_t>(m);
    QuantMatmulTilingSwat<mm::DataType::DT_FLOAT8_E4M3FN, mm::DataType::DT_FLOAT8_E4M3FN> tilingEngine;
    tilingEngine.SetOptimizeEnable(false);
    tilingEngine.SetMTailAlignEnable(true);

    // Limit baseM to tpSizeM (per-rank M slice), matching experimental tiling logic.
    constexpr uint64_t CUBE_BLOCK_NUM = 16;
    constexpr uint64_t MAX_BASIC_BLOCK_SIZE = 256;
    uint64_t tpSizeM = totalM / static_cast<uint64_t>(rankNum);
    uint64_t dynMaxBaseM = std::min(MAX_BASIC_BLOCK_SIZE, (tpSizeM / CUBE_BLOCK_NUM) * CUBE_BLOCK_NUM);
    if (dynMaxBaseM < CUBE_BLOCK_NUM) {
        dynMaxBaseM = std::min(MAX_BASIC_BLOCK_SIZE, (totalM / 2 / CUBE_BLOCK_NUM) * CUBE_BLOCK_NUM);
    }
    if (dynMaxBaseM < CUBE_BLOCK_NUM) {
        dynMaxBaseM = std::min(MAX_BASIC_BLOCK_SIZE, totalM);
    }
    tilingEngine.SetMaxBaseM(dynMaxBaseM);

    tilingEngine.GetTilingData(totalM, static_cast<uint64_t>(n), static_cast<uint64_t>(k), false, false,
                               tilingData->mmTile);

    *usedCoreNum = tilingData->mmTile.usedCoreNum;
    INFO_LOG("Tiling: M=%d, K=%d, N=%d, rankNum=%d, dynMaxBaseM=%lu, usedCoreNum=%u", m, k, n, rankNum, dynMaxBaseM,
             *usedCoreNum);

    // opType=8 is ALL_TO_ALL (not 6 which is ALL_GATHER)
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(hcomName, 8, "AlltoAll=level0:fullmesh;level1:pairwise");
    mc2CcTilingConfig.SetCommEngine(3);
    int ret = mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling);
    CHECK_RET(ret == ACL_SUCCESS, ERROR_LOG("GetTiling mc2InitTiling failed. ret = %d", ret); delete tilingData;
              return ret);
    ret = mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling);
    CHECK_RET(ret == ACL_SUCCESS, ERROR_LOG("GetTiling mc2CcTiling failed. ret = %d", ret); delete tilingData;
              return ret);

    int mmTileSize = sizeof(QuantMatmulTilingData);
    ret = aclrtMalloc(deviceTilingAddr, mmTileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, ERROR_LOG("aclrtMalloc TilingData failed. ret = %d", ret); delete tilingData;
              return ret);
    ret = aclrtMemcpy(*deviceTilingAddr, mmTileSize, &tilingData->mmTile, mmTileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, ERROR_LOG("aclrtMemcpy TilingData failed. ret = %d", ret);
              aclrtFree(*deviceTilingAddr); *deviceTilingAddr = nullptr; delete tilingData; return ret);

    HcclComm commHandle;
    ret = HcomGetCommHandleByGroup(hcomName, reinterpret_cast<void **>(&commHandle));
    CHECK_RET(ret == ACL_SUCCESS, ERROR_LOG("HcomGetCommHandleByGroup failed. ret = %d", ret);
              aclrtFree(*deviceTilingAddr); *deviceTilingAddr = nullptr; delete tilingData; return ret);

    void *mc2Context = nullptr;
    ret = HcclAllocComResourceByTiling(commHandle, stream, tilingData, &mc2Context);
    CHECK_RET(ret == ACL_SUCCESS, ERROR_LOG("HcclAllocComResourceByTiling failed. ret = %d", ret);
              aclrtFree(*deviceTilingAddr); *deviceTilingAddr = nullptr; delete tilingData; return ret);
    if (mc2Context == nullptr) {
        ERROR_LOG("mc2Context is nullptr");
        aclrtFree(*deviceTilingAddr);
        *deviceTilingAddr = nullptr;
        delete tilingData;
        return -1;
    }
    *deviceContextAddr = mc2Context;

    delete tilingData;
    return ACL_SUCCESS;
}

struct Args {
    uint32_t rankId;
    uint32_t rankDim;
    HcclComm hcclComm;
    aclrtStream stream;
    aclrtContext context;
    int m;
    int k;
    int n;
    std::string mode;
};

int LaunchOneThread(Args &args)
{
    int ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, ERROR_LOG("aclrtSetCurrentContext failed. ret = %d", ret); return ret);

    char hcomName[128] = {0};
    ret = HcclGetCommName(args.hcclComm, hcomName);
    CHECK_RET(ret == ACL_SUCCESS, ERROR_LOG("HcclGetCommName failed. ret = %d", ret); return ret);
    INFO_LOG("rank=%d hcomName=%s", args.rankId, hcomName);

    void *tilingAddr = nullptr;
    void *mc2ContextAddr = nullptr;
    uint32_t usedCoreNum = 0;
    ret = CreateTilingDataAndContext(hcomName, args.stream, args.m, args.k, args.n, args.rankDim, &tilingAddr,
                                     &mc2ContextAddr, &usedCoreNum);
    CHECK_RET(ret == ACL_SUCCESS, ERROR_LOG("CreateTilingDataAndContext failed. ret = %d", ret); return ret);

    uint64_t m = args.m, k = args.k, n = args.n, rankNum = args.rankDim;
    uint64_t tpSizeM = m / rankNum;
    uint64_t sizeA = m * k * sizeof(uint8_t);
    // B: NZ layout {ceil(N/32), ceil(K/16), 16, 32}
    uint64_t ntNz = CeilDivHost(n, 32);
    uint64_t ktNz = CeilDivHost(k, 16);
    uint64_t sizeB = ntNz * ktNz * 16 * 32 * sizeof(uint8_t);
    uint64_t sizeScaleA = m * CeilDivHost(k, ::MXFP_DIVISOR_SIZE) * ::MXFP_MULTI_BASE_SIZE * sizeof(uint8_t);
    // ScaleB: NZ layout {ceil(K/64), N, 2}
    uint64_t sizeScaleB = CeilDivHost(k, ::MXFP_DIVISOR_SIZE) * n * ::MXFP_MULTI_BASE_SIZE * sizeof(uint8_t);
    // Output per rank: [M/rankNum, N] as BF16
    uint64_t sizeOutput = tpSizeM * n * sizeof(uint16_t);
    // Workspace: matmul output [M, N] as BF16 (NZ layout uses workspace, not y)
    uint64_t sizeWorkspace = m * n * sizeof(uint16_t);

    void *deviceA = nullptr, *deviceB = nullptr, *deviceScaleA = nullptr, *deviceScaleB = nullptr;
    void *deviceOutput = nullptr, *deviceWorkspace = nullptr;

    ret = aclrtMalloc(&deviceA, sizeA, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, ERROR_LOG("aclrtMalloc deviceA failed. ret = %d", ret); return ret);
    ret = aclrtMalloc(&deviceB, sizeB, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, ERROR_LOG("aclrtMalloc deviceB failed. ret = %d", ret); return ret);
    ret = aclrtMalloc(&deviceScaleA, sizeScaleA, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, ERROR_LOG("aclrtMalloc deviceScaleA failed. ret = %d", ret); return ret);
    ret = aclrtMalloc(&deviceScaleB, sizeScaleB, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, ERROR_LOG("aclrtMalloc deviceScaleB failed. ret = %d", ret); return ret);
    ret = aclrtMalloc(&deviceOutput, sizeOutput, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, ERROR_LOG("aclrtMalloc deviceOutput failed. ret = %d", ret); return ret);
    ret = aclrtMalloc(&deviceWorkspace, sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, ERROR_LOG("aclrtMalloc deviceWorkspace failed. ret = %d", ret); return ret);

    std::vector<uint8_t> hostA(sizeA, 0);
    std::vector<uint8_t> hostB(sizeB, 0);
    std::vector<uint8_t> hostScaleA(sizeScaleA, 0);
    std::vector<uint8_t> hostScaleB(sizeScaleB, 0);

    char exePath[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", exePath, PATH_MAX - 1);
    std::string baseDir = ".";
    if (len > 0) {
        exePath[len] = '\0';
        baseDir = exePath;
        size_t lastSlash = baseDir.find_last_of('/');
        if (lastSlash != std::string::npos && lastSlash > 0) {
            baseDir.resize(lastSlash);
        }
    }
    std::string inputDir = baseDir + "/input/" + std::to_string(args.rankId);
    if (!ReadFile(inputDir + "/input_a.bin", hostA.data(), sizeA)) {
        INFO_LOG("rank=%d input_a.bin missing, using zeros", args.rankId);
    }
    if (!ReadFile(inputDir + "/input_b.bin", hostB.data(), sizeB)) {
        INFO_LOG("rank=%d input_b.bin missing, using zeros", args.rankId);
    }
    if (!ReadFile(inputDir + "/input_scaleA.bin", hostScaleA.data(), sizeScaleA)) {
        INFO_LOG("rank=%d input_scaleA.bin missing, using zeros", args.rankId);
    }
    if (!ReadFile(inputDir + "/input_scaleB.bin", hostScaleB.data(), sizeScaleB)) {
        INFO_LOG("rank=%d input_scaleB.bin missing, using zeros", args.rankId);
    }

    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceScaleA, sizeScaleA, hostScaleA.data(), sizeScaleA, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceScaleB, sizeScaleB, hostScaleB.data(), sizeScaleB, ACL_MEMCPY_HOST_TO_DEVICE));

    INFO_LOG("rank=%d Ready to launch kernel: M=%lu, N=%lu, K=%lu, tpSizeM=%lu, rankDim=%d, usedCoreNum=%u",
             args.rankId, m, n, k, tpSizeM, args.rankDim, usedCoreNum);

    {
        std::unique_lock<std::mutex> lock(gSetupMtx);
        gSetupCount++;
        if (gSetupCount >= static_cast<int>(args.rankDim)) {
            gSetupCv.notify_all();
        } else {
            gSetupCv.wait(lock, [&]() { return gSetupCount >= static_cast<int>(args.rankDim); });
        }
    }

    QuantMatmulReduceScatterUbmemKernel<<<usedCoreNum, nullptr, args.stream>>>(
        (GM_ADDR)mc2ContextAddr, (GM_ADDR)deviceA, (GM_ADDR)deviceB, (GM_ADDR)deviceScaleA, (GM_ADDR)deviceScaleB,
        (GM_ADDR)deviceOutput, (GM_ADDR)deviceWorkspace, (GM_ADDR)tilingAddr);

    ret = aclrtSynchronizeStream(args.stream);
    CHECK_RET(ret == ACL_SUCCESS, ERROR_LOG("aclrtSynchronizeStream failed. ret = %d", ret); return ret);
    INFO_LOG("rank=%d kernel done", args.rankId);

    size_t outputSize = static_cast<size_t>(tpSizeM) * n * sizeof(uint16_t);
    std::vector<uint8_t> hostOutput(outputSize, 0);
    ACL_CHECK(aclrtMemcpy(hostOutput.data(), outputSize, deviceOutput, outputSize, ACL_MEMCPY_DEVICE_TO_HOST));

    std::string outputDir = baseDir + "/output/" + std::to_string(args.rankId);
    {
        std::string cmd = "mkdir -p " + outputDir;
        system(cmd.c_str());
    }
    WriteFile(outputDir + "/npu_out.bin", hostOutput.data(), outputSize);

    int32_t iterations = (args.mode == "perf") ? BENCHMARK_ITERATIONS : 1;
    if (iterations > 1) {
        aclrtEvent kernelStartEvent = nullptr, kernelEndEvent = nullptr;
        ACL_CHECK(aclrtCreateEvent(&kernelStartEvent));
        ACL_CHECK(aclrtCreateEvent(&kernelEndEvent));
        ACL_CHECK(aclrtRecordEvent(kernelStartEvent, args.stream));
        for (int i = 1; i < iterations; ++i) {
            QuantMatmulReduceScatterUbmemKernel<<<usedCoreNum, nullptr, args.stream>>>(
                (GM_ADDR)mc2ContextAddr, (GM_ADDR)deviceA, (GM_ADDR)deviceB, (GM_ADDR)deviceScaleA,
                (GM_ADDR)deviceScaleB, (GM_ADDR)deviceOutput, (GM_ADDR)deviceWorkspace, (GM_ADDR)tilingAddr);
        }
        ACL_CHECK(aclrtRecordEvent(kernelEndEvent, args.stream));
        ACL_CHECK(aclrtSynchronizeStream(args.stream));

        float kernelElapsedMs = 0.0F;
        ACL_CHECK(aclrtEventElapsedTime(&kernelElapsedMs, kernelStartEvent, kernelEndEvent));
        double kernelElapsedUs = static_cast<double>(kernelElapsedMs) * 1000.0;
        INFO_LOG("rank=%d Kernel perf: %.3f us (avg over %d iterations)", args.rankId,
                 kernelElapsedUs / (iterations - 1), iterations - 1);
        if (kernelEndEvent)
            aclrtDestroyEvent(kernelEndEvent);
        if (kernelStartEvent)
            aclrtDestroyEvent(kernelStartEvent);
    } else {
        INFO_LOG("rank=%d Kernel completed! (precision mode)", args.rankId);
    }

    aclrtFree(deviceA);
    aclrtFree(deviceB);
    aclrtFree(deviceScaleA);
    aclrtFree(deviceScaleB);
    aclrtFree(deviceOutput);
    aclrtFree(deviceWorkspace);
    aclrtFree(mc2ContextAddr);
    aclrtFree(tilingAddr);

    {
        std::unique_lock<std::mutex> lock(gBarrierMtx);
        gBarrierCount++;
        if (gBarrierCount >= static_cast<int>(args.rankDim)) {
            gBarrierCv.notify_all();
        } else {
            gBarrierCv.wait(lock, [&]() { return gBarrierCount >= static_cast<int>(args.rankDim); });
        }
    }
    HcclCommDestroy(args.hcclComm);

    INFO_LOG("rank=%d resources freed", args.rankId);
    return ACL_SUCCESS;
}

int main(int argc, char *argv[])
{
    int m = 0, k = 0, n = 0, rankNum = 0;
    std::string mode;
    try {
        ParseArgs(argc, argv, &m, &k, &n, &rankNum, mode);
    } catch (const std::invalid_argument &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }
    INFO_LOG("Config: DEV_NUM=%d, M=%d, N=%d, K=%d, mode=%s", rankNum, m, n, k, mode.c_str());

    int ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, ERROR_LOG("aclInit failed. ret = %d", ret); return ret);

    uint32_t devNum = static_cast<uint32_t>(rankNum);
    std::vector<aclrtStream> stream(devNum);
    std::vector<aclrtContext> context(devNum);
    for (uint32_t rankId = 0; rankId < devNum; rankId++) {
        ret = aclrtSetDevice(rankId);
        CHECK_RET(ret == ACL_SUCCESS, ERROR_LOG("aclrtSetDevice failed. ret = %d", ret); return ret);
        ret = aclrtCreateContext(&context[rankId], rankId);
        CHECK_RET(ret == ACL_SUCCESS, ERROR_LOG("aclrtCreateContext failed. ret = %d", ret); return ret);
        ret = aclrtCreateStream(&stream[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, ERROR_LOG("aclrtCreateStream failed. ret = %d", ret); return ret);
    }

    std::vector<int32_t> devices(devNum);
    for (uint32_t i = 0; i < devNum; i++) {
        devices[i] = static_cast<int32_t>(i);
    }

    std::vector<HcclComm> comms(devNum);
    ret = HcclCommInitAll(devNum, devices.data(), comms.data());
    CHECK_RET(ret == ACL_SUCCESS, ERROR_LOG("HcclCommInitAll failed. ret = %d", ret); return ret);

    std::vector<Args> args(devNum);
    std::vector<std::unique_ptr<std::thread>> threads(devNum);
    std::vector<int> returnCodes(devNum, 0);
    for (uint32_t rankId = 0; rankId < devNum; rankId++) {
        args[rankId].rankId = rankId;
        args[rankId].rankDim = devNum;
        args[rankId].hcclComm = comms[rankId];
        args[rankId].context = context[rankId];
        args[rankId].stream = stream[rankId];
        args[rankId].m = m;
        args[rankId].k = k;
        args[rankId].n = n;
        args[rankId].mode = mode;
        threads[rankId].reset(
            new std::thread([&args, &returnCodes, rankId]() { returnCodes[rankId] = LaunchOneThread(args[rankId]); }));
    }
    for (uint32_t rankId = 0; rankId < devNum; rankId++)
        if (threads[rankId])
            threads[rankId]->join();
    for (uint32_t rankId = 0; rankId < devNum; rankId++) {
        aclrtDestroyStream(stream[rankId]);
        aclrtResetDevice(rankId);
        aclrtDestroyContext(context[rankId]);
    }
    aclFinalize();

    int finalRet = 0;
    for (uint32_t rankId = 0; rankId < devNum; rankId++) {
        if (returnCodes[rankId] != 0) {
            finalRet = 1;
        }
    }
    std::cout << "All workers finished. Status: " << (finalRet == 0 ? "SUCCESS" : "FAILURE") << std::endl;
    return finalRet;
}
