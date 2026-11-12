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
#include "hccl/hccl_rank_graph.h"
#include "apace_st_utils.h"
#include "apace/utils/apace_constant.h"
#include "apace/utils/apace_common_utils.h"
#include "apace/tiling/matmul_tiling/quant_matmul_tiling.h"
#include "apace/tiling/matmul_tiling/quant_matmul_tiling_data.h"
#include "apace/utils/comm_channel_builder.h"
#include "tiling/hccl/hccl_tiling.h"
#include "apace_qmrs_kernel_launcher.h"

extern "C" HcclResult HcclCommInitAll(uint32_t commNum, int32_t* devIds, HcclComm* comm);
extern "C" HcclResult HcclCommDestroy(HcclComm comm);
extern "C" HcclResult HcclGetCommName(HcclComm comm, char* commName);
extern "C" HcclResult HcomGetCommHandleByGroup(const char* groupName, void** commHandle);
extern "C" HcclResult HcclAllocComResourceByTiling(HcclComm comm, void* stream, void* mc2Tiling, void** commContext);

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

inline uint64_t CeilDivHost(uint64_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

STXDtype ParseXdtype(const std::string& s)
{
    if (s == "e4m3" || s == "fp8_e4m3")
        return STXDtype::kE4M3;
    if (s == "e5m2" || s == "fp8_e5m2")
        return STXDtype::kE5M2;
    if (s == "fp4" || s == "fp4_e2m1")
        return STXDtype::kFP4;
    throw std::invalid_argument("ERROR: x1dtype/x2dtype must be e4m3|fp8_e4m3|e5m2|fp8_e5m2|fp4|fp4_e2m1, got " + s);
}

STYDtype ParseYdtype(const std::string& s)
{
    if (s == "bf16")
        return STYDtype::kBF16;
    if (s == "fp16")
        return STYDtype::kFP16;
    throw std::invalid_argument("ERROR: ydtype must be bf16|fp16, got " + s);
}

STX2Layout ParseLayout(const std::string& s)
{
    if (s == "dn")
        return STX2Layout::kDN;
    if (s == "nd")
        return STX2Layout::kND;
    if (s == "zn")
        return STX2Layout::kZN;
    if (s == "nz")
        return STX2Layout::kNZ;
    throw std::invalid_argument("ERROR: x2layout must be dn|nd|zn|nz, got " + s);
}

inline bool IsTransB(STX2Layout ly)
{
    return (ly == STX2Layout::kDN || ly == STX2Layout::kZN);
}

void ParseArgs(int argc, char* argv[], int* m, int* k, int* n, int* rankNum, STXDtype* xa, STXDtype* xb, STYDtype* yt,
               STX2Layout* ly, std::string& mode)
{
    if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        std::cerr << "Usage: " << argv[0] << " m k n rankNum [x1dtype x2dtype ydtype x2layout] [mode]" << std::endl;
        std::cerr << "  m: row of matrix A (total M, must be divisible by rankNum)" << std::endl;
        std::cerr << "  k: col of matrix A (= row of matrix B, full K)" << std::endl;
        std::cerr << "  n: col of matrix B (total N)" << std::endl;
        std::cerr << "  rankNum: number of ranks" << std::endl;
        std::cerr << "  x1dtype/x2dtype: optional, e4m3|e5m2|fp4 (default e4m3/e4m3)" << std::endl;
        std::cerr << "  ydtype: optional, bf16|fp16 (default bf16)" << std::endl;
        std::cerr << "  x2layout: optional, dn|nd|zn|nz (default nz)" << std::endl;
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
    } catch (const std::invalid_argument&) {
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
        throw std::invalid_argument("ERROR: K must be divisible by 32 for MX quantization");
    }
    if (CeilDiv(*k, static_cast<int>(::MXFP_DIVISOR_SIZE)) % 2 != 0) {
        throw std::invalid_argument("ERROR: CeilDiv(K, 64) must be an even number");
    }
    *xa = STXDtype::kE4M3;
    *xb = STXDtype::kE4M3;
    *yt = STYDtype::kBF16;
    *ly = STX2Layout::kNZ;
    if (argc >= 9) {
        *xa = ParseXdtype(argv[5]);
        *xb = ParseXdtype(argv[6]);
        *yt = ParseYdtype(argv[7]);
        *ly = ParseLayout(argv[8]);
    }
    mode = (argc >= 10) ? std::string(argv[9]) : std::string("precision");
    if (mode != "precision" && mode != "perf") {
        throw std::invalid_argument("ERROR: mode must be 'precision' or 'perf'");
    }
    // fp4 约束：仅 fp4+fp4，且仅 DN/ND（不支持 ZN/NZ）
    if (*xa == STXDtype::kFP4 || *xb == STXDtype::kFP4) {
        if (*xa != STXDtype::kFP4 || *xb != STXDtype::kFP4) {
            throw std::invalid_argument("ERROR: FP4 requires both x1 and x2 to be fp4");
        }
        if (*ly == STX2Layout::kZN || *ly == STX2Layout::kNZ) {
            throw std::invalid_argument("ERROR: FP4 does not support ZN/NZ x2 layout");
        }
    }
}

int CreateTilingDataAndContext(HcclComm comm, int m, int k, int n, int rankNum, STXDtype xa, bool transB,
                               void** deviceTilingAddr, void** deviceContextAddr, uint32_t* usedCoreNum)
{
    QuantMatmulTilingData mmTile{};

    // Each rank computes the full M x N matmul (not M*rankNum).
    // The AlltoAll epilogue scatters the result, and ReduceAdd reduces.
    uint64_t totalM = static_cast<uint64_t>(m);
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
    // SWAT 引擎按 x1 dtype 分发（对齐 op tiling：fp8 统一 e4m3，fp4 用 e2m1）
    if (xa == STXDtype::kFP4) {
        QuantMatmulTilingSwat<mm::DataType::DT_FLOAT4_E2M1, mm::DataType::DT_FLOAT4_E2M1, mm::BiasDataType::DT_FLOAT>
            tilingEngine;
        tilingEngine.SetOptimizeEnable(false);
        tilingEngine.SetMTailAlignEnable(true);
        tilingEngine.SetMaxBaseM(dynMaxBaseM);
        tilingEngine.GetTilingData(totalM, static_cast<uint64_t>(n), static_cast<uint64_t>(k), false, transB, mmTile);
    } else {
        QuantMatmulTilingSwat<mm::DataType::DT_FLOAT8_E4M3FN, mm::DataType::DT_FLOAT8_E4M3FN,
                              mm::BiasDataType::DT_FLOAT>
            tilingEngine;
        tilingEngine.SetOptimizeEnable(false);
        tilingEngine.SetMTailAlignEnable(true);
        tilingEngine.SetMaxBaseM(dynMaxBaseM);
        tilingEngine.GetTilingData(totalM, static_cast<uint64_t>(n), static_cast<uint64_t>(k), false, transB, mmTile);
    }

    *usedCoreNum = mmTile.usedCoreNum;
    INFO_LOG("Tiling: M=%d, K=%d, N=%d, rankNum=%d, dynMaxBaseM=%lu, usedCoreNum=%u", m, k, n, rankNum, dynMaxBaseM,
             *usedCoreNum);

    void* commBuffer = nullptr;
    uint64_t commBufferSize = 0;
    auto ret = HcclGetHcclBuffer(comm, &commBuffer, &commBufferSize);
    CHECK_RET(ret == ::HCCL_SUCCESS && commBuffer != nullptr, ERROR_LOG("HcclGetHcclBuffer failed. ret = %d", ret);
              return -1);
    uint64_t requiredBytes = APACE_MTE_STATE_WIN_SIZE + static_cast<uint64_t>(rankNum - 1) * APACE_EP_RANK_OFFSET_STEP +
                             static_cast<uint64_t>(m) * static_cast<uint64_t>(n) * sizeof(uint16_t);
    CHECK_RET(commBufferSize >= requiredBytes,
              ERROR_LOG("Communication buffer too small: available=%lu, required=%lu", commBufferSize, requiredBytes);
              return -1);

    // UBMEM builder exchanges directly accessible addresses and zeroes the buffer.
    CommChannelBuilder<ChannelMode::UBMEM> builder(comm);
    Apace::AivComm::CommUbmemContext hostContext{};
    void* ubmemContext =
        builder.CreateDeviceContext(&hostContext, sizeof(hostContext), "apace_qmrs_ubmem", &hostContext);
    CHECK_RET(ubmemContext != nullptr, ERROR_LOG("CreateDeviceContext failed"); return -1);

    // gSetupCv ensures all ranks complete channel setup before launching the kernel.
    int mmTileSize = sizeof(mmTile);
    int aclRet = aclrtMalloc(deviceTilingAddr, mmTileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, ERROR_LOG("aclrtMalloc TilingData failed. ret = %d", aclRet); return aclRet);
    aclRet = aclrtMemcpy(*deviceTilingAddr, mmTileSize, &mmTile, mmTileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, ERROR_LOG("aclrtMemcpy TilingData failed. ret = %d", aclRet);
              aclrtFree(*deviceTilingAddr); *deviceTilingAddr = nullptr; return aclRet);
    *deviceContextAddr = ubmemContext;
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
    STXDtype x1dtype;
    STXDtype x2dtype;
    STYDtype ydtype;
    STX2Layout layout;
    std::string mode;
};

int LaunchOneThread(Args& args)
{
    int ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, ERROR_LOG("aclrtSetCurrentContext failed. ret = %d", ret); return ret);

    char hcomName[128] = {0};
    ret = HcclGetCommName(args.hcclComm, hcomName);
    CHECK_RET(ret == ACL_SUCCESS, ERROR_LOG("HcclGetCommName failed. ret = %d", ret); return ret);
    INFO_LOG("rank=%d hcomName=%s", args.rankId, hcomName);

    void* tilingAddr = nullptr;
    void* ubmemContextAddr = nullptr;
    uint32_t usedCoreNum = 0;
    ret = CreateTilingDataAndContext(args.hcclComm, args.m, args.k, args.n, args.rankDim, args.x1dtype,
                                     IsTransB(args.layout), &tilingAddr, &ubmemContextAddr, &usedCoreNum);
    CHECK_RET(ret == ACL_SUCCESS, ERROR_LOG("CreateTilingDataAndContext failed. ret = %d", ret); return ret);

    uint64_t m = args.m, k = args.k, n = args.n, rankNum = args.rankDim;
    uint64_t tpSizeM = m / rankNum;
    // dtype 位宽：fp8 1B/元素；fp4 打包 2 元素/字节（沿 A 的 K、B 的 last dim）
    bool isFp4 = (args.x1dtype == STXDtype::kFP4);
    // A：[M, K]，fp4 打包 → m*(k/2) 字节
    uint64_t sizeA = m * k / (isFp4 ? 2 : 1);
    // B 按布局：DN [N,K]（fp4 打包 K）、ND [K,N]（fp4 打包 N）、ZN/NZ 分形 {*,*,16,32}（fp8 only）
    uint64_t sizeB = 0;
    if (args.layout == STX2Layout::kDN) {
        sizeB = n * k / (isFp4 ? 2 : 1);
    } else if (args.layout == STX2Layout::kND) {
        sizeB = k * n / (isFp4 ? 2 : 1);
    } else if (args.layout == STX2Layout::kZN) {
        // [N,K] NZ：{ceil(K/32), ceil(N/16), 16, 32}
        sizeB = CeilDivHost(k, 32) * CeilDivHost(n, 16) * 16 * 32;
    } else {
        // [K,N] NZ：{ceil(N/32), ceil(K/16), 16, 32}
        sizeB = CeilDivHost(n, 32) * CeilDivHost(k, 16) * 16 * 32;
    }
    uint64_t kd = CeilDivHost(k, ::MXFP_DIVISOR_SIZE);
    uint64_t sizeScaleA = m * kd * ::MXFP_MULTI_BASE_SIZE;
    // ScaleB 语义随 transB：DN/ZN [n,kd,2]；ND/NZ [kd,n,2]（字节数相同）
    uint64_t sizeScaleB = kd * n * ::MXFP_MULTI_BASE_SIZE;
    // Output per rank: [M/rankNum, N]，y dtype bf16/fp16 均 2B
    uint64_t sizeOutput = tpSizeM * n * sizeof(uint16_t);
    // Workspace: matmul output [M, N]，y dtype 2B
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

    LaunchQmrsKernel(args.x1dtype, args.x2dtype, args.ydtype, args.layout, usedCoreNum, args.stream,
                     (GM_ADDR)ubmemContextAddr, (GM_ADDR)deviceA, (GM_ADDR)deviceB, (GM_ADDR)deviceScaleA,
                     (GM_ADDR)deviceScaleB, (GM_ADDR)deviceOutput, (GM_ADDR)deviceWorkspace, (GM_ADDR)tilingAddr);

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
            LaunchQmrsKernel(args.x1dtype, args.x2dtype, args.ydtype, args.layout, usedCoreNum, args.stream,
                             (GM_ADDR)ubmemContextAddr, (GM_ADDR)deviceA, (GM_ADDR)deviceB, (GM_ADDR)deviceScaleA,
                             (GM_ADDR)deviceScaleB, (GM_ADDR)deviceOutput, (GM_ADDR)deviceWorkspace,
                             (GM_ADDR)tilingAddr);
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

int main(int argc, char* argv[])
{
    int m = 0, k = 0, n = 0, rankNum = 0;
    STXDtype xa = STXDtype::kE4M3, xb = STXDtype::kE4M3;
    STYDtype yt = STYDtype::kBF16;
    STX2Layout ly = STX2Layout::kNZ;
    std::string mode;
    try {
        ParseArgs(argc, argv, &m, &k, &n, &rankNum, &xa, &xb, &yt, &ly, mode);
    } catch (const std::invalid_argument& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }
    const char* xdNames[] = {"e4m3", "e5m2", "fp4"};
    const char* ydNames[] = {"bf16", "fp16"};
    const char* lyNames[] = {"dn", "nd", "zn", "nz"};
    INFO_LOG("Config: DEV_NUM=%d, M=%d, N=%d, K=%d, x1=%s x2=%s y=%s layout=%s mode=%s", rankNum, m, n, k,
             xdNames[static_cast<int>(xa)], xdNames[static_cast<int>(xb)], ydNames[static_cast<int>(yt)],
             lyNames[static_cast<int>(ly)], mode.c_str());
    // (Config 已在 ParseArgs 后打印)

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
        args[rankId].x1dtype = xa;
        args[rankId].x2dtype = xb;
        args[rankId].ydtype = yt;
        args[rankId].layout = ly;
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
