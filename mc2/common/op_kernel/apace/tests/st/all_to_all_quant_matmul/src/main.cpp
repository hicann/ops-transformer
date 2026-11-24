/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <limits.h>
#include <unistd.h>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include <sys/wait.h>

#include "kernel_basic_intf.h"
#include "acl/acl.h"
#include "hccl/hccl.h"
#include "hccl/hccl_comm.h"
#include "hccl/hccl_res.h"
#include "hccl/hccl_types.h"
#include "hccl/hccl_rank_graph.h"
#include <cstdlib>
#include "apace_st_utils.h"
#include "apace/utils/comm_channel_builder.h"
#include "../../utils/root_info_exchanger.h"
#include "apace/tiling/matmul_tiling_swat.h"
#include "apace/tiling/matmul_tiling/quant_matmul_tiling.h"
#include "apace/kernel/fusions/all_to_all_matmul/all_to_all_matmul_tiling_data.h"
#include "apace_a2a_kernel_launcher.h"
#include "apace/core/aiv_comm/collective_comm_context.h"

#define HCCL_CHECK(status) \
    do { \
        HcclResult ret = (status); \
        if (ret != HCCL_SUCCESS) { \
            std::cerr << __FILE__ << ":" << __LINE__ << " HcclResult:" << ret << std::endl; \
        } \
    } while (0)

inline uint64_t CeilDiv(uint32_t a, uint32_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

void printUsage(const std::string& programName)
{
    std::cerr << "Usage: " << programName
              << " m k n rankNum [mode] [headMSize] [bufferCount] [type] [dtype] [localMatmul]" << std::endl;
    std::cerr << "Args: " << std::endl;
    std::cerr << "  m: row of matrix A (per-rank output rows)" << std::endl;
    std::cerr << "  k: col of matrix A (total K, distributed across ranks)" << std::endl;
    std::cerr << "  n: col of matrix B" << std::endl;
    std::cerr << "  rankNum: number of ranks" << std::endl;
    std::cerr << "  mode: optional, 'precision' (default) | 'perf'" << std::endl;
    std::cerr << "  headMSize: optional, long block M size (0 = auto from tiling, non-quant only)" << std::endl;
    std::cerr << "  bufferCount: optional, win buffer slot reuse count, 0=no reuse (default 0)" << std::endl;
    std::cerr << "  type: optional, 'quant' (default) | 'nonquant'" << std::endl;
    std::cerr << "  dtype: optional (nonquant), 'fp16' (default) | 'bf16'" << std::endl;
    std::cerr << "  localMatmul: optional (nonquant), 0=REMOTE | 1=DEFERRED_SYNC | 2=GM累加 (default 0)" << std::endl;
    std::cerr << "Example: " << programName << " 1024 1024 5120 4 precision 0 0 nonquant bf16 0" << std::endl;
    std::cerr << "         " << programName << " 16384 32768 4096 2 precision 512 8 quant 0 0 0" << std::endl;
}

void parseArguments(int argc, char* argv[], int& m, int& k, int& n, int& rankNum, std::string& mode, int& headMSize,
                    int& bufferCount, std::string& type, std::string& dtype, int& localMatmul)
{
    if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        printUsage(argv[0]);
        exit(1);
    }
    if (argc < 5) {
        throw std::invalid_argument("ERROR: Lacks Arguments");
    }
    try {
        m = std::stoi(argv[1]);
        k = std::stoi(argv[2]);
        n = std::stoi(argv[3]);
        rankNum = std::stoi(argv[4]);
    } catch (const std::invalid_argument&) {
        throw std::invalid_argument("ERROR: m k n rankNum must be Integer");
    }

    mode = (argc >= 6) ? std::string(argv[5]) : std::string("precision");
    if (mode != "precision" && mode != "perf") {
        throw std::invalid_argument("ERROR: mode must be 'precision' or 'perf'");
    }

    headMSize = (argc >= 7) ? std::stoi(argv[6]) : 0;
    if (headMSize < 0) {
        throw std::invalid_argument("ERROR: headMSize must be non-negative (0 = auto from tiling)");
    }

    bufferCount = (argc >= 8) ? std::stoi(argv[7]) : 0;
    if (bufferCount < 0) {
        throw std::invalid_argument("ERROR: bufferCount must be >= 0");
    }

    type = (argc >= 9) ? std::string(argv[8]) : std::string("quant");
    if (type != "quant" && type != "nonquant") {
        throw std::invalid_argument("ERROR: type must be 'quant' or 'nonquant'");
    }

    dtype = (argc >= 10) ? std::string(argv[9]) : std::string("fp16");
    if (dtype != "fp16" && dtype != "bf16") {
        throw std::invalid_argument("ERROR: dtype must be 'fp16' or 'bf16'");
    }

    localMatmul = (argc >= 11) ? std::stoi(argv[10]) : 0;
    if (localMatmul < 0 || localMatmul > 2) {
        throw std::invalid_argument("ERROR: localMatmul must be in [0, 2]");
    }

    if (m <= 0 || k <= 0 || n <= 0 || rankNum <= 0) {
        throw std::invalid_argument("ERROR: m k n rankNum must be positive");
    }

    if (k % rankNum != 0) {
        throw std::invalid_argument("ERROR: k must be divisible by rankNum");
    }

    uint32_t ka = k / rankNum;
    if (CeilDiv(ka, 64) % 2 != 0) {
        throw std::invalid_argument("ERROR: CeilDiv(Ka, 64) must be an even number");
    }
}

// ---------------- 非量化路径：填充 tileMmTilingData ----------------
// 返回实际采用的 headMSize（0 = 从 tiling 自动推导）
uint32_t fillNonQuantTiling(AllToAllMatmulTilingData& tilingData, int m, int k, int n, int rankNum, int headMSizeArg,
                            int bufferCountArg, int localMatmul, uint32_t& headTileCntOut)
{
    uint32_t ka = k / rankNum;
    MatmulTilingSwat tilingEngine;
    // 禁用尾块优化（与量化 ST 一致）：mBaseTailSplitCnt 保持 1，
    // 尾块单块落在 headTileSize 边界，避免跨 tile 块导致 comm 窗口读串
    tilingEngine.SetOptimizeEnable(false);
    // 非量化 tiling 引擎按数据类型字节宽度推导 baseK（fp16/bf16 = 2 字节）
    tilingEngine.GetTilingData(m, n, ka, sizeof(uint16_t), false, true, tilingData.tileMmTilingData, true);

    // headMSize：命令行显式指定（>0）则直接采用（含钳制不超过 m）；未指定（0）则从 tiling 推导
    uint32_t headMSize = static_cast<uint32_t>(headMSizeArg);
    if (headMSize == 0) {
        auto nTile = CeilDiv(n, tilingData.tileMmTilingData.baseN);
        headMSize = CeilDiv(tilingData.tileMmTilingData.usedCoreNum, nTile) * tilingData.tileMmTilingData.baseM;
    }
    if (headMSize > static_cast<uint32_t>(m)) {
        headMSize = static_cast<uint32_t>(m);
    }
    uint32_t headTileCnt = static_cast<uint32_t>(m) / headMSize;
    uint32_t tailMSize = static_cast<uint32_t>(m) % headMSize;
    uint32_t tailTileCnt = (tailMSize > 0) ? 1 : 0;

    tilingData.commTilingData.splitAxisTileCnt = static_cast<uint64_t>(headTileCnt);
    tilingData.commTilingData.splitAxisTileSize = static_cast<uint64_t>(headMSize);
    tilingData.commTilingData.splitAxisTailSize = static_cast<uint64_t>(tailMSize);
    tilingData.commTilingData.splitAxisTailCnt = static_cast<uint64_t>(tailTileCnt);
    tilingData.commTilingData.nonSplitAxisSize = ka;
    tilingData.commTilingData.slotNum = static_cast<uint64_t>(bufferCountArg); // win buffer 复用槽位数
    tilingData.localMatmul = static_cast<uint32_t>(localMatmul);

    headTileCntOut = headTileCnt + tailTileCnt;
    return headMSize;
}

// ---------------- 量化路径：填充 tileQbmmTilingData + scaleCommTilingData ----------------
uint32_t fillQuantTiling(AllToAllQuantMatmulTilingData& tilingData, int m, int k, int n, int rankNum, int headMSizeArg,
                         int bufferCountArg, uint32_t& headTileCntOut)
{
    uint32_t ka = k / rankNum;
    QuantMatmulTilingSwat<mm::DataType::DT_FLOAT8_E4M3FN, mm::DataType::DT_FLOAT8_E4M3FN, mm::BiasDataType::DT_FLOAT>
        tilingEngine;
    tilingEngine.SetOptimizeEnable(false);
    tilingEngine.GetTilingData(m, n, ka, false, true, tilingData.tileQbmmTilingData);

    auto nTile = CeilDiv(n, tilingData.tileQbmmTilingData.baseN);
    uint32_t headMSize = static_cast<uint32_t>(headMSizeArg);
    if (headMSize == 0) {
        headMSize = CeilDiv(tilingData.tileQbmmTilingData.usedCoreNum, nTile) * tilingData.tileQbmmTilingData.baseM;
    }
    if (headMSize > static_cast<uint32_t>(m)) {
        headMSize = static_cast<uint32_t>(m);
    }
    uint32_t headTileCnt = static_cast<uint32_t>(m) / headMSize;
    uint32_t tailMSize = static_cast<uint32_t>(m) % headMSize;
    uint32_t tailTileCnt = (tailMSize > 0) ? 1 : 0;

    tilingData.commTilingData.splitAxisTileCnt = static_cast<uint64_t>(headTileCnt);
    tilingData.commTilingData.splitAxisTileSize = static_cast<uint64_t>(headMSize);
    tilingData.commTilingData.splitAxisTailSize = static_cast<uint64_t>(tailMSize);
    tilingData.commTilingData.splitAxisTailCnt = static_cast<uint64_t>(tailTileCnt);
    tilingData.commTilingData.nonSplitAxisSize = ka;
    tilingData.commTilingData.slotNum = static_cast<uint64_t>(bufferCountArg);
    tilingData.localMatmul = 0;

    tilingData.scaleCommTilingData = tilingData.commTilingData;
    tilingData.scaleCommTilingData.nonSplitAxisSize = ka / 32;
    tilingData.scaleCommTilingData.slotNum = 0;

    headTileCntOut = headTileCnt + tailTileCnt;
    return headMSize;
}

int runAllToAllMatmul(int rankNum, int rankId, int m, int k, int n, const std::string& mode, int headMSizeArg,
                      int bufferCountArg, const std::string& type, const std::string& dtype, int localMatmul)
{
    const char* ipport = "tcp://127.0.0.1:8998";
    INFO_LOG("rankNum=%d, rankId=%d, ipport=%s, mode=%s, headMSize=%d, bufferCount=%d, type=%s, dtype=%s, "
             "localMatmul=%d",
             rankNum, rankId, ipport, mode.c_str(), headMSizeArg, bufferCountArg, type.c_str(), dtype.c_str(),
             localMatmul);

    uint32_t ka = k / rankNum;

    uint32_t headMSize = 0;
    uint32_t tileCnt = 0;
    uint32_t coreNum = 0;
    if (type == "quant") {
        AllToAllQuantMatmulTilingData tilingData;
        headMSize = fillQuantTiling(tilingData, m, k, n, rankNum, headMSizeArg, bufferCountArg, tileCnt);
        coreNum = tilingData.tileQbmmTilingData.usedCoreNum;
        INFO_LOG("TileCnt=%u, HeadMSize=%u, bufferCount=%u (quant, E4M3E4M3)", tileCnt, headMSize, bufferCountArg);
        INFO_LOG("tiling: usedCoreNum=%u baseM=%u baseN=%u baseK=%u dbL0c=%u",
                 tilingData.tileQbmmTilingData.usedCoreNum, tilingData.tileQbmmTilingData.baseM,
                 tilingData.tileQbmmTilingData.baseN, tilingData.tileQbmmTilingData.baseK,
                 tilingData.tileQbmmTilingData.dbL0c);

        ACL_CHECK(aclInit(nullptr));
        int32_t deviceId = rankId;
        ACL_CHECK(aclrtSetDevice(deviceId));
        aclrtStream stream = nullptr;
        ACL_CHECK(aclrtCreateStream(&stream));

        RootInfoExchanger exchanger(static_cast<uint32_t>(rankId), static_cast<uint32_t>(rankNum), ipport);
        HcclRootInfo rootInfo;
        if (!exchanger.Exchange(rootInfo)) {
            ERROR_LOG("rank %d exchange rootInfo failed", rankId);
            return -1;
        }

        HcclCommConfig config;
        HcclCommConfigInit(&config);
        config.hcclWorldRankID = static_cast<uint32_t>(rankId);
        HcclComm comm = nullptr;
        HCCL_CHECK(HcclCommInitRootInfoConfig(static_cast<uint32_t>(rankNum), &rootInfo, static_cast<uint32_t>(rankId),
                                              &config, &comm));
        if (comm == nullptr) {
            ERROR_LOG("rank %d HcclCommInitRootInfoConfig failed", rankId);
            return -1;
        }

        CommChannelBuilder builder(comm);
        CommContext hostCtx = {};
        const char* ctxTag = "all_to_all_quant_matmul";

        CommContext* devContext = reinterpret_cast<CommContext*>(
            builder.CreateDeviceContext(&hostCtx, sizeof(CommContext), ctxTag, &hostCtx.udmaCtx, &hostCtx.ubmemCtx));
        if (devContext == nullptr) {
            ERROR_LOG("rank %d CreateDeviceContext failed", rankId);
            return -1;
        }

        exchanger.Barrier();
        exchanger.Close();

        std::vector<uint8_t> hostA(m * k, 0);
        std::vector<uint8_t> hostB(k * n, 0);
        std::vector<uint8_t> hostScaleA(m * CeilDiv(k, 64) * 2, 0);
        std::vector<uint8_t> hostScaleB(CeilDiv(k, 64) * 2 * n, 0);
        std::vector<uint16_t> hostOutput(m * n, 0);

        auto sizeA = static_cast<size_t>(1) * hostA.size() * sizeof(uint8_t);
        auto sizeB = static_cast<size_t>(1) * hostB.size() * sizeof(uint8_t);
        auto sizeScaleA = static_cast<size_t>(1) * hostScaleA.size() * sizeof(uint8_t);
        auto sizeScaleB = static_cast<size_t>(1) * hostScaleB.size() * sizeof(uint8_t);
        auto sizeOutput = static_cast<size_t>(1) * hostOutput.size() * sizeof(uint16_t);

        char exePath[PATH_MAX];
        ssize_t len = readlink("/proc/self/exe", exePath, sizeof(exePath) - 1);
        std::string baseDir = ".";
        if (len > 0) {
            exePath[len] = '\0';
            baseDir = exePath;
            size_t lastSlash = baseDir.find_last_of('/');
            if (lastSlash != std::string::npos) {
                baseDir.resize(lastSlash);
            }
        }
        std::string inputDir = baseDir + "/input/" + std::to_string(rankId);
        std::string outputDir = baseDir + "/output/" + std::to_string(rankId);
        ReadFile(inputDir + "/input_a.bin", hostA.data(), sizeA);
        ReadFile(inputDir + "/input_b.bin", hostB.data(), sizeB);
        ReadFile(inputDir + "/input_scaleA.bin", hostScaleA.data(), sizeScaleA);
        ReadFile(inputDir + "/input_scaleB.bin", hostScaleB.data(), sizeScaleB);

        GM_ADDR deviceA = nullptr;
        GM_ADDR deviceB = nullptr;
        GM_ADDR deviceScaleA = nullptr;
        GM_ADDR deviceScaleB = nullptr;
        GM_ADDR deviceOutput = nullptr;

        ACL_CHECK(aclrtMalloc((void**)&deviceA, sizeA, ACL_MEM_MALLOC_HUGE_ONLY));
        ACL_CHECK(aclrtMalloc((void**)&deviceB, sizeB, ACL_MEM_MALLOC_HUGE_ONLY));
        ACL_CHECK(aclrtMalloc((void**)&deviceScaleA, sizeScaleA, ACL_MEM_MALLOC_HUGE_ONLY));
        ACL_CHECK(aclrtMalloc((void**)&deviceScaleB, sizeScaleB, ACL_MEM_MALLOC_HUGE_ONLY));
        ACL_CHECK(aclrtMalloc((void**)&deviceOutput, sizeOutput, ACL_MEM_MALLOC_HUGE_ONLY));

        ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));
        ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));
        ACL_CHECK(aclrtMemcpy(deviceScaleA, sizeScaleA, hostScaleA.data(), sizeScaleA, ACL_MEMCPY_HOST_TO_DEVICE));
        ACL_CHECK(aclrtMemcpy(deviceScaleB, sizeScaleB, hostScaleB.data(), sizeScaleB, ACL_MEMCPY_HOST_TO_DEVICE));

        auto LaunchKernel = [&]() {
            AllToAllQuantMatmulKernelE4M3E4M3_Udma<<<coreNum, nullptr, stream>>>(
                devContext, deviceA, deviceScaleA, deviceB, deviceScaleB, deviceOutput, tilingData);
        };

        if (mode == "precision") {
            LaunchKernel();

            ACL_CHECK(aclrtSynchronizeStream(stream));

            ACL_CHECK(aclrtMemcpy(hostOutput.data(), sizeOutput, deviceOutput, sizeOutput, ACL_MEMCPY_DEVICE_TO_HOST));

            WriteFile(outputDir + "/npu_out.bin", hostOutput.data(), sizeOutput);
        } else {
            constexpr int64_t HEAVY_BLOCK_NUM = 56;
            constexpr int64_t BOOST_ELEM_COUNT = 10000LL * 10000LL;
            size_t boostSize = static_cast<size_t>(BOOST_ELEM_COUNT) * sizeof(uint16_t);
            GM_ADDR boostIn = nullptr;
            GM_ADDR boostOut = nullptr;
            ACL_CHECK(aclrtMalloc((void**)&boostIn, boostSize, ACL_MEM_MALLOC_HUGE_ONLY));
            ACL_CHECK(aclrtMalloc((void**)&boostOut, boostSize, ACL_MEM_MALLOC_HUGE_ONLY));
            std::vector<uint16_t> boostHost(BOOST_ELEM_COUNT, 0x3F00);
            ACL_CHECK(aclrtMemcpy(boostIn, boostSize, boostHost.data(), boostSize, ACL_MEMCPY_HOST_TO_DEVICE));

            constexpr int64_t CACHE_FLUSH_ELEM_COUNT = 128LL * 1024 * 1024;
            size_t cacheFlushSize = static_cast<size_t>(CACHE_FLUSH_ELEM_COUNT) * sizeof(uint16_t);
            GM_ADDR cacheFlush = nullptr;
            ACL_CHECK(aclrtMalloc((void**)&cacheFlush, cacheFlushSize, ACL_MEM_MALLOC_HUGE_ONLY));
            std::vector<uint16_t> cacheFlushHost(CACHE_FLUSH_ELEM_COUNT, 0x0000);
            ACL_CHECK(aclrtMemcpy(cacheFlush, cacheFlushSize, cacheFlushHost.data(), cacheFlushSize,
                                  ACL_MEMCPY_HOST_TO_DEVICE));

            constexpr int PERF_LOOP_COUNT = 10;
            std::vector<double> durations(PERF_LOOP_COUNT);
            for (int i = 0; i < PERF_LOOP_COUNT; ++i) {
                auto t0 = std::chrono::steady_clock::now();
                LaunchKernel();
                auto t1 = std::chrono::steady_clock::now();
                durations[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
            }

            double sum = 0, minv = 1e9, maxv = 0;
            for (int i = 1; i < PERF_LOOP_COUNT; ++i) {
                sum += durations[i];
                minv = std::min(minv, durations[i]);
                maxv = std::max(maxv, durations[i]);
            }
            double avg = sum / (PERF_LOOP_COUNT - 1);
            INFO_LOG("[Rank %d] headMSize=%u tileCnt=%u | per-iter ms: avg=%.3f min=%.3f max=%.3f (excl warmup)",
                     rankId, headMSize, tileCnt, avg, minv, maxv);

            ACL_CHECK(aclrtSynchronizeStream(stream));
            ACL_CHECK(aclrtMemcpy(hostOutput.data(), sizeOutput, deviceOutput, sizeOutput, ACL_MEMCPY_DEVICE_TO_HOST));
            WriteFile(outputDir + "/npu_out.bin", hostOutput.data(), sizeOutput);

            aclrtFree(boostIn);
            aclrtFree(boostOut);
            aclrtFree(cacheFlush);
        }

        aclrtFree(deviceA);
        aclrtFree(deviceScaleA);
        aclrtFree(deviceB);
        aclrtFree(deviceScaleB);
        aclrtFree(deviceOutput);

        HcclCommDestroy(comm);

        ACL_CHECK(aclrtDestroyStream(stream));
        ACL_CHECK(aclrtResetDevice(deviceId));
        ACL_CHECK(aclFinalize());

        return 0;
    }

    // ---- 非量化 ----
    AllToAllMatmulTilingData tilingData;
    headMSize = fillNonQuantTiling(tilingData, m, k, n, rankNum, headMSizeArg, bufferCountArg, localMatmul, tileCnt);
    coreNum = tilingData.tileMmTilingData.usedCoreNum;
    INFO_LOG("TileCnt=%u, HeadMSize=%u, bufferCount=%u, dtype=%s, localMatmul=%d", tileCnt, headMSize, bufferCountArg,
             dtype.c_str(), localMatmul);
    INFO_LOG("tiling: usedCoreNum=%u baseM=%u baseN=%u baseK=%u l0cDB=%u", tilingData.tileMmTilingData.usedCoreNum,
             tilingData.tileMmTilingData.baseM, tilingData.tileMmTilingData.baseN, tilingData.tileMmTilingData.baseK,
             tilingData.tileMmTilingData.l0cDB);

    ACL_CHECK(aclInit(nullptr));
    int32_t deviceId = rankId;
    ACL_CHECK(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));

    RootInfoExchanger exchanger(static_cast<uint32_t>(rankId), static_cast<uint32_t>(rankNum), ipport);
    HcclRootInfo rootInfo;
    if (!exchanger.Exchange(rootInfo)) {
        ERROR_LOG("rank %d exchange rootInfo failed", rankId);
        return -1;
    }

    HcclCommConfig config;
    HcclCommConfigInit(&config);
    config.hcclWorldRankID = static_cast<uint32_t>(rankId);
    HcclComm comm = nullptr;
    HCCL_CHECK(HcclCommInitRootInfoConfig(static_cast<uint32_t>(rankNum), &rootInfo, static_cast<uint32_t>(rankId),
                                          &config, &comm));
    if (comm == nullptr) {
        ERROR_LOG("rank %d HcclCommInitRootInfoConfig failed", rankId);
        return -1;
    }

    CommChannelBuilder builder(comm);
    CommContext hostCtx = {};
    const char* ctxTag = "all_to_all_matmul";

    CommContext* devContext = reinterpret_cast<CommContext*>(
        builder.CreateDeviceContext(&hostCtx, sizeof(CommContext), ctxTag, &hostCtx.udmaCtx, &hostCtx.ubmemCtx));
    if (devContext == nullptr) {
        ERROR_LOG("rank %d CreateDeviceContext failed", rankId);
        return -1;
    }

    exchanger.Barrier();
    exchanger.Close();

    std::vector<uint16_t> hostA(m * k, 0);
    std::vector<uint16_t> hostB(n * k, 0);
    std::vector<uint16_t> hostBias(n, 0);
    std::vector<uint16_t> hostOutput(m * n, 0);

    auto sizeA = static_cast<size_t>(1) * hostA.size() * sizeof(uint16_t);
    auto sizeB = static_cast<size_t>(1) * hostB.size() * sizeof(uint16_t);
    auto sizeBias = static_cast<size_t>(1) * hostBias.size() * sizeof(uint16_t);
    auto sizeOutput = static_cast<size_t>(1) * hostOutput.size() * sizeof(uint16_t);

    char exePath[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", exePath, sizeof(exePath) - 1);
    std::string baseDir = ".";
    if (len > 0) {
        exePath[len] = '\0';
        baseDir = exePath;
        size_t lastSlash = baseDir.find_last_of('/');
        if (lastSlash != std::string::npos) {
            baseDir.resize(lastSlash);
        }
    }
    std::string inputDir = baseDir + "/input/" + std::to_string(rankId);
    std::string outputDir = baseDir + "/output/" + std::to_string(rankId);
    if (!ReadFile(inputDir + "/input_a.bin", hostA.data(), sizeA)) {
        ERROR_LOG("rank %d failed to read input_a.bin", rankId);
        return -1;
    }
    if (!ReadFile(inputDir + "/input_b.bin", hostB.data(), sizeB)) {
        ERROR_LOG("rank %d failed to read input_b.bin", rankId);
        return -1;
    }
    if (!ReadFile(inputDir + "/input_bias.bin", hostBias.data(), sizeBias)) {
        ERROR_LOG("rank %d failed to read input_bias.bin", rankId);
        return -1;
    }

    GM_ADDR deviceA = nullptr;
    GM_ADDR deviceB = nullptr;
    GM_ADDR deviceBias = nullptr;
    GM_ADDR deviceOutput = nullptr;

    ACL_CHECK(aclrtMalloc((void**)&deviceA, sizeA, ACL_MEM_MALLOC_HUGE_ONLY));
    ACL_CHECK(aclrtMalloc((void**)&deviceB, sizeB, ACL_MEM_MALLOC_HUGE_ONLY));
    ACL_CHECK(aclrtMalloc((void**)&deviceBias, sizeBias, ACL_MEM_MALLOC_HUGE_ONLY));
    ACL_CHECK(aclrtMalloc((void**)&deviceOutput, sizeOutput, ACL_MEM_MALLOC_HUGE_ONLY));

    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceBias, sizeBias, hostBias.data(), sizeBias, ACL_MEMCPY_HOST_TO_DEVICE));

    // 按 dtype 选择 fp16/bf16 内核（两者数据宽度均为 2 字节，host 侧 buffer/tiling 一致）
    auto LaunchKernel = [&]() {
        if (dtype == "bf16") {
            AllToAllMmKernel_Bf16_Udma<<<coreNum, nullptr, stream>>>(devContext, deviceA, deviceB, deviceOutput,
                                                                     deviceBias, tilingData);
        } else {
            AllToAllMmKernel_Udma<<<coreNum, nullptr, stream>>>(devContext, deviceA, deviceB, deviceOutput, deviceBias,
                                                                tilingData);
        }
    };

    if (mode == "precision") {
        LaunchKernel();

        ACL_CHECK(aclrtSynchronizeStream(stream));

        ACL_CHECK(aclrtMemcpy(hostOutput.data(), sizeOutput, deviceOutput, sizeOutput, ACL_MEMCPY_DEVICE_TO_HOST));

        WriteFile(outputDir + "/npu_out.bin", hostOutput.data(), sizeOutput);
    } else {
        constexpr int64_t BOOST_ELEM_COUNT = 10000LL * 10000LL;
        size_t boostSize = static_cast<size_t>(BOOST_ELEM_COUNT) * sizeof(uint16_t);
        GM_ADDR boostIn = nullptr;
        GM_ADDR boostOut = nullptr;
        ACL_CHECK(aclrtMalloc((void**)&boostIn, boostSize, ACL_MEM_MALLOC_HUGE_ONLY));
        ACL_CHECK(aclrtMalloc((void**)&boostOut, boostSize, ACL_MEM_MALLOC_HUGE_ONLY));
        std::vector<uint16_t> boostHost(BOOST_ELEM_COUNT, 0x3F00);
        ACL_CHECK(aclrtMemcpy(boostIn, boostSize, boostHost.data(), boostSize, ACL_MEMCPY_HOST_TO_DEVICE));

        constexpr int64_t CACHE_FLUSH_ELEM_COUNT = 128LL * 1024 * 1024;
        size_t cacheFlushSize = static_cast<size_t>(CACHE_FLUSH_ELEM_COUNT) * sizeof(uint16_t);
        GM_ADDR cacheFlush = nullptr;
        ACL_CHECK(aclrtMalloc((void**)&cacheFlush, cacheFlushSize, ACL_MEM_MALLOC_HUGE_ONLY));
        std::vector<uint16_t> cacheFlushHost(CACHE_FLUSH_ELEM_COUNT, 0x0000);
        ACL_CHECK(
            aclrtMemcpy(cacheFlush, cacheFlushSize, cacheFlushHost.data(), cacheFlushSize, ACL_MEMCPY_HOST_TO_DEVICE));

        constexpr int PERF_LOOP_COUNT = 10;
        std::vector<double> durations(PERF_LOOP_COUNT);
        for (int i = 0; i < PERF_LOOP_COUNT; ++i) {
            auto t0 = std::chrono::steady_clock::now();
            LaunchKernel();
            auto t1 = std::chrono::steady_clock::now();
            durations[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
        }

        double sum = 0, minv = 1e9, maxv = 0;
        for (int i = 1; i < PERF_LOOP_COUNT; ++i) {
            sum += durations[i];
            minv = std::min(minv, durations[i]);
            maxv = std::max(maxv, durations[i]);
        }
        double avg = sum / (PERF_LOOP_COUNT - 1);
        INFO_LOG("[Rank %d] localMatmul=%d headMSize=%u tileCnt=%u | per-iter ms: avg=%.3f min=%.3f max=%.3f "
                 "(excl warmup)",
                 rankId, localMatmul, headMSize, tileCnt, avg, minv, maxv);

        ACL_CHECK(aclrtSynchronizeStream(stream));
        ACL_CHECK(aclrtMemcpy(hostOutput.data(), sizeOutput, deviceOutput, sizeOutput, ACL_MEMCPY_DEVICE_TO_HOST));
        WriteFile(outputDir + "/npu_out.bin", hostOutput.data(), sizeOutput);

        aclrtFree(boostIn);
        aclrtFree(boostOut);
        aclrtFree(cacheFlush);
    }

    aclrtFree(deviceA);
    aclrtFree(deviceB);
    aclrtFree(deviceBias);
    aclrtFree(deviceOutput);

    HcclCommDestroy(comm);

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(deviceId));
    ACL_CHECK(aclFinalize());

    return 0;
}

int main(int argc, char* argv[])
{
    int m, k, n, rankNum, headMSize, bufferCount, localMatmul;
    std::string mode, type, dtype;
    try {
        parseArguments(argc, argv, m, k, n, rankNum, mode, headMSize, bufferCount, type, dtype, localMatmul);
    } catch (const std::invalid_argument& e) {
        std::cerr << e.what() << std::endl;
        printUsage(argv[0]);
        return -1;
    }

    INFO_LOG("Master (PID=%d) will fork %d processes (mode=%s, headMSize=%d, bufferCount=%d, type=%s, dtype=%s, "
             "localMatmul=%d)",
             getpid(), rankNum, mode.c_str(), headMSize, bufferCount, type.c_str(), dtype.c_str(), localMatmul);

    std::vector<pid_t> pids(rankNum);
    for (int rankId = 0; rankId < rankNum; ++rankId) {
        pid_t pid = fork();

        if (pid < 0) {
            ERROR_LOG("Fork failed for rank %d", rankId);
            exit(-1);
        } else if (pid == 0) {
            int ret =
                runAllToAllMatmul(rankNum, rankId, m, k, n, mode, headMSize, bufferCount, type, dtype, localMatmul);
            exit(ret);
        } else {
            pids[rankId] = pid;
            INFO_LOG("Forked Rank %d -> PID %d", rankId, pid);
        }
    }

    int status;
    bool all_success = true;
    for (int rankId = 0; rankId < rankNum; ++rankId) {
        pid_t pid = pids[rankId];
        waitpid(pid, &status, 0);
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            all_success = false;
            ERROR_LOG("Worker PID %d failed", pid);
        }
    }

    std::cout << "All workers finished. Status: " << (all_success ? "SUCCESS" : "FAILURE") << std::endl;
    return all_success ? 0 : -1;
}
