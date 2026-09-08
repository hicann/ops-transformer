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
 * \file main.cpp
 * \brief ST host entry for QuantMatmul + AllToAll fusion (URMA variant)
 *
 * Operator semantics: MXFP MatMul + AllToAll.
 *   - Each rank holds A_i [M, K] and the full B_i [K, N]
 *   - Each rank computes C_i = dequant(A_i) x dequant(B_i) (+ bias) -> [M, N]
 *   - AllToAll scatters N-slices of C_i: rank j receives C_i[:, j*Np:(j+1)*Np]
 *     from every rank i, concatenated along M -> [rankNum*M, Np] (BF16/FP16)
 */

#include <cstring>
#include <iostream>
#include <iomanip>
#include <limits.h>
#include <unistd.h>
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
#include "apace/utils/apace_constant.h"
#include "apace/tiling/quant_matmul_tiling_swat.h"
#include "apace/kernel/fusions/quant_matmul_all_to_all/matmul_all_to_all_tiling_data.h"
#include "kernel_launcher.h"
#include "apace/utils/comm_channel_builder.h"
#include "apace/core/aiv_comm/collective_comm_context.h"
#include "../../utils/root_info_exchanger.h"

static constexpr int32_t BENCHMARK_ITERATIONS = 10;
static constexpr int32_t MXFP_K_ALIGN_SIZE = 32;
static constexpr uint16_t HCCL_ROOT_INFO_PORT = 8998;
static constexpr double MS_TO_US = 1000.0;

enum QuantType {
    QUANT_E2M1_E2M1 = 0,
    QUANT_E4M3_E4M3 = 1,
    QUANT_E5M2_E5M2 = 2,
    QUANT_E4M3_E5M2 = 3,
    QUANT_E5M2_E4M3 = 4,
};

struct AllToAllArgs {
    int m{0};
    int k{0};
    int n{0};
    int rankNum{0};
    int quantType{QUANT_E4M3_E4M3};
    int isFp16{0};
    int isBias{0};
    int tileCnt{1};
    int tileSize{0};
    int tailSize{0};
    int isLocalDelayed{1};
};

void PrintAllToAllUsage(const std::string &programName)
{
    std::cerr << "Usage: " << programName
              << " m k n rankNum [quantType] [isFp16] [isBias] [tileCnt] [tileSize]"
                 " [tailSize] [isLocalDelayed]"
              << std::endl;
    std::cerr << "Args: " << std::endl;
    std::cerr << "  m: row of matrix A" << std::endl;
    std::cerr << "  k: col of matrix A (= row of matrix B, full K, not split)" << std::endl;
    std::cerr << "  n: col of matrix B (total N, split to N/rankNum per rank)" << std::endl;
    std::cerr << "  rankNum: number of ranks" << std::endl;
    std::cerr << "  quantType: 0=E2M1E2M1(FP4), 1=E4M3E4M3, 2=E5M2E5M2, 3=E4M3E5M2, 4=E5M2E4M3 (default: 1)"
              << std::endl;
    std::cerr << "  isFp16: output float16 instead of bfloat16 (0/1, default: 0)" << std::endl;
    std::cerr << "  isBias: enable bias input (0/1, default: 0)" << std::endl;
    std::cerr << "  tileCnt: number of regular tiles (default: 1)" << std::endl;
    std::cerr << "  tileSize: M rows per tile (default: m / tileCnt)" << std::endl;
    std::cerr << "  tailSize: M rows per tail tile, 0=no tail (default: 0)" << std::endl;
    std::cerr << "  isLocalDelayed: defer self batch to end (0/1, default: 1)" << std::endl;
    std::cerr << "Example: " << programName << " 2048 3584 4096 4 1 0 0 4 512 0 1" << std::endl;
}

void ParseArgs(int argc, char *argv[], AllToAllArgs &args)
{
    if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        PrintAllToAllUsage(argv[0]);
        exit(1);
    }
    if (argc < 5) {
        throw std::invalid_argument("ERROR: Lacks Arguments");
    }
    try {
        args.m = std::stoi(argv[1]);
        args.k = std::stoi(argv[2]);
        args.n = std::stoi(argv[3]);
        args.rankNum = std::stoi(argv[4]);
        args.quantType = (argc >= 6) ? std::stoi(argv[5]) : QUANT_E4M3_E4M3;
        args.isFp16 = (argc >= 7) ? std::stoi(argv[6]) : 0;
        args.isBias = (argc >= 8) ? std::stoi(argv[7]) : 0;
        args.tileCnt = (argc >= 9) ? std::stoi(argv[8]) : 1;
        args.tileSize = (argc >= 10) ? std::stoi(argv[9]) : 0;
        args.tailSize = (argc >= 11) ? std::stoi(argv[10]) : 0;
        args.isLocalDelayed = (argc >= 12) ? std::stoi(argv[11]) : 1;
    } catch (const std::invalid_argument &) {
        throw std::invalid_argument(
            "ERROR: m k n rankNum quantType isFp16 isBias tileCnt tileSize tailSize isLocalDelayed must "
            "be Integer");
    }

    if (args.m <= 0 || args.k <= 0 || args.n <= 0 || args.rankNum <= 0 || args.tileCnt <= 0) {
        throw std::invalid_argument("ERROR: m k n rankNum tileCnt must be positive");
    }
    if (args.k % MXFP_K_ALIGN_SIZE != 0) {
        throw std::invalid_argument("ERROR: K must be divisible by 32 for MXFP quantization");
    }
    if (CeilDiv(args.k, static_cast<int>(::MXFP_DIVISOR_SIZE)) % ::MXFP_MULTI_BASE_SIZE != 0) {
        throw std::invalid_argument("ERROR: CeilDiv(K, 64) must be an even number");
    }
    if (args.n % args.rankNum != 0) {
        throw std::invalid_argument("ERROR: n must be divisible by rankNum for AllToAll");
    }
    if (args.quantType < QUANT_E2M1_E2M1 || args.quantType > QUANT_E5M2_E4M3) {
        throw std::invalid_argument("ERROR: quantType must be 0-4");
    }
    if (args.isFp16 != 0 && args.isFp16 != 1) {
        throw std::invalid_argument("ERROR: isFp16 must be 0 or 1");
    }
    if (args.isBias != 0 && args.isBias != 1) {
        throw std::invalid_argument("ERROR: isBias must be 0 or 1");
    }
    if (args.isLocalDelayed != 0 && args.isLocalDelayed != 1) {
        throw std::invalid_argument("ERROR: isLocalDelayed must be 0 or 1");
    }
    if (args.rankNum == 1 && args.isLocalDelayed != 0) {
        throw std::invalid_argument("ERROR: isLocalDelayed=1 requires rankNum >= 2");
    }

    if (args.tileSize <= 0) {
        args.tileSize = args.m / args.tileCnt;
    }
    if (args.tileSize <= 0) {
        throw std::invalid_argument("ERROR: tileSize must be positive, check m >= tileCnt");
    }
    int64_t rem =
        static_cast<int64_t>(args.m) - static_cast<int64_t>(args.tileSize) * static_cast<int64_t>(args.tileCnt);
    if (rem < 0) {
        throw std::invalid_argument("ERROR: tileSize * tileCnt exceeds m");
    }
    if (args.tailSize > 0) {
        if (rem % args.tailSize != 0) {
            throw std::invalid_argument("ERROR: remainder not divisible by tailSize");
        }
    } else if (rem != 0) {
        throw std::invalid_argument("ERROR: m must be divisible by tileSize * tileCnt when tailSize=0 or set "
                                    "tailSize>0");
    }
}

static void LaunchKernel(uint32_t usedCoreNum, aclrtStream stream, CommContext *hcommCtx, GM_ADDR deviceA,
                         GM_ADDR deviceScaleA, GM_ADDR deviceB, GM_ADDR deviceScaleB, GM_ADDR deviceBias,
                         GM_ADDR deviceOutput, const MatmulAllToAllTilingData &tilingData, int quantType, int isFp16,
                         int isLocalDelayed)
{
#define A2A_MX_LAUNCH(kernelName) \
    kernelName<<<usedCoreNum, nullptr, stream>>>(hcommCtx, deviceA, deviceScaleA, deviceB, deviceScaleB, deviceBias, \
                                                 deviceOutput, tilingData)
#define A2A_MX_LAUNCH_QUANT(typeSuffix) \
    do { \
        switch (quantType) { \
            case QUANT_E2M1_E2M1: \
                A2A_MX_LAUNCH(MatmulAllToAllMxKernelE2M1E2M1##typeSuffix); \
                break; \
            case QUANT_E4M3_E4M3: \
                A2A_MX_LAUNCH(MatmulAllToAllMxKernelE4M3E4M3##typeSuffix); \
                break; \
            case QUANT_E5M2_E5M2: \
                A2A_MX_LAUNCH(MatmulAllToAllMxKernelE5M2E5M2##typeSuffix); \
                break; \
            case QUANT_E4M3_E5M2: \
                A2A_MX_LAUNCH(MatmulAllToAllMxKernelE4M3E5M2##typeSuffix); \
                break; \
            case QUANT_E5M2_E4M3: \
                A2A_MX_LAUNCH(MatmulAllToAllMxKernelE5M2E4M3##typeSuffix); \
                break; \
            default: \
                break; \
        } \
    } while (0)
    if (isLocalDelayed != 0) {
        if (isFp16 != 0) {
            A2A_MX_LAUNCH_QUANT(Fp16Delay);
        } else {
            A2A_MX_LAUNCH_QUANT(Bf16Delay);
        }
    } else {
        if (isFp16 != 0) {
            A2A_MX_LAUNCH_QUANT(Fp16NoDelay);
        } else {
            A2A_MX_LAUNCH_QUANT(Bf16NoDelay);
        }
    }
#undef A2A_MX_LAUNCH_QUANT
#undef A2A_MX_LAUNCH
}

static void ReleaseStreamDevice(aclrtStream stream, int32_t deviceId)
{
    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int RunMatmulAllToAll(const AllToAllArgs &args, int rankId)
{
    std::string ipport = "tcp://127.0.0.1:" + std::to_string(HCCL_ROOT_INFO_PORT);
    int rankNum = args.rankNum;
    int m = args.m;
    int k = args.k;
    int n = args.n;

    INFO_LOG("rankNum=%d, rankId=%d, ipport=%s, quantType=%d, isFp16=%d, isBias=%d, tileCnt=%d, "
             "tileSize=%d, tailSize=%d, isLocalDelayed=%d",
             rankNum, rankId, ipport.c_str(), args.quantType, args.isFp16, args.isBias, args.tileCnt, args.tileSize,
             args.tailSize, args.isLocalDelayed);

    uint32_t np = static_cast<uint32_t>(n) / static_cast<uint32_t>(rankNum);

    MatmulAllToAllTilingData tilingData = {};
    uint32_t tileSize = static_cast<uint32_t>(args.tileSize);
    uint32_t tileCnt = static_cast<uint32_t>(args.tileCnt);
    uint32_t tailSize = static_cast<uint32_t>(args.tailSize);
    uint32_t rem = static_cast<uint32_t>(static_cast<uint64_t>(m) -
                                         static_cast<uint64_t>(tileSize) * static_cast<uint64_t>(tileCnt));
    uint32_t tailCnt = (tailSize > 0) ? rem / tailSize : 0;

    auto &commTd = tilingData.commTilingData;
    commTd.splitAxisTileSize = static_cast<uint64_t>(tileSize);
    commTd.splitAxisTileCnt = static_cast<uint64_t>(tileCnt);
    commTd.splitAxisTailSize = static_cast<uint64_t>(tailSize);
    commTd.splitAxisTailCnt = static_cast<uint64_t>(tailCnt);
    commTd.nonSplitAxisSize = static_cast<uint64_t>(np);

    INFO_LOG("CommTurnSet: tileCnt=%u tileM=%u tailM=%u tailCnt=%u", tileCnt, tileSize, tailSize, tailCnt);

    if (aclInit(nullptr) != ACL_ERROR_NONE) {
        ERROR_LOG("rank %d aclInit failed", rankId);
        return -1;
    }
    int32_t deviceId = rankId;
    if (aclrtSetDevice(deviceId) != ACL_ERROR_NONE) {
        ERROR_LOG("rank %d aclrtSetDevice failed", rankId);
        return -1;
    }
    aclrtStream stream = nullptr;
    if (aclrtCreateStream(&stream) != ACL_ERROR_NONE) {
        ERROR_LOG("rank %d aclrtCreateStream failed", rankId);
        ReleaseStreamDevice(stream, deviceId);
        return -1;
    }

    RootInfoExchanger exchanger(static_cast<uint32_t>(rankId), static_cast<uint32_t>(rankNum), ipport);
    HcclRootInfo rootInfo;
    if (!exchanger.Exchange(rootInfo)) {
        ERROR_LOG("rank %d exchange rootInfo failed", rankId);
        memset(&rootInfo, 0, sizeof(rootInfo));
        ReleaseStreamDevice(stream, deviceId);
        return -1;
    }

    HcclCommConfig config;
    HcclCommConfigInit(&config);
    config.hcclWorldRankID = static_cast<uint32_t>(rankId);
    HcclComm comm = nullptr;
    HcclResult hcclRet = HcclCommInitRootInfoConfig(static_cast<uint32_t>(rankNum), &rootInfo,
                                                    static_cast<uint32_t>(rankId), &config, &comm);
    if (hcclRet != HCCL_SUCCESS || comm == nullptr) {
        ERROR_LOG("rank %d HcclCommInitRootInfoConfig failed, HcclResult:%d", rankId, static_cast<int>(hcclRet));
        memset(&rootInfo, 0, sizeof(rootInfo));
        ReleaseStreamDevice(stream, deviceId);
        return -1;
    }
    memset(&rootInfo, 0, sizeof(rootInfo));

    CommChannelBuilder<> builder(comm);
    CommContext hostCtx = {};

    const char *ctxTag = "quant_matmul_all_to_all";

    CommContext *devContext = reinterpret_cast<CommContext *>(
        builder.CreateDeviceContext(&hostCtx, sizeof(CommContext), ctxTag, &hostCtx.udmaCtx, &hostCtx.ubmemCtx));
    if (devContext == nullptr) {
        ERROR_LOG("rank %d CreateDeviceContext failed", rankId);
        HcclCommDestroy(comm);
        ReleaseStreamDevice(stream, deviceId);
        return -1;
    }

    exchanger.Barrier();
    exchanger.Close();

    uint32_t remoteBatch =
        (args.isLocalDelayed != 0) ? static_cast<uint32_t>(rankNum) - 1 : static_cast<uint32_t>(rankNum);
    uint32_t mmTileM = static_cast<uint32_t>(commTd.splitAxisTileSize);
    INFO_LOG("isLocalDelayed = %d, quantType = %d, isBias = %d, isFp16 = %d", args.isLocalDelayed, args.quantType,
             args.isBias, args.isFp16);

#define DO_TILING(dTypeA, dTypeB) \
    do { \
        QuantMatmulTilingSwat<dTypeA, dTypeB> tilingEngine; \
        tilingEngine.GetTilingData(mmTileM, np, static_cast<uint64_t>(k), tilingData.tileQbmmTilingData, remoteBatch); \
        if (tailSize > 0) { \
            tilingEngine.GetTilingData(static_cast<uint64_t>(tailSize), np, static_cast<uint64_t>(k), \
                                       tilingData.tailQbmmTilingData, remoteBatch); \
        } \
        if (args.isLocalDelayed) { \
            tilingEngine.GetTilingData(static_cast<uint64_t>(m), np, static_cast<uint64_t>(k), \
                                       tilingData.localQbmmTilingData, 1); \
        } \
    } while (0)

    switch (args.quantType) {
        case QUANT_E2M1_E2M1:
            DO_TILING(mm::DataType::DT_FLOAT4_E2M1, mm::DataType::DT_FLOAT4_E2M1);
            break;
        case QUANT_E4M3_E4M3:
            DO_TILING(mm::DataType::DT_FLOAT8_E4M3FN, mm::DataType::DT_FLOAT8_E4M3FN);
            break;
        case QUANT_E5M2_E5M2:
            DO_TILING(mm::DataType::DT_FLOAT8_E5M2, mm::DataType::DT_FLOAT8_E5M2);
            break;
        case QUANT_E4M3_E5M2:
            DO_TILING(mm::DataType::DT_FLOAT8_E4M3FN, mm::DataType::DT_FLOAT8_E5M2);
            break;
        case QUANT_E5M2_E4M3:
            DO_TILING(mm::DataType::DT_FLOAT8_E5M2, mm::DataType::DT_FLOAT8_E4M3FN);
            break;
        default:
            break;
    }
#undef DO_TILING

    uint32_t usedCoreNum = tilingData.tileQbmmTilingData.usedCoreNum;
    if (usedCoreNum < static_cast<uint32_t>(rankNum)) {
        ERROR_LOG("rank %d usedCoreNum(%u) < rankNum(%d), TeamBarrier requires at least rankNum blocks", rankId,
                  usedCoreNum, rankNum);
        HcclCommDestroy(comm);
        ReleaseStreamDevice(stream, deviceId);
        return -1;
    }
    INFO_LOG("Tiling: usedCoreNum=%u", usedCoreNum);

    bool isFp4 = (args.quantType == QUANT_E2M1_E2M1);
    uint64_t sizeShift = isFp4 ? 1 : 0;
    auto sizeA = static_cast<size_t>((static_cast<uint64_t>(m) * k) >> sizeShift);
    auto sizeB = static_cast<size_t>((static_cast<uint64_t>(k) * n) >> sizeShift);
    uint64_t scaleKGroups = CeilDiv(static_cast<uint64_t>(k), static_cast<uint64_t>(::MXFP_DIVISOR_SIZE));
    auto sizeScaleA = static_cast<size_t>(m) * scaleKGroups * ::MXFP_MULTI_BASE_SIZE;
    auto sizeScaleB = static_cast<size_t>(scaleKGroups) * ::MXFP_MULTI_BASE_SIZE * n;
    auto sizeBias = (args.isBias != 0) ? static_cast<size_t>(n) * sizeof(float) : 0;
    auto sizeOutput = static_cast<size_t>(rankNum) * m * np * sizeof(uint16_t);

    std::vector<uint8_t> hostA(sizeA, 0);
    std::vector<uint8_t> hostB(sizeB, 0);
    std::vector<uint8_t> hostScaleA(sizeScaleA, 0);
    std::vector<uint8_t> hostScaleB(sizeScaleB, 0);
    std::vector<float> hostBias((args.isBias != 0) ? n : 0, 0.0F);
    std::vector<uint16_t> hostOutput(sizeOutput / sizeof(uint16_t), 0);

    GM_ADDR deviceA = nullptr;
    GM_ADDR deviceB = nullptr;
    GM_ADDR deviceScaleA = nullptr;
    GM_ADDR deviceScaleB = nullptr;
    GM_ADDR deviceBias = nullptr;
    GM_ADDR deviceOutput = nullptr;
    aclrtEvent kernelStartEvent = nullptr;
    aclrtEvent kernelEndEvent = nullptr;
    auto releaseAll = [&]() {
        if (kernelEndEvent != nullptr) {
            aclrtDestroyEvent(kernelEndEvent);
        }
        if (kernelStartEvent != nullptr) {
            aclrtDestroyEvent(kernelStartEvent);
        }
        if (deviceOutput != nullptr) {
            aclrtFree(deviceOutput);
        }
        if (deviceBias != nullptr) {
            aclrtFree(deviceBias);
        }
        if (deviceScaleB != nullptr) {
            aclrtFree(deviceScaleB);
        }
        if (deviceScaleA != nullptr) {
            aclrtFree(deviceScaleA);
        }
        if (deviceB != nullptr) {
            aclrtFree(deviceB);
        }
        if (deviceA != nullptr) {
            aclrtFree(deviceA);
        }
        HcclCommDestroy(comm);
        ReleaseStreamDevice(stream, deviceId);
    };

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
    auto readInput = [&](const std::string &fileName, void *buffer, size_t size) -> bool {
        std::string path = inputDir + "/" + fileName;
        if (!ReadFile(path, buffer, size)) {
            ERROR_LOG("rank %d read %s failed. path = %s", rankId, fileName.c_str(), path.c_str());
            return false;
        }
        return true;
    };
    if (!readInput("input_a.bin", hostA.data(), sizeA) || !readInput("input_b.bin", hostB.data(), sizeB) ||
        !readInput("input_scaleA.bin", hostScaleA.data(), sizeScaleA) ||
        !readInput("input_scaleB.bin", hostScaleB.data(), sizeScaleB) ||
        (args.isBias != 0 && !readInput("input_bias.bin", hostBias.data(), sizeBias))) {
        releaseAll();
        return -1;
    }

    if (aclrtMalloc((void **)&deviceA, sizeA, ACL_MEM_MALLOC_HUGE_ONLY) != ACL_ERROR_NONE) {
        ERROR_LOG("rank %d aclrtMalloc deviceA failed", rankId);
        releaseAll();
        return -1;
    }
    if (aclrtMalloc((void **)&deviceB, sizeB, ACL_MEM_MALLOC_HUGE_ONLY) != ACL_ERROR_NONE) {
        ERROR_LOG("rank %d aclrtMalloc deviceB failed", rankId);
        releaseAll();
        return -1;
    }
    if (aclrtMalloc((void **)&deviceScaleA, sizeScaleA, ACL_MEM_MALLOC_HUGE_ONLY) != ACL_ERROR_NONE) {
        ERROR_LOG("rank %d aclrtMalloc deviceScaleA failed", rankId);
        releaseAll();
        return -1;
    }
    if (aclrtMalloc((void **)&deviceScaleB, sizeScaleB, ACL_MEM_MALLOC_HUGE_ONLY) != ACL_ERROR_NONE) {
        ERROR_LOG("rank %d aclrtMalloc deviceScaleB failed", rankId);
        releaseAll();
        return -1;
    }
    if (args.isBias != 0) {
        if (aclrtMalloc((void **)&deviceBias, sizeBias, ACL_MEM_MALLOC_HUGE_ONLY) != ACL_ERROR_NONE) {
            ERROR_LOG("rank %d aclrtMalloc deviceBias failed", rankId);
            releaseAll();
            return -1;
        }
    }
    if (aclrtMalloc((void **)&deviceOutput, sizeOutput, ACL_MEM_MALLOC_HUGE_ONLY) != ACL_ERROR_NONE) {
        ERROR_LOG("rank %d aclrtMalloc deviceOutput failed", rankId);
        releaseAll();
        return -1;
    }

    if (aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_ERROR_NONE) {
        ERROR_LOG("rank %d aclrtMemcpy deviceA failed", rankId);
        releaseAll();
        return -1;
    }
    if (aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_ERROR_NONE) {
        ERROR_LOG("rank %d aclrtMemcpy deviceB failed", rankId);
        releaseAll();
        return -1;
    }
    if (aclrtMemcpy(deviceScaleA, sizeScaleA, hostScaleA.data(), sizeScaleA, ACL_MEMCPY_HOST_TO_DEVICE) !=
        ACL_ERROR_NONE) {
        ERROR_LOG("rank %d aclrtMemcpy deviceScaleA failed", rankId);
        releaseAll();
        return -1;
    }
    if (aclrtMemcpy(deviceScaleB, sizeScaleB, hostScaleB.data(), sizeScaleB, ACL_MEMCPY_HOST_TO_DEVICE) !=
        ACL_ERROR_NONE) {
        ERROR_LOG("rank %d aclrtMemcpy deviceScaleB failed", rankId);
        releaseAll();
        return -1;
    }
    if (args.isBias != 0) {
        if (aclrtMemcpy(deviceBias, sizeBias, hostBias.data(), sizeBias, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_ERROR_NONE) {
            ERROR_LOG("rank %d aclrtMemcpy deviceBias failed", rankId);
            releaseAll();
            return -1;
        }
    }

    if (aclrtCreateEvent(&kernelStartEvent) != ACL_ERROR_NONE) {
        ERROR_LOG("rank %d aclrtCreateEvent kernelStartEvent failed", rankId);
        releaseAll();
        return -1;
    }
    if (aclrtCreateEvent(&kernelEndEvent) != ACL_ERROR_NONE) {
        ERROR_LOG("rank %d aclrtCreateEvent kernelEndEvent failed", rankId);
        releaseAll();
        return -1;
    }
    if (aclrtRecordEvent(kernelStartEvent, stream) != ACL_ERROR_NONE) {
        ERROR_LOG("rank %d aclrtRecordEvent kernelStartEvent failed", rankId);
        releaseAll();
        return -1;
    }

    for (int i = 0; i < BENCHMARK_ITERATIONS; ++i) {
        LaunchKernel(usedCoreNum, stream, devContext, deviceA, deviceScaleA, deviceB, deviceScaleB, deviceBias,
                     deviceOutput, tilingData, args.quantType, args.isFp16, args.isLocalDelayed);
    }
    std::cout << "LaunchKernel finished" << std::endl;
    if (aclrtRecordEvent(kernelEndEvent, stream) != ACL_ERROR_NONE) {
        ERROR_LOG("rank %d aclrtRecordEvent kernelEndEvent failed", rankId);
        releaseAll();
        return -1;
    }
    if (aclrtSynchronizeStream(stream) != ACL_ERROR_NONE) {
        ERROR_LOG("rank %d aclrtSynchronizeStream failed", rankId);
        releaseAll();
        return -1;
    }

    float kernelElapsedMs = 0.0F;
    if (aclrtEventElapsedTime(&kernelElapsedMs, kernelStartEvent, kernelEndEvent) != ACL_ERROR_NONE) {
        ERROR_LOG("rank %d aclrtEventElapsedTime failed", rankId);
        releaseAll();
        return -1;
    }
    double kernelElapsedUs = static_cast<double>(kernelElapsedMs) * MS_TO_US;

    if (aclrtMemcpy(hostOutput.data(), sizeOutput, deviceOutput, sizeOutput, ACL_MEMCPY_DEVICE_TO_HOST) !=
        ACL_ERROR_NONE) {
        ERROR_LOG("rank %d aclrtMemcpy deviceOutput to host failed", rankId);
        releaseAll();
        return -1;
    }

    bool writeOk = WriteFile(outputDir + "/npu_out.bin", hostOutput.data(), sizeOutput);
    if (!writeOk) {
        ERROR_LOG("rank %d write npu_out.bin failed. path = %s", rankId, (outputDir + "/npu_out.bin").c_str());
    }

    std::cout << std::fixed << std::setprecision(3) << "[Rank " << rankId
              << "] Kernel elapsed time: " << kernelElapsedUs / BENCHMARK_ITERATIONS << " us (avg over "
              << BENCHMARK_ITERATIONS << " iterations)" << std::endl;

    if (kernelEndEvent != nullptr) {
        aclrtDestroyEvent(kernelEndEvent);
    }
    if (kernelStartEvent != nullptr) {
        aclrtDestroyEvent(kernelStartEvent);
    }

    aclrtFree(deviceA);
    aclrtFree(deviceScaleA);
    aclrtFree(deviceB);
    aclrtFree(deviceScaleB);
    if (deviceBias != nullptr) {
        aclrtFree(deviceBias);
    }
    aclrtFree(deviceOutput);

    HcclCommDestroy(comm);

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(deviceId));
    ACL_CHECK(aclFinalize());

    return writeOk ? 0 : -1;
}

int main(int argc, char *argv[])
{
    AllToAllArgs args;
    try {
        ParseArgs(argc, argv, args);
    } catch (const std::invalid_argument &e) {
        std::cerr << e.what() << std::endl;
        PrintAllToAllUsage(argv[0]);
        return -1;
    }

    INFO_LOG("Master (PID=%d) will fork %d processes (quantType=%d, isFp16=%d, isBias=%d, tileCnt=%d, "
             "tileSize=%d, tailSize=%d, isLocalDelayed=%d)",
             getpid(), args.rankNum, args.quantType, args.isFp16, args.isBias, args.tileCnt, args.tileSize,
             args.tailSize, args.isLocalDelayed);

    std::vector<pid_t> pids(args.rankNum);
    for (int rankId = 0; rankId < args.rankNum; ++rankId) {
        pid_t pid = fork();

        if (pid < 0) {
            ERROR_LOG("Fork failed for rank %d", rankId);
            exit(-1);
        } else if (pid == 0) {
            int ret = RunMatmulAllToAll(args, rankId);
            exit(ret);
        } else {
            pids[rankId] = pid;
            INFO_LOG("Forked Rank %d -> PID %d", rankId, pid);
        }
    }

    int status;
    bool allSuccess = true;
    for (int rankId = 0; rankId < args.rankNum; ++rankId) {
        pid_t pid = pids[rankId];
        waitpid(pid, &status, 0);
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            allSuccess = false;
            ERROR_LOG("Worker PID %d failed", pid);
        }
    }

    std::cout << "All workers finished. Status: " << (allSuccess ? "SUCCESS" : "FAILURE") << std::endl;
    return allSuccess ? 0 : -1;
}
