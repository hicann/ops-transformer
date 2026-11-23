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
 * \brief quantMatmulReduceScatter ST host 主程序
 *
 * 每 rank 计算 C_i=bf16(dequant(A_i) @ dequant(B)), 按 M 轴执行 AllToAll,
 * 目标 rank 对收到的部分和做 ReduceSum, 输出 [m/rankSize,n] BF16.
 * K 不切分, 通信数据为 BF16 C, 不是量化输入 A/B.
 * 流程: fork 多 rank → HCCL 建链 → Tiling → Kernel 执行 → host 拷回并落盘 → Python 比对
 */

#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits.h>
#include <stdexcept>
#include <type_traits>
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
#include "apace/utils/comm_channel_builder.h"
#include "apace/utils/apace_constant.h"
#include "../../../utils/root_info_exchanger.h"
#include "apace/tiling/matmul_tiling/quant_matmul_tiling.h"
#include "apace_mrs_urma_kernel_launcher.h"
#include "apace/core/aiv_comm/collective_comm_context.h"

#define HCCL_CHECK(status) \
    do { \
        HcclResult ret = (status); \
        if (ret != HCCL_SUCCESS) { \
            std::cerr << __FILE__ << ":" << __LINE__ << " HcclResult:" << ret << std::endl; \
        } \
    } while (0)

// ACL 失败传播 (bool 语义): 打印并 return false
// 失败走 return false, "return -1" 在 bool 函数里会转成 true
#define QMRS_ACL_RET(status) \
    do { \
        aclError aclRet = (status); \
        if (aclRet != ACL_SUCCESS) { \
            ERROR_LOG("%s:%d aclError:%d", __FILE__, __LINE__, static_cast<int>(aclRet)); \
            return false; \
        } \
    } while (0)

inline uint64_t CeilDiv(uint32_t a, uint32_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

constexpr int QMRS_DEFAULT_HEAD_M_SIZE = 512;

void PrintQmrsUsage(const std::string& programName)
{
    std::cerr << "Usage: " << programName << " m k n rankNum [mode] [headMSize] [quantType]" << std::endl;
    std::cerr << "Args: " << std::endl;
    std::cerr << "  m: row of matrix A (must be divisible by rankNum)" << std::endl;
    std::cerr << "  k: col of matrix A (full K, NOT split across ranks; CeilDiv(k, 64) must be even)" << std::endl;
    std::cerr << "  n: col of matrix B" << std::endl;
    std::cerr << "  rankNum: number of ranks, in [2, 16]" << std::endl;
    std::cerr << "  mode: optional, 'precision' (default) | 'precision2' (double-launch residue test) | 'perf'"
              << std::endl;
    std::cerr << "  headMSize: optional, absent = " << QMRS_DEFAULT_HEAD_M_SIZE
              << " (default), <=0 = auto tileM, positive = override" << std::endl;
    std::cerr
        << "  quantType: optional, 0=E2M1E2M1 (MXFP4) | 1=E4M3E4M3 (default) | 2=E5M2E5M2 | 3=E4M3E5M2 | 4=E5M2E4M3"
        << std::endl;
    std::cerr << "Example: " << programName << " 2048 3584 4096 4" << std::endl;
}

// rankNum 范围: [2, QMRS_URMA_MAX_RANK_NUM]
constexpr int QMRS_MIN_RANK_NUM = 2;
// headMSize 和 paddedTailMSize 按 16 行对齐
constexpr int QMRS_MIN_ROW_ALIGN = 16;
constexpr const char* QMRS_MODE_PRECISION = "precision";
constexpr const char* QMRS_MODE_PRECISION2 = "precision2";
constexpr const char* QMRS_MODE_PERF = "perf";
// 默认端口由 master PID 派生, 避免连续用例争用; QMRS_HCCL_PORT 可显式覆盖, 缺省走 8998 兜底
constexpr unsigned QMRS_HCCL_PORT_BASE = 20000U;
constexpr unsigned QMRS_HCCL_PORT_SPAN = 20000U;
constexpr const char* QMRS_HCCL_PORT_FALLBACK = "8998";
// WriteNbi 的 len 必须小于 256MB, host 按最大 tile 提前校验 (adv_api/hcomm/hcomm.h 约束)
constexpr uint64_t QMRS_URMA_PUT_MAX_BYTES = 256ULL * 1024 * 1024;

// 输出固定 BF16; 支持 4 种 FP8 A/B 组合和 E2M1E2M1 (MXFP4, 每字节打包两个元素)
enum QuantType {
    QUANT_E2M1_E2M1 = 0, // MXFP4: A/B 均使用 fp4x2_e2m1_t, 每字节打包两个元素
    QUANT_E4M3_E4M3 = 1, // 默认
    QUANT_E5M2_E5M2 = 2,
    QUANT_E4M3_E5M2 = 3,
    QUANT_E5M2_E4M3 = 4,
};

void ParseArgs(int argc, char* argv[], int& m, int& k, int& n, int& rankNum, std::string& mode, int& headMSize,
               int& quantType)
{
    if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        PrintQmrsUsage(argv[0]);
        exit(1);
    }
    if (argc > 8) {
        throw std::invalid_argument("ERROR: too many arguments");
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
    } catch (const std::out_of_range&) {
        throw std::invalid_argument("ERROR: m k n rankNum exceeds int range");
    }

    mode = (argc >= 6) ? std::string(argv[5]) : std::string(QMRS_MODE_PRECISION);
    if (mode != QMRS_MODE_PRECISION && mode != QMRS_MODE_PRECISION2 && mode != QMRS_MODE_PERF) {
        throw std::invalid_argument("ERROR: mode must be 'precision', 'precision2' or 'perf'");
    }

    try {
        headMSize = (argc >= 7) ? std::stoi(argv[6]) : QMRS_DEFAULT_HEAD_M_SIZE;
    } catch (const std::invalid_argument&) {
        throw std::invalid_argument("ERROR: headMSize must be Integer");
    } catch (const std::out_of_range&) {
        throw std::invalid_argument("ERROR: headMSize exceeds int range");
    }

    try {
        quantType = (argc >= 8) ? std::stoi(argv[7]) : QUANT_E4M3_E4M3;
    } catch (const std::invalid_argument&) {
        throw std::invalid_argument("ERROR: quantType must be Integer");
    } catch (const std::out_of_range&) {
        throw std::invalid_argument("ERROR: quantType exceeds int range");
    }
    if (quantType < QUANT_E2M1_E2M1 || quantType > QUANT_E5M2_E4M3) {
        throw std::invalid_argument("ERROR: quantType must be in [0, 4]");
    }

    if (m <= 0 || k <= 0 || n <= 0 || rankNum <= 0) {
        throw std::invalid_argument("ERROR: m k n rankNum must be positive");
    }

    // m 必须被 rankNum 整除(M 轴按 rankSize 切分用于 AllToAll)
    if (m % rankNum != 0) {
        throw std::invalid_argument("ERROR: m must be divisible by rankNum");
    }

    if (rankNum < QMRS_MIN_RANK_NUM || rankNum > QMRS_URMA_MAX_RANK_NUM) {
        throw std::invalid_argument("ERROR: rankNum must be in [2, 16] (kernel MAX_RANK_SIZE bound)");
    }

    if (CeilDiv(k, MXFP_DIVISOR_SIZE) % MXFP_MULTI_BASE_SIZE != 0) {
        throw std::invalid_argument("ERROR: CeilDiv(k, MXFP_DIVISOR_SIZE) must be an even number");
    }
}

// tileM = min(ceil(max(blockNum,1) / max(floor(n/256),1)) * 256, chunkM)
uint32_t FormulaHeadMSize(uint32_t n, uint32_t chunkM, uint32_t blockNum)
{
    constexpr uint32_t BASE_TILE_M_SIZE = 256U; // 自动 tileM 的估算粒度
    const uint32_t nTiles =
        (n / BASE_TILE_M_SIZE) > 1U ? (n / BASE_TILE_M_SIZE) : 1U; // N 方向按 256 划分的块数, 至少为 1
    const uint32_t blocks = blockNum > 1U ? blockNum : 1U;
    uint32_t tileM = CeilDiv(blocks, nTiles) * BASE_TILE_M_SIZE;
    if (tileM > chunkM) {
        tileM = chunkM;
    }
    return tileM;
}

struct DeviceBuffers {
    GM_ADDR a{nullptr};
    GM_ADDR scaleA{nullptr};
    GM_ADDR b{nullptr};
    GM_ADDR scaleB{nullptr};
    GM_ADDR output{nullptr};
    GM_ADDR workspace{nullptr}; // [m,n] BF16, 保存本 rank 的 C_i 部分和
};

static void LaunchQmrsKernel(uint32_t usedCoreNum, aclrtStream stream, CommContext* devContext,
                             const DeviceBuffers& buf, const QuantMatmulReduceScatterTilingData& tilingData,
                             int quantType)
{
#define QMRS_LAUNCH(kernelName) \
    kernelName<<<usedCoreNum, nullptr, stream>>>(devContext, buf.a, buf.scaleA, buf.b, buf.scaleB, buf.workspace, \
                                                 buf.output, tilingData)
    switch (quantType) {
        case QUANT_E2M1_E2M1:
            QMRS_LAUNCH(QuantMatmulReduceScatterKernelE2M1E2M1);
            break;
        case QUANT_E4M3_E4M3:
            QMRS_LAUNCH(QuantMatmulReduceScatterKernelE4M3E4M3);
            break;
        case QUANT_E5M2_E5M2:
            QMRS_LAUNCH(QuantMatmulReduceScatterKernelE5M2E5M2);
            break;
        case QUANT_E4M3_E5M2:
            QMRS_LAUNCH(QuantMatmulReduceScatterKernelE4M3E5M2);
            break;
        case QUANT_E5M2_E4M3:
            QMRS_LAUNCH(QuantMatmulReduceScatterKernelE5M2E4M3);
            break;
        default:
            break;
    }
#undef QMRS_LAUNCH
}

// 各阶段共享的执行状态
struct QmrsSession {
    int rankNum;
    int rankId;
    int m;
    int k;
    int n;
    std::string mode;
    int quantType;

    // tiling 结果
    uint32_t chunkM{0};
    uint32_t headMSize{0};
    uint32_t tailMSize{0};
    uint32_t paddedTailMSize{0}; // 尾 tile 的 matmul 布局行数按 16 对齐, 通信仍使用 tailMSize
    uint32_t headTileCnt{0};
    uint32_t tailTileCnt{0};
    uint32_t tileCnt{0};
    uint64_t winReservedBytes{0};
    uint64_t commSlotNum{0};
    uint64_t tileMaxBytes{0};
    uint64_t winRequiredBytes{0};
    QuantMatmulReduceScatterTilingData tilingData{};

    // HCCL / 设备
    HcclComm comm{nullptr};
    CommContext* devContext{nullptr};
    CommContext hostCtx{};
    aclrtStream stream{nullptr};
    int32_t deviceId{0};

    // host 数据
    std::vector<uint8_t> hostA;
    std::vector<uint8_t> hostB;
    std::vector<uint8_t> hostScaleA;
    std::vector<uint8_t> hostScaleB;
    std::vector<uint16_t> hostOutput;
    std::string inputDir;
    std::string outputDir;
    size_t sizeWorkspace{0};
};

// 通过 integral_constant 将 quantType 分派到对应的模板实例
template <typename F>
static void DispatchQuantType(int quantType, F&& fn)
{
    using DT = mm::DataType;
    using E2M1 = std::integral_constant<DT, DT::DT_FLOAT4_E2M1>;
    using E4M3 = std::integral_constant<DT, DT::DT_FLOAT8_E4M3FN>;
    using E5M2 = std::integral_constant<DT, DT::DT_FLOAT8_E5M2>;
    switch (quantType) {
        case QUANT_E2M1_E2M1:
            fn(E2M1{}, E2M1{});
            break;
        case QUANT_E5M2_E5M2:
            fn(E5M2{}, E5M2{});
            break;
        case QUANT_E4M3_E5M2:
            fn(E4M3{}, E5M2{});
            break;
        case QUANT_E5M2_E4M3:
            fn(E5M2{}, E4M3{});
            break;
        case QUANT_E4M3_E4M3:
            fn(E4M3{}, E4M3{});
            break;
        default:
            break;
    }
}

// <=0 时按核数估算 headMSize; 先截到 chunkM, 再向下对齐 16 行, 对齐结果为 0 时取 16
static void ResolveHeadMSize(QmrsSession& session, int headMSizeArg)
{
    if (headMSizeArg <= 0) {
        // 自动估算只需 usedCoreNum, 先对完整 [m,n,k] 计算一次 tiling
        QuantMatmulTilingData fullTiling;
        DispatchQuantType(session.quantType, [&](auto dTypeA, auto dTypeB) {
            QuantMatmulTilingSwat<decltype(dTypeA)::value, decltype(dTypeB)::value, mm::BiasDataType::DT_FLOAT>
                fullTilingEngine;
            fullTilingEngine.GetTilingData(session.m, session.n, session.k, false, true, fullTiling);
        });
        session.headMSize = FormulaHeadMSize(static_cast<uint32_t>(session.n), session.chunkM, fullTiling.usedCoreNum);
        if (session.rankId == 0) {
            INFO_LOG("FormulaHeadMSize: n=%u chunkM=%u blockNum=%u -> tileM=%u", static_cast<uint32_t>(session.n),
                     session.chunkM, fullTiling.usedCoreNum, session.headMSize);
        }
    } else {
        session.headMSize = static_cast<uint32_t>(headMSizeArg);
        if (session.headMSize > session.chunkM) {
            session.headMSize = session.chunkM;
        }
    }
    session.headMSize = (session.headMSize / QMRS_MIN_ROW_ALIGN) * QMRS_MIN_ROW_ALIGN;
    if (session.headMSize == 0) {
        session.headMSize = QMRS_MIN_ROW_ALIGN;
    }
}

// 计算 win 前缀、槽位数和总容量
static void ComputeWinCapacity(QmrsSession& session)
{
    // 最多使用 3 个槽; tile 数不足 3 时每个 tile 独占一槽
    // win 头部保留区公式 (与 kernel 侧 winDataOffset 计算同式):
    //   按两个 rankSize×32B 区域计算保留量并对齐到 WIN_DATA_ALIGN (128B);
    //   这不表示 localFlag/teamSyncCounter 实际位于 URMA win 内
    session.winReservedBytes =
        (WIN_BARRIER_REGION_NUM * static_cast<uint64_t>(session.rankNum) * WIN_BARRIER_FLAG_SIZE + WIN_DATA_ALIGN - 1) /
        WIN_DATA_ALIGN * WIN_DATA_ALIGN;
    uint64_t tileMaxMSize = (session.headMSize > session.tailMSize) ? session.headMSize : session.tailMSize;
    session.tileMaxBytes = tileMaxMSize * static_cast<uint64_t>(session.n) * sizeof(uint16_t);
    session.commSlotNum = (session.tileCnt < QMRS_MAX_COMM_SLOT_NUM) ? session.tileCnt : QMRS_MAX_COMM_SLOT_NUM;
    session.winRequiredBytes =
        session.winReservedBytes + session.commSlotNum * static_cast<uint64_t>(session.rankNum) * session.tileMaxBytes;
}

// 每次计算覆盖所有 rank 段, QBMM 逻辑 M = rankNum * paddedTileM
static void BuildQbmmTiling(QmrsSession& session)
{
    DispatchQuantType(session.quantType, [&](auto dTypeA, auto dTypeB) {
        using TilingEngine =
            QuantMatmulTilingSwat<decltype(dTypeA)::value, decltype(dTypeB)::value, mm::BiasDataType::DT_FLOAT>;
        TilingEngine headTilingEngine;
        headTilingEngine.EnableBaseMHalving(true);
        headTilingEngine.SetMTailAlignEnable(true);
        uint32_t mainHeadLogicalM = static_cast<uint32_t>(session.rankNum) * session.headMSize;
        headTilingEngine.GetTilingData(mainHeadLogicalM, session.n, session.k, false, true,
                                       session.tilingData.tileQbmmTilingData);
        if (session.tailTileCnt > 0 && session.tailMSize > 0) {
            TilingEngine tailTilingEngine;
            tailTilingEngine.SetMTailAlignEnable(true);
            // 尾块使用 paddedTailMSize 做 QBMM tiling, realTileM 仍取真实 tailMSize
            uint32_t mainTailLogicalM = static_cast<uint32_t>(session.rankNum) * session.paddedTailMSize;
            tailTilingEngine.GetTilingData(mainTailLogicalM, session.n, session.k, false, true,
                                           session.tilingData.tailQbmmTilingData);
        } else {
            session.tilingData.tailQbmmTilingData = {};
        }
    });
}

static void LogTilingSummary(const QmrsSession& session)
{
    if (session.rankId != 0) {
        return;
    }
    INFO_LOG("chunkM=%u, headMSize=%u, headTileCnt=%u, tailMSize=%u (padded %u), tailTileCnt=%u, tileCnt=%u, "
             "rankSize=%d",
             session.chunkM, session.headMSize, session.headTileCnt, session.tailMSize, session.paddedTailMSize,
             session.tailTileCnt, session.tileCnt, session.rankNum);
    auto printTiling = [](const char* name, const QuantMatmulTilingData& t) {
        INFO_LOG("[Tiling] %s: m=%u n=%u k=%u baseM=%u baseN=%u baseK=%u stepK=%u usedCoreNum=%u dbL0c=%u "
                 "nBufferNum=%u scaleKL1=%u mTailTile=%u nTailTile=%u mBaseTailSplitCnt=%u nBaseTailSplitCnt=%u",
                 name, t.m, t.n, t.k, t.baseM, t.baseN, t.baseK, t.stepK, t.usedCoreNum, t.dbL0c, t.nBufferNum,
                 t.scaleKL1, t.mTailTile, t.nTailTile, t.mBaseTailSplitCnt, t.nBaseTailSplitCnt);
    };
    printTiling("tileQbmmTilingData (blockDim source)", session.tilingData.tileQbmmTilingData);
    if (session.tailTileCnt > 0 && session.tailMSize > 0) {
        printTiling("tailQbmmTilingData", session.tilingData.tailQbmmTilingData);
    }
}

// 每 rank 的 chunkM = m/rankNum, 再划分为 head tile 和至多一个 tail tile
static bool BuildTilingData(QmrsSession& session, int headMSizeArg)
{
    session.chunkM = static_cast<uint32_t>(session.m) / static_cast<uint32_t>(session.rankNum);

    ResolveHeadMSize(session, headMSizeArg);

    session.headTileCnt = session.chunkM / session.headMSize;
    session.tailMSize = session.chunkM % session.headMSize;
    // 尾块 QBMM 按 16 行补齐; PUT, ReduceSum 和 win 容量使用真实 tailMSize
    session.paddedTailMSize = (session.tailMSize > 0) ? (session.tailMSize + QMRS_MIN_ROW_ALIGN - 1) /
                                                            QMRS_MIN_ROW_ALIGN * QMRS_MIN_ROW_ALIGN :
                                                        0;
    session.tailTileCnt = (session.tailMSize > 0) ? 1 : 0;
    session.tileCnt = session.headTileCnt + session.tailTileCnt;

    ComputeWinCapacity(session);

    session.tilingData.commTilingData.splitAxisTileCnt = static_cast<uint64_t>(session.headTileCnt);
    session.tilingData.commTilingData.splitAxisTileSize = static_cast<uint64_t>(session.headMSize);
    session.tilingData.commTilingData.splitAxisTailSize = static_cast<uint64_t>(session.tailMSize);
    session.tilingData.commTilingData.splitAxisTailCnt = static_cast<uint64_t>(session.tailTileCnt);
    session.tilingData.commTilingData.nonSplitAxisSize = static_cast<uint64_t>(session.n);

    BuildQbmmTiling(session);
    LogTilingSummary(session);

    // 当前 kernel launch 约定 usedCoreNum > rankNum
    if (session.tilingData.tileQbmmTilingData.usedCoreNum <= static_cast<uint32_t>(session.rankNum)) {
        ERROR_LOG("tileQbmmTilingData.usedCoreNum=%u must be > rankNum=%d",
                  session.tilingData.tileQbmmTilingData.usedCoreNum, session.rankNum);
        return false;
    }
    return true;
}

static bool SetupHcclContext(QmrsSession& session)
{
    const char* portEnv = getenv("QMRS_HCCL_PORT");
    const std::string ipport =
        std::string("tcp://127.0.0.1:") + ((portEnv != nullptr) ? portEnv : QMRS_HCCL_PORT_FALLBACK);

    if (aclInit(nullptr) != ACL_SUCCESS) {
        ERROR_LOG("rank %d aclInit failed", session.rankId);
        return false;
    }
    session.deviceId = session.rankId;
    if (aclrtSetDevice(session.deviceId) != ACL_SUCCESS) {
        ERROR_LOG("rank %d aclrtSetDevice failed", session.rankId);
        return false;
    }
    if (aclrtCreateStream(&session.stream) != ACL_SUCCESS) {
        ERROR_LOG("rank %d aclrtCreateStream failed", session.rankId);
        return false;
    }

    RootInfoExchanger exchanger(static_cast<uint32_t>(session.rankId), static_cast<uint32_t>(session.rankNum),
                                ipport.c_str());
    HcclRootInfo rootInfo;
    if (!exchanger.Exchange(rootInfo)) {
        ERROR_LOG("rank %d exchange rootInfo failed", session.rankId);
        return false;
    }

    HcclCommConfig config;
    HcclCommConfigInit(&config);
    config.hcclWorldRankID = static_cast<uint32_t>(session.rankId);
    HCCL_CHECK(HcclCommInitRootInfoConfig(static_cast<uint32_t>(session.rankNum), &rootInfo,
                                          static_cast<uint32_t>(session.rankId), &config, &session.comm));
    if (session.comm == nullptr) {
        ERROR_LOG("rank %d HcclCommInitRootInfoConfig failed", session.rankId);
        return false;
    }

    CommChannelBuilder builder(session.comm);
    const char* ctxTag = "quant_matmul_reduce_scatter";
    session.devContext = reinterpret_cast<CommContext*>(builder.CreateDeviceContext(
        &session.hostCtx, sizeof(CommContext), ctxTag, &session.hostCtx.udmaCtx, &session.hostCtx.ubmemCtx));
    if (session.devContext == nullptr) {
        ERROR_LOG("rank %d CreateDeviceContext failed", session.rankId);
        return false;
    }

    // win 容量校验: 对照 HCCL 通道实际缓冲大小 (防 PUT 越界写远端 win)
    void* hcclBuf = nullptr;
    uint64_t hcclBufSize = 0;
    if (HcclGetHcclBuffer(session.comm, &hcclBuf, &hcclBufSize) != HCCL_SUCCESS || hcclBuf == nullptr) {
        ERROR_LOG("rank %d HcclGetHcclBuffer failed", session.rankId);
        return false;
    }
    if (session.winRequiredBytes > hcclBufSize) {
        ERROR_LOG("rank %d win required=%lu bytes exceeds actual hccl buffer=%lu bytes. Reduce m/n or rankNum.",
                  session.rankId, static_cast<unsigned long>(session.winRequiredBytes),
                  static_cast<unsigned long>(hcclBufSize));
        return false;
    }
    if (session.tileMaxBytes >= QMRS_URMA_PUT_MAX_BYTES) {
        ERROR_LOG("rank %d single PUT block=%lu bytes must be < URMA limit %lu bytes. Reduce headMSize or n.",
                  session.rankId, static_cast<unsigned long>(session.tileMaxBytes),
                  static_cast<unsigned long>(QMRS_URMA_PUT_MAX_BYTES));
        return false;
    }
    INFO_LOG("[Rank %d] win required=%lu MB / hccl buffer=%lu MB", session.rankId,
             static_cast<unsigned long>(session.winRequiredBytes / (1024UL * 1024UL)),
             static_cast<unsigned long>(hcclBufSize / (1024UL * 1024UL)));

    exchanger.Barrier();
    exchanger.Close();
    INFO_LOG("[Rank %d] HCCL setup complete", session.rankId);
    return true;
}

// 输入输出位于 executable 所在目录的 input/<rankId> 和 output/<rankId>
static bool LoadInputFiles(QmrsSession& session)
{
    // A ND 与 B DN 均沿 K 打包; MXFP4 每字节两个元素, FP8 每字节一个元素
    const size_t packFactor = (session.quantType == QUANT_E2M1_E2M1) ? 2U : 1U;
    session.hostA.resize(static_cast<size_t>(session.m) * session.k / packFactor);
    session.hostB.resize(static_cast<size_t>(session.n) * session.k / packFactor);
    session.hostScaleA.resize(static_cast<size_t>(session.m) * CeilDiv(session.k, MXFP_DIVISOR_SIZE) *
                              MXFP_MULTI_BASE_SIZE);
    session.hostScaleB.resize(static_cast<size_t>(session.n) * CeilDiv(session.k, MXFP_DIVISOR_SIZE) *
                              MXFP_MULTI_BASE_SIZE);
    session.hostOutput.resize(static_cast<size_t>(session.chunkM) * session.n);

    const size_t sizeA = session.hostA.size() * sizeof(uint8_t);
    const size_t sizeB = session.hostB.size() * sizeof(uint8_t);
    const size_t sizeScaleA = session.hostScaleA.size() * sizeof(uint8_t);
    const size_t sizeScaleB = session.hostScaleB.size() * sizeof(uint8_t);

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
    session.inputDir = baseDir + "/input/" + std::to_string(session.rankId);
    session.outputDir = baseDir + "/output/" + std::to_string(session.rankId);

    if (!ReadFile(session.inputDir + "/input_a.bin", session.hostA.data(), sizeA) ||
        !ReadFile(session.inputDir + "/input_b.bin", session.hostB.data(), sizeB) ||
        !ReadFile(session.inputDir + "/input_scaleA.bin", session.hostScaleA.data(), sizeScaleA) ||
        !ReadFile(session.inputDir + "/input_scaleB.bin", session.hostScaleB.data(), sizeScaleB)) {
        return false;
    }

    // quant_type.txt 必须与命令行 quantType 一致
    {
        std::ifstream quantTypeFile(session.inputDir + "/quant_type.txt");
        if (!quantTypeFile.is_open()) {
            ERROR_LOG("quant_type.txt missing: %s", (session.inputDir + "/quant_type.txt").c_str());
            return false;
        }
        int fileQuantType = -1;
        quantTypeFile >> fileQuantType;
        if (quantTypeFile.fail() || fileQuantType != session.quantType) {
            ERROR_LOG("quantType mismatch: argv=%d vs input dir=%d", session.quantType, fileQuantType);
            return false;
        }
    }
    return true;
}

// 申请设备内存并拷贝输入, 失败由上层统一释放
static bool PrepareDeviceMemory(QmrsSession& session, DeviceBuffers& buf)
{
    const size_t sizeA = session.hostA.size() * sizeof(uint8_t);
    const size_t sizeB = session.hostB.size() * sizeof(uint8_t);
    const size_t sizeScaleA = session.hostScaleA.size() * sizeof(uint8_t);
    const size_t sizeScaleB = session.hostScaleB.size() * sizeof(uint8_t);
    size_t sizeC = static_cast<size_t>(session.chunkM) * session.n * sizeof(uint16_t);
    session.sizeWorkspace = static_cast<size_t>(session.m) * session.n * sizeof(uint16_t);

    // malloc 失败时释放已申请项并置空, 避免 CleanupSession 二次 free
    auto freeAndNull = [](GM_ADDR& ptr) {
        if (ptr != nullptr) {
            aclrtFree(ptr);
            ptr = nullptr;
        }
    };
    auto mallocOrCleanup = [&](GM_ADDR* ptr, size_t size, const char* name) -> bool {
        if (aclrtMalloc((void**)ptr, size, ACL_MEM_MALLOC_HUGE_ONLY) != ACL_SUCCESS) {
            std::cerr << "aclrtMalloc failed: " << name << " (" << size << " bytes)" << std::endl;
            freeAndNull(buf.a);
            freeAndNull(buf.b);
            freeAndNull(buf.scaleA);
            freeAndNull(buf.scaleB);
            freeAndNull(buf.output);
            freeAndNull(buf.workspace);
            return false;
        }
        return true;
    };
    if (!mallocOrCleanup(&buf.a, sizeA, "deviceA") || !mallocOrCleanup(&buf.b, sizeB, "deviceB") ||
        !mallocOrCleanup(&buf.scaleA, sizeScaleA, "deviceScaleA") ||
        !mallocOrCleanup(&buf.scaleB, sizeScaleB, "deviceScaleB") ||
        !mallocOrCleanup(&buf.output, sizeC, "deviceOutput") ||
        !mallocOrCleanup(&buf.workspace, session.sizeWorkspace, "deviceWorkspace")) {
        return false;
    }
    INFO_LOG("[Rank %d] workspace size=%zu bytes (m=%d, n=%d)", session.rankId, session.sizeWorkspace, session.m,
             session.n);

    QMRS_ACL_RET(aclrtMemcpy(buf.a, sizeA, session.hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));
    QMRS_ACL_RET(aclrtMemcpy(buf.b, sizeB, session.hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));
    QMRS_ACL_RET(aclrtMemcpy(buf.scaleA, sizeScaleA, session.hostScaleA.data(), sizeScaleA, ACL_MEMCPY_HOST_TO_DEVICE));
    QMRS_ACL_RET(aclrtMemcpy(buf.scaleB, sizeScaleB, session.hostScaleB.data(), sizeScaleB, ACL_MEMCPY_HOST_TO_DEVICE));
    INFO_LOG("[Rank %d] data loaded to device", session.rankId);
    return true;
}

// perf 默认用 256MB memset 冲刷 L2, 且不计入设备事件耗时; QMRS_L2_FLUSH=0 可关闭
struct L2Flusher {
    static constexpr uint64_t FLUSH_BYTES = 256UL * 1024 * 1024;
    GM_ADDR buf{nullptr};
    bool enabled{false};

    bool Setup(const QmrsSession& session)
    {
        const char* flushEnv = getenv("QMRS_L2_FLUSH");
        enabled = (session.mode == QMRS_MODE_PERF) && !((flushEnv != nullptr) && std::string(flushEnv) == "0");
        if (!enabled) {
            return true;
        }
        if (aclrtMalloc((void**)&buf, FLUSH_BYTES, ACL_MEM_MALLOC_HUGE_ONLY) != ACL_SUCCESS) {
            std::cerr << "aclrtMalloc failed: l2FlushBuf (" << FLUSH_BYTES << " bytes)" << std::endl;
            return false;
        }
        INFO_LOG("[Rank %d] L2 flush enabled: %lu MB memset between iterations", session.rankId,
                 static_cast<unsigned long>(FLUSH_BYTES / (1024UL * 1024UL)));
        return true;
    }

    bool Flush(const QmrsSession& session)
    {
        if (!enabled) {
            return true;
        }
        // maxCount 是缓冲区容量, count 是本次写入字节数
        const aclError aclRet = aclrtMemsetAsync(buf, FLUSH_BYTES, 0, FLUSH_BYTES, session.stream);
        if (aclRet != ACL_SUCCESS) {
            ERROR_LOG("%s:%d aclError:%d", __FILE__, __LINE__, static_cast<int>(aclRet));
            return false;
        }
        return true;
    }

    void Teardown()
    {
        if (buf != nullptr) {
            aclrtFree(buf);
        }
    }
};

// precision/precision2 保存 win 和 workspace 用于定位
static void DumpDebugArtifacts(const QmrsSession& session, const DeviceBuffers& buf)
{
    // win_dump 保存 URMA 前缀和远端 PUT 槽位; workspace_dump 保存本 rank 完整 C_i, 第 r 段发往 rank r
    const uint64_t winDumpBytes = session.winRequiredBytes;
    std::vector<uint8_t> hostWin(winDumpBytes, 0);
    void* winDevPtr = reinterpret_cast<void*>(session.hostCtx.udmaCtx.commBufferAddrs[session.rankId]);
    aclError dumpRet = (winDevPtr == nullptr) ? ACL_ERROR_INVALID_PARAM :
                                                aclrtMemcpy(hostWin.data(), winDumpBytes, winDevPtr, winDumpBytes,
                                                            ACL_MEMCPY_DEVICE_TO_HOST);
    if (dumpRet == ACL_SUCCESS) {
        if (!WriteFile(session.outputDir + "/win_dump.bin", hostWin.data(), winDumpBytes)) {
            ERROR_LOG("WriteFile failed: %s", (session.outputDir + "/win_dump.bin").c_str());
        }
    } else {
        ERROR_LOG("win dump skipped: base=%p, ret=%d", winDevPtr, static_cast<int>(dumpRet));
    }

    if (buf.workspace != nullptr) {
        std::vector<uint8_t> hostWs(session.sizeWorkspace, 0);
        const aclError wsRet = aclrtMemcpy(hostWs.data(), session.sizeWorkspace, buf.workspace, session.sizeWorkspace,
                                           ACL_MEMCPY_DEVICE_TO_HOST);
        if (wsRet == ACL_SUCCESS) {
            if (!WriteFile(session.outputDir + "/workspace_dump.bin", hostWs.data(), session.sizeWorkspace)) {
                ERROR_LOG("WriteFile failed: %s", (session.outputDir + "/workspace_dump.bin").c_str());
            }
        } else {
            ERROR_LOG("workspace dump skipped: ret=%d", static_cast<int>(wsRet));
        }
    }
}

static bool WriteOutputToHost(QmrsSession& session, const DeviceBuffers& buf, const char* fileName)
{
    QMRS_ACL_RET(aclrtMemcpy(session.hostOutput.data(), session.hostOutput.size() * sizeof(uint16_t), buf.output,
                             session.hostOutput.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST));
    const std::string path = session.outputDir + "/" + fileName;
    if (!WriteFile(path, session.hostOutput.data(), session.hostOutput.size() * sizeof(uint16_t))) {
        ERROR_LOG("WriteFile failed: %s", path.c_str());
        return false;
    }
    return true;
}

// precision2 连续 launch 两次并逐元素比较; 两种 precision 模式均保存输出和定位 dump
static bool RunPrecisionFlow(QmrsSession& session, const DeviceBuffers& buf)
{
    std::vector<uint16_t> iter1Output;
    if (session.mode == QMRS_MODE_PRECISION2) {
        if (!WriteOutputToHost(session, buf, "npu_out_iter1.bin")) {
            return false;
        }
        iter1Output = session.hostOutput;
        INFO_LOG("[Rank %d] launch 1 output archived (npu_out_iter1.bin), launching second time", session.rankId);
        LaunchQmrsKernel(session.tilingData.tileQbmmTilingData.usedCoreNum, session.stream, session.devContext, buf,
                         session.tilingData, session.quantType);
        QMRS_ACL_RET(aclrtSynchronizeStream(session.stream));
        INFO_LOG("[Rank %d] second launch synced", session.rankId);
    }

    // 每 rank 只拷回 chunkM × n 的归约结果
    if (!WriteOutputToHost(session, buf, "npu_out.bin")) {
        return false;
    }

    if (session.mode == QMRS_MODE_PRECISION2) {
        if (session.hostOutput != iter1Output) {
            ERROR_LOG("[Rank %d] precision2 FAILED: launch2 output != launch1 output (%zu elems)", session.rankId,
                      session.hostOutput.size());
            DumpDebugArtifacts(session, buf);
            return false;
        }
        INFO_LOG("[Rank %d] precision2: launch2 output == launch1 output (%zu elems)", session.rankId,
                 session.hostOutput.size());
    }

    DumpDebugArtifacts(session, buf);
    return true;
}

// 首次 launch 作为 warmup; 后续迭代分别用设备事件计时, L2 flush 不计入耗时
static bool RunPerfFlow(QmrsSession& session, const DeviceBuffers& buf, L2Flusher& flusher)
{
    constexpr int32_t BENCHMARK_ITERATIONS = 20;
    aclrtEvent kernelStartEvent = nullptr;
    aclrtEvent kernelEndEvent = nullptr;
    QMRS_ACL_RET(aclrtCreateEvent(&kernelStartEvent));
    QMRS_ACL_RET(aclrtCreateEvent(&kernelEndEvent));
    double totalElapsedUs = 0.0;
    for (int32_t i = 1; i < BENCHMARK_ITERATIONS; ++i) {
        QMRS_ACL_RET(aclrtRecordEvent(kernelStartEvent, session.stream));
        LaunchQmrsKernel(session.tilingData.tileQbmmTilingData.usedCoreNum, session.stream, session.devContext, buf,
                         session.tilingData, session.quantType);
        QMRS_ACL_RET(aclrtRecordEvent(kernelEndEvent, session.stream));
        QMRS_ACL_RET(aclrtSynchronizeStream(session.stream));
        float iterElapsedMs = 0.0F;
        QMRS_ACL_RET(aclrtEventElapsedTime(&iterElapsedMs, kernelStartEvent, kernelEndEvent));
        const double iterElapsedUs = static_cast<double>(iterElapsedMs) * 1000.0;
        totalElapsedUs += iterElapsedUs;
        INFO_LOG("[Rank %d] iteration %d/%d: %.3f us", session.rankId, i + 1, BENCHMARK_ITERATIONS, iterElapsedUs);
        if (!flusher.Flush(session)) {
            aclrtDestroyEvent(kernelEndEvent);
            aclrtDestroyEvent(kernelStartEvent);
            return false;
        }
    }
    aclrtDestroyEvent(kernelEndEvent);
    aclrtDestroyEvent(kernelStartEvent);
    INFO_LOG("[Rank %d] Kernel perf: %.3f us (avg over %d iterations)", session.rankId,
             totalElapsedUs / (BENCHMARK_ITERATIONS - 1), BENCHMARK_ITERATIONS - 1);

    if (!WriteOutputToHost(session, buf, "npu_out.bin")) {
        return false;
    }
    return true;
}

// 成功和失败路径统一清理设备内存, HCCL 和 ACL 资源
static void CleanupSession(const QmrsSession& session, const DeviceBuffers& buf, L2Flusher& flusher)
{
    flusher.Teardown();
    for (GM_ADDR ptr : {buf.a, buf.scaleA, buf.b, buf.scaleB, buf.output, buf.workspace}) {
        if (ptr != nullptr) { // 部分申请失败时仅回收已分配项
            aclrtFree(ptr);
        }
    }
    // 部分初始化的失败路径: 仅释放已建立的资源 (comm/stream 未建立时跳过)
    if (session.comm != nullptr) {
        HcclCommDestroy(session.comm);
    }
    if (session.stream != nullptr) {
        ACL_CHECK(aclrtDestroyStream(session.stream));
    }
    ACL_CHECK(aclrtResetDevice(session.deviceId));
    ACL_CHECK(aclFinalize());
}

int RunQuantMatmulReduceScatter(int rankNum, int rankId, int m, int k, int n, const std::string& mode, int headMSizeArg,
                                int quantType)
{
    INFO_LOG("rankNum=%d, rankId=%d, mode=%s, headMSize=%d (<=0: formula), quantType=%d", rankNum, rankId, mode.c_str(),
             headMSizeArg, quantType);

    QmrsSession session;
    session.rankNum = rankNum;
    session.rankId = rankId;
    session.m = m;
    session.k = k;
    session.n = n;
    session.mode = mode;
    session.quantType = quantType;

    DeviceBuffers buf;
    L2Flusher flusher;
    bool flowOk = false;
    do {
        if (!BuildTilingData(session, headMSizeArg)) {
            break;
        }
        if (!SetupHcclContext(session)) {
            break;
        }
        if (!LoadInputFiles(session)) {
            break;
        }
        if (!PrepareDeviceMemory(session, buf)) {
            break;
        }
        if (!flusher.Setup(session)) {
            break;
        }

        INFO_LOG("[Rank %d] launching kernel (usedCoreNum=%u, mode=%s)", session.rankId,
                 session.tilingData.tileQbmmTilingData.usedCoreNum, session.mode.c_str());
        LaunchQmrsKernel(session.tilingData.tileQbmmTilingData.usedCoreNum, session.stream, session.devContext, buf,
                         session.tilingData, session.quantType);
        INFO_LOG("[Rank %d] kernel launched, syncing...", session.rankId);

        const aclError syncRet = aclrtSynchronizeStream(session.stream);
        if (syncRet != ACL_SUCCESS) {
            ERROR_LOG("%s:%d aclError:%d", __FILE__, __LINE__, static_cast<int>(syncRet));
            break;
        }
        INFO_LOG("[Rank %d] stream synced", session.rankId);

        if (!flusher.Flush(session)) {
            break;
        }

        flowOk = (session.mode == QMRS_MODE_PERF) ? RunPerfFlow(session, buf, flusher) : RunPrecisionFlow(session, buf);
    } while (false);

    CleanupSession(session, buf, flusher);
    return flowOk ? 0 : -1;
}

int main(int argc, char* argv[])
{
    int m;
    int k;
    int n;
    int rankNum;
    int headMSize;
    int quantType = QUANT_E4M3_E4M3;
    std::string mode;
    try {
        ParseArgs(argc, argv, m, k, n, rankNum, mode, headMSize, quantType);
    } catch (const std::invalid_argument& e) {
        std::cerr << e.what() << std::endl;
        PrintQmrsUsage(argv[0]);
        return -1;
    }

    INFO_LOG("Master (PID=%d) will fork %d processes (mode=%s, headMSize=%d, quantType=%d)", getpid(), rankNum,
             mode.c_str(), headMSize, quantType);

    // fork 前设置根信息交换端口, 供各 rank 继承
    if (getenv("QMRS_HCCL_PORT") == nullptr) {
        const std::string portEnv =
            std::to_string(QMRS_HCCL_PORT_BASE + static_cast<unsigned>(getpid()) % QMRS_HCCL_PORT_SPAN);
        setenv("QMRS_HCCL_PORT", portEnv.c_str(), 1);
    }

    std::vector<pid_t> pids(rankNum);
    for (int rankId = 0; rankId < rankNum; ++rankId) {
        pid_t pid = fork();

        if (pid < 0) {
            ERROR_LOG("Fork failed for rank %d", rankId);
            exit(-1);
        } else if (pid == 0) {
            int ret = RunQuantMatmulReduceScatter(rankNum, rankId, m, k, n, mode, headMSize, quantType);
            exit(ret);
        } else {
            pids[rankId] = pid;
            INFO_LOG("Forked Rank %d -> PID %d", rankId, pid);
        }
    }

    int status = 0;
    bool allSuccess = true;
    for (int rankId = 0; rankId < rankNum; ++rankId) {
        pid_t pid = pids[rankId];
        pid_t waitRet = waitpid(pid, &status, 0);
        if (waitRet == -1) {
            allSuccess = false;
            ERROR_LOG("waitpid failed for PID %d", pid);
            continue;
        }
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            allSuccess = false;
            ERROR_LOG("Worker PID %d failed", pid);
        }
    }

    std::cout << "All workers finished. Status: " << (allSuccess ? "SUCCESS" : "FAILURE") << std::endl;
    return allSuccess ? 0 : -1;
}
