/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_grouped_matmul_dequant_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_QUANT_MATMUL_DEQUANT_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_QUANT_MATMUL_DEQUANT_H
#include <cstdint>
#include <vector>

namespace optiling {

constexpr int32_t FLOAT16_PERBLOCK = 16;
constexpr int32_t BLOCK_SIZE = 32;
constexpr int32_t NUM_OF_AICORE = 8;
constexpr int32_t NUMBER_2 = 2;
constexpr int32_t NUMBER_3 = 3;
constexpr int32_t NUMBER_4 = 4;
constexpr int32_t NUMBER_5 = 5;
constexpr int32_t NUMBER_16 = 16;
constexpr int32_t NUMBER_256 = 256;
constexpr int32_t NUMBER_128 = 128;

constexpr int32_t K_FRACTAL_INT8 = 32;
constexpr int32_t NM_FRACTAL_INT8 = 16;
constexpr int32_t L0C_FRACTAL = 16;
constexpr int32_t L0_ADDR_ALIGN = 512;

constexpr int32_t GEMV_THRESHOLD = 8;
constexpr int32_t REDUCE_THRESHOLD = 32 * 1024;

constexpr int32_t SYS_WORKSPACE_310P = 2 * 1024 * 1024;
constexpr uint64_t L2_HEADROOM_BYTES = 14ULL * 1024 * 1024;

constexpr uint32_t QGMMD_FLAG_DYNAMIC_QUANT = 1u << 0;
constexpr uint32_t QGMMD_FLAG_HAS_SMOOTH = 1u << 1;
constexpr uint32_t QGMMD_FLAG_PER_TOKEN = 1u << 2;
constexpr uint32_t QGMMD_FLAG_SCALE_IS_INT64 = 1u << 3;
constexpr uint32_t QGMMD_FLAG_BLOCKED_ZN = 1u << 4;
constexpr uint32_t QGMMD_FLAG_SMALL_M = 1u << 5;
constexpr uint32_t QGMMD_FLAG_L2_WEIGHT_FITS = 1u << 6;
constexpr uint32_t QGMMD_FLAG_HAS_X_RAW_L1 = 1u << 7;
constexpr uint32_t QGMMD_FLAG_USE_L2_HINT = 1u << 8;
constexpr uint32_t QGMMD_FLAG_DBL0C = 1u << 9;
constexpr uint32_t QGMMD_FLAG_USE_XZN_WORKSPACE = 1u << 10;

constexpr uint64_t TILING_KEY_GEMV = 10000001;
constexpr uint64_t TILING_KEY_NORMAL = 10000002;
constexpr uint64_t TILING_KEY_GROUPED = 10000003;
constexpr uint64_t TILING_KEY_SWIFT = 10000004;
constexpr uint64_t TILING_KEY_GROUPED_DQ_PERTOKEN_OPT = 10000005;

constexpr int32_t SWIFT_M_MIN = 20;
constexpr int32_t SWIFT_M_MAX = 192;
constexpr int32_t SWIFT_K_MIN = 512;
constexpr int32_t SWIFT_K_MAX = 5120;
constexpr int32_t SWIFT_N_MIN = 1025;
constexpr int32_t SWIFT_N_MAX = 5120;
constexpr float GMM_GEMV_THRESHOLD_C1 = -7.916f;
constexpr float GMM_GEMV_THRESHOLD_C2 = 22.07f;
constexpr int32_t GMM_GMEV_THRESHOLD_MIN = 7;
constexpr int32_t GMM_GMEV_THRESHOLD_MAX = 20;
constexpr uint32_t BATCH_MODE = 1;

enum class QuantMatmulDequantTilingKey : uint64_t {
    GEMV = TILING_KEY_GEMV,
    NORMAL = TILING_KEY_NORMAL,
    GROUPED = TILING_KEY_GROUPED,
    SWIFT = TILING_KEY_SWIFT,
    GROUPED_DQ_PERTOKEN_OPT = TILING_KEY_GROUPED_DQ_PERTOKEN_OPT,
    UNDFINED = 10000099
};

struct QuantMatmulDequantCompileInfo {
    uint32_t coreNum;
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l2Size;
    uint64_t l0CSize;
    uint64_t l0ASize;
    uint64_t l0BSize;
    platform_ascendc::SocVersion socVersion;
    std::string socVersionStr = "";
};
struct QuantMatmulDequantParam {
    // platform
    uint64_t CoreNum;
    uint64_t UBSize;
    uint64_t L1Size;
    uint64_t L0ASize;
    uint64_t L0BSize;
    uint64_t L0CSize;

    // shape
    bool perToken;
    bool dynamicQuant;
    bool smoothScale;
    uint64_t originE;
    uint64_t originM;
    uint64_t originN;
    uint64_t originK;
    uint64_t originKAligned32;
    uint64_t originKAligned512;
    uint64_t fracK;
    uint64_t fracN;

    // dynamic
    uint64_t dynamicBaseK;
    uint64_t dynamicIterK;
    uint64_t dynamicBaseKTail;

    // gemv
    uint64_t singleCoreFracN;
    uint64_t singleCoreFracNTail;
    uint64_t baseFracN;
    uint64_t baseFracK;
    uint64_t baseFracNL0C;
    uint64_t ubBaseK;
    uint64_t ubIterK;
    uint64_t ubBaseKTail;

    // normal
    uint64_t fracM;
    uint64_t tailM;
    uint64_t processXKBaseNMax;
    uint64_t processXKBaseN;
    uint64_t processXKloop;
    uint64_t processXKloopPerfracM;
    uint64_t processXKTailN;
    bool MMmod;
    uint64_t MCoreNum;
    uint64_t NCoreNum;
    uint64_t singleCoreM;
    uint64_t singleCoreN;
    uint64_t singleCoreMTail;
    uint64_t singleCoreNTail;
    uint64_t baseMNum;
    uint64_t baseNNum;
    uint64_t baseKNum;

    // swift
    uint64_t swiftGEMVThreshold;
    uint64_t baseNNum_2;
    uint64_t baseKNum_2;
    uint64_t baseN;
    uint64_t baseN_2;
    uint64_t baseK;
    uint64_t baseK_2;
    uint64_t baseKTail;
    uint64_t baseKTail_2;

    uint64_t ubKMask;

    uint32_t isXScaleHalf;
    bool scaleIsInt64;

    // Optimized dynamic per-token grouped path.
    uint32_t Mb;
    uint32_t Nb;
    uint32_t Kb;
    uint32_t kPasses;
    uint32_t wChunkKPasses;
    uint32_t phaseXChunk;
    uint32_t phaseXFullPairs;
    uint32_t phaseXTail;
    uint32_t maxOneTurnToken;
    uint32_t maxWeightColOneTurn;
    uint32_t perCoreIters;
    uint32_t remainderIters;
    uint32_t weightBlockN1;
    uint32_t nBlockSize;
    uint32_t flags;
    bool smallM;
    bool l2WeightFits;
    bool hasXRawL1;
    bool useL2Hint;
    bool dbL0c;
    bool useXZNWorkspace;
    uint64_t xZNWorkspaceBytes;
};

BEGIN_TILING_DATA_DEF(QuantMatmulDequantTilingData)

TILING_DATA_FIELD_DEF(uint32_t, CoreNum);
TILING_DATA_FIELD_DEF(uint32_t, perToken);
TILING_DATA_FIELD_DEF(uint32_t, dynamicQuant);
TILING_DATA_FIELD_DEF(uint32_t, smoothScale);

TILING_DATA_FIELD_DEF(uint32_t, originE);
TILING_DATA_FIELD_DEF(uint32_t, originM);
TILING_DATA_FIELD_DEF(uint32_t, originN);
TILING_DATA_FIELD_DEF(uint32_t, originK);
TILING_DATA_FIELD_DEF(uint32_t, originKAligned32);
TILING_DATA_FIELD_DEF(uint32_t, originKAligned512);
TILING_DATA_FIELD_DEF(uint32_t, fracN);
TILING_DATA_FIELD_DEF(uint32_t, fracK);
// dynamic
TILING_DATA_FIELD_DEF(uint32_t, dynamicBaseK);
TILING_DATA_FIELD_DEF(uint32_t, dynamicIterK);
TILING_DATA_FIELD_DEF(uint32_t, dynamicBaseKTail);
// gemv
TILING_DATA_FIELD_DEF(uint32_t, singleCoreFracN);
TILING_DATA_FIELD_DEF(uint32_t, singleCoreFracNTail);
TILING_DATA_FIELD_DEF(uint32_t, baseFracN);
TILING_DATA_FIELD_DEF(uint32_t, baseFracK);
TILING_DATA_FIELD_DEF(uint32_t, baseFracNL0C);
TILING_DATA_FIELD_DEF(uint32_t, ubBaseK);
TILING_DATA_FIELD_DEF(uint32_t, ubIterK);
TILING_DATA_FIELD_DEF(uint32_t, ubBaseKTail);
// normal
TILING_DATA_FIELD_DEF(uint32_t, fracM);
TILING_DATA_FIELD_DEF(uint32_t, tailM);
TILING_DATA_FIELD_DEF(uint32_t, processXKBaseNMax);
TILING_DATA_FIELD_DEF(uint32_t, processXKBaseN);
TILING_DATA_FIELD_DEF(uint32_t, processXKloop);
TILING_DATA_FIELD_DEF(uint32_t, processXKloopPerfracM);
TILING_DATA_FIELD_DEF(uint32_t, processXKTailN);
TILING_DATA_FIELD_DEF(uint32_t, MMmod);
TILING_DATA_FIELD_DEF(uint32_t, MCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, NCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, singleCoreM);
TILING_DATA_FIELD_DEF(uint32_t, singleCoreN);
TILING_DATA_FIELD_DEF(uint32_t, singleCoreMTail);
TILING_DATA_FIELD_DEF(uint32_t, singleCoreNTail);
TILING_DATA_FIELD_DEF(uint32_t, baseMNum);
TILING_DATA_FIELD_DEF(uint32_t, baseNNum);
TILING_DATA_FIELD_DEF(uint32_t, baseKNum);
// swift
TILING_DATA_FIELD_DEF(uint32_t, swiftGEMVThreshold);
TILING_DATA_FIELD_DEF(uint32_t, baseNNum_2);
TILING_DATA_FIELD_DEF(uint32_t, baseKNum_2);
TILING_DATA_FIELD_DEF(uint32_t, baseK);
TILING_DATA_FIELD_DEF(uint32_t, baseKTail);
TILING_DATA_FIELD_DEF(uint32_t, baseK_2);
TILING_DATA_FIELD_DEF(uint32_t, baseKTail_2);

TILING_DATA_FIELD_DEF(uint64_t, ubKMask);
TILING_DATA_FIELD_DEF(uint32_t, isXScaleHalf);
// Fields consumed only by the optimized dynamic per-token grouped kernel path.
TILING_DATA_FIELD_DEF(uint32_t, coreNum);
TILING_DATA_FIELD_DEF(uint32_t, M);
TILING_DATA_FIELD_DEF(uint32_t, K);
TILING_DATA_FIELD_DEF(uint32_t, N);
TILING_DATA_FIELD_DEF(uint32_t, E);
TILING_DATA_FIELD_DEF(uint32_t, Mb);
TILING_DATA_FIELD_DEF(uint32_t, Nb);
TILING_DATA_FIELD_DEF(uint32_t, Kb);
TILING_DATA_FIELD_DEF(uint32_t, kPasses);
TILING_DATA_FIELD_DEF(uint32_t, wChunkKPasses);
TILING_DATA_FIELD_DEF(uint32_t, phaseXChunk);
TILING_DATA_FIELD_DEF(uint32_t, phaseXFullPairs);
TILING_DATA_FIELD_DEF(uint32_t, phaseXTail);
TILING_DATA_FIELD_DEF(uint32_t, maxOneTurnToken);
TILING_DATA_FIELD_DEF(uint32_t, maxWeightColOneTurn);
TILING_DATA_FIELD_DEF(uint32_t, perCoreIters);
TILING_DATA_FIELD_DEF(uint32_t, remainderIters);
TILING_DATA_FIELD_DEF(uint32_t, weightBlockN1);
TILING_DATA_FIELD_DEF(uint32_t, nBlockSize);
TILING_DATA_FIELD_DEF(uint32_t, flags);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(QuantGroupedMatmulDequant, QuantMatmulDequantTilingData)

class QuantMatmulDequantTiling {
public:
    ge::graphStatus runTiling(gert::TilingContext *context, bool is_grouped = false);

private:
    bool GetPlatformInfo(gert::TilingContext *context);

    bool GetCheckAttr(gert::TilingContext *context);

    bool CheckInOutShapes(gert::TilingContext *context);

    bool GetTilingData(gert::TilingContext *context);

    bool GetOptimizedPertokenDynamicTilingData(gert::TilingContext *context);

    bool SetTilingData(gert::TilingContext *context);

    bool SetLaunchInfo(gert::TilingContext *context);

    bool isGrouped;
    QuantMatmulDequantTilingKey tilingKey;
    QuantMatmulDequantTilingData tilingData;
    QuantMatmulDequantParam _Params;
};

} // namespace optiling
#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_QUANT_MATMUL_DEQUANT_H
