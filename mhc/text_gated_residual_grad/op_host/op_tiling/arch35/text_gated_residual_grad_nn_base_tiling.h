/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Adapted from ops-nn d1d5c991c199d430f3b825546439799ea2a04f87:
// matmul/mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.cpp.
// This is its ordinary ND base policy for A2/A3, without standalone full-load,
// split-K, AOE repository or L2 partition selectors.
#ifndef HC_GRAD_NN_BASE_TILING_H
#define HC_GRAD_NN_BASE_TILING_H
#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>
namespace HcGradNnBase {
namespace ops {
inline uint64_t CeilAlign(uint64_t a, uint64_t b)
{
    return (a + b - 1) / b * b;
}
inline uint64_t FloorAlign(uint64_t a, uint64_t b)
{
    return a / b * b;
}
inline uint64_t FloorDiv(uint64_t a, uint64_t b)
{
    return a / b;
}
} // namespace ops
struct MathUtil {
    static uint64_t CeilDivision(uint64_t a, uint64_t b)
    {
        return (a + b - 1) / b;
    }
};
enum class CalcType {
    M_BY_BASE_NK,
    MN_BY_BASE_K
};
enum class MatmulV3Trans {
    NO_TRANS,
    A_TRANS,
    B_TRANS,
    AB_TRANS
};
constexpr uint64_t BASIC_ALIGN_16 = 16;
constexpr uint64_t BASIC_ALIGN_256 = 256;
constexpr uint64_t BASIC_ALIGN_512 = 512;
constexpr uint64_t DB_SIZE = 2;
constexpr uint64_t DB_OFF_SIZE = 1;
constexpr uint64_t LOC_DATA_SIZE = 4;
constexpr uint64_t BASIC_BLOCK_SIZE_256 = 256;
constexpr uint64_t BASIC_BLOCK_SIZE_128 = 128;
constexpr uint64_t BASIC_BLOCK_K_128_BYTE = 128;
constexpr uint64_t BIAS_TABLE_NUM = 256;
constexpr uint64_t DATA_SIZE_FP32 = 4;
constexpr uint64_t L0C_SIZE_256_KB = 262144;
constexpr uint64_t NUM_HALF = 2;
constexpr uint64_t BASE_STEP = 1;
constexpr uint64_t ITER_COL_FIRST = 0;
constexpr uint64_t SMALL_SHAPE_THRES = 256;
constexpr uint64_t SMALL_SHAPE_LOWER_THRES = 128;
constexpr uint64_t L0C_THRES = 32768;
constexpr uint64_t BLOCK = 128;
constexpr uint64_t LOW_PRECISION_BYTES = sizeof(uint16_t);
constexpr uint64_t L1_RPC_RESERVE_BYTES = 256;
constexpr uint64_t NARROW_M_THRESHOLD = 64;
constexpr uint64_t NARROW_M_BASE_N_FLOOR = 1024;
constexpr size_t TILE_M_INDEX = 0;
constexpr size_t TILE_N_INDEX = 1;
constexpr size_t TILE_K_INDEX = 2;

struct RunInfo {
    uint64_t baseM = BASIC_BLOCK_SIZE_128, baseN = BASIC_BLOCK_SIZE_256;
    uint64_t baseK = BASIC_BLOCK_K_128_BYTE / LOW_PRECISION_BYTES, usedCoreNum = 1;
    uint64_t stepM = BASE_STEP, stepN = BASE_STEP, iterateOrder = ITER_COL_FIRST, dbL0c = DB_OFF_SIZE;
    uint64_t depthA1 = 1, depthB1 = 1, stepKa = BASE_STEP, stepKb = BASE_STEP;
    uint64_t singleCoreM = BASIC_BLOCK_SIZE_128, singleCoreN = BASIC_BLOCK_SIZE_256, singleCoreK = 1;
};
inline void ResetBase(RunInfo& runInfo, const uint64_t l0CSize, const uint64_t dTypeSize = LOW_PRECISION_BYTES)
{
    runInfo.baseM = (l0CSize == L0C_SIZE_256_KB) ? BASIC_BLOCK_SIZE_256 : BASIC_BLOCK_SIZE_128;
    runInfo.baseN = BASIC_BLOCK_SIZE_256; // 256 is better base
    runInfo.baseK = BASIC_BLOCK_K_128_BYTE / dTypeSize;
    runInfo.stepM = BASE_STEP;
    runInfo.stepN = BASE_STEP;
    runInfo.iterateOrder = ITER_COL_FIRST;
    runInfo.dbL0c = DB_OFF_SIZE;
}
inline uint64_t CalBaseSize(uint64_t cnt, uint64_t totalCoreNum, uint64_t size, uint64_t maxBase)
{
    uint64_t curCnt = 0UL;
    if (cnt == 0UL) {
        curCnt = 1UL;
    } else {
        curCnt = totalCoreNum / cnt;
        curCnt = std::max(curCnt, 1UL);
    }
    uint64_t base = ops::CeilAlign(std::max(size / curCnt, 1UL), BASIC_ALIGN_16);
    base = std::min(base, maxBase);
    return base;
}
class Policy {
public:
    RunInfo Make(uint64_t m, uint64_t n, uint64_t k, bool transA, uint64_t cores, uint64_t l1, uint64_t l0a,
                 uint64_t l0b, uint64_t l0c)
    {
        args_.mValue = m;
        args_.nValue = n;
        args_.kValue = k;
        args_.isATrans = transA;
        compileInfo_.aicNum = cores;
        compileInfo_.l1Size = l1 - L1_RPC_RESERVE_BYTES;
        compileInfo_.l0ASize = l0a;
        compileInfo_.l0BSize = l0b;
        compileInfo_.l0CSize = l0c;
        trans_ = transA ? MatmulV3Trans::A_TRANS : MatmulV3Trans::NO_TRANS;
        m256Align_ = (m * aDtypeSize_ % BASIC_ALIGN_256) == 0;
        n256Align_ = (n * bDtypeSize_ % BASIC_ALIGN_256) == 0;
        kA256Align_ = (k * aDtypeSize_ % BASIC_ALIGN_256) == 0;
        kB256Align_ = kA256Align_;
        ResetBase(runInfo_, l0c);
        runInfo_.singleCoreK = k;
        SetBaseBlockTiling();
        const bool small = (ops::CeilAlign(m, BASIC_BLOCK_SIZE_128) / runInfo_.baseM) *
                               (ops::CeilAlign(n, BASIC_BLOCK_SIZE_256) / runInfo_.baseN) <
                           cores;
        if (small || m < SMALL_SHAPE_THRES || n < SMALL_SHAPE_THRES)
            DoSmallShapeTiling();
        CalL1Tiling();
        return runInfo_;
    }

private:
    struct Args {
        uint64_t mValue = 1, nValue = 1, kValue = 1;
        bool isATrans = false, isBTrans = false, hasBias = false;
    } args_;
    struct CompileInfo {
        uint64_t aicNum = 1, l1Size = 524032, l0ASize = 65536, l0BSize = 65536, l0CSize = 131072, btSize = 1024;
        bool supportL0c2out = true;
    } compileInfo_;
    RunInfo runInfo_;
    uint64_t aDtypeSize_ = LOW_PRECISION_BYTES, bDtypeSize_ = LOW_PRECISION_BYTES;
    uint64_t basicBlockBaseM_ = BASIC_BLOCK_SIZE_128;
    bool m256Align_ = false, n256Align_ = false, kA256Align_ = false, kB256Align_ = false;
    MatmulV3Trans trans_ = MatmulV3Trans::NO_TRANS;
    std::vector<std::vector<uint64_t>> calcMBasic_{{128, 256, 128}, {128, 128, 256}, {64, 512, 64}};
    std::vector<std::vector<uint64_t>> calcMNBasic_{{64, 64, 512}, {128, 128, 256}, {128, 256, 128}};
    bool CheckBTSize(uint64_t)
    {
        return true;
    } // This fusion has no MatMul bias.
    void SetBaseBlockTiling();
    void DoSmallShapeTiling();
    void FormulaicTilingNoTrans();
    void FormulaicBaseBlockTiling();
    void BalanceBaseBlockTiling();
    template <CalcType T>
    void CalcBase(const std::vector<std::vector<uint64_t>>&);
    void CalBaseMBaseN(uint64_t&, uint64_t&, uint64_t = BASIC_BLOCK_SIZE_128, uint64_t = BASIC_BLOCK_SIZE_256);
    void CalL1Tiling();
    void UpdateL1TilingStepK(uint64_t&);
};
void Policy::SetBaseBlockTiling()
{
    runInfo_.usedCoreNum = compileInfo_.aicNum;
    runInfo_.baseM = BASIC_BLOCK_SIZE_128;
    runInfo_.baseN = BASIC_BLOCK_SIZE_256;
    runInfo_.baseK = BASIC_BLOCK_K_128_BYTE / aDtypeSize_;
    if (compileInfo_.l0CSize == L0C_SIZE_256_KB) {
        runInfo_.baseM = BASIC_BLOCK_SIZE_256;
        return;
    }
    if (!args_.isATrans && !args_.isBTrans) {
        return;
    }
    if (args_.isATrans && args_.isBTrans) {
        runInfo_.baseM = BASIC_BLOCK_SIZE_256;
        runInfo_.baseN = BASIC_BLOCK_SIZE_128;
        return;
    }
    if (false) {
        return;
    }
    const uint64_t largeTileAxisThreshold = (compileInfo_.aicNum / NUM_HALF) * BASIC_BLOCK_SIZE_256;
    const bool mEnableLarge = args_.mValue >= largeTileAxisThreshold;
    const bool nEnableLarge = args_.nValue >= largeTileAxisThreshold;
    if (mEnableLarge && nEnableLarge) {
        uint64_t sizeMLarge = MathUtil::CeilDivision(args_.mValue, BASIC_BLOCK_SIZE_256) * args_.nValue +
                              MathUtil::CeilDivision(args_.nValue, BASIC_BLOCK_SIZE_128) * args_.mValue;
        uint64_t sizeNLarge = MathUtil::CeilDivision(args_.mValue, BASIC_BLOCK_SIZE_128) * args_.nValue +
                              MathUtil::CeilDivision(args_.nValue, BASIC_BLOCK_SIZE_256) * args_.mValue;
        if (sizeMLarge < sizeNLarge) {
            runInfo_.baseM = BASIC_BLOCK_SIZE_256;
            runInfo_.baseN = BASIC_BLOCK_SIZE_128;
        }
        return;
    }
    if (mEnableLarge && !nEnableLarge) {
        runInfo_.baseM = BASIC_BLOCK_SIZE_256;
        runInfo_.baseN = BASIC_BLOCK_SIZE_128;
        return;
    }
    if (!mEnableLarge && !nEnableLarge) {
        if (args_.mValue > args_.nValue) {
            runInfo_.baseM = BASIC_BLOCK_SIZE_256;
            runInfo_.baseN = BASIC_BLOCK_SIZE_128;
        }
    }
    return;
}
void Policy::DoSmallShapeTiling()
{
    FormulaicBaseBlockTiling();
    if (!(compileInfo_.l0CSize == L0C_SIZE_256_KB)) {
        if (args_.mValue < SMALL_SHAPE_THRES && args_.mValue > SMALL_SHAPE_LOWER_THRES &&
            runInfo_.baseN > SMALL_SHAPE_LOWER_THRES) {
            runInfo_.baseN = SMALL_SHAPE_LOWER_THRES;
        } else if (args_.nValue < SMALL_SHAPE_THRES && args_.nValue > SMALL_SHAPE_LOWER_THRES &&
                   runInfo_.baseM > SMALL_SHAPE_LOWER_THRES) {
            runInfo_.baseM = SMALL_SHAPE_LOWER_THRES;
        } else if (runInfo_.baseM * runInfo_.baseN > L0C_THRES) {
            runInfo_.baseM = runInfo_.baseM > SMALL_SHAPE_LOWER_THRES && runInfo_.baseM < SMALL_SHAPE_THRES ?
                                 SMALL_SHAPE_LOWER_THRES :
                                 runInfo_.baseM;
            runInfo_.baseN = runInfo_.baseN > SMALL_SHAPE_LOWER_THRES && runInfo_.baseN < SMALL_SHAPE_THRES ?
                                 SMALL_SHAPE_LOWER_THRES :
                                 runInfo_.baseN;
        }
        if (runInfo_.baseM * runInfo_.baseN > L0C_THRES) {
            runInfo_.baseM = L0C_THRES / runInfo_.baseN;
        }
    }
}
void Policy::FormulaicTilingNoTrans()
{
    uint64_t nCore = MathUtil::CeilDivision(args_.nValue, runInfo_.baseN); // Number of base blocks along N.
    CalcBase<CalcType::M_BY_BASE_NK>(calcMBasic_);
    // Handle unaligned operands.
    if (!kA256Align_ && !n256Align_) {
        // Convert both A and B from ND to NZ and choose base M, N, and K for load balance.
        BalanceBaseBlockTiling();
    } else if (!kA256Align_) {
        // Convert A from ND to NZ. Align baseN * depth to 256 bytes and allow any baseM split.
        runInfo_.baseN = BASIC_BLOCK_SIZE_256;
        runInfo_.baseK = BASIC_BLOCK_SIZE_128 / aDtypeSize_;
        runInfo_.baseM = CalBaseSize(nCore, compileInfo_.aicNum, args_.mValue, basicBlockBaseM_);
    } else if (!n256Align_) {
        // Convert B from ND to NZ. Align baseK * depth to 256 bytes and allow any baseM and baseN splits.
        runInfo_.baseK = BASIC_BLOCK_SIZE_128 / aDtypeSize_;
        CalBaseMBaseN(runInfo_.baseM, runInfo_.baseN);
    }
}
void Policy::FormulaicBaseBlockTiling()
{
    uint64_t mCore = MathUtil::CeilDivision(args_.mValue, runInfo_.baseM); // Number of base blocks along M.
    uint64_t nCore = MathUtil::CeilDivision(args_.nValue, runInfo_.baseN); // Number of base blocks along N.

    // Use ND directly when the contiguous dimensions are aligned to 256 bytes.
    switch (trans_) {
        case MatmulV3Trans::NO_TRANS: {
            FormulaicTilingNoTrans();
            break;
        }
        case MatmulV3Trans::A_TRANS: {
            // For [K, M] and [K, N], align M and N to 256 bytes. K can be split freely.
            // If aligned operands already saturate MTE2, unaligned operands put more pressure on MTE2. Extra cores will
            // not help.
            if (!m256Align_ && !n256Align_) {
                // Convert both A and B from ND to NZ and choose base M, N, and K for load balance.
                BalanceBaseBlockTiling();
            } else if (!m256Align_) {
                // Convert A from ND to NZ. Align baseN * depth to 256 bytes and allow any baseM split.
                runInfo_.baseM = CalBaseSize(nCore, compileInfo_.aicNum, args_.mValue, basicBlockBaseM_);
            } else if (!n256Align_) {
                runInfo_.baseN = CalBaseSize(mCore, compileInfo_.aicNum, args_.nValue, BASIC_BLOCK_SIZE_256);
            }
            break;
        }
        case MatmulV3Trans::B_TRANS: {
            // For [M, K] and [N, K], align K to 256 bytes. M and N can be split freely.
            // Distribute work evenly.
            CalcBase<CalcType::MN_BY_BASE_K>(calcMNBasic_);
            if (!kA256Align_) {
                // Convert both A and B from ND to NZ and choose base M, N, and K for load balance.
                BalanceBaseBlockTiling();
            }
            break;
        }
        case MatmulV3Trans::AB_TRANS: {
            // For [K, M] and [N, K], align M and K to 256 bytes. N can be split freely.
            runInfo_.baseN = CalBaseSize(mCore, compileInfo_.aicNum, args_.nValue, BASIC_BLOCK_SIZE_256);
            if (!m256Align_ && kB256Align_) {
                // Convert both A and B from ND to NZ and choose base M, N, and K for load balance.
                BalanceBaseBlockTiling();
            } else if (!m256Align_) {
                CalBaseMBaseN(runInfo_.baseM, runInfo_.baseN);
            }
            break;
        }
        default:
            break;
    }
    // When baseN >= 512, the ND-to-NZ vmuls dstRepeatStride exceeds 255. Only single-repeat instructions can be used,
    // which lowers efficiency.
    if (true && !compileInfo_.supportL0c2out) {
        runInfo_.baseN = std::min(runInfo_.baseN, BASIC_BLOCK_SIZE_256);
    }
    return;
}
void Policy::BalanceBaseBlockTiling()
{
    ResetBase(runInfo_, compileInfo_.l0CSize, aDtypeSize_);
    uint64_t nCore = 1;
    uint64_t mCore = 1;
    if (args_.mValue >= basicBlockBaseM_) {
        mCore = std::max(MathUtil::CeilDivision(args_.mValue, basicBlockBaseM_), 1UL);
        nCore = std::max(compileInfo_.aicNum / mCore, 1UL);
        runInfo_.baseN = ops::CeilAlign(MathUtil::CeilDivision(args_.nValue, nCore), BASIC_ALIGN_16);
        runInfo_.baseN = std::min(runInfo_.baseN, BASIC_BLOCK_SIZE_256);
        uint64_t baseKa = compileInfo_.l0ASize / DB_SIZE / aDtypeSize_ / BASIC_BLOCK_SIZE_256;
        uint64_t baseKb = ops::FloorDiv(compileInfo_.l0BSize, DB_SIZE * bDtypeSize_ * runInfo_.baseN);
        runInfo_.baseK = ops::FloorAlign(std::min(baseKa, baseKb), BASIC_ALIGN_16);
    } else {
        const uint64_t minBaseN = args_.mValue >= NARROW_M_THRESHOLD ? basicBlockBaseM_ : NARROW_M_BASE_N_FLOOR;
        runInfo_.baseM = ops::CeilAlign(args_.mValue, BASIC_ALIGN_16);
        uint64_t nTile = std::max(args_.nValue / compileInfo_.aicNum, 1UL);
        runInfo_.baseN = ops::CeilAlign(nTile, BASIC_ALIGN_16);
        runInfo_.baseN = std::max(runInfo_.baseN, minBaseN);
        uint64_t maxBaseN = ops::FloorDiv(compileInfo_.l0CSize, LOC_DATA_SIZE * runInfo_.baseM);
        maxBaseN = ops::FloorAlign(maxBaseN, BASIC_ALIGN_16);
        if (args_.hasBias && compileInfo_.supportL0c2out) {
            maxBaseN = std::min(maxBaseN, compileInfo_.btSize / DATA_SIZE_FP32);
        }
        runInfo_.baseN = std::min(runInfo_.baseN, maxBaseN);
        nCore = MathUtil::CeilDivision(args_.nValue, runInfo_.baseN);
        uint64_t tailCoreNum = 0;
        if (runInfo_.baseN > BASIC_BLOCK_SIZE_256) {
            tailCoreNum = MathUtil::CeilDivision(args_.nValue, BASIC_BLOCK_SIZE_256) % compileInfo_.aicNum;
        }
        if (nCore % compileInfo_.aicNum < tailCoreNum) {
            runInfo_.baseN = BASIC_BLOCK_SIZE_256;
        }
        uint64_t baseKa = ops::FloorDiv(compileInfo_.l0ASize, DB_SIZE * aDtypeSize_ * runInfo_.baseM);
        uint64_t baseKb = ops::FloorDiv(compileInfo_.l0BSize, DB_SIZE * bDtypeSize_ * runInfo_.baseN);
        runInfo_.baseK = ops::FloorAlign(std::min(baseKa, baseKb), BASIC_ALIGN_16);
    }

    if (runInfo_.baseM <= 0 || runInfo_.baseN <= 0 || runInfo_.baseK <= 0 || runInfo_.baseM % BASIC_ALIGN_16 != 0 ||
        runInfo_.baseN % BASIC_ALIGN_16 != 0 || runInfo_.baseK % BASIC_ALIGN_16 != 0) {
        ResetBase(runInfo_, compileInfo_.l0CSize, aDtypeSize_);
    }
    return;
}
template <CalcType T>
void Policy::CalcBase(const std::vector<std::vector<uint64_t>>& baseMNK)
{
    std::vector<std::vector<uint64_t>> calBaseMNK;
    size_t index = 0;
    uint64_t minLoadSize = std::numeric_limits<uint64_t>::max();
    uint64_t nCnt = 1;
    uint64_t mCnt = 1;
    for (size_t i = 0; i < baseMNK.size(); ++i) {
        std::vector<uint64_t> tempBaseMNK(baseMNK[i].begin(), baseMNK[i].end());
        if (T == CalcType::M_BY_BASE_NK) {
            nCnt = MathUtil::CeilDivision(args_.nValue, baseMNK[i][TILE_N_INDEX]);
            // baseM is the largest M block that fits after allocating cores along N.
            tempBaseMNK[TILE_M_INDEX] = CalBaseSize(nCnt, compileInfo_.aicNum, args_.mValue, baseMNK[i][TILE_M_INDEX]);
            mCnt = MathUtil::CeilDivision(args_.mValue, tempBaseMNK[TILE_M_INDEX]);
            calBaseMNK.push_back(tempBaseMNK);
        } else {
            CalBaseMBaseN(tempBaseMNK[TILE_M_INDEX], tempBaseMNK[TILE_N_INDEX], calcMNBasic_[i][TILE_M_INDEX],
                          calcMNBasic_[i][TILE_N_INDEX]);
            calBaseMNK.push_back(tempBaseMNK);
            mCnt = MathUtil::CeilDivision(args_.mValue, tempBaseMNK[TILE_M_INDEX]);
            nCnt = MathUtil::CeilDivision(args_.nValue, tempBaseMNK[TILE_N_INDEX]);
        }
        if (args_.hasBias && !CheckBTSize(calBaseMNK[i][TILE_N_INDEX])) {
            continue;
        }

        uint64_t tmpTailCore = (mCnt * nCnt) % compileInfo_.aicNum;
        const uint64_t inputTileSpan = tempBaseMNK[TILE_M_INDEX] + tempBaseMNK[TILE_N_INDEX];
        uint64_t tmpLoadSize = inputTileSpan * (mCnt * nCnt / compileInfo_.aicNum);
        tmpLoadSize += (tmpTailCore > 0) ? inputTileSpan : 0;
        if (tmpLoadSize < minLoadSize) {
            index = i;
            minLoadSize = tmpLoadSize;
        }
    }
    runInfo_.baseM = calBaseMNK[index][TILE_M_INDEX];
    runInfo_.baseN = calBaseMNK[index][TILE_N_INDEX];
    runInfo_.baseK = calBaseMNK[index][TILE_K_INDEX] / aDtypeSize_;
}
void Policy::CalBaseMBaseN(uint64_t& baseM, uint64_t& baseN, uint64_t maxBaseM, uint64_t maxBaseN)
{
    // Update the base block size along the axis with more base blocks.
    uint64_t mCore = MathUtil::CeilDivision(args_.mValue, maxBaseM);
    uint64_t nCore = MathUtil::CeilDivision(args_.nValue, maxBaseN);
    if (mCore < nCore) {
        nCore = std::max(compileInfo_.aicNum / mCore, 1UL);
        baseN = ops::CeilAlign(std::max(args_.nValue / nCore, 1UL), BASIC_ALIGN_16);
        if (baseN > maxBaseN) {
            baseN = maxBaseN;
        }
    } else {
        mCore = std::max(compileInfo_.aicNum / nCore, 1UL);
        baseM = ops::CeilAlign(std::max(args_.mValue / mCore, 1UL), BASIC_ALIGN_16);
        if (baseM > maxBaseM) {
            baseM = maxBaseM;
        }
    }

    baseM = ops::CeilAlign(std::min(args_.mValue, baseM), BASIC_ALIGN_16);
    baseN = ops::CeilAlign(std::min(args_.nValue, baseN), BASIC_ALIGN_16);
    return;
}
void Policy::CalL1Tiling()
{
    // The RPC reservation is available to a standalone operator.
    const uint64_t totalL1Size = compileInfo_.l1Size + L1_RPC_RESERVE_BYTES;
    uint64_t reserveBTSize = args_.hasBias ? BIAS_TABLE_NUM * DATA_SIZE_FP32 : 0;
    runInfo_.depthA1 = totalL1Size / NUM_HALF / runInfo_.baseM / runInfo_.baseK / aDtypeSize_; // 2: half of l1
    runInfo_.depthB1 = totalL1Size / NUM_HALF / runInfo_.baseN / runInfo_.baseK / bDtypeSize_; // 2: half of l1

    uint64_t depthASize = runInfo_.depthA1 * runInfo_.baseM * runInfo_.baseK * aDtypeSize_;
    uint64_t depthBSize = runInfo_.depthB1 * runInfo_.baseN * runInfo_.baseK * bDtypeSize_;
    if (depthASize + depthBSize > totalL1Size - reserveBTSize) {
        if (runInfo_.baseM <= runInfo_.baseN) {
            runInfo_.depthA1 = runInfo_.depthA1 / NUM_HALF; // 2: adjust deptch for l1 buffer
        } else {
            runInfo_.depthB1 = runInfo_.depthB1 / NUM_HALF; // 2: adjust deptch for l1 buffer
        }
    }
    runInfo_.stepKa = runInfo_.depthA1 / DB_SIZE;
    runInfo_.stepKb = runInfo_.depthB1 / DB_SIZE;

    UpdateL1TilingStepK(runInfo_.stepKa);
    UpdateL1TilingStepK(runInfo_.stepKb);

    if (runInfo_.stepKa >= runInfo_.stepKb) {
        runInfo_.stepKa = runInfo_.stepKa / runInfo_.stepKb * runInfo_.stepKb;
    } else {
        runInfo_.stepKb = runInfo_.stepKb / runInfo_.stepKa * runInfo_.stepKa;
    }
    runInfo_.depthA1 = runInfo_.stepKa * DB_SIZE; // depth % (stepKa * stepM) == 0
    runInfo_.depthB1 = runInfo_.stepKb * DB_SIZE; // depth % (stepKb * stepN) == 0
    runInfo_.singleCoreM = runInfo_.baseM;
    runInfo_.singleCoreN = runInfo_.baseN;
    return;
}
void Policy::UpdateL1TilingStepK(uint64_t& stepK)
{
    if (stepK * runInfo_.baseK >= args_.kValue) {
        return;
    }

    if (stepK * runInfo_.baseK * aDtypeSize_ > BASIC_ALIGN_512) {
        if ((stepK * runInfo_.baseK * aDtypeSize_ % BASIC_ALIGN_512) != 0 &&
            (BASIC_ALIGN_512 % (runInfo_.baseK * aDtypeSize_)) == 0) {
            while (stepK * runInfo_.baseK * aDtypeSize_ % BASIC_ALIGN_512 != 0 && stepK > 1) {
                stepK--;
            }
        }
    } else if (stepK * runInfo_.baseK * aDtypeSize_ > BASIC_ALIGN_256) {
        if ((stepK * runInfo_.baseK * aDtypeSize_ % BASIC_ALIGN_256) != 0 &&
            (BASIC_ALIGN_256 % (runInfo_.baseK * aDtypeSize_)) == 0) {
            while (stepK * runInfo_.baseK * aDtypeSize_ % BASIC_ALIGN_256 != 0 && stepK > 1) {
                stepK--;
            }
        }
    }
}
} // namespace HcGradNnBase
#endif
