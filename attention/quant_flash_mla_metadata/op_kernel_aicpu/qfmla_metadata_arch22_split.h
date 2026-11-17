/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QFMLA_METADATA_ARCH22_SPLIT_H
#define QFMLA_METADATA_ARCH22_SPLIT_H

#include <cstdint>
#include <string>
#include <vector>

#include "qfmla_metadata_arch22_layout.h"

namespace aicpu {
namespace arch22 {

struct Arch22ScheduleInput {
    std::vector<int64_t> cacheSeqlens;
    std::vector<int64_t> cuSeqlensQ;
    std::vector<int64_t> sequsedQ;
    int32_t batchSize = 0;
    int32_t numHeadsQ = 0;
    int32_t numHeadsKv = 1;
    int32_t maxSeqlenQ = -1;
    int32_t maxSeqlenKv = -1;
    int32_t maskMode = 0;
    int32_t headDimQk = 576;
    std::string layoutQ = "TND"; // layout_q: BSH/BSND/TND
    std::string socVersion = "Ascend910B";
    uint32_t aicCoreNum = 24;
    uint32_t aivCoreNum = 48;
};

struct Arch22FaCoreRecord {
    uint32_t sIdxStart = 0;
    uint32_t sIdxEnd = 0;
    uint32_t bEnd = 0;
    uint32_t s1OuterEnd = 0;
    uint32_t s2End = 0;
    uint32_t startKvSplitPos = 0;
};

struct Arch22FdRecord {
    uint32_t bIdx = 0;
    uint32_t s1Idx = 0;
    uint32_t kvSplitNum = 0;
    uint32_t s1SplitNum = 0;
    uint32_t s1LastPartSize = 0;
    uint32_t s1IdxEnd = 0;
    uint32_t s1IdxEndSplit = 0;
    uint32_t workspaceIdx = 0;
};

struct Arch22ScheduleResult {
    uint32_t balanceMode = 0;
    bool splitKVFlag = false;
    uint32_t kvSplitPart = 0;
    uint32_t fdBalanceFlag = 0;
    uint32_t sInnerSize = 0;
    uint32_t sInnerSizeAlign = 0;
    uint32_t sInnerLoopTimes = 0;
    uint32_t sInnerSizeTail = 0;
    uint32_t s1SplitSize = 0;
    uint32_t groupSplitSize = 0;
    uint32_t gOuter = 1;
    uint32_t s1Outer = 1;
    uint32_t usedCoreNum = 0;
    uint32_t sInnerLoopSize = 0;
    uint32_t maxActualseq = 0;
    bool hasZeroActualseq = false;
    bool isSameActualseq = true;
    uint64_t qSeqSquareSum = 0;
    uint32_t qSeqSize = 1;
    uint32_t tSeqSize = 0;
    uint32_t seqSize = 0;
    uint32_t coreNum = 0;
    uint32_t actualLenQDims = 0;
    uint32_t actualLenDims = 0;
    int32_t maskMode = 0; // mask_mode
    uint32_t blockSize = 128;
    uint32_t numHeadsQ = 0;
    uint32_t numHeadsKv = 1;
    uint32_t nNumOfQInOneGroup = 1;
    uint32_t batchSize = 0;
    uint32_t outputLayout = 0;
    uint32_t kvLayout = 0;
    uint32_t needInitOutput = 0;
    std::vector<Arch22FaCoreRecord> faRecords;
    uint32_t tndFDCoreArrLen = 0;
    uint32_t usedVecNumOfFd = 0;
    std::vector<Arch22FdRecord> fdRecords;
};

class Arch22IfaSplit {
public:
    bool Run(const Arch22ScheduleInput& input, Arch22ScheduleResult& result);

private:
    enum class IfaLayout : uint32_t {
        BSH_BSND = 0,
        BNSD = 1,
        NZ = 2,
        TND = 3,
        NBSD = 4,
        NTD = 5
    };
    enum class IfaPerfMode : uint32_t {
        NORMAL = 0,
        BMM_ALL_BY_VEC,         // 1
        C1_V1,                  // 2
        CUBE_VIEW_MM,           // 3
        CUBE_VIEW_MM_FULL_LOAD, // 4
        CUBE_VIEW_MM_MLA,       // 5
        CUBE_VIEW_MM_DD         // 6
    };
    enum class IfaSocVersion : uint32_t {
        SOC_ASCEND_910B = 0,
        SOC_ASCEND_310P = 1,
        SOC_ASCEND_950 = 2
    };
    enum class IfaAmlaMode : uint32_t {
        DISABLE_AMLA = 0,
        AMLA = 1,
        AMLA_3BUF = 2
    };

    struct ValidityConfigFD {
        std::vector<int32_t> validBatchSizes;
        std::vector<int32_t> validQSeqSizes;
        int32_t numHeads = 0;
        int32_t numKvHeads = 0;
        int32_t headDim = 0;
        int32_t headDimV = 0;
        int32_t sparseMode = 0;
        int64_t expectedActualSeqLength = 0;
    };

    struct ActualSeqInfo {
        std::vector<int64_t> actualSeqQ;
        int64_t maxActualseqkv = 0;
        ActualSeqInfo(uint32_t bSize, int64_t maxVal)
            : actualSeqQ(bSize),
              maxActualseqkv(maxVal)
        {}
    };
    struct SeqTilingInfo {
        std::vector<uint32_t> s1OuterNum;
        std::vector<uint32_t> s2OuterNum;
        uint32_t lastValidBIdx = 0U;
        uint64_t avgS2Length = 1U;
        explicit SeqTilingInfo(uint32_t bSize)
            : s1OuterNum(bSize),
              s2OuterNum(bSize)
        {}
    };
    struct BalancedSplitTilingInfo {
        std::vector<uint64_t> coreLoad;
        uint64_t accumS2Length = 0U;
        uint32_t currCoreIdx = 0U;
        bool needUpdate = false;
        uint32_t maxKvSplitPart = 0U;
        uint32_t tndFDCoreArrLen = 0U;
        explicit BalancedSplitTilingInfo(uint32_t coreNum)
            : coreLoad(coreNum)
        {}
    };
    struct TilingIndexes {
        uint32_t bIdx = 0U;
        uint32_t s1Idx = 0U;
        uint32_t s2Idx = 0U;
        TilingIndexes() = default;
        TilingIndexes(uint32_t b, uint32_t s1, uint32_t s2)
            : bIdx(b),
              s1Idx(s1),
              s2Idx(s2)
        {}
    };

    bool ParseInputs(const Arch22ScheduleInput& input);
    bool QKVPreProcess4TND();
    bool ProcessActualSeqLen();
    bool PreProcess();
    bool Split();
    void FillResult(Arch22ScheduleResult& result);

    bool SplitBalanced();
    bool SplitBalancedFd();
    void GetActualSeqInfo(const std::vector<int64_t>& actualSeqKv, ActualSeqInfo& actualSeqInfo) const;
    void GetSeqTilingInfo(const std::vector<int64_t>& actualSeqKv, const ActualSeqInfo& actualSeqInfo,
                          SeqTilingInfo& seqTilingInfo) const;
    bool FillBalancedSplitCoreInfo(const TilingIndexes& tilingIdx, BalancedSplitTilingInfo& tilingInfo);
    void EndSplitForCurrentCore(const TilingIndexes& tilingIdx, const SeqTilingInfo& seqTilingInfo,
                                uint32_t& currKvSplitPart, BalancedSplitTilingInfo& tilingInfo);
    void SplitBalancedForEachHead(uint32_t bIdx, const SeqTilingInfo& seqTilingInfo,
                                  BalancedSplitTilingInfo& tilingInfo);
    void SplitBalancedForEachHeadFd(uint32_t bIdx, const SeqTilingInfo& seqTilingInfo,
                                    BalancedSplitTilingInfo& tilingInfo, std::vector<int64_t>& gS1SplitNumOfFdHead,
                                    uint32_t s1);
    void SplitFDMLa(uint32_t tndFDCoreArrLen, std::vector<int64_t>& gS1SplitNumOfFdHead, uint32_t* s2SplitNumOfFdHead,
                    uint32_t aivCoreNum, SeqTilingInfo& seqTilingInfo);
    bool IsKvZeroBatchSplit(bool needUpdate, uint32_t lastValidBMoreIdx, uint32_t bSize,
                            const std::vector<uint32_t>& s1OuterNum, const std::vector<uint32_t>& s2OuterNum) const;

    bool SplitUnbalanced();
    bool SplitBN();
    bool SplitBN_V0();
    bool SplitBNS();
    bool gqaKvNzFlagForSplit() const;

    void GetEstimatedLoad(int64_t& estimatedLoad) const;
    std::vector<int64_t> InitSparseValidArray(const int64_t* actualLens) const;
    bool BalanceLoad(const std::vector<int64_t>& sparseValidArray, int64_t totalSize, int64_t validAivNum,
                     std::vector<int64_t>& localValue, std::vector<int64_t>& sparseStartIdx) const;
    void InitLoadValue(const std::vector<int64_t>& sparseValidArray, int64_t totalSize, int64_t validAivNum,
                       const std::vector<int64_t>& sparseStartIdx, std::vector<int64_t>& localValue) const;
    void SetSparseStartIdx(const std::vector<int64_t>& sparseValidArray, int64_t totalSize, int64_t validAivNum,
                           uint32_t* sparseStartIdx, int64_t splitFactorSize);

    bool IsBalanceSplitCore();
    void IsFdBalanceCase();
    bool CheckCommonConditions(const ValidityConfigFD& config) const;
    bool CheckBatchAndQSeqSize(const ValidityConfigFD& config) const;
    bool CheckHeadDimensions(const ValidityConfigFD& config) const;
    bool CheckQuantizationFlags(int32_t sparseMode) const;
    bool CheckActualSeqLengths(int64_t expectedActualSeqLength) const;
    bool IsValidFlag();
    bool IsValidFlag3B();
    bool IsValidFlag560B();
    bool IsFlashDecode(uint32_t coreNum, IfaPerfMode perfMode);
    bool CheckCoreOkFlag(uint32_t coreNum, IfaPerfMode perfMode) const;
    bool DealSameSeqEachBatch() const;
    bool HasZeroSeqBatch() const;
    void SetCoreNum();
    void SetupPerfMode();
    IfaAmlaMode GetAmlaMode() const;

    bool CalcInnerSize(uint32_t seqSize);

    // ---- 输入 ----
    std::vector<int64_t> cacheSeqlens_;
    std::vector<int64_t> cuSeqlensQ_;
    std::vector<int64_t> ifaCuSeqQ_;
    std::vector<int64_t> sequsedQ_;
    uint32_t batchSize_ = 0U;
    uint32_t batchSizeQ_ = 0U;
    uint32_t numHeads_ = 0U;
    uint32_t numKvHeads_ = 1U;
    uint32_t headDim_ = 576U;
    uint32_t headDimV_ = 576U;
    int32_t maxSeqlenQ_ = -1;
    int32_t maxSeqlenKv_ = -1;
    int32_t maskMode_ = 0;
    uint32_t blockSize_ = 128U;
    uint32_t aicNum_ = 0U;
    uint32_t aivNum_ = 0U;
    uint32_t cvRatio_ = 1U;
    IfaLayout inputLayout_ = IfaLayout::TND;
    IfaLayout inputKvLayout_ = IfaLayout::NZ;
    IfaLayout outputLayout_ = IfaLayout::TND;
    IfaSocVersion socVersion_ = IfaSocVersion::SOC_ASCEND_910B;
    IfaPerfMode perfMode_ = IfaPerfMode::CUBE_VIEW_MM_MLA;
    bool ropeFlag_ = true;
    bool antiQuantFlag_ = false;
    bool quantFlag_ = true;
    bool gqaMtpFlag_ = false;
    bool slidingFlag_ = false;
    bool pageAttentionFlag_ = true;
    bool batchContinuousFlag_ = true;
    bool isWorkspace_ = false;
    bool sysPrefixFlag_ = false;
    bool isSysPrefixTiling_ = false;
    bool softmaxLseFlag_ = false;
    uint32_t sparseMode_ = 0U;
    bool attenMaskFlag_ = false;
    bool balanceModeFlag_ = false;
    uint32_t fdBalanceFlag_ = 0U;
    bool isSameSeqAllKVTensor_ = true;
    uint32_t qSeqSize_ = 1U;
    uint32_t tSeqSize_ = 0U;
    uint32_t seqSize_ = 0U;
    uint32_t sMax_ = 0U;
    uint32_t actualLenDims_ = 0U;
    uint32_t actualLenQDims_ = 0U;
    uint32_t maxActualseq_ = 0U;
    bool hasZeroActualseq_ = false;
    bool isSameActualseq_ = true;
    bool actualSeqLenFlag_ = false;
    uint64_t qSeqSquareSum_ = 0U;
    uint32_t coreNum_ = 0U;
    uint32_t nNumOfQInOneGroup_ = 1U;
    uint32_t groupSplitSize_ = 0U;
    uint32_t gOuter_ = 1U;
    uint32_t s1Outer_ = 1U;
    uint32_t s1SplitSize_ = 0U;
    uint32_t gMax_ = 128U;
    IfaAmlaMode amlaMode_ = IfaAmlaMode::DISABLE_AMLA;
    uint32_t sInnerSize_ = 0U;
    uint32_t sInnerSizeAlign_ = 0U;
    uint32_t sInnerLoopTimes_ = 0U;
    uint32_t sInnerSizeTail_ = 0U;
    bool splitKVFlag_ = false;
    uint32_t kvSplit_ = 0U;
    uint32_t kvSplitPart_ = 1U;
    uint32_t usedCoreNum_ = 0U;
    uint32_t formerCoreNum_ = 0U;
    uint32_t blockSplitBn2Range_ = 0U;
    uint32_t tailSplitedBatchRange_ = 0U;
    uint32_t sInnerLoopSize_ = 0U;
    static constexpr uint32_t MAX_CORE_NUM = 50U;
    static constexpr uint32_t MAX_AIC_CORE_NUM = 26U;
    static constexpr uint32_t MAX_AIC_CORE_NUM_X2 = 52U;
    uint32_t startIdxEachCore_[MAX_CORE_NUM] = {};
    uint32_t coreBEnd_[MAX_AIC_CORE_NUM] = {};
    uint32_t coreS1OuterEnd_[MAX_AIC_CORE_NUM] = {};
    uint32_t coreS2End_[MAX_AIC_CORE_NUM] = {};
    uint32_t balanceFDCoreBArr_[MAX_AIC_CORE_NUM] = {};
    uint32_t balanceFDCoreS1Arr_[MAX_AIC_CORE_NUM] = {};
    uint32_t balanceFDCoreKVSplitArr_[MAX_AIC_CORE_NUM] = {};
    uint32_t balanceFDCoreStartKVSplitNum_[MAX_AIC_CORE_NUM] = {};
    uint32_t gS1IdxEndOfFdHead_[MAX_AIC_CORE_NUM_X2] = {};
    uint32_t gS1IdxEndOfFdHeadSplit_[MAX_AIC_CORE_NUM_X2] = {};
    uint32_t gS1SplitNumOfFdHeadMla_[MAX_AIC_CORE_NUM_X2] = {};
    uint32_t gS1LastPartSizeOfFdHead_[MAX_AIC_CORE_NUM_X2] = {};
    uint32_t tndFDCoreArrLen_ = 0U;
    uint32_t usedVecNumOfFd_ = 0U;
};

bool Arch22Schedule(const Arch22ScheduleInput& input, Arch22ScheduleResult& result);

bool GenMetaDataArch22(const Arch22ScheduleResult& result, int32_t* metadataPtr, int64_t rowSize, int64_t dimNum);

} // namespace arch22
} // namespace aicpu

#endif
