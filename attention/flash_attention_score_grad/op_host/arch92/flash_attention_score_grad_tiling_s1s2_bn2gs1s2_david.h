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
 * \file flash_attention_score_grad_tiling_s1s2_bn2gs1s2_david.h
 * \brief
 */

#pragma once

#include "flash_attention_score_grad_tiling_common_david.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "../../op_kernel/arch92/flash_attention_score_grad_template_tiling_key_david.h"
#include "../../op_kernel/arch92/flash_attention_score_grad_tiling_data_david.h"
#include "op_host/tiling_type.h"

using namespace optiling::fag;
using namespace Ops::Transformer::OpTiling;
namespace optiling {
namespace fag {
enum class ConstAxisTemplateType {
    AlignTo16 = 1,
    AlignTo32 = 2,
    AlignTo64 = 3,
    AlignTo128 = 4,
    AlignTo256 = 5,
    AlignTo768 = 6,

    AlignTo48 = 7,
    AlignTo80 = 8,
    AlignTo96 = 9,
    AlignTo112 = 10,
    AlignTo160 = 11,
    AlignTo192 = 12,
    AlignTo224 = 13,
    Other = 0
};

enum class ConstAxisTemplateNum {
    NUM1 = 1,
    NUM16 = 16,
    NUM32 = 32,
    NUM64 = 64,
    NUM80 = 80,
    NUM96 = 96,
    NUM112 = 112,
    NUM128 = 128,
    NUM114 = 144,
    NUM160 = 160,
    NUM176 = 176,
    NUM192 = 192,
    NUM208 = 208,
    NUM224 = 224,
    NUM240 = 240,
    NUM256 = 256,
    NUM512 = 512,
    NUM768 = 768,
};

enum class SplitAxisEnum {
    BN2GS1S2 = 0,
    BN2 = 1,
    B = 2,
    N2 = 3,
    BN2GS1 = 4,
    BN2S2 = 5,
    NONE = 9
};

enum class SparseType : uint8_t {
    DENSE = 0,
    CASUAL = 1,
    BAND = 2,
    UNSUPPORTED = 3 // 超L2优化暂不支持sparse的场景
};

enum class DeterSparseType : uint32_t {
    NO_DETER = 0,  // 非确定性
    DETER_OLD = 1, // 确定性老实现方案
    DETER_DENSE = 2,
    DETER_CAUSAL = 3,
    DETER_BAND = 4
};

enum class AttenMaskShapeType {
    ATTENMASKBN2GS1S2,
    ATTENMASKBS1S2,
    ATTENMASKS1S2,
    ATTENMASKTT = 99
};

enum class PseLayoutType {
    pseS1S2 = 0,
    pse1S2 = 1,
    pseSlopeBn = 2,
    pseSlopeN = 3
};

struct DeterPrefixData {
    std::vector<int64_t> prefix0 = {0};
    std::vector<int64_t> prefix1 = {0};
    std::vector<int64_t> prefix2 = {0};
    std::vector<int64_t> deterPrefix = {0};
    std::vector<int64_t> deterPrefixAlign = {0};
    std::vector<int64_t> mNewList = {};
    std::vector<int64_t> nNewList = {};
    std::vector<int64_t> pNewList = {};
    std::vector<int64_t> qNewList = {};
};

struct TndBandDeterRoundInfo {
    int64_t lastBatchId = 0;
    uint64_t lastValidRound = 0;
    uint64_t coreFirstBatchLastRound = 0;
    uint64_t coreLastBatchStartRound = 0;
};

struct TndBaseInfo {
    // for swizzle
    uint64_t tndSwizzleS1S2PrefixSum[TND_SWIZZLE_PREFIX_NUM] = {0};
    uint64_t tndSwizzleS1S2AlignPrefixSum[TND_SWIZZLE_PREFIX_NUM] = {0};
    uint64_t tndS2BlockPrefixSum[TND_SWIZZLE_PREFIX_NUM] = {0};
    // new
    uint64_t tndValidBlockPrefixSum[TND_SWIZZLE_PREFIX_NUM] = {0};
    uint64_t totalValidS2TaskNum = 0;
    uint64_t totalValidS2TileNum = 0;
    uint64_t validS2MaxCoreTileNum = 0;
    uint32_t validS2ActiveCoreNum = 0;
    uint32_t columnScheduleMode = TND_VALID_S2_COLUMN_SIMPLE_CYCLIC;
    bool validS2ColumnEligible = false;
    bool isTndSwizzle = false;

    // no swizzle
    uint64_t tndStartBIdx[CORE_LIST_NUM] = {0};
    uint64_t tndS1S2PrefixSum[CORE_LIST_NUM] = {0};
    uint64_t tndS1S2AlignPrefixSum[CORE_LIST_NUM] = {0};
    uint64_t tndPrefixSum[CORE_LIST_NUM] = {0};
    uint64_t tndPrefixValidSum[CORE_LIST_NUM] = {0};

    bool isS1GreaterThanS2 = true;
    bool isS1LessThanS2 = true;
    bool isSeqExistZero = false;
};

constexpr uint32_t ARRAY_LENGTH = 3;
constexpr uint32_t DETER_LENGTH = 4;
struct FuzzyBaseInfoParamsDavid { // 频繁使用的基础参数
    uint64_t coreNum;
    uint64_t aicNum;
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0aSize;
    uint64_t l0cSize;
    uint64_t l2CacheSize;

    int64_t b;
    int64_t s1;
    int64_t n1;
    int64_t n2;
    int64_t s2;
    int64_t g;
    int64_t d;
    int64_t d1; // head dim of value
    int64_t ropeD = 0;

    int64_t s1Outer;
    int64_t s1Inner;
    int64_t s1CvInner;
    int64_t s1Tail;
    int64_t s1CvTail;

    int64_t s2Outer;
    uint32_t s1CvRatio = 1;
    uint32_t s2CvRatio = 1;
    int64_t cvS2Inner;
    int64_t s2Inner;
    int64_t s2Tail;
    int64_t s2CvTail;

    uint32_t sfmgdInner;
    int64_t t1 = 0;
    int64_t t2 = 0;
    int64_t sumS1S2Product = 0;

    uint32_t queryType;
    uint32_t pseOptional;
    uint32_t pseType = static_cast<uint32_t>(PseType::PSE_OUTER_ADD_MUL_TYPE);
    uint32_t pseShapeType = 0;
    uint32_t pseLayoutType = 0;
    uint32_t pseDtype = 0;
    uint32_t attenMaskOptional;
    uint32_t attenMaskShapeType = 0;
    uint32_t attenMaskDtype = 0;
    uint32_t attenMaskCompressMode;
    uint32_t attenMaskS1Size = 0;
    uint32_t attenMaskS2Size = 0;
    uint32_t dropoutIsDivisibleBy8 = 0;
    uint32_t layoutType;
    float scaleValue;
    float keepProb;
    uint32_t bandIdx;
    int64_t seed;
    int64_t offset;
    DtypeEnum outDtype;
    int64_t keepProbUint8;
    uint32_t sinkOptional = 0;
    uint64_t sinkSize = 0;
    uint64_t s1SinkOuter = 0;
    uint64_t s2SinkOuter = 0;
    bool sValueZeroUnderTND = false;
    uint64_t tailZeroCount = 0;

    uint32_t calTypeSize;
    int64_t s1Token;
    int64_t s2Token;
    uint32_t blockOuter;
    int64_t blockFactor;
    int64_t maxValidBBLen = 0;
    bool noNeedDeter = true;
    uint64_t dqIsNeedDeter[32];
    uint64_t dkDvIsNeedDeter[32];

    uint64_t qSize;
    uint64_t kSize;
    uint64_t vSize;
    uint64_t qRopeSize = 0;
    uint64_t kRopeSize = 0;
    uint64_t dropMaskSize;
    uint8_t dropMaskOuter;

    int64_t blockStarts[CORE_LIST_NUM];
    int64_t blockEnds[CORE_LIST_NUM];
    uint32_t sparseMode;
    uint32_t prefixN[BATCH_MAX_SIZE] = {0};

    std::vector<int64_t> actualSeqQlen;
    std::vector<int64_t> actualSeqKvlen;

    bool isSparse;
    uint32_t tmpBufferSize = 0;

    DtypeEnum inputDtype;
    bool isDeterministic = false;
    bool isBn2 = false;
    bool isBn2MultiBlk = false;
    bool hasRope = false;
    bool isAllSame = false;
    bool isInvalidCol = false;
    bool isInvalidRow = false;
    uint32_t deterSparseType;
    uint8_t sparseType;
    bool isS1S2Same = false;
    bool coreDivide = false;
    int64_t deterMaxRound = 0;
    // 每个 batch 的前缀面积总和 prefix,
    // 小于128b传完整的前缀和，大于128b的，按步长传部分前缀和，在kernel内组装完整的前缀和
    int64_t deterPrefixThreshold = 128;
    int64_t deterPrefixStep = 1;
    int64_t deterPrefix[132] = {0};
    int64_t deterPrefixAlign[132] = {0};
    int64_t deterPrefix0[132] = {0};
    int64_t deterPrefix1[132] = {0};
    int64_t deterPrefix2[132] = {0};
    // 确定性计算需要全核同步的轮次
    uint64_t startNeedSyncRound[CORE_LIST_NUM];
    uint64_t endNeedSyncRound[CORE_LIST_NUM];
    int64_t separateDkOffset[CORE_LIST_NUM];

    SplitAxisEnum splitAxis = SplitAxisEnum::BN2GS1S2;
    ConstAxisTemplateNum s1TemplateType = ConstAxisTemplateNum::NUM128;
    ConstAxisTemplateNum s2TemplateType = ConstAxisTemplateNum::NUM128;
    ConstAxisTemplateNum dTemplateType = ConstAxisTemplateNum::NUM64;

    int64_t qStartIdx;
    int64_t kvStartIdx;
    uint32_t tndMaxSumLayout = 0; // 0: BNS8, 1: TND

    // swizzle
    int64_t segmentS1BlockNum = 1;
    bool enableSwizzle = false;
    bool isSwizzle = false;
};

class FlashAttentionScoreGradTilingUs1s2Bs2David : public TilingBaseClass {
public:
    explicit FlashAttentionScoreGradTilingUs1s2Bs2David(gert::TilingContext* context_)
        : TilingBaseClass(context_)
    {}
    ~FlashAttentionScoreGradTilingUs1s2Bs2David() {}
    FlashAttentionScoreGradS1S2BNGS1S2BaseParamsDavid* s1s2BNGS1S2BaseParams_ = nullptr;
    FlashAttentionScoreGradS1S2BNGS1S2SplitCoreParamsDavid* s1s2BNGS1S2SplitCoreParams_ = nullptr;
    BlockNumListParamsDavid* s1s2BNGS1S2BlockNumList_ = nullptr;
    PreParamsDavid* preTilingData_ = nullptr;
    PostParamsDavid* postTilingData_ = nullptr;
    PreSfmgParamsDavid* preSfmgTilingData_ = nullptr;
    DeterParamDavid* deterParam_ = nullptr;
    TndSwizzleParamDavid* tndSwizzleParam_ = nullptr;

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

    void DoSplit();
    virtual ge::graphStatus DoSparse();
    ge::graphStatus InitTilingData();
    void GetIsDeterArr();
    bool IsValid(int64_t blockIdx);
    void GetOffset(int64_t& currentDqOffset, int64_t& currentDkDvOffset, int64_t blockIdx);
    void JudgeIsNeedDeter(std::array<int64_t, CORE_LIST_NUM>& dqOffset, std::array<int64_t, CORE_LIST_NUM>& dkDvOffset,
                          std::array<int64_t, CORE_LIST_NUM>& dqOffsetpre,
                          std::array<int64_t, CORE_LIST_NUM>& dkDvOffsetpre, int64_t calcNum, bool& noNeedDeter,
                          bool& dqNeedDeterpre, bool& dkDvNeedDeterpre);
    bool CheckFuzzyArgsLegal(uint32_t s1Inner, uint32_t s2Inner);
    std::tuple<uint32_t, uint32_t, uint32_t> FuzzyForBestSplit();
    virtual ge::graphStatus GetSparseBlockInfo();
    void DoPreTiling();
    void DoPostTiling();
    void DoPreSfmgTiling();
    void DetermineMode();
    void TransTilingSplitMode(uint8_t mode, int64_t& m, int64_t& n);
    bool SupportTNDBns2(DeterPrefixData& deterPrefixData, int64_t round);
    void CalcleActualToken(int64_t batchIdx, int64_t& actualCalcS1Token, int64_t& actualCalcS2Token);
    uint32_t GetDeterSparseTilingKey();
    uint8_t GetSparseType();
    bool IsNewDeter();
    bool SetPrefixSparseParams();
    void CalcleDeterParam();
    void CalcleCausalDeterParam();
    void CalcleBandDeterParam();
    void CalcleTNDDeterParam();
    void CalcleTNDBandDeterParam();

    void CalcleTNDCausalDeterPrefix(DeterPrefixData& deterPrefixData, int64_t& m0Max, int64_t& m1Max, int64_t& m2Max);
    void CalcleTNDCausalDeterParam();
    void CalcleTNDCausalDeterParamNormal(DeterPrefixData& deterPrefixData, const int64_t m0Max, const int64_t m1Max,
                                         const int64_t m2Max);
    void CalcleTNDCausalDeterParamGQA(DeterPrefixData& deterPrefixData, const int64_t m0Max, const int64_t m1Max,
                                      const int64_t m2Max);
    void CalcleTNDDenseDeterParam();
    void CalcleTNDDenseBns2DeterParam(DeterPrefixData& deterPrefixData);
    void CalcleTNDDeterSyncRounds(std::vector<std::pair<uint64_t, uint64_t>>& syncRounds,
                                  std::vector<std::pair<uint64_t, uint64_t>>& syncRoundRanges);
    void CalcleTNDDenseDeterSplitDkOffset(DeterPrefixData& deterPrefixData,
                                          std::vector<std::pair<uint64_t, uint64_t>>& syncRounds,
                                          std::vector<std::pair<uint64_t, uint64_t>>& syncRoundRanges);
    void UpdateSeparateDkOffset(
        std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>& coordinateInfo,
        TndBandDeterRoundInfo& tndBandDeterRoundInfo);
    void UpdateSeparateDkOffsetLargeM(
        std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>& coordinateInfo,
        TndBandDeterRoundInfo& tndBandDeterRoundInfo);
    void UpdateSeparateDkOffsetSmallM(
        std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>& coordinateInfo,
        TndBandDeterRoundInfo& tndBandDeterRoundInfo);
    void CalcleTNDBandDeterSplitDkOffset(DeterPrefixData& deterPrefixData,
                                         std::vector<std::pair<uint64_t, uint64_t>>& syncRounds,
                                         std::vector<std::pair<uint64_t, uint64_t>>& syncRoundRanges);
    void CalcleTNDBandDeterPrefix(DeterPrefixData& deterPrefixData, int64_t N11, int64_t& mnMax);
    void CalcleTNDBandBns2DeterParam(DeterPrefixData& deterPrefixData);
    void SetCoreRoundInfo(TndBandDeterRoundInfo& tndBandDeterRoundInfo, uint64_t round, int64_t batchId);
    void SetPseLayout();
    void GetCommonS1S2OuterIndex(int64_t (*parseInfo)[ARRAY_LENGTH], int64_t gTail, int64_t& s1oIdx, int64_t& s2oIdx);
    void SetSparsePrefixBlockInterval(int64_t bIdx, int64_t nIdx,
                                      std::vector<std::vector<std::pair<int64_t, int64_t>>>& s1ValidIdx,
                                      int64_t (&blockStarts)[CORE_LIST_NUM], int64_t (&blockEnds)[CORE_LIST_NUM],
                                      uint32_t& coreNum, int64_t& tmepBlock);
    void CalValidUnpadBlockInfo(int64_t batchIdx, std::vector<std::vector<std::vector<int64_t>>>& calculatedBlockInfo);
    void GetUnpadS1S2OuterIndex(int64_t& s1oIdx, int64_t& s2oIdx, int64_t gTail, int64_t bIdx,
                                std::vector<std::vector<std::vector<int64_t>>>& calculatedBlockInfo);
    std::vector<uint64_t> CalculateSyncRound(std::vector<std::pair<uint64_t, uint64_t>> syncRounds);
    int64_t GetKeyOffset(const int64_t* kvValue, int64_t w, int64_t y);
    bool IsSeparateS2(
        std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>& coordinateInfo);
    bool SupportTrans2BS2N2GD();
    template <const uint32_t deterSparseType>
    std::tuple<int64_t, int64_t, int64_t> CalTNDDenseIndex(DeterPrefixData& deterPrefixData, int64_t coreId,
                                                           int64_t roundId, int64_t N1);
    ge::graphStatus CheckAttenMaskShape();
    ge::graphStatus ProcessSinkInfo();
    ge::graphStatus CheckSoftmaxMaxSumTndShape();
    ge::graphStatus ProcessPseNormal(const char* inputLayout);
    ge::graphStatus ProcessPseInfo(const char* inputLayout);
    ge::graphStatus ProcessSparseModeInfo();
    ge::graphStatus ProcessTokensInfo();
    ge::graphStatus SaveToTilingData();
    ge::graphStatus GetSparsePrefixBlockInfo();
    ge::graphStatus GetSparseUnpadBlockInfo();
    ge::graphStatus ProcessPseSparseMode8();
    ge::graphStatus ProcessInnerPseInfo(size_t pseShapeDim);
    ge::graphStatus SetAttenMaskShapeType(const gert::StorageShape* attenMaskShape, size_t dimNum);
    ge::graphStatus CheckUnpadTokensInfo();
    int64_t FindBandIdx();
    void FillBlockInfo(std::vector<std::vector<std::vector<int64_t>>>& calculatedBlockInfo,
                       std::vector<std::vector<int64_t>>& totalBlockInfo);
    // Tnd Swizzle
    void BuildTndBn2gs1s2SwizzleParam(const std::vector<std::vector<std::vector<int64_t>>>& calculatedBlockInfo);
    int64_t GetActualS2Outer(int64_t batchIdx);
    void CalcTndCubeColumnRange(int64_t batchIdx, int64_t s2Idx, int64_t& begin, int64_t& end);

    bool CheckPrefixNExist(const int64_t bIdx, const int64_t prefixN,
                           std::vector<std::vector<std::pair<int64_t, int64_t>>>& s2ValidIdx);
    void GetCommS1S2OuterInfo(const int64_t s1, const int64_t s2, int64_t (*parseInfo)[ARRAY_LENGTH]);
    void GetParseS1S2OuterInfo(int64_t (*parseInfo)[ARRAY_LENGTH]);
    bool SetSparseParams();
    void GetCommS1S2OuterInfo(const int64_t prefixN, std::vector<std::pair<int64_t, int64_t>>& s2ValidIdx);
    void PrintShapeInfo();
    bool CheckSparseModeValue();
    bool CheckVarLenSparseModeValue();
    std::pair<uint32_t, uint32_t> GetS1S2TemplateType();
    uint32_t GetDTemplateType();
    void SetQKVStartIdx();
    ge::graphStatus QuantScaleShapeValidCheck();
    ge::graphStatus QuantScaleDtypeValidCheck();
    bool CheckExceedL2Cache();
    bool CheckIsLargeInvalidBlk(FuzzyBaseInfoParamsDavid& fBaseParams);
    int64_t GetTotalPerBatchNum(FuzzyBaseInfoParamsDavid& fBaseParams, uint8_t sparseType);
    void SetSplitAxis();
    ge::graphStatus DoBn2MultiBlkSparse();
    ge::graphStatus GetBlockInfoOfTNDForBn2();
    float BinarySearchMaxBlockNumPerCore(const std::vector<std::vector<float>>& acturalBlockInfo);
    bool IsPossible(const std::vector<std::vector<float>>& acturalBlockInfo, const float possibleMax);
    bool CaclePerCoreBlockInfoBn2(const std::vector<std::vector<int64_t>>& totalBlockInfo,
                                  const std::vector<std::vector<float>>& acturalBlockInfo,
                                  const float maxBlockNumPerCore, int64_t (&blockStarts)[CORE_LIST_NUM],
                                  int64_t (&blockEnds)[CORE_LIST_NUM]);
    void FillBlockInfoLoadBalanceForBn2(std::vector<std::vector<int64_t>>& totalBlockInfo,
                                        std::vector<std::vector<float>>& acturalBlockInfo);
    ge::graphStatus GetSparseBlockInfoBn2();

    FuzzyBaseInfoParamsDavid fBaseParams;
    platform_ascendc::SocVersion socVersion;
    NpuArch npuArch = NpuArch::DAV_RESV;
    TndBaseInfo tndBaseInfo;
};

class FlashAttentionScoreGradTilingUnpaddedAttensionDavid : public FlashAttentionScoreGradTilingUs1s2Bs2David {
public:
    explicit FlashAttentionScoreGradTilingUnpaddedAttensionDavid(gert::TilingContext* context_)
        : FlashAttentionScoreGradTilingUs1s2Bs2David(context_)
    {}
    ~FlashAttentionScoreGradTilingUnpaddedAttensionDavid() {}

protected:
    bool IsCapable() override;
};

} // namespace fag
} // namespace optiling
