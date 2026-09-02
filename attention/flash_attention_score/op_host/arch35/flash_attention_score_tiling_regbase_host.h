/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ARCH35_FLASH_ATTENTION_SCORE_TILING_REGBASE_H_
#define ARCH35_FLASH_ATTENTION_SCORE_TILING_REGBASE_H_

#include <numeric>
#include <alog_pub.h>
#include <tiling/tiling_api.h>
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_type.h"
#include "../../op_kernel/arch35/flash_attention_score_template_tiling_key.h"
#include "../../../common/op_kernel/arch35/flash_attention_score_tiling_regbase_arch35.h"
#include "err/ops_err.h"
#include "platform/soc_spec.h"

using namespace Ops::Transformer::OpTiling;
namespace optiling {
namespace FA {
static const int64_t DN_D_64 = 64L;
static const int64_t DN_S1_128 = 128L;
static const uint32_t NUM_10 = 10;
static const uint32_t DIM_NUM_2 = 2;
static const uint32_t DIM_NUM_3 = 3;
static const int64_t ALIGNED_NUM_8 = 8L;
static const int64_t DROP_MASK_CAL_SIZE = 6L;
static const int64_t NUM_64 = 64L;
static const int64_t NUM_128 = 128L;
static const int64_t NUM_192 = 192L;
static const int64_t NUM_256 = 256L;
static const int64_t NUM_320 = 320L;
static const int64_t NUM_384 = 384L;
static const int64_t NUM_448 = 448L;
static const int64_t NUM_512 = 512L;
static const int64_t NUM_768 = 768L;
static const int64_t NUM_1024 = 1024L;
static const int64_t NUM_1536 = 1536L;
static const int64_t MIN_DN_S2 = 256L;
static const int64_t MIN_D_TO_USE_WORKSPACE = 128L;
static const int64_t DN_MIN_D_TO_USE_WORKSPACE = 192L;
static const int64_t D_TEMPLATE_SPLIT_SIZE = 64L;
static const int64_t PING_PONG_VALUE = 3L;
static const int64_t DATA_TYPE_FP32 = 4L;
static const int64_t DATA_TYPE_FP8 = 1L;
static const int64_t GM_ALIGN = 512;
static const int64_t FRACTAL_NUM = 16L;
static const int64_t PSE_DIM_NUM = 4L;
static const int64_t BYTE_BIT_NUM = 8UL;
static const int64_t HEAD_DIM_MAX_VALUE = 768L;
static const size_t PSE_INPUT_INDEX = 3UL;
static const size_t DROP_MASK_INPUT_INDEX = 4UL;
static const size_t ATTENTION_MASK_INPUT_INDEX = 6UL;
static const size_t PREFIX_INPUT_INDEX = 7UL;
static const size_t ACTUAL_SEQ_LENGTH_INPUT_INDEX = 8UL;
static const size_t ACTUAL_SEQ_LENGTH_KV_INPUT_INDEX = 9UL;
static const size_t Q_START_IDX_INPUT_INDEX = 10UL;
static const size_t KV_START_IDX_INPUT_INDEX = 11UL;
static const size_t ATTEN_OUT_INDEX = 3UL;
static const size_t ATTENTION_MASK_DIM_NUM_4 = 4UL;
static const size_t ATTENTION_MASK_DIM_NUM_2 = 2UL;
static const int64_t HIGH_PERF_BLOCK_SIZE = 128L;
static const uint32_t PSE_ALIBI_S_SIZE = 1024;
static constexpr size_t WORK_SPACE_RESERVE_SIZE = 16 * 1024 * 1024;
static const int64_t ATTEN_MASK_S1_REV_INDEX = 2L;
static const int64_t ATTEN_MASK_COMPRESS_LIMIT = 2048L;
static const int64_t ATTEN_MASK_COMPRESS_PREFIX_LIMIT = 3072L;
static const int64_t SLOPE_BN_DIM_NUM = 2L;
static const int64_t SLOPE_N_DIM_NUM = 1L;
static const int64_t INVALID_ROW_SPARSE_RATIO = 6L;
static const int64_t D_Q_SCALE_INDEX = 12L;
static const int64_t D_K_SCALE_INDEX = 13L;
static const int64_t D_V_SCALE_INDEX = 14L;
static const int64_t QUERY_ROPE_INDEX = 15L;
static const int64_t KEY_ROPE_INDEX = 16L;
static const int64_t SINK_INPUT_INDEX = 17L;
static const int64_t P_SCALE_INDEX = 18L;
static const int64_t D_SCALE_DIM_NUM_4 = 4L;
static const int64_t D_SCALE_DIM_NUM_0 = 0L;
static const int64_t D_SCALE_DIM_NUM_1 = 1L;
static const int64_t D_SCALE_DIM_NUM_2 = 2L;
static const int64_t D_SCALE_DIM_NUM_3 = 3L;
static const int64_t QUANT_BLOCK_SIZE = 128L;
static const int64_t QUANT_K_BLOCK_SIZE = 256L;
static const int64_t QUANT_V_BLOCK_SIZE = 512L;
static const int64_t COMPRESS_ATTEN_MASK_SIZE = 2048 * 2048;

enum class LayoutType : uint8_t {
    NONE = 0,
    LAYOUT_BSH = 1,
    LAYOUT_BSND = 1,
    LAYOUT_SBH = 2,
    LAYOUT_BNSD = 3,
    LAYOUT_TND = 4,
};

enum class AttenMaskShapeType : uint8_t {
    ATTEN_B_N2_G_S1_S2 = 0,
    ATTEN_B_1_1_S1_S2 = 1,
    ATTEN_1_1_1_S1_S2 = 2,
    ATTEN_1_1_1_T_T = 99,
};

enum class PseShapeType : uint8_t {
    PSE_B_N2_G_S1_S2 = 0,
    PSE_B_N2_G_1_S2 = 1,
    PSE_B_N2_G_SLOPE,
    PSE_1_N2_G_SLOPE
};

enum class SparseMode : uint8_t {
    NO_MASK = 0,
    ALL_MASK,
    LEFT_UP_CAUSAL,
    RIGHT_DOWN_CAUSAL,
    BAND,
    PREFIX,
    PREFIX_COMPRESS,
    RIGHT_DOWN_CAUSAL_BAND,
    BAND_LEFT_UP_CAUSAL
};

enum class AttenMaskCompressMode : uint8_t {
    NO_COMPRESS_MODE = 0,
    LEFT_UP_CAUSAL_MODE,
    RIGHT_DOWN_CAUSAL_MODE,
    BAND_MODE,
    PREFIX_MODE,
    RIGHT_DOWN_CAUSAL_BAND_MODE = 5,
    BAND_LEFT_UP_CAUSAL_MODE
};

enum class ImplMode : uint8_t {
    AA_HIGH_PRECISION = 0,
    AA_HIGH_PERFORMANCE = 1,
    AA_INVALID_LINE_HIGH_PRECISION = 2
};

enum class PseType : uint8_t {
    PSE_OUTER_MUL_ADD_TYPE = 0, // v2 default
    PSE_OUTER_ADD_MUL_TYPE,     // v1 current usage
    PSE_INNER_MUL_ADD_TYPE,
    PSE_INNER_MUL_ADD_SQRT_TYPE,
    PSE_INVALID_TYPE,
    PSE_NONE_TYPE = 9
};

enum class PseEncodeType : uint8_t {
    PSE_ENCODE_NONE = 0,
    PSE_ENCODE_ALIBI_S2_FULL = 0x11, // shape: (1024, S2)
};

enum class DTemplateType : uint32_t {
    NONALIGNED = 0,
    ALIGNED_16 = 16,
    ALIGNED_32 = 32,
    ALIGNED_48 = 48,
    ALIGNED_64 = 64,
    ALIGNED_80 = 80,
    ALIGNED_96 = 96,
    ALIGNED_128 = 128,
    ALIGNED_160 = 160,
    ALIGNED_192 = 192,
    ALIGNED_224 = 224,
    ALIGNED_256 = 256,
    ALIGNED_320 = 320,
    ALIGNED_384 = 384,
    ALIGNED_448 = 448,
    ALIGNED_768 = 768,
    DTEMPLATEBOTTOM
};

enum class STemplateType : uint32_t {
    NONALIGNED = 0,
    ALIGNED_16 = 16,
    ALIGNED_32 = 32,
    ALIGNED_64 = 64,
    ALIGNED_128 = 128,
    ALIGNED_256 = 256,
    ALIGNED_512 = 512,
    STEMPLATEBOTTOM
};

enum class ConstTemplateType : uint8_t {
    NONALIGNED128 = 0,
    ALIGNED_128 = 1,
    ONLY_128 = 2,
    LESS_THAN_128 = 3,
    ALIGNED_64 = 4,
    ALIGNED_80 = 5,
    ALIGNED_16 = 6,
    ALIGNED_256 = 7,
    ALIGNED_96 = 8,
};

struct FACompileInfoCommon {
    uint32_t aivNum;
    uint32_t aicNum;
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0aSize;
    uint64_t l0bSize;
    uint64_t l0cSize;
    uint64_t l2CacheSize;
    int64_t coreNum;
    platform_ascendc::SocVersion socVersion;
    uint32_t rsvd;
};

enum class SplitCoreMode : uint8_t {
    SQ_SINGLE_CORE_FIRST = 0, // 传统分核
    SQ_MULTI_CORE_FIRST = 1, // 全新分核：对于N和S较大场景，采取顺序分配或者两两配对的分核方式；
};

template <typename T>
static auto AlignUp(T num1, T num2) -> T
{
    if (num2 == 0) {
        return 0;
    }
    if (num1 < 0) {
        return -(-num1 / num2) * num2;
    }
    return (num1 + num2 - 1) / num2 * num2;
}

template <typename T>
static auto AlignDown(T num1, T num2) -> T
{
    if (num2 == 0) {
        return 0;
    }
    return num1 / num2 * num2;
}

template <typename T>
static auto CeilDivision(T num1, T num2) -> T
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2;
}

template <typename T>
static auto CeilDiv(const T n1, const T n2) -> T
{
    if (n1 == 0) {
        return 0;
    }
    return (n2 != 0) ? (((n1 - 1) / n2) + 1) : n1;
}

template <typename T>
static auto CalcTailSize(T num1, T num2) -> T
{
    if (num2 == 0) {
        return 0;
    }
    T mod = num1 % num2;
    return mod != 0 ? mod : num2;
}

class FlashAttentionScoreTilingRegbase : public TilingBaseClass {
public:
    explicit FlashAttentionScoreTilingRegbase(gert::TilingContext *context)
        : TilingBaseClass(context)
    {
        Reset();
    }
    ~FlashAttentionScoreTilingRegbase() override = default;

    void Reset(gert::TilingContext *context) override
    {
        Reset();
        TilingBaseClass::Reset(context);
    }

protected:
    void Reset()
    {
        bmmDtype = matmul_tiling::DataType::DT_FLOAT;
        bmm1OutDtype = matmul_tiling::DataType::DT_FLOAT;
        bmm2OutDtype = matmul_tiling::DataType::DT_FLOAT;

        inputDtype = ge::DT_FLOAT16;
        inputDtypeBytes = ge::GetSizeByDataType(inputDtype);
        calcTypeSize = inputDtypeBytes;

        tilingKeyDType = DtypeEnum::FLOAT16;

        bSize = 0LL;
        gSize = 0LL;
        dSize = 0LL;
        dSizeV = 0LL;
        dSizeRope = 0LL;
        n1Size = 0LL;
        n2Size = 0LL;
        s1Size = 0LL;
        s2Size = 0LL;
        accumS1 = 0LL;
        accumS2 = 0LL;
        bandIndex = 0LL;
        dropTotalSize = 0LL;
        realT1Size = 0LL;

        s1StrideSize = 0LL;
        s2StrideSize = 0LL;
        preTokens = std::numeric_limits<int32_t>::max();
        nextTokens = std::numeric_limits<int32_t>::max();
        sparseMode = static_cast<int64_t>(SparseMode::NO_MASK);
        pseType = static_cast<int64_t>(PseType::PSE_OUTER_ADD_MUL_TYPE);
        pseAlibiBaseS1 = 0;
        pseAlibiBaseS2 = 0;
        tndSoftmaxOut = 0;
        qStartIdx = 0;
        kvStartIdx = 0;
        keepProb = 1.0f;
        scaleValue = 1.0f;
        attenMaskCompressMode = static_cast<uint8_t>(AttenMaskCompressMode::NO_COMPRESS_MODE);
        isHighPercision = true;

        s1BasicBlock = std::numeric_limits<int64_t>::max();
        s2BasicBlock = std::numeric_limits<int64_t>::max();
        dBasicBlock = std::numeric_limits<int64_t>::max();

        maxValidS2Len = 0LL;
        opName = nullptr;
        inputLayout = nullptr;
        softmaxOutLayout = nullptr;
    }

    bool IsCapable() override
    {
        return true;
    }
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    virtual bool GetSparseInfo(SparseEnum &sparseType);
    void EnableBandInvalidLineImplMode();
    bool SparseModeProcess(SparseEnum &sparseType);
    void SetSparseTilingInfo(SparseEnum &sparseType);
    bool SparseBandModeCheck(int64_t maxS1Val, int64_t maxS2Val, int64_t minS1Val, int64_t minS2Val,
                             SparseEnum &sparseType);
    bool PretokenAndNexttokenAdjustment(SparseEnum &sparseType);

    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override = 0;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override = 0;
    // 7、保存Tiling数据，//
    // 由于这个类中不保存TilingData，子类中需要调用这个类的PostTiling并额外设置RawTilingData的DataSize
    ge::graphStatus PostTiling() override;

    bool GetActualSeqLenData(int64_t inputIdx, std::vector<int64_t> &res, int64_t &actualLen) const;

    // 关于TilingData的校验需要在子类中实现
    virtual ge::graphStatus CheckContext();
    virtual bool AnalyzeDtype();
    bool AnalyzeAttrs();
    bool AnalyzeLayout();
    bool AnalyzeTndLayout(const gert::Shape &queryShape, const gert::Shape &keyShape, const gert::Shape &valueShape);
    bool Analyze3DimLayout(const gert::Shape &queryShape, const gert::Shape &keyShape, const gert::Shape &valueShape,
                           size_t layoutLen, const gert::Shape *queryRopeShape);
    bool Analyze4DimLayout(const gert::Shape &queryShape, const gert::Shape &keyShape, const gert::Shape &valueShape,
                           size_t layoutLen, const gert::Shape *queryRopeShape);
    bool AnalyzeTndPseOptionalInput(PseShapeType &pseShapeType, const gert::Shape &pseShapeDims, size_t pseDimNum,
                                    int64_t pseBSize);
    bool AnalyzeGeneralPseOptionalInput(PseShapeType &pseShapeType, const gert::Shape &pseShapeDims, size_t pseDimNum,
                                        int64_t pseBSize);
    bool AnalyzePseOptionalInput();
    bool Analyze4DimAttenOptionalInput(AttenMaskShapeType &attenMaskShapeType,
                                       const gert::Shape &attenMaskStorageShape);
    bool Analyze2DimAttenOptionalInput(AttenMaskShapeType &attenMaskShapeType,
                                       const gert::Shape &attenMaskStorageShape);
    bool AnalyzeAttenOptionalInputDimNumLimit(const gert::Shape &attenMaskStorageShape, size_t attenMaskDimNum);
    bool AnalyzeAttenOptionalInput();
    bool AnalyzeDropOptionalInput();
    bool AnalyzeFp8OptionalInput();
    bool AnalyzeRopeOptionalInput();
    bool AnalyzeOptionalInput();
    bool AnalyzeSinkOptionalInput();
    virtual void CalcS1S2BasicBlock() = 0;
    virtual void CalcDBasicBlock() = 0;
    virtual void CalcDVBasicBlock();
    virtual int64_t CalcTotalSize();
    virtual void AnalyzeOptionalDn() {}
    virtual ge::graphStatus SetQKVStartIdx();
    virtual void SetOutputDtype();
    virtual void SetSplitCoreModeParam();
    virtual void CalcThresholdForS2Size();
    virtual bool IsUseSplitCoreMode(SparseMode inputSparseMode);
    virtual void SetMultiCoreParamsRegbase(int64_t totalSize, int64_t coreNum);
    virtual void SetSparseParamsRegbase(int64_t maxCoreNum);
    virtual bool SetPseAlibiParamsRegbase();
    virtual bool InitSparseValidArray(std::vector<int64_t> &sparseValidArray, int64_t bIdx);
    virtual bool SetSparseStartIdx(const std::vector<int64_t> &sparseValidArray,
                                   MultiCoreParamsRegbase &multiCoreParamsRegbase, int64_t maxCoreNum);
    void SetPrefixSparseStartIdx(const std::vector<std::vector<int64_t>> &sparseValidArray,
                                 MultiCoreParamsRegbase &multiCoreParamsRegbase, int64_t maxCoreNum);
    int64_t PrintSparseMaxMinLoadPerCore(const std::vector<int64_t> &sparseValidArray, int64_t *sparseStartIdx,
                                         int32_t validAivNum, int64_t avgLoadSize);
    bool PartitionSparseData(const std::vector<int64_t> &sparseRollingArray, int64_t sparseRollingArraySum,
                             int64_t sparseArraySize, int64_t loadMaxEachCore, std::vector<int64_t> &partitionResult);
    SparseEnum GetPrefixNList(std::ostringstream &failReason);

    uint32_t aivNum;
    uint32_t aicNum;
    platform_ascendc::SocVersion socVersion;
    NpuArch npuArch = NpuArch::DAV_RESV;

    matmul_tiling::DataType bmmDtype = matmul_tiling::DataType::DT_FLOAT;
    matmul_tiling::DataType bmm1OutDtype = matmul_tiling::DataType::DT_FLOAT;
    matmul_tiling::DataType bmm2OutDtype = matmul_tiling::DataType::DT_FLOAT;

    ge::DataType inputDtype;
    int64_t inputDtypeBytes;
    int64_t calcTypeSize;

    bool isHighPercision; // fp16 high percision mode

    DtypeEnum tilingKeyDType;
    LayoutType tilingKeyLayout;
    ImplMode implMode;

    int64_t bSize;
    int64_t gSize;
    int64_t dSize;
    int64_t dSizeV;
    int64_t dSizeRope;
    int64_t n1Size;
    int64_t n2Size;
    int64_t s1Size;
    int64_t s2Size;
    int64_t s1StrideSize; // query Shape S inner axes, for bmm1
    int64_t s2StrideSize; // key Shape S inner axes, for bmm1
    int64_t preTokens;
    int64_t nextTokens;
    int64_t s1SparseValidSize;
    int64_t s2SparseValidSize;
    int64_t sparseMode;
    int64_t pseType;
    int64_t pseAlibiBaseS1;
    int64_t pseAlibiBaseS2;
    uint8_t tndSoftmaxOut;
    int64_t qStartIdx;
    int64_t kvStartIdx;
    int64_t accumS1;
    int64_t dropTotalSize;
    int64_t accumS2;
    int64_t bandIndex;
    int64_t realT1Size;
    std::vector<int64_t> actualSeqLenData;
    std::vector<int64_t> actualSeqLenKvData;
    float keepProb;
    int64_t keepProbUint8;
    int64_t seed;
    int64_t offset;
    int64_t outDtype;
    float scaleValue;
    uint8_t attenMaskCompressMode;

    int64_t s1BasicBlock;
    int64_t s2BasicBlock;
    int64_t dBasicBlock;
    int64_t dVBasicBlock;

    int64_t maxValidS2Len;
    int64_t thresholdS2Size = std::numeric_limits<int64_t>::max();
    int64_t firstFullLoadS1OuterIdx = -1;
    SplitCoreMode splitCoreMode = SplitCoreMode::SQ_SINGLE_CORE_FIRST;
    uint64_t l2CacheSize = 134217728; // 128M

    const char *templateName = "base";
    const char *opName;
    const char *inputLayout;
    const char *softmaxOutLayout;
    const int64_t *prefixNData;

    bool isSparseValidSizeAligned = false;
    bool hasPse = false;
    bool hasAttenMask = false;
    bool hasDropOut = false;
    bool dropMaskOuter = false;
    bool regbase = false;
    bool hasRope = false;
    bool hasSink = false;
    bool optionalDn = false;

    DTemplateType dTemplateType = DTemplateType::DTEMPLATEBOTTOM;
    DTemplateType dVTemplateType = DTemplateType::DTEMPLATEBOTTOM;
    FlashAttentionScoreSimplifiedTilingData *tilingData =
        context_->GetTilingData<FlashAttentionScoreSimplifiedTilingData>();
    InputParamsRegbase *inputParamsRegbase_ = &tilingData->inputParamsRegbase;
    MultiCoreParamsRegbase *multiCoreParamsRegbase_ = &tilingData->multiCoreParamsRegbase;
    DropmaskParamsRegbase *dropmaskParamsRegbase_ = &tilingData->dropmaskParamsRegbase;
};
} // namespace FA
} // namespace optiling

#endif // ARCH35_FLASH_ATTENTION_SCORE_TILING_REGBASE_H_
