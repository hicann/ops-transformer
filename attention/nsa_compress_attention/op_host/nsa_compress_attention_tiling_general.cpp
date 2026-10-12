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
 * \file nsa_compress_attention_tiling_general.cpp
 * \brief
 */

#include "log/log.h"
#include "err/ops_err.h"
#include <numeric>
#include "op_host/data_copy_transpose_tiling.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_type.h"
#include "nsa_compress_attention_tiling.h"
#include "nsa_compress_attention_tiling_common.h"
using namespace Ops::Transformer::OpTiling;
namespace optiling {
namespace NsaCompressAttention {
const int64_t DIM_0 = 0L;
const int64_t DIM_1 = 1L;
const int64_t DIM_2 = 2L;
const int64_t DIM_3 = 3L;
const int64_t CV_RATIO = 2L;
const uint64_t BLOCK_BYTE = 32;
const uint64_t SIZE_1024 = 1024;
const uint32_t MODE_7 = 7;

const int64_t FRACTAL_NUM = 16L;
const size_t ATTENTION_MASK_INPUT_INDEX = 3UL;
const size_t TOPK_MASK_INPUT_INDEX = 7UL;
const size_t ACTUAL_SEQ_LENGTH_INPUT_INDEX = 4UL;
const size_t ACTUAL_SEQ_LENGTH_CMP_KV_INPUT_INDEX = 5UL;
const size_t ATTEN_OUT_INDEX = 2UL;

const int64_t MAX_VAR_LEN_SEQ_LEN = 4096L;
const int64_t BMM1_DEPTH_A1_2 = 2L;
const int64_t BMM1_DEPTH_A1_3 = 3L;
const int64_t HEAD_DIM_MAX_VALUE = 768L;
const int64_t NSA_BASE_S1G_SIZE = 128L;
const int64_t NSA_DOUBLE_BUFFER = 2L;

constexpr size_t WORK_SPACE_RESERVE_SIZE = 16 * 1024 * 1024;

enum LayoutType : uint8_t {
    None = 0,
    LAYOUT_BSH = 1,
    LAYOUT_BSND = 1,
    LAYOUT_SBH = 2,
    LAYOUT_BNSD = 3,
    LAYOUT_TND = 4,
};

enum SparseMode : uint8_t {
    NO_MASK = 0,
    ALL_MASK
};

template <typename T>
static T AlignUp(T num1, T num2)
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
static T CeilDivision(T num1, T num2)
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2;
}

template <typename T>
static T CeilDiv(const T n1, const T n2)
{
    if (n1 == 0) {
        return 0;
    }
    return (n2 != 0) ? (((n1 - 1) / n2) + 1) : n1;
}

template <typename T>
static T CalcTailSize(T num1, T num2)
{
    if (num2 == 0) {
        return 0;
    }
    T mod = num1 % num2;
    return mod != 0 ? mod : num2;
}

class NsaCompressAttentionTilingBase : public TilingBaseClass {
public:
    explicit NsaCompressAttentionTilingBase(gert::TilingContext* context)
        : TilingBaseClass(context)
    {
        Reset();
    }
    ~NsaCompressAttentionTilingBase() override = default;

    void Reset(gert::TilingContext* context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    bool IsCapable() override
    {
        return true;
    }
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;

    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override = 0;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;

    void Reset();

    void GetActualSeqLenData(int64_t inputIdx, std::array<int64_t, MAX_VAR_LEN_SEQ_LEN>& res, int64_t& actualLen) const;

    virtual bool IsTemplateMatched() const
    {
        return false;
    }

    ge::graphStatus CheckContext();
    ge::graphStatus CheckAttr();
    virtual bool AnalyzeDtype();
    bool AnalyzeAttrs();
    bool AnalyzeLayout();
    bool CheckUbSize(uint64_t& s2ScoreLoopLen, uint64_t& outerLoop);
    bool Analyze3DimLayout(const gert::Shape& queryShape, const gert::Shape& keyShape, const gert::Shape& valueShape,
                           size_t layoutLen);
    bool Analyze4DimLayout(const gert::Shape& queryShape, const gert::Shape& keyShape, const gert::Shape& valueShape,
                           size_t layoutLen);
    bool AnalyzeOptionalInput();
    bool MatchTemplate();
    virtual void CalcS1S2BasicBlock();
    virtual bool SetBmm1TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock,
                                    matmul_tiling::MatmulApiTiling& bmm1) = 0;
    virtual bool SetBmm2TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t tmpDBasicBlock,
                                    matmul_tiling::MatmulApiTiling& bmm2) = 0;
    bool SetMatMulTiling(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t tmpDBasicBlock,
                         matmul_tiling::MatmulApiTiling& bmm1, matmul_tiling::MatmulApiTiling& bmm2);
    bool SetMatMulTiling(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t tmpDBasicBlock);
    virtual void SetCoreParams();
    virtual void SetMultiCoreParams();
    virtual void SetSoftMaxTiling();
    virtual void TopKTiling();
    virtual void SetImportanceScoreParams();
    uint32_t aivNum;
    uint32_t aicNum;
    uint64_t l2CacheSize = 0;

    matmul_tiling::DataType bmmDtype = matmul_tiling::DataType::DT_FLOAT;
    matmul_tiling::DataType bmm1OutDtype = matmul_tiling::DataType::DT_FLOAT;
    matmul_tiling::DataType bmm2OutDtype = matmul_tiling::DataType::DT_FLOAT;

    ge::DataType inputDtype;
    int64_t inputDtypeBytes;
    int64_t calcTypeSize;

    LayoutType tilingKeyLayout;

    int64_t bSize;
    int64_t gSize;
    int64_t dSize;
    int64_t d2Size;
    int64_t n1Size;
    int64_t n2Size;
    int64_t s1Size;
    int64_t s2Size;
    int64_t sparseMode;
    int64_t maxS1Val;
    int64_t accumS1;
    int64_t accumS1BlockNum;
    int64_t maxS2Val;
    int64_t accumS2;
    std::array<int64_t, MAX_VAR_LEN_SEQ_LEN> actualSeqLenData;
    std::array<int64_t, MAX_VAR_LEN_SEQ_LEN> actualSeqLenKvData;
    float scaleValue;
    uint8_t attenMaskExistFlag;

    int64_t alignedS1;
    int64_t alignedS2;
    int64_t alignedD;
    int64_t alignedD2;

    int64_t s1BasicBlock;
    int64_t s2BasicBlock;
    int64_t s1VecBasicBlock;
    int64_t dBasicBlock;

    const char* templateName = "base";
    const char* opName;
    const char* inputLayout;

    int32_t mm1BaseM;
    int32_t mm1BaseN;
    int32_t mm1BaseK;
    int32_t mm2BaseM;
    int32_t mm2BaseN;
    int32_t mm2BaseK;

    int64_t compressBlockSize; // l
    int64_t compressStride;    // d
    int64_t selectBlockSize;   // l'
    int64_t selectBlockCount;  // n

    bool hasAttenMask = false;
    bool hasTopkMask = false;
    bool isSameAB = false;
    bool enableBestBlock = false;
    NsaCompressAttentionGeneralTilingData tilingData;
};

ge::graphStatus NsaCompressAttentionTilingBase::GetPlatformInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        auto compileInfoPtr = context_->GetCompileInfo<NsaCompressAttentionCompileInfo>();
        OP_CHECK_IF(compileInfoPtr == nullptr, OPS_REPORT_VECTOR_INNER_ERR(opName, "compileInfoPtr is null."),
                    return ge::GRAPH_FAILED);

        aivNum = compileInfoPtr->aivNum;
        aicNum = compileInfoPtr->aicNum;
        aicoreParams_.ubSize = compileInfoPtr->ubSize;
        aicoreParams_.l1Size = compileInfoPtr->l1Size;
        aicoreParams_.l0cSize = compileInfoPtr->l0cSize;
        l2CacheSize = compileInfoPtr->l2CacheSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        aivNum = ascendcPlatform.GetCoreNumAiv();
        aicNum = ascendcPlatform.GetCoreNumAic();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, aicoreParams_.ubSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, aicoreParams_.l1Size);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, aicoreParams_.l0cSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2CacheSize);
    }
    OP_LOGI(context_, "get platform from compileInfo.aivNum(%u) aicNum(%u) ubSize(%lu) l1Size(%lu) l0cSize(%lu).",
            aivNum, aicNum, aicoreParams_.ubSize, aicoreParams_.l1Size, aicoreParams_.l0cSize);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NsaCompressAttentionTilingBase::CheckContext()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    size_t idx = 0;
    auto scaleValuePtr = attrs->GetAttrPointer<float>(idx++);
    auto headNumPtr = attrs->GetAttrPointer<float>(idx++);
    auto inputLayoutPtr = attrs->GetAttrPointer<int64_t>(idx++);
    auto sparseModePtr = attrs->GetAttrPointer<int64_t>(idx++);
    auto compressBlockSizePtr = attrs->GetAttrPointer<int64_t>(idx++);
    auto compressStridePtr = attrs->GetAttrPointer<char>(idx++);
    auto selectBlockSizePtr = attrs->GetAttrPointer<int64_t>(idx++);
    auto selectBlockCountPtr = attrs->GetAttrPointer<char>(idx++);
    size_t* workspaces = context_->GetWorkspaceSizes(1);

    OP_CHECK_NULL_WITH_CONTEXT(context_, scaleValuePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, headNumPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputLayoutPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, sparseModePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, compressBlockSizePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, compressStridePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, selectBlockSizePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, selectBlockCountPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);

    auto queryShape = context_->GetInputShape(0);
    auto queryDesc = context_->GetInputDesc(0);
    auto keyShape = context_->GetInputShape(1);
    auto attenOutShape = context_->GetOutputShape(ATTEN_OUT_INDEX);

    OP_CHECK_NULL_WITH_CONTEXT(context_, queryShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, queryDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, keyShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attenOutShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData()->GetData());
    OP_CHECK_IF(context_->GetRawTilingData()->GetCapacity() < tilingData.GetDataSize(),
                OPS_REPORT_VECTOR_INNER_ERR(opName, "context tiling data capacity %zu < actual tiling data size %zu.",
                                            context_->GetRawTilingData()->GetCapacity(), tilingData.GetDataSize()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NsaCompressAttentionTilingBase::GetShapeAttrsInfo()
{
    opName = context_->GetNodeName();
    OP_LOGD(opName,
            "TilingContext: "
            "%s.",
            GetTilingContextDebugStr().c_str());
    OP_CHECK_IF(CheckContext() != ge::GRAPH_SUCCESS, OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid context."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!AnalyzeAttrs() || !AnalyzeDtype() || !AnalyzeLayout() || !AnalyzeOptionalInput(),
                OPS_REPORT_VECTOR_INNER_ERR(opName, "fail to analyze context info."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckAttr() != ge::GRAPH_SUCCESS, OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid attr."),
                return ge::GRAPH_FAILED);

    alignedS1 = AlignUp(s1Size, FRACTAL_NUM);
    alignedS2 = AlignUp(s2Size, FRACTAL_NUM);
    alignedD = AlignUp(dSize, FRACTAL_NUM);
    alignedD2 = AlignUp(d2Size, FRACTAL_NUM);

    OP_CHECK_IF(alignedS1 <= 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid alignedS1 %ld.", alignedS1),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(alignedS2 <= 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid alignedS2 %ld.", alignedS2),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(alignedD <= 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid alignedD %ld.", alignedD),
                return ge::GRAPH_FAILED);

    auto& inputParams = tilingData.inputParams;
    inputParams.set_bSize(bSize);
    inputParams.set_n2Size(n2Size);
    inputParams.set_gSize(gSize);
    inputParams.set_s1Size(s1Size);
    inputParams.set_s2Size(s2Size);
    inputParams.set_dSize(dSize);
    inputParams.set_scaleValue(scaleValue);
    inputParams.set_alignedS2(alignedS2);
    inputParams.set_selectBlockCount(selectBlockCount);
    OP_LOGD(context_,
            "input params: bn2gs1s2d"
            "[%ld, %ld, %ld, %ld, %ld, %ld], scaleValue[%f],"
            "alignedS2:%ld, selectBlockCount:%ld.",
            bSize, n2Size, gSize, s1Size, s2Size, dSize, scaleValue, alignedS2, selectBlockCount);

    return ge::GRAPH_SUCCESS;
}

void NsaCompressAttentionTilingBase::Reset()
{
    tilingData.SetDataPtr(context_->GetRawTilingData()->GetData());

    bmmDtype = matmul_tiling::DataType::DT_FLOAT;
    bmm1OutDtype = matmul_tiling::DataType::DT_FLOAT;
    bmm2OutDtype = matmul_tiling::DataType::DT_FLOAT;

    inputDtype = ge::DT_FLOAT16;
    inputDtypeBytes = ge::GetSizeByDataType(inputDtype);
    calcTypeSize = inputDtypeBytes;

    tilingKeyLayout = LayoutType::LAYOUT_BNSD;

    bSize = 0LL;
    gSize = 0LL;
    dSize = 0LL;
    n1Size = 0LL;
    n2Size = 0LL;
    s1Size = 0LL;
    s2Size = 0LL;
    maxS1Val = 0LL;
    accumS1 = 0LL;
    accumS2 = 0LL;
    maxS2Val = 0LL;

    sparseMode = static_cast<int64_t>(NO_MASK);
    scaleValue = 1.0f;
    attenMaskExistFlag = 0;

    alignedS1 = 0LL;
    alignedS2 = 0LL;
    alignedD = 0LL;
    alignedD2 = 0LL;

    compressBlockSize = 0LL;
    compressStride = 0LL;
    selectBlockSize = 0LL;
    selectBlockCount = 0LL;

    s1BasicBlock = std::numeric_limits<int64_t>::max();
    s2BasicBlock = std::numeric_limits<int64_t>::max();
    dBasicBlock = std::numeric_limits<int64_t>::max();

    opName = nullptr;
    inputLayout = nullptr;
}

bool NsaCompressAttentionTilingBase::AnalyzeDtype()
{
    inputDtype = context_->GetInputDesc(0)->GetDataType();
    inputDtypeBytes = ge::GetSizeByDataType(inputDtype);
    switch (inputDtype) {
        case ge::DT_FLOAT16:
            bmmDtype = matmul_tiling::DataType::DT_FLOAT16;
            bmm1OutDtype = matmul_tiling::DataType::DT_FLOAT;
            calcTypeSize = ge::GetSizeByDataType(ge::DT_FLOAT);
            break;
        case ge::DT_FLOAT:
            bmmDtype = matmul_tiling::DataType::DT_FLOAT;
            bmm1OutDtype = matmul_tiling::DataType::DT_FLOAT;
            calcTypeSize = ge::GetSizeByDataType(inputDtype);
            break;
        case ge::DT_BF16:
            bmmDtype = matmul_tiling::DataType::DT_BF16;
            bmm1OutDtype = matmul_tiling::DataType::DT_FLOAT;
            calcTypeSize = ge::GetSizeByDataType(ge::DT_FLOAT);
            break;
        default:
            OPS_REPORT_VECTOR_INNER_ERR(opName, "not support input dtype: %s for now",
                                        ge::TypeUtils::DataTypeToSerialString(inputDtype).c_str());
            return false;
    }

    bmm2OutDtype = bmm1OutDtype;
    return true;
}

bool NsaCompressAttentionTilingBase::AnalyzeAttrs()
{
    auto attrs = context_->GetAttrs();
    size_t idx = 0;
    auto scaleValuePtr = attrs->GetAttrPointer<float>(idx++);
    auto n1SizePtr = attrs->GetAttrPointer<uint32_t>(idx++);
    inputLayout = attrs->GetAttrPointer<char>(idx++);
    auto sparseModePtr = attrs->GetAttrPointer<char>(idx++);
    auto compressBlockSizePtr = attrs->GetAttrPointer<char>(idx++);
    auto compressStridePtr = attrs->GetAttrPointer<char>(idx++);
    auto selectBlockSizePtr = attrs->GetAttrPointer<char>(idx++);
    auto selectBlockCountPtr = attrs->GetAttrPointer<char>(idx++);

    scaleValue = *scaleValuePtr;
    n1Size = *n1SizePtr;
    sparseMode = *sparseModePtr;
    selectBlockCount = *selectBlockCountPtr;

    compressBlockSize = *compressBlockSizePtr;
    compressStride = *compressStridePtr;
    selectBlockSize = *selectBlockSizePtr;
    selectBlockCount = *selectBlockCountPtr;

    OP_CHECK_IF(n1Size == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "Head num is zero."), return false);
    OP_LOGD(context_,
            "attrs: scale_value[%f]"
            " head_num[%ld] input_layout[%s] sparse_mode[%ld], select_block_count[%ld].",
            scaleValue, n1Size, inputLayout, sparseMode, selectBlockCount);
    return true;
}

ge::graphStatus NsaCompressAttentionTilingBase::CheckAttr()
{
    uint64_t minS2Val = *std::min_element(actualSeqLenKvData.begin(), actualSeqLenKvData.begin() + bSize);
    uint64_t isM = selectBlockSize / compressStride;
    uint64_t minSelS2Val = CeilDiv(minS2Val, isM);
    OP_CHECK_IF(selectBlockCount < 1 || selectBlockCount > 32 || static_cast<uint64_t>(selectBlockCount) > minSelS2Val,
                OPS_REPORT_VECTOR_INNER_ERR(
                    opName, "selectBlockCount must be greater than 1 and less than 32 and less than min(SelSkv)."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        compressBlockSize < 16U || compressBlockSize > 128U || compressBlockSize % 16U != 0,
        OPS_REPORT_VECTOR_INNER_ERR(
            opName, "compressBlockSize must be a multiple of 16 and compressBlockSize must not be greater than 128"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(compressStride < 16U || compressStride > 64U || compressStride % 16U != 0,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "compressStride must be 16 or 32 or 48 or 64"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        selectBlockSize < 16U || selectBlockSize > 128U || selectBlockSize % 16U != 0,
        OPS_REPORT_VECTOR_INNER_ERR(
            opName, "selectBlockSize must be a multiple of 16 and selectBlockSize must not be greater than 128"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        compressStride > selectBlockSize || compressStride > compressBlockSize || compressBlockSize > selectBlockSize,
        OPS_REPORT_VECTOR_INNER_ERR(opName, "The selectBlockSize >= compressBlockSize >= compressStride must be met."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(selectBlockSize % compressStride != 0,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "selectBlockSize must be divisible by compressStride"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool NsaCompressAttentionTilingBase::AnalyzeLayout()
{
    auto& queryShape = context_->GetInputShape(0)->GetStorageShape();
    auto& keyShape = context_->GetInputShape(1)->GetStorageShape();
    auto& valueShape = context_->GetInputShape(2)->GetStorageShape();

    size_t layoutLen = strlen(inputLayout);
    OP_LOGD(context_,
            "Get input_layout"
            " [%s].",
            inputLayout);
    OP_CHECK_IF(!Analyze3DimLayout(queryShape, keyShape, valueShape, layoutLen) ||
                    !Analyze4DimLayout(queryShape, keyShape, valueShape, layoutLen),
                OPS_REPORT_VECTOR_INNER_ERR(opName, "Get unsupported layout: %s", inputLayout), return false);
    OP_CHECK_IF(gSize == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "gSize is zero."), return false);
    OP_CHECK_IF(n2Size == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "n2Size is zero."), return false);
    OP_CHECK_IF(dSize > HEAD_DIM_MAX_VALUE || dSize <= 0L,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "dSize is not in range:(0, 512]."), return false);
    OP_CHECK_IF(
        n1Size % n2Size != 0,
        OPS_REPORT_VECTOR_INNER_ERR(opName, "n1Size [%ld] should be a multiple of n2Size [%ld].", n1Size, n2Size),
        return false);
    return true;
}

void NsaCompressAttentionTilingBase::GetActualSeqLenData(int64_t inputIdx,
                                                         std::array<int64_t, MAX_VAR_LEN_SEQ_LEN>& res,
                                                         int64_t& actualLen) const
{
    auto actualSeqLenTensor = context_->GetOptionalInputTensor(inputIdx);
    if (actualSeqLenTensor == nullptr) {
        OP_LOGW(context_, "[%s]actualSeqLenTensor is null pointer", templateName);
        return;
    }
    auto& actualSeqLenShape = actualSeqLenTensor->GetShape().GetStorageShape();
    if (actualSeqLenShape.GetDimNum() != 1) {
        OP_LOGW(context_, "[%s]actualSeqLenShape is invalid %lu %ld", templateName, actualSeqLenShape.GetDimNum(),
                actualSeqLenShape.GetDim(0));
        return;
    }
    /* Get Data from tensor. */
    const int64_t* value = actualSeqLenTensor->GetData<int64_t>();
    if (value == nullptr) {
        OP_LOGW(context_, "[%s]actualSeqLenTensor data is null pointer", templateName);
        return;
    }
    res[0] = value[0];
    actualLen++;
    for (int64_t i = 1; i < actualSeqLenShape.GetDim(0); ++i) {
        auto qLen = value[i] - value[i - 1];
        res[i] = qLen < 0 ? 0 : qLen;
        actualLen++;
    }
}

bool NsaCompressAttentionTilingBase::Analyze3DimLayout(const gert::Shape& queryShape, const gert::Shape& keyShape,
                                                       const gert::Shape& valueShape, size_t layoutLen)
{
    if (layoutLen == 3UL) {
        if (strcmp(inputLayout, "TND") == 0) {
            int64_t actualSeqQLen = 0;
            int64_t actualSeqKVLen = 0;
            int64_t t1Size = queryShape.GetDim(DIM_1); // N2, T, G, D
            int64_t t2Size = keyShape.GetDim(DIM_0);
            std::fill(actualSeqLenData.begin(), actualSeqLenData.end(), 0);
            std::fill(actualSeqLenKvData.begin(), actualSeqLenKvData.end(), 0);
            GetActualSeqLenData(ACTUAL_SEQ_LENGTH_INPUT_INDEX, actualSeqLenData, actualSeqQLen);
            GetActualSeqLenData(ACTUAL_SEQ_LENGTH_CMP_KV_INPUT_INDEX, actualSeqLenKvData, actualSeqKVLen);
            OP_CHECK_IF(actualSeqQLen != actualSeqKVLen,
                        OPS_REPORT_VECTOR_INNER_ERR(opName, "VarLen scene, q is not equal kv."), return false);
            bSize = actualSeqQLen;
            accumS1 = std::accumulate(actualSeqLenData.begin(), actualSeqLenData.end(), 0LL);
            accumS2 = std::accumulate(actualSeqLenKvData.begin(), actualSeqLenKvData.end(), 0LL);
            OP_CHECK_IF(
                t1Size < accumS1 || t2Size < accumS2,
                OPS_REPORT_VECTOR_INNER_ERR(
                    opName,
                    "Query T(%ld) and key T(%ld) need larger than respectively sum of seqLen(%ld) and sekvLen(%ld).",
                    t1Size, t2Size, accumS1, accumS2),
                return false);
            maxS1Val = *std::max_element(actualSeqLenData.begin(), actualSeqLenData.end());
            maxS2Val = *std::max_element(actualSeqLenKvData.begin(), actualSeqLenKvData.end());
            s1Size = maxS1Val;
            s2Size = maxS2Val;
            n2Size = keyShape.GetDim(DIM_1);
            OP_CHECK_IF(n2Size == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "N2 is zero."), return false);
            gSize = queryShape.GetDim(DIM_2);
            dSize = queryShape.GetDim(DIM_3);
            d2Size = valueShape.GetDim(DIM_2);
            OP_CHECK_IF(n1Size != queryShape.GetDim(0) * gSize,
                        OPS_REPORT_VECTOR_INNER_ERR(opName, "head_num is [%ld], but got query head_num [%ld].", n1Size,
                                                    queryShape.GetDim(0) * gSize),
                        return false);
            tilingData.inputParams.set_layoutType(LAYOUT_TND);
            tilingKeyLayout = LayoutType::LAYOUT_TND;
        } else {
            return false;
        }
    }

    return true;
}

bool NsaCompressAttentionTilingBase::Analyze4DimLayout([[maybe_unused]] const gert::Shape& queryShape,
                                                       [[maybe_unused]] const gert::Shape& keyShape,
                                                       [[maybe_unused]] const gert::Shape& valueShape, size_t layoutLen)
{
    if (layoutLen == 4UL) {
        return false;
    }
    return true;
}

bool NsaCompressAttentionTilingBase::AnalyzeOptionalInput()
{
    auto attenMaskInput = context_->GetOptionalInputDesc(ATTENTION_MASK_INPUT_INDEX);
    auto attenMaskShape = context_->GetOptionalInputShape(ATTENTION_MASK_INPUT_INDEX);
    if (attenMaskInput != nullptr && attenMaskShape != nullptr && attenMaskShape->GetStorageShape().GetDimNum() != 0) {
        attenMaskExistFlag = 1;
        hasAttenMask = true;
        auto attenMaskType = attenMaskInput->GetDataType();
        OP_CHECK_IF(attenMaskType != ge::DT_BOOL && attenMaskType != ge::DT_UINT8,
                    OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid attenMask dtype[%s], only support bool or uint8.",
                                                ge::TypeUtils::DataTypeToSerialString(attenMaskType).c_str()),
                    return false);
    }
    auto topkMaskInput = context_->GetOptionalInputDesc(TOPK_MASK_INPUT_INDEX);
    auto topkMaskShape = context_->GetOptionalInputShape(TOPK_MASK_INPUT_INDEX);
    if (topkMaskInput != nullptr && topkMaskShape != nullptr && topkMaskShape->GetStorageShape().GetDimNum() != 0) {
        hasTopkMask = true;
        auto topkMaskType = topkMaskInput->GetDataType();
        OP_CHECK_IF(topkMaskType != ge::DT_BOOL && topkMaskType != ge::DT_UINT8,
                    OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid topkMask dtype[%s], only support bool or uint8.",
                                                ge::TypeUtils::DataTypeToSerialString(topkMaskType).c_str()),
                    return false);
    }
    return true;
}

ge::graphStatus NsaCompressAttentionTilingBase::DoOpTiling()
{
    if (!MatchTemplate()) {
        OP_LOGI(context_, "template not matched.");
        return ge::GRAPH_PARAM_INVALID;
    }

    SetCoreParams();
    SetMultiCoreParams();

    return ge::GRAPH_SUCCESS;
}

bool NsaCompressAttentionTilingBase::MatchTemplate()
{
    CalcS1S2BasicBlock();
    if (IsTemplateMatched()) {
        OP_LOGD(context_, "[%s]final basic block: [%ld, %ld].", templateName, s1BasicBlock, s2BasicBlock);
        return true;
    }
    return false;
}

void NsaCompressAttentionTilingBase::CalcS1S2BasicBlock()
{
    return;
}

void NsaCompressAttentionTilingBase::SetCoreParams()
{
    auto& coreParams = tilingData.coreParams;
    coreParams.set_s1BaseSize(s1BasicBlock);
    coreParams.set_s1BaseTailSize(CalcTailSize(s1Size, s1BasicBlock));
    coreParams.set_s1OuterSize(CeilDivision(s1Size, s1BasicBlock));
    coreParams.set_d2Size(d2Size);
}

void NsaCompressAttentionTilingBase::SetMultiCoreParams()
{
    return;
}

ge::graphStatus NsaCompressAttentionTilingBase::DoLibApiTiling()
{
    if (!SetMatMulTiling(s1BasicBlock, s2BasicBlock, dBasicBlock)) {
        return ge::GRAPH_FAILED;
    }
    SetSoftMaxTiling();
    SetImportanceScoreParams();
    TopKTiling();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NsaCompressAttentionTilingBase::PostTiling()
{
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize()); // already check capcity in CheckContext
    auto blockDim = this->CalcTschBlockDim(tilingData.multiCoreParams.get_coreNum(), aicNum, aivNum);
    context_->SetBlockDim(blockDim);
    size_t* workspaces = context_->GetWorkspaceSizes(1);

    OP_LOGD(context_, "[%s] final workspace size: %zu.", templateName, workspaces[0]);
    OP_LOGD(opName, "[%s] tiling data:%s.", templateName, GetTilingDataDebugStr().c_str());
    OP_LOGD(context_, "[%s] tiling data size: %zu.", templateName, tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

bool NsaCompressAttentionTilingBase::SetMatMulTiling(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock,
                                                     int64_t tmpDBasicBlock, matmul_tiling::MatmulApiTiling& bmm1,
                                                     matmul_tiling::MatmulApiTiling& bmm2)
{
    if (!SetBmm1TilingInput(tmpS1BasicBlock, tmpS2BasicBlock, bmm1) ||
        !SetBmm2TilingInput(tmpS1BasicBlock, tmpS2BasicBlock, tmpDBasicBlock, bmm2)) {
        return false;
    }

    if (bmm1.GetTiling(tilingData.bmm1TilingData) == -1) {
        OP_LOGE(context_, "BMM1 tiling failed.");
        return false;
    }

    if (isSameAB && tilingData.bmm1TilingData.get_baseK() >= tilingData.bmm1TilingData.get_singleCoreK() &&
        tilingData.bmm1TilingData.get_depthA1() == BMM1_DEPTH_A1_2) {
        tilingData.bmm1TilingData.set_depthA1(BMM1_DEPTH_A1_3);
    }

    tilingData.bmm1TilingData.set_shareMode(0);
    tilingData.bmm1TilingData.set_shareL1Size(aicoreParams_.l1Size);
    tilingData.bmm1TilingData.set_shareL0CSize(aicoreParams_.l0cSize);

    if (bmm2.GetTiling(tilingData.bmm2TilingData) == -1) {
        OP_LOGE(context_, "BMM2 tiling failed.");
        return false;
    }

    tilingData.bmm2TilingData.set_shareMode(0);
    tilingData.bmm2TilingData.set_shareL1Size(aicoreParams_.l1Size);
    tilingData.bmm2TilingData.set_shareL0CSize(aicoreParams_.l0cSize);

    mm1BaseM = bmm1.GetBaseM();
    mm1BaseN = bmm1.GetBaseN();
    mm1BaseK = bmm1.GetBaseK();
    mm2BaseM = bmm2.GetBaseM();
    mm2BaseN = bmm2.GetBaseN();
    mm2BaseK = bmm2.GetBaseK();

    return true;
}

bool NsaCompressAttentionTilingBase::SetMatMulTiling(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock,
                                                     int64_t tmpDBasicBlock)
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        matmul_tiling::MatmulApiTiling bmm1(ascendcPlatform);
        matmul_tiling::MatmulApiTiling bmm2(ascendcPlatform);
        return SetMatMulTiling(tmpS1BasicBlock, tmpS2BasicBlock, tmpDBasicBlock, bmm1, bmm2);
    } else {
        OP_LOGD(context_, "platform info is null, use default info to generate matmul tiling.");
        matmul_tiling::MatmulApiTiling bmm1;
        matmul_tiling::MatmulApiTiling bmm2;
        return SetMatMulTiling(tmpS1BasicBlock, tmpS2BasicBlock, tmpDBasicBlock, bmm1, bmm2);
    }
}

void NsaCompressAttentionTilingBase::SetSoftMaxTiling()
{
    std::vector<int64_t> shapeVec = {s1BasicBlock * gSize, s2BasicBlock};
    ge::Shape srcShape(shapeVec);
    AscendC::SoftMaxTilingFunc(srcShape, sizeof(float), BLOCK_BYTE * SIZE_1024, tilingData.softmaxTilingData);
}

void NsaCompressAttentionTilingBase::TopKTiling()
{
    auto& importanceScoreParams = tilingData.importanceScoreParams;
    uint32_t S2size = s2BasicBlock / importanceScoreParams.get_isM(); // 4096/(l'/d)
    uint32_t S2sizeAlign32 = (S2size + BLOCK_BYTE - 1) / BLOCK_BYTE * BLOCK_BYTE;
    int32_t k = selectBlockCount;
    uint32_t topkBase =
        std::min((aicoreParams_.ubSize - 2 * SIZE_1024 - 20 * S2sizeAlign32) / (4 * S2sizeAlign32 + 8 * k),
                 static_cast<uint64_t>(s1VecBasicBlock));
    uint32_t dtypesize = 4; // float类型
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    AscendC::TopKTilingFunc(ascendcPlatform,
                            S2sizeAlign32, // inner
                            topkBase,      // outter
                            k,             // topk
                            dtypesize,
                            false, // isinitindex
                            AscendC::TopKMode::TOPK_NORMAL,
                            true, // islargest
                            tilingData.topkTilingData);
    uint32_t maxsize = 0;
    uint32_t minsize = 0;
    AscendC::GetTopKMaxMinTmpSize(ascendcPlatform,
                                  S2sizeAlign32, // inner
                                  topkBase,      // outter
                                  false,         // isReuseSource
                                  false,         // isInitIndex
                                  AscendC::TopKMode::TOPK_NORMAL,
                                  true, // islargest
                                  dtypesize, maxsize, minsize);
    tilingData.importanceScoreParams.set_tmpUbSize(static_cast<uint64_t>(maxsize));
}

bool NsaCompressAttentionTilingBase::CheckUbSize(uint64_t& s2ScoreLoopLen, uint64_t& outerLoop)
{
    uint64_t useUbSize = 0;
    uint64_t s2Aligned64B = (outerLoop + 15) / 16 * 16;
    if (gSize >= 4) {
        useUbSize += s1VecBasicBlock * gSize * s2ScoreLoopLen * sizeof(float) * 2 +
                     s2Aligned64B * s1VecBasicBlock * gSize * sizeof(float) * 2 +
                     s1VecBasicBlock * AlignUp(outerLoop, BLOCK_BYTE) * sizeof(uint8_t);
    } else {
        useUbSize += s1VecBasicBlock * gSize * s2ScoreLoopLen * sizeof(float) * 2 +
                     s2Aligned64B * s1VecBasicBlock * gSize * sizeof(float) * 2 +
                     s1VecBasicBlock * AlignUp(outerLoop, BLOCK_BYTE) * sizeof(uint8_t) + 16 * 1024;
    }
    if (useUbSize > aicoreParams_.ubSize) {
        return false;
    }
    return true;
}

void NsaCompressAttentionTilingBase::SetImportanceScoreParams()
{
    using namespace std;
    uint64_t isM = selectBlockSize / compressStride;
    uint64_t isN = compressBlockSize / compressStride;
    int64_t S2PerScore = isM + isN - 1;

    uint64_t maskDT = 1;
    uint64_t splitS2 = 0;
    if (gSize >= 4) {
        splitS2 = aicoreParams_.ubSize /
                  (sizeof(float) * s1VecBasicBlock * gSize * 2 +
                   s1VecBasicBlock * (gSize * 2 * sizeof(float) + maskDT) / isM); // s2方向可以加载的列数
    } else {
        splitS2 = (aicoreParams_.ubSize - 16 * 1024) /
                  (sizeof(float) * s1VecBasicBlock * gSize * 2 +
                   s1VecBasicBlock * (gSize * 2 * sizeof(float) + maskDT) / isM); // 增加 select tmpBuf
    }
    uint64_t s2ScoreLoopTimes = (splitS2 - S2PerScore) / isM + 1; // s2方向基本块需要loop的次数
    uint64_t s2ScoreLoopLen = s2ScoreLoopTimes * isM + isN - 1;   // s2方向基本块一次加载的长度
    s2ScoreLoopLen = s2ScoreLoopLen / FRACTAL_NUM * FRACTAL_NUM;  // datacopy 32B 对齐 且transpose 16 对齐

    uint64_t outerLoop =
        (s2ScoreLoopLen - S2PerScore) / isM + 1; // 需要满足(isM + isN - 1) + (outerLoop - 1) * isM <= s2ScoreLoopLen
    uint64_t innerLoop = isM + isN - 1;

    // 用实际outerLoop根据kernel算实际ub使用量，若大于ubSize则调整s2ScoreLoopLen - 16
    if (!CheckUbSize(s2ScoreLoopLen, outerLoop)) {
        s2ScoreLoopLen -= FRACTAL_NUM;
        outerLoop = (s2ScoreLoopLen - S2PerScore) / isM + 1;
    }
    uint64_t impScoreNums = CeilDiv(static_cast<uint64_t>(s2BasicBlock), isM);
    uint64_t s2Loop = CeilDiv(impScoreNums, outerLoop);
    uint64_t lastLoopLen = s2Size % (outerLoop * isM) + innerLoop - isM;
    lastLoopLen = (lastLoopLen + FRACTAL_NUM - 1) / FRACTAL_NUM * FRACTAL_NUM;
    uint64_t lastOuterLoop = impScoreNums - outerLoop * (s2Loop - 1);

    auto& importanceScoreParams = tilingData.importanceScoreParams;
    importanceScoreParams.set_isM(isM);
    importanceScoreParams.set_isN(isN);
    importanceScoreParams.set_outerLoop(outerLoop);
    importanceScoreParams.set_lastOuterLoop(lastOuterLoop);
    importanceScoreParams.set_innerLoop(innerLoop);
    importanceScoreParams.set_s2ScoreLoopLen(s2ScoreLoopLen);
    importanceScoreParams.set_lastLoopLen(lastLoopLen);
    importanceScoreParams.set_s2Loop(s2Loop);

    const uint32_t stackBufferSize = 0;

    std::vector<int64_t> shapeVec = {static_cast<int64_t>(s1VecBasicBlock * gSize),
                                     static_cast<int64_t>(s2ScoreLoopLen)};
    ge::Shape srcShape(shapeVec);
    AscendC::GetConfusionTransposeTilingInfo(srcShape, stackBufferSize, sizeof(float), MODE_7,
                                             importanceScoreParams.confusionTransposeTilingData);

    std::vector<int64_t> shapeVec2 = {static_cast<int64_t>((outerLoop + 15) / 16 * 16),
                                      static_cast<int64_t>(s1VecBasicBlock * gSize)};
    ge::Shape srcShape2(shapeVec2);
    AscendC::GetConfusionTransposeTilingInfo(srcShape2, stackBufferSize, sizeof(float), MODE_7,
                                             importanceScoreParams.confusionTransposeTilingData2);

    std::vector<int64_t> shapeVecLast = {static_cast<int64_t>(s1VecBasicBlock * gSize),
                                         static_cast<int64_t>(lastLoopLen)};
    ge::Shape srcShapeLast(shapeVecLast);
    AscendC::GetConfusionTransposeTilingInfo(srcShapeLast, stackBufferSize, sizeof(float), MODE_7,
                                             importanceScoreParams.confusionTransposeTilingDataLast);

    std::vector<int64_t> shapeVecLast2 = {static_cast<int64_t>((lastOuterLoop + 15) / 16 * 16),
                                          static_cast<int64_t>(s1VecBasicBlock * gSize)};
    ge::Shape srcShapeLast2(shapeVecLast2);
    AscendC::GetConfusionTransposeTilingInfo(srcShapeLast2, stackBufferSize, sizeof(float), MODE_7,
                                             importanceScoreParams.confusionTransposeTilingDataLast2);
}

ge::graphStatus NsaCompressAttentionTilingBase::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

class NsaCompressAttentionVarLenScoreTiling : public NsaCompressAttentionTilingBase {
public:
    explicit NsaCompressAttentionVarLenScoreTiling(gert::TilingContext* context)
        : NsaCompressAttentionTilingBase(context)
    {
        templateName = "NsaCompressAttentionVarLenScore";
    }
    ~NsaCompressAttentionVarLenScoreTiling() override = default;

protected:
    void CalcS1S2BasicBlock() override
    {
        s1BasicBlock = NSA_BASE_S1G_SIZE / gSize;
        s1VecBasicBlock = s1BasicBlock / CV_RATIO;
        s2BasicBlock = alignedS2; // 也是s2最大值
    }

    bool SetBmm1TilingInput([[maybe_unused]] int64_t tmpS1BasicBlock, [[maybe_unused]] int64_t tmpS2BasicBlock,
                            matmul_tiling::MatmulApiTiling& bmm1) override
    {
        bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, true);
        bmm1.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm1OutDtype);

        bmm1.SetShape(NSA_BASE_S1G_SIZE, s2Size, dSize);
        bmm1.SetOrgShape(s1Size, s2Size, dSize);
        bmm1.SetBias(false);
        if (bmm1.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize) != 0) {
            return false;
        }
        return true;
    }

    bool SetBmm2TilingInput([[maybe_unused]] int64_t tmpS1BasicBlock, [[maybe_unused]] int64_t tmpS2BasicBlock,
                            [[maybe_unused]] int64_t tmpDBasicBlock, matmul_tiling::MatmulApiTiling& bmm2) override
    {
        bmm2.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm2.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm2OutDtype);
        bmm2.SetShape(NSA_BASE_S1G_SIZE, dSize, s2Size);
        bmm2.SetOrgShape(s1Size, dSize, s2Size);
        bmm2.SetBias(false);
        if (bmm2.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize) != 0) {
            return false;
        }
        return true;
    }

    uint64_t GetTilingKey() const override
    {
        return GET_TILINGKEY(tilingKeyLayout, SparseEnum::ANY, hasAttenMask, hasTopkMask);
    }

    bool IsCapable() override
    {
        if (tilingKeyLayout != LayoutType::LAYOUT_TND) {
            return false;
        }
        isSameAB = true;
        return true;
    }

    void SetMultiCoreParams() override
    {
        auto& multiCoreParams = tilingData.multiCoreParams;
        auto& inputParams = tilingData.inputParams;
        accumS1BlockNum = 0;
        for (int64_t i = 0; i < bSize; i++) {
            OP_LOGD(context_,
                    "[%s]"
                    "actualSeqLenData data %ld is %ld.",
                    templateName, i, actualSeqLenData[i]);
            OP_LOGD(context_,
                    "[%s]"
                    "actualSeqLenKvData data %ld is %ld.",
                    templateName, i, actualSeqLenKvData[i]);
            accumS1BlockNum += CeilDivision(actualSeqLenData[i], s1BasicBlock);
        }
        int64_t totalSize = accumS1BlockNum * inputParams.get_n2Size();
        int64_t actualUsedAivNum = std::min(totalSize, static_cast<int64_t>(aivNum));
        int64_t actualUsedAicNum = std::min(totalSize, static_cast<int64_t>(aicNum));
        int64_t actualUsedAiCoreNum = isSameAB ? actualUsedAicNum * 2 : actualUsedAivNum;
        int64_t actualSplitAiCoreNum = isSameAB ? actualUsedAicNum : actualUsedAivNum;
        multiCoreParams.set_coreNum(static_cast<int32_t>(actualUsedAiCoreNum));
        multiCoreParams.set_totalSize(totalSize);
        multiCoreParams.set_splitFactorSize(CeilDivision(totalSize, actualSplitAiCoreNum));
        multiCoreParams.set_splitFactorTailSize(CalcTailSize(totalSize, multiCoreParams.get_splitFactorSize()));
    }

    bool IsTemplateMatched() const override
    {
        return true;
    }

    ge::graphStatus GetWorkspaceSize() override
    {
        size_t* workspaces = context_->GetWorkspaceSizes(1);
        int64_t mm1BaseCount = NSA_BASE_S1G_SIZE * alignedS2;

        uint64_t pWorkspaceSize64 =
            static_cast<uint64_t>(mm1BaseCount) * sizeof(float) * NSA_DOUBLE_BUFFER * static_cast<uint64_t>(aicNum);
        uint64_t stage1ResWorkspaceSize64 =
            static_cast<uint64_t>(mm1BaseCount) * sizeof(inputDtype) * static_cast<uint64_t>(aicNum);
        uint64_t baseS1G = static_cast<uint64_t>(NSA_BASE_S1G_SIZE) / static_cast<uint64_t>(gSize);
        uint64_t alignedS2Div = CeilDiv(static_cast<uint64_t>(alignedS2), tilingData.importanceScoreParams.get_isM());
        uint64_t impScoreWorkspaceSize64 =
            baseS1G * alignedS2Div * sizeof(float) * NSA_DOUBLE_BUFFER * static_cast<uint64_t>(aicNum);

        uint64_t totalWorkspace64 =
            pWorkspaceSize64 + impScoreWorkspaceSize64 + stage1ResWorkspaceSize64 + WORK_SPACE_RESERVE_SIZE;
        workspaces[0] = static_cast<size_t>(totalWorkspace64);
        return ge::GRAPH_SUCCESS;
    }

    void SetSoftMaxTiling() override
    {
        return;
    }
};

// NOTE manually initialize tiling data in hostapi scenario in highest priority template
REGISTER_OPS_TILING_TEMPLATE(NsaCompressAttention, NsaCompressAttentionVarLenScoreTiling, 0);
} // namespace NsaCompressAttention
} // namespace optiling
