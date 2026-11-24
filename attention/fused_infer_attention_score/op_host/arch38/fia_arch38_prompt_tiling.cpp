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
 * \file fia_arch38_prompt_tiling.cpp
 * \brief
 */
#include "fia_arch38_prompt_tiling.h"
#include <queue>
#include <vector>

#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "op_host/data_copy_transpose_tiling.h"
#include "log/log.h"
#include "err/ops_err.h"
#include "../../op_kernel/arch38/fia_arch38_policy.h"
#include <unordered_map>
#include <string>

namespace {
static const std::unordered_map<ge::DataType, std::string> g_strDataTypeArch38 = {
    {ge::DT_FLOAT, "DT_FLOAT"},
    {ge::DT_FLOAT16, "DT_FLOAT16"},
    {ge::DT_INT8, "DT_INT8"},
    {ge::DT_INT16, "DT_INT16"},
    {ge::DT_UINT16, "DT_UINT16"},
    {ge::DT_UINT8, "DT_UINT8"},
    {ge::DT_INT32, "DT_INT32"},
    {ge::DT_INT64, "DT_INT64"},
    {ge::DT_UINT32, "DT_UINT32"},
    {ge::DT_UINT64, "DT_UINT64"},
    {ge::DT_BOOL, "DT_BOOL"},
    {ge::DT_DOUBLE, "DT_DOUBLE"},
    {ge::DT_STRING, "DT_STRING"},
    {ge::DT_DUAL_SUB_INT8, "DT_DUAL_SUB_INT8"},
    {ge::DT_DUAL_SUB_UINT8, "DT_DUAL_SUB_UINT8V"},
    {ge::DT_COMPLEX64, "DT_COMPLEX64"},
    {ge::DT_COMPLEX128, "DT_COMPLEX128"},
    {ge::DT_QINT8, "DT_QINT8"},
    {ge::DT_QINT16, "DT_QINT16"},
    {ge::DT_QINT32, "DT_QINT32"},
    {ge::DT_QUINT8, "DT_QUINT8"},
    {ge::DT_QUINT16, "DT_QUINT16"},
    {ge::DT_RESOURCE, "DT_RESOURCE"},
    {ge::DT_STRING_REF, "DT_STRING_REF"},
    {ge::DT_DUAL, "DT_DUAL"},
    {ge::DT_VARIANT, "DT_VARIANT"},
    {ge::DT_BF16, "DT_BF16"},
    {ge::DT_HIFLOAT8, "DT_HIFLOAT8"},
    {ge::DT_FLOAT8_E4M3FN, "DT_FLOAT8_E4M3FN"},
    {ge::DT_UNDEFINED, "DT_UNDEFINED"},
};
} // namespace

namespace optiling {
namespace arch38 {
std::string GetPfaDataTypeStr(ge::DataType type)
{
    ge::DataType findDype = (g_strDataTypeArch38.find(type) == g_strDataTypeArch38.end()) ? ge::DT_UNDEFINED : type;
    return g_strDataTypeArch38.at(findDype);
}

bool IsArch38IncreScenarioSupported(const FiaArch38IncreContext& ifaContext, const gert::TilingContext* context)
{
    // 只校验 IFA 增量场景特有边界；平台级约束（layout/head size/dQ==dV/dtype/D256 mask）
    // 由 PFA 模板的 CheckArch38PlatformSupported 统一兜底，此处不重复。
    const char* opName = context->GetNodeName();
    // PagedAttention lays key out by block, its shape carries no real kv seq length for the check below
    OP_CHECK_IF((ifaContext.blockTable.tensor != nullptr),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "blockTable", "not null",
                                                      "PagedAttention is not supported on this platform."),
                return false);

    OP_CHECK_IF(((ifaContext.query.shape == nullptr) || (ifaContext.key.shape == nullptr) ||
                 (ifaContext.value.shape == nullptr)),
                OP_LOGE_WITHOUT_REPORT(opName, "shape of query, key or value is null."), return false);

    const auto& queryShape = ifaContext.query.shape->GetStorageShape();
    const auto& keyShape = ifaContext.key.shape->GetStorageShape();
    const auto& valueShape = ifaContext.value.shape->GetStorageShape();
    OP_CHECK_IF(((queryShape.GetDimNum() != BNSD_DIM_NUM) || (keyShape.GetDimNum() != BNSD_DIM_NUM) ||
                 (valueShape.GetDimNum() != BNSD_DIM_NUM)),
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                    opName, "query, key and value",
                    (std::to_string(queryShape.GetDimNum()) + ", " + std::to_string(keyShape.GetDimNum()) + ", " +
                     std::to_string(valueShape.GetDimNum()))
                        .c_str(),
                    "BNSD layout requires 4-dim query, key and value shapes."),
                return false);

    const int64_t sQ = queryShape.GetDim(BNSD_DIM_S);
    OP_CHECK_IF((sQ != 1),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "query", std::to_string(sQ).c_str(),
                                                      "query seq length (S dimension) must be 1."),
                return false);

    const int64_t sKV = keyShape.GetDim(BNSD_DIM_S);
    OP_CHECK_IF(((sKV <= 1) || (sKV > ARCH38_IFA_SUPPORTED_MAX_SKV)),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    opName, "key", std::to_string(sKV).c_str(),
                    ("kv seq length must be in (1, " + std::to_string(ARCH38_IFA_SUPPORTED_MAX_SKV) + "].").c_str()),
                return false);

    OP_CHECK_IF((ifaContext.numHeads == nullptr), OP_LOGE_WITHOUT_REPORT(opName, "numHeads got from ge is null."),
                return false);
    const int64_t nQ = static_cast<int64_t>(*ifaContext.numHeads);
    // numKeyValueHeads defaults to numHeads when it is absent or left as 0
    const int64_t nKV = ((ifaContext.kvHeadNums != nullptr) && (*ifaContext.kvHeadNums != 0U)) ?
                            static_cast<int64_t>(*ifaContext.kvHeadNums) :
                            nQ;
    OP_CHECK_IF(((nQ <= 0) || (nQ > ARCH38_IFA_SUPPORTED_MAX_N) || (nKV <= 0) || (nKV > ARCH38_IFA_SUPPORTED_MAX_N)),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    opName, "numHeads, numKeyValueHeads", (std::to_string(nQ) + ", " + std::to_string(nKV)).c_str(),
                    ("head num must be in (0, " + std::to_string(ARCH38_IFA_SUPPORTED_MAX_N) + "].").c_str()),
                return false);

    return true;
}
} // namespace arch38
} // namespace optiling

using namespace ge;
using namespace AscendC;
using namespace matmul_tiling;
namespace optiling {

namespace arch38 {
namespace {
inline void ReportTilingFailure(const gert::TilingContext* context, const char* reason)
{
    OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "%s\n", reason);
    const std::vector<const char*> msgKey = {"op_name", "reason"};
    const std::vector<const char*> msgvalue = {context->GetNodeName(), reason};
    REPORT_PREDEFINED_ERR_MSG("EZ1008", msgKey, msgvalue);
}
} // namespace
constexpr uint32_t INPUT_QKV_SHAPE_MIN_DIMS = 3;
constexpr uint32_t INPUT_QKV_SHAPE_MAX_DIMS = 4;

constexpr uint32_t BYTE_BLOCK = 32; // The block size of datacopy, which moves data at the block granularity.

constexpr uint32_t MASKDIM_2 = 2;
constexpr uint32_t MASKDIM_3 = 3;
constexpr uint32_t MASKDIM_4 = 4;
constexpr uint32_t PSESHIFTDIM_4 = 4;

constexpr uint32_t KV_CACHE_DIM_NUMS_4 = 4;
constexpr uint32_t KV_CACHE_DIM_NUMS_3 = 3;
constexpr uint32_t KV_CACHE_DIM_0 = 0;
constexpr uint32_t KV_CACHE_DIM_1 = 1;
constexpr uint32_t KV_CACHE_DIM_2 = 2;
constexpr uint32_t KV_CACHE_DIM_3 = 3;

constexpr uint64_t EMPTY_KV_TILING_KEY = 20;
constexpr uint32_t LOOP_BEGIN_NUM = 0;
constexpr uint32_t SPARSE_MODE_NO_MASK = 0;
constexpr uint32_t SPARSE_MODE_ALL_MASK = 1;
constexpr uint32_t SPARSE_MODE_LEFT_UP = 2;
constexpr uint32_t SPARSE_MODE_RIGHT_DOWN = 3;
constexpr uint32_t SPARSE_MODE_BAND = 4;
constexpr int64_t SPARSE_MODE_INT_MAX = 2147483647;
constexpr uint32_t SPARSE_OPTIMIZE_ATTENTION_SIZE = 2048;
// The current requirement is a multiple of 128, and to prevent cross block handling, the mm base is also set to 128.
constexpr int32_t BLOCK_SIZE_BASE = 128;
constexpr int32_t BLOCK_SIZE_MAX = 512;

constexpr uint32_t SOUTER_FACTOR_SUB = 32;
constexpr uint32_t SOUTER_FACTOR_DEFAULT = 64;
constexpr uint32_t SINNER_FACTOR_SUB = 64;
constexpr uint32_t SINNER_FACTOR_DEFAULT = 128;
constexpr uint32_t SINNER_FACTOR_DOUBLE = 256;

uint32_t CV_RATIO = 2U; // Vector : Cube
constexpr uint32_t MATMUL_NORM_MIN_SEQ = 128;
constexpr uint32_t MATMUL_NORM_MIN_HEADSIZE = 128;

constexpr uint32_t BLIMIT = 65536;
constexpr uint32_t NLIMIT = 256;      // n <= 256
constexpr uint32_t SLIMIT = 20971520; // s, kvs <= 20MB
constexpr uint32_t DLIMIT = 512;      // D <= 512
constexpr uint32_t HLIMIT = 65535;    // warning: H <= 65536

// Validated scenario boundary of this platform, out-of-range cases are rejected instead of silently mis-computed.
constexpr int64_t ARCH38_SUPPORTED_MAX_SEQ = 4096;
constexpr int64_t ARCH38_SUPPORTED_MAX_N = 128;

constexpr uint32_t MLA_QKD_SIZE = 192;
constexpr uint32_t MLA_VD_SIZE = 128; // typical scene for PFA MLA, can be deleted after subsequent generalization.

static const int64_t GM_ALIGN = 512;
static const int64_t SLOPE_N_DIM_NUM = 1L;
uint64_t BENCHMARK_TILING_KEY = 1000000000000000000;
constexpr uint32_t ATTR_SPARSE_MODE = 6;
constexpr uint32_t ATTR_INNER_PRECISE = 7;
constexpr int64_t PSE_TYPE_2_TILING_V2 = 2;
constexpr int64_t PSE_TYPE_3_TILING_V2 = 3;
constexpr uint32_t QUERY_SHAPE_DIM_D_128_TILING_V2 = 128;
constexpr int32_t ROPE_DIMENSION_SIZE_TILING_V2 = 64;

constexpr uint32_t BATCH_MODE_SCHEDULE = 1;

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
static auto CeilDivision(T num1, T num2) -> T
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2;
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

template <typename T>
static auto CeilDiv(const T n1, const T n2) -> T
{
    if (n1 == 0) {
        return 0;
    }
    return (n2 != 0) ? (((n1 - 1) / n2) + 1) : n1;
}

enum class LayoutType : uint8_t {
    NONE = 0,
    LAYOUT_BSH = 1,
    LAYOUT_BSND = 1,
    LAYOUT_SBH = 2,
    LAYOUT_BNSD = 3,
    LAYOUT_TND = 4,
};

enum class PfaSparseEnum : uint8_t {
    PFA_ALL = 0,
    PFA_NONE = 1,
    PFA_ANY = 2,
    PFA_CAUSAL = 3,
    PFA_BAND = 4,
    PFA_PREFIX = 5,
    PFA_BAND_COMPRESS = 6,
    PFA_RIGHT_DOWN_CAUSAL = 7,
    PFA_RIGHT_DOWN_CAUSAL_BAND = 8,
    PFA_BAND_LEFT_UP_CAUSAL = 9
};

enum PfaAttenMaskCompressMode : uint8_t {
    PFA_NO_COMPRESS_MODE = 0,
    PFA_LEFT_UP_CAUSAL_MODE,
    PFA_RIGHT_DOWN_CAUSAL_MODE,
    PFA_BAND_MODE,
    PFA_PREFIX_MODE,
    PFA_RIGHT_DOWN_CAUSAL_BAND_MODE,
    PFA_BAND_LEFT_UP_CAUSAL_MODE
};

enum class PfaPseType : uint8_t {
    PSE_OUTER_MUL_ADD_TYPE = 0, // v2 default
    PSE_OUTER_ADD_MUL_TYPE,     // v1 current usage
    PSE_INNER_MUL_ADD_TYPE,
    PSE_INNER_MUL_ADD_SQRT_TYPE,
    PSE_INVALID_TYPE,
    PSE_NONE_TYPE = 9
};

enum class PfaPseShapeType : uint8_t {
    PSE_B_N2_G_S1_S2 = 0,
    PSE_B_N2_G_1_S2 = 1,
    PSE_B_N2_G_SLOPE,
    PSE_1_N2_G_SLOPE
};

bool FiaArch38PromptTiling::CheckNonEmptyShapeExceptions(const FiaArch38ContextParams& contextKeyParams,
                                                         const gert::StorageShape* shape,
                                                         const std::string& sName) const
{
    OP_CHECK_IF(shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(contextKeyParams.opName, sName.c_str()), return false);
    OP_CHECK_IF(shape->GetStorageShape().GetShapeSize() == gert::Shape::kInvalidDimValue,
                ReportTilingFailure(context_, "get the shape size of tensor failed"), return false);
    return true;
}

void FiaArch38PromptTiling::FiaArch38InitOutputSplit(int64_t totalSize, FiaArch38FlashAttentionTilingData& tilingData)
{
    FiaArch38AttentionInitOutputParams* initParams = &tilingData.fiaArch38AttentionInitOutputParams;
    // Upward rounding, coreNum has been verified to be non-zero when obtained.
    uint32_t singleCoreSize = (totalSize + coreNum - 1) / (coreNum);

    if (enablePostQuant) {
        // requiring that the number of points allocated to each kernel must be even.
        singleCoreSize = ((singleCoreSize + 1) / 2) * 2; // 2 : fill in 0
    }

    initParams->set_singleCoreSize(singleCoreSize);
    initParams->set_totalOutputSize(totalSize);
}

bool FiaArch38PromptTiling::CheckEmptyTensor(FiaArch38ContextParams& contextKeyParams) const
{
    return (contextKeyParams.keyInputShape->GetStorageShape().GetShapeSize() == 0) ||
           (contextKeyParams.valueInputShape->GetStorageShape().GetShapeSize() == 0) ||
           (contextKeyParams.outputShape->GetStorageShape().GetShapeSize() == 0) ||
           (contextKeyParams.emptyTensor == 1U);
}

ge::graphStatus FiaArch38PromptTiling::SetEmptyTensor(FiaArch38ContextParams& contextKeyParams, uint64_t& tilingKey,
                                                      uint32_t& numBlocksToBeSet,
                                                      FiaArch38FlashAttentionTilingData& tilingData)
{
    tilingKey = EMPTY_KV_TILING_KEY;
    FiaArch38InitOutputSplit(contextKeyParams.outputShape->GetStorageShape().GetShapeSize(), tilingData);
    tilingData.fiaArch38AttentionInitOutputParams.set_needInit(1);

    auto platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr, OP_LOGE_WITHOUT_REPORT(contextKeyParams.opName, "platformInfoPtr is null!"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    numBlocksToBeSet = ascendcPlatform.CalcTschBlockDim(coreNum, aicNum, coreNum);

    size_t* workspace = contextKeyParams.workspaceSize;
    const size_t sysWorkspaceSize = 16 * 1024 * 1024; // minimum size required by workspace
    workspace[0] = sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

bool FiaArch38PromptTiling::CheckIODataType(FiaArch38ContextParams& contextKeyParams)
{
    outputType = contextKeyParams.outputDataType;
    inputType = contextKeyParams.inputDataType;

    std::vector<ge::DataType> allowedDtypes = {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, ge::DT_HIFLOAT8,
                                               ge::DT_FLOAT8_E4M3FN};
    std::vector<uint32_t> dataTypeSizeArray = {FLOAT16SIZE, BFLOAT16SIZE, INT8SIZE, FLOAT8SIZE, FLOAT8SIZE, FLOAT8SIZE};

    auto inputTypeCheck = std::find(allowedDtypes.begin(), allowedDtypes.end(), inputType);
    auto outputTypeCheck = std::find(allowedDtypes.begin(), allowedDtypes.end(), outputType);
    if (inputTypeCheck != allowedDtypes.end()) {
        uint32_t inputTypeIndex = std::distance(allowedDtypes.begin(), inputTypeCheck);
        dataTypeSize = dataTypeSizeArray[inputTypeIndex];
    }
    maskElemSize = dataTypeSize;
    if (outputTypeCheck != allowedDtypes.end()) {
        uint32_t outputTypeIndex = std::distance(allowedDtypes.begin(), outputTypeCheck);
        outputDataTypeSize = dataTypeSizeArray[outputTypeIndex];
    }

    if (inputType != ge::DT_FLOAT16) {
        OP_LOGW(contextKeyParams.opName, "innerprecise will not take effect when input type is %s!",
                GetPfaDataTypeStr(inputType).c_str());
    }
    OP_CHECK_IF((inputTypeCheck == allowedDtypes.end() || outputTypeCheck == allowedDtypes.end()),
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    contextKeyParams.opName, "inputType, outputType",
                    (GetPfaDataTypeStr(inputType) + ", " + GetPfaDataTypeStr(outputType)).c_str(),
                    "inputType and outputType should be DT_FLOAT16, DT_BF16 or DT_INT8"),
                return false);
    return true;
}

bool FiaArch38PromptTiling::SetInputLayout(const char* layout)
{
    if (layout == nullptr) {
        inputLayout = InputLayout::BSH;
        return true;
    }

    std::string layoutStr(layout);
    if (layoutStr == "" || layoutStr == "BSH") {
        inputLayout = InputLayout::BSH;
    } else if (layoutStr == "TND") {
        inputLayout = InputLayout::TND;
    } else if (layoutStr == "BSND") {
        inputLayout = InputLayout::BSND;
    } else if (layoutStr == "BNSD" || layoutStr == "BNSD_BSND") { // Reuse BNSD process for BNSD_BSND
        inputLayout = InputLayout::BNSD;
    } else {
        return false;
    }

    return true;
}

int64_t GetMaxSeq(const gert::Tensor* actualSeqLength)
{
    int64_t max = actualSeqLength->GetData<int64_t>()[0];
    for (int i = 1; i < actualSeqLength->GetShapeSize(); ++i) {
        max = std::max(max, actualSeqLength->GetData<int64_t>()[i] - actualSeqLength->GetData<int64_t>()[i - 1]);
    }
    return max;
}

bool FiaArch38PromptTiling::SetShape(FiaArch38ContextParams& contextKeyParams, const gert::StorageShape* shape,
                                     const std::string inputName, int64_t& b, int64_t& n, int64_t& s, int64_t& d,
                                     int64_t& h, int64_t& t) const
{
    // when enablePA, k_v has two shapes:
    // 1. [>=blockNums, N, blockSize, D]
    // 2. [>=blockNums, blockSize, H]
    if (enablePA && (inputName != "query")) {
        if (shape->GetStorageShape().GetDimNum() == 3) { // 3 for dim num
            b = shape->GetStorageShape().GetDim(0);
            s = shape->GetStorageShape().GetDim(1);
            h = shape->GetStorageShape().GetDim(2); // 2 for H dim
            n = static_cast<int64_t>(*contextKeyParams.numKeyValueHeads);
            n = n > 0 ? n : static_cast<int64_t>(*contextKeyParams.headsNumber);
            d = n > 0 ? h / n : 0;
        } else {
            b = shape->GetStorageShape().GetDim(0);
            n = shape->GetStorageShape().GetDim(1);
            s = shape->GetStorageShape().GetDim(2); // 2 for Sequence length
            d = shape->GetStorageShape().GetDim(3); // 3 for D dim
            h = n * d;
        }
    } else if ((inputLayout == InputLayout::BNSD)) {
        b = shape->GetStorageShape().GetDim(0);
        n = shape->GetStorageShape().GetDim(1);
        s = shape->GetStorageShape().GetDim(2); // 2 for Sequence length
        d = shape->GetStorageShape().GetDim(3); // 3 for D dim
        h = n * d;
    } else if ((inputLayout == InputLayout::BSH)) {
        b = shape->GetStorageShape().GetDim(0);
        s = shape->GetStorageShape().GetDim(1);
        h = shape->GetStorageShape().GetDim(2); // 2 for H dim
        if (inputName == "query") {
            n = static_cast<int64_t>(*contextKeyParams.headsNumber);
        } else {
            n = static_cast<int64_t>(*contextKeyParams.numKeyValueHeads);
            n = n > 0 ? n : static_cast<int64_t>(*contextKeyParams.headsNumber);
        }
        d = n > 0 ? h / n : 0;
    } else if ((inputLayout == InputLayout::BSND)) {
        b = shape->GetStorageShape().GetDim(0);
        s = shape->GetStorageShape().GetDim(1);
        n = shape->GetStorageShape().GetDim(2); // 2 for head dim
        d = shape->GetStorageShape().GetDim(3); // 3 for D dim
        h = n * d;
    } else if ((inputLayout == InputLayout::TND)) {
        b = static_cast<int64_t>(contextKeyParams.actualSequenceLengthQ->GetShapeSize());
        s = (inputName == "query") ? GetMaxSeq(contextKeyParams.actualSequenceLengthQ) :
                                     GetMaxSeq(contextKeyParams.actualSequenceLengthKV);
        t = shape->GetStorageShape().GetDim(0);
        n = shape->GetStorageShape().GetDim(1);
        d = shape->GetStorageShape().GetDim(2); // 2 for D dim
        h = n * d;
    } else {
        return false;
    }
    return true;
}

bool FiaArch38PromptTiling::GetAndCheckShape(FiaArch38ContextParams& contextKeyParams, PFAShapeInfo& shapeInfo,
                                             const gert::StorageShape* shape, const std::string& sName) const
{
    std::string layoutStr(contextKeyParams.layout);
    OP_CHECK_IF(
        (shape->GetStorageShape().GetDimNum() != 3) &&
            (inputLayout == InputLayout::BSH || inputLayout == InputLayout::TND),
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            contextKeyParams.opName, sName.c_str(), std::to_string(shape->GetStorageShape().GetDimNum()).c_str(),
            (std::string("dim num of ") + sName + " must be 3 when inputLayout is " + layoutStr).c_str()),
        return false);
    OP_CHECK_IF(
        (shape->GetStorageShape().GetDimNum() != 4) &&
            (inputLayout != InputLayout::BSH && inputLayout != InputLayout::TND),
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            contextKeyParams.opName, sName.c_str(), std::to_string(shape->GetStorageShape().GetDimNum()).c_str(),
            (std::string("dim num of ") + sName + " must be 4 when inputLayout is " + layoutStr).c_str()),
        return false);

    int64_t b = 0;
    int64_t n = 0;
    int64_t s = 0;
    int64_t d = 0;
    int64_t h = 0;
    int64_t t = 0;
    if (!SetShape(contextKeyParams, shape, sName, b, n, s, d, h, t)) {
        return false;
    }
    OP_CHECK_IF(
        (b > BLIMIT || b <= 0),
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(contextKeyParams.opName, sName.c_str(), std::to_string(b).c_str(),
                                                  "batch size of tensor must be within (0, BLIMIT]"),
        return false);
    OP_CHECK_IF(
        (s <= 0),
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(contextKeyParams.opName, sName.c_str(), std::to_string(s).c_str(),
                                                  "seq size of tensor must be greater than 0"),
        return false);
    if (s > SLIMIT) {
        OP_LOGW(contextKeyParams.opName, "sequence size of %s should <= 20M, but seq = %ld.", sName.c_str(), s);
    }
    if (inputLayout == InputLayout::BSH) {
        OP_CHECK_IF(
            (d > DLIMIT || d <= 0),
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(contextKeyParams.opName, sName.c_str(), std::to_string(d).c_str(),
                                                      "d size of tensor must be within (0, DLIMIT]"),
            return false);
        OP_CHECK_IF(
            (n > NLIMIT || n <= 0),
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(contextKeyParams.opName, sName.c_str(), std::to_string(n).c_str(),
                                                      "n size of tensor must be within (0, NLIMIT]"),
            return false);
    } else {
        OP_CHECK_IF(
            (n > NLIMIT || n <= 0),
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(contextKeyParams.opName, sName.c_str(), std::to_string(n).c_str(),
                                                      "n size of tensor must be within (0, NLIMIT]"),
            return false);
        OP_CHECK_IF(
            (d > DLIMIT || d <= 0),
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(contextKeyParams.opName, sName.c_str(), std::to_string(d).c_str(),
                                                      "d size of tensor must be within (0, DLIMIT]"),
            return false);
    }
    if (((inputLayout == InputLayout::BSH) || (inputLayout == InputLayout::BSND)) && (h > HLIMIT)) {
        OP_LOGW(contextKeyParams.opName,
                "layout = %s, h(%ld) size of %s should be less than or equal to 65535, "
                "which may cause precision problem! Please use BNSD or BNSD_BSND instead.",
                contextKeyParams.layout, h, sName.c_str());
    }
    OP_CHECK_IF(
        (inputLayout == InputLayout::TND) && (t <= 0),
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(contextKeyParams.opName, sName.c_str(), std::to_string(t).c_str(),
                                                  "when layout is TND, t must be greater than 0"),
        return false);
    shapeInfo.b = static_cast<uint32_t>(b);
    shapeInfo.n = static_cast<uint64_t>(n);
    shapeInfo.s = static_cast<uint64_t>(s);
    shapeInfo.d = static_cast<uint32_t>(d);
    shapeInfo.h = static_cast<uint64_t>(h);
    shapeInfo.t = static_cast<uint32_t>(t);
    return true;
}

bool FiaArch38PromptTiling::GetAndCheckRopeShape(FiaArch38ContextParams& contextKeyParams, PFAShapeInfo& shapeInfo,
                                                 PFAShapeInfo& ropeShapeInfo, const gert::StorageShape* shape,
                                                 const std::string& sName, const std::string& rName) const
{
    std::string layoutStr(contextKeyParams.layout);
    OP_CHECK_IF(
        (shape->GetStorageShape().GetDimNum() != 3) &&
            (inputLayout == InputLayout::BSH || inputLayout == InputLayout::TND),
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            contextKeyParams.opName, sName.c_str(), std::to_string(shape->GetStorageShape().GetDimNum()).c_str(),
            (std::string("dim num of ") + sName + " must be 3 when inputLayout is " + layoutStr).c_str()),
        return false);
    OP_CHECK_IF(
        (shape->GetStorageShape().GetDimNum() != 4) &&
            (inputLayout != InputLayout::BSH && inputLayout != InputLayout::TND),
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            contextKeyParams.opName, sName.c_str(), std::to_string(shape->GetStorageShape().GetDimNum()).c_str(),
            (std::string("dim num of ") + sName + " must be 4 when inputLayout is " + layoutStr).c_str()),
        return false);

    OP_CHECK_IF((shapeInfo.d != 128 && shapeInfo.d != 512),
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    contextKeyParams.opName, sName.c_str(), std::to_string(shapeInfo.d).c_str(),
                    "when queryRope and keyRope is not null, d must be 128 or 512"),
                return false);

    int64_t b = 0;
    int64_t n = 0;
    int64_t s = 0;
    int64_t d = 0;
    int64_t h = 0;
    int64_t t = 0;
    if (!SetShape(contextKeyParams, shape, sName, b, n, s, d, h, t)) {
        return false;
    }

    if (inputLayout == InputLayout::BSH) {
        OP_CHECK_IF((d != 64),
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                        contextKeyParams.opName, rName.c_str(), std::to_string(d).c_str(),
                        "when queryRope and keyRope is not null, d must be 64 when layout is BSH"),
                    return false);
    } else {
        OP_CHECK_IF(
            (d != 64),
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(contextKeyParams.opName, rName.c_str(), std::to_string(d).c_str(),
                                                      "when queryRope and keyRope is not null, d must be 64"),
            return false);
    }

    if (!enableTensorList || sName == "query") {
        OP_CHECK_IF(
            (b != shapeInfo.b),
            OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(contextKeyParams.opName, (rName + ", " + sName).c_str(),
                                                       (std::to_string(b) + ", " + std::to_string(shapeInfo.b)).c_str(),
                                                       "batch size of rope must be equal to batch size of tensor"),
            return false);
    }
    OP_CHECK_IF(
        (n != shapeInfo.n),
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(contextKeyParams.opName, (rName + ", " + sName).c_str(),
                                                   (std::to_string(n) + ", " + std::to_string(shapeInfo.n)).c_str(),
                                                   "head number of rope must be equal to head number of tensor"),
        return false);
    OP_CHECK_IF(
        (s != shapeInfo.s),
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(contextKeyParams.opName, (rName + ", " + sName).c_str(),
                                                   (std::to_string(s) + ", " + std::to_string(shapeInfo.s)).c_str(),
                                                   "seq length of rope must be equal to seq length of tensor"),
        return false);
    OP_CHECK_IF(
        (t != shapeInfo.t),
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(contextKeyParams.opName, (rName + ", " + sName).c_str(),
                                                   (std::to_string(t) + ", " + std::to_string(shapeInfo.t)).c_str(),
                                                   "t of rope must be equal to t of tensor"),
        return false);

    if (((inputLayout == InputLayout::BSH) || (inputLayout == InputLayout::BSND)) &&
        (static_cast<uint32_t>(h) > HLIMIT)) {
        OP_LOGW(contextKeyParams.opName,
                "layout = %s, h(%ld) size of %s should be less than or equal to 65535, "
                "which may cause precision problem! Please use BNSD or BNSD_BSND instead.",
                contextKeyParams.layout, h, sName.c_str());
    }
    ropeShapeInfo.b = static_cast<uint32_t>(b);
    ropeShapeInfo.n = static_cast<uint64_t>(n);
    ropeShapeInfo.s = static_cast<uint64_t>(s);
    ropeShapeInfo.d = static_cast<uint32_t>(d);
    ropeShapeInfo.h = static_cast<uint64_t>(h);
    ropeShapeInfo.t = static_cast<uint32_t>(t);
    return true;
}

bool FiaArch38PromptTiling::CheckQueryOutParamsConsistency(const FiaArch38ContextParams& contextKeyParams,
                                                           const gert::StorageShape* queryShape,
                                                           const gert::StorageShape* outShape) const
{
    const size_t queryDimNum = queryShape->GetStorageShape().GetDimNum();
    const size_t outDimNum = outShape->GetStorageShape().GetDimNum();
    int64_t tmpqueryDim = 0;
    int64_t outDim = 0;
    std::string layoutStr(contextKeyParams.layout);

    OP_CHECK_IF(queryDimNum != outDimNum,
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                    contextKeyParams.opName, "query, attention_out",
                    (std::to_string(queryDimNum) + ", " + std::to_string(outDimNum)).c_str(),
                    "query dim num must be consistent with output dim num"),
                return false);
    OP_CHECK_IF(
        (queryDimNum < INPUT_QKV_SHAPE_MIN_DIMS) || (queryDimNum > INPUT_QKV_SHAPE_MAX_DIMS),
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(contextKeyParams.opName, "query", std::to_string(queryDimNum).c_str(),
                                                 "query dim num is invalid, only support 3 or 4 dimensions"),
        return false);

    for (uint32_t i = 0; i < queryDimNum; ++i) {
        if ((i == queryDimNum - 1) && enablePFAMLA) {
            continue;
        }
        if (layoutStr == "BNSD_BSND") {
            if (i == 1) { // query:N, output:S
                tmpqueryDim = queryShape->GetStorageShape().GetDim(i);
                outDim = outShape->GetStorageShape().GetDim(i + 1);
            } else if (i == 2) { // 2 for current queryDimNum; Q:N, Output:S
                tmpqueryDim = queryShape->GetStorageShape().GetDim(i);
                outDim = outShape->GetStorageShape().GetDim(i - 1);
            }
        } else {
            tmpqueryDim = queryShape->GetStorageShape().GetDim(i);
            outDim = outShape->GetStorageShape().GetDim(i);
        }
        OP_CHECK_IF(tmpqueryDim != outDim,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        contextKeyParams.opName, "query, attention_out",
                        (std::to_string(tmpqueryDim) + ", " + std::to_string(outDim)).c_str(),
                        "query and output shape must be equal in the same dim"),
                    return false);
    }
    return true;
}

bool FiaArch38PromptTiling::CheckKVDataType(FiaArch38ContextParams& contextKeyParams) const
{
    ge::DataType keyDataType = contextKeyParams.kDataType;
    ge::DataType valueDataType = contextKeyParams.vDataType;

    std::vector<ge::DataType> allowedDtypes = {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, ge::DT_HIFLOAT8,
                                               ge::DT_FLOAT8_E4M3FN};

    auto keyTypeCheck = std::find(allowedDtypes.begin(), allowedDtypes.end(), keyDataType);

    OP_CHECK_IF((keyDataType != valueDataType),
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(contextKeyParams.opName, "key, value",
                                                       (GetPfaDataTypeStr(contextKeyParams.kDataType) + ", " +
                                                        GetPfaDataTypeStr(contextKeyParams.vDataType))
                                                           .c_str(),
                                                       "key and value must have the same data type"),
                return false);
    OP_CHECK_IF((keyTypeCheck == allowedDtypes.end()),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(contextKeyParams.opName, "key",
                                                      GetPfaDataTypeStr(contextKeyParams.kDataType).c_str(),
                                                      "key dtype only supports DT_FLOAT16, DT_BF16 or DT_INT8"),
                return false);
    return true;
}

bool FiaArch38PromptTiling::CheckKeyValueParamsConsistency(FiaArch38ContextParams& contextKeyParams,
                                                           const gert::StorageShape* keyShape,
                                                           const gert::StorageShape* valueShape) const
{
    if (enableTensorList) {
        return true;
    }
    const size_t keyDimNum = keyShape->GetStorageShape().GetDimNum();
    const size_t valueDimNum = valueShape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(keyDimNum != valueDimNum,
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                    contextKeyParams.opName, "key, value",
                    (std::to_string(keyDimNum) + ", " + std::to_string(valueDimNum)).c_str(),
                    "key dim num must be consistent with value dim num"),
                return false);
    OP_CHECK_IF(
        (keyDimNum < INPUT_QKV_SHAPE_MIN_DIMS) || (keyDimNum > INPUT_QKV_SHAPE_MAX_DIMS),
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(contextKeyParams.opName, "key", std::to_string(keyDimNum).c_str(),
                                                 "key dim num is invalid, only support 3 or 4 dimensions"),
        return false);
    for (uint32_t i = 0; i < keyDimNum; ++i) {
        if ((i == keyDimNum - 1) && enablePFAMLA) {
            continue;
        }
        int64_t tmpKeyDim = keyShape->GetStorageShape().GetDim(i);
        int64_t tmpValueDim = valueShape->GetStorageShape().GetDim(i);
        OP_CHECK_IF(tmpKeyDim != tmpValueDim,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        contextKeyParams.opName, "key, value",
                        (std::to_string(tmpKeyDim) + ", " + std::to_string(tmpValueDim)).c_str(),
                        "key and value shape must be equal in the same dim"),
                    return false);
    }
    return true;
}

bool FiaArch38PromptTiling::CheckInputDimAndHeadNum(FiaArch38ContextParams& contextKeyParams, const uint32_t nQAttr,
                                                    const uint32_t nKVAttr)
{
    uint32_t nQ = nQAttr;
    uint32_t nKV = nKVAttr;
    if (nKVAttr == 0U) { // Detected that nKVAttr is the default value, which means that the customer did not pass in.
        nKV = nQAttr;
    }

    const gert::StorageShape* queryShape = contextKeyParams.queryInputShape;
    const gert::StorageShape* keyShape = contextKeyParams.keyInputShape;
    const gert::StorageShape* valueShape = contextKeyParams.valueInputShape;
    uint32_t queryShapeHeadNum = nQ;
    uint32_t keyShapeHeadNum = nKV;
    uint32_t valueShapeHeadNum = nKV;
    const size_t queryDim = queryShape->GetStorageShape().GetDimNum();
    const size_t keyDim = keyShape->GetStorageShape().GetDimNum();
    const size_t valueDim = valueShape->GetStorageShape().GetDimNum();
    const size_t nIdx =
        (inputLayout == InputLayout::BNSD || inputLayout == InputLayout::TND) ? 1U : 2U; // BNSD/TND: 1; BSND:2

    if (((inputLayout == InputLayout::BNSD) || (inputLayout == InputLayout::BSND)) && (!enablePA)) {
        if ((queryDim == 4) && (keyDim == 4) && (valueDim == 4)) { // dim num: 4
            queryShapeHeadNum = queryShape->GetStorageShape().GetDim(nIdx);
            keyShapeHeadNum = keyShape->GetStorageShape().GetDim(nIdx);
            valueShapeHeadNum = valueShape->GetStorageShape().GetDim(nIdx);
        } else {
            OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                contextKeyParams.opName, "query, key, value",
                (std::to_string(queryDim) + ", " + std::to_string(keyDim) + ", " + std::to_string(valueDim)).c_str(),
                "query, key and value must be 4D for BNSD or BSND format");
            return false;
        }
    } else if ((inputLayout == InputLayout::TND) && (!enablePA)) {
        if ((queryDim == 3) && (keyDim == 3) && (valueDim == 3)) { // dim num: 3
            queryShapeHeadNum = queryShape->GetStorageShape().GetDim(nIdx);
            keyShapeHeadNum = keyShape->GetStorageShape().GetDim(nIdx);
            valueShapeHeadNum = valueShape->GetStorageShape().GetDim(nIdx);
        } else {
            OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                contextKeyParams.opName, "query, key, value",
                (std::to_string(queryDim) + ", " + std::to_string(keyDim) + ", " + std::to_string(valueDim)).c_str(),
                "query, key and value must be 3D for TND format");
            return false;
        }
    }

    OP_CHECK_IF(nQ > 256U, // The maximum limit for head is 256.
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(contextKeyParams.opName, "num_heads", std::to_string(nQ).c_str(),
                                                      "numHeads should not be more than 256"),
                return false);
    OP_CHECK_IF(
        queryShapeHeadNum != nQ,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "query, num_heads",
                                               (std::to_string(queryShapeHeadNum) + ", " + std::to_string(nQ)).c_str(),
                                               "numHeads in query shape must be equal to numHeads in attr"),
        return false);
    OP_CHECK_IF(
        keyShapeHeadNum != nKV,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "key, num_key_value_heads",
                                               (std::to_string(keyShapeHeadNum) + ", " + std::to_string(nKV)).c_str(),
                                               "numHeads in key shape must match numKeyValueHeads in attr"),
        return false);
    OP_CHECK_IF(
        valueShapeHeadNum != nKV,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "value, num_key_value_heads",
                                               (std::to_string(valueShapeHeadNum) + ", " + std::to_string(nKV)).c_str(),
                                               "numHeads in value shape must match numKeyValueHeads in attr"),
        return false);
    if (enableIFAMLA || enableIFA) {
        gSize = nQ / nKV;
    }
    if ((nQ / nKV) > pfaMergeGLimit) {
        enablePFAMerge = false;
    }
    return true;
}

bool FiaArch38PromptTiling::SetAndCheckHeadNumRatio(FiaArch38ContextParams& contextKeyParams,
                                                    FiaArch38FlashAttentionTilingData& tilingData)
{
    const int64_t nQ = *contextKeyParams.headsNumber;
    const int64_t nKV = *contextKeyParams.numKeyValueHeads;

    if ((nQ < 0) || (nKV < 0)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "num_heads, num_key_value_heads",
                                               (std::to_string(nQ) + ", " + std::to_string(nKV)).c_str(),
                                               "numHeads and numKeyValueHeads must not be negative");
        return false;
    }

    if (!CheckInputDimAndHeadNum(contextKeyParams, nQ, nKV)) {
        return false;
    }

    if (nKV == 0) { // Detected that nKV is the default value, which means that the customer did not pass in.
        tilingData.fiaArch38AttentionBaseParams.set_headNumRatio(1);
        tilingData.fiaArch38AttentionBaseParams.set_gOfMla(1);
        return true;
    }

    if (enableIFAMLA || enableIFA) {
        tilingData.fiaArch38AttentionBaseParams.set_headNumRatio(1);
        tilingData.fiaArch38AttentionBaseParams.set_gOfMla(gSize);
        return true;
    } else {
        tilingData.fiaArch38AttentionBaseParams.set_gOfMla(1);
    }

    if (nQ % nKV != 0) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "num_heads, num_key_value_heads",
                                               (std::to_string(nQ) + ", " + std::to_string(nKV)).c_str(),
                                               "numHeads must be divisible by numKeyValueHeads");
        return false;
    } else {
        if (nQ / nKV > 64 && (!enablePFAMLA)) { // G cannot be greater than 64.
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(contextKeyParams.opName, "num_heads / num_key_value_heads",
                                                  std::to_string(nQ / nKV).c_str(),
                                                  "numHeads / numKeyValueHeads must not be larger than 64");
            return false;
        }
        tilingData.fiaArch38AttentionBaseParams.set_headNumRatio(nQ / nKV);
        return true;
    }
}

bool FiaArch38PromptTiling::CheckPostQuantShape(const FiaArch38ContextParams& contextKeyParams,
                                                const gert::StorageShape* quantOffset2Shape,
                                                const ge::DataType quantScale2Type, int64_t quantScale2ShapeSize,
                                                const PFAShapeInfo& queryShapeInfo,
                                                const PFAShapeInfo& valueShapeInfo) const
{
    // dtype verification
    OP_CHECK_IF(
        (quantOffset2Shape != nullptr) && (quantScale2Type != contextKeyParams.quantOffset2Type),
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            contextKeyParams.opName, "quant_scale2, quant_offset2",
            (GetPfaDataTypeStr(quantScale2Type) + ", " + GetPfaDataTypeStr(contextKeyParams.quantOffset2Type)).c_str(),
            "post quant scale and offset must have the same data type"),
        return false);

    OP_CHECK_IF((quantScale2Type != ge::DT_BF16) && (quantScale2Type != ge::DT_FLOAT),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(contextKeyParams.opName, "quant_scale2",
                                                      GetPfaDataTypeStr(quantScale2Type).c_str(),
                                                      "post quant scale dtype only supports bf16 and fp32"),
                return false);

    if (inputType == ge::DT_BF16) {
        OP_CHECK_IF(quantScale2Type != ge::DT_FLOAT && quantScale2Type != ge::DT_BF16,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        contextKeyParams.opName, "quant_scale2", GetPfaDataTypeStr(quantScale2Type).c_str(),
                        "post quant scale dtype only supports float32 and bf16 for this query dtype"),
                    return false);
    } else {
        OP_CHECK_IF(quantScale2Type != ge::DT_FLOAT,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        contextKeyParams.opName, "quant_scale2", GetPfaDataTypeStr(quantScale2Type).c_str(),
                        "post quant scale dtype only supports float32 for this query dtype"),
                    return false);
    }
    // shape verification
    if (quantOffset2Shape != nullptr) {
        int64_t quantOffset2ShapeSize = quantOffset2Shape->GetStorageShape().GetShapeSize();
        OP_CHECK_IF(quantScale2ShapeSize != quantOffset2ShapeSize,
                    OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                        contextKeyParams.opName, "quant_scale2, quant_offset2",
                        (std::to_string(quantScale2ShapeSize) + ", " + std::to_string(quantOffset2ShapeSize)).c_str(),
                        "quant_scale2 and quant_offset2 dimension multiply results must be equal"),
                    return false);
    }

    uint64_t quantScale2ShapeSizePerChannel =
        static_cast<uint64_t>(queryShapeInfo.n) * static_cast<uint64_t>(valueShapeInfo.d);
    OP_CHECK_IF(
        quantScale2ShapeSize < 0 || ((static_cast<uint64_t>(quantScale2ShapeSize) != 1U) &&
                                     (static_cast<uint64_t>(quantScale2ShapeSize) != quantScale2ShapeSizePerChannel)),
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            contextKeyParams.opName, "quant_scale2, quant_offset2", std::to_string(quantScale2ShapeSize).c_str(),
            "post quant scale2/offset2 dimension multiply result only supports 1 or qN * vD"),
        return false);
    return true;
}

bool FiaArch38PromptTiling::CheckPerTensorQuantParams(const FiaArch38ContextParams& contextKeyParams) const
{
    const gert::StorageShape* deqScale1Shape = contextKeyParams.deqScale1Shape;
    const gert::StorageShape* quantScale1Shape = contextKeyParams.scale1Shape;
    const gert::StorageShape* deqScale2Shape = contextKeyParams.deqScale2Shape;
    const ge::DataType inputParamsType = contextKeyParams.inputDataType;
    OP_CHECK_IF((inputParamsType != ge::DT_INT8) && (inputParamsType != ge::DT_HIFLOAT8) &&
                    (inputParamsType != ge::DT_FLOAT8_E4M3FN),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    contextKeyParams.opName, "query", GetPfaDataTypeStr(contextKeyParams.inputDataType).c_str(),
                    "input type must be INT8, HIFLOAT8 or FLOAT8_E4M3FN in per-tensor quant scenario"),
                return false);
    OP_CHECK_IF((deqScale1Shape == nullptr) || (quantScale1Shape == nullptr) || (deqScale2Shape == nullptr),
                OP_LOGE_WITH_INVALID_INPUT(contextKeyParams.opName, "deq_scale1, quant_scale1, deq_scale2"),
                return false);
    OP_CHECK_IF(
        (deqScale1Shape != nullptr && deqScale1Shape->GetStorageShape().GetShapeSize() == 0) ||
            (quantScale1Shape != nullptr && quantScale1Shape->GetStorageShape().GetShapeSize() == 0) ||
            (deqScale2Shape != nullptr && deqScale2Shape->GetStorageShape().GetShapeSize() == 0),
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(contextKeyParams.opName, "deq_scale1, quant_scale1, deq_scale2", "0",
                                                   "quant scale inputs must not be empty in per-tensor quant scenario"),
        return false);
    return true;
}

bool FiaArch38PromptTiling::CheckPerblockQuantParams(const FiaArch38ContextParams& contextKeyParams,
                                                     const PFAShapeInfo& queryShapeInfo,
                                                     const PFAShapeInfo& keyShapeInfo,
                                                     const PFAShapeInfo& valueShapeInfo) const
{
    const ge::DataType dequantScaleQueryType = contextKeyParams.dequantScaleQueryType;
    const ge::DataType keyAntiquantScaleType = contextKeyParams.keyAntiquantScaleType;
    const ge::DataType valueAntiquantScaleType = contextKeyParams.valueAntiquantScaleType;
    const gert::StorageShape* dequantScaleQueryShape = contextKeyParams.dequantScaleQueryShape;
    const gert::StorageShape* keyAntiquantScaleShape = contextKeyParams.keyAntiquantScaleShape;
    const gert::StorageShape* valueAntiquantScaleshape = contextKeyParams.valueAntiquantScaleShape;
    OP_CHECK_IF(
        (contextKeyParams.inputDataType != ge::DT_HIFLOAT8) && (contextKeyParams.inputDataType != ge::DT_FLOAT8_E4M3FN),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            contextKeyParams.opName, "query", GetPfaDataTypeStr(inputType).c_str(),
            "input type must be HIFLOAT8 or FLOAT8_E4M3FN in per-block quant scenario"),
        return false);
    OP_CHECK_IF((dequantScaleQueryType != ge::DT_FLOAT) || (keyAntiquantScaleType != ge::DT_FLOAT) ||
                    (valueAntiquantScaleType != ge::DT_FLOAT),
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    contextKeyParams.opName, "dequant_scale_query, key_antiquant_scale, value_antiquant_scale",
                    (GetPfaDataTypeStr(dequantScaleQueryType) + ", " + GetPfaDataTypeStr(keyAntiquantScaleType) + ", " +
                     GetPfaDataTypeStr(valueAntiquantScaleType))
                        .c_str(),
                    "dequant scale type must be DT_FLOAT in per-block quant scenario"),
                return false);
    OP_CHECK_IF((dequantScaleQueryShape == nullptr) || (keyAntiquantScaleShape == nullptr) ||
                    (valueAntiquantScaleshape == nullptr),
                OP_LOGE_WITH_INVALID_INPUT(contextKeyParams.opName,
                                           "dequant_scale_query, key_antiquant_scale, value_antiquant_scale"),
                return false);
    OP_CHECK_IF(
        (dequantScaleQueryShape->GetStorageShape().GetDim(0) != queryShapeInfo.b) ||
            (dequantScaleQueryShape->GetStorageShape().GetDim(1) != queryShapeInfo.n) ||
            (dequantScaleQueryShape->GetStorageShape().GetDim(2) !=
             CeilDivision(queryShapeInfo.s, static_cast<uint64_t>(128U))) || // 2 is the dim of dequantscale along s1.
            (dequantScaleQueryShape->GetStorageShape().GetDim(3) != 1),      // 3 is the dim of dequantscale along d.
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            contextKeyParams.opName, "dequant_scale_query",
            ("[" + std::to_string(dequantScaleQueryShape->GetStorageShape().GetDim(0)) + ", " +
             std::to_string(dequantScaleQueryShape->GetStorageShape().GetDim(1)) + ", " +
             std::to_string(dequantScaleQueryShape->GetStorageShape().GetDim(2)) + ", " +
             std::to_string(dequantScaleQueryShape->GetStorageShape().GetDim(3)) + "]")
                .c_str(),
            "dequantScaleQueryShape must be (b, n1, ceil(s1/128), 1) in per-block quant scenario"),
        return false);
    OP_CHECK_IF(
        (keyAntiquantScaleShape->GetStorageShape().GetDim(0) != keyShapeInfo.b) ||
            (keyAntiquantScaleShape->GetStorageShape().GetDim(1) != keyShapeInfo.n) ||
            (keyAntiquantScaleShape->GetStorageShape().GetDim(2) !=
             CeilDivision(keyShapeInfo.s, static_cast<uint64_t>(256U))) || //  2 is the dim of dequantscale along s2.
            (keyAntiquantScaleShape->GetStorageShape().GetDim(3) != 1),    // 3 is the dim of dequantscale along d.
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            contextKeyParams.opName, "key_antiquant_scale",
            ("[" + std::to_string(keyAntiquantScaleShape->GetStorageShape().GetDim(0)) + ", " +
             std::to_string(keyAntiquantScaleShape->GetStorageShape().GetDim(1)) + ", " +
             std::to_string(keyAntiquantScaleShape->GetStorageShape().GetDim(2)) + ", " +
             std::to_string(keyAntiquantScaleShape->GetStorageShape().GetDim(3)) + "]")
                .c_str(),
            "keyAntiquantScaleShape must be (b, n2, ceil(s2/256), 1) in per-block quant scenario"),
        return false);
    OP_CHECK_IF(
        (valueAntiquantScaleshape->GetStorageShape().GetDim(0) != valueShapeInfo.b) ||
            (valueAntiquantScaleshape->GetStorageShape().GetDim(1) != valueShapeInfo.n) ||
            (valueAntiquantScaleshape->GetStorageShape().GetDim(2) !=
             CeilDivision(valueShapeInfo.s, static_cast<uint64_t>(256U))) || // 2 is the dim of dequantscale along s2.
            (valueAntiquantScaleshape->GetStorageShape().GetDim(3) != 1),    // 3 is the dim of dequantscale along d.
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            contextKeyParams.opName, "value_antiquant_scale",
            ("[" + std::to_string(valueAntiquantScaleshape->GetStorageShape().GetDim(0)) + ", " +
             std::to_string(valueAntiquantScaleshape->GetStorageShape().GetDim(1)) + ", " +
             std::to_string(valueAntiquantScaleshape->GetStorageShape().GetDim(2)) + ", " +
             std::to_string(valueAntiquantScaleshape->GetStorageShape().GetDim(3)) + "]")
                .c_str(),
            "valueAntiquantScaleShape must be (b, n2, ceil(s2/256), 1) when enablePertensorQuant is true"),
        return false);
    return true;
}

bool FiaArch38PromptTiling::CheckPostQuantParams(const FiaArch38ContextParams& contextKeyParams,
                                                 const PFAShapeInfo& queryShapeInfo,
                                                 const PFAShapeInfo& valueShapeInfo) const
{
    const gert::StorageShape* quantScale2Shape = contextKeyParams.scale2Shape;
    const gert::StorageShape* quantOffset2Shape = contextKeyParams.offset2Shape;
    const ge::DataType quantScale2Type = contextKeyParams.quantScale2Type;
    uint64_t h = queryShapeInfo.h;
    uint64_t n = queryShapeInfo.n;

    int64_t quantScale2ShapeSize = 0;
    uint32_t quantD = 0;
    uint64_t queryD = h / n;

    OP_CHECK_IF(outputType != ge::DT_INT8 && outputType != ge::DT_FLOAT8_E4M3FN && outputType != ge::DT_HIFLOAT8,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(contextKeyParams.opName, "attention_out",
                                                      GetPfaDataTypeStr(outputType).c_str(),
                                                      "output type only supports int8, fp8_e4m3fn or hifloat8"),
                return false);

    // Basic verification: quantScale2 must be inputted and not an empty tensor
    OP_CHECK_IF(quantScale2Shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(contextKeyParams.opName, "quant_scale2"),
                return false);
    quantScale2ShapeSize = quantScale2Shape->GetStorageShape().GetShapeSize();
    quantD = quantScale2ShapeSize / n;
    OP_CHECK_IF(quantScale2ShapeSize == 0,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(contextKeyParams.opName, "quant_scale2", "0",
                                                          "quant_scale2 must not be empty in post quant scenario"),
                return false);

    // altert unsupported situation(post quant per-tensor + BF16 + BSH + D unalign)
    if ((contextKeyParams.inputDataType == ge::DT_BF16) && (quantScale2ShapeSize == 1) &&
        (inputLayout == InputLayout::BSH) && (queryD % BYTE_BLOCK != 0)) {
        OP_LOGW(contextKeyParams.opName,
                "post quant per-tensor doesn't support D unaligned(%u), when qkv is bf16 and layout is BSH.", queryD);
    }
    OP_CHECK_IF(!CheckPostQuantShape(contextKeyParams, quantOffset2Shape, quantScale2Type, quantScale2ShapeSize,
                                     queryShapeInfo, valueShapeInfo),
                OP_LOGE_WITHOUT_REPORT(contextKeyParams.opName, "post quant params check failed!"), return false);
    return true;
}

bool FiaArch38PromptTiling::CheckAntiquantParamsShape(FiaArch38ContextParams& contextKeyParams) const
{
    const gert::StorageShape* antiquantScaleShape = contextKeyParams.antiquantScaleShape;
    // 伪量化收编至ifa tiling，pfa接口kv若传入伪量化dtype均会在此被拦截
    OP_CHECK_IF(contextKeyParams.antiquantScale == nullptr || antiquantScaleShape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(contextKeyParams.opName, "antiquant_scale"), return false);
    return true;
}

bool FiaArch38PromptTiling::GetAndCheckPrefixShape(FiaArch38ContextParams& contextKeyParams,
                                                   const PFAShapeInfo& queryShapeInfo, PFAShapeInfo& prefixShapeInfo,
                                                   FiaArch38FlashAttentionTilingData& tilingData) const
{
    int64_t prefixSeqInnerSize = 0;
    int64_t bPrefix = 0U;
    int64_t nPrefix = 0U;
    int64_t dPrefix = 0U;
    int64_t hPrefix = 0U;
    int64_t tPrefix = 0;

    if (!SetShape(contextKeyParams, contextKeyParams.queryInputShape, "query", bPrefix, nPrefix, prefixSeqInnerSize,
                  dPrefix, hPrefix, tPrefix)) {
        return false;
    }
    OP_CHECK_IF(
        (bPrefix != 1),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(contextKeyParams.opName, "prefix batch", std::to_string(bPrefix).c_str(),
                                              "prefix batch num only supports 1"),
        return false);
    if (inputLayout == InputLayout::BSH) {
        OP_CHECK_IF((hPrefix != queryShapeInfo.h / tilingData.fiaArch38AttentionBaseParams.get_headNumRatio()),
                    OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                        contextKeyParams.opName, "key_shared_prefix, key",
                        (std::to_string(hPrefix) + ", " +
                         std::to_string(queryShapeInfo.h / tilingData.fiaArch38AttentionBaseParams.get_headNumRatio()))
                            .c_str(),
                        "prefix H must be the same as KV H"),
                    return false);
    } else {
        OP_CHECK_IF((nPrefix != queryShapeInfo.n / tilingData.fiaArch38AttentionBaseParams.get_headNumRatio()) ||
                        (dPrefix != queryShapeInfo.h / queryShapeInfo.n),
                    OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                        contextKeyParams.opName, "key_shared_prefix, key",
                        (std::to_string(nPrefix) + ", " + std::to_string(dPrefix) + ", " +
                         std::to_string(queryShapeInfo.n / tilingData.fiaArch38AttentionBaseParams.get_headNumRatio()) +
                         ", " + std::to_string(queryShapeInfo.h / queryShapeInfo.n))
                            .c_str(),
                        "prefix N and D must be the same as KV N and D"),
                    return false);
    }
    prefixShapeInfo.b = static_cast<uint32_t>(bPrefix);
    prefixShapeInfo.n = static_cast<uint64_t>(nPrefix);
    prefixShapeInfo.s = static_cast<uint32_t>(prefixSeqInnerSize);
    prefixShapeInfo.d = static_cast<uint32_t>(dPrefix);
    prefixShapeInfo.h = static_cast<uint64_t>(hPrefix);
    return true;
}

bool FiaArch38PromptTiling::CheckKeyValuePrefixConsistency(FiaArch38ContextParams& contextKeyParams,
                                                           const gert::StorageShape* keyShape) const
{
    size_t prefixKeyDim = contextKeyParams.keySharedPrefix->GetStorageShape().GetDimNum();
    size_t prefixValueDim = contextKeyParams.valueSharedPrefix->GetStorageShape().GetDimNum();
    size_t KVDim = keyShape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(
        ((prefixKeyDim != KVDim) || (prefixKeyDim != prefixValueDim)),
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            contextKeyParams.opName, "key_shared_prefix, value_shared_prefix, key",
            (std::to_string(prefixKeyDim) + ", " + std::to_string(prefixValueDim) + ", " + std::to_string(KVDim))
                .c_str(),
            "dim num of key_shared_prefix and value_shared_prefix must be the same as KV"),
        return false);
    for (uint32_t i = 0; i < prefixKeyDim; i++) {
        int64_t tmpPrefixKeyDim = contextKeyParams.keySharedPrefix->GetStorageShape().GetDim(i);
        int64_t tmpPrefixValueDim = contextKeyParams.valueSharedPrefix->GetStorageShape().GetDim(i);
        OP_CHECK_IF(((tmpPrefixKeyDim == 0) || (tmpPrefixValueDim == 0)),
                    OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                        contextKeyParams.opName, "key_shared_prefix, value_shared_prefix",
                        (std::to_string(tmpPrefixKeyDim) + ", " + std::to_string(tmpPrefixValueDim)).c_str(),
                        "key_shared_prefix and value_shared_prefix do not support empty tensor"),
                    return false);
        OP_CHECK_IF(((tmpPrefixKeyDim != tmpPrefixValueDim)),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        contextKeyParams.opName, "key_shared_prefix, value_shared_prefix",
                        (std::to_string(tmpPrefixKeyDim) + ", " + std::to_string(tmpPrefixValueDim)).c_str(),
                        "shape of key_shared_prefix must be the same as value_shared_prefix"),
                    return false);
    }
    return true;
}

bool FiaArch38PromptTiling::CheckActSharedPrefix(FiaArch38ContextParams& contextKeyParams, const uint32_t sPrefix,
                                                 const uint32_t sKV)
{
    size_t prefixDimNum = contextKeyParams.actualSharedPrefixLen->GetStorageShape().GetDimNum();
    OP_CHECK_IF((prefixDimNum != 1),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(contextKeyParams.opName, "actual_shared_prefix_len",
                                                         std::to_string(prefixDimNum).c_str(),
                                                         "actualSharedPrefixLen dim num must be 1"),
                return false);
    int64_t prefixShapeSize = contextKeyParams.actualSharedPrefixLen->GetStorageShape().GetShapeSize();
    OP_CHECK_IF((prefixShapeSize != 1),
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(contextKeyParams.opName, "actual_shared_prefix_len",
                                                          std::to_string(prefixShapeSize).c_str(),
                                                          "actualSharedPrefixLen length must be 1"),
                return false);
    OP_CHECK_IF((contextKeyParams.actualSharedPrefixLen->GetData<int64_t>() == nullptr),
                OP_LOGE_WITHOUT_REPORT(contextKeyParams.opName, "input actualSharedPrefixLen GetData is nullptr!"),
                return false);
    actualSharedPrefixLen = contextKeyParams.actualSharedPrefixLen->GetData<int64_t>()[0];
    OP_CHECK_IF((actualSharedPrefixLen > sPrefix) || (actualSharedPrefixLen < 0),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(contextKeyParams.opName, "actual_shared_prefix_len",
                                                      std::to_string(actualSharedPrefixLen).c_str(),
                                                      "actualSharedPrefixLen must be in range [0, sPrefix]"),
                return false);

    if (actualSharedPrefixLen + sKV > SLIMIT) {
        OP_LOGW(contextKeyParams.opName, "Seq size of actualSharedPrefixLen(%ld) + kv_s(%u) should <= 20m",
                actualSharedPrefixLen, sKV);
    }
    return true;
}

bool FiaArch38PromptTiling::CheckPAKeyValueShape(FiaArch38ContextParams& contextKeyParams, int64_t& keyDim1,
                                                 const PFAShapeInfo& queryShapeInfo, const gert::StorageShape* keyShape,
                                                 const gert::StorageShape* valueShape, const size_t keyDim,
                                                 const int32_t* blockSize, int64_t blockNumValid,
                                                 int32_t headNumRatio) const
{
    int64_t keyDim2 = keyShape->GetStorageShape().GetDim(KV_CACHE_DIM_1);
    int64_t keyDim3 = keyShape->GetStorageShape().GetDim(KV_CACHE_DIM_2);
    int64_t keyDim4 = 0;
    int64_t valueDim1 = valueShape->GetStorageShape().GetDim(KV_CACHE_DIM_0);
    int64_t valueDim2 = valueShape->GetStorageShape().GetDim(KV_CACHE_DIM_1);
    int64_t valueDim3 = valueShape->GetStorageShape().GetDim(KV_CACHE_DIM_2);
    int64_t valueDim4 = 0;
    if (keyDim == KV_CACHE_DIM_NUMS_4) {
        keyDim4 = keyShape->GetStorageShape().GetDim(KV_CACHE_DIM_3);
        valueDim4 = valueShape->GetStorageShape().GetDim(KV_CACHE_DIM_3);
    }
    OP_CHECK_IF(
        (keyDim == KV_CACHE_DIM_NUMS_4) && ((keyDim1 != valueDim1) || (keyDim2 != valueDim2) ||
                                            (keyDim3 != valueDim3) || ((keyDim4 != valueDim4) && (!enablePFAMLA))),
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            contextKeyParams.opName, "key, value",
            (std::to_string(keyDim1) + ", " + std::to_string(keyDim2) + ", " + std::to_string(keyDim3) + ", " +
             std::to_string(keyDim4) + ", " + std::to_string(valueDim1) + ", " + std::to_string(valueDim2) + ", " +
             std::to_string(valueDim3) + ", " + std::to_string(valueDim4))
                .c_str(),
            "the dim num of key and value are inconsistent when PA enable"),
        return false);
    OP_CHECK_IF((keyDim == KV_CACHE_DIM_NUMS_3) &&
                    ((keyDim1 != valueDim1) || (keyDim2 != valueDim2) || ((keyDim3 != valueDim3) && (!enablePFAMLA))),
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                    contextKeyParams.opName, "key, value",
                    (std::to_string(keyDim1) + ", " + std::to_string(keyDim2) + ", " + std::to_string(keyDim3) + ", " +
                     std::to_string(valueDim1) + ", " + std::to_string(valueDim2) + ", " + std::to_string(valueDim3))
                        .c_str(),
                    "the dim num of key and value are inconsistent when PA enable"),
                return false);
    std::string layoutStr(contextKeyParams.layout);
    if (inputLayout == InputLayout::BNSD || inputLayout == InputLayout::TND) {
        OP_CHECK_IF(
            ((keyDim != KV_CACHE_DIM_NUMS_3) && (keyDim != KV_CACHE_DIM_NUMS_4)),
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(contextKeyParams.opName, "key, value",
                                                   (std::to_string(keyDim1) + ", " + std::to_string(keyDim2) + ", " +
                                                    std::to_string(keyDim3) + ", " + std::to_string(keyDim4))
                                                       .c_str(),
                                                   "the layout of key and value is invalid when PA enable"),
            return false);
    } else if (inputLayout == InputLayout::BSH || inputLayout == InputLayout::BSND) {
        OP_CHECK_IF(
            keyDim != 3,
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(contextKeyParams.opName, "key, value",
                                                   (std::to_string(keyDim1) + ", " + std::to_string(keyDim2) + ", " +
                                                    std::to_string(keyDim3) + ", " + std::to_string(keyDim4))
                                                       .c_str(),
                                                   "the layout of key and value is invalid when PA enable"),
            return false);
    }
    return true;
}

bool FiaArch38PromptTiling::CheckPACacheShape(FiaArch38ContextParams& contextKeyParams, const size_t keyDim,
                                              const PFAShapeInfo& shapeInfo, const gert::StorageShape* shape,
                                              const int32_t* blockSize, int64_t blockNumValid, int32_t headNumRatio,
                                              const std::string& sName)
{
    std::string layoutStr(contextKeyParams.layout);
    int64_t dim1 = shape->GetStorageShape().GetDim(KV_CACHE_DIM_0);
    int64_t dim2 = shape->GetStorageShape().GetDim(KV_CACHE_DIM_1);
    int64_t dim3 = shape->GetStorageShape().GetDim(KV_CACHE_DIM_2);
    int64_t dim4 = 0;
    int64_t tempBlockSize = dim2;
    int64_t tempH = dim3;
    int64_t tempN = 0;
    int64_t tempD = 0;
    if (keyDim == 3) {    // dim num: 3
        paLayoutType = 1; // If it is three-dimensional, paLayoutType = 1
        OP_CHECK_IF(
            ((dim1 < blockNumValid) || (tempBlockSize != *blockSize) || (tempH * headNumRatio != shapeInfo.h)),
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                contextKeyParams.opName, sName.c_str(),
                ("[" + std::to_string(dim1) + ", " + std::to_string(dim2) + ", " + std::to_string(dim3) + "]").c_str(),
                "the dim must be [>=blockNumValid, blockSize, h/headNumRatio] when PA enable"),
            return false);
        // In the BSH input of the PA scenario, it is required that the h of the KV matrix does not exceed 65535.
        // The dim and dim3 of the K/V have already been verified to be equal, so only the K matrix is verified here.
        OP_CHECK_IF(dim3 > HLIMIT,
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                        contextKeyParams.opName, sName.c_str(), std::to_string(dim3).c_str(),
                        "layout of key/value is BSH, h must not be greater than 65535 when PA enable"),
                    return false);
    } else {
        dim4 = shape->GetStorageShape().GetDim(3); // 3: The third dimension.
        tempN = dim2;
        tempBlockSize = dim3;
        tempD = dim4;
        paLayoutType = 0; // If it is four-dimensional, paLayoutType = 0
        OP_CHECK_IF(((dim1 < blockNumValid) || (tempN * headNumRatio != shapeInfo.n) || (tempBlockSize != *blockSize) ||
                     (tempD != (shapeInfo.h / shapeInfo.n))),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        contextKeyParams.opName, sName.c_str(),
                        ("[" + std::to_string(dim1) + ", " + std::to_string(dim2) + ", " + std::to_string(dim3) + ", " +
                         std::to_string(dim4) + "]")
                            .c_str(),
                        "the dim must be [>=blockNumValid, n/headNumRatio, blockSize, h/n] when PA enable"),
                    return false);
    }
    return true;
}

bool FiaArch38PromptTiling::CheckBlockTableShape(FiaArch38ContextParams& contextKeyParams, PFAShapeInfo& queryShapeInfo,
                                                 PFAShapeInfo& queryRopeShapeInfo, const int32_t* blockSize,
                                                 const gert::StorageShape* blockTableShape,
                                                 FiaArch38FlashAttentionTilingData& tilingData)
{
    const gert::Tensor* actSeqLenKV = contextKeyParams.actualSequenceLengthKV;
    int32_t actualSeqKVPerBatch = 0;
    int32_t blockNumPerBatch = 0;
    int64_t blockNumValid = 0;
    int32_t maxBlockNumPerBatch = 0;
    for (uint32_t i = 0; i < queryShapeInfo.b; i++) {
        actualSeqKVPerBatch = (actSeqLenKV->GetShapeSize() > 1) ?
                                  static_cast<int32_t>(actSeqLenKV->GetData<int64_t>()[i]) :
                                  static_cast<int32_t>(actSeqLenKV->GetData<int64_t>()[0]);
        blockNumPerBatch = (actualSeqKVPerBatch + *blockSize - 1) / *blockSize;
        blockNumValid += blockNumPerBatch;
        if (blockNumPerBatch > maxBlockNumPerBatch) {
            maxBlockNumPerBatch = blockNumPerBatch;
        }
    }
    const gert::StorageShape* keyShape = contextKeyParams.keyInputShape;
    const gert::StorageShape* valueShape = contextKeyParams.valueInputShape;
    const size_t keyDim = keyShape->GetStorageShape().GetDimNum();
    int64_t keyDim1 = keyShape->GetStorageShape().GetDim(KV_CACHE_DIM_0);
    int64_t headNumRatio = (enableIFAMLA || enableIFA) ?
                               gSize :
                               static_cast<int64_t>(tilingData.fiaArch38AttentionBaseParams.get_headNumRatio());
    if (!CheckPAKeyValueShape(contextKeyParams, keyDim1, queryShapeInfo, keyShape, valueShape, keyDim, blockSize,
                              blockNumValid, headNumRatio)) {
        return false;
    }
    if (!CheckPACacheShape(contextKeyParams, keyDim, queryShapeInfo, keyShape, blockSize, blockNumValid, headNumRatio,
                           "key")) {
        return false;
    }
    if (enableIFAMLA) {
        const gert::StorageShape* keyRopeShape = contextKeyParams.keyRopeInputShape;
        const size_t keyRopeDim = keyRopeShape->GetStorageShape().GetDimNum();
        OP_CHECK_IF((keyRopeDim != keyDim),
                    OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                        contextKeyParams.opName, "key_rope, key",
                        (std::to_string(keyRopeDim) + ", " + std::to_string(keyDim)).c_str(),
                        "the dim of keyRope must be equal to the dim of key when PA enable"),
                    return false);
        if (!CheckPACacheShape(contextKeyParams, keyDim, queryRopeShapeInfo, keyRopeShape, blockSize, blockNumValid,
                               headNumRatio, "keyRope")) {
            return false;
        }
    }

    // check blockTable Shape
    size_t blockTableDim = blockTableShape->GetStorageShape().GetDimNum();
    int64_t blockTableDim1 = blockTableShape->GetStorageShape().GetDim(0);
    string blockTableStr = "";
    for (size_t i = 0; i < blockTableDim; i++) {
        blockTableStr += std::to_string(blockTableShape->GetStorageShape().GetDim(i));
        blockTableStr += ",";
    }
    blockTableStr.pop_back();

    OP_CHECK_IF(blockTableDim != 2, // PA BlockTable dim size
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(contextKeyParams.opName, "block_table",
                                                         std::to_string(blockTableDim).c_str(),
                                                         "the dim of block table must be 2 when PA enable"),
                return false);

    OP_CHECK_IF(((blockTableDim1 != queryShapeInfo.b) || (blockTableDim2 < maxBlockNumPerBatch)),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    contextKeyParams.opName, "block_table",
                    ("[" + std::to_string(blockTableDim1) + ", " + std::to_string(blockTableDim2) + "]").c_str(),
                    "block table shape should be [b, >=maxBlockNumPerBatch] when PA enable"),
                return false);

    paBlockNumSum = keyDim1;
    S2 = maxBlockNumPerBatch * (*blockSize);
    if (S2 > SLIMIT) {
        OP_LOGW(contextKeyParams.opName, "seq should <= 20m, but kv seq = %u when PA enable", S2);
    }
    return true;
}

bool FiaArch38PromptTiling::CheckMaskShape(FiaArch38ContextParams& contextKeyParams, const int32_t* sparseMode,
                                           int64_t& attenMaskBatch, int64_t& attenMaskS1, int64_t& attenMaskS2,
                                           bool& checkMask, const uint32_t sQ, const uint32_t sK,
                                           const uint32_t batchSize, std::string& strMaskShape) const
{
    const gert::StorageShape* attenMaskShape = contextKeyParams.attentionMaskShape;
    size_t attenMaskDim = attenMaskShape->GetStorageShape().GetDimNum();
    int64_t attenMaskN = 1U;
    if (attenMaskDim == MASKDIM_2) {
        if (enableIFA) {
            attenMaskBatch = attenMaskShape->GetStorageShape().GetDim(0);
            attenMaskS1 = 1;
            attenMaskS2 = attenMaskShape->GetStorageShape().GetDim(1);
            strMaskShape = std::to_string(attenMaskBatch) + ", " + std::to_string(attenMaskS2);
        } else {
            attenMaskS1 = attenMaskShape->GetStorageShape().GetDim(0);
            attenMaskS2 = attenMaskShape->GetStorageShape().GetDim(1);
            strMaskShape = std::to_string(attenMaskS1) + ", " + std::to_string(attenMaskS2);
        }
    } else if (attenMaskDim == MASKDIM_3) {
        attenMaskBatch = attenMaskShape->GetStorageShape().GetDim(0);
        attenMaskS1 = attenMaskShape->GetStorageShape().GetDim(1);
        attenMaskS2 = attenMaskShape->GetStorageShape().GetDim(2); // 2: When the dim is 3, the second dimension is S2.
        strMaskShape =
            std::to_string(attenMaskBatch) + ", " + std::to_string(attenMaskS1) + ", " + std::to_string(attenMaskS2);
    } else if (attenMaskDim == MASKDIM_4) {
        attenMaskBatch = attenMaskShape->GetStorageShape().GetDim(0);
        attenMaskN = attenMaskShape->GetStorageShape().GetDim(1);
        attenMaskS1 = attenMaskShape->GetStorageShape().GetDim(2); // 2: When the dim is 4, the second dimension is S1.
        attenMaskS2 = attenMaskShape->GetStorageShape().GetDim(3); // 3: When the dim is 4, the third dimension is S2.
        strMaskShape = std::to_string(attenMaskBatch) + ", " + std::to_string(attenMaskN) + ", " +
                       std::to_string(attenMaskS1) + ", " + std::to_string(attenMaskS2);
    } else {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(contextKeyParams.opName, "atten_mask",
                                                 std::to_string(attenMaskDim).c_str(),
                                                 "attenMask dim must be 2, 3 or 4");
        return false;
    }

    if (enableIFA) {
        checkMask = (attenMaskBatch == batchSize) && (attenMaskS1 == 1) && (attenMaskS2 >= S2);
    } else if (isDefaultSparseMode || (sparseMode != nullptr && *sparseMode == SPARSE_MODE_ALL_MASK)) {
        checkMask = (attenMaskS1 >= sQ) && (attenMaskS2 >= sK) && (attenMaskBatch == 1 || attenMaskBatch == batchSize);
    } else if ((sparseMode != nullptr) &&
               ((*sparseMode == SPARSE_MODE_LEFT_UP) || (*sparseMode == SPARSE_MODE_RIGHT_DOWN) ||
                (*sparseMode == SPARSE_MODE_BAND))) {
        checkMask = (attenMaskBatch == 1) && (attenMaskN == 1) && (attenMaskS1 == SPARSE_OPTIMIZE_ATTENTION_SIZE) &&
                    (attenMaskS2 == SPARSE_OPTIMIZE_ATTENTION_SIZE);
    }
    return true;
}

void FiaArch38PromptTiling::SetSparseModeData(FiaArch38ContextParams& contextKeyParams,
                                              const gert::StorageShape* attenMaskShape,
                                              FiaArch38FlashAttentionTilingData& tilingData, const int32_t* sparseMode,
                                              const int64_t* preTokens, const int64_t* nextTokens)
{
    size_t attenMaskBatch = 1;
    if ((attenMaskShape != nullptr) && (attenMaskShape->GetStorageShape().GetDimNum() != MASKDIM_2)) {
        // 2: target dimension of attenMask
        attenMaskBatch = attenMaskShape->GetStorageShape().GetDim(0);
    }
    if ((attenMaskShape != nullptr) && enableIFA && (attenMaskShape->GetStorageShape().GetDimNum() == MASKDIM_2)) {
        attenMaskBatch = attenMaskShape->GetStorageShape().GetDim(0); // IFA mask shape (B,S2)
    }
    tilingData.fiaArch38AttentionSingleCoreParams.set_attenMaskBatch(attenMaskBatch);

    if (*preTokens > SPARSE_MODE_INT_MAX) {
        sparsePreTokens = SPARSE_MODE_INT_MAX;
    } else if (*preTokens < -(SPARSE_MODE_INT_MAX)) {
        sparsePreTokens = -(SPARSE_MODE_INT_MAX);
    } else {
        sparsePreTokens = *preTokens;
    }

    if (*nextTokens > SPARSE_MODE_INT_MAX) {
        sparseNextTokens = SPARSE_MODE_INT_MAX;
    } else if (*nextTokens < -(SPARSE_MODE_INT_MAX)) {
        sparseNextTokens = -(SPARSE_MODE_INT_MAX);
    } else {
        sparseNextTokens = *nextTokens;
    }
    if ((attenMaskShape != nullptr) && (sparseMode != nullptr)) {
        if (*sparseMode == SPARSE_MODE_LEFT_UP) {
            sparsePreTokens = SPARSE_MODE_INT_MAX;
            sparseNextTokens = 0;
            // Right down tokens are calculated on the kernel side.
        } else if (*sparseMode == SPARSE_MODE_RIGHT_DOWN) {
            sparsePreTokens = SPARSE_MODE_INT_MAX;
        } else if (*sparseMode == SPARSE_MODE_ALL_MASK) {
            sparsePreTokens = SPARSE_MODE_INT_MAX;
            sparseNextTokens = SPARSE_MODE_INT_MAX;
        } else if (*sparseMode == SPARSE_MODE_BAND) {
            isBandMode = true;
        }
        if (enableIFA && !enableIFAMLA) {
            sparseModeVal = SPARSE_MODE_NO_MASK;
            sparsePreTokens = SPARSE_MODE_INT_MAX;
            sparseNextTokens = SPARSE_MODE_INT_MAX;
        }
        sparseModeVal = *sparseMode;
        OP_LOGI(contextKeyParams.opName, "sparseMode is %d.", sparseModeVal);
    }
    if ((sparseMode != nullptr) && (*sparseMode == SPARSE_MODE_NO_MASK)) {
        // sparse mode need process attention mask when empty tensor scenes as same.
        if (((attenMaskShape != nullptr) && (attenMaskShape->GetStorageShape().GetShapeSize() == 0)) ||
            (attenMaskShape == nullptr)) {
            sparsePreTokens = SPARSE_MODE_INT_MAX;
            sparseNextTokens = SPARSE_MODE_INT_MAX;
            sparseModeVal = *sparseMode;
        }
    }
    if (isDefaultSparseMode && enableLeftPadding) {
        // For scenes with sparse mode=0 and left padding, the attention mask part is fully calculated
        sparsePreTokens = SPARSE_MODE_INT_MAX;
        sparseNextTokens = SPARSE_MODE_INT_MAX;
    }
}

bool FiaArch38PromptTiling::CheckMaskShapeCrossSparse(FiaArch38ContextParams& contextKeyParams,
                                                      const int32_t* sparseMode, uint64_t sQ, const uint64_t sK,
                                                      const uint32_t batchSize)
{
    if ((contextKeyParams.fromTilingSink != 0) || (!enableMask)) {
        return true;
    }
    if (enableIFA || enableIFAMLA || enablePFAMerge) {
        sQ /= gSize; // 合轴场景使用原始的seq长度校验
    }
    int64_t attenMaskBatch = 1;
    int64_t attenMaskS1 = 0;
    int64_t attenMaskS2 = 0;
    bool checkMask = 0;
    std::string strMaskShape;
    if (!CheckMaskShape(contextKeyParams, sparseMode, attenMaskBatch, attenMaskS1, attenMaskS2, checkMask, sQ, sK,
                        batchSize, strMaskShape)) {
        return false;
    }
    if (enableIFA) {
        OP_CHECK_IF(!checkMask,
                    OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                        contextKeyParams.opName, "atten_mask",
                        (std::to_string(attenMaskBatch) + ", " + std::to_string(attenMaskS1) + ", " +
                         std::to_string(attenMaskS2))
                            .c_str(),
                        "attenMask batch, Q_S and KV_S are invalid for the given sparse mode"),
                    return false);
    } else {
        if (isDefaultSparseMode || ((sparseMode != nullptr) && (*sparseMode == SPARSE_MODE_ALL_MASK))) {
            OP_CHECK_IF(!checkMask,
                        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                            contextKeyParams.opName, "atten_mask",
                            (std::to_string(attenMaskBatch) + ", " + std::to_string(attenMaskS1) + ", " +
                             std::to_string(attenMaskS2))
                                .c_str(),
                            "attenMask batch, Q_S and KV_S are invalid for the given sparse mode"),
                        return false);
        }
        if ((sparseMode != nullptr) &&
            ((*sparseMode == SPARSE_MODE_LEFT_UP) || (*sparseMode == SPARSE_MODE_RIGHT_DOWN) ||
             (*sparseMode == SPARSE_MODE_BAND)) &&
            !checkMask) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                contextKeyParams.opName, "atten_mask", strMaskShape.c_str(),
                "attenMask shape must be (2048, 2048), (1, 2048, 2048) or (1, 1, 2048, 2048) for this sparse mode");
            return false;
        }
    }
    attenMaskShapeType = attenMaskBatch > 1 ? 1 : 2; // 1 for multi-batch and 2 for 1 batch, same as fa
    return true;
}

bool FiaArch38PromptTiling::CheckPFAMerge(FiaArch38ContextParams& contextKeyParams,
                                          const PFAShapeInfo& queryShapeInfo) const
{
    if ((queryShapeInfo.s <= 1U) || (queryShapeInfo.s > pfaMergeQsLimit)) {
        return false;
    }

    const int64_t nQ = *contextKeyParams.headsNumber;
    const int64_t nKV = *contextKeyParams.numKeyValueHeads;
    if ((nKV > 0) && (static_cast<uint32_t>(nQ / nKV) > pfaMergeGLimit)) {
        return false;
    }

    if (enableMask || enablePseShift || enablePA || enableAlibiPse) {
        return false;
    }
    return true;
}

bool FiaArch38PromptTiling::CheckIO(FiaArch38ContextParams& contextKeyParams, PFAShapeInfo& queryShapeInfo,
                                    PFAShapeInfo& valueShapeInfo)
{
    const gert::StorageShape* queryShape = contextKeyParams.queryInputShape;
    const gert::StorageShape* outShape = contextKeyParams.outputShape;
    const gert::StorageShape* valueShape = contextKeyParams.valueInputShape;
    // check dtype
    if (!CheckIODataType(contextKeyParams)) {
        return false;
    }

    // check layout
    OP_CHECK_IF((!SetInputLayout(contextKeyParams.layout)),
                OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(contextKeyParams.opName, "input_layout", contextKeyParams.layout,
                                                       "input layout must be one of TND, BSH, BNSD, BSND or BNSD_BSND"),
                return false);

    // check query shape size and bnsd
    OP_CHECK_IF((!GetAndCheckShape(contextKeyParams, queryShapeInfo, queryShape, "query")),
                OP_LOGE_WITHOUT_REPORT(contextKeyParams.opName, "Get and check query shape failed."), return false);
    if (queryShapeInfo.s == 1 && !enableAlibiPse) {
        enableIFA = true;
    }
    enablePFAMerge = CheckPFAMerge(contextKeyParams, queryShapeInfo);
    // check value shape
    if (enablePA) { // only get shape
        int64_t b = 0;
        int64_t n = 0;
        int64_t s = 0;
        int64_t d = 0;
        int64_t h = 0;
        int64_t t = 0;
        if (!SetShape(contextKeyParams, valueShape, "value", b, n, s, d, h, t)) {
            return false;
        }
        valueShapeInfo.d = d;
    } else {
        OP_CHECK_IF((!GetAndCheckShape(contextKeyParams, valueShapeInfo, valueShape, "value")),
                    OP_LOGE_WITHOUT_REPORT(contextKeyParams.opName, "Get and check value shape failed."), return false);
    }
    if (queryShapeInfo.d == MLA_QKD_SIZE && valueShapeInfo.d == MLA_VD_SIZE) {
        enablePFAMLA = true;
        enablePFAMerge = false;
    }
    // check query and output consistency
    OP_CHECK_IF((!CheckQueryOutParamsConsistency(contextKeyParams, queryShape, outShape)),
                OP_LOGE_WITHOUT_REPORT(contextKeyParams.opName, "Query and output consistency check failed."),
                return false);

    return true;
}

bool FiaArch38PromptTiling::CheckKV(FiaArch38ContextParams& contextKeyParams, PFAShapeInfo& keyShapeInfo,
                                    PFAShapeInfo& valueShapeInfo) const
{
    const gert::StorageShape* keyShape = contextKeyParams.keyInputShape;
    const gert::StorageShape* valueShape = contextKeyParams.valueInputShape;

    // check dtype
    if (!CheckKVDataType(contextKeyParams)) {
        return false;
    }

    // check key shape size and bnsd
    OP_CHECK_IF((!enablePA && !GetAndCheckShape(contextKeyParams, keyShapeInfo, keyShape, "key")),
                OP_LOGE_WITHOUT_REPORT(contextKeyParams.opName, "Get and check key shape failed."), return false);

    // check value shape size and bnsd
    OP_CHECK_IF((!enablePA && !GetAndCheckShape(contextKeyParams, valueShapeInfo, valueShape, "value")),
                OP_LOGE_WITHOUT_REPORT(contextKeyParams.opName, "Get and check value shape failed."), return false);

    // check key and value consistency
    OP_CHECK_IF((!CheckKeyValueParamsConsistency(contextKeyParams, keyShape, valueShape)),
                OP_LOGE_WITHOUT_REPORT(contextKeyParams.opName, "Key and value consistency check failed."),
                return false);

    return true;
}

bool FiaArch38PromptTiling::CheckQueryAndKey(FiaArch38ContextParams& contextKeyParams, PFAShapeInfo& queryShapeInfo,
                                             PFAShapeInfo& keyShapeInfo, FiaArch38FlashAttentionTilingData& tilingData)
{
    // check numhead ratio
    if (!SetAndCheckHeadNumRatio(contextKeyParams, tilingData)) {
        return false;
    }

    // check b and s for not PA
    OP_CHECK_IF((contextKeyParams.fromTilingSink == 0) && (queryShapeInfo.b != keyShapeInfo.b) && (!enableTensorList) &&
                    (!enablePA),
                OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                    contextKeyParams.opName, "query, key, value",
                    (std::to_string(queryShapeInfo.b) + ", " + std::to_string(keyShapeInfo.b)).c_str(),
                    "query batch must be equal to key and value batch"),
                return false);
    // check d size
    OP_CHECK_IF((contextKeyParams.fromTilingSink == 0) && (queryShapeInfo.d != keyShapeInfo.d) && (!enableTensorList) &&
                    (!enablePA),
                OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                    contextKeyParams.opName, "query, key",
                    (std::to_string(queryShapeInfo.d) + ", " + std::to_string(keyShapeInfo.d)).c_str(),
                    "query d size must be equal to key d size"),
                return false);
    return true;
}

bool FiaArch38PromptTiling::CheckIFAMLA(FiaArch38ContextParams& contextKeyParams,
                                        const PFAShapeInfo& queryShapeInfo) const
{
    constexpr uint32_t maxQuerySeqLenInIfaMla = 16U; // ifa mla场景qS最大支持16
    OP_CHECK_IF((queryShapeInfo.s > maxQuerySeqLenInIfaMla || queryShapeInfo.s < 1),
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    contextKeyParams.opName, "query", std::to_string(queryShapeInfo.s).c_str(),
                    "when enable ifa mla, query sequence length must be in range [1, maxQuerySeqLenInIfaMla]"),
                return false);
    static const std::set<uint32_t> supportNumHeadInIfaMla = {32U, 64U, 128U}; // ifa mla场景qN支持范围
    OP_CHECK_IF((supportNumHeadInIfaMla.find(queryShapeInfo.n) == supportNumHeadInIfaMla.end()),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(contextKeyParams.opName, "num_heads",
                                                      std::to_string(queryShapeInfo.n).c_str(),
                                                      "when enable ifa mla, query heads num must be 32, 64 or 128"),
                return false);
    const int64_t nKV =
        *contextKeyParams.numKeyValueHeads; // ifa mla场景不支持g = 1, 因此在nKV用默认值0, nQ替代也属于异常场景
    OP_CHECK_IF((nKV != 1U),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(contextKeyParams.opName, "num_key_value_heads",
                                                      std::to_string(nKV).c_str(),
                                                      "when enable ifa mla, key and value heads num must be 1"),
                return false);
    return true;
}

bool FiaArch38PromptTiling::CheckRope(FiaArch38ContextParams& contextKeyParams, PFAShapeInfo& queryShapeInfo,
                                      PFAShapeInfo& keyShapeInfo, PFAShapeInfo& queryRopeShapeInfo)
{
    if (contextKeyParams.queryRopeInputShape == nullptr && contextKeyParams.keyRopeInputShape == nullptr) {
        return true;
    }
    OP_CHECK_IF((contextKeyParams.queryRopeInputShape == nullptr && contextKeyParams.keyRopeInputShape != nullptr),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "query_rope, key_rope",
                                                       "queryRope absent, keyRope present",
                                                       "queryRope and keyRope must be both present or both absent"),
                return false);
    OP_CHECK_IF((contextKeyParams.queryRopeInputShape != nullptr && contextKeyParams.keyRopeInputShape == nullptr),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "query_rope, key_rope",
                                                       "queryRope present, keyRope absent",
                                                       "queryRope and keyRope must be both present or both absent"),
                return false);
    const gert::StorageShape* queryRopeShape = contextKeyParams.queryRopeInputShape;
    const gert::StorageShape* keyRopeShape = contextKeyParams.keyRopeInputShape;

    // check queryRope shape
    OP_CHECK_IF((!GetAndCheckRopeShape(contextKeyParams, queryShapeInfo, queryRopeShapeInfo, queryRopeShape, "query",
                                       "queryRope")),
                OP_LOGE_WITHOUT_REPORT(contextKeyParams.opName, "Get and check queryRope shape failed."), return false);
    // check keyRope shape
    PFAShapeInfo keyRopeShapeInfo;
    OP_CHECK_IF((!enablePA && !GetAndCheckRopeShape(contextKeyParams, keyShapeInfo, keyRopeShapeInfo, keyRopeShape,
                                                    "key", "keyRope")),
                OP_LOGE_WITHOUT_REPORT(contextKeyParams.opName, "Get and check queryRope shape failed."), return false);
    enableIFA = false;
    enablePFAMerge = false;
    if (queryShapeInfo.d == QUERY_SHAPE_DIM_D_128_TILING_V2) {
        enablePFARope = true;
    } else {
        enableIFAMLA = true;
    }
    OP_LOGI(contextKeyParams.opName, "enableIFAMLA is %d, enablePA is %d, enableMask is %d", enableIFAMLA, enablePA,
            enableMask);
    return true;
}

bool FiaArch38PromptTiling::CheckQuant(FiaArch38ContextParams& contextKeyParams, PFAShapeInfo& queryShapeInfo,
                                       PFAShapeInfo& keyShapeInfo, const PFAShapeInfo& valueShapeInfo) const
{
    const gert::StorageShape* deqScale1Shape = contextKeyParams.deqScale1Shape;
    const gert::StorageShape* quantScale1Shape = contextKeyParams.scale1Shape;
    const gert::StorageShape* deqScale2Shape = contextKeyParams.deqScale2Shape;
    // per-tensor quant check
    if (enablePertensorQuant) {
        OP_CHECK_IF(!CheckPerTensorQuantParams(contextKeyParams),
                    OP_LOGE_WITHOUT_REPORT(contextKeyParams.opName, "per-tensor quant params check failed!"),
                    return false);
    }

    // per-block quant check
    if (enablePerblockQuant) {
        OP_CHECK_IF(!CheckPerblockQuantParams(contextKeyParams, queryShapeInfo, keyShapeInfo, valueShapeInfo),
                    OP_LOGE_WITHOUT_REPORT(contextKeyParams.opName, "per-block quant params check failed!"),
                    return false);
    }

    OP_CHECK_IF(
        ((contextKeyParams.inputDataType == ge::DT_INT8) && (contextKeyParams.outputDataType == ge::DT_FLOAT16) &&
         ((contextKeyParams.scale2Shape != nullptr) || (contextKeyParams.offset2Shape != nullptr))),
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            contextKeyParams.opName, "quant_scale2, quant_offset2", "present",
            "when query dtype is int8 and output dtype is fp16, quant_scale2 and quant_offset2 must be null"),
        return false);

    // quant/dequant dtype check
    if ((inputType == ge::DT_FLOAT16) && enablePostQuant) {
        OP_CHECK_IF((deqScale1Shape != nullptr) || (quantScale1Shape != nullptr) || (deqScale2Shape != nullptr),
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        contextKeyParams.opName, "dequant_scale1, quant_scale1, dequant_scale2", "present",
                        "when input dtype is fp16 and postQuant is enabled, dequantScale1, quantScale1 and "
                        "dequantScale2 must be null"),
                    return false);
    }

    // post-quant check
    if (enablePostQuant) {
        OP_CHECK_IF(!CheckPostQuantParams(contextKeyParams, queryShapeInfo, valueShapeInfo),
                    OP_LOGE_WITHOUT_REPORT(contextKeyParams.opName, "post quant params check failed!"), return false);
    }
    OP_CHECK_IF(enableKVAntiquant && !CheckAntiquantParamsShape(contextKeyParams),
                OP_LOGE_WITHOUT_REPORT(contextKeyParams.opName, "antiquant params check failed!"), return false);
    return true;
}

bool FiaArch38PromptTiling::CheckPrefix(FiaArch38ContextParams& contextKeyParams, PFAShapeInfo& queryShapeInfo,
                                        PFAShapeInfo& keyShapeInfo, FiaArch38FlashAttentionTilingData& tilingData)
{
    PFAShapeInfo prefixShapeInfo;
    tilingData.fiaArch38AttentionBaseParams.set_prefixSeqInnerSize(0);
    if (!isKVHasPrefix) {
        tilingData.fiaArch38AttentionBaseParams.set_isActualSharedPrefixLenNull(1);
        actualSharedPrefixLen = 0;
        return true;
    }
    // The prefix does not support tensorlist, PA, or left padding
    OP_CHECK_IF(enableTensorList,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "tensorlist, prefix",
                                                       "tensorlist with system prefix",
                                                       "when tensorlist is used, system prefix is not supported"),
                return false);
    OP_CHECK_IF(enableLeftPadding,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "prefix, leftPadding",
                                                       "system prefix with left padding",
                                                       "when system prefix is used, left padding is not supported"),
                return false);
    OP_CHECK_IF((contextKeyParams.inputDataType == ge::DT_INT8) && (contextKeyParams.kDataType == ge::DT_INT8),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    contextKeyParams.opName, "query, key/value", "both int8",
                    "when system prefix is used, query and key/value should not both be int8"),
                return false);

    // get prefix shape
    const gert::StorageShape* keyShape = contextKeyParams.keyInputShape;

    OP_CHECK_IF(!GetAndCheckPrefixShape(contextKeyParams, queryShapeInfo, prefixShapeInfo, tilingData),
                OP_LOGE_WITHOUT_REPORT(contextKeyParams.opName, "Get and check prefix shape failed."), return false);

    // check prefix kv consistency
    OP_CHECK_IF(!CheckKeyValuePrefixConsistency(contextKeyParams, keyShape),
                OP_LOGE_WITHOUT_REPORT(contextKeyParams.opName, "key value prefix consistency check failed."),
                return false);

    // check actSharedPrefix
    if ((contextKeyParams.fromTilingSink == 0U) && (contextKeyParams.actualSharedPrefixLen != nullptr) &&
        (contextKeyParams.actualSharedPrefixLen->GetStorageShape().GetShapeSize() > 0) &&
        !CheckActSharedPrefix(contextKeyParams, prefixShapeInfo.s, keyShapeInfo.s)) {
        tilingData.fiaArch38AttentionBaseParams.set_isActualSharedPrefixLenNull(0);
    } else {
        tilingData.fiaArch38AttentionBaseParams.set_isActualSharedPrefixLenNull(1);
        actualSharedPrefixLen = prefixShapeInfo.s;
    }
    tilingData.fiaArch38AttentionBaseParams.set_prefixSeqInnerSize(prefixShapeInfo.s);
    return true;
}

bool FiaArch38PromptTiling::CheckActSeq(const FiaArch38ContextParams& contextKeyParams,
                                        const PFAShapeInfo& queryShapeInfo) const
{
    if (inputLayout != InputLayout::TND) {
        return true;
    }

    const gert::Tensor* actSeqLen = contextKeyParams.actualSequenceLengthQ;
    const gert::Tensor* actSeqLenKV = contextKeyParams.actualSequenceLengthKV;
    OP_CHECK_IF(actSeqLen == nullptr, OP_LOGE_WITH_INVALID_INPUT(contextKeyParams.opName, "actual_seq_lengths"),
                return false);
    OP_CHECK_IF(actSeqLenKV == nullptr, OP_LOGE_WITH_INVALID_INPUT(contextKeyParams.opName, "actual_seq_lengths_kv"),
                return false);

    auto batchOfQuery = actSeqLen->GetShapeSize();
    auto batchOfKey = actSeqLenKV->GetShapeSize();
    OP_CHECK_IF(
        batchOfQuery != batchOfKey,
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
            contextKeyParams.opName, "actual_seq_lengths, actual_seq_lengths_kv",
            (std::to_string(batchOfQuery) + ", " + std::to_string(batchOfKey)).c_str(),
            "when layout is TND, the batch size of actualSequenceLengthQ and actualSequenceLengthKV must be equal"),
        return false);

    int64_t lastActSeq = 0;
    int64_t lastActSeqKV = 0;
    uint32_t batchSize = queryShapeInfo.b; // actSeqLengthSize and actSeqLengthKVSize are equal
    for (uint32_t i = LOOP_BEGIN_NUM; i < batchSize; ++i) {
        int64_t curActSeq = actSeqLen->GetData<int64_t>()[i];
        int64_t curActSeqKV = actSeqLenKV->GetData<int64_t>()[i];
        OP_CHECK_IF(
            curActSeq < 0,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(contextKeyParams.opName, "actual_seq_lengths",
                                                  std::to_string(curActSeq).c_str(), "actualSeqLengths must be >= 0"),
            return false);
        OP_CHECK_IF(((i >= 1) && (curActSeq < lastActSeq)),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        contextKeyParams.opName, "actual_seq_lengths", std::to_string(curActSeq).c_str(),
                        "when layout is TND, actualSeqLengths must not be decreasing"),
                    return false);
        OP_CHECK_IF(curActSeqKV < 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(contextKeyParams.opName, "actual_seq_lengths_kv",
                                                          std::to_string(curActSeqKV).c_str(),
                                                          "actualSeqLengthsKv must be >= 0"),
                    return false);
        if (!enablePA) {
            OP_CHECK_IF(((i >= 1) && (curActSeqKV < lastActSeqKV)),
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            contextKeyParams.opName, "actual_seq_lengths_kv", std::to_string(curActSeqKV).c_str(),
                            "when layout is TND, actualSeqLengthsKv must not be decreasing"),
                        return false);
        }
        lastActSeq = curActSeq;
        lastActSeqKV = curActSeqKV;
    }

    const gert::StorageShape* queryShape = contextKeyParams.queryInputShape;
    const gert::StorageShape* keyShape = contextKeyParams.keyInputShape;
    OP_CHECK_IF(actSeqLen->GetData<int64_t>()[batchSize - 1] != queryShape->GetStorageShape().GetDim(0),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    contextKeyParams.opName, "actual_seq_lengths",
                    std::to_string(actSeqLen->GetData<int64_t>()[batchSize - 1]).c_str(),
                    "when layout is TND, the last element of actual_seq_lengths must equal T"),
                return false);
    if (!enablePA) {
        OP_CHECK_IF(actSeqLenKV->GetData<int64_t>()[batchSize - 1] != keyShape->GetStorageShape().GetDim(0),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        contextKeyParams.opName, "actual_seq_lengths_kv",
                        std::to_string(actSeqLenKV->GetData<int64_t>()[batchSize - 1]).c_str(),
                        "when layout is TND, the last element of actual_seq_lengths_kv must equal T"),
                    return false);
    }

    return true;
}

bool FiaArch38PromptTiling::CheckActSeqLen(FiaArch38ContextParams& contextKeyParams, PFAShapeInfo& queryShapeInfo,
                                           const PFAShapeInfo& keyShapeInfo)
{
    const gert::Tensor* actSeqLen = contextKeyParams.actualSequenceLengthQ;
    const gert::Tensor* actSeqLenKV = contextKeyParams.actualSequenceLengthKV;

    constexpr int64_t actSeqLenDimsQMin = 1;  // The min value of the length of actualSeqQ is 1
    constexpr int64_t actSeqLenDimsKVMin = 1; // The min value of the length of actualSeqKV is 1

    std::string layoutStr(contextKeyParams.layout);
    if (enableActSeqLen) { // check the length of actual_seq_lengthsQ, whether is 1 or batch size
        OP_CHECK_IF(enableIFAMLA && (inputLayout != InputLayout::TND),
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "input_layout, actual_seq_lengths",
                                                           (layoutStr + ", present").c_str(),
                                                           "actual_seq_lengths is only supported when layout is TND"),
                    return false);
        OP_CHECK_IF((actSeqLenDims < queryShapeInfo.b) && (actSeqLenDims > actSeqLenDimsQMin),
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                        contextKeyParams.opName, "actual_seq_lengths", std::to_string(actSeqLenDims).c_str(),
                        "the dim of actual_seq_lengths must be 1 or greater than or equal to batch size"),
                    return false);
        uint32_t actSeqLengthSize = std::min(static_cast<uint32_t>(actSeqLenDims), queryShapeInfo.b);
        for (uint32_t i = LOOP_BEGIN_NUM; i < actSeqLengthSize; ++i) {
            int64_t actSeqTmp = actSeqLen->GetData<int64_t>()[i];
            if (inputLayout == InputLayout::TND && i >= 1) {
                actSeqTmp -= actSeqLen->GetData<int64_t>()[i - 1];
            }
            OP_CHECK_IF(actSeqTmp < 0 || actSeqTmp > queryShapeInfo.s,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(contextKeyParams.opName, "actual_seq_lengths",
                                                              std::to_string(actSeqTmp).c_str(),
                                                              "actual_seq_lengths must be in range [0, s]"),
                        return false);
            // query act seq len padding情况下不支持合轴
            if (actSeqTmp < queryShapeInfo.s) {
                enablePFAMerge = false;
            }
        }
    }

    if (enableActSeqLenKV) { // check the length of actual_seq_lengthsKV,whether is 1 or batch size
        // Batch of key is 1 under tensorlist mode, the number of actual_seq_lengthsKV should be compared
        // with batch of query, batch of key is equal to query under non-tensorlist mode.
        OP_CHECK_IF((actSeqLenKVDims < queryShapeInfo.b) && (actSeqLenKVDims > actSeqLenDimsKVMin),
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                        contextKeyParams.opName, "actual_seq_lengths_kv", std::to_string(actSeqLenKVDims).c_str(),
                        "the dim of actual_seq_lengths_kv must be 1 or greater than or equal to batch size"),
                    return false);
        uint32_t actSeqLengthKVSize = std::min(static_cast<uint32_t>(actSeqLenKVDims), queryShapeInfo.b);
        for (uint32_t i = LOOP_BEGIN_NUM; i < actSeqLengthKVSize; ++i) {
            int64_t actSeqKVTmp = static_cast<int64_t>(actSeqLenKV->GetData<int64_t>()[i]);
            if (inputLayout == InputLayout::TND && !enablePA && i >= 1) {
                actSeqKVTmp -= static_cast<int64_t>(actSeqLenKV->GetData<int64_t>()[i - 1]);
            }
            if (!enableTensorList && !enablePA) {
                int64_t actSeqKVMax = static_cast<int64_t>(keyShapeInfo.s);
                OP_CHECK_IF((actSeqKVTmp < 0) || (actSeqKVTmp > actSeqKVMax),
                            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                                contextKeyParams.opName, "actual_seq_lengths_kv", std::to_string(actSeqKVTmp).c_str(),
                                "actual_seq_lengths_kv must be in range [0, actSeqKVMax]"),
                            return false);
            } else if (enableTensorList) {
                int64_t actSeqKVMax;
                if ((inputLayout == InputLayout::BSND) || (inputLayout == InputLayout::BSH)) {
                    actSeqKVMax = contextKeyParams.kTensorList[i]->GetStorageShape().GetDim(1);
                } else {
                    actSeqKVMax = contextKeyParams.kTensorList[i]->GetStorageShape().GetDim(2); // 2: Dim S for BNSD
                }
                OP_CHECK_IF((actSeqKVTmp < 0) || (actSeqKVTmp > actSeqKVMax),
                            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                                contextKeyParams.opName, "actual_seq_lengths_kv", std::to_string(actSeqKVTmp).c_str(),
                                "actual_seq_lengths_kv must be in range [0, actSeqKVMax]"),
                            return false);
            }
        }
    }
    return true;
}

bool FiaArch38PromptTiling::CheckPATypeAndShape(FiaArch38ContextParams& contextKeyParams, PFAShapeInfo& queryShapeInfo,
                                                PFAShapeInfo& queryRopeShapeInfo,
                                                FiaArch38FlashAttentionTilingData& tilingData)
{
    // The interception that is mutually exclusive with the left padding has been implemented in FIA.
    OP_CHECK_IF(enableTensorList,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "block_table, tensorlist",
                                                       "block_table present with tensorlist",
                                                       "tensorlist is not supported when blockTable is not null"),
                return false);
    OP_CHECK_IF((!enableActSeqLenKV) && (contextKeyParams.fromTilingSink == 0),
                OP_LOGE_WITH_INVALID_INPUT(contextKeyParams.opName, "actual_seq_lengths_kv"), return false);

    const gert::StorageShape* blockTableShape = contextKeyParams.blockTableShape;
    // check blockTable shape is nullptr
    OP_CHECK_IF(blockTableShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(contextKeyParams.opName, "block_table"),
                return false);

    // check blockTable each dim cannot be 0
    OP_CHECK_IF(
        blockTableShape->GetStorageShape().GetShapeSize() == 0,
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(contextKeyParams.opName, "block_table",
                                                   (std::to_string(blockTableShape->GetStorageShape().GetDim(0)) +
                                                    ", " + std::to_string(blockTableShape->GetStorageShape().GetDim(1)))
                                                       .c_str(),
                                                   "blockTable each dim must not be 0"),
        return false);

    // check Dtype
    ge::DataType blockTableType = contextKeyParams.blockTableType;
    OP_CHECK_IF((blockTableType != ge::DT_INT32),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(contextKeyParams.opName, "block_table",
                                                      GetPfaDataTypeStr(blockTableType).c_str(),
                                                      "blockTable only supports int32 when PA enable"),
                return false);

    // check block table shape
    const int32_t* blockSize = contextKeyParams.blockSize;
    // Tiling sinking scene, workspace needs to be calculated, at this time, blockTableDim2 * blockSize is used as S2.
    blockTableDim2 = static_cast<int32_t>(blockTableShape->GetStorageShape().GetDim(1));
    // PFA PA blockSize % 128 == 0
    OP_CHECK_IF((!enableIFAMLA && !enableIFA && !(queryShapeInfo.s == 1 && enableAlibiPse) &&
                 (*blockSize % BLOCK_SIZE_BASE != 0 || *blockSize < BLOCK_SIZE_BASE || *blockSize > BLOCK_SIZE_MAX)),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    contextKeyParams.opName, "block_size", std::to_string(*blockSize).c_str(),
                    "block size must be a multiple of BLOCK_SIZE_BASE and within the allowed range"),
                return false);
    // IFA PA blockSize % 16 == 0
    ifaBlockSizeBase /= static_cast<int32_t>(dataTypeSize);
    OP_CHECK_IF(((enableIFAMLA || enableIFA || (queryShapeInfo.s == 1 && enableAlibiPse)) &&
                 (*blockSize % ifaBlockSizeBase != 0 || *blockSize < ifaBlockSizeBase || *blockSize > BLOCK_SIZE_MAX)),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    contextKeyParams.opName, "block_size", std::to_string(*blockSize).c_str(),
                    "block size must be a multiple of ifaBlockSizeBase and within the allowed range"),
                return false);

    if (contextKeyParams.fromTilingSink != 0) {
        S2 = blockTableDim2 * (*blockSize);
        return true;
    }
    if (!CheckBlockTableShape(contextKeyParams, queryShapeInfo, queryRopeShapeInfo, blockSize, blockTableShape,
                              tilingData)) {
        return false;
    }

    tilingData.fiaArch38AttentionBaseParams.set_PAlayoutType(paLayoutType);
    return true;
}

bool FiaArch38PromptTiling::CheckPseShiftTypeAndShape(FiaArch38ContextParams& contextKeyParams, uint32_t b, uint32_t n,
                                                      uint32_t s1, uint32_t s2)
{
    const gert::StorageShape* pseShiftShape = contextKeyParams.pseShiftShape;
    if (!CheckNonEmptyShapeExceptions(contextKeyParams, pseShiftShape, "pseShift")) {
        return false;
    }
    std::string layoutStr(contextKeyParams.layout);
    pseShiftElemType = contextKeyParams.pseShiftDataType;

    OP_CHECK_IF((inputType == ge::DT_FLOAT16 && pseShiftElemType != ge::DT_FLOAT16),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(contextKeyParams.opName, "pse_shift",
                                                      GetPfaDataTypeStr(pseShiftElemType).c_str(),
                                                      "when query dtype is fp16, pse shift type must be fp16"),
                return false);

    OP_CHECK_IF((inputType == ge::DT_BF16 && pseShiftElemType != ge::DT_BF16),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(contextKeyParams.opName, "pse_shift",
                                                      GetPfaDataTypeStr(pseShiftElemType).c_str(),
                                                      "when query dtype is bf16, pse shift type must be bf16"),
                return false);

    OP_CHECK_IF((inputType == ge::DT_INT8 && pseShiftElemType != ge::DT_FLOAT16),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(contextKeyParams.opName, "pse_shift",
                                                      GetPfaDataTypeStr(pseShiftElemType).c_str(),
                                                      "when query dtype is int8, pse shift type must be fp16"),
                return false);

    if (pseShiftElemType == ge::DT_FLOAT16) {
        pseShiftElemSize = FLOAT16SIZE;
    } else if (pseShiftElemType == ge::DT_BF16) {
        pseShiftElemSize = BFLOAT16SIZE;
    }
    pseShiftTypeByteNum = BYTE_BLOCK / pseShiftElemSize;

    size_t pseShiftDim = pseShiftShape->GetStorageShape().GetDimNum();
    OP_CHECK_IF((pseShiftDim != PSESHIFTDIM_4),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(contextKeyParams.opName, "pse_shift",
                                                         std::to_string(pseShiftDim).c_str(),
                                                         "pse shift shape must be 4 dimension"),
                return false);

    pseShiftBatch = pseShiftShape->GetStorageShape().GetDim(0);
    int64_t pseShiftN = pseShiftShape->GetStorageShape().GetDim(1); // 1: The sirst dimension is N.
    pseShiftS1 = pseShiftShape->GetStorageShape().GetDim(2);        // 2: The second dimension is S1.
    pseShiftS2 = pseShiftShape->GetStorageShape().GetDim(3);        // 3: The third dimension is S2.
    OP_CHECK_IF(
        ((pseShiftBatch != 1 && pseShiftBatch != b) || (pseShiftN != n) || (pseShiftS1 < s1) || (pseShiftS2 < s2)),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            contextKeyParams.opName, "pse_shift",
            ("[" + std::to_string(pseShiftBatch) + ", " + std::to_string(pseShiftN) + ", " +
             std::to_string(pseShiftS1) + ", " + std::to_string(pseShiftS2) + "]")
                .c_str(),
            "pse shift shape must be [1 or b, n, >=s1, >=s2]"),
        return false);
    return true;
}

bool FiaArch38PromptTiling::CheckInnerPrecise(FiaArch38ContextParams& contextKeyParams,
                                              FiaArch38FlashAttentionTilingData& tilingData)
{
    // 0: Invalid plural number; 4: Invalid if greater than or equal to 4; 0,1,2,3 are effective values for
    // innerPrecise.
    if (innerPrecise >= 4 || innerPrecise < 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(contextKeyParams.opName, "inner_precise",
                                              std::to_string(innerPrecise).c_str(),
                                              "inner_precise should be 0, 1, 2 or 3");
        return false;
    }
    // Determine if the bit1 bit of innerPrecise requires invalid correction.
    tilingData.fiaArch38AttentionBaseParams.set_isRowInvalid(
        (static_cast<uint32_t>(static_cast<uint32_t>(innerPrecise) >> 1U)) & 1U);
    // Determine the bit0 bit of innerPrecise, high-performance or high-precision mode.
    innerPrecise = HIGH_PRECISION; // High-precision contains high-performance and high-precision mode.

    innerPrecise = HIGH_PERFORMANCE;

    // FP16 pse is forced to enter high-precision mode.
    if ((contextKeyParams.pseShift != nullptr) && (inputType == ge::DT_FLOAT16) && (innerPrecise == HIGH_PERFORMANCE)) {
        innerPrecise = HIGH_PRECISION;
        OP_LOGW(contextKeyParams.opName,
                "when the input is fp16 and enable pseshift, the mode is forcibly switched to high-precision!");
    }
    if (inputType != ge::DT_FLOAT16) {
        OP_LOGW(contextKeyParams.opName, "innerPrecise will not take effect when input type is %s!",
                GetPfaDataTypeStr(inputType).c_str());
    }
    if ((inputType == ge::DT_FLOAT16) && (innerPrecise == HIGH_PERFORMANCE)) {
        softmaxDataTypeSize = FLOAT16SIZE; // The default size is fp32.
    }
    return true;
}

bool FiaArch38PromptTiling::CheckMaskTypeAndShape(FiaArch38ContextParams& contextKeyParams,
                                                  FiaArch38FlashAttentionTilingData& tilingData)
{
    const gert::StorageShape* attenMaskShape = contextKeyParams.attentionMaskShape;
    OP_CHECK_IF((attenMaskShape->GetStorageShape().GetShapeSize() == gert::Shape::kInvalidDimValue),
                ReportTilingFailure(context_, "get the shape size of atten_mask failed"), return false);

    auto maskDataType = contextKeyParams.maskDataType;
    std::vector<ge::DataType> allowedMaskDtypes = {ge::DT_FLOAT16, ge::DT_BOOL, ge::DT_INT8, ge::DT_UINT8};
    std::vector<uint32_t> maskTypeSizeArray = {FLOAT16SIZE, BOOLSIZE, INT8SIZE, UINT8SIZE};

    auto maskTypeCheck = std::find(allowedMaskDtypes.begin(), allowedMaskDtypes.end(), maskDataType);

    OP_CHECK_IF((maskTypeCheck == allowedMaskDtypes.end()),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(contextKeyParams.opName, "atten_mask",
                                                      GetPfaDataTypeStr(maskDataType).c_str(),
                                                      "maskType should be DT_FLOAT16, DT_BOOL, DT_INT8 or DT_UINT8"),
                return false);
    int32_t maskTypeIndex = std::distance(allowedMaskDtypes.begin(), maskTypeCheck);
    maskElemSize = maskTypeSizeArray[maskTypeIndex];

    // When in fp16 high-precision mode, the mask type only supports bool or int8.
    OP_CHECK_IF(((inputType == ge::DT_FLOAT16) && (innerPrecise == HIGH_PRECISION)) && (maskDataType != ge::DT_BOOL) &&
                    (maskDataType != ge::DT_INT8) && (maskDataType != ge::DT_UINT8),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    contextKeyParams.opName, "atten_mask", GetPfaDataTypeStr(maskDataType).c_str(),
                    "when innerprecise is 2/3 and input type is fp16, maskType should be bool, int8 or uint8"),
                return false);
    // When bf16, the mask type only supports bool or int8.
    OP_CHECK_IF((inputType == ge::DT_BF16) && (maskDataType != ge::DT_BOOL) && (maskDataType != ge::DT_INT8) &&
                    (maskDataType != ge::DT_UINT8),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    contextKeyParams.opName, "atten_mask", GetPfaDataTypeStr(maskDataType).c_str(),
                    "when input type is bfloat16, maskType should be bool, int8 or uint8"),
                return false);

    int64_t maskKVsSize = 2048; // 2048 : default the last frist dim.
    int64_t maskQsSize = 2048;  // 2048 : default the last second dim.
    // 1: last frist dim, 2: last second dim
    maskKVsSize = attenMaskShape->GetStorageShape().GetDim(attenMaskShape->GetStorageShape().GetDimNum() - 1);
    maskQsSize = attenMaskShape->GetStorageShape().GetDim(attenMaskShape->GetStorageShape().GetDimNum() -
                                                          2); // 2 for Q dim index

    if (enableIFA) {
        maskQsSize = 1;
    }

    tilingData.fiaArch38AttentionBaseParams.set_maskKVsSize(maskKVsSize);
    tilingData.fiaArch38AttentionBaseParams.set_maskQsSize(maskQsSize);
    return true;
}

void FiaArch38PromptTiling::SetSparseType(uint32_t qS)
{
    if (sparseModeVal == SPARSE_MODE_NO_MASK) {
        if (sparsePreTokens >= qS && sparseNextTokens == 0) {
            sparseType = static_cast<uint8_t>(PfaSparseEnum::PFA_CAUSAL);
        } else if (sparsePreTokens >= qS && !enablePA && sparseNextTokens >= S2) {
            sparseType = static_cast<uint8_t>(PfaSparseEnum::PFA_ALL);
        } else {
            sparseType = static_cast<uint8_t>(PfaSparseEnum::PFA_BAND);
        }
    } else if (sparseModeVal == SPARSE_MODE_ALL_MASK) {
        sparseType = static_cast<uint8_t>(PfaSparseEnum::PFA_ALL);
    } else if (sparseModeVal == SPARSE_MODE_LEFT_UP) {
        sparseType = static_cast<uint8_t>(PfaSparseEnum::PFA_CAUSAL);
    } else if (sparseModeVal == SPARSE_MODE_RIGHT_DOWN) {
        if (!enablePA && qS == S2) {
            sparseType = static_cast<uint8_t>(PfaSparseEnum::PFA_CAUSAL);
        } else {
            sparseType = static_cast<uint8_t>(PfaSparseEnum::PFA_BAND);
        }
    } else if (sparseModeVal == SPARSE_MODE_BAND) {
        sparseType = static_cast<uint8_t>(PfaSparseEnum::PFA_BAND);
    }
}

bool FiaArch38PromptTiling::CheckSparseMode(FiaArch38ContextParams& contextKeyParams, uint32_t qS,
                                            FiaArch38FlashAttentionTilingData& tilingData)
{
    const int32_t* sparseMode = contextKeyParams.sparseMode;
    const int64_t* nextTokens = contextKeyParams.nextToken;
    const int64_t* preTokens = contextKeyParams.preToken;

    bool sparseCheck = false;
    if (sparseMode != nullptr) {
        sparseCheck = ((*sparseMode != SPARSE_MODE_NO_MASK) && (*sparseMode != SPARSE_MODE_LEFT_UP) &&
                       (*sparseMode != SPARSE_MODE_RIGHT_DOWN) && (*sparseMode != SPARSE_MODE_ALL_MASK) &&
                       (*sparseMode != SPARSE_MODE_BAND));
        OP_CHECK_IF(sparseCheck,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        contextKeyParams.opName, "sparse_mode", std::to_string(*sparseMode).c_str(),
                        "sparse_mode is out of range, only 0, 1, 2, 3, 4 are supported"),
                    return false);
    }
    const gert::StorageShape* attenMaskShape = contextKeyParams.attentionMaskShape;
    SetSparseModeData(contextKeyParams, attenMaskShape, tilingData, sparseMode, preTokens, nextTokens);

    OP_CHECK_IF(((attenMaskShape != nullptr) && (sparseMode != nullptr) && (*sparseMode == SPARSE_MODE_BAND) &&
                 (*preTokens < 0 && *nextTokens < 0)),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    contextKeyParams.opName, "pre_tokens, next_tokens",
                    (std::to_string(*preTokens) + ", " + std::to_string(*nextTokens)).c_str(),
                    "preTokens and nextTokens must not be negative in band mode"),
                return false);

    OP_CHECK_IF((sparsePreTokens < 0) && (sparseNextTokens < 0),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    contextKeyParams.opName, "pre_tokens, next_tokens",
                    (std::to_string(sparsePreTokens) + ", " + std::to_string(sparseNextTokens)).c_str(),
                    "preTokens and nextTokens must not be negative"),
                return false);

    OP_CHECK_IF((sparseNextTokens * (-1)) > sparsePreTokens,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    contextKeyParams.opName, "pre_tokens, next_tokens",
                    (std::to_string(sparsePreTokens) + ", " + std::to_string(sparseNextTokens)).c_str(),
                    "nexttoken line should be higher than pretoken line"),
                return false);

    OP_CHECK_IF(isDefaultSparseMode && (sparseNextTokens < 0) && (sparseNextTokens * (-1)) >= static_cast<int32_t>(qS),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    contextKeyParams.opName, "next_tokens", std::to_string(sparseNextTokens).c_str(),
                    "nextTokens absolute value should be smaller than the length of query"),
                return false);

    OP_CHECK_IF(
        isDefaultSparseMode && (sparsePreTokens < 0) &&
            (sparsePreTokens * (-1) >= (static_cast<int32_t>(S2) + static_cast<int32_t>(actualSharedPrefixLen))),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            contextKeyParams.opName, "pre_tokens", std::to_string(sparsePreTokens).c_str(),
            "preTokens absolute value should be smaller than the length of key and value"),
        return false);

    SetSparseType(qS);
    return true;
}

bool FiaArch38PromptTiling::CheckPACrossover(FiaArch38ContextParams& contextKeyParams, PFAShapeInfo& queryShapeInfo)
{
    if (enablePA) {
        OP_CHECK_IF((isKVHasPrefix),
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        contextKeyParams.opName, "prefix, block_table", "system prefix with page attention",
                        "when system prefix is used, page attention is not supported"),
                    return false);
        OP_CHECK_IF((enableKVAntiquant),
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "block_table, antiquant_scale",
                                                           "block_table present with antiquant",
                                                           "antiquant is not supported when blockTable is not null"),
                    return false);
    }
    if (enableActSeqLenKV && !enableTensorList) {
        const gert::Tensor* actSeqLenKV = contextKeyParams.actualSequenceLengthKV;
        uint32_t actSeqLenKVSize = std::min(static_cast<uint32_t>(actSeqLenKVDims), queryShapeInfo.b);
        for (uint32_t i = LOOP_BEGIN_NUM; i < actSeqLenKVSize; ++i) {
            OP_CHECK_IF(
                enablePA && (actSeqLenKV->GetData<int64_t>()[i] < 0),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(contextKeyParams.opName, "actual_seq_lengths_kv",
                                                      std::to_string(actSeqLenKV->GetData<int64_t>()[i]).c_str(),
                                                      "actual_seq_lengths_kv must be >= 0"),
                return false);
        }
    }
    return true;
}

bool FiaArch38PromptTiling::CheckMaskCrossIFAMLA(FiaArch38ContextParams& contextKeyParams, const int32_t* sparseMode,
                                                 uint64_t queryS)
{
    if (sparseMode == nullptr) {
        return true;
    }
    if (queryS == 1U) {
        OP_CHECK_IF(!((*sparseMode == SPARSE_MODE_NO_MASK) && (!enableMask)),
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        contextKeyParams.opName, "sparse_mode, atten_mask",
                        (std::to_string(*sparseMode) + ", " + std::string(enableMask ? "present" : "absent")).c_str(),
                        "only sparse 0 without mask is supported when ifa mla and query sequence length is 1"),
                    return false);
    } else {
        OP_CHECK_IF(!(((*sparseMode == SPARSE_MODE_RIGHT_DOWN) && (enableMask)) ||
                      ((*sparseMode == SPARSE_MODE_NO_MASK) && (!enableMask))),
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        contextKeyParams.opName, "sparse_mode, atten_mask",
                        (std::to_string(*sparseMode) + ", " + std::string(enableMask ? "present" : "absent")).c_str(),
                        "only sparse 3 with mask or sparse 0 without mask is supported when ifa mla and "
                        "query sequence length is > 1"),
                    return false);
    }
    return true;
}

bool FiaArch38PromptTiling::CheckMaskCrossover(FiaArch38ContextParams& contextKeyParams, PFAShapeInfo& queryShapeInfo,
                                               FiaArch38FlashAttentionTilingData& tilingData)
{
    auto maskDataType = contextKeyParams.maskDataType;
    const int32_t* sparseMode = contextKeyParams.sparseMode;
    const gert::StorageShape* attenMaskShape = contextKeyParams.attentionMaskShape;
    if ((sparseMode != nullptr) && (*sparseMode == SPARSE_MODE_LEFT_UP || *sparseMode == SPARSE_MODE_RIGHT_DOWN ||
                                    *sparseMode == SPARSE_MODE_ALL_MASK || *sparseMode == SPARSE_MODE_BAND)) {
        OP_CHECK_IF(((attenMaskShape != nullptr) && (attenMaskShape->GetStorageShape().GetShapeSize() == 0)) ||
                        (attenMaskShape == nullptr),
                    OP_LOGE_WITH_INVALID_INPUT(contextKeyParams.opName, "atten_mask"), return false);

        // When sparse=2, 3, 4, the mask type only supports bool, int8, uint8
        OP_CHECK_IF((*sparseMode != SPARSE_MODE_ALL_MASK) && (maskDataType != ge::DT_BOOL) &&
                        (maskDataType != ge::DT_INT8) && (maskDataType != ge::DT_UINT8),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        contextKeyParams.opName, "atten_mask", GetPfaDataTypeStr(maskDataType).c_str(),
                        "maskType should be bool, int8 or uint8 for this sparse mode"),
                    return false);
    }
    // FP16 mask type does not support invalid line correction.
    OP_CHECK_IF((maskDataType == ge::DT_FLOAT16 &&
                 static_cast<bool>(tilingData.fiaArch38AttentionBaseParams.get_isRowInvalid())),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(contextKeyParams.opName, "atten_mask",
                                                      GetPfaDataTypeStr(maskDataType).c_str(),
                                                      "maskType should not be float16 when innerPrecise is 2 or 3"),
                return false);
    if (!CheckMaskShapeCrossSparse(contextKeyParams, sparseMode, queryShapeInfo.s, S2 + actualSharedPrefixLen,
                                   queryShapeInfo.b)) {
        return false;
    }
    if (enableIFAMLA && (!CheckMaskCrossIFAMLA(contextKeyParams, sparseMode, queryShapeInfo.s / gSize))) {
        return false;
    }
    return true;
}

bool FiaArch38PromptTiling::CheckTNDLayoutCrossover(FiaArch38ContextParams& contextKeyParams)
{
    if (inputLayout != InputLayout::TND) {
        return true;
    }

    OP_CHECK_IF(enableLeftPadding,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "input_layout, leftPadding",
                                                       "TND with left padding",
                                                       "when layout is TND, left padding is not supported"),
                return false);

    OP_CHECK_IF(enableTensorList,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "input_layout, tensorlist",
                                                       "TND with tensorlist",
                                                       "when layout is TND, tensorlist is not supported"),
                return false);

    OP_CHECK_IF(enablePseShift,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "input_layout, pse_shift",
                                                       "TND with pse", "when layout is TND, pse is not supported"),
                return false);

    return true;
}

bool FiaArch38PromptTiling::ParseActualSeqLengths(FiaArch38ContextParams& contextKeyParams,
                                                  PFAShapeInfo& queryShapeInfo, std::vector<int64_t>& actualSeqLengths,
                                                  std::vector<int64_t>& actualSeqLengthsKV)
{
    uint32_t lenDims = queryShapeInfo.b; // The current length of the actSeqLen array is equal to batch size b.
    const gert::Tensor* actSeqLenData = contextKeyParams.actualSequenceLengthQ;
    const gert::Tensor* actSeqLenDataKV = contextKeyParams.actualSequenceLengthKV;
    actSeqLenDims = (actSeqLenData != nullptr) ? actSeqLenData->GetShapeSize() : 0;
    actSeqLenKVDims = (actSeqLenDataKV != nullptr) ? actSeqLenDataKV->GetShapeSize() : 0;

    if (inputLayout == InputLayout::TND) {
        if ((actSeqLenData == nullptr) || (actSeqLenDataKV == nullptr)) {
            return false;
        }
        middleActualSeqLengths = static_cast<uint32_t>(actSeqLenData->GetData<int64_t>()[lenDims - 1]);
    }
    for (uint32_t i = LOOP_BEGIN_NUM; i < lenDims; i++) {
        if (inputLayout == InputLayout::TND) {
            actualSeqLengths[i] = (i == 0) ? static_cast<uint32_t>(actSeqLenData->GetData<int64_t>()[0]) :
                                             static_cast<uint32_t>(actSeqLenData->GetData<int64_t>()[i]) -
                                                 static_cast<uint32_t>(actSeqLenData->GetData<int64_t>()[i - 1]);
            actualSeqLengthsKV[i] = static_cast<uint32_t>(actSeqLenDataKV->GetData<int64_t>()[i]);
            if (enableIFAMLA) {
                actualSeqLengths[i] *= gSize;
            }
            if (!enablePA && i >= 1) {
                actualSeqLengthsKV[i] -= static_cast<uint32_t>(actSeqLenDataKV->GetData<int64_t>()[i - 1]);
            }
            if (actualSeqLengthsKV[i] != S2) {
                needInit = 1;
            }
            continue;
        }

        if (!enableActSeqLen) {
            actualSeqLengths[i] = queryShapeInfo.s;
        } else if (enableIFAMLA) {
            actualSeqLengths[i] = (actSeqLenDims > 1) ? static_cast<uint32_t>(actSeqLenData->GetData<int64_t>()[i]) :
                                                        static_cast<uint32_t>(actSeqLenData->GetData<int64_t>()[0]);
            if (actualSeqLengths[i] != queryShapeInfo.s / gSize) {
                needInit = 1;
                OP_CHECK_IF((sparseNextTokens < 0) &&
                                (sparseNextTokens * (-1) >= static_cast<int32_t>(actualSeqLengths[i])) &&
                                isDefaultSparseMode,
                            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                                contextKeyParams.opName, "next_tokens", std::to_string(sparseNextTokens).c_str(),
                                "nexttoken absolute value should be smaller than the actual length of query"),
                            return false);
            }
            if (inputLayout == InputLayout::BSND || inputLayout == InputLayout::BSH) {
                actualSeqLengths[i] *= gSize;
            } else {
                actualSeqLengths[i] = queryShapeInfo.s;
            }
        } else {
            actualSeqLengths[i] = (actSeqLenDims > 1) ? static_cast<uint32_t>(actSeqLenData->GetData<int64_t>()[i]) :
                                                        static_cast<uint32_t>(actSeqLenData->GetData<int64_t>()[0]);
            if (actualSeqLengths[i] != queryShapeInfo.s) {
                needInit = 1;
                OP_CHECK_IF((sparseNextTokens < 0) &&
                                (sparseNextTokens * (-1) >= static_cast<int32_t>(actualSeqLengths[i])) &&
                                isDefaultSparseMode,
                            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                                contextKeyParams.opName, "next_tokens", std::to_string(sparseNextTokens).c_str(),
                                "nexttoken absolute value should be smaller than the actual length of query"),
                            return false);
            }
        }
        middleActualSeqLengths += actualSeqLengths[i];
        if (!enableActSeqLenKV) { // The user did not input act_seq_kv
            if (!enableTensorList) {
                actualSeqLengthsKV[i] = S2;
            } else {
                if ((inputLayout == InputLayout::BSND) || (inputLayout == InputLayout::BSH)) {
                    actualSeqLengthsKV[i] = contextKeyParams.kTensorList[i]->GetStorageShape().GetDim(1);
                } else { // 2: Obtain the second dimension
                    actualSeqLengthsKV[i] = contextKeyParams.kTensorList[i]->GetStorageShape().GetDim(2);
                }
            }
        } else {
            actualSeqLengthsKV[i] = (actSeqLenKVDims > 1) ?
                                        static_cast<uint32_t>(actSeqLenDataKV->GetData<int64_t>()[i]) :
                                        static_cast<uint32_t>(actSeqLenDataKV->GetData<int64_t>()[0]);
            if (actualSeqLengthsKV[i] != S2) {
                needInit = 1;
            }
        }
        maxActualseqKV = std::max(maxActualseqKV, actualSeqLengthsKV[i]);
    }

    return true;
}

bool FiaArch38PromptTiling::CheckMultiFeatureCrossover(FiaArch38ContextParams& contextKeyParams,
                                                       PFAShapeInfo& queryShapeInfo,
                                                       std::vector<int64_t>& actualSeqLengths,
                                                       std::vector<int64_t>& actualSeqLengthsKV,
                                                       FiaArch38FlashAttentionTilingData& tilingData)
{
    if (!ParseActualSeqLengths(contextKeyParams, queryShapeInfo, actualSeqLengths, actualSeqLengthsKV)) {
        return false;
    }

    uint32_t lenDims = queryShapeInfo.b; // The current length of the actSeqLen array is equal to batch size b.
    int64_t preTokensPerbatch = 0;
    int64_t nextTokensPerbatch = 0;
    for (uint32_t i = LOOP_BEGIN_NUM; i < lenDims; i++) {
        OP_CHECK_IF(isDefaultSparseMode && (sparsePreTokens < 0) &&
                        (sparsePreTokens * (-1) >= (actualSeqLengthsKV[i] + actualSharedPrefixLen)),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        contextKeyParams.opName, "pre_tokens", std::to_string(sparsePreTokens).c_str(),
                        "preToken absolute value should be smaller than the actual length of key and value"),
                    return false);
        if (sparseModeVal == SPARSE_MODE_RIGHT_DOWN) {
            preTokensPerbatch = SPARSE_MODE_INT_MAX;
            if (enableIFAMLA) {
                nextTokensPerbatch = actualSeqLengthsKV[i] + actualSharedPrefixLen - actualSeqLengths[i] / gSize;
            } else {
                nextTokensPerbatch = actualSeqLengthsKV[i] + actualSharedPrefixLen - actualSeqLengths[i];
            }
        } else if (sparseModeVal == SPARSE_MODE_BAND) {
            preTokensPerbatch = sparsePreTokens - actualSeqLengthsKV[i] - actualSharedPrefixLen + actualSeqLengths[i];
            nextTokensPerbatch = sparseNextTokens + actualSeqLengthsKV[i] + actualSharedPrefixLen - actualSeqLengths[i];
        } else {
            preTokensPerbatch = sparsePreTokens;
            nextTokensPerbatch = sparseNextTokens;
        }
        if ((nextTokensPerbatch < 0) ||
            (actualSeqLengths[i] > (actualSeqLengthsKV[i] + actualSharedPrefixLen + preTokensPerbatch))) {
            needInit = 1;
        }

        OP_LOGI(contextKeyParams.opName, "preTokensPerbatch[%u] is %ld, nextTokensPerbatch[%u] is %ld", i,
                preTokensPerbatch, i, nextTokensPerbatch);

        OP_CHECK_IF((isBandMode && (sparseNextTokens < 0) &&
                     (sparseNextTokens * (-1) >= actualSeqLengthsKV[i] + actualSharedPrefixLen)),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        contextKeyParams.opName, "next_tokens", std::to_string(sparseNextTokens).c_str(),
                        "nextTokens absolute value should be smaller than the actual length of key and value "
                        "in band mode"),
                    return false);

        OP_CHECK_IF((isBandMode && (sparsePreTokens < 0) && (sparsePreTokens * (-1) >= actualSeqLengths[i])),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        contextKeyParams.opName, "pre_tokens", std::to_string(sparsePreTokens).c_str(),
                        "preTokens absolute value should be smaller than the actual length of query in band mode"),
                    return false);

        OP_LOGI(
            contextKeyParams.opName,
            "actualSeqLengths[%u] is %ld, actualSeqLengthsKV[%u] is %ld, actualSharedPrefixLen is %ld, needInit is %u",
            i, actualSeqLengths[i], i, actualSeqLengthsKV[i], actualSharedPrefixLen, needInit);

        tilingData.fiaArch38AttentionInitOutputParams.set_needInit(needInit);

        if (enableAlibiPse) {
            OP_CHECK_IF(
                (actualSeqLengths[i] != actualSeqLengthsKV[i]),
                OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                    contextKeyParams.opName, "actual_seq_lengths, actual_seq_lengths_kv",
                    (std::to_string(actualSeqLengths[i]) + ", " + std::to_string(actualSeqLengthsKV[i])).c_str(),
                    "when pseType is 2/3, actualSeqLengths must be equal to actualSeqLengthsKv"),
                return false);
        }
    }
    return true;
}

bool FiaArch38PromptTiling::CheckPerblockCrossover(FiaArch38ContextParams& contextKeyParams)
{
    if (!enablePerblockQuant) {
        return true;
    }
    OP_CHECK_IF(enableActSeqLen,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "actual_seq_lengths, per-block quant",
                                                       "actualSeqLen present with per-block quant",
                                                       "ActSeqLen is not supported in per-block quant scenario"),
                return false);
    OP_CHECK_IF(
        enableActSeqLenKV,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "actual_seq_lengths_kv, per-block quant",
                                               "actualSeqLenKV present with per-block quant",
                                               "ActSeqLenKV is not supported in per-block quant scenario"),
        return false);
    OP_CHECK_IF((innerPrecise == 2) || (innerPrecise == 3),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(contextKeyParams.opName, "inner_precise",
                                                      std::to_string(innerPrecise).c_str(),
                                                      "innerPrecise must be 0 or 1 in per-block quant scenario"),
                return false);
    OP_CHECK_IF(enablePA,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "block_table, per-block quant",
                                                       "PA enabled with per-block quant",
                                                       "PA is not supported in per-block quant scenario"),
                return false);
    OP_CHECK_IF(enableTensorList,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "tensorlist, per-block quant",
                                                       "tensorlist with per-block quant",
                                                       "tensorlist is not supported in per-block quant scenario"),
                return false);
    OP_CHECK_IF(enableLeftPadding,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "leftPadding, per-block quant",
                                                       "left padding with per-block quant",
                                                       "leftpadding is not supported in per-block quant scenario"),
                return false);
    OP_CHECK_IF(enablePFAMLA,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "query_rope/key_rope, per-block quant",
                                                       "PFAMLA with per-block quant",
                                                       "PFAMLA is not supported in per-block quant scenario"),
                return false);
    OP_CHECK_IF(enablePFARope,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "query_rope/key_rope, per-block quant",
                                                       "PFARope with per-block quant",
                                                       "PFARope is not supported in per-block quant scenario"),
                return false);
    OP_CHECK_IF(enableMask,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "atten_mask, per-block quant",
                                                       "mask with per-block quant",
                                                       "mask is not supported in per-block quant scenario"),
                return false);
    OP_CHECK_IF(enablePseShift,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "pse_shift, per-block quant",
                                                       "PseShift with per-block quant",
                                                       "PseShift is not supported in per-block quant scenario"),
                return false);
    OP_CHECK_IF(enableAlibiPse,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "pse_shift, per-block quant",
                                                       "AlibiPse with per-block quant",
                                                       "AlibiPse is not supported in per-block quant scenario"),
                return false);
    return true;
}

void FiaArch38PromptTiling::SetTilingDataAttribute(FiaArch38ContextParams& contextKeyParams,
                                                   FiaArch38FlashAttentionTilingData& tilingData)
{
    tilingData.fiaArch38AttentionBaseParams.set_preTokens(sparsePreTokens);
    tilingData.fiaArch38AttentionBaseParams.set_nextTokens(sparseNextTokens);
    tilingData.fiaArch38AttentionBaseParams.set_sparseMode(static_cast<uint32_t>(sparseModeVal));

    bool isActualSeqLengthsNull = contextKeyParams.fromTilingSink == 0 ? !enableActSeqLen : true;
    bool isActualSeqLengthsKVNull = contextKeyParams.fromTilingSink == 0 ? !enableActSeqLenKV : true;
    tilingData.fiaArch38AttentionBaseParams.set_isActualSeqLengthsNull(static_cast<uint32_t>(isActualSeqLengthsNull));
    tilingData.fiaArch38AttentionBaseParams.set_isActualSeqLengthsKVNull(
        static_cast<uint32_t>(isActualSeqLengthsKVNull));
    tilingData.fiaArch38AttentionBaseParams.set_actualSeqLengthsSize(actSeqLenDims);
    tilingData.fiaArch38AttentionBaseParams.set_actualSeqLengthsKVSize(actSeqLenKVDims);

    tilingData.fiaArch38AttentionBaseParams.set_usePseShift(usePseShift);
    tilingData.fiaArch38AttentionBaseParams.set_pseShiftTypeByteNum(pseShiftTypeByteNum);
    tilingData.fiaArch38AttentionBaseParams.set_pseMaskMaxSize(pseMaskMaxSize);
    tilingData.fiaArch38AttentionSingleCoreParams.set_pseShiftBatch(pseShiftBatch);
    tilingData.fiaArch38AttentionBaseParams.set_pseShiftS1Size(pseShiftS1);
    tilingData.fiaArch38AttentionBaseParams.set_pseShiftS2Size(pseShiftS2);

    tilingData.fiaArch38AttentionBaseParams.set_isKvContinuous(contextKeyParams.isKvContinuous);
    tilingData.fiaArch38AttentionBaseParams.set_isQHasLeftPadding(contextKeyParams.queryPaddingSize != nullptr ? 1 : 0);
    tilingData.fiaArch38AttentionBaseParams.set_isKVHasLeftPadding(contextKeyParams.kvPaddingSize != nullptr ? 1 : 0);

    tilingData.fiaArch38AttentionBaseParams.set_fromFused((contextKeyParams.fromFused == FROM_FUSED_FLAG) ? 1 : 0);
    tilingData.fiaArch38AttentionBaseParams.set_isBSNDOut(contextKeyParams.isBSNDOut);
    tilingData.fiaArch38AttentionBaseParams.set_isSoftMaxLseEnable(contextKeyParams.isSoftMaxLseEnable);

    uint32_t originHeadSize = enableIFAMLA ? tilingData.fiaArch38AttentionBaseParams.get_headSize() :
                                             tilingData.fiaArch38AttentionBaseParams.get_qkHeadSize();
    uint32_t blockElementCnt = BYTE_BLOCK / dataTypeSize;
    if (originHeadSize % blockElementCnt !=
        0) { // Determine if D is aligned with 32B, using fp16 type with 16 elements.
        tilingData.fiaArch38AttentionBaseParams.set_alignedHeadSize(
            ((originHeadSize + blockElementCnt - 1) / blockElementCnt) * blockElementCnt);
        isDNoTail = false;
    } else {
        tilingData.fiaArch38AttentionBaseParams.set_alignedHeadSize(originHeadSize);
    }
}

void FiaArch38PromptTiling::GetEnableDN(FiaArch38FlashAttentionTilingData& tilingData, PFAShapeInfo& queryShapeInfo,
                                        const PFAShapeInfo& valueShapeInfo, std::vector<int64_t>& actualSeqLengths,
                                        std::vector<int64_t>& actualSeqLengthsKV)
{
    // 使能 DN：镜像 kernel 模板条件 !hasAtten && s1BaseSize == 128 && dTemplate == Aligned128，
    // 即 !enableMask && AlignUp(d, 64) == 128 && sOuter == 128。
    (void)valueShapeInfo;
    (void)actualSeqLengths;
    (void)actualSeqLengthsKV;
    constexpr uint32_t dLimitDN = 128;
    constexpr uint32_t sOuterBaseSizeDN = 128;
    constexpr uint32_t sOuterLimitDN = 64;
    // kernel 侧由模板参数推导 useDn：!hasAtten && s1BaseSize == 128 && dTemplate == Aligned128。
    // 此处镜像同一规则，保证 DN 的 tilingKey 位始终与 kernel 路径一致
    // （sOuter == 128 <=> s1BaseSize == 128；不在此判断 IFA，kernel 侧看不到该信息）。
    const uint32_t sOuter = tilingData.fiaArch38AttentionSingleCoreParams.get_singleProcessSOuterSize();
    enableDN = (AlignUp(queryShapeInfo.d, sOuterLimitDN) == dLimitDN) && !enableMask && (sOuter == sOuterBaseSizeDN);

    // arch35 同步遗留的两段 sInner=256 扩展（d==64 / perblock fp8）在 arch38 不可达，已删除：
    // - d==64 不满足 enableDN 的 AlignUp(d, 64) == 128，D64 不会走 DN；
    // - perblock(fp8/hif8) 已被 CheckArch38ScenarioSupported 的 dtype 门禁拒绝。
    // 若未来放开 D64-DN 或 fp8 perblock，需要同时恢复 sInner=256 与对应的 tilingKey/模板。
}

void FiaArch38PromptTiling::SetTilingData(FiaArch38ContextParams& contextKeyParams, PFAShapeInfo& queryShapeInfo,
                                          PFAShapeInfo& queryRopeShapeInfo, PFAShapeInfo& valueShapeInfo,
                                          FiaArch38FlashAttentionTilingData& tilingData)
{
    typeByteNum = BYTE_BLOCK / dataTypeSize;
    outputTypeByteNum = BYTE_BLOCK / outputDataTypeSize;
    softmaxTypeByteNum = BYTE_BLOCK / softmaxDataTypeSize;
    maskTypeByteNum = BYTE_BLOCK / maskElemSize;
    tilingData.fiaArch38AttentionBaseParams.set_typeByteNum(typeByteNum);
    tilingData.fiaArch38AttentionBaseParams.set_maskTypeByteNum(maskTypeByteNum);
    tilingData.fiaArch38AttentionBaseParams.set_softmaxTypeByteNum(softmaxTypeByteNum);
    tilingData.fiaArch38AttentionBaseParams.set_outputTypeByteNum(outputTypeByteNum);

    uint32_t deqScaleTypeFlag = (contextKeyParams.deqScaleType == DT_UINT64) ? 0U : 1U;
    uint32_t deqScale2TypeFlag = (contextKeyParams.deqScale2Type == DT_UINT64) ? 0U : 1U;
    tilingData.fiaArch38AttentionBaseParams.set_deqScaleFlag(deqScaleTypeFlag);
    tilingData.fiaArch38AttentionBaseParams.set_deqScale2Flag(deqScale2TypeFlag);
    tilingData.fiaArch38AttentionBaseParams.set_scaleValue(*contextKeyParams.scaleValue);
    // Perchannel judgment to be adapted, maintain the existing logic firstly.
    tilingData.fiaArch38AttentionBaseParams.set_isQuant2Perchannel(0);
    tilingData.fiaArch38AttentionBaseParams.set_isQuant2BF16(0);
    tilingData.fiaArch38AttentionBaseParams.set_isQuant2FP16(0);
    // IFA flag
    tilingData.fiaArch38AttentionBaseParams.set_isIFA(enableIFA);
    if (enablePostQuant) {
        if (contextKeyParams.scale2Shape->GetStorageShape().GetShapeSize() > 1) {
            tilingData.fiaArch38AttentionBaseParams.set_isQuant2Perchannel(1);
        }
        if (contextKeyParams.quantScale2Type == ge::DT_BF16) {
            tilingData.fiaArch38AttentionBaseParams.set_isQuant2BF16(1);
        }
        if (contextKeyParams.quantScale2Type == ge::DT_FLOAT16 && contextKeyParams.hasKeyAntiquantScale &&
            contextKeyParams.hasValueAntiquantScale) {
            tilingData.fiaArch38AttentionBaseParams.set_isQuant2FP16(1);
        }
    }

    if (enablePA) {
        tilingData.fiaArch38AttentionBaseParams.set_blockSize(*contextKeyParams.blockSize);
    } else {
        tilingData.fiaArch38AttentionBaseParams.set_blockSize(BLOCK_SIZE_BASE);
    }
    tilingData.fiaArch38AttentionBaseParams.set_blockTableDim2(blockTableDim2);
    tilingData.fiaArch38AttentionBaseParams.set_PABlockNumSum(paBlockNumSum);
    uint32_t isLayoutSH = (inputLayout == InputLayout::SH) ? 1U : 0U;
    tilingData.fiaArch38AttentionBaseParams.set_isLayoutSH(isLayoutSH);
    tilingData.fiaArch38AttentionBaseParams.set_dimNumOfseq(queryShapeInfo.b);
    tilingData.fiaArch38AttentionBaseParams.set_headSize(queryShapeInfo.d);
    tilingData.fiaArch38AttentionBaseParams.set_ropeHeadSize(queryRopeShapeInfo.d);
    if (enableIFAMLA) {
        tilingData.fiaArch38AttentionBaseParams.set_qkHeadSize(queryShapeInfo.d + queryRopeShapeInfo.d);
    } else {
        tilingData.fiaArch38AttentionBaseParams.set_qkHeadSize(queryShapeInfo.d);
    }
    tilingData.fiaArch38AttentionBaseParams.set_vHeadSize(valueShapeInfo.d);
    tilingData.fiaArch38AttentionBaseParams.set_seqInnerSize(S2);
    tilingData.fiaArch38AttentionBaseParams.set_seqSize(queryShapeInfo.s);
    tilingData.fiaArch38AttentionBaseParams.set_headNumSize(queryShapeInfo.n);
    tilingData.fiaArch38AttentionBaseParams.set_batchSize(queryShapeInfo.b);

    SetTilingDataAttribute(contextKeyParams, tilingData);
}

void FiaArch38PromptTiling::InferTilingMod(const FiaArch38ContextParams& contextKeyParams,
                                           std::vector<int64_t>& actualSeqLengths,
                                           std::vector<int64_t>& actualSeqLengthsKV, uint32_t actualSeqArrayLen,
                                           uint32_t d)
{
    // Determine whether to use the norm template
    int64_t minActualSeqLengths = INT64_MAX;
    int64_t minActualSeqLengthsKV = INT64_MAX;
    for (uint32_t i = 0; i < actualSeqArrayLen; ++i) {
        minActualSeqLengths = std::min(minActualSeqLengths, actualSeqLengths[i]);
        minActualSeqLengthsKV = std::min(minActualSeqLengthsKV, actualSeqLengthsKV[i]);
    }
    if ((minActualSeqLengths >= MATMUL_NORM_MIN_SEQ) && (minActualSeqLengthsKV >= MATMUL_NORM_MIN_SEQ) &&
        (d == MATMUL_NORM_MIN_HEADSIZE) && (inputType == ge::DT_FLOAT16) &&
        (contextKeyParams.kDataType == ge::DT_FLOAT16) && (contextKeyParams.maskDataType == ge::DT_BOOL) &&
        (outputType == ge::DT_FLOAT16) && (usePseShift == 0) && (inputLayout == InputLayout::BNSD) &&
        (sparseModeVal == SPARSE_MODE_BAND) && (!enablePA)) {
        // Currently, only the matmul norm template is open for the X1 scenario
        enableMatmulNorm = true;
    }
}

void FiaArch38PromptTiling::InferSplitCoreMode()
{
    splitCoreMode = SplitCoreMode::SPLIT_NBS_VECTOR;
}

void FiaArch38PromptTiling::InferConstantization()
{
    isConstantization = true;
}

void FiaArch38PromptTiling::GetMatMulType(matmul_tiling::DataType& mmInputType,
                                          matmul_tiling::DataType& mmOutputType) const
{
    if (inputType == ge::DT_FLOAT16 && innerPrecise == HIGH_PRECISION) {
        mmInputType = matmul_tiling::DataType::DT_FLOAT16;
        mmOutputType = matmul_tiling::DataType::DT_FLOAT;
    } else if (inputType == ge::DT_BF16) {
        mmInputType = matmul_tiling::DataType::DT_BF16;
        mmOutputType = matmul_tiling::DataType::DT_FLOAT;
    } else if (inputType == ge::DT_INT8) {
        mmInputType = matmul_tiling::DataType::DT_INT8;
        mmOutputType = matmul_tiling::DataType::DT_INT32;
    }
}

bool FiaArch38PromptTiling::EnableMTE2BmmPipe(FiaArch38FlashAttentionTilingData& tilingData,
                                              matmul_tiling::MatmulApiTiling& bmm, TCubeTiling& bmmTilingData,
                                              uint32_t sOuterFactor, uint32_t sInnerFactor) const
{
    // When the size is greater than 16, use xiaoe speculative inference.
    if (tilingData.fiaArch38AttentionBaseParams.get_seqSize() > 16) {
        return true;
    }
    uint32_t baseK = 32U;
    uint32_t headSize = tilingData.fiaArch38AttentionBaseParams.get_qkHeadSize();
    if (headSize % baseK != 0) {
        return true;
    }

    uint32_t baseM = std::min(uint32_t(128), sOuterFactor);
    uint32_t baseN = std::min(uint32_t(512), sInnerFactor);
    if (enablePA) {
        baseN = BLOCK_SIZE_BASE;
    }
    int32_t ret = 0;
    ret = bmm.SetFixSplit(baseM, baseN, baseK);
    OP_CHECK_IF(ret != 0, ReportTilingFailure(context_, "bmm SetFixSplit failed"), return false);
    bool res = bmm.GetTiling(bmmTilingData) != -1;
    return res;
}

void FiaArch38PromptTiling::EnableBmmDoubleBuffer(TCubeTiling& bmmTilingData) const
{
    if ((bmmTilingData.get_depthA1() == 1) && (bmmTilingData.get_depthB1() == 1)) {
        bmmTilingData.set_depthA1(2); // 2 : depthA1
        bmmTilingData.set_depthB1(2); // 2 : depthB1
    }
}

bool FiaArch38PromptTiling::FiaArch38CheckBmm1(FiaArch38FlashAttentionTilingData& tilingData,
                                               TCubeTiling& bmm1TilingData, int64_t l1SizeRemain, int64_t l0CSize,
                                               uint32_t sOuterFactor, uint32_t sInnerFactor, bool autoBaseMNK)
{
    if (splitCoreMode == SplitCoreMode::SPLIT_NBS_CUBE) {
        sOuterFactor = sOuterFactor * CV_RATIO;
    }
    matmul_tiling::MatmulApiTiling bmm1(ascendPlatformInfo);

    matmul_tiling::DataType bmm1InputType = matmul_tiling::DataType::DT_FLOAT16;
    matmul_tiling::DataType bmm1OutputType = matmul_tiling::DataType::DT_FLOAT16;
    GetMatMulType(bmm1InputType, bmm1OutputType);
    bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm1InputType, false);
    bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm1InputType, true);
    bmm1.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND_ALIGN, bmm1OutputType);

    int32_t ret = bmm1.SetShape(sOuterFactor, sInnerFactor, tilingData.fiaArch38AttentionBaseParams.get_qkHeadSize());
    OP_CHECK_IF(ret != 0, ReportTilingFailure(context_, "bmm1 SetShape failed"), return false);
    int32_t ratio = tilingData.fiaArch38AttentionBaseParams.get_headNumRatio();
    int32_t strideQ = tilingData.fiaArch38AttentionBaseParams.get_qkHeadSize() *
                      tilingData.fiaArch38AttentionBaseParams.get_headNumSize();
    if (ratio == 0) {
        return false;
    }
    int32_t strideK = strideQ / ratio;
    if ((inputLayout == InputLayout::BSH) || (inputLayout == InputLayout::BSND) || (inputLayout == InputLayout::TND)) {
        if (enableKVAntiquant || (inputLayout == InputLayout::TND && enablePA && paLayoutType == 0)) {
            bmm1.SetOrgShape(tilingData.fiaArch38AttentionBaseParams.get_seqSize(),
                             tilingData.fiaArch38AttentionBaseParams.get_seqInnerSize(), strideQ,
                             tilingData.fiaArch38AttentionBaseParams.get_qkHeadSize());
        } else if (enableIFAMLA || enableIFA) {
            bmm1.SetOrgShape(tilingData.fiaArch38AttentionBaseParams.get_seqSize(),
                             tilingData.fiaArch38AttentionBaseParams.get_seqInnerSize(),
                             tilingData.fiaArch38AttentionBaseParams.get_qkHeadSize(), strideK);
        } else {
            bmm1.SetOrgShape(tilingData.fiaArch38AttentionBaseParams.get_seqSize(),
                             tilingData.fiaArch38AttentionBaseParams.get_seqInnerSize(), strideQ, strideK);
        }
    } else if (inputLayout == InputLayout::BNSD) {
        if (enablePA && paLayoutType == 1) { // The left matrix of PA is BNSD, and the right matrix is BSH.
            bmm1.SetOrgShape(tilingData.fiaArch38AttentionBaseParams.get_seqSize(),
                             tilingData.fiaArch38AttentionBaseParams.get_seqInnerSize(),
                             tilingData.fiaArch38AttentionBaseParams.get_qkHeadSize(), strideK);
        } else {
            bmm1.SetOrgShape(tilingData.fiaArch38AttentionBaseParams.get_seqSize(),
                             tilingData.fiaArch38AttentionBaseParams.get_seqInnerSize(),
                             tilingData.fiaArch38AttentionBaseParams.get_qkHeadSize());
        }
    }

    bmm1.SetBias(false);
    ret = bmm1.SetBufferSpace(l1SizeRemain, l0CSize);
    OP_CHECK_IF(ret != 0, ReportTilingFailure(context_, "bmm1 SetBufferSpace failed"), return false);
    if (!enableIFAMLA && enablePA) {
        ret = bmm1.SetFixSplit(sOuterFactor, BLOCK_SIZE_BASE);
    } else {
        ret = bmm1.SetFixSplit(sOuterFactor, sInnerFactor);
    }
    OP_CHECK_IF(ret != 0, ReportTilingFailure(context_, "bmm1 SetFixSplit failed"), return false);

    ret = bmm1.GetTiling(bmm1TilingData);
    if (autoBaseMNK) {
        uint32_t baseM = std::min(uint32_t(128), sOuterFactor);
        uint32_t baseN = std::min(uint32_t(128), sInnerFactor);
        if (enableMatmulNorm) {
            uint32_t baseK = 128U;
            ret = bmm1.SetFixSplit(baseM, baseN, baseK);
            OP_CHECK_IF(ret != 0, ReportTilingFailure(context_, "bmm1 SetFixSplit failed"), return false);
            ret = bmm1.GetTiling(bmm1TilingData);
        } else {
            uint32_t baseK = 64U;
            if (enablePA) {
                baseN = BLOCK_SIZE_BASE;
            }
            if (ret != 0) {
                ret = bmm1.SetFixSplit(baseM, baseN, baseK);
                OP_CHECK_IF(ret != 0, ReportTilingFailure(context_, "bmm1 SetFixSplit failed"), return false);
                ret = bmm1.GetTiling(bmm1TilingData);
            }
        }
    }
    // Get tiling fail for bmm1.
    OP_CHECK_IF(ret != 0, ReportTilingFailure(context_, "bmm1 GetTiling failed"), return false);

    bmm1TilingData.set_shareMode(0);
    bmm1TilingData.set_shareL1Size(l1SizeRemain);
    bmm1TilingData.set_shareL0CSize(l0CSize);

    bmm1TilingData.set_shareUbSize(0);
    EnableBmmDoubleBuffer(bmm1TilingData); // Open the double buffer for BMM1 calculation, and BMM1's MTE2 can be bound.
    // Open MTE2 Matmul pipeline.
    bool res = EnableMTE2BmmPipe(tilingData, bmm1, bmm1TilingData, sOuterFactor, sInnerFactor);
    // EnableMTE2BmmPipe fail.
    OP_CHECK_IF(res == false, OP_LOGE_WITHOUT_REPORT(contextKeyParamsPtr->opName, "EnableMTE2BmmPipe failed!"),
                return false);

    return true;
}

bool FiaArch38PromptTiling::AdjustCVTilingCVDiff(const FiaArch38ContextParams& contextKeyParams, uint32_t& sOuterFactor,
                                                 uint32_t& sInnerFactor, uint32_t& softmaxSOuterFactor,
                                                 FiaArch38FlashAttentionTilingData& tilingData,
                                                 const PFAShapeInfo& queryShapeInfo)
{
    uint32_t minFactor = SOUTER_FACTOR_DEFAULT;
    uint32_t rectangleFactor = SINNER_FACTOR_DEFAULT;
    softmaxSOuterFactor = SOUTER_FACTOR_DEFAULT;

    auto compileInfoPtr = contextKeyParams.compileInfoPtr;
    bool res = FiaArch38ComputeCVDiffParams(tilingData, compileInfoPtr->l1Size, compileInfoPtr->l0CSize, minFactor,
                                            rectangleFactor);
    OP_CHECK_IF(res == false,
                OP_LOGE_WITHOUT_REPORT(contextKeyParamsPtr->opName, "FiaArch38ComputeCVDiffParams failed!"),
                return false);

    if (tilingData.fiaArch38AttentionBaseParams.get_vHeadSize() <= 128 && !enablePFAMLA) { // 128 for D size
        bool checkDtype =
            contextKeyParams.inputDataType == ge::DT_FLOAT16 || contextKeyParams.inputDataType == ge::DT_BF16;
        bool checkQueryAndValueS = queryShapeInfo.s <= SOUTER_FACTOR_DEFAULT;
        uint32_t sparseMode = tilingData.fiaArch38AttentionBaseParams.get_sparseMode();
        int32_t preTokens = tilingData.fiaArch38AttentionBaseParams.get_preTokens();
        int32_t nextTokens = tilingData.fiaArch38AttentionBaseParams.get_nextTokens();
        if (sparseMode == SPARSE_MODE_NO_MASK) {
            preTokens = (preTokens > 0) ? 0 : preTokens;
        } else if (sparseMode == SPARSE_MODE_BAND) {
            nextTokens = (nextTokens > 0) ? 0 : nextTokens;
        }
        // actual calculation area is smaller than SINNER_FACTOR_DEFAULT in SPARSE_MODE_LEFT_UP mode
        bool checkSparseMode = ((sparseMode == SPARSE_MODE_ALL_MASK) || (sparseMode == SPARSE_MODE_RIGHT_DOWN) ||
                                (((sparseMode == SPARSE_MODE_NO_MASK) || (sparseMode == SPARSE_MODE_BAND)) &&
                                 (preTokens + nextTokens > static_cast<int32_t>(SINNER_FACTOR_DEFAULT))));
        // 小 s（64/256）仅有 int8 输出的模板；fp16->fp16 固定走大 s（128/128）。
        if (checkDtype && checkQueryAndValueS && checkSparseMode && !enablePFARope && !enablePerblockQuant &&
            (contextKeyParams.outputDataType == ge::DT_INT8)) {
            minFactor = SOUTER_FACTOR_SUB;
            rectangleFactor = SINNER_FACTOR_DOUBLE;
            softmaxSOuterFactor = SOUTER_FACTOR_SUB;
        }
    } else if (tilingData.fiaArch38AttentionBaseParams.get_vHeadSize() > ARCH38_SUPPORTED_D_128 && !enableIFAMLA &&
               !enableIFA) {
        // D>128 with atten_mask: enforce s_outer/s_inner = 64/128
        // (UB cannot accommodate 128/128 tiling + D256 features + mask data)
        if (enableMask) {
            minFactor = SOUTER_FACTOR_SUB;
            rectangleFactor = SINNER_FACTOR_DEFAULT;
        } else {
            minFactor = SOUTER_FACTOR_DEFAULT;
            rectangleFactor = SINNER_FACTOR_DEFAULT;
        }
        softmaxSOuterFactor = SOUTER_FACTOR_SUB;
    } else if (enableIFAMLA || (enableIFA && tilingData.fiaArch38AttentionBaseParams.get_vHeadSize() >
                                                 ARCH38_SUPPORTED_D_128)) { // IFA VD > 128
        if (enableIFAMLA || enableMask) {
            minFactor = SOUTER_FACTOR_SUB;
        } else {
            minFactor = SOUTER_FACTOR_DEFAULT;
        }
        rectangleFactor = SINNER_FACTOR_DEFAULT;
        softmaxSOuterFactor = SOUTER_FACTOR_SUB;
    }

    minFactor *= 2; // 2 for cv ratio equals 1:1, previous calculation based on cv ratio equals 1:2

    sOuterFactor = minFactor;
    sInnerFactor = rectangleFactor;

    return true;
}

bool FiaArch38PromptTiling::FiaArch38CheckBmm2(FiaArch38FlashAttentionTilingData& tilingData,
                                               TCubeTiling& bmm2TilingData, int64_t l1SizeRemain, int64_t l0CSize,
                                               uint32_t sOuterFactor, uint32_t sInnerFactor, uint32_t dSplitFactor,
                                               bool autoBaseMNK)
{
    int32_t ret = 0;
    matmul_tiling::MatmulApiTiling bmm2(ascendPlatformInfo);
    matmul_tiling::DataType bmm2InputType = matmul_tiling::DataType::DT_FLOAT16;
    matmul_tiling::DataType bmm2OutputType = matmul_tiling::DataType::DT_FLOAT16;
    GetMatMulType(bmm2InputType, bmm2OutputType);
    bmm2.SetAType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND, bmm2InputType, false);
    bmm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm2InputType, false);
    bmm2.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND_ALIGN, bmm2OutputType);
    ret = bmm2.SetShape(sOuterFactor, tilingData.fiaArch38AttentionBaseParams.get_vHeadSize(), sInnerFactor);

    OP_CHECK_IF(ret != 0, ReportTilingFailure(context_, "bmm2 set SetShape failed"), return false);
    int32_t ratio = tilingData.fiaArch38AttentionBaseParams.get_headNumRatio();
    if (ratio == 0) {
        return false;
    }
    int32_t strideV = tilingData.fiaArch38AttentionBaseParams.get_vHeadSize() *
                      tilingData.fiaArch38AttentionBaseParams.get_headNumSize() / ratio;
    if ((inputLayout == InputLayout::BSH) || (inputLayout == InputLayout::BSND) || (inputLayout == InputLayout::TND)) {
        bmm2.SetOrgShape(tilingData.fiaArch38AttentionBaseParams.get_seqSize(), strideV,
                         tilingData.fiaArch38AttentionBaseParams.get_seqInnerSize());
        if (enableKVAntiquant || (inputLayout == InputLayout::TND && enablePA && paLayoutType == 0)) {
            bmm2.SetOrgShape(tilingData.fiaArch38AttentionBaseParams.get_seqSize(),
                             tilingData.fiaArch38AttentionBaseParams.get_vHeadSize(),
                             tilingData.fiaArch38AttentionBaseParams.get_seqInnerSize());
        }
    } else if (inputLayout == InputLayout::BNSD) {
        if (enablePA && paLayoutType == 1) { // The left matrix of PA is BNSD, and the right matrix is of PA is BSH.
            bmm2.SetOrgShape(tilingData.fiaArch38AttentionBaseParams.get_seqSize(), strideV,
                             tilingData.fiaArch38AttentionBaseParams.get_seqInnerSize());
        } else {
            bmm2.SetOrgShape(tilingData.fiaArch38AttentionBaseParams.get_seqSize(),
                             tilingData.fiaArch38AttentionBaseParams.get_vHeadSize(),
                             tilingData.fiaArch38AttentionBaseParams.get_seqInnerSize());
        }
    }

    bmm2.SetBias(false);
    ret = bmm2.SetBufferSpace(l1SizeRemain, l0CSize);
    OP_CHECK_IF(ret != 0, ReportTilingFailure(context_, "bmm2 set SetBufferSpace failed"), return false);

    if (autoBaseMNK) {
        if (enableMatmulNorm) {
            uint32_t baseM = std::min(uint32_t(128), sOuterFactor);
            uint32_t baseN = std::min(uint32_t(128), tilingData.fiaArch38AttentionBaseParams.get_vHeadSize());
            uint32_t baseK = 128U;
            ret = bmm2.SetFixSplit(baseM, baseN, baseK);
            OP_CHECK_IF(ret != 0, ReportTilingFailure(context_, "bmm2 SetFixSplit failed"), return false);
        }
        ret = bmm2.GetTiling(bmm2TilingData);
    } else {
        if ((isDNoTail) || (splitS2 == 0)) {
            ret = bmm2.SetFixSplit(sOuterFactor, dSplitFactor);
        } else {
            ret = bmm2.SetFixSplit(sOuterFactor, tilingData.fiaArch38AttentionBaseParams.get_alignedHeadSize());
        }
        OP_CHECK_IF(ret != 0, ReportTilingFailure(context_, "bmm2 SetFixSplit failed"), return false);
        ret = bmm2.GetTiling(bmm2TilingData);
    }
    OP_CHECK_IF(ret != 0, ReportTilingFailure(context_, "bmm2 set GetTiling failed"), return false);
    bmm2TilingData.set_shareMode(0);
    bmm2TilingData.set_shareL1Size(l1SizeRemain);
    bmm2TilingData.set_shareL0CSize(l0CSize);
    OP_CHECK_IF(ret != 0, ReportTilingFailure(context_, "bmm2 set shareL0CSize failed"), return false);
    bmm2TilingData.set_shareUbSize(0);
    return true;
}

bool FiaArch38PromptTiling::FiaArch38ComputeCVDiffParams(FiaArch38FlashAttentionTilingData& tilingData, int64_t l1Size,
                                                         int64_t l0CSize, uint32_t& sOuterFactor,
                                                         uint32_t& sInnerFactor)
{
    constexpr uint32_t dSplitFactorBmm2 = 128U;
    int32_t l1SizeRemain = l1Size;
    bool res = FiaArch38CheckBmm1(tilingData, tilingData.bmm1TilingDataRect, l1SizeRemain, l0CSize, sOuterFactor,
                                  sInnerFactor, true);
    OP_CHECK_IF(!res, OP_LOGE_WITHOUT_REPORT(contextKeyParamsPtr->opName, "FiaArch38CheckBmm1 failed!"), return false);

    res = FiaArch38CheckBmm2(tilingData, tilingData.bmm2TilingDataRect, l1SizeRemain, l0CSize, sOuterFactor,
                             sInnerFactor, dSplitFactorBmm2, true);
    OP_CHECK_IF(!res, OP_LOGE_WITHOUT_REPORT(contextKeyParamsPtr->opName, "FiaArch38CheckBmm2 failed!"), return false);

    return true;
}

void FiaArch38PromptTiling::GetPreNextTokensLeftUp(FiaArch38FlashAttentionTilingData& tilingData,
                                                   int64_t actualSeqLength, int64_t actualSeqLengthKV,
                                                   int64_t& preTokensLeftUp, int64_t& nextTokensLeftUp) const
{
    FiaArch38AttentionBaseParams* baseParams = &tilingData.fiaArch38AttentionBaseParams;
    if (baseParams->get_sparseMode() == SPARSE_MODE_RIGHT_DOWN) {
        preTokensLeftUp = SPARSE_MODE_INT_MAX;
        if (enableIFAMLA) {
            if (inputLayout == InputLayout::BSND || inputLayout == InputLayout::BSH ||
                inputLayout == InputLayout::TND) {
                nextTokensLeftUp = actualSeqLengthKV * gSize - actualSeqLength;
            } else { // BNSD场景下分核不做优化
                nextTokensLeftUp = SPARSE_MODE_INT_MAX;
            }
        } else {
            nextTokensLeftUp = actualSeqLengthKV - actualSeqLength;
        }
    } else if (baseParams->get_sparseMode() == SPARSE_MODE_BAND) {
        preTokensLeftUp = baseParams->get_preTokens() - actualSeqLengthKV + actualSeqLength;
        nextTokensLeftUp = baseParams->get_nextTokens() + actualSeqLengthKV - actualSeqLength;
    } else {
        preTokensLeftUp = baseParams->get_preTokens();
        nextTokensLeftUp = baseParams->get_nextTokens();
    }
}

void FiaArch38PromptTiling::UpdateTilingKeyMatmulCfg(uint64_t& tilingKey) const
{
    constexpr uint64_t tilingKeySplitCoreNBSVectorValue = 0;
    constexpr uint64_t tilingKeySplitCoreNBSCubeValue = static_cast<uint64_t>(1e6); // Position 6

    if (splitCoreMode == SplitCoreMode::SPLIT_NBS_VECTOR) {
        tilingKey += tilingKeySplitCoreNBSVectorValue;
    } else if (splitCoreMode == SplitCoreMode::SPLIT_NBS_CUBE) {
        tilingKey += tilingKeySplitCoreNBSCubeValue;
    } else {
        tilingKey += tilingKeySplitCoreNBSVectorValue;
    }
}

void FiaArch38PromptTiling::UpdateTilingKeyMaskCfg(FiaArch38FlashAttentionTilingData& tilingData,
                                                   uint64_t& tilingKey) const
{
    constexpr uint64_t tilingKeyDisableMask = 0;
    constexpr uint64_t tilingKeyEnableMaskNoBand = static_cast<uint64_t>(1e8); // Position 8
    constexpr uint64_t tilingKeyEnableMaskBand = static_cast<uint64_t>(2e8);

    if (tilingData.fiaArch38AttentionBaseParams.get_useMask() == 0U) {
        tilingKey += tilingKeyDisableMask;
    } else if (tilingData.fiaArch38AttentionBaseParams.get_useMask() == 1U &&
               tilingData.fiaArch38AttentionBaseParams.get_sparseMode() == SPARSE_MODE_BAND) {
        tilingKey += tilingKeyEnableMaskBand;
    } else {
        tilingKey += tilingKeyEnableMaskNoBand;
    }
}

void FiaArch38PromptTiling::UpdateTilingKeyPseCfg(uint64_t& tilingKey) const
{
    constexpr uint64_t tilingKeyDisablePse = 0;
    constexpr uint64_t tilingKeyEnablePse = static_cast<uint64_t>(1e9);          // Position 9
    constexpr uint64_t tilingKeyEnableAlibiPse = static_cast<uint64_t>(2e9);     // Position 9
    constexpr uint64_t tilingKeyEnableAlibiPseSqrt = static_cast<uint64_t>(3e9); // Position 9

    if (usePseShift == 0U) {
        tilingKey += tilingKeyDisablePse;
    } else if (usePseShift == 1U) {
        tilingKey += tilingKeyEnablePse;
    } else if (usePseShift == 2U) {
        tilingKey += tilingKeyEnableAlibiPse;
    } else if (usePseShift == 3U) {
        tilingKey += tilingKeyEnableAlibiPseSqrt;
    }
}

void FiaArch38PromptTiling::UpdateTilingKeyDSizeConst(FiaArch38FlashAttentionTilingData& tilingData,
                                                      uint64_t& tilingKey) const
{
    constexpr uint64_t tilingKeyDSizeConstValue64 = static_cast<uint64_t>(1e12); // Position 12
    constexpr uint64_t tilingKeyDSizeConstValue128 = static_cast<uint64_t>(2e12);
    constexpr uint64_t tilingKeyDSizeConstValue256 = static_cast<uint64_t>(3e12);
    constexpr uint64_t tilingKeyDSizeConstValue512 = static_cast<uint64_t>(4e12);
    constexpr uint64_t tilingKeyDSizeConstValue576 = static_cast<uint64_t>(5e12); // for IFA MLA
    uint32_t dSize = tilingData.fiaArch38AttentionBaseParams.get_qkHeadSize();
    if (dSize <= 64) { // 64: headsize is 64
        tilingKey += tilingKeyDSizeConstValue64;
    } else if (dSize <= 128) { // 128: headsize 128
        tilingKey += tilingKeyDSizeConstValue128;
    } else if (dSize <= 256) { // 256: headsize 256
        tilingKey += tilingKeyDSizeConstValue256;
    } else if (dSize <= 512) { // 512: headsize 512
        tilingKey += tilingKeyDSizeConstValue512;
    } else if (dSize <= 576) { // 576: headsize 576
        tilingKey += tilingKeyDSizeConstValue576;
    }
}

void FiaArch38PromptTiling::UpdateTilingKeySInnerConst(FiaArch38FlashAttentionTilingData& tilingData,
                                                       uint64_t& tilingKey) const
{
    constexpr uint64_t tilingKeySInnerConstValue128 = static_cast<uint64_t>(1e13); // Position 13
    constexpr uint64_t tilingKeySInnerConstValue256 = static_cast<uint64_t>(2e13);

    if (tilingData.fiaArch38AttentionSingleCoreParams.get_singleProcessSInnerSize() == 128) { // 128: sinner is 128
        tilingKey += tilingKeySInnerConstValue128;
    } else if (tilingData.fiaArch38AttentionSingleCoreParams.get_singleProcessSInnerSize() ==
               256) { // 256: sinner is 256
        tilingKey += tilingKeySInnerConstValue256;
    }
}

void FiaArch38PromptTiling::UpdateTilingKeySOuterConst(FiaArch38FlashAttentionTilingData& tilingData,
                                                       uint64_t& tilingKey)
{
    constexpr uint64_t tilingKeySOuterConstValue64 = static_cast<uint64_t>(1e14); // Position 14
    constexpr uint64_t tilingKeySOuterConstValue96 = static_cast<uint64_t>(2e14);
    constexpr uint64_t tilingKeySOuterConstValue128 = static_cast<uint64_t>(3e14);

    if (!isConstantization) {
        return;
    }
    if (tilingData.fiaArch38AttentionSingleCoreParams.get_singleProcessSOuterSize() == 64) { // 64:souter32
        tilingKey += tilingKeySOuterConstValue64;
    } else if (tilingData.fiaArch38AttentionSingleCoreParams.get_singleProcessSOuterSize() == 96) { // 96:souter48*2
        tilingKey += tilingKeySOuterConstValue96;
    } else if (tilingData.fiaArch38AttentionSingleCoreParams.get_singleProcessSOuterSize() == 128) { // 128:souter64*2
        tilingKey += tilingKeySOuterConstValue128;
    }
}

void FiaArch38PromptTiling::UpdateTilingKeyValueDSizeConst(FiaArch38FlashAttentionTilingData& tilingData,
                                                           uint64_t& tilingKey) const
{
    constexpr uint64_t tilingKeyDSizeConstValue64 = static_cast<uint64_t>(1e15); // Position 15
    constexpr uint64_t tilingKeyDSizeConstValue128 = static_cast<uint64_t>(2e15);
    constexpr uint64_t tilingKeyDSizeConstValue256 = static_cast<uint64_t>(3e15);
    constexpr uint64_t tilingKeyDSizeConstValue512 = static_cast<uint64_t>(4e15);
    uint32_t dSize = tilingData.fiaArch38AttentionBaseParams.get_vHeadSize();
    if (dSize <= 64) { // 64: headsize is 64
        tilingKey += tilingKeyDSizeConstValue64;
    } else if (dSize <= 128) { // 128: headsize 128
        tilingKey += tilingKeyDSizeConstValue128;
    } else if (dSize <= 256) { // 256: headsize 256
        tilingKey += tilingKeyDSizeConstValue256;
    } else if (dSize <= 512) { // 512: headsize 512
        tilingKey += tilingKeyDSizeConstValue512;
    }
}

int64_t FiaArch38PromptTiling::GetSInnerBlockNums(int64_t sInnerIndexStart, int64_t sInnerIndexEnd,
                                                  int64_t innerBlockNums) const
{
    int64_t sInnerBlockNums = 0;

    if (sInnerIndexEnd < 0) {
        sInnerBlockNums = 0;
    } else if (sInnerIndexEnd < innerBlockNums) {
        sInnerBlockNums = (sInnerIndexStart < 0) ? (sInnerIndexEnd + 1) : (sInnerIndexEnd - sInnerIndexStart + 1);
    } else {
        sInnerBlockNums = (sInnerIndexStart < 0) ?
                              innerBlockNums :
                              (sInnerIndexStart < innerBlockNums ? innerBlockNums - sInnerIndexStart : 0);
    }

    return sInnerBlockNums;
}

int64_t FiaArch38PromptTiling::GetCutBlockNums(int64_t blockSeqLengthKV, int64_t blockSeqLength, int64_t sInner,
                                               int64_t sOuter, int64_t token) const
{
    int64_t blockNums = 0;
    int64_t blockToken = 0;
    int64_t outDivIn = 0;
    if (sInner != 0) {
        blockToken = token > 0 ? ((token + sInner - 1) / sInner * sInner) : (token / sInner * sInner);
        outDivIn = sOuter > sInner ? sOuter / sInner : 1;
    }
    int64_t InDivOut = 0;
    if (sOuter != 0) {
        InDivOut = sInner > sOuter ? sInner / sOuter : 1;
    }
    int64_t tolerance = 0;
    int64_t smallSize = 0;
    if (outDivIn >= 1) {
        tolerance = outDivIn;
        smallSize = sInner;
    } else {
        tolerance = InDivOut;
        smallSize = sOuter;
    }
    int64_t innerCutBlockNums = (blockSeqLengthKV - blockToken) / smallSize - tolerance;
    int64_t innerCutBlockLeftNums = -blockToken / smallSize - tolerance;
    int64_t innerCutBlockDownNums = (blockSeqLengthKV - blockSeqLength - blockToken) / smallSize - tolerance;
    blockNums += (innerCutBlockNums > 0) ?
                     (innerCutBlockNums % tolerance + innerCutBlockNums) * (innerCutBlockNums / tolerance + 1) / 2 :
                     0; // 2: The denominator of the arithmetic sequence summation formula
    blockNums -= (innerCutBlockLeftNums > 0) ? (innerCutBlockLeftNums % tolerance + innerCutBlockLeftNums) *
                                                   (innerCutBlockLeftNums / tolerance + 1) / 2 :
                                               0; // 2: The denominator of the arithmetic sequence summation formula
    blockNums -= (innerCutBlockDownNums > 0) ? (innerCutBlockDownNums % tolerance + innerCutBlockDownNums) *
                                                   (innerCutBlockDownNums / tolerance + 1) / 2 :
                                               0; // 2: The denominator of the arithmetic sequence summation formula
    return blockNums;
}

// 函数内部不处理prefix逻辑，prefix场景下入参需自行传入actualSeqLengthKV + prefix
void FiaArch38PromptTiling::FixParamWithRowInvalid(int64_t& actualSeqLength, int64_t actualSeqLengthKV,
                                                   int64_t& preTokensLeftUp, int64_t& nextTokensLeftUp) const
{
    // 若出现行无效，需要重新计算nexttokens，pretokens，actualseqlen，以便正确计算分核核数
    int64_t nextTokensError = (nextTokensLeftUp < 0) ? -nextTokensLeftUp : 0;
    int64_t preTokensError = (actualSeqLength > actualSeqLengthKV + preTokensLeftUp) ?
                                 (actualSeqLength - actualSeqLengthKV - preTokensLeftUp) :
                                 0;

    // 若出现上方行无效，需要重新计算nexttokens，pretokens，actualseqlen
    nextTokensLeftUp += nextTokensError;
    preTokensLeftUp -= nextTokensError;
    actualSeqLength -= nextTokensError;

    // 若出现下方行无效，需要重新计算actualseqlen
    actualSeqLength -= preTokensError;
}

int64_t FiaArch38PromptTiling::GetCalcBlockNumsOneHead(int64_t actualSeqLength, int64_t actualSeqLengthKV,
                                                       uint32_t sOuterSize, uint32_t sInnerSize,
                                                       int64_t preTokensLeftUp, int64_t nextTokensLeftUp,
                                                       bool isAttenMaskUsed) const
{
    if (!isAttenMaskUsed) {
        int64_t outerBlockNums = (actualSeqLength + sOuterSize - 1) / sOuterSize;
        int64_t innerBlockNums = (actualSeqLengthKV + sInnerSize - 1) / sInnerSize;
        int64_t toCalcBlockNums = innerBlockNums * outerBlockNums;
        return toCalcBlockNums;
    } else {
        int64_t innerBlockNums =
            (actualSeqLengthKV + static_cast<int64_t>(sInnerSize) - 1) / static_cast<int64_t>(sInnerSize);
        int64_t blockSeqLengthKV = innerBlockNums * static_cast<int64_t>(sInnerSize);
        int64_t outerBlockNums =
            (actualSeqLength + static_cast<int64_t>(sOuterSize) - 1) / static_cast<int64_t>(sOuterSize);
        int64_t blockSeqLength = outerBlockNums * static_cast<int64_t>(sOuterSize);
        int64_t toCalcBlockNums = innerBlockNums * outerBlockNums;
        // Must meet this condition : pretoken + nexttoken > 0
        toCalcBlockNums -= GetCutBlockNums(blockSeqLengthKV, blockSeqLength, static_cast<int64_t>(sInnerSize),
                                           static_cast<int64_t>(sOuterSize), nextTokensLeftUp);
        toCalcBlockNums -=
            GetCutBlockNums(blockSeqLengthKV, blockSeqLength, static_cast<int64_t>(sInnerSize),
                            static_cast<int64_t>(sOuterSize), blockSeqLengthKV - blockSeqLength + preTokensLeftUp);
        return toCalcBlockNums;
    }
}

void FiaArch38PromptTiling::ComputeSplitNBSeq(FiaArch38FlashAttentionTilingData& tilingData, uint32_t batchSize,
                                              const size_t tilingElementArrayLen,
                                              std::vector<int64_t>& actualSeqLengths,
                                              std::vector<int64_t>& actualSeqLengthsKV, uint32_t sOuterSize,
                                              uint32_t sInnerSize, double coreWightTarget, uint32_t& curCore)
{
    FiaArch38AttentionBaseParams* baseParams = &tilingData.fiaArch38AttentionBaseParams;
    std::vector<uint32_t> coreSposEnd(tilingElementArrayLen, 0U);
    std::vector<uint32_t> coreSposStart(tilingElementArrayLen, 0U);
    std::vector<uint32_t> coreSidEnd(tilingElementArrayLen, 0U);
    std::vector<uint32_t> coreSidStart(tilingElementArrayLen, 0U);
    std::vector<uint32_t> coreNidEnd(tilingElementArrayLen, 0U);
    std::vector<uint32_t> coreNidStart(tilingElementArrayLen, 0U);
    std::vector<int64_t> sparseStartIdx(tilingElementArrayLen, 0L);
    std::vector<uint32_t> bnStartIdx(tilingElementArrayLen, 0U);
    std::vector<int64_t> gS1StartIdx(tilingElementArrayLen, 0L);
    // Temporary algorithm to be optimized
    int64_t curWight = 0;
    curCore = 0;
    uint32_t tmpCoreNidEnd = 0; // actual seq为0时不分配核
    uint32_t tmpCoreSidEnd = 0;
    uint32_t tmpCoreSposEnd = 0;
    for (uint32_t sIdx = 0; sIdx < batchSize; sIdx++) {
        for (uint32_t headNum = 0; headNum < baseParams->get_headNumSize(); headNum++) {
            // 针对行无效情况修正actualseqlen
            int64_t preTokensLeftUp = 0;
            int64_t nextTokensLeftUp = 0;
            GetPreNextTokensLeftUp(tilingData, actualSeqLengths[sIdx], actualSeqLengthsKV[sIdx] + actualSharedPrefixLen,
                                   preTokensLeftUp, nextTokensLeftUp);
            int64_t actualSeqLength = actualSeqLengths[sIdx];
            int64_t actualSeqLengthKV = actualSeqLengthsKV[sIdx];
            FixParamWithRowInvalid(actualSeqLength, actualSeqLengthKV + actualSharedPrefixLen, preTokensLeftUp,
                                   nextTokensLeftUp);

            int64_t outerBlockNums = (actualSeqLength + sOuterSize - 1) / sOuterSize;
            int64_t innerBlockNums = (actualSeqLengthKV + sInnerSize - 1) / sInnerSize +
                                     (actualSharedPrefixLen + sInnerSize - 1) / sInnerSize;
            for (uint32_t sOuterIndex = 0; sOuterIndex < outerBlockNums; sOuterIndex++) {
                int64_t dif = static_cast<int64_t>(coreWightTarget * double(curCore + 1)) - curWight;
                int64_t sInnerIndexStart =
                    -(preTokensLeftUp > 0 ?
                          (preTokensLeftUp + static_cast<int64_t>(sInnerSize) - 1) / static_cast<int64_t>(sInnerSize) :
                          preTokensLeftUp / static_cast<int64_t>(sInnerSize));
                int64_t sInnerIndexEnd =
                    nextTokensLeftUp > 0 ?
                        (nextTokensLeftUp + static_cast<int64_t>(sInnerSize) - 1) / static_cast<int64_t>(sInnerSize) :
                        nextTokensLeftUp / static_cast<int64_t>(sInnerSize);

                // The number of innerBlock blocks in each outBlock row represents the calculation amount of each
                // outBlock row.
                int64_t sInnerBlockNums = GetSInnerBlockNums(sInnerIndexStart, sInnerIndexEnd, innerBlockNums);
                if (sInnerBlockNums - dif > dif && !(tmpCoreNidEnd == 0 && tmpCoreSidEnd == 0 && tmpCoreSposEnd == 0)) {
                    coreNidEnd[curCore] = tmpCoreNidEnd;
                    coreSidEnd[curCore] = tmpCoreSidEnd;
                    coreSposEnd[curCore] = tmpCoreSposEnd;
                    curCore += 1;
                    coreNidStart[curCore] = headNum;
                    coreSidStart[curCore] = sIdx;
                    coreSposStart[curCore] = sOuterIndex;
                    bnStartIdx[curCore] = sIdx * baseParams->get_headNumSize() + headNum;
                    gS1StartIdx[curCore] = sOuterIndex;
                }
                tmpCoreNidEnd = headNum + 1;
                tmpCoreSidEnd = sIdx + 1;
                tmpCoreSposEnd = sOuterIndex + 1;

                curWight += sInnerBlockNums;
                preTokensLeftUp -= sOuterSize;
                nextTokensLeftUp += sOuterSize;
            }
        }
    }
    coreNidEnd[curCore] = tmpCoreNidEnd;
    coreSidEnd[curCore] = tmpCoreSidEnd;
    coreSposEnd[curCore] = tmpCoreSposEnd;
    bnStartIdx[curCore + 1] = batchSize * baseParams->get_headNumSize();
    gS1StartIdx[curCore + 1] = tmpCoreSposEnd;

    FiaArch38AttentionSeqParams* seqParams = &tilingData.fiaArch38AttentionSeqParams;
    seqParams->set_CoreHeadNumTail(coreNidStart.data());
    seqParams->set_actualS1(coreNidEnd.data());
    seqParams->set_actualCoreNums(coreSidStart.data());
    seqParams->set_singleCoreHeadNumSize(coreSidEnd.data());
    seqParams->set_coreSeqPosStart(coreSposStart.data());
    seqParams->set_coreSeqPosEnd(coreSposEnd.data());

    faTilingAdapter.multiCoreParamsRegbase.set_bnStartIdx(bnStartIdx.data());
    faTilingAdapter.multiCoreParamsRegbase.set_sparseStartIdx(gS1StartIdx.data());
}

void FiaArch38PromptTiling::SetMultiCoreParamsRegbase(int64_t totalSize, int64_t actualUsedCoreNum)
{
    faTilingAdapter.multiCoreParamsRegbase.set_coreNum(static_cast<int32_t>(actualUsedCoreNum));
    faTilingAdapter.multiCoreParamsRegbase.set_totalSize(totalSize);
    faTilingAdapter.multiCoreParamsRegbase.set_splitFactorSize(CeilDivision(totalSize, actualUsedCoreNum));
    faTilingAdapter.multiCoreParamsRegbase.set_splitFactorTailSize(
        CalcTailSize(totalSize, faTilingAdapter.multiCoreParamsRegbase.get_splitFactorSize()));
}

void FiaArch38PromptTiling::FiaArch38SplitNBSeq(FiaArch38FlashAttentionTilingData& tilingData,
                                                std::vector<int64_t>& actualSeqLengths,
                                                std::vector<int64_t>& actualSeqLengthsKV, bool isAttenMaskUsed)
{
    FiaArch38AttentionBaseParams* baseParams = &tilingData.fiaArch38AttentionBaseParams;
    FiaArch38AttentionSingleCoreParams* singleCoreParams = &tilingData.fiaArch38AttentionSingleCoreParams;
    uint32_t curCoreNum = coreNum;
    uint32_t batchSize = baseParams->get_dimNumOfseq();
    uint32_t sOuterSize = singleCoreParams->get_singleProcessSOuterSize();
    uint32_t sInnerSize = singleCoreParams->get_singleProcessSInnerSize();

    if (splitCoreMode == SplitCoreMode::SPLIT_NBS_CUBE) { // From the perspective of cube
        sOuterSize = sOuterSize * CV_RATIO;
        curCoreNum = curCoreNum / CV_RATIO;
    }

    int64_t totalBlockNumsOneHead = 0; // The calculation amount of all sequences for a single head

    std::vector<uint32_t> sInnerLoopTimes(batchSize);
    uint32_t multiSmaxsInnerLoopTimes = 0U;
    for (uint32_t sIdx = 0; sIdx < batchSize; sIdx++) {
        int64_t actualSeqLengthsTmp = actualSeqLengths[sIdx]; // 用于存放减去行无效后，真实的actseqlen
        int64_t preTokensLeftUp = 0;
        int64_t nextTokensLeftUp = 0;
        GetPreNextTokensLeftUp(tilingData, actualSeqLengths[sIdx], actualSeqLengthsKV[sIdx] + actualSharedPrefixLen,
                               preTokensLeftUp, nextTokensLeftUp);

        // 计算各sparse mode情况下，减去行无效后真实的actseqlen
        FixParamWithRowInvalid(actualSeqLengthsTmp, actualSeqLengthsKV[sIdx], preTokensLeftUp, nextTokensLeftUp);

        // sinner方向块数，prefix和origin是分开切的。
        sInnerLoopTimes[sIdx] = (actualSeqLengthsKV[sIdx] + sInnerSize - 1) / sInnerSize +
                                (actualSharedPrefixLen + sInnerSize - 1) / sInnerSize;
        multiSmaxsInnerLoopTimes = std::max(multiSmaxsInnerLoopTimes, sInnerLoopTimes[sIdx]);

        totalBlockNumsOneHead +=
            GetCalcBlockNumsOneHead(actualSeqLengthsTmp, actualSeqLengthsKV[sIdx], sOuterSize, sInnerSize,
                                    preTokensLeftUp, nextTokensLeftUp, isAttenMaskUsed);
    }
    singleCoreParams->set_multiSmaxsInnerLoopTimes(multiSmaxsInnerLoopTimes);

    // Amount of computation per core
    double coreWightTarget = (double(totalBlockNumsOneHead * baseParams->get_headNumSize()) / double(curCoreNum));

    int64_t s1OuterSize = (baseParams->get_seqSize() + sOuterSize - 1) / sOuterSize;
    faTilingAdapter.multiCoreParamsRegbase.set_s1OuterSize(s1OuterSize);

    // The tiling structure element needs to have a length greater than or equal to the length specified
    // by TILING_DATA_FIELD_DEF_ARR. If the tiling structure definition specifies a length of 64,
    // the vector definition needs to compare its size with coreNum and take the larger value
    const size_t tilingElementArrayLen =
        (static_cast<size_t>(curCoreNum) > 64UL) ? static_cast<size_t>(curCoreNum) : 64UL;
    uint32_t curIndx = 0;
    ComputeSplitNBSeq(tilingData, batchSize, tilingElementArrayLen, actualSeqLengths, actualSeqLengthsKV, sOuterSize,
                      sInnerSize, coreWightTarget, curIndx);

    uint32_t actualCoreNums = (splitCoreMode == SplitCoreMode::SPLIT_NBS_CUBE) ? (curIndx + 1) * CV_RATIO : curIndx + 1;
    singleCoreParams->set_actualCoreNums(actualCoreNums);
    int64_t sinnerBlocknum = (baseParams->get_seqInnerSize() + sInnerSize - 1) / sInnerSize;
    SetMultiCoreParamsRegbase((totalBlockNumsOneHead / sinnerBlocknum) * baseParams->get_headNumSize(),
                              static_cast<int64_t>((curIndx + 1)));
}

void FiaArch38PromptTiling::FiaArch38InitSoftmaxLseOutputSplit(int64_t totalSize,
                                                               FiaArch38FlashAttentionTilingData& tilingData) const
{
    FiaArch38AttentionInitOutputParams* initParams = &tilingData.fiaArch38AttentionInitOutputParams;
    initParams->set_totalSoftMaxLseOutputSize(totalSize);
}

void FiaArch38PromptTiling::UpdateTilingKeyFlag(const FiaArch38ContextParams& contextKeyParams,
                                                uint64_t& tilingKey) const
{
    uint64_t binaryFlag = 0;
    auto queryDtype = contextKeyParams.inputDataType;
    auto kvDtype = contextKeyParams.kDataType;

    if ((queryDtype == ge::DT_FLOAT16) && (kvDtype == ge::DT_INT8)) {
        // 4bit flag bit, the leftmost side indicates whether to perform inverse quantization operation,
        // with a corresponding value of 2**3 = 8, and the remaining 3bit is reserved
        binaryFlag += 8U;
    }
    // If inverse quantization is performed, tilingKey should increase by 8*1e11.
    tilingKey += (binaryFlag * static_cast<uint64_t>(1e11));
    return;
}

bool FiaArch38PromptTiling::TilingGetTilingKeyAttentionAscendC(uint64_t& tilingKey,
                                                               FiaArch38ContextParams& contextKeyParams,
                                                               FiaArch38FlashAttentionTilingData& tilingData)
{
    auto inputDataType = contextKeyParams.inputDataType; // input q
    auto attenMaskElemType = contextKeyParams.maskDataType;
    auto outputDataType = contextKeyParams.outputDataType; // output tensor
    tilingData.fiaArch38AttentionBaseParams.set_attenMaskElemType(attenMaskElemType);

    tilingKey = 1012U; // 1012：CV diff, +1000; new_tiling, +10; not distinguishing between tail and total, +2.
    // fp16 high precision mode, regarded as a type 600.
    tilingKey += ((inputDataType == ge::DT_FLOAT16) && (innerPrecise == HIGH_PRECISION)) ? 600U : 0;
    tilingKey += (inputDataType == ge::DT_BF16) ? 100U : 0;          // 100: bf16
    tilingKey += (inputDataType == ge::DT_INT8) ? 200U : 0;          // 200: int8
    tilingKey += (inputDataType == ge::DT_HIFLOAT8) ? 400U : 0;      // 400: hifloat8
    tilingKey += (inputDataType == ge::DT_FLOAT8_E4M3FN) ? 900U : 0; // 900: float8_e4m3
    tilingKey += (enablePerblockQuant == true) ? 1U : 0;             // 0: FP8 perblock dequant
    // When the output dtype is bf16, tilingKey should increase by 10000.
    tilingKey += (outputDataType == ge::DT_BF16) ? static_cast<uint64_t>(1e4) : 0;
    tilingKey += (outputDataType == ge::DT_INT8) ? static_cast<uint64_t>(2e4) :
                                                   0; // 20000: The situation of outputDataType == ge::DT_INT8
    tilingKey += (outputDataType == ge::DT_HIFLOAT8) ? static_cast<uint64_t>(4e4) :
                                                       0; // 40000: The situation of outputDataType == ge::DT_HIFLOAT8
    tilingKey += (outputDataType == ge::DT_FLOAT8_E4M3FN) ?
                     static_cast<uint64_t>(6e4) :
                     0; // 60000: The situation of outputDataType == ge::DT_FLOAT8_E4M3FN
    // When the inputLayout is TND, plus 2e5.
    // When the inputLayout is BSH or BSND, plus 1e5.
    uint64_t tilingKeyLayout = (inputLayout == InputLayout::TND) ? static_cast<uint64_t>(2e5) :
                               ((inputLayout == InputLayout::BSH) || (inputLayout == InputLayout::BSND)) ?
                                                                   static_cast<uint64_t>(1e5) :
                                                                   0U;
    tilingKey += tilingKeyLayout;
    UpdateTilingKeyMatmulCfg(tilingKey);
    UpdateTilingKeyDSizeConst(tilingData, tilingKey);
    UpdateTilingKeyValueDSizeConst(tilingData, tilingKey);
    UpdateTilingKeySInnerConst(tilingData, tilingKey);
    UpdateTilingKeySOuterConst(tilingData, tilingKey);
    UpdateTilingKeyMaskCfg(tilingData, tilingKey);
    UpdateTilingKeyPseCfg(tilingKey);
    UpdateTilingKeyFlag(contextKeyParams, tilingKey);

    if (enablePA) {
        tilingKey += static_cast<uint64_t>(2e7); // 2e7: the situation of PA
    }

    if (enableIFAMLA) {
        tilingKey += static_cast<uint64_t>(3e7); // 3e7: the situation of IFA MLA
    }

    if (enableDN) {
        tilingKey += static_cast<uint64_t>(4e7); // 4e7: the situation of DN
    }

    if (isKVHasPrefix) {
        tilingKey += static_cast<uint64_t>(1e8); // 1e8: the situation of prefix
    }

    if (enableFlashDecode) {
        tilingKey += static_cast<uint64_t>(1e10); // 1e10: the situation of FD
    }

    return true;
};

size_t FiaArch38PromptTiling::GetPFAWorkSpaceSize(FiaArch38FlashAttentionTilingData& tilingData)
{
    constexpr uint32_t S1_BASIC_BLOCK_FACTOR = 2;              // S1 基本块 = SOuter × 2
    constexpr uint32_t WORKSPACE_BMM_FACTOR = 3;               // bmm2/vec2 在 workspace 中占 3 份
    constexpr uint32_t PA_WORKSPACE_ALIGN_FACTOR = 2 * 2 * 64; // PA 场景额外 workspace 对齐（2 bmm × 2 × 64B）

    size_t sysWorkspaceSize = 0;
    size_t curWorkspaceSize = 0;
    const uint64_t defaultSysWorkspaceSize910B = 16U * 1024U * 1024U;
    sysWorkspaceSize = defaultSysWorkspaceSize910B; // sys workspace size default value

    uint32_t kvSplitPart = faTilingAdapter.inputParamsRegbase.get_kvSplitPart();
    auto batchSize = tilingData.fiaArch38AttentionBaseParams.get_batchSize();
    auto headNumSize = tilingData.fiaArch38AttentionBaseParams.get_headNumSize();
    auto vHeadSize = tilingData.fiaArch38AttentionBaseParams.get_vHeadSize();
    size_t accumOutSize = batchSize * gSize * headNumSize * kvSplitPart * vHeadSize * sizeof(float);
    size_t logSumExpSize = batchSize * gSize * headNumSize * kvSplitPart * BYTE_BLOCK * 2;

    int64_t bmm2Bytes = 0;
    int64_t vec2Bytes = 0;
    uint32_t dSize = tilingData.fiaArch38AttentionBaseParams.get_vHeadSize();
    uint32_t dVBasicBlock = 0;
    if (dSize <= FiaPolicy::DV_TEMPLATE_64) {
        dVBasicBlock = FiaPolicy::DV_TEMPLATE_64;
    } else if (dSize <= FiaPolicy::DV_TEMPLATE_128) {
        dVBasicBlock = FiaPolicy::DV_TEMPLATE_128;
    } else if (dSize <= FiaPolicy::DV_TEMPLATE_256) {
        dVBasicBlock = FiaPolicy::DV_TEMPLATE_256;
    } else if (dSize <= FiaPolicy::DV_TEMPLATE_512) {
        dVBasicBlock = FiaPolicy::DV_TEMPLATE_512;
    }
    int64_t bmm2ResBlockSize = dVBasicBlock;
    if (dVBasicBlock > FiaPolicy::DV_TEMPLATE_256) {
        bmm2ResBlockSize = FiaPolicy::DV_TEMPLATE_512;
    }
    uint32_t s1BasicBlock =
        tilingData.fiaArch38AttentionSingleCoreParams.get_singleProcessSOuterSize() * S1_BASIC_BLOCK_FACTOR;
    uint32_t sOuter = tilingData.fiaArch38AttentionSingleCoreParams.get_singleProcessSOuterSize();
    uint32_t calcTypeSize = softmaxDataTypeSize;
    // 与 kernel 共用同一判定（dV 模板桶 + S1 基本块 + mask），避免 host/kernel 条件漂移。
    bool forceBmm2UB = FiaPolicy::IsForceBmm2UB(enableMask, dVBasicBlock, sOuter);
    if (!forceBmm2UB &&
        ((!enableDN && dSize > FiaPolicy::DV_TEMPLATE_128) || (enableDN && dSize > FiaPolicy::DV_TEMPLATE_192))) {
        bmm2Bytes = static_cast<int64_t>(s1BasicBlock) * bmm2ResBlockSize * calcTypeSize;
        if (dVBasicBlock > FiaPolicy::DV_TEMPLATE_256) {
            vec2Bytes = static_cast<int64_t>(s1BasicBlock) * dVBasicBlock * calcTypeSize;
        }
    }
    bmm2Bytes = AlignUp(bmm2Bytes, GM_ALIGN);
    vec2Bytes = AlignUp(vec2Bytes, GM_ALIGN);
    curWorkspaceSize = static_cast<size_t>((bmm2Bytes + vec2Bytes) * WORKSPACE_BMM_FACTOR * coreNum) +
                       sysWorkspaceSize + accumOutSize + logSumExpSize;
    if (enablePA) {
        curWorkspaceSize += static_cast<uint64_t>(coreNum) * PA_WORKSPACE_ALIGN_FACTOR;
    }
    return curWorkspaceSize;
}

ge::graphStatus FiaArch38PromptTiling::SetPlatMemoryInfo(FiaArch38ContextParams& contextKeyParams)
{
    // In subsequent version, contextKeyParams will be written as a member variable of the class.
    auto compileInfoPtr = contextKeyParams.compileInfoPtr;
    contextKeyParamsPtr = &contextKeyParams;

    OP_CHECK_IF(compileInfoPtr == nullptr, ReportTilingFailure(context_, "compileInfoPtr is null"),
                return ge::GRAPH_FAILED);

    coreNum = compileInfoPtr->aivNum;
    OP_CHECK_IF(coreNum == 0, ReportTilingFailure(context_, "coreNum is 0"), return ge::GRAPH_FAILED);
    aivNum = compileInfoPtr->aivNum;
    OP_CHECK_IF(aivNum == 0, ReportTilingFailure(context_, "aivNum is 0"), return ge::GRAPH_FAILED);
    aicNum = compileInfoPtr->aicNum;
    OP_CHECK_IF(aicNum == 0, ReportTilingFailure(context_, "aicNum is 0"), return ge::GRAPH_FAILED);
    ascendPlatformInfo.socVersion = compileInfoPtr->socShortName;
    ascendPlatformInfo.l1Size = compileInfoPtr->l1Size;
    ascendPlatformInfo.l0CSize = compileInfoPtr->l0CSize;
    ascendPlatformInfo.l0ASize = compileInfoPtr->l0ASize;
    ascendPlatformInfo.l0BSize = compileInfoPtr->l0BSize;
    ascendPlatformInfo.ubSize = compileInfoPtr->ubSize;
    OP_LOGI(contextKeyParams.opName,
            "ascendPlatformInfo:aivNum = %u, aicNum = %u, l1Size = %lu, l0CSize = %lu,"
            "l0ASize = %lu, l0BSize = %lu, ubSize = %lu!",
            aivNum, aicNum, compileInfoPtr->l1Size, compileInfoPtr->l0CSize, compileInfoPtr->l0ASize,
            compileInfoPtr->l0BSize, compileInfoPtr->ubSize);
    CV_RATIO = 1U; // Vector : Cube
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaArch38PromptTiling::SetAttributeInfo(FiaArch38ContextParams& contextKeyParams)
{
    // antiquant check, temporary solution
    if (contextKeyParams.hasKeyAntiquantScale || contextKeyParams.hasValueAntiquantScale) {
        enableKVAntiquant = false;
    } else {
        if (contextKeyParams.inputDataType == ge::DT_FLOAT16 && contextKeyParams.kDataType == ge::DT_INT8) {
            enableKVAntiquant = true;
        }
        OP_CHECK_IF(contextKeyParams.inputDataType == ge::DT_BF16 && contextKeyParams.kDataType == ge::DT_INT8,
                    OP_LOGE_WITH_INVALID_INPUT(contextKeyParams.opName, "key_antiquant_scale, value_antiquant_scale"),
                    return ge::GRAPH_FAILED);
    }

    // prefix check
    isKVHasPrefix =
        (contextKeyParams.keySharedPrefix != nullptr) && (contextKeyParams.valueSharedPrefix != nullptr) ? true : false;
    OP_CHECK_IF((!isKVHasPrefix &&
                 ((contextKeyParams.keySharedPrefix != nullptr) || contextKeyParams.valueSharedPrefix != nullptr)),
                OP_LOGE_WITH_INVALID_INPUT(contextKeyParams.opName, "key_shared_prefix, value_shared_prefix"),
                return ge::GRAPH_FAILED);

    // actSeqLen check
    const gert::Tensor* actSeqLenData = contextKeyParams.actualSequenceLengthQ;
    const gert::Tensor* actSeqLenDataKV = contextKeyParams.actualSequenceLengthKV;
    actSeqLenDims = (actSeqLenData != nullptr) ? actSeqLenData->GetShapeSize() : 0;
    actSeqLenKVDims = (actSeqLenDataKV != nullptr) ? actSeqLenDataKV->GetShapeSize() : 0;
    enableActSeqLen =
        !((actSeqLenDims == 0) || (actSeqLenData == nullptr) || (actSeqLenData->GetData<int64_t>() == nullptr));
    enableActSeqLenKV =
        !((actSeqLenKVDims == 0) || (actSeqLenDataKV == nullptr) || (actSeqLenDataKV->GetData<int64_t>() == nullptr));

    // PA check
    if (contextKeyParams.blockTable != nullptr) {
        OP_CHECK_IF(contextKeyParams.blockSize == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(contextKeyParams.opName, "block_size",
                                                             "block_size is required when PA is enabled"),
                    return ge::GRAPH_FAILED);
        enablePA = true;
    }

    // Pse
    const gert::StorageShape* pseShiftShape = contextKeyParams.pseShiftShape;
    enablePseShift = (contextKeyParams.fromTilingSink == 0) && (contextKeyParams.pseShift != nullptr) &&
                     (pseShiftShape != nullptr) && (pseShiftShape->GetStorageShape().GetShapeSize() > 0);

    if (contextKeyParams.pseType != nullptr) {
        pseType = *contextKeyParams.pseType;
        OP_CHECK_IF(
            (pseType != 0) && (pseType != 2) && (pseType != 3),
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(contextKeyParams.opName, "pse_type", std::to_string(pseType).c_str(),
                                                  "pse_type is not supported, must be 0, 2 or 3"),
            return ge::GRAPH_FAILED);
        faTilingAdapter.inputParamsRegbase.set_pseType(pseType);
    }
    if (pseType == PSE_TYPE_2_TILING_V2 || pseType == PSE_TYPE_3_TILING_V2) {
        enableAlibiPse = true;
        enablePseShift = false;
    }

    // innerprecise, 910B defaults to high-performance.
    const int64_t* innerPrecisePtr = contextKeyParams.innerPrecisePtr;
    innerPrecise = innerPrecisePtr != nullptr ? static_cast<int32_t>(*innerPrecisePtr) : HIGH_PERFORMANCE;

    // mask check
    const gert::StorageShape* attenMaskShape = contextKeyParams.attentionMaskShape;
    enableMask = (contextKeyParams.attentionMask != nullptr) && (attenMaskShape != nullptr) &&
                 (attenMaskShape->GetStorageShape().GetShapeSize() != 0);

    // sparsemode check
    const int32_t* sparseMode = contextKeyParams.sparseMode;
    isDefaultSparseMode = (sparseMode == nullptr) || ((sparseMode != nullptr) && (*sparseMode == SPARSE_MODE_NO_MASK));

    // tensorlist check
    enableTensorList = contextKeyParams.isKvContinuous == 1 ? false : true;

    // LeftPadding check
    enableLeftPadding = ((contextKeyParams.queryPaddingSize != nullptr) || (contextKeyParams.kvPaddingSize != nullptr));

    // postQuant check
    if (contextKeyParams.outputDataType != ge::DT_BF16 && contextKeyParams.outputDataType != ge::DT_FLOAT16) {
        enablePostQuant = true;
    }

    // per-tensor or per-block check
    if (contextKeyParams.inputDataType == ge::DT_INT8) {
        enablePertensorQuant = true;
    }

    const int64_t* keyAntiquantMode = contextKeyParams.keyAntiquantMode;
    const int64_t* queryQuantMode = contextKeyParams.queryQuantMode;
    const int64_t* valueAntiquantMode = contextKeyParams.valueAntiquantMode;
    if (contextKeyParams.inputDataType == ge::DT_HIFLOAT8 || contextKeyParams.inputDataType == ge::DT_FLOAT8_E4M3FN) {
        if (*keyAntiquantMode == 7 && *queryQuantMode == 7 && *valueAntiquantMode == 7) { // 7: FP8 perblock quant
            enablePerblockQuant = true;
        } else {
            enablePertensorQuant = true;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaArch38PromptTiling::CheckRopeInvalid(const FiaArch38ContextParams& contextKeyParams) const
{
    const gert::StorageShape* queryRopeShapeParam = contextKeyParams.queryRopeInputShape;
    const gert::StorageShape* keyRopeShapeParam = contextKeyParams.keyRopeInputShape;
    if ((queryRopeShapeParam != nullptr) && (queryRopeShapeParam->GetStorageShape().GetShapeSize() == 0)) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            contextKeyParams.opName, "query_rope", "0",
            "queryRope must not be an empty tensor when queryRopeShape is not null");
        return ge::GRAPH_FAILED;
    }
    if ((keyRopeShapeParam != nullptr) && (keyRopeShapeParam->GetStorageShape().GetShapeSize() == 0)) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(contextKeyParams.opName, "key_rope", "0",
                                                  "keyRope must not be an empty tensor when keyRopeShape is not null");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaArch38PromptTiling::CheckTensorInvalid(const FiaArch38ContextParams& contextKeyParams) const
{
    if (!CheckNonEmptyShapeExceptions(contextKeyParams, contextKeyParams.queryInputShape, "query")) {
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF((contextKeyParams.queryInputShape->GetStorageShape().GetShapeSize() == 0) &&
                    (contextKeyParams.outputShape->GetStorageShape().GetShapeSize() != 0),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "query, attention_out",
                                                       "query empty, attention_out not empty",
                                                       "query input is empty but output is not empty"),
                return ge::GRAPH_FAILED);
    if (!CheckNonEmptyShapeExceptions(contextKeyParams, contextKeyParams.keyInputShape, "key")) {
        return ge::GRAPH_FAILED;
    }
    if (!CheckNonEmptyShapeExceptions(contextKeyParams, contextKeyParams.valueInputShape, "value")) {
        return ge::GRAPH_FAILED;
    }
    if (!CheckNonEmptyShapeExceptions(contextKeyParams, contextKeyParams.outputShape, "output")) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

bool FiaArch38PromptTiling::CheckAlibiPseShiftTypeAndShape(FiaArch38ContextParams& contextKeyParams, uint32_t n)
{
    const gert::StorageShape* pseShape = contextKeyParams.pseShiftShape;
    if (!CheckNonEmptyShapeExceptions(contextKeyParams, pseShape, "pseShift")) {
        return false;
    }
    auto& inputParams = faTilingAdapter.inputParamsRegbase;

    pseShiftElemType = contextKeyParams.pseShiftDataType;

    OP_CHECK_IF((pseShiftElemType != ge::DT_FLOAT),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(contextKeyParams.opName, "pse_shift",
                                                      GetPfaDataTypeStr(pseShiftElemType).c_str(),
                                                      "when pseType is 2/3, pse shift type must be float"),
                return false);

    // 0: (B,N2,G,S1,S2), 1: (B,N2,G,1,S2)
    PfaPseShapeType pseShapeType = PfaPseShapeType::PSE_B_N2_G_1_S2;
    if (pseShape != nullptr && pseShape->GetStorageShape().GetDimNum() != 0) {
        auto& pseShapeDims = pseShape->GetStorageShape();
        int64_t pseDimNum = pseShapeDims.GetDimNum();
        int64_t pseBSize = 0;
        if (pseType == static_cast<int64_t>(PfaPseType::PSE_INNER_MUL_ADD_TYPE) ||
            pseType == static_cast<int64_t>(PfaPseType::PSE_INNER_MUL_ADD_SQRT_TYPE)) {
            OP_CHECK_IF(pseDimNum != SLOPE_N_DIM_NUM,
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(contextKeyParams.opName, "pse_shift",
                                                                 std::to_string(pseDimNum).c_str(),
                                                                 "when pseType is 2/3, the dim of pseShift must be 1"),
                        return false);
            int64_t pseShiftN = pseShape->GetStorageShape().GetDim(0); // 0: The first dimension is N.
            OP_CHECK_IF(pseShiftN != n,
                        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                            contextKeyParams.opName, "pse_shift", std::to_string(pseShiftN).c_str(),
                            "when pseType is 2/3, the length of pseShift must be equal to query head number"),
                        return false);
            if (pseDimNum == 1) {
                pseShapeType = PfaPseShapeType::PSE_1_N2_G_SLOPE;
                pseBSize = 1;
            }
        }
        inputParams.set_pseBSize(static_cast<uint32_t>(pseBSize));
    }

    inputParams.set_pseShapeType(static_cast<uint8_t>(pseShapeType));
    return true;
}

ge::graphStatus FiaArch38PromptTiling::CheckSingleAttribute(FiaArch38ContextParams& contextKeyParams,
                                                            PFAShapeInfo& queryShapeInfo, PFAShapeInfo& keyShapeInfo,
                                                            PFAShapeInfo& valueShapeInfo,
                                                            PFAShapeInfo& queryRopeShapeInfo,
                                                            FiaArch38FlashAttentionTilingData& tilingData)
{
    if (!CheckIO(contextKeyParams, queryShapeInfo, valueShapeInfo)) {
        OP_LOGE_WITHOUT_REPORT(contextKeyParams.opName, "Check query/ouput failed!");
        return ge::GRAPH_FAILED;
    }
    if (!CheckKV(contextKeyParams, keyShapeInfo, valueShapeInfo)) {
        OP_LOGE_WITHOUT_REPORT(contextKeyParams.opName, "Check key/value failed!");
        return ge::GRAPH_FAILED;
    }
    if (!CheckRope(contextKeyParams, queryShapeInfo, keyShapeInfo, queryRopeShapeInfo)) {
        OP_LOGE_WITHOUT_REPORT(contextKeyParams.opName, "Check queryRope/keyRope failed!");
        return ge::GRAPH_FAILED;
    }
    if (!CheckQueryAndKey(contextKeyParams, queryShapeInfo, keyShapeInfo, tilingData)) {
        OP_LOGE_WITHOUT_REPORT(contextKeyParams.opName, "Check query and key consistency failed!");
        return ge::GRAPH_FAILED;
    }
    if (enableIFAMLA && (!CheckIFAMLA(contextKeyParams, queryShapeInfo))) {
        return false;
    }
    // print shape info
    OP_LOGI(contextKeyParams.opName,
            "Tiling Info: Q_B is %u, KV_B is %u, Q_N is %u, KV_N is %u, Q_S is %u, KV_S is %u, H is %u, D is %u, "
            "headNumRatio = %u",
            queryShapeInfo.b, keyShapeInfo.b, queryShapeInfo.n, keyShapeInfo.n, queryShapeInfo.s, S2, queryShapeInfo.h,
            queryShapeInfo.d, tilingData.fiaArch38AttentionBaseParams.get_headNumRatio());

    // quant/dequant/antiquant
    if (!CheckQuant(contextKeyParams, queryShapeInfo, keyShapeInfo, valueShapeInfo)) {
        OP_LOGE_WITHOUT_REPORT(contextKeyParams.opName, "Check quant failed!");
        return ge::GRAPH_FAILED;
    }

    // prefix check
    if (!CheckPrefix(contextKeyParams, queryShapeInfo, keyShapeInfo, tilingData)) {
        OP_LOGE_WITHOUT_REPORT(contextKeyParams.opName, "Check prefix failed!");
        return ge::GRAPH_FAILED;
    }

    // actSeq check
    if (!CheckActSeq(contextKeyParams, queryShapeInfo)) {
        OP_LOGE_WITHOUT_REPORT(contextKeyParams.opName, "Check actual sequence failed!");
        return ge::GRAPH_FAILED;
    }

    // actSeqLen check
    if ((contextKeyParams.fromTilingSink == static_cast<uint32_t>(0)) &&
        !CheckActSeqLen(contextKeyParams, queryShapeInfo, keyShapeInfo)) {
        OP_LOGE_WITHOUT_REPORT(contextKeyParams.opName, "Check actual sequence length failed!");
        return ge::GRAPH_FAILED;
    }

    // PA check
    if (!enablePA) {
        if (enableTensorList) {
            S2 = contextKeyParams.maxKVs;
        } else {
            S2 = keyShapeInfo.s;
        }
    } else if (!CheckPATypeAndShape(contextKeyParams, queryShapeInfo, queryRopeShapeInfo, tilingData)) {
        return ge::GRAPH_FAILED;
    }

    // pse check
    if (enablePseShift) {
        if (!CheckPseShiftTypeAndShape(contextKeyParams, queryShapeInfo.b, queryShapeInfo.n, queryShapeInfo.s, S2)) {
            return ge::GRAPH_FAILED;
        }
        usePseShift = 1;
    } else if (enableAlibiPse) {
        if (!CheckAlibiPseShiftTypeAndShape(contextKeyParams, queryShapeInfo.n)) {
            return ge::GRAPH_FAILED;
        }
        if (pseType == PSE_TYPE_2_TILING_V2) {
            usePseShift = static_cast<uint32_t>(PSE_TYPE_2_TILING_V2);
        } else if (pseType == PSE_TYPE_3_TILING_V2) {
            usePseShift = static_cast<uint32_t>(PSE_TYPE_3_TILING_V2);
        }
    }
    // innerPrecise check
    if (!CheckInnerPrecise(contextKeyParams, tilingData)) {
        return ge::GRAPH_FAILED;
    }

    // mask check
    if (enableMask) {
        if (!CheckMaskTypeAndShape(contextKeyParams, tilingData)) {
            return ge::GRAPH_FAILED;
        }
        tilingData.fiaArch38AttentionBaseParams.set_useMask(1);
    } else {
        tilingData.fiaArch38AttentionBaseParams.set_useMask(0);
    }

    // sparseMode check
    if (!CheckSparseMode(contextKeyParams, queryShapeInfo.s, tilingData)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

bool FiaArch38PromptTiling::CheckAlibiPseCrossover(FiaArch38ContextParams& contextKeyParams)
{
    if (!enableAlibiPse) {
        return true;
    }

    OP_CHECK_IF(enableLeftPadding,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "pse_type, leftPadding",
                                                       "pseType 2/3 with left padding",
                                                       "when pseType is 2/3, left padding is not supported"),
                return false);

    OP_CHECK_IF((enableIFAMLA || enablePFARope),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "pse_type, query_rope/key_rope",
                                                       (std::to_string(pseType) + ", present").c_str(),
                                                       "when pseType is 2/3, rope is not supported"),
                return false);

    OP_CHECK_IF(
        (enablePFAMLA),
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(contextKeyParams.opName, "query, key, value", "192, 192, 128",
                                                   "when pseType is 2/3, query, key and value D must be the same"),
        return false);

    return true;
}

bool FiaArch38PromptTiling::CheckArch38PlatformSupported(const FiaArch38ContextParams& contextKeyParams,
                                                         const PFAShapeInfo& queryShapeInfo,
                                                         const PFAShapeInfo& valueShapeInfo) const
{
    // 以下约束只与平台已注册的模板/tilingKey 有关，不依赖 IFA/合轴改写后的 n/s，
    // 因此 IFA / MLA / 合轴路径也必须满足，不能放在 early-return 之后。
    OP_CHECK_IF((inputLayout != InputLayout::BNSD),
                OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(contextKeyParams.opName, "input_layout", contextKeyParams.layout,
                                                       "only BNSD layout is supported on this platform"),
                return false);

    OP_CHECK_IF((!IsArch38SupportedHeadSize(static_cast<int64_t>(queryShapeInfo.d)) ||
                 !IsArch38SupportedHeadSize(static_cast<int64_t>(valueShapeInfo.d))),
                OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                    contextKeyParams.opName, "query, value",
                    (std::to_string(queryShapeInfo.d) + ", " + std::to_string(valueShapeInfo.d)).c_str(),
                    "head size only supports 64, 80, 128 or 256 on this platform"),
                return false);

    // 本平台注册的模板中 query d 恒等于 value d。
    OP_CHECK_IF((queryShapeInfo.d != valueShapeInfo.d),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    contextKeyParams.opName, "query.d, value.d",
                    (std::to_string(queryShapeInfo.d) + ", " + std::to_string(valueShapeInfo.d)).c_str(),
                    "query d and value d must be equal on this platform."),
                return false);

    // 仅放行已注册 tiling key 的 dtype 组合，避免分发到不存在的 kernel。
    OP_CHECK_IF(
        !IsArch38SupportedDataType(contextKeyParams.inputDataType, contextKeyParams.kDataType,
                                   contextKeyParams.outputDataType, static_cast<int64_t>(queryShapeInfo.d)),
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            contextKeyParams.opName, "query, key, attentionOut",
            (GetPfaDataTypeStr(contextKeyParams.inputDataType) + ", " + GetPfaDataTypeStr(contextKeyParams.kDataType) +
             ", " + GetPfaDataTypeStr(contextKeyParams.outputDataType))
                .c_str(),
            "only fp16/int8 query/key/value with fp16/int8 output is supported on this platform, "
            "and head size 256 only supports fp16 query/key/value with fp16 output."),
        return false);

    // head size 256 仅有带 mask 场景的 tiling key。
    OP_CHECK_IF(((static_cast<int64_t>(queryShapeInfo.d) == ARCH38_SUPPORTED_D_256) && (!enableMask)),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(contextKeyParams.opName, "atten_mask", "absent",
                                                      "head size 256 requires atten_mask on this platform."),
                return false);

    return true;
}

bool FiaArch38PromptTiling::CheckArch38ScenarioSupported(const FiaArch38ContextParams& contextKeyParams,
                                                         const PFAShapeInfo& queryShapeInfo,
                                                         const PFAShapeInfo& valueShapeInfo) const
{
    // 本平台没有 queryRope/keyRope 对应的模板：所有 tiling key 都以 hasRope == false 实例化，
    // 若放行 PFARope 场景会静默忽略 rope 输入。
    OP_CHECK_IF(enablePFARope,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(contextKeyParams.opName, "queryRope/keyRope", "not empty",
                                                      "queryRope/keyRope is not supported on this platform."),
                return false);

    // 本平台同样没有 MLA 对应的模板：entry 中不存在 +3e7（MLA）key，也没有 d=192/576 的组合，
    // 放行只会生成未注册的 key 并分发失败，此处显式拒绝。
    OP_CHECK_IF((enableIFAMLA || enablePFAMLA),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(contextKeyParams.opName, "queryRope/keyRope or query.d, value.d",
                                                      "MLA", "MLA is not supported on this platform."),
                return false);

    // 平台级约束（layout / head size / dQ == dV / dtype / D256 mask）对所有路径生效。
    if (!CheckArch38PlatformSupported(contextKeyParams, queryShapeInfo, valueShapeInfo)) {
        return false;
    }

    // 带 mask 的 band（sparseMode=4）会生成 +2e8 的 tilingKey，本平台没有注册；
    // 不带 mask 的 band 仍走普通的 no-mask key，IFA / 合轴可继续使用。
    OP_CHECK_IF(
        (enableMask && (sparseModeVal == SPARSE_MODE_BAND)),
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "sparse_mode, atten_mask", "band mask present",
                                               "band mask is not supported on this platform"),
        return false);

    // IFA / 合轴在调用点前已按 gSize 改写 queryShapeInfo.n/.s，下面 qs == kvs、qs > 1 等
    // PFA 原始形状约束不适用，因此这里只跳过形状相关检查（平台级约束已在上方执行）。
    if (enableIFA || enablePFAMerge) {
        return true;
    }

    OP_CHECK_IF((queryShapeInfo.b != 1U && queryShapeInfo.b != 9U),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(contextKeyParams.opName, "query batch",
                                                      std::to_string(queryShapeInfo.b).c_str(),
                                                      "batch size must be 1 or 9 on this platform"),
                return false);

    OP_CHECK_IF((queryShapeInfo.s != static_cast<uint64_t>(S2)),
                OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                    contextKeyParams.opName, "query, key/value",
                    (std::to_string(queryShapeInfo.s) + ", " + std::to_string(S2)).c_str(),
                    "query seq length must equal kv seq length on this platform"),
                return false);

    OP_CHECK_IF(((queryShapeInfo.s <= 1UL) || (S2 <= 1U)),
                OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                    contextKeyParams.opName, "query, key/value",
                    (std::to_string(queryShapeInfo.s) + ", " + std::to_string(S2)).c_str(),
                    "query and kv seq length must be greater than 1 on this platform"),
                return false);

    OP_CHECK_IF((queryShapeInfo.s > static_cast<uint64_t>(ARCH38_SUPPORTED_MAX_SEQ)),
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    contextKeyParams.opName, "query", std::to_string(queryShapeInfo.s).c_str(),
                    "seq length must be less than or equal to ARCH38_SUPPORTED_MAX_SEQ on this platform"),
                return false);

    const int64_t nQ = *contextKeyParams.headsNumber;
    const int64_t nKV = (*contextKeyParams.numKeyValueHeads > 0) ? *contextKeyParams.numKeyValueHeads : nQ;
    OP_CHECK_IF(
        ((nQ <= 0) || (nQ > ARCH38_SUPPORTED_MAX_N) || (nKV <= 0) || (nKV > ARCH38_SUPPORTED_MAX_N)),
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "num_heads, num_key_value_heads",
                                               (std::to_string(nQ) + ", " + std::to_string(nKV)).c_str(),
                                               "head num must be in (0, ARCH38_SUPPORTED_MAX_N] on this platform"),
        return false);

    // Guarantees the kernel never takes the InitOutput branch.
    OP_CHECK_IF((needInit != 0U),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    contextKeyParams.opName, "actual_seq_lengths, pre_tokens, leftPadding", "needInit = 1",
                    "output must be fully covered by the main loop, check actual_seq_lengths / preTokens / "
                    "left padding configuration"),
                return false);

    OP_CHECK_IF((sparseModeVal != SPARSE_MODE_NO_MASK),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(contextKeyParams.opName, "sparse_mode",
                                                      std::to_string(sparseModeVal).c_str(),
                                                      "attention mask only supports sparseMode = 0 on this platform"),
                return false);

    if (!enableMask) {
        return true;
    }

    // With sparseMode = 0, nextTokensPerbatch always equals this attribute, no per-batch conversion needed.
    OP_CHECK_IF((sparseNextTokens != 0),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(contextKeyParams.opName, "next_tokens",
                                                      std::to_string(sparseNextTokens).c_str(),
                                                      "attention mask requires nextTokens = 0"),
                return false);

    OP_CHECK_IF((sparsePreTokens <= 0),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(contextKeyParams.opName, "pre_tokens",
                                                      std::to_string(sparsePreTokens).c_str(),
                                                      "attention mask requires preTokens > 0"),
                return false);

    return true;
}

ge::graphStatus FiaArch38PromptTiling::CheckCrossoverAttribute(FiaArch38ContextParams& contextKeyParams,
                                                               PFAShapeInfo& queryShapeInfo,
                                                               std::vector<int64_t>& actualSeqLengths,
                                                               std::vector<int64_t>& actualSeqLengthsKV,
                                                               FiaArch38FlashAttentionTilingData& tilingData)
{
    // PA and prefix,antiquant,actseqlenKV features crossover
    if (!CheckPACrossover(contextKeyParams, queryShapeInfo)) {
        return ge::GRAPH_FAILED;
    }

    // Mask and innerprecise,sparseMode feature crossover
    if (!CheckMaskCrossover(contextKeyParams, queryShapeInfo, tilingData)) {
        return ge::GRAPH_FAILED;
    }

    // TNDLayout and left padding, tensorlist, pse feature crossover
    if (!CheckTNDLayoutCrossover(contextKeyParams)) {
        return ge::GRAPH_FAILED;
    }

    if (!CheckAlibiPseCrossover(contextKeyParams)) {
        return ge::GRAPH_FAILED;
    }

    if ((contextKeyParams.fromTilingSink == 0) &&
        !CheckMultiFeatureCrossover(contextKeyParams, queryShapeInfo, actualSeqLengths, actualSeqLengthsKV,
                                    tilingData)) {
        return ge::GRAPH_FAILED;
    }

    if (!CheckPerblockCrossover(contextKeyParams)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaArch38PromptTiling::AdjustTilingData(FiaArch38ContextParams& contextKeyParams,
                                                        FiaArch38FlashAttentionTilingData& tilingData,
                                                        const PFAShapeInfo& queryShapeInfo)
{
    uint32_t sOuterFactor = 0;
    uint32_t sInnerFactor = 0;
    uint32_t softmaxSInnerFactor = 0;
    uint32_t softmaxSOuterFactor = 0;
    // Currently, there will be no D splitting scenario, and split D = 0 is default when splitting.
    auto ret = AdjustCVTilingCVDiff(contextKeyParams, sOuterFactor, sInnerFactor, softmaxSOuterFactor, tilingData,
                                    queryShapeInfo);
    OP_CHECK_IF(!ret, ReportTilingFailure(context_, "adjust tiling cv diff fail"), return ge::GRAPH_FAILED);

    // fp16 输入的部分形状没有注册 tiling key，在此 fail fast。
    // 该判断需与 fia_arch38_entry_regbase.h 的 key 列表保持一致。
    const bool isFp16Input = (inputType == ge::DT_FLOAT16);
    const bool isSmallSOuter = (sOuterFactor == 64U);
    const bool isHeadSize64 = (static_cast<int64_t>(queryShapeInfo.d) == ARCH38_SUPPORTED_D_64);
    // fp16 -> fp16：d = 64 且无 mask 的组合没有注册 key（小 s 已固定走大 s）。
    OP_CHECK_IF((isFp16Input && (outputType == ge::DT_FLOAT16) && isHeadSize64 && (!enableMask)),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    contextKeyParams.opName, "input_dtype, output_dtype, head_size, atten_mask",
                    (std::string(GetPfaDataTypeStr(contextKeyParams.inputDataType)) + ", " +
                     GetPfaDataTypeStr(outputType) + ", " + std::to_string(queryShapeInfo.d) + ", absent")
                        .c_str(),
                    "fp16->fp16 with head size 64 without atten_mask has no registered tiling key on this platform."),
                return ge::GRAPH_FAILED);
    // fp16 -> int8：只注册了小 s（64/256）模板，且 d = 64 带 mask 的组合没有注册 key。
    OP_CHECK_IF((isFp16Input && (outputType == ge::DT_INT8) && ((!isSmallSOuter) || (isHeadSize64 && enableMask))),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    contextKeyParams.opName, "output_dtype, sOuter, head_size, atten_mask",
                    (std::string(GetPfaDataTypeStr(outputType)) + ", " + std::to_string(sOuterFactor) + ", " +
                     std::to_string(queryShapeInfo.d) + ", " + (enableMask ? "true" : "false"))
                        .c_str(),
                    "fp16->int8 only supports sOuter = 64, and head size 64 combined with atten_mask "
                    "has no registered tiling key on this platform."),
                return ge::GRAPH_FAILED);

    softmaxSInnerFactor = sInnerFactor;

    tilingData.fiaArch38AttentionBaseParams.set_softmaxOuterSize(softmaxSOuterFactor);
    tilingData.fiaArch38AttentionSingleCoreParams.set_singleProcessSOuterSize(sOuterFactor);
    tilingData.fiaArch38AttentionSingleCoreParams.set_singleProcessSInnerSize(sInnerFactor);
    tilingData.fiaArch38AttentionBaseParams.set_splitS2(splitS2);

    sOuterFactorTiling = sOuterFactor;
    softmaxSInnerFactorTiling = softmaxSInnerFactor;
    softmaxSOuterFactorTiling = softmaxSOuterFactor;
    return ge::GRAPH_SUCCESS;
}

bool FiaArch38PromptTiling::IsFlashDecode() const
{
    return false;
}

ge::graphStatus FiaArch38PromptTiling::SplitBNS(FiaArch38FlashAttentionTilingData& tilingData, uint64_t bng)
{
    uint64_t batchSize = tilingData.fiaArch38AttentionBaseParams.get_batchSize();
    uint64_t headNumSize = tilingData.fiaArch38AttentionBaseParams.get_headNumSize() * gSize;
    uint64_t headDimAlign = AlignUp(tilingData.fiaArch38AttentionBaseParams.get_vHeadSize(), BYTE_BLOCK);
    uint32_t kvSplitLimit = 256U; // 256: 经验值
    int64_t kvSplitPart = 0;
    if (bng != 0) {
        kvSplitPart = aicNum / bng;
    }
    while (((maxActualseqKV / kvSplitPart) < kvSplitLimit) && (kvSplitPart > 1)) {
        kvSplitPart--;
    }

    faTilingAdapter.inputParamsRegbase.set_kvSplitPart(kvSplitPart);
    faTilingAdapter.inputParamsRegbase.set_accumOutSize(batchSize * headNumSize * kvSplitPart * headDimAlign);
    faTilingAdapter.inputParamsRegbase.set_logSumExpSize(batchSize * headNumSize * kvSplitPart *
                                                         (BYTE_BLOCK / sizeof(float)));

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaArch38PromptTiling::ComputeTilingData(FiaArch38ContextParams& contextKeyParams,
                                                         std::vector<int64_t>& actualSeqLengths,
                                                         std::vector<int64_t>& actualSeqLengthsKV,
                                                         FiaArch38FlashAttentionTilingData& tilingData)
{
    // Compute tiling data.
    bool isAttenMaskUsed = (contextKeyParams.attentionMaskShape != nullptr);
    FiaArch38SplitNBSeq(tilingData, actualSeqLengths, actualSeqLengthsKV, isAttenMaskUsed);

    if (needInit == 1) {
        FiaArch38InitOutputSplit(contextKeyParams.outputShape->GetStorageShape().GetShapeSize(), tilingData);
    }

    if (contextKeyParams.isSoftMaxLseEnable) {
        FiaArch38InitSoftmaxLseOutputSplit(contextKeyParams.lseoutputShape->GetStorageShape().GetShapeSize(),
                                           tilingData);
    }

    if (enableIFA && !enablePFAMerge) {
        FiaArch38AttentionSingleCoreParams* singleCoreParams = &tilingData.fiaArch38AttentionSingleCoreParams;
        uint32_t sOuterSize = singleCoreParams->get_singleProcessSOuterSize();
        uint64_t batchSize = tilingData.fiaArch38AttentionBaseParams.get_batchSize();
        uint64_t headNumKVSize = tilingData.fiaArch38AttentionBaseParams.get_headNumSize(); // IFA kv N
        uint64_t bng = batchSize * headNumKVSize * (gSize + sOuterSize - 1) / sOuterSize;
        if (IsFlashDecode()) {
            enableFlashDecode = true;
            return SplitBNS(tilingData, bng);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaArch38PromptTiling::ComputeTilingKey(uint64_t& tilingKey, FiaArch38ContextParams& contextKeyParams,
                                                        uint32_t& numBlocksToBeSet,
                                                        FiaArch38FlashAttentionTilingData& tilingData)
{
    bool tilingRet = TilingGetTilingKeyAttentionAscendC(tilingKey, contextKeyParams, tilingData);
    OP_CHECK_IF(!tilingRet, ReportTilingFailure(context_, "Get tilingKey fail"), return ge::GRAPH_FAILED);

    auto platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr, OP_LOGE_WITHOUT_REPORT(contextKeyParams.opName, "platformInfoPtr is null!"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    numBlocksToBeSet = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);

    size_t* workspaces = contextKeyParams.workspaceSize;
    workspaces[0] = GetPFAWorkSpaceSize(tilingData);
    OP_LOGI(contextKeyParams.opName, "The Tiling key is %lu", tilingKey);
    return ge::GRAPH_SUCCESS;
}

void FiaArch38PromptTiling::SetAttenMaskCompressMode()
{
    static std::map<uint32_t, uint8_t> sparseToCompressModeMap = {
        {SPARSE_MODE_NO_MASK, PfaAttenMaskCompressMode::PFA_NO_COMPRESS_MODE},
        {SPARSE_MODE_ALL_MASK, PfaAttenMaskCompressMode::PFA_NO_COMPRESS_MODE},
        {SPARSE_MODE_LEFT_UP, PfaAttenMaskCompressMode::PFA_LEFT_UP_CAUSAL_MODE},
        {SPARSE_MODE_RIGHT_DOWN, PfaAttenMaskCompressMode::PFA_RIGHT_DOWN_CAUSAL_MODE},
        {SPARSE_MODE_BAND, PfaAttenMaskCompressMode::PFA_BAND_MODE}};
    auto itr = sparseToCompressModeMap.find(sparseModeVal);
    if (itr == sparseToCompressModeMap.end()) {
        faTilingAdapter.inputParamsRegbase.set_attenMaskCompressMode(0);
    } else {
        faTilingAdapter.inputParamsRegbase.set_attenMaskCompressMode(itr->second);
    }
}

void FiaArch38PromptTiling::SetLayoutType()
{
    static std::map<InputLayout, LayoutType> layoutStrToLayoutTypeMap = {
        {InputLayout::BSH, LayoutType::LAYOUT_BSH},
        {InputLayout::TND, LayoutType::LAYOUT_TND},
        {InputLayout::BSND, LayoutType::LAYOUT_BSND},
        {InputLayout::BNSD, LayoutType::LAYOUT_BNSD},
    };
    auto itr = layoutStrToLayoutTypeMap.find(inputLayout);
    if (itr == layoutStrToLayoutTypeMap.end()) {
        faTilingAdapter.inputParamsRegbase.set_layoutType(static_cast<uint8_t>(0));
    } else {
        faTilingAdapter.inputParamsRegbase.set_layoutType(static_cast<uint8_t>(itr->second));
    }
}

ge::graphStatus FiaArch38PromptTiling::SetQKVStartIdx(FiaArch38ContextParams& contextKeyParams)
{
    auto& inputParams = faTilingAdapter.inputParamsRegbase;
    inputParams.set_qStartIdx(0);
    inputParams.set_kvStartIdx(0);
    if (!enableAlibiPse) {
        return ge::GRAPH_SUCCESS;
    }
    int64_t qStartIdx = 0;
    int64_t kvStartIdx = 0;
    auto qStartIdxTensor = contextKeyParams.qStartIdx;
    if (qStartIdxTensor != nullptr) {
        if (qStartIdxTensor->GetShapeSize() >= 1) {
            const int64_t* value = qStartIdxTensor->GetData<int64_t>();
            if (value != nullptr) {
                qStartIdx = value[0];
                OP_CHECK_IF(qStartIdx > INT32_MAX || qStartIdx < INT32_MIN,
                            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(contextKeyParams.opName, "q_start_idx",
                                                                  std::to_string(qStartIdx).c_str(),
                                                                  "qStartIdx must be in range [INT32_MIN, INT32_MAX]"),
                            return ge::GRAPH_FAILED);
                inputParams.set_qStartIdx(qStartIdx);
            }
        }
    }

    auto kvStartIdxTensor = contextKeyParams.kvStartIdx;
    if (kvStartIdxTensor != nullptr) {
        if (kvStartIdxTensor->GetShapeSize() >= 1) {
            const int64_t* kvValue = kvStartIdxTensor->GetData<int64_t>();
            if (kvValue != nullptr) {
                kvStartIdx = kvValue[0];
                OP_CHECK_IF(kvStartIdx > INT32_MAX || kvStartIdx < INT32_MIN,
                            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(contextKeyParams.opName, "kv_start_idx",
                                                                  std::to_string(kvStartIdx).c_str(),
                                                                  "kvStartIdx must be in range [INT32_MIN, INT32_MAX]"),
                            return ge::GRAPH_FAILED);
                inputParams.set_kvStartIdx(kvStartIdx);
            }
        }
    }
    // 当kvStartIdx - qStartIdx超出范围后，由于编译器不支持大数值类型转换，kernel侧int_64转float类型时可能发生截断。
    OP_CHECK_IF(
        kvStartIdx - qStartIdx > INT32_MAX || kvStartIdx - qStartIdx < INT32_MIN,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "q_start_idx, kv_start_idx",
                                               (std::to_string(qStartIdx) + ", " + std::to_string(kvStartIdx)).c_str(),
                                               "kvStartIdx - qStartIdx must be in range [INT32_MIN, INT32_MAX]"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void FiaArch38PromptTiling::PFATilingDataconvert(FiaArch38FlashAttentionTilingData& tilingData)
{
    SetLayoutType();
    auto& inputParams = faTilingAdapter.inputParamsRegbase;
    inputParams.set_bSize(tilingData.fiaArch38AttentionBaseParams.get_batchSize());
    // 将GS1合轴与不合轴场景下，有不同含义的n2Size、gSize与s1Size参数，转化为各自实际的值
    if (enableIFAMLA || enableIFA || enablePFAMerge) {
        inputParams.set_n2Size(tilingData.fiaArch38AttentionBaseParams.get_headNumSize());
        inputParams.set_gSize(tilingData.fiaArch38AttentionBaseParams.get_gOfMla());
        inputParams.set_s1Size(tilingData.fiaArch38AttentionBaseParams.get_seqSize() /
                               tilingData.fiaArch38AttentionBaseParams.get_gOfMla());
    } else {
        inputParams.set_n2Size(tilingData.fiaArch38AttentionBaseParams.get_headNumSize() /
                               tilingData.fiaArch38AttentionBaseParams.get_headNumRatio());
        inputParams.set_gSize(tilingData.fiaArch38AttentionBaseParams.get_headNumRatio());
        inputParams.set_s1Size(tilingData.fiaArch38AttentionBaseParams.get_seqSize());
    }
    inputParams.set_s2Size(tilingData.fiaArch38AttentionBaseParams.get_seqInnerSize());
    inputParams.set_alignedS2(0); // 默认值
    inputParams.set_dSize(tilingData.fiaArch38AttentionBaseParams.get_headSize());
    inputParams.set_dSizeV(tilingData.fiaArch38AttentionBaseParams.get_vHeadSize());
    inputParams.set_dSizeRope(ROPE_DIMENSION_SIZE_TILING_V2);
    inputParams.set_scaleValue(tilingData.fiaArch38AttentionBaseParams.get_scaleValue());
    inputParams.set_preTokens(tilingData.fiaArch38AttentionBaseParams.get_preTokens());
    inputParams.set_nextTokens(tilingData.fiaArch38AttentionBaseParams.get_nextTokens());
    inputParams.set_pseS1Size(tilingData.fiaArch38AttentionBaseParams.get_pseShiftS1Size());
    inputParams.set_pseS2Size(tilingData.fiaArch38AttentionBaseParams.get_pseShiftS2Size());
    if (!enableAlibiPse) {
        inputParams.set_pseBSize(tilingData.fiaArch38AttentionSingleCoreParams.get_pseShiftBatch());
        inputParams.set_pseShapeType(0); // 对应训练 PSE_B_N2_G_S1_S2
        inputParams.set_pseType(0);      // 对应训练 PSE_OUTER_MUL_ADD_TYPE
    }
    inputParams.set_bandIndex(0);      // 训练代码中在TND场景生效，用于计算s2方向循环的起始位置
    inputParams.set_pseEncodeType(0);  // 默认值
    inputParams.set_pseAlibiBaseS1(0); // 默认值
    inputParams.set_pseAlibiBaseS2(0); // 默认值
    inputParams.set_attenMaskShapeType(attenMaskShapeType);
    inputParams.set_attenMaskDataType(1); // 默认值
    SetAttenMaskCompressMode();
    inputParams.set_implMode(static_cast<uint8_t>(HIGH_PRECISION));
    inputParams.set_sparseType(sparseType);
    inputParams.set_needDropMaskOp(0); // 默认值
    inputParams.set_keepProb(0);       // 默认值
    inputParams.set_keepProbUint8(0);  // 默认值
    inputParams.set_dropMaskOuter(0);  // 默认值
    inputParams.set_remain(0);         // 默认值
    inputParams.set_attenMaskS2Size(tilingData.fiaArch38AttentionBaseParams.get_maskKVsSize());
    inputParams.set_rsv1(0);              // 默认值
    inputParams.set_s1SparseValidSize(0); // 临时默认值
    inputParams.set_s2SparseValidSize(0); // 临时默认值
    inputParams.set_seed(0);              // 默认值
    inputParams.set_offset(0);            // 默认值

    // PFA
    inputParams.set_blockSize(tilingData.fiaArch38AttentionBaseParams.get_blockSize());
    inputParams.set_blockTableDim2(tilingData.fiaArch38AttentionBaseParams.get_blockTableDim2());
    inputParams.set_paBlockNumSum(tilingData.fiaArch38AttentionBaseParams.get_PABlockNumSum());
    inputParams.set_prefixSeqInnerSize(tilingData.fiaArch38AttentionBaseParams.get_prefixSeqInnerSize());
    inputParams.set_paLayoutType(tilingData.fiaArch38AttentionBaseParams.get_PAlayoutType());
    inputParams.set_attenMaskS1Size(tilingData.fiaArch38AttentionBaseParams.get_maskQsSize());
    inputParams.set_isActualSeqLengthsNull(tilingData.fiaArch38AttentionBaseParams.get_isActualSeqLengthsNull());
    inputParams.set_isActualSeqLengthsKVNull(tilingData.fiaArch38AttentionBaseParams.get_isActualSeqLengthsKVNull());
    inputParams.set_actualSeqLengthsSize(tilingData.fiaArch38AttentionBaseParams.get_actualSeqLengthsSize());
    inputParams.set_actualSeqLengthsKVSize(tilingData.fiaArch38AttentionBaseParams.get_actualSeqLengthsKVSize());
    inputParams.set_deqScaleFlag(tilingData.fiaArch38AttentionBaseParams.get_deqScaleFlag());
    inputParams.set_deqScale2Flag(tilingData.fiaArch38AttentionBaseParams.get_deqScale2Flag());
    inputParams.set_isKvContinuous(tilingData.fiaArch38AttentionBaseParams.get_isKvContinuous());
    inputParams.set_fromFused(tilingData.fiaArch38AttentionBaseParams.get_fromFused());
    inputParams.set_isBSNDOut(tilingData.fiaArch38AttentionBaseParams.get_isBSNDOut());
    inputParams.set_isGqa((tilingData.fiaArch38AttentionBaseParams.get_isIFA() != 0) || enablePFAMerge);
    inputParams.set_isSoftMaxLseEnable(tilingData.fiaArch38AttentionBaseParams.get_isSoftMaxLseEnable());
    inputParams.set_isActualSharedPrefixLenNull(
        tilingData.fiaArch38AttentionBaseParams.get_isActualSharedPrefixLenNull());
    inputParams.set_isQHasLeftPadding(tilingData.fiaArch38AttentionBaseParams.get_isQHasLeftPadding());
    inputParams.set_isKVHasLeftPadding(tilingData.fiaArch38AttentionBaseParams.get_isKVHasLeftPadding());
    inputParams.set_ropeHeadSize(tilingData.fiaArch38AttentionBaseParams.get_ropeHeadSize());
    inputParams.set_isRowInvalid(static_cast<uint8_t>(tilingData.fiaArch38AttentionBaseParams.get_isRowInvalid()));
    inputParams.set_headNumRatio(tilingData.fiaArch38AttentionBaseParams.get_headNumRatio());

    auto& initOutputParams = faTilingAdapter.initOutputParams;
    initOutputParams.set_singleCoreSize(tilingData.fiaArch38AttentionInitOutputParams.get_singleCoreSize());
    initOutputParams.set_totalOutputSize(tilingData.fiaArch38AttentionInitOutputParams.get_totalOutputSize());
    initOutputParams.set_totalSoftMaxLseOutputSize(
        tilingData.fiaArch38AttentionInitOutputParams.get_totalSoftMaxLseOutputSize());
    initOutputParams.set_needInit(static_cast<uint8_t>(tilingData.fiaArch38AttentionInitOutputParams.get_needInit()));
    initOutputParams.set_isOneN(static_cast<uint8_t>(tilingData.fiaArch38AttentionInitOutputParams.get_isOneN()));

    inputParams.set_isPostQuantPerChnl(tilingData.fiaArch38AttentionBaseParams.get_isQuant2Perchannel());
    inputParams.set_isPostQuantBF16(tilingData.fiaArch38AttentionBaseParams.get_isQuant2BF16());
    inputParams.set_antiquantPerTensorFlag(0);
    inputParams.set_antiquantPerHeadFlag(0);
    inputParams.set_antiquantParaSeqSize(1);
}

ge::graphStatus FiaArch38PromptTiling::FiaArch38SetTilingData(gert::TilingContext* context)
{
    faTilingAdapter.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(faTilingAdapter.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaArch38PromptTiling::RunBigKernelTilingWithParams(FiaArch38ContextParams& contextKeyParams,
                                                                    uint64_t& tilingKey, uint32_t& numBlocksToBeSet,
                                                                    FiaArch38FlashAttentionTilingData& tilingData)
{
    // set memory parameters
    if (SetPlatMemoryInfo(contextKeyParams) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // set sttributes switch
    if (SetAttributeInfo(contextKeyParams) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // rope ptr is empty or rope is invalid
    if (CheckRopeInvalid(contextKeyParams) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // tensor ptr is empty or tensor is invalid
    if (CheckTensorInvalid(contextKeyParams) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    } else if (CheckEmptyTensor(contextKeyParams)) {
        return SetEmptyTensor(contextKeyParams, tilingKey, numBlocksToBeSet, tilingData);
    }
    // Input, output check
    PFAShapeInfo queryShapeInfo;
    PFAShapeInfo keyShapeInfo;
    PFAShapeInfo valueShapeInfo;
    PFAShapeInfo queryRopeShapeInfo;

    // Check kinds of single attribute
    if (CheckSingleAttribute(contextKeyParams, queryShapeInfo, keyShapeInfo, valueShapeInfo, queryRopeShapeInfo,
                             tilingData) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // 同IFA合轴实现
    if (enablePFAMerge) {
        if (*contextKeyParams.numKeyValueHeads > 0) {
            gSize = (*contextKeyParams.headsNumber) / (*contextKeyParams.numKeyValueHeads);
        } else {
            gSize = 1U;
        }
        tilingData.fiaArch38AttentionBaseParams.set_headNumRatio(1);
        tilingData.fiaArch38AttentionBaseParams.set_gOfMla(gSize);
    }
    if (enableIFAMLA || enableIFA || enablePFAMerge) {
        queryShapeInfo.n = queryShapeInfo.n / gSize;
        queryShapeInfo.s = queryShapeInfo.s * gSize;
    }

    // multi feature crossover check
    std::vector<int64_t> actualSeqLengths(queryShapeInfo.b);
    std::vector<int64_t> actualSeqLengthsKV(queryShapeInfo.b);

    // Check crossover attribute
    if (CheckCrossoverAttribute(contextKeyParams, queryShapeInfo, actualSeqLengths, actualSeqLengthsKV, tilingData) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // needInit is available only after CheckCrossoverAttribute, so the scenario gate is placed here.
    if (!CheckArch38ScenarioSupported(contextKeyParams, queryShapeInfo, valueShapeInfo)) {
        return ge::GRAPH_FAILED;
    }

    // print shape info
    OP_LOGI(contextKeyParams.opName,
            "inputLayout is %s, innerPrecise is %d, scaleValue is %f, preTokens is %ld, nextTokens is %ld",
            string(contextKeyParams.layout).c_str(), innerPrecise, *contextKeyParams.scaleValue,
            *contextKeyParams.preToken, *contextKeyParams.nextToken);

    // set tilingData
    SetTilingData(contextKeyParams, queryShapeInfo, queryRopeShapeInfo, valueShapeInfo, tilingData);

    // Infering whether the tiling mode is S2 full load, CV diff, and whether to use the matmul norm template.
    InferTilingMod(contextKeyParams, actualSeqLengths, actualSeqLengthsKV, queryShapeInfo.b, queryShapeInfo.d);
    InferSplitCoreMode();
    // Whether to enable constant templates
    InferConstantization();

    if (AdjustTilingData(contextKeyParams, tilingData, queryShapeInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // DN check
    GetEnableDN(tilingData, queryShapeInfo, valueShapeInfo, actualSeqLengths, actualSeqLengthsKV);

    if (ComputeTilingData(contextKeyParams, actualSeqLengths, actualSeqLengthsKV, tilingData) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // Compute tiling key.
    if (ComputeTilingKey(tilingKey, contextKeyParams, numBlocksToBeSet, tilingData) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    PFATilingDataconvert(tilingData);
    if (SetQKVStartIdx(contextKeyParams) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaArch38PromptTiling::DoSubOpTiling(FiaArch38FlashAttentionTilingData& tilingData,
                                                     FiaArch38ContextParams& contextParamsForPFATiling)
{
    uint64_t tilingKey = 7;
    uint32_t numBlocksToBeSet;
    auto ret = RunBigKernelTilingWithParams(contextParamsForPFATiling, tilingKey, numBlocksToBeSet, tilingData);
    tilingKey += BENCHMARK_TILING_KEY;
    context_->SetTilingKey(tilingKey);
    context_->SetBlockDim(numBlocksToBeSet);
    FiaArch38SetTilingData(context_);
    // 使用SyncAll，需要设置为batchmode模式，所有核同时启动，否则多流方式下执行可能会卡死
    context_->SetScheduleMode(BATCH_MODE_SCHEDULE);
    return ret;
}

} // namespace arch38
} // namespace optiling
