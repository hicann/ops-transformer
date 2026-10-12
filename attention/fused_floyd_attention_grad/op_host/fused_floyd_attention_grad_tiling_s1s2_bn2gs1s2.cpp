/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fused_floyd_attention_grad_tiling_s1s2_bn2gs1s2.h"
#include "op_host/tiling_type.h"
#include "op_host/tiling_templates_registry.h"
#include "err/ops_err.h"
#include "op_host/tiling_base.h"

using namespace Ops::Transformer::OpTiling;

namespace optiling {
namespace FFAG {
constexpr uint32_t INITIAL_S1_SPLIT_NUM = 128; // to avoid repeat max value 255
constexpr uint32_t INITIAL_S2_SPLIT_NUM = 64;
constexpr uint32_t MUL_CORE_SYNC_BUFFER = 16 * 1024;

constexpr uint32_t EMPTY_TENSOR = 0;
constexpr uint32_t NORMAL_TENSOR = 1;

constexpr uint32_t MAX_BASIC_BLOCK_SIZE = 1024;
constexpr uint32_t TEMP_BUFFER_REMAIN_SIZE = 1024 * 2;

constexpr uint32_t INPUT_FORMAT_BN2GS2D = 0; // BNSD

constexpr uint32_t CORE_INIT_NUM = 40;
constexpr uint32_t MATMUL_SIZE = 8 * 1024;

constexpr uint32_t INPUT_ALIGN = 16;
constexpr uint32_t WORKSPACE_NUM_ALIGN = 256;
constexpr int64_t GM_ALIGN = 512;

constexpr uint32_t SOFTMAX_PERF = 64;

constexpr uint32_t TOTAL_BLOCK_DIMENSION = 2;
constexpr uint32_t CALCULATED_BLOCK_DIMENSION = 4;
constexpr uint32_t BEGIN_IDX = 0;
constexpr uint32_t END_IDX = 1;
constexpr uint32_t SUM_S1S2 = 2;
constexpr uint32_t SUM_ALL = 3;
constexpr uint32_t LENGTH_IDX = 2;
constexpr uint32_t BASIC_BLOCK_MULTIPLE = 15;
constexpr uint32_t POST_NZ_COEX_NODE = 10;
constexpr uint32_t POST_COEX_NODE = 3;
constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t POST_NZ_RESERVED_N = 4;

constexpr uint32_t FP16_BYTES = 2;
constexpr uint32_t FP16_BLOCK_NUMS = 16;
constexpr uint32_t FP32_BYTES = 4;
constexpr uint32_t FP32_BLOCK_NUMS = 8;
constexpr uint32_t SHAPE_INFO = 32;
constexpr uint32_t C0_SIZE = 16;
constexpr uint32_t BLOCK_SIZE = 32;

constexpr uint32_t MATMAL_INPUT_NUMS = 2;
constexpr uint32_t S1CV_RATIO_DEFAULT = 1;
constexpr uint32_t S2CV_RATIO_DEFAULT = 8;
constexpr uint32_t CV_RATIO_2 = 2;
constexpr uint32_t CV_RATIO_4 = 4;
constexpr uint32_t CV_RATIO_16 = 16;
constexpr uint32_t WORKSPACE_BUFFER = 20 * 1024 * 1024;
constexpr uint32_t PSE_ALIBI_S2_LIMIT_SIZE = 1024;
constexpr uint32_t BIT_NUMS = 8;
constexpr uint32_t S2_NZ_SIZE = 128;
constexpr uint32_t MM12_ND2NZ_SIZE = 5000;
constexpr uint32_t ASCENDC_API_TEMP_BUFFER = 32 * 1024 + 1024; // ND2ND NEED ANOTHER 1K
constexpr uint32_t API_BOOL_ALIGN = 32;                        // ASCEND API ATTENMASK OR DROPOUT LAST DIM ALIGN
constexpr uint32_t SYNC_GLOBAL_WORKSPACE_SIZE = 16 * 1024;
constexpr uint32_t ADDR_ALIGN_SIZE = 512;
constexpr uint32_t FIX_BASEMN_128 = 128;
constexpr uint32_t FIX_BASEMN_256 = 256;
constexpr uint32_t EVERY_BLOCK_INFO_DIM = 4;
constexpr uint32_t S1OUTER_IDX = 0;
constexpr uint32_t S2OUTER_IDX = 1;
constexpr uint32_t S1TAIL_IDX = 2;
constexpr uint32_t S2TAIL_IDX = 3;
// SFMG_DB_SIZE = 18 + 18 + 9+ 9 + 18 = 72K（GraphAB的DB是74K）, CLC1占用18K
constexpr uint32_t SFMG_DB_CLC1_UBSIZE = 18 * 1024;

bool FusedFloydAttentionGradTilingS1s2Bn2gs1s2::IsCapable()
{
    // 基础模板 全部支持
    return true;
}

uint64_t FusedFloydAttentionGradTilingS1s2Bn2gs1s2::GetTilingKey() const
{
    auto dtypeValue = DtypeEnum::FLOAT32;
    if (fBaseParams.mode == BF16) {
        dtypeValue = DtypeEnum::BFLOAT16;
    } else if (fBaseParams.mode == FP32) {
        dtypeValue = DtypeEnum::FLOAT32;
    } else {
        dtypeValue = DtypeEnum::FLOAT16_PRECISION;
    }
    auto attenMaskCfg = fBaseParams.attenMaskOptional == EMPTY_TENSOR ? OptionEnum::DISABLE : OptionEnum::ENABLE;
    LayoutEnum inputLayout = LayoutEnum::BNSD;

    auto pseValue = fBaseParams.pseOptional == NORMAL_TENSOR ? OptionEnum::ENABLE : OptionEnum::DISABLE;
    auto dropValue = fBaseParams.keepProb < 1 ? OptionEnum::ENABLE : OptionEnum::DISABLE;
    auto mm1IsNZOut = fBaseParams.mm1IsNZOut ? OptionEnum::ENABLE : OptionEnum::DISABLE;
    auto mm2IsNZOut = fBaseParams.mm2IsNZOut ? OptionEnum::ENABLE : OptionEnum::DISABLE;
    auto tndS1Pingpong = OptionEnum::DISABLE;
    // tilingKey的12,13位为SameAB模板的标识占位
    uint64_t tilingKey = GET_TILINGKEY(AxisEnum::S2, AxisEnum::S1, AxisEnum::S2, dtypeValue, inputLayout,
                                       SparseEnum::ALL, dropValue, pseValue, attenMaskCfg, mm1IsNZOut, mm2IsNZOut,
                                       OptionEnum::DISABLE, OptionEnum::DISABLE, tndS1Pingpong);
    OP_LOGI(context_, "FAGTiling S1s2Bn2gs1s2 DoTiling success, tiling is %lu.", tilingKey);
    return tilingKey;
}

ge::graphStatus FusedFloydAttentionGradTilingS1s2Bn2gs1s2::GetPlatformInfo()
{
    // 待公共模板实现后，会删除该函数  直接继承基类
    uint32_t coreNum = CORE_INIT_NUM; // 40 is init core num

    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const FlashAttentionScoreGradCompileInfo*>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile_info is null"), return ge::GRAPH_FAILED);

        fBaseParams.coreNum = compileInfoPtr->aivNum;
        fBaseParams.aicNum = compileInfoPtr->aicNum;
        fBaseParams.ubSize = compileInfoPtr->ubSize;
        fBaseParams.l1Size = compileInfoPtr->l1Size;
        fBaseParams.l0aSize = compileInfoPtr->l0aSize;
        fBaseParams.l0cSize = compileInfoPtr->l0cSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        coreNum = ascendcPlatform.GetCoreNumAiv();

        fBaseParams.coreNum = coreNum;
        fBaseParams.aicNum = ascendcPlatform.GetCoreNumAic();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, fBaseParams.ubSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, fBaseParams.l1Size);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, fBaseParams.l0aSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, fBaseParams.l0cSize);
    }
    OP_CHECK_IF((fBaseParams.coreNum == 0) || (fBaseParams.aicNum == 0),
                OP_LOGE(context_, "num of coreNum(aivNum) is %ld, num of aicNum is %ld.", fBaseParams.coreNum,
                        fBaseParams.aicNum),
                return ge::GRAPH_FAILED);

    fBaseParams.ubSize -= MATMUL_SIZE;
    OP_CHECK_IF(fBaseParams.ubSize <= 0, OP_LOGE(context_, "ubSize is invalid."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedFloydAttentionGradTilingS1s2Bn2gs1s2::GetBaseShapeInfo()
{
    // 待公共模板实现后，会删除该函数  直接继承基类
    const gert::StorageShape* queryShape = context_->GetInputShape(QUERY); // [B, N2, G, S1, D]
    const gert::StorageShape* keyShape = context_->GetInputShape(KEY_1);   // [B, N2, 1, S2, D]

    const char* inputLayout = "BNSD";

    OP_CHECK_IF(queryShape == nullptr, OP_LOGE(context_, "queryShape is nullptr."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(keyShape == nullptr, OP_LOGE(context_, "keyShape is nullptr."), return ge::GRAPH_FAILED);

    if (strcmp(inputLayout, "BNSD") == 0) {
        OP_LOGD(context_, "inputLayout == BNSD queryShape");
        fBaseParams.layoutType = INPUT_FORMAT_BN2GS2D;
        fBaseParams.b = queryShape->GetStorageShape().GetDim(0);
        fBaseParams.n2 = keyShape->GetStorageShape().GetDim(1);
        fBaseParams.g = keyShape->GetStorageShape().GetDim(2);
        fBaseParams.s1 = queryShape->GetStorageShape().GetDim(3);
        fBaseParams.d = queryShape->GetStorageShape().GetDim(4);
        fBaseParams.s2 = keyShape->GetStorageShape().GetDim(3);
        OP_LOGD(context_, "inputLayout == BNSD queryShape", "%ld, %ld, %ld, %ld,",
                queryShape->GetStorageShape().GetDim(DIM_0), queryShape->GetStorageShape().GetDim(DIM_1),
                queryShape->GetStorageShape().GetDim(DIM_2), queryShape->GetStorageShape().GetDim(DIM_3));
    } else {
        OP_LOGE(context_, "FAG inputLayout is invalid");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(fBaseParams.n2 == 0, OP_LOGE(context_, "n2 is 0."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedFloydAttentionGradTilingS1s2Bn2gs1s2::GetShapeAttrsInfo()
{
    OP_CHECK_IF(context_ == nullptr, OP_LOGE(context_, "context is nullptr."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetAttrs() == nullptr, OP_LOGE(context_, "GetAttrs is nullptr."), return ge::GRAPH_FAILED);

    auto ret = GetBaseShapeInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    fBaseParams.n1 = fBaseParams.n2 * fBaseParams.g;
    fBaseParams.s1Align = (fBaseParams.s1 + INPUT_ALIGN - 1) / INPUT_ALIGN * INPUT_ALIGN;
    fBaseParams.s2Align = (fBaseParams.s2 + INPUT_ALIGN - 1) / INPUT_ALIGN * INPUT_ALIGN;

    // Direct multiplication is safe: CheckInputShapeValid enforces documented bounds
    // (B<=2048, N%16, M/K 128-aligned <=1M, D=32/64/128), products fit in int64.
    fBaseParams.qSize = fBaseParams.b * fBaseParams.n2 * fBaseParams.g * fBaseParams.s1 * fBaseParams.d;
    fBaseParams.kvSize = fBaseParams.b * fBaseParams.n2 * fBaseParams.g * fBaseParams.s2 * fBaseParams.d;
    fBaseParams.k1v1Size = fBaseParams.b * fBaseParams.n2 * fBaseParams.s2 * fBaseParams.s1 * fBaseParams.d;
    fBaseParams.dropMaskSize = fBaseParams.b * fBaseParams.n2 * fBaseParams.g * fBaseParams.s2 * fBaseParams.s1;

    // mBaseParams is used for matmal tiling module
    OP_CHECK_IF(context_->GetInputDesc(QUERY) == nullptr, OP_LOGE(context_, "InputDesc of query is nullptr."),
                return ge::GRAPH_FAILED);
    auto queryType = context_->GetInputDesc(QUERY)->GetDataType();
    fBaseParams.queryType = queryType;
    fBaseParams.isBf16 = queryType == ge::DT_BF16 ? true : false;
    if (queryType == ge::DT_FLOAT) {
        fBaseParams.dataTypeSize = FP32_BYTES; // init date type fp32 is 4
        fBaseParams.dataBlockNum = FP32_BLOCK_NUMS;
        fBaseParams.calTypeSize = FP32_BYTES; // init cal type fp32 is 4
        fBaseParams.calBlockNum = FP32_BLOCK_NUMS;
    } else {
        fBaseParams.dataTypeSize = FP16_BYTES;
        fBaseParams.dataBlockNum = FP16_BLOCK_NUMS;
        fBaseParams.calTypeSize = FP32_BYTES;
        fBaseParams.calBlockNum = FP32_BLOCK_NUMS;
    }

    fBaseParams.mm1IsNZOut = false; // 先支持ND
    fBaseParams.mm2IsNZOut = false; // 支持ND
    fBaseParams.dataBlockNum = BYTE_BLOCK / fBaseParams.dataTypeSize;
    fBaseParams.calBlockNum = BYTE_BLOCK / fBaseParams.calTypeSize;

    // nz out
    int64_t qSizeAlign = fBaseParams.qSize;
    int64_t kvSizeAlign = fBaseParams.kvSize;
    int64_t k1v1SizeAlign = fBaseParams.k1v1Size;
    if (fBaseParams.mm2IsNZOut) {
        int64_t dAlign = (fBaseParams.d + C0_SIZE - 1) / C0_SIZE * C0_SIZE;
        qSizeAlign = (fBaseParams.qSize / fBaseParams.d) * dAlign;
        kvSizeAlign = (fBaseParams.kvSize / fBaseParams.d) * dAlign;
    }
    fBaseParams.qSizeAlign = qSizeAlign;
    fBaseParams.kvSizeAlign = kvSizeAlign;
    fBaseParams.k1v1SizeAlign = k1v1SizeAlign;

    fBaseParams.scaleValue = *(context_->GetAttrs()->GetAttrPointer<float>(0));
    fBaseParams.keepProb = 1;
    OP_CHECK_IF((fBaseParams.keepProb <= 0 || fBaseParams.keepProb > 1), OP_LOGE(context_, "keepProb is illegal."),
                return ge::GRAPH_FAILED);

    fBaseParams.dropoutIsDivisibleBy8 = 1;

    // token_info
    fBaseParams.s1Token = 65536;
    fBaseParams.s2Token = 65536;
    fBaseParams.sparseMode = NO_MASK;
    fBaseParams.attenMaskOptional = EMPTY_TENSOR;
    auto attenMask = context_->GetOptionalInputDesc(ATTEN_MASK);
    if (attenMask != nullptr) {
        fBaseParams.attenMaskOptional = NORMAL_TENSOR;
        auto attenMaskType = attenMask->GetDataType();
        OP_CHECK_IF(attenMaskType != ge::DT_BOOL && attenMaskType != ge::DT_UINT8,
                    OP_LOGE(context_, "invalid attenMask dtype[%s], only support bool or uint8.",
                            ge::TypeUtils::DataTypeToSerialString(attenMaskType).c_str()),
                    return ge::GRAPH_FAILED);
        fBaseParams.attenMaskDtype = ATTEN_MASK_TYPE_U8_BOOL;
    }

    fBaseParams.isSparse = false;
    OP_LOGD(context_, "FAG S1s2Bn2gs1s2 sparse mode = %u, sparse %s.", fBaseParams.sparseMode,
            fBaseParams.isSparse ? "enable" : "disable");

    return CheckInputShapeValid(context_, fBaseParams.b, fBaseParams.n2, fBaseParams.g, fBaseParams.s1, fBaseParams.s2,
                                fBaseParams.d);
}

ge::graphStatus FusedFloydAttentionGradTilingS1s2Bn2gs1s2::DoOpTiling()
{
    auto ret = DoSplit();
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGW(context_, "get DoSplit fail."), return ret);

    ret = DoSparse();
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGW(context_, "get DoSparse fail."), return ret);

    ret = DoPreTiling();
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGW(context_, "get DoPreTiling fail."), return ret);

    ret = DoPostTiling();
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGW(context_, "get DoPostTiling fail."), return ret);
    DetermineMode();

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedFloydAttentionGradTilingS1s2Bn2gs1s2::DoSplit()
{
    fBaseParams.s1CvRatio = 1; // 1
    fBaseParams.s2CvRatio = 1; // 8
    if (fBaseParams.d == 64) { // d size is 64
        fBaseParams.s2CvRatio = CV_RATIO_16;
        if (fBaseParams.s1 >= 256) {     // 256 is s1 size
            if (fBaseParams.s2 <= 128) { // 128 is s2 size
                fBaseParams.s1CvRatio = CV_RATIO_4;
                fBaseParams.s2CvRatio = CV_RATIO_2;
            }
        }
    }
    fBaseParams.s1CvRatio = 1; // 1
    fBaseParams.s2CvRatio = 1; // 8

    uint32_t s1Inner = 64;
    uint32_t s2Inner = 128;
    fBaseParams.gInner = 16;
    fBaseParams.bmmS1base = 8;
    if (fBaseParams.d == 32) {
        fBaseParams.bmmS1base = 16;
    }

    uint32_t tmpBufferSize = (fBaseParams.ubSize - s1Inner * s2Inner * BASIC_BLOCK_MULTIPLE -
                              s1Inner * SHAPE_INFO * fBaseParams.calTypeSize) /
                             BYTE_BLOCK * BYTE_BLOCK;
    if (fBaseParams.mm1IsNZOut) {
        tmpBufferSize = tmpBufferSize - TEMP_BUFFER_REMAIN_SIZE;
    }
    fBaseParams.tmpBufferSize = tmpBufferSize;

    uint32_t s1CvInner = s1Inner * fBaseParams.s1CvRatio;
    OP_CHECK_IF(s1CvInner == 0, OP_LOGE(context_, "divisor s1CvInner is 0."), return ge::GRAPH_FAILED);
    int64_t s1Outer = (fBaseParams.s1 + s1CvInner - 1) / s1CvInner;
    uint32_t s1TailTmp = fBaseParams.s1 % s1Inner;
    uint32_t s1CvTailTmp = fBaseParams.s1 % s1CvInner;
    fBaseParams.s1Tail = s1TailTmp == 0 ? s1Inner : s1TailTmp;
    fBaseParams.s1CvTail = s1CvTailTmp == 0 ? s1CvInner : s1CvTailTmp;
    fBaseParams.s1Inner = s1Inner;
    fBaseParams.s1CvInner = s1CvInner;
    fBaseParams.s1Outer = s1Outer;

    uint32_t cvS2Inner = s2Inner * fBaseParams.s2CvRatio;
    OP_CHECK_IF(cvS2Inner == 0, OP_LOGE(context_, "divisor cvS2Inner is 0."), return ge::GRAPH_FAILED);
    int64_t s2Outer = (fBaseParams.s2 + cvS2Inner - 1) / cvS2Inner;
    uint32_t s2TailTmp = fBaseParams.s2 % s2Inner;
    uint32_t s2CvTailTmp = fBaseParams.s2 % cvS2Inner;
    fBaseParams.s2Tail = s2TailTmp == 0 ? s2Inner : s2TailTmp;
    fBaseParams.s2CvTail = s2CvTailTmp == 0 ? cvS2Inner : s2CvTailTmp;
    fBaseParams.s2Outer = s2Outer;
    fBaseParams.cvS2Inner = cvS2Inner;
    fBaseParams.s2Inner = s2Inner;

    fBaseParams.gOuter = (fBaseParams.g + fBaseParams.gInner - 1) / fBaseParams.gInner;
    fBaseParams.gTail = fBaseParams.g - (fBaseParams.gOuter - 1) * fBaseParams.gInner;
    fBaseParams.baseMN = fBaseParams.gInner * s1Inner * s2Inner;

    OP_CHECK_IF((fBaseParams.baseMN == 0 || fBaseParams.s2Outer == 0 || fBaseParams.s1Outer == 0),
                OP_LOGE(context_, "baseMN or s2Outer or s1Outer is 0."), return ge::GRAPH_FAILED);

    uint32_t sfmgdInner = 128; // 先写死

    OP_CHECK_IF(sfmgdInner == 0, OP_LOGE(context_, "divisor sfmgdInner is 0."), return ge::GRAPH_FAILED);
    uint32_t sfmgdOuter = (fBaseParams.d + sfmgdInner - 1) / sfmgdInner;
    uint32_t sfmgdTailTmp = fBaseParams.d % sfmgdInner;
    uint32_t sfmgdTail = sfmgdTailTmp == 0 ? sfmgdInner : sfmgdTailTmp;
    fBaseParams.sfmgdOuter = sfmgdOuter;
    fBaseParams.sfmgdInner = sfmgdInner;
    fBaseParams.sfmgdTail = sfmgdTail;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedFloydAttentionGradTilingS1s2Bn2gs1s2::DoSparse()
{
    int64_t blockStarts[CORE_LIST_NUM];
    int64_t blockEnds[CORE_LIST_NUM];
    // block split
    int64_t fusedOuter = static_cast<int64_t>(fBaseParams.b) * fBaseParams.n2 * fBaseParams.gOuter *
                         fBaseParams.s1Outer * fBaseParams.s2Outer; // 总块数
    // coreNum前面已做非0校验
    int64_t blockFactor = (fusedOuter + fBaseParams.coreNum - 1) / fBaseParams.coreNum; // 单核块数
    // 除0校验
    OP_CHECK_IF(blockFactor == 0, OP_LOGE(context_, "divisor blockFactor is 0."), return ge::GRAPH_FAILED);
    int64_t blockOuter = (fusedOuter + blockFactor - 1) / blockFactor; // 实际核数
    // 防止下标越界
    OP_CHECK_IF(blockOuter > CORE_LIST_NUM, OP_LOGE(context_, "blockStarts and blockEnds array bound."),
                return ge::GRAPH_FAILED);

    fBaseParams.blockOuter = blockOuter;
    fBaseParams.blockFactor = blockFactor;

    for (int64_t i = 0; i < blockOuter; i++) {
        blockStarts[i] = blockFactor * i;
        blockEnds[i] = std::min(blockFactor * (i + 1), fusedOuter);
    }
    for (int64_t i = blockOuter; i < CORE_LIST_NUM; i++) {
        blockStarts[i] = 0;
        blockEnds[i] = 0;
    }

    std::copy(std::begin(blockStarts), std::end(blockStarts), std::begin(fBaseParams.blockStarts));
    std::copy(std::begin(blockEnds), std::end(blockEnds), std::begin(fBaseParams.blockEnds));

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedFloydAttentionGradTilingS1s2Bn2gs1s2::DoLibApiTiling()
{
    // mm tiling
    matmul_tiling::MatmulApiTiling mm1;
    matmul_tiling::MatmulApiTiling mm2;
    matmul_tiling::MatmulApiTiling mm3;
    matmul_tiling::DataType inputAType = matmul_tiling::DataType::DT_FLOAT;
    if (fBaseParams.mode == FP32) {
        inputAType = matmul_tiling::DataType::DT_FLOAT;
    } else {
        inputAType = matmul_tiling::DataType::DT_BFLOAT16;
    }
    mm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, inputAType, false);
    mm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, inputAType, true);
    mm1.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    int64_t mmS1 = fBaseParams.s1;
    if (fBaseParams.mm1IsNZOut) {
        mmS1 = fBaseParams.s1Inner * fBaseParams.s1CvRatio;
    }
    // format left[B, N2, G, S1, D] right[B, N2, 1, S2, D] result[B, N2, G, S1, S2]
    mm1.SetOrgShape(mmS1, fBaseParams.s2, fBaseParams.d);
    mm1.SetShape(fBaseParams.s1Inner * fBaseParams.s1CvRatio, fBaseParams.s2Inner * fBaseParams.s2CvRatio,
                 fBaseParams.d);
    mm1.SetBias(false);

    if (fBaseParams.cvS2Inner > 128) { // 128 for perf when s2 cv ratio
        if (fBaseParams.d > 64) {      // 64 for d
            uint32_t minBaseM = std::min(fBaseParams.s1CvInner, FIX_BASEMN_256);
            mm1.SetFixSplit(minBaseM, FIX_BASEMN_128, -1); // 128 for baseN
        } else {
            uint32_t minBaseM = std::min(fBaseParams.s1CvInner, FIX_BASEMN_128);
            mm1.SetFixSplit(minBaseM, FIX_BASEMN_256, -1); // 256 for baseN
        }
    } else {
        mm1.SetFixSplit(-1, -1, -1);
    }

    OP_CHECK_IF(mm1.GetTiling(tilingData.mm1TilingData) != 0, OP_LOGE(context_, "matmul1 tilingData get fail."),
                return ge::GRAPH_FAILED);
    SetMatmulTilingBufferInfo(tilingData.mm1TilingData);

    matmul_tiling::MatmulApiTiling bmm1k1;
    SetBmm1k1TilingInput(bmm1k1, inputAType);
    OP_CHECK_IF(bmm1k1.GetTiling(tilingData.bmm1TilingData) != 0, OP_LOGE(context_, "bmm1k1 tilingData get fail."),
                return ge::GRAPH_FAILED);
    SetMatmulTilingBufferInfo(tilingData.bmm1TilingData);

    matmul_tiling::MatmulApiTiling bmm2k1;
    SetBmm2k1TilingInput(bmm2k1, inputAType);
    OP_CHECK_IF(bmm2k1.GetTiling(tilingData.bmm2TilingData) != 0, OP_LOGE(context_, "bmm2k1 tilingData get fail."),
                return ge::GRAPH_FAILED);
    SetMatmulTilingBufferInfo(tilingData.bmm2TilingData);

    matmul_tiling::MatmulApiTiling bmm3k1v1;
    SetBmm3k1V1TilingInput(bmm3k1v1, inputAType);
    OP_CHECK_IF(bmm3k1v1.GetTiling(tilingData.bmm3TilingData) != 0, OP_LOGE(context_, "bmm3k1v1 tilingData get fail."),
                return ge::GRAPH_FAILED);
    SetMatmulTilingBufferInfo(tilingData.bmm3TilingData);

    // format left[B, N2, G, S1, S2] right[B, N2, G, S1, D] result[B, N2, G, S2, D]
    // matmal3/5
    mm2.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, inputAType, true);
    mm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, inputAType, false);
    auto outFormat = fBaseParams.mm2IsNZOut ? matmul_tiling::CubeFormat::NZ : matmul_tiling::CubeFormat::ND;
    mm2.SetCType(matmul_tiling::TPosition::GM, outFormat, matmul_tiling::DataType::DT_FLOAT);

    mm2.SetOrgShape(fBaseParams.s2, fBaseParams.d, fBaseParams.s1);

    mm2.SetShape(fBaseParams.s2Inner * fBaseParams.s2CvRatio, fBaseParams.d,
                 fBaseParams.s1Inner * fBaseParams.s1CvRatio);
    mm2.SetBias(false);

    mm2.SetFixSplit(-1, -1, -1);

    OP_CHECK_IF(mm2.GetTiling(tilingData.mm2TilingData) != 0, OP_LOGE(context_, "matmul2 tilingData get fail."),
                return ge::GRAPH_FAILED);
    SetMatmulTilingBufferInfo(tilingData.mm2TilingData);

    // format left[B, N2, G, S1, S2] right[B, N2, 1, S2, D] result[B, N2, G, S1, D]
    // matmal4
    mm3.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, inputAType, false);
    mm3.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, inputAType, false);
    mm3.SetCType(matmul_tiling::TPosition::GM, outFormat, matmul_tiling::DataType::DT_FLOAT);

    mm3.SetOrgShape(fBaseParams.s1, fBaseParams.d, fBaseParams.s2);

    mm3.SetShape(fBaseParams.s1Inner * fBaseParams.s1CvRatio, fBaseParams.d,
                 fBaseParams.s2Inner * fBaseParams.s2CvRatio);
    mm3.SetBias(false);
    if (fBaseParams.mm1IsNZOut && fBaseParams.mm2IsNZOut) {
        int64_t dAlign = (fBaseParams.d + FP16_BLOCK_NUMS - 1) / FP16_BLOCK_NUMS * FP16_BLOCK_NUMS;
        uint32_t minBaseN = std::min(static_cast<uint32_t>(dAlign), FIX_BASEMN_256);
        mm3.SetFixSplit(-1, minBaseN, -1);
    } else {
        mm3.SetFixSplit(-1, -1, -1);
    }
    OP_CHECK_IF(mm3.GetTiling(tilingData.mm3TilingData) != 0, OP_LOGE(context_, "matmul3 tilingData get fail."),
                return ge::GRAPH_FAILED);
    SetMatmulTilingBufferInfo(tilingData.mm3TilingData);

    uint32_t cvS2Inner = fBaseParams.s2Inner * fBaseParams.s2CvRatio;
    uint32_t s2VSize = cvS2Inner > 256 ? 256 : cvS2Inner;
    uint32_t s1VecSize =
        std::min(((INITIAL_S1_SPLIT_NUM * INITIAL_S2_SPLIT_NUM + s2VSize - 1) / s2VSize), fBaseParams.s1Inner);

    auto softmaxShape = ge::Shape({s1VecSize, s2VSize});
    AscendC::SoftMaxTilingFunc(softmaxShape, fBaseParams.calTypeSize, fBaseParams.tmpBufferSize,
                               tilingData.softmaxTilingData);
    AscendC::SoftMaxGradTilingFunc(softmaxShape, fBaseParams.calTypeSize, fBaseParams.tmpBufferSize,
                                   tilingData.softmaxGradTilingData, true);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedFloydAttentionGradTilingS1s2Bn2gs1s2::SetBmm1k1TilingInput(matmul_tiling::MatmulApiTiling& bmm1k1,
                                                                                matmul_tiling::DataType inputAType)
{
    // BNGS1D x BNS2S1D（中间做batch轴）, M=G， N = S2， K = D，batch = S1
    bmm1k1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, inputAType, false);
    bmm1k1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, inputAType, true);
    bmm1k1.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);

    bmm1k1.SetShape(fBaseParams.gInner, fBaseParams.cvS2Inner, fBaseParams.d); // 32 * 32 * 128
    bmm1k1.SetOrgShape(fBaseParams.g, fBaseParams.s2, fBaseParams.d);
    bmm1k1.SetBias(false);
    bmm1k1.SetBufferSpace(-1, -1, -1);
    // B S N G D   SBNGD
    bmm1k1.SetALayout(1, fBaseParams.g, 1, fBaseParams.s1, fBaseParams.d);
    bmm1k1.SetBLayout(1, fBaseParams.s2, 1, fBaseParams.s1, fBaseParams.d);
    bmm1k1.SetCLayout(1, fBaseParams.gInner, 1, fBaseParams.s1CvInner, fBaseParams.cvS2Inner);
    bmm1k1.SetBatchNum(fBaseParams.bmmS1base);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedFloydAttentionGradTilingS1s2Bn2gs1s2::SetBmm2k1TilingInput(matmul_tiling::MatmulApiTiling& bmm2k1,
                                                                                matmul_tiling::DataType inputAType)
{
    // BNGS1S2 x BNS2S1D（中间做batch轴）= BNGS1D, M=G， N = D， K = S2，batch = S1
    bmm2k1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, inputAType, false);
    bmm2k1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, inputAType, false);
    bmm2k1.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);

    // GS2 x S2D = GD
    bmm2k1.SetShape(fBaseParams.gInner, fBaseParams.d, fBaseParams.cvS2Inner); // M=g, N=D, k = S1 batch = s1
    bmm2k1.SetOrgShape(fBaseParams.g, fBaseParams.d, fBaseParams.s2);
    bmm2k1.SetBias(false);
    bmm2k1.SetBufferSpace(-1, -1, -1);
    // 输入顺序B S N G D   SBNGD
    bmm2k1.SetALayout(1, fBaseParams.gInner, 1, fBaseParams.s1CvInner, fBaseParams.cvS2Inner);
    bmm2k1.SetBLayout(1, fBaseParams.s2, 1, fBaseParams.s1, fBaseParams.d);
    bmm2k1.SetCLayout(1, fBaseParams.g, 1, fBaseParams.s1, fBaseParams.d);
    bmm2k1.SetBatchNum(fBaseParams.bmmS1base);

    return true;
}

ge::graphStatus FusedFloydAttentionGradTilingS1s2Bn2gs1s2::SetBmm3k1V1TilingInput(
    matmul_tiling::MatmulApiTiling& bmm3k1v1, matmul_tiling::DataType inputAType)
{
    // BNGS1S2 x BNGS1D（中间做batch轴）= BNS2S1D,  M=S2， N = D， K = G，batch = S1
    bmm3k1v1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, inputAType, true);
    bmm3k1v1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, inputAType, false);
    bmm3k1v1.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);

    // GS2 x GD = S2D
    bmm3k1v1.SetShape(fBaseParams.cvS2Inner, fBaseParams.d, fBaseParams.gInner); // M=S2， N = D， K = G，batch = S1
    bmm3k1v1.SetOrgShape(fBaseParams.s2, fBaseParams.d, fBaseParams.g);
    bmm3k1v1.SetBias(false);
    bmm3k1v1.SetBufferSpace(-1, -1, -1);
    // 输入顺序B S N G D   SBNGD
    bmm3k1v1.SetALayout(1, fBaseParams.gInner, 1, fBaseParams.s1CvInner, fBaseParams.cvS2Inner); // BNGS1S2
    bmm3k1v1.SetBLayout(1, fBaseParams.g, 1, fBaseParams.s1, fBaseParams.d);                     // BNGS1D
    bmm3k1v1.SetCLayout(1, fBaseParams.s2, 1, fBaseParams.s1, fBaseParams.d);                    // BNS2S1D
    bmm3k1v1.SetBatchNum(fBaseParams.bmmS1base);

    return true;
}

void FusedFloydAttentionGradTilingS1s2Bn2gs1s2::SetMatmulTilingBufferInfo(TCubeTiling& mmTiling)
{
    mmTiling.set_shareMode(0);
    mmTiling.set_shareL1Size(fBaseParams.l1Size);
    mmTiling.set_shareL0CSize(fBaseParams.l0cSize);
}

ge::graphStatus FusedFloydAttentionGradTilingS1s2Bn2gs1s2::GetWorkspaceSize()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    size_t workspaceSize = MUL_CORE_SYNC_BUFFER;
    uint32_t s1Inner = std::min(INITIAL_S1_SPLIT_NUM, fBaseParams.s1Align);
    OP_CHECK_IF(s1Inner <= 0, OP_LOGE(context_, "s1Inner is less than or equal to 0, s1Inner is %u.", s1Inner),
                return ge::GRAPH_FAILED);

    // matmal3 q
    workspaceSize =
        (workspaceSize + static_cast<size_t>(fBaseParams.qSizeAlign) * FP32_BYTES + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    // matmal3 k
    workspaceSize =
        (workspaceSize + static_cast<size_t>(fBaseParams.kvSizeAlign) * FP32_BYTES + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    // matmal3 v
    workspaceSize =
        (workspaceSize + static_cast<size_t>(fBaseParams.kvSizeAlign) * FP32_BYTES + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    // mask bool workspace size
    // matmal3 k1
    workspaceSize =
        (workspaceSize + static_cast<size_t>(fBaseParams.k1v1SizeAlign) * FP32_BYTES + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    // matmal3 v1
    workspaceSize =
        (workspaceSize + static_cast<size_t>(fBaseParams.k1v1SizeAlign) * FP32_BYTES + GM_ALIGN) / GM_ALIGN * GM_ALIGN;

    if (fBaseParams.dropoutIsDivisibleBy8 == 0) {
        workspaceSize =
            (workspaceSize + static_cast<size_t>(fBaseParams.dropMaskSize) + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    }
    // matmal1/matmal2 workspace size
    size_t vectorCoreNum = fBaseParams.coreNum;
    workspaceSize = (workspaceSize +
                     vectorCoreNum * fBaseParams.gInner * fBaseParams.s1CvRatio * fBaseParams.s2CvRatio *
                         fBaseParams.baseMN * FP32_BYTES * MATMAL_INPUT_NUMS +
                     GM_ALIGN) /
                    GM_ALIGN * GM_ALIGN;
    // CV ratio workspace size fp16
    // drop workspace size
    workspaceSize = (workspaceSize +
                     vectorCoreNum * fBaseParams.gInner * fBaseParams.s1CvRatio * fBaseParams.s2CvRatio *
                         fBaseParams.baseMN * fBaseParams.dataTypeSize * 2 // 2 means pingpong
                     + GM_ALIGN) /
                    GM_ALIGN * GM_ALIGN;
    // mul workspace size
    workspaceSize = (workspaceSize +
                     vectorCoreNum * fBaseParams.gInner * fBaseParams.s1CvRatio * fBaseParams.s2CvRatio *
                         fBaseParams.baseMN * fBaseParams.dataTypeSize * 2 // 2 means pingpong
                     + GM_ALIGN) /
                    GM_ALIGN * GM_ALIGN;
    workspaceSize += WORKSPACE_BUFFER;
    workspaces[0] = workspaceSize;

    if (fBaseParams.pseType == PSE_INNER_MUL_ADD_TYPE || fBaseParams.pseType == PSE_INNER_MUL_ADD_SQRT_TYPE) {
        fBaseParams.pseAlibiBaseS2 = PSE_ALIBI_S2_LIMIT_SIZE;
        int64_t s2Tail = fBaseParams.s2 % PSE_ALIBI_S2_LIMIT_SIZE;
        if (s2Tail != 0) {
            fBaseParams.pseAlibiBaseS1 =
                std::min(static_cast<int64_t>(s1Inner), UB_BASIC_LIMIT_SIZE / AlignUp(s2Tail, FRACTAL_NUM));
        } else {
            fBaseParams.pseAlibiBaseS1 =
                std::min(static_cast<int64_t>(s1Inner), UB_BASIC_LIMIT_SIZE / fBaseParams.pseAlibiBaseS2);
        }
        fBaseParams.pseAlibiBaseS1 = std::max(fBaseParams.pseAlibiBaseS1, UB_BASIC_LIMIT_SIZE / s1Inner);
        int64_t pseAlibiBytes =
            AlignUp(fBaseParams.pseAlibiBaseS2 * fBaseParams.pseAlibiBaseS1 * 2, GM_ALIGN) * fBaseParams.coreNum;
        workspaces[0] += pseAlibiBytes;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedFloydAttentionGradTilingS1s2Bn2gs1s2::PostTiling()
{
    SaveToTilingData();
    auto blockdim = CalcTschBlockDim(tilingData.s1s2BNGS1S2SplitCoreParams.get_blockOuter(), fBaseParams.aicNum,
                                     fBaseParams.coreNum);
    OP_CHECK_IF(
        blockdim == 0,
        OP_LOGE(context_, "blockdim is 0, aicNum is %ld, aivNum is %ld.", fBaseParams.aicNum, fBaseParams.coreNum),
        return ge::GRAPH_FAILED);
    context_->SetBlockDim(blockdim);
    context_->SetScheduleMode(1);

    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedFloydAttentionGradTilingS1s2Bn2gs1s2::SaveToTilingData()
{
    tilingData.s1s2BNGS1S2BaseParams.set_coreNum(fBaseParams.coreNum);

    // set tilingdata baseinfo
    tilingData.s1s2BNGS1S2BaseParams.set_b(fBaseParams.b);
    tilingData.s1s2BNGS1S2BaseParams.set_n2(fBaseParams.n2);
    tilingData.s1s2BNGS1S2BaseParams.set_g(fBaseParams.g);
    tilingData.s1s2BNGS1S2BaseParams.set_s1(fBaseParams.s1);
    tilingData.s1s2BNGS1S2BaseParams.set_d(fBaseParams.d);
    tilingData.s1s2BNGS1S2BaseParams.set_s2(fBaseParams.s2);

    tilingData.s1s2BNGS1S2BaseParams.set_pseOptional(fBaseParams.pseOptional);
    tilingData.s1s2BNGS1S2BaseParams.set_pseType(fBaseParams.pseType);
    tilingData.s1s2BNGS1S2BaseParams.set_pseShapeType(fBaseParams.pseShapeType);
    tilingData.s1s2BNGS1S2BaseParams.set_pseDtype(fBaseParams.pseDtype);
    tilingData.s1s2BNGS1S2BaseParams.set_attenMaskOptional(fBaseParams.attenMaskOptional);
    tilingData.s1s2BNGS1S2BaseParams.set_attenMaskShapeType(fBaseParams.attenMaskShapeType);
    tilingData.s1s2BNGS1S2BaseParams.set_attenMaskDtype(fBaseParams.attenMaskDtype);
    tilingData.s1s2BNGS1S2BaseParams.set_scaleValue(fBaseParams.scaleValue);
    tilingData.s1s2BNGS1S2BaseParams.set_keepProb(fBaseParams.keepProb);

    // fBaseParams.s1Token int64_t类型   tilingData.s1s2BNGS1S2BaseParams.s1Token  int32_t类型 防止溢出
    tilingData.s1s2BNGS1S2BaseParams.set_s1Token(fBaseParams.s1Token > INT32_MAX ? INT32_MAX : fBaseParams.s1Token);
    tilingData.s1s2BNGS1S2BaseParams.set_s2Token(fBaseParams.s2Token > INT32_MAX ? INT32_MAX : fBaseParams.s2Token);

    tilingData.s1s2BNGS1S2BaseParams.set_sparseMode(fBaseParams.sparseMode);
    tilingData.s1s2BNGS1S2BaseParams.set_isSparse(fBaseParams.isSparse);
    tilingData.s1s2BNGS1S2BaseParams.set_attenMaskS2Size(fBaseParams.attenMaskS2Size);
    tilingData.s1s2BNGS1S2BaseParams.set_attenMaskCompressMode(fBaseParams.attenMaskCompressMode);

    // s1/s2 split
    tilingData.s1s2BNGS1S2SplitCoreParams.set_s1CvRatio(fBaseParams.s1CvRatio);
    tilingData.s1s2BNGS1S2SplitCoreParams.set_s1Outer(fBaseParams.s1Outer);
    tilingData.s1s2BNGS1S2SplitCoreParams.set_s1Inner(fBaseParams.s1Inner);
    tilingData.s1s2BNGS1S2SplitCoreParams.set_s1CvInner(fBaseParams.s1CvInner);
    tilingData.s1s2BNGS1S2SplitCoreParams.set_s1Tail(fBaseParams.s1Tail);
    tilingData.s1s2BNGS1S2SplitCoreParams.set_s1CvTail(fBaseParams.s1CvTail);
    tilingData.s1s2BNGS1S2SplitCoreParams.set_s2Outer(fBaseParams.s2Outer);
    tilingData.s1s2BNGS1S2SplitCoreParams.set_s2CvRatio(fBaseParams.s2CvRatio);
    tilingData.s1s2BNGS1S2SplitCoreParams.set_s2Inner(fBaseParams.s2Inner);
    tilingData.s1s2BNGS1S2SplitCoreParams.set_s2Tail(fBaseParams.s2Tail);
    tilingData.s1s2BNGS1S2SplitCoreParams.set_sfmgdOuter(fBaseParams.sfmgdOuter);
    tilingData.s1s2BNGS1S2SplitCoreParams.set_sfmgdFactor(fBaseParams.sfmgdInner);
    tilingData.s1s2BNGS1S2SplitCoreParams.set_sfmgdTail(fBaseParams.sfmgdTail);
    tilingData.s1s2BNGS1S2SplitCoreParams.set_gInner(fBaseParams.gInner);
    tilingData.s1s2BNGS1S2SplitCoreParams.set_gTail(fBaseParams.gTail);
    tilingData.s1s2BNGS1S2SplitCoreParams.set_gOuter(fBaseParams.gOuter);
    tilingData.s1s2BNGS1S2SplitCoreParams.set_bmmS1base(fBaseParams.bmmS1base);

    tilingData.s1s2BNGS1S2SplitCoreParams.set_baseMN(fBaseParams.baseMN);
    tilingData.s1s2BNGS1S2SplitCoreParams.set_bandIdx(fBaseParams.bandIdx);
    tilingData.s1s2BNGS1S2BlockNumList.set_blockStarts(fBaseParams.blockStarts);
    tilingData.s1s2BNGS1S2BlockNumList.set_blockEnds(fBaseParams.blockEnds);
    tilingData.s1s2BNGS1S2SplitCoreParams.set_blockOuter(fBaseParams.blockOuter);

    if (fBaseParams.pseType == PSE_INNER_MUL_ADD_TYPE || fBaseParams.pseType == PSE_INNER_MUL_ADD_SQRT_TYPE) {
        tilingData.s1s2BNGS1S2BaseParams.set_pseAlibiBaseS1(fBaseParams.pseAlibiBaseS1);
        tilingData.s1s2BNGS1S2BaseParams.set_pseAlibiBaseS2(fBaseParams.pseAlibiBaseS2);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedFloydAttentionGradTilingS1s2Bn2gs1s2::DoPreTiling()
{
    uint32_t castBufferLen = 60 * 1024;
    uint32_t outputBufferLen = 30 * 1024;
    uint32_t inputBufferLen = 4 * 1024;
    int64_t singleUBProcessNum = castBufferLen / 2;

    int64_t maskSize = AlignTo(fBaseParams.dropMaskSize, static_cast<int64_t>(BOOL_BLOCK_NUMS));
    int64_t singleCoreNum = AlignTo(CeilCommon(maskSize, static_cast<int64_t>(fBaseParams.blockOuter)),
                                    static_cast<int64_t>(BOOL_BLOCK_NUMS));
    int64_t maskUsedCoreNum = static_cast<int64_t>(CeilCommon(maskSize, singleCoreNum));

    int64_t tailCoreNum = maskSize - (maskUsedCoreNum - 1) * singleCoreNum;
    tailCoreNum = AlignTo(tailCoreNum, static_cast<int64_t>(BOOL_BLOCK_NUMS));

    int64_t singleCoreUBLoop = static_cast<int64_t>(CeilCommon(singleCoreNum, singleUBProcessNum));
    int64_t tailCoreUBLoop = static_cast<int64_t>(CeilCommon(tailCoreNum, singleUBProcessNum));

    int64_t singleCoreUBLastLoopNum = static_cast<int64_t>(singleCoreNum - (singleCoreUBLoop - 1) * singleUBProcessNum);
    int64_t tailCoreUBLastLoopNum = static_cast<int64_t>(tailCoreNum - (tailCoreUBLoop - 1) * singleUBProcessNum);

    tilingData.preTilingData.set_maskCoreNum(maskUsedCoreNum);
    tilingData.preTilingData.set_castBufferLen(castBufferLen);
    tilingData.preTilingData.set_outputBufferLen(outputBufferLen);
    tilingData.preTilingData.set_inputBufferLen(inputBufferLen);
    tilingData.preTilingData.set_singleUBProcessNum(static_cast<int64_t>(singleUBProcessNum));
    tilingData.preTilingData.set_maskSingleCoreNum(singleCoreNum); // size == num
    tilingData.preTilingData.set_maskSingleCoreLoop(singleCoreUBLoop);
    tilingData.preTilingData.set_maskLastLoopNum(singleCoreUBLastLoopNum);
    tilingData.preTilingData.set_maskTailCoreLoop(tailCoreUBLoop);
    tilingData.preTilingData.set_maskTailCoreLastLoopNum(tailCoreUBLastLoopNum);

    OP_CHECK_IF(maskUsedCoreNum == 0, OP_LOGE(context_, "divisor maskUsedCoreNum is 0."), return ge::GRAPH_FAILED);
    int64_t qPreBlockFactor = (fBaseParams.qSizeAlign + maskUsedCoreNum - 1) / maskUsedCoreNum;
    OP_CHECK_IF(qPreBlockFactor == 0, OP_LOGE(context_, "divisor qPreBlockFactor is 0."), return ge::GRAPH_FAILED);
    int64_t qPreBlockTotal = (fBaseParams.qSizeAlign + qPreBlockFactor - 1) / qPreBlockFactor;
    int64_t qPreTailNumTmp = fBaseParams.qSizeAlign % qPreBlockFactor;
    int64_t qPreTailNum = qPreTailNumTmp == 0 ? qPreBlockFactor : qPreTailNumTmp;

    int64_t kvPreBlockFactor = (fBaseParams.kvSizeAlign + maskUsedCoreNum - 1) / maskUsedCoreNum;
    OP_CHECK_IF(kvPreBlockFactor == 0, OP_LOGE(context_, "divisor kvPreBlockFactor is 0."), return ge::GRAPH_FAILED);
    int64_t kvPreBlockTotal = (fBaseParams.kvSizeAlign + kvPreBlockFactor - 1) / kvPreBlockFactor;
    int64_t kvPreTailNumTmp = fBaseParams.kvSizeAlign % kvPreBlockFactor;
    int64_t kvPreTailNum = kvPreTailNumTmp == 0 ? kvPreBlockFactor : kvPreTailNumTmp;

    int64_t k1v1PreBlockFactor = (fBaseParams.k1v1SizeAlign + maskUsedCoreNum - 1) / maskUsedCoreNum;
    OP_CHECK_IF(k1v1PreBlockFactor == 0, OP_LOGE(context_, "divisor k1v1PreBlockFactor is 0."),
                return ge::GRAPH_FAILED);
    int64_t k1v1PreBlockTotal = (fBaseParams.k1v1SizeAlign + k1v1PreBlockFactor - 1) / k1v1PreBlockFactor;
    int64_t k1v1PreTailNumTmp = fBaseParams.k1v1SizeAlign % k1v1PreBlockFactor;
    int64_t k1v1PreTailNum = k1v1PreTailNumTmp == 0 ? k1v1PreBlockFactor : k1v1PreTailNumTmp;

    int64_t maskPreBlockTotal = (fBaseParams.dropMaskSize);
    tilingData.preTilingData.set_qPreBlockFactor(qPreBlockFactor);
    tilingData.preTilingData.set_qPreBlockTotal(qPreBlockTotal);
    tilingData.preTilingData.set_qPreBlockTail(qPreTailNum);
    tilingData.preTilingData.set_kvPreBlockFactor(kvPreBlockFactor);
    tilingData.preTilingData.set_kvPreBlockTotal(kvPreBlockTotal);
    tilingData.preTilingData.set_kvPreBlockTail(kvPreTailNum);
    tilingData.preTilingData.set_k1v1PreBlockFactor(k1v1PreBlockFactor);
    tilingData.preTilingData.set_k1v1PreBlockTotal(k1v1PreBlockTotal);
    tilingData.preTilingData.set_k1v1PreBlockTail(k1v1PreTailNum);
    tilingData.preTilingData.set_dropoutIsDivisibleBy8(fBaseParams.dropoutIsDivisibleBy8);
    tilingData.preTilingData.set_maskPreBlockTotal(maskPreBlockTotal);

    int64_t dropBeginAddr = SYNC_GLOBAL_WORKSPACE_SIZE;
    dropBeginAddr =
        (dropBeginAddr + (fBaseParams.qSize) * sizeof(float) + ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
    dropBeginAddr =
        (dropBeginAddr + (fBaseParams.kvSize) * sizeof(float) + ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
    dropBeginAddr =
        (dropBeginAddr + (fBaseParams.kvSize) * sizeof(float) + ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
    tilingData.preTilingData.set_dropBeginAddr(dropBeginAddr);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedFloydAttentionGradTilingS1s2Bn2gs1s2::DoPostTiling()
{
    int64_t dAlign = (fBaseParams.d + FP16_BLOCK_NUMS - 1) / FP16_BLOCK_NUMS * FP16_BLOCK_NUMS;
    int64_t curPostCoexNode = fBaseParams.mm2IsNZOut ? POST_NZ_COEX_NODE : POST_COEX_NODE;
    int64_t nzReservedSize = fBaseParams.mm2IsNZOut ? dAlign / C0_SIZE * BLOCK_SIZE * POST_NZ_RESERVED_N : 0;
    int64_t postUbBaseSize = (fBaseParams.ubSize - 2 * nzReservedSize) / curPostCoexNode /
                             BUFFER_NUM / // 开DB预留2份nzReservedSize
                             WORKSPACE_NUM_ALIGN * WORKSPACE_NUM_ALIGN;

    int64_t qPostBaseNum = fBaseParams.mm2IsNZOut ?
                               (postUbBaseSize / fBaseParams.dataTypeSize / dAlign * fBaseParams.d) :
                               (postUbBaseSize / fBaseParams.dataTypeSize);
    OP_CHECK_IF(qPostBaseNum == 0, OP_LOGE(context_, "divisor qPostBaseNum is 0."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(fBaseParams.blockOuter == 0, OP_LOGE(context_, "divisor fBaseParams.blockOuter is 0."),
                return ge::GRAPH_FAILED);
    int64_t qPostBlockTotal = fBaseParams.qSize;
    int64_t qPostTailNumTmp = qPostBlockTotal % qPostBaseNum;
    int64_t qPostTailNum = qPostTailNumTmp == 0 ? qPostBaseNum : qPostTailNumTmp;
    int64_t qPostBlockOuterTotal = (qPostBlockTotal + qPostBaseNum - 1) / qPostBaseNum;
    int64_t qPostBlockFactor = (qPostBlockOuterTotal + fBaseParams.blockOuter - 1) / fBaseParams.blockOuter;

    int64_t kvPostBaseNum = qPostBaseNum;
    OP_CHECK_IF(kvPostBaseNum == 0, OP_LOGE(context_, "divisor kvPostBaseNum is 0."), return ge::GRAPH_FAILED);
    int64_t kvPostBlockTotal = fBaseParams.kvSize;
    int64_t kvPostTailNumTmp = kvPostBlockTotal % kvPostBaseNum;
    int64_t kvPostTailNum = kvPostTailNumTmp == 0 ? kvPostBaseNum : kvPostTailNumTmp;
    int64_t kvPostBlockOuterTotal = (kvPostBlockTotal + kvPostBaseNum - 1) / kvPostBaseNum;
    int64_t kvPostBlockFactor = (kvPostBlockOuterTotal + fBaseParams.blockOuter - 1) / fBaseParams.blockOuter;

    int64_t k1v1PostBaseNum = qPostBaseNum;
    OP_CHECK_IF(k1v1PostBaseNum == 0, OP_LOGE(context_, "divisor k1v1PostBaseNum is 0."), return ge::GRAPH_FAILED);
    int64_t k1v1PostBlockTotal = fBaseParams.k1v1Size;
    int64_t k1v1PostTailNumTmp = k1v1PostBlockTotal % k1v1PostBaseNum;
    int64_t k1v1PostTailNum = k1v1PostTailNumTmp == 0 ? k1v1PostBaseNum : k1v1PostTailNumTmp;
    int64_t k1v1PostBlockOuterTotal = (k1v1PostBlockTotal + k1v1PostBaseNum - 1) / k1v1PostBaseNum;
    int64_t k1v1PostBlockFactor = (k1v1PostBlockOuterTotal + fBaseParams.blockOuter - 1) / fBaseParams.blockOuter;

    tilingData.postTilingData.set_scaleValue(fBaseParams.scaleValue);
    tilingData.postTilingData.set_coreNum(fBaseParams.coreNum);
    tilingData.postTilingData.set_postUbBaseSize(postUbBaseSize);
    tilingData.postTilingData.set_nzReservedSize(nzReservedSize);
    tilingData.postTilingData.set_qPostBlockFactor(qPostBlockFactor);
    tilingData.postTilingData.set_qPostBlockTotal(qPostBlockTotal);
    tilingData.postTilingData.set_qPostBaseNum(qPostBaseNum);
    tilingData.postTilingData.set_qPostTailNum(qPostTailNum);
    tilingData.postTilingData.set_qSizeAlign(fBaseParams.qSizeAlign);

    tilingData.postTilingData.set_kvPostBlockFactor(kvPostBlockFactor);
    tilingData.postTilingData.set_kvPostBlockTotal(kvPostBlockTotal);
    tilingData.postTilingData.set_kvPostBaseNum(kvPostBaseNum);
    tilingData.postTilingData.set_kvPostTailNum(kvPostTailNum);
    tilingData.postTilingData.set_kvSizeAlign(fBaseParams.kvSizeAlign);

    tilingData.postTilingData.set_k1v1PostBlockFactor(k1v1PostBlockFactor);
    tilingData.postTilingData.set_k1v1PostBlockTotal(k1v1PostBlockTotal);
    tilingData.postTilingData.set_k1v1PostBaseNum(k1v1PostBaseNum);
    tilingData.postTilingData.set_k1v1PostTailNum(k1v1PostTailNum);
    tilingData.postTilingData.set_k1v1SizeAlign(fBaseParams.k1v1SizeAlign);

    int64_t workspaceOffsets = MUL_CORE_SYNC_BUFFER;
    tilingData.postTilingData.set_dqWorkSpaceOffset(workspaceOffsets);

    workspaceOffsets = (workspaceOffsets + fBaseParams.qSizeAlign * sizeof(float) + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    tilingData.postTilingData.set_dkWorkSpaceOffset(workspaceOffsets);

    workspaceOffsets = (workspaceOffsets + fBaseParams.kvSizeAlign * sizeof(float) + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    tilingData.postTilingData.set_dvWorkSpaceOffset(workspaceOffsets);

    workspaceOffsets = (workspaceOffsets + fBaseParams.kvSizeAlign * sizeof(float) + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    tilingData.postTilingData.set_dk1WorkSpaceOffset(workspaceOffsets);

    workspaceOffsets = (workspaceOffsets + fBaseParams.k1v1SizeAlign * sizeof(float) + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    tilingData.postTilingData.set_dv1WorkSpaceOffset(workspaceOffsets);

    tilingData.postTilingData.set_b(fBaseParams.b);
    tilingData.postTilingData.set_n2(fBaseParams.n2);
    tilingData.postTilingData.set_g(fBaseParams.g);
    tilingData.postTilingData.set_s1(fBaseParams.s1);
    tilingData.postTilingData.set_s2(fBaseParams.s2);
    tilingData.postTilingData.set_d(fBaseParams.d);

    return ge::GRAPH_SUCCESS;
}

void FusedFloydAttentionGradTilingS1s2Bn2gs1s2::DetermineMode()
{
    // 当前fp16都走高精度
    if (fBaseParams.queryType == ge::DT_FLOAT) {
        fBaseParams.mode = FP32;
    } else if (fBaseParams.queryType == ge::DT_BF16) {
        fBaseParams.mode = BF16;
    } else if (fBaseParams.queryType == ge::DT_FLOAT16) {
        fBaseParams.mode = INHP;
    } else {
        fBaseParams.mode = FP16;
    }
}

REGISTER_OPS_TILING_TEMPLATE(FusedFloydAttentionGrad, FusedFloydAttentionGradTilingS1s2Bn2gs1s2, 16000);
} // namespace FFAG
} // namespace optiling
