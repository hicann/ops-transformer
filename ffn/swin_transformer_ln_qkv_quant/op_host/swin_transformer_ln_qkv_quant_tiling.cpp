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
 * \file swin_transformer_ln_qkv_quant_tiling.cpp
 * \brief
 */
#include "register/op_def_registry.h"
#include "platform/platform_info.h"
#include "swin_transformer_ln_qkv_quant_tiling.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "log/log.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_base.h"
#include "util/math_util.h"

using namespace std;
using namespace ge;
using namespace matmul_tiling;
namespace optiling {
constexpr static uint32_t USE_CORE_THRESHOLD = 8;
constexpr static uint32_t BUFFER_NUM = 2;
constexpr static uint32_t BLOCK_UINT_16 = 16;
constexpr static uint32_t RESVERD_BUFF = 64 * 1024;
class SwinTransformerLnQkvQuantTilingCompute {
public:
    enum class ProcessMode {
        NORMAL_MODE = 0UL,
        SPLIT_N_MODE,
        LN_PRE_ALL_SHARE_MODE,
        LN_INDEPENDT_MODE,
        INVALID_MODE
    };
    SwinTransformerLnQkvQuantTilingData tilingData;
    ge::graphStatus SwinTransformerLnQkvQuantTilingMainProc(gert::TilingContext* context);
    bool transposeB = true;
    int32_t sizePerHead = 32;  // 32 is Block Unit
    float epsilon = 0.000001f; // 0.000001 is default epsilon
    bool allPreProcessA = false;
    bool dbBuffer = (BUFFER_NUM > 1) ? true : false;
    ProcessMode templateId = ProcessMode::LN_INDEPENDT_MODE;
    int64_t bLength;
    int64_t sLength;
    int64_t hLength;
    int64_t headNum;
    int64_t oriHeight;
    int64_t oriWeight;
    int64_t hWinSize;
    int64_t wWinSize;
    bool bTransOpt;

protected:
    ge::graphStatus SwinTransformerLnQkvQuantSaveTilingData(gert::TilingContext* context);
    ge::graphStatus SwinTransformerLnQkvQuantSetWorkSpace(gert::TilingContext* context);
    ge::graphStatus SwinTransformerLnQkvQuantSetMatmulTilingData(gert::TilingContext* context, TCubeTiling mmtilingData,
                                                                 uint64_t l1SizePlatform, uint64_t l0SizePlatform);
    ge::graphStatus SwinTransformerLnQkvQuantSetMatmulMutiTilingData(gert::TilingContext* context,
                                                                     TCubeTiling mmtilingData);
    int32_t SwinTransformerLnQkvQuantGetMatmulTmpSize(gert::TilingContext* context, TCubeTiling& mmtilingData,
                                                      int32_t m, int32_t n, int32_t k);
    void SwinTransformerLnQkvQuantSetTilingKey(gert::TilingContext* context);
    int32_t SwinTransformerLnQkvQuantGetUsedUbSize(int32_t mSize, int32_t kSize, int32_t splitN);
    ge::graphStatus GetBaseParams(gert::TilingContext* context);
    ge::graphStatus IsSupport(gert::TilingContext* context);
}; // class SwinTransformerLnQkvQuantTilingCompute

ge::graphStatus SwinTransformerLnQkvQuantTilingCompute::SwinTransformerLnQkvQuantSaveTilingData(
    gert::TilingContext* context)
{
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwinTransformerLnQkvQuantTilingCompute::SwinTransformerLnQkvQuantSetWorkSpace(
    gert::TilingContext* context)
{
    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaces == nullptr, OP_LOGE("SwinTransformerLnQkvQuantTiling", "failed to get workspace size"),
                return ge::GRAPH_FAILED);
    workspaces[0] = sizePerHead * sizePerHead * sizePerHead * sizePerHead;
    return ge::GRAPH_SUCCESS;
}

const std::map<int32_t, SwinTransformerLnQkvQuantTilingCompute::ProcessMode> TEMPLATE_MAP = {
    {0, SwinTransformerLnQkvQuantTilingCompute::ProcessMode::INVALID_MODE},
    {1, SwinTransformerLnQkvQuantTilingCompute::ProcessMode::NORMAL_MODE},
    {2, SwinTransformerLnQkvQuantTilingCompute::ProcessMode::NORMAL_MODE},
    {3, SwinTransformerLnQkvQuantTilingCompute::ProcessMode::NORMAL_MODE},
    {4, SwinTransformerLnQkvQuantTilingCompute::ProcessMode::NORMAL_MODE},
    {5, SwinTransformerLnQkvQuantTilingCompute::ProcessMode::NORMAL_MODE},
    {6, SwinTransformerLnQkvQuantTilingCompute::ProcessMode::NORMAL_MODE},
    {7, SwinTransformerLnQkvQuantTilingCompute::ProcessMode::NORMAL_MODE},
    {8, SwinTransformerLnQkvQuantTilingCompute::ProcessMode::NORMAL_MODE},
    {9, SwinTransformerLnQkvQuantTilingCompute::ProcessMode::NORMAL_MODE},
    {10, SwinTransformerLnQkvQuantTilingCompute::ProcessMode::NORMAL_MODE},
    {11, SwinTransformerLnQkvQuantTilingCompute::ProcessMode::NORMAL_MODE},
    {12, SwinTransformerLnQkvQuantTilingCompute::ProcessMode::NORMAL_MODE},
    {13, SwinTransformerLnQkvQuantTilingCompute::ProcessMode::NORMAL_MODE},
    {14, SwinTransformerLnQkvQuantTilingCompute::ProcessMode::NORMAL_MODE},
    {15, SwinTransformerLnQkvQuantTilingCompute::ProcessMode::NORMAL_MODE},
    {16, SwinTransformerLnQkvQuantTilingCompute::ProcessMode::NORMAL_MODE},
    {17, SwinTransformerLnQkvQuantTilingCompute::ProcessMode::NORMAL_MODE},
    {18, SwinTransformerLnQkvQuantTilingCompute::ProcessMode::NORMAL_MODE},
    {19, SwinTransformerLnQkvQuantTilingCompute::ProcessMode::NORMAL_MODE},
    {20, SwinTransformerLnQkvQuantTilingCompute::ProcessMode::NORMAL_MODE},
    {21, SwinTransformerLnQkvQuantTilingCompute::ProcessMode::NORMAL_MODE},
    {22, SwinTransformerLnQkvQuantTilingCompute::ProcessMode::NORMAL_MODE},
    {23, SwinTransformerLnQkvQuantTilingCompute::ProcessMode::NORMAL_MODE},
    {24, SwinTransformerLnQkvQuantTilingCompute::ProcessMode::NORMAL_MODE},
    {25, SwinTransformerLnQkvQuantTilingCompute::ProcessMode::NORMAL_MODE},
    {26, SwinTransformerLnQkvQuantTilingCompute::ProcessMode::NORMAL_MODE},
    {27, SwinTransformerLnQkvQuantTilingCompute::ProcessMode::NORMAL_MODE},
    {28, SwinTransformerLnQkvQuantTilingCompute::ProcessMode::NORMAL_MODE},
    {29, SwinTransformerLnQkvQuantTilingCompute::ProcessMode::NORMAL_MODE},
    {30, SwinTransformerLnQkvQuantTilingCompute::ProcessMode::NORMAL_MODE},
    {31, SwinTransformerLnQkvQuantTilingCompute::ProcessMode::NORMAL_MODE},
    {32, SwinTransformerLnQkvQuantTilingCompute::ProcessMode::NORMAL_MODE},
};

/**********************************************
 *
 * [templatedId][A_Dtype][B_Dtype][C_Dtype][A_Trans][B_Trans]
 *
 ************************************************/
void SwinTransformerLnQkvQuantTilingCompute::SwinTransformerLnQkvQuantSetTilingKey(gert::TilingContext* context)
{
    uint64_t tilingKey = 0;
    tilingKey = static_cast<uint64_t>(templateId) + transposeB * 100000; // transposeB is 100000 bit
    context->SetTilingKey(tilingKey);
    OP_LOGI(context->GetNodeName(), "SwinTransformerLnQkvQuant tilingKey is : %ld.", tilingKey);
    return;
}

int32_t SwinTransformerLnQkvQuantTilingCompute::SwinTransformerLnQkvQuantGetUsedUbSize(int32_t mSize, int32_t kSize,
                                                                                       int32_t splitN)
{
    int32_t bufferForLn = (mSize + BLOCK_UINT_16 - 1) / BLOCK_UINT_16 * BLOCK_UINT_16;
    int32_t allUbSize = 0;
    if (splitN == 0) {
        allUbSize += bufferForLn * kSize * sizeof(uint16_t) * BUFFER_NUM; // 2 fp16 in out
        allUbSize += bufferForLn * kSize * sizeof(int8_t);                // int 8 A
        allUbSize += bufferForLn * kSize * sizeof(float) * BUFFER_NUM;    // fp32 in + tmpSub
    } else {
        bufferForLn = BLOCK_UINT_16;
        allUbSize += bufferForLn * kSize * sizeof(uint16_t) * BUFFER_NUM; // fp16 in out
        allUbSize += bufferForLn * kSize * sizeof(int8_t);                // int 8 A
        allUbSize += bufferForLn * kSize * sizeof(float) * BUFFER_NUM;    // fp32 in + tmpSub
    }
    return allUbSize;
}

ge::graphStatus SwinTransformerLnQkvQuantTilingCompute::SwinTransformerLnQkvQuantSetMatmulTilingData(
    gert::TilingContext* context, TCubeTiling mmtilingData, uint64_t l1SizePlatform, uint64_t l0CSizePlatform)
{
    auto platformInfo = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    matmul_tiling::MatmulApiTiling tiling(platformInfo);
    tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT8,
                    transposeB);
    tiling.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND,
                    matmul_tiling::DataType::DT_FLOAT16);
    tiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT32);
    tiling.SetAType(matmul_tiling::TPosition::TSCM, matmul_tiling::CubeFormat::NZ, matmul_tiling::DataType::DT_INT8);

    tiling.SetShape(tilingData.mmInfo.get_mmSizeM(), tilingData.mmInfo.get_mmSizeN(), tilingData.mmInfo.get_mmSizeK());
    tiling.SetOrgShape(tilingData.mmInfo.get_mmSizeM(), tilingData.mmInfo.get_mmSizeN(),
                       tilingData.mmInfo.get_mmSizeK());
    tiling.SetBias(true);
    tiling.SetBufferSpace(l1SizePlatform, l0CSizePlatform, -1);
    tiling.SetDequantType(matmul_tiling::DequantType::TENSOR);
    int32_t ret = tiling.GetTiling(mmtilingData);
    if (ret == -1) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

int32_t SwinTransformerLnQkvQuantTilingCompute::SwinTransformerLnQkvQuantGetMatmulTmpSize(gert::TilingContext* context,
                                                                                          TCubeTiling& mmtilingData,
                                                                                          int32_t m, int32_t n,
                                                                                          int32_t k)
{
    auto platformInfo = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    matmul_tiling::MatmulApiTiling tiling(platformInfo);
    tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT8,
                    transposeB);
    tiling.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND,
                    matmul_tiling::DataType::DT_FLOAT16);
    tiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT32);
    tiling.SetAType(matmul_tiling::TPosition::TSCM, matmul_tiling::CubeFormat::NZ, matmul_tiling::DataType::DT_INT8);

    tiling.SetShape(m, n, k);
    tiling.SetOrgShape(m, n, k);
    tiling.SetBias(true);
    tiling.SetBufferSpace(-1, -1, -1);
    tiling.SetDequantType(matmul_tiling::DequantType::TENSOR);
    int32_t ret = tiling.GetTiling(mmtilingData);
    if (ret == -1) {
        return ret;
    }
    return 0;
}

ge::graphStatus SwinTransformerLnQkvQuantTilingCompute::SwinTransformerLnQkvQuantSetMatmulMutiTilingData(
    gert::TilingContext* context, TCubeTiling mmTilingData)
{
    auto platformInfo = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    matmul_tiling::MatmulApiTiling tiling(platformInfo);
    tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT8,
                    transposeB);
    tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    tiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT32);
    tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT8);

    tiling.SetShape(tilingData.opBaseInfo.get_bSize() * tilingData.opBaseInfo.get_sSize(), tilingData.get_weightN(),
                    tilingData.get_weightK());
    tiling.SetOrgShape(tilingData.opBaseInfo.get_bSize() * tilingData.opBaseInfo.get_sSize(), tilingData.get_weightN(),
                       tilingData.get_weightK());
    tiling.SetBias(true);
    tiling.SetBufferSpace(-1, -1, -1);
    tiling.SetDequantType(matmul_tiling::DequantType::TENSOR);
    int32_t ret = tiling.GetTiling(mmTilingData);
    if (ret == -1) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwinTransformerLnQkvQuantTilingCompute::GetBaseParams(gert::TilingContext* context)
{
    const gert::StorageShape* x1Shape = context->GetInputShape(0);
    int32_t dataSize = 1;
    OP_CHECK_IF(x1Shape == nullptr, OP_LOGE("SwinTransformerLnQkvQuant", "The INPUT x1Shape param is null!"),
                return ge::GRAPH_FAILED);
    for (uint32_t i = 0; i < x1Shape->GetStorageShape().GetDimNum(); ++i) {
        dataSize *= x1Shape->GetStorageShape().GetDim(i);
    }
    tilingData.set_size(dataSize);
    bLength = x1Shape->GetStorageShape().GetDim(0);
    sLength = x1Shape->GetStorageShape().GetDim(1);
    hLength = x1Shape->GetStorageShape().GetDim(2); // 2 shape dim2

    const gert::StorageShape* weightShape = context->GetInputShape(3); // 3 is weight
    OP_CHECK_IF(weightShape == nullptr, OP_LOGE("SwinTransformerLnQkvQuant", "The INPUT weight param is null!"),
                return ge::GRAPH_FAILED);
    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE("SwinTransformerLnQkvQuant", "The attrs param is null!"),
                return ge::GRAPH_FAILED);
    size_t idx = 0;
    headNum = *attrs->GetAttrPointer<int64_t>(idx++);
    auto seqLength = attrs->GetAttrPointer<int64_t>(idx++);
    auto epsilonOpt = attrs->GetAttrPointer<float>(idx++);
    oriHeight = *attrs->GetAttrPointer<int64_t>(idx++);
    oriWeight = *attrs->GetAttrPointer<int64_t>(idx++);
    hWinSize = *attrs->GetAttrPointer<int64_t>(idx++);
    wWinSize = *attrs->GetAttrPointer<int64_t>(idx++);
    bTransOpt = *attrs->GetAttrPointer<bool>(idx++);
    sizePerHead = (seqLength != nullptr) ? (*seqLength) : sizePerHead;
    transposeB = bTransOpt;
    if ((transposeB == false) || (hWinSize < 7) || (hWinSize > 32) || // win [7 - 32]
        (wWinSize < 7) || (wWinSize > 32)) {                          // win [7 - 32]
        return ge::GRAPH_FAILED;
    }
    auto patchHeight = oriHeight / hWinSize;
    auto patchWeight = oriWeight / wWinSize;

    tilingData.opBaseInfo.set_hWinSize(hWinSize);
    tilingData.opBaseInfo.set_headNum(headNum);
    tilingData.opBaseInfo.set_wWinSize(wWinSize);
    tilingData.opBaseInfo.set_sizePerHead(sizePerHead);
    tilingData.set_epsilon((epsilonOpt != nullptr) ? (*epsilonOpt) : epsilon);

    tilingData.opBaseInfo.set_patchHeight(patchHeight);
    tilingData.opBaseInfo.set_patchWeight(patchWeight);
    if ((patchHeight == 0) || (patchWeight == 0)) {
        return ge::GRAPH_FAILED;
    }
    if (transposeB) {
        tilingData.set_weightK(weightShape->GetStorageShape().GetDim(1));
        tilingData.set_weightN(weightShape->GetStorageShape().GetDim(0));
    } else {
        tilingData.set_weightK(weightShape->GetStorageShape().GetDim(0));
        tilingData.set_weightN(weightShape->GetStorageShape().GetDim(1));
    }
    if ((sizePerHead != 32) && (sizePerHead != 64)) { // seqLengh = 32 / 64
        return ge::GRAPH_FAILED;
    }
    if ((hLength > 1024) || (bLength > 32)) { // K <= 1024, b <=32
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwinTransformerLnQkvQuantTilingCompute::IsSupport(gert::TilingContext* context)
{
    if (bLength <= 0) {
        return ge::GRAPH_FAILED;
    }
    if (oriHeight * oriWeight != sLength) {
        OP_LOGW(context, "SwinTransformerLnQkvQuant is not support this input shape.");
        return ge::GRAPH_FAILED;
    }
    if ((oriHeight % hWinSize != 0) || (oriWeight % wWinSize != 0)) {
        return ge::GRAPH_FAILED;
    }
    if (transposeB == false) {
        return ge::GRAPH_FAILED;
    }
    if (sLength * hLength == 0) {
        return ge::GRAPH_FAILED;
    }
    if ((headNum < 1) || (headNum > 32)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwinTransformerLnQkvQuantTilingCompute::SwinTransformerLnQkvQuantTilingMainProc(
    gert::TilingContext* context)
{
    auto platformInfo = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto checkRet = GetBaseParams(context);
    if (checkRet != ge::GRAPH_SUCCESS) {
        OP_LOGE("SwinTransformerLnQkvQuantTilingProc", "inputPara is invalid!");
        return ge::GRAPH_FAILED;
    }
    if (IsSupport(context) != ge::GRAPH_SUCCESS) {
        OP_LOGE("SwinTransformerLnQkvQuantTilingProc", "inputPara is not support!");
        return ge::GRAPH_FAILED;
    }
    tilingData.opBaseInfo.set_bSize(bLength);
    tilingData.opBaseInfo.set_sSize(sLength);
    tilingData.opBaseInfo.set_hSize(hLength);
    uint32_t coreNum = USE_CORE_THRESHOLD;
    uint64_t ubSizePlatform;
    uint64_t l1SizePlatForm;
    uint64_t l0CSizePlatForm;
    uint64_t l0ASizePlatForm;

    coreNum = (platformInfo.GetCoreNum() != 0) ? platformInfo.GetCoreNum() : coreNum;
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1SizePlatForm);
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0CSizePlatForm);
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0ASizePlatForm);

    int64_t ubSize = ubSizePlatform;
    int64_t l1Size = l1SizePlatForm;
    int64_t l0CSize = l0CSizePlatForm;
    auto weightN = tilingData.get_weightN();
    if (weightN == 0) {
        return ge::GRAPH_FAILED;
    }
    std::vector<int64_t> shape_vec = {1, wWinSize, hLength};

    uint32_t typeSize = sizeof(uint16_t);

    uint32_t maxValueQuant = 0;
    uint32_t minValueQuant = 0;

    int64_t maxUbSizeForLn = ubSize - 6 * 1024; // resverd 6 * 1024

    int64_t blockNum = coreNum;
    auto iter = TEMPLATE_MAP.find(headNum);
    templateId = (iter != TEMPLATE_MAP.end()) ? iter->second : ProcessMode::LN_INDEPENDT_MODE;
    bool solutionFlag = false;
    bool lnSolutionFlag = false;
    switch (templateId) {
        case ProcessMode::NORMAL_MODE: {
            int64_t lnBsSize = bLength * sLength;
            blockNum = lnBsSize / wWinSize;
            blockNum = (blockNum <= coreNum) ? blockNum : coreNum;
            int64_t singleCoreLnBs = (lnBsSize + blockNum - 1) / blockNum;
            singleCoreLnBs = (singleCoreLnBs + wWinSize - 1) / (wWinSize) * (wWinSize);
            blockNum = (lnBsSize + singleCoreLnBs - 1) / singleCoreLnBs; // update blockNum after align to winSize
            blockNum = (blockNum <= coreNum) ? blockNum : coreNum;
            int64_t kSizePerLoop = hLength;
            int64_t nSizePerLoop = weightN;
            int64_t mSizePerLoop = wWinSize;
            int64_t maxMnSize = l0CSize / typeSize;                            // to Ub
            maxMnSize = (maxMnSize > (ubSize / 2)) ? (ubSize / 2) : maxMnSize; // 2 IS SPLIT FOR A AND B
            int32_t lnResverdBuffer = (hLength <= 256) ?
                                          RESVERD_BUFF :
                                          (hLength * sizeof(float) * 2 + hLength * typeSize * 4); // 2 PART AND 4 PART
            int32_t matmulUbMaxSize = maxUbSizeForLn - lnResverdBuffer;
            uint32_t splitN = 0;
            for (; splitN < 3 * (headNum); ++splitN) { // q k v sum is 3
                for (uint32_t multipleM = (singleCoreLnBs / wWinSize); multipleM >= 1; --multipleM) {
                    mSizePerLoop = multipleM * (wWinSize);
                    if (mSizePerLoop == 0) {
                        break;
                    }
                    if ((mSizePerLoop * kSizePerLoop) > (l1Size / 2)) { // 2 is for a and B
                        continue;
                    }
                    if (mSizePerLoop >= 256) { // 256 is cube max
                        continue;
                    }
                    TCubeTiling mmtilingData;
                    int32_t ret = SwinTransformerLnQkvQuantGetMatmulTmpSize(context, mmtilingData, mSizePerLoop,
                                                                            nSizePerLoop, kSizePerLoop);
                    if (ret == -1) {
                        continue;
                    }
                    matmul_tiling::SysTilingTempBufSize bufSize;
                    ret = MatmulGetTmpBufSize(mmtilingData, bufSize);
                    int32_t mmOutUbSize = mSizePerLoop * nSizePerLoop * sizeof(uint16_t);
                    int32_t mmUseAllSize = mmOutUbSize + bufSize.ubSize;
                    mmUseAllSize += ((splitN == 0) ? (mSizePerLoop + BLOCK_UINT_16 - 1) / BLOCK_UINT_16 *
                                                         BLOCK_UINT_16 * kSizePerLoop * typeSize :
                                                     BLOCK_UINT_16 * kSizePerLoop * typeSize);
                    int32_t lnUseUbSize = SwinTransformerLnQkvQuantGetUsedUbSize(mSizePerLoop, kSizePerLoop, splitN);
                    if ((mmUseAllSize > matmulUbMaxSize) || (lnUseUbSize > matmulUbMaxSize)) {
                        continue;
                    }
                    if ((singleCoreLnBs > mSizePerLoop) && (mSizePerLoop < BLOCK_UINT_16)) {
                        continue;
                    }
                    if (mmOutUbSize <= maxMnSize) {
                        solutionFlag = true;
                        break;
                    }
                }
                if (solutionFlag == true) {
                    break;
                }
                nSizePerLoop = (splitN == 0) ? (nSizePerLoop / 3) : (nSizePerLoop / 2); // 3 is qkv 2 is resverd
            }
            int64_t lnBaseM = mSizePerLoop;
            int64_t lnBaseK = kSizePerLoop;
            int64_t lnGammaBetaSize = 2 * hLength * (sizeof(float) + sizeof(uint16_t));
            int64_t quantSize = hLength * (sizeof(uint16_t)) * 2; // scale + offset is 2
            int64_t bufferMForLn = lnBaseM;
            int64_t minWinSize = (splitN) ? (7) : std::min(wWinSize, (int64_t)16); // 7 is min win , 16 is align
            int64_t stepLnBaseM = (splitN) ? 8 : wWinSize;                         // 8 is best
            for (; lnBaseK >= sizePerHead; lnBaseK -= sizePerHead) {
                lnBaseM = mSizePerLoop;
                for (; lnBaseM >= minWinSize; lnBaseM -= stepLnBaseM) {
                    if (lnBaseM < 1) {
                        break;
                    }
                    bufferMForLn = (lnBaseM + BLOCK_UINT_16 - 1) / BLOCK_UINT_16 * BLOCK_UINT_16;
                    int64_t allUbSize = lnGammaBetaSize + quantSize;
                    allUbSize += bufferMForLn * lnBaseK * sizeof(uint16_t) * BUFFER_NUM;
                    allUbSize += bufferMForLn * lnBaseK * sizeof(int8_t);
                    allUbSize += bufferMForLn * lnBaseK * sizeof(float) * BUFFER_NUM;
                    if (splitN == 0) {
                        mSizePerLoop = lnBaseM;
                    }
                    allUbSize += ((bufferMForLn + 64 - 1) / 64 * 64) * sizeof(float); // float block is 64
                    std::vector<int64_t> shape_quant = {1, bufferMForLn, lnBaseK};
                    ge::Shape srcShape(shape_quant);
                    AscendC::GetAscendQuantMaxMinTmpSize(srcShape, typeSize, maxValueQuant, minValueQuant);
                    allUbSize += minValueQuant;
                    if (allUbSize < maxUbSizeForLn) {
                        lnSolutionFlag = true;
                        break;
                    }
                }
                if (lnSolutionFlag == true) {
                    break;
                }
            }
            if (lnBaseK == 0) {
                lnBaseK = BLOCK_UINT_16;
                bufferMForLn = BLOCK_UINT_16;
                mSizePerLoop = wWinSize;
            }
            int64_t lnMloopNum = (mSizePerLoop == lnBaseM) ? 1 : (mSizePerLoop + bufferMForLn - 1) / bufferMForLn;
            int64_t lnKloopNum = (kSizePerLoop + lnBaseK - 1) / lnBaseK;
            if (lnMloopNum == 1) {
                tilingData.opBaseInfo.set_lnBaseM(lnBaseM);
            } else {
                tilingData.opBaseInfo.set_lnBaseM(bufferMForLn);
            }
            tilingData.opBaseInfo.set_lnBaseK(lnBaseK);
            tilingData.opBaseInfo.set_lnBufferM(bufferMForLn);
            tilingData.opBaseInfo.set_lnBufferK(lnBaseK);
            tilingData.opBaseInfo.set_lnMSubLoop(lnMloopNum);
            tilingData.opBaseInfo.set_lnKSubLoop(lnKloopNum);
            tilingData.set_tmpBufferForQuant(minValueQuant);
            OP_LOGI(context->GetNodeName(), "lnBaseM: %ld, mSizePerLoop:%ld, nSizePerLoop:%ld, kSizePerLoop:%ld",
                    lnBaseM, mSizePerLoop, nSizePerLoop, kSizePerLoop);
            OP_LOGI(context->GetNodeName(), "bufferMForLn: %ld, lnMloopNum:%ld", bufferMForLn, lnMloopNum);
            if ((solutionFlag == true) && (lnSolutionFlag == true)) {
                tilingData.mmInfo.set_mmSizeM(mSizePerLoop);
                tilingData.mmInfo.set_mmSizeN(nSizePerLoop);
                tilingData.mmInfo.set_mmSizeK(kSizePerLoop);
                tilingData.opBaseInfo.set_singleCoreLnBsSize(singleCoreLnBs);
                ubSizePlatform = maxUbSizeForLn - (lnGammaBetaSize + quantSize) - minValueQuant -
                                 bufferMForLn * lnBaseK * (sizeof(uint16_t)) -
                                 mSizePerLoop * nSizePerLoop * sizeof(uint16_t);
                l1SizePlatForm = l1Size - mSizePerLoop * kSizePerLoop;
                SwinTransformerLnQkvQuantSetMatmulTilingData(context, tilingData.mmTilingParams, l1SizePlatForm,
                                                             l0CSizePlatForm);
                tilingData.set_tmpShareBufferForLn(ubSizePlatform + mSizePerLoop * nSizePerLoop * sizeof(uint16_t));
                OP_LOGI(context->GetNodeName(), "tmpShareBufferForLn: %d", tilingData.get_tmpShareBufferForLn());
            }
            break;
        }
        default:
            break;
    }
    SwinTransformerLnQkvQuantSetWorkSpace(context);
    context->SetBlockDim(static_cast<int64_t>(blockNum));

    SwinTransformerLnQkvQuantSetTilingKey(context);
    SwinTransformerLnQkvQuantSaveTilingData(context);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingFuncForSwinTransformerLnQkvQuant(gert::TilingContext* context)
{
    SwinTransformerLnQkvQuantTilingCompute tilingCompute;
    auto ret = tilingCompute.SwinTransformerLnQkvQuantTilingMainProc(context);

    OP_LOGI(context->GetNodeName(), "SwinTransformerLnQkvQuant tiling end.");
    return ret;
} // TilingFunc

ge::graphStatus TilingPrepareForSwinTransformerLnQkvQuant(gert::TilingParseContext* context)
{
    // auto compileInfo = GetCompileInfoPtr<SwinTransformerLnQkvQuantCompileInfo>(context);
    auto compileInfo = context->GetCompiledInfo<SwinTransformerLnQkvQuantCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = (uint32_t)platformInfo.GetCoreNum();
    uint64_t ubSizePlatform;
    uint64_t l1SizePlatForm;
    uint64_t l0CSizePlatForm;
    uint64_t l0ASizePlatForm;

    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1SizePlatForm);
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0CSizePlatForm);
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0ASizePlatForm);
    compileInfo->coreNum = coreNum;

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SwinTransformerLnQkvQuant)
    .Tiling(TilingFuncForSwinTransformerLnQkvQuant)
    .TilingParse<SwinTransformerLnQkvQuantCompileInfo>(TilingPrepareForSwinTransformerLnQkvQuant);
} // namespace optiling
