/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/*!
 * \file paged_attention_checker.cpp
 * \brief
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../fused_infer_attention_score_tiling_constants.h"
#include "paged_attention_checker.h"

namespace optiling {
using std::map;
using std::string;
using std::pair;
using namespace ge;
using namespace AscendC;
using namespace arch35FIA;

// 公共校验函数
// check blocktable dtype
ge::graphStatus PagedAttentionChecker::CheckBlockTableDtype(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.opParamInfo.blockTable.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    const gert::CompileTimeTensorDesc *blockTableDesc = fiaInfo.opParamInfo.blockTable.desc;
    OP_CHECK_IF(blockTableDesc->GetDataType() != ge::DT_INT32,
                OP_LOGE(fiaInfo.opName, "When page attention enable, blockTable datatype only support INT32."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// check blockTable shape size
ge::graphStatus PagedAttentionChecker::CheckBlockTableShapeSize(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.opParamInfo.blockTable.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    const gert::Shape blockTableShape = fiaInfo.opParamInfo.blockTable.tensor->GetStorageShape();

    // check dim num
    OP_CHECK_IF(blockTableShape.GetDimNum() != 2,
                OP_LOGE(fiaInfo.opName, "When page attention enable, the dim num(%zu) of blockTable should be 2.",
                        blockTableShape.GetDimNum()),
                return ge::GRAPH_FAILED);

    // check blockTable each dim cannot be 0
    OP_CHECK_IF(
        blockTableShape.GetShapeSize() == 0,
        OP_LOGE(fiaInfo.opName, "When page attention enable, blockTable each dim can not be 0, now is [%ld, %ld].",
                blockTableShape.GetDim(0), blockTableShape.GetDim(1)),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// check blocksize
ge::graphStatus PagedAttentionChecker::CheckBlockSize(const FiaTilingInfo &fiaInfo)
{
    OP_CHECK_IF(fiaInfo.opParamInfo.blockSize == nullptr,
                OP_LOGE(fiaInfo.opName, "When page attention enable, blockSize should not be null."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(fiaInfo.blockSize <= 0,
                OP_LOGE(fiaInfo.opName, "When page attention enable, blockSize(%d) should be > 0.", fiaInfo.blockSize),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckBlockTableExistence(const FiaTilingInfo &fiaInfo)
{
    OP_CHECK_IF(fiaInfo.opParamInfo.blockTable.tensor == nullptr,
                OP_LOGE(fiaInfo.opName, "When page attention enable, block table should not be null."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckFeatureSupport(const FiaTilingInfo &fiaInfo)
{
    OP_CHECK_IF((fiaInfo.opParamInfo.queryPaddingSize.tensor != nullptr) ||
                    (fiaInfo.opParamInfo.kvPaddingSize.tensor != nullptr),
                OP_LOGE(fiaInfo.opName, "When page attention enable, left padding is not supported."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((fiaInfo.opParamInfo.keySharedPrefix.tensor != nullptr) ||
                    (fiaInfo.opParamInfo.valueSharedPrefix.tensor != nullptr),
                OP_LOGE(fiaInfo.opName, "When page attention enable, system prefix is not supported."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(fiaInfo.isQKVDDifferent,
                OP_LOGE(fiaInfo.opName, "When page attention enable, the head dim of query and key should be equal."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckSeqLengthKVExistence(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.isMaxWorkspace) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(fiaInfo.opParamInfo.actualSeqLengths.tensor == nullptr ||
                    fiaInfo.opParamInfo.actualSeqLengths.tensor->GetData<int64_t>() == nullptr ||
                    fiaInfo.opParamInfo.actualSeqLengths.tensor->GetShapeSize() == 0,
                OP_LOGE(fiaInfo.opName, "When page attention enable, actualSeqLengths of KV should not be null."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

int64_t PagedAttentionChecker::GetMaxBlockNumPerBatch(const FiaTilingInfo &fiaInfo)
{
    const int32_t blockSize = fiaInfo.blockSize;
    const gert::Tensor *actSeqLenKV = fiaInfo.opParamInfo.actualSeqLengths.tensor;
    uint32_t actualSeqLengthsKVSize = static_cast<uint32_t>(actSeqLenKV->GetShapeSize());
    int64_t actualSeqKVPerBatch = 0;
    int64_t blockNumPerBatch = 0;
    int64_t maxBlockNumPerBatch = 0;
    uint32_t loop = std::min(actualSeqLengthsKVSize, fiaInfo.bSize);
    if (actSeqLenKV->GetData<int64_t>() == nullptr) {
        return 0;
    }
    for (uint32_t i = 0; i < loop; i++) {
        actualSeqKVPerBatch = actSeqLenKV->GetData<int64_t>()[i];
        blockNumPerBatch = (actualSeqKVPerBatch + blockSize - 1) / blockSize;
        if (blockNumPerBatch > maxBlockNumPerBatch) {
            maxBlockNumPerBatch = blockNumPerBatch;
        }
    }
    return maxBlockNumPerBatch;
}

// check mask shape
ge::graphStatus PagedAttentionChecker::CheckMaskShape(const FiaTilingInfo &fiaInfo)
{
    if ((fiaInfo.sparseMode == SPARSE_MODE_NO_MASK || fiaInfo.sparseMode == SPARSE_MODE_ALL_MASK) &&
        fiaInfo.opParamInfo.attenMask.tensor != nullptr) {
        if (fiaInfo.opParamInfo.actualSeqLengths.tensor == nullptr) {
            return ge::GRAPH_SUCCESS;
        }
        const gert::Shape attenMaskShape = fiaInfo.opParamInfo.attenMask.tensor->GetStorageShape();
        int64_t maxBlockNumPerBatch = GetMaxBlockNumPerBatch(fiaInfo);
        uint32_t attenMaskDimNum = attenMaskShape.GetDimNum();
        OP_CHECK_IF(attenMaskDimNum > 0 &&
                        (attenMaskShape.GetDim(attenMaskDimNum - 1) < maxBlockNumPerBatch * fiaInfo.blockSize),
                    OP_LOGE(fiaInfo.opName,
                            "When page attention enable and attenMask enable (sparseMode = %u), the last dimension of "
                            "attenMask(%ld) should be greater than maxBlockNumPerBatch(%ld) * blockSize(%d).",
                            fiaInfo.sparseMode, attenMaskShape.GetDim(attenMaskDimNum - 1), maxBlockNumPerBatch,
                            fiaInfo.blockSize),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// check pa cache shape
ge::graphStatus PagedAttentionChecker::CheckPACacheShape(const FiaTilingInfo &fiaInfo, const gert::Shape tempShape,
                                                         const std::string &inputName)
{
    uint32_t shapeDim = tempShape.GetDimNum();
    int64_t tempBlockSize = 0;
    int64_t tempH = 0;
    int64_t tempN = 0;
    int64_t tempD = 0;
    int64_t tempD0 = 0;
    int64_t tempD1 = 0;

    uint32_t compareD = 0;
    if (inputName == "key") {
        compareD = fiaInfo.qkHeadDim;
    } else if (inputName == "keyRope") {
        compareD = fiaInfo.ropeHeadDim;
    } else {
        compareD = fiaInfo.vHeadDim;
    }

    if (shapeDim == DIM_NUM_3) { // [blockNums, blockSize, H]
        tempBlockSize = tempShape.GetDim(DIM_NUM_1);
        tempH = tempShape.GetDim(DIM_NUM_2);
        OP_CHECK_IF(tempBlockSize != fiaInfo.blockSize,
                    OP_LOGE(fiaInfo.opName, "When page attention enable, blocksize of %s(%u) should be %u",
                            inputName.c_str(), tempBlockSize, fiaInfo.blockSize),
                    return ge::GRAPH_FAILED);

        if (fiaInfo.inputKvType == ge::DT_INT4) {
            OP_CHECK_IF(tempH != fiaInfo.n2Size * compareD,
                        OP_LOGE(fiaInfo.opName,
                                "When page attention enable, if input kv dataType is INT32, "
                                "the axis H of %s(%u) should be %u, "
                                "if input kv dataType is INT4, the axis H of %s(%u) should be %u",
                                inputName.c_str(), tempH / NUM8, fiaInfo.n2Size * compareD / NUM8, inputName.c_str(),
                                tempH, fiaInfo.n2Size * compareD),
                        return ge::GRAPH_FAILED);

            OP_CHECK_IF(tempH > H_LIMIT,
                        OP_LOGE(fiaInfo.opName,
                                "When page attention enable and layout is BSH, "
                                "if input kv dataType is INT32, the axis H of %s(%ld) should not > %u."
                                "if input kv dataType is INT4, the axis H of %s(%ld) should not > %u.",
                                inputName.c_str(), tempH / NUM8, H_LIMIT / NUM8, inputName.c_str(), tempH, H_LIMIT),
                        return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(tempH != fiaInfo.n2Size * compareD,
                        OP_LOGE(fiaInfo.opName, "When page attention enable, the axis H(%u) of kvCache should be %u",
                                tempH, fiaInfo.n2Size * compareD),
                        return ge::GRAPH_FAILED);

            OP_CHECK_IF(tempH > H_LIMIT,
                        OP_LOGE(fiaInfo.opName,
                                "When page attention enable and layout is BSH, "
                                "the axis H of %s(%ld) should not > %u.",
                                inputName.c_str(), tempH, H_LIMIT),
                        return ge::GRAPH_FAILED);
        }
    } else if (shapeDim == DIM_NUM_4) { // [blockNums, N, blockSize, D]
        tempN = tempShape.GetDim(DIM_NUM_1);
        tempBlockSize = tempShape.GetDim(DIM_NUM_2);
        tempD = tempShape.GetDim(DIM_NUM_3);

        OP_CHECK_IF(tempN != fiaInfo.n2Size,
                    OP_LOGE(fiaInfo.opName, "When page attention enable, the axis N(%u) of kvCache should be %u", tempN,
                            fiaInfo.n2Size),
                    return ge::GRAPH_FAILED);

        OP_CHECK_IF(tempBlockSize != fiaInfo.blockSize,
                    OP_LOGE(fiaInfo.opName, "When page attention enable, blocksize(%u) of kvCache should be %u",
                            tempBlockSize, fiaInfo.blockSize),
                    return ge::GRAPH_FAILED);

        if (fiaInfo.inputKvType == ge::DT_INT4) {
            OP_CHECK_IF(tempD != compareD,
                        OP_LOGE(fiaInfo.opName,
                                "When page attention enable, if input kv dataType is INT32, "
                                "the axis D of %s(%u) should be %u, "
                                "if input kv dataType is INT4, the axis D of %s(%u) should be %u",
                                inputName.c_str(), tempD / NUM8, compareD / NUM8, inputName.c_str(), tempD, compareD),
                        return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(tempD != compareD,
                        OP_LOGE(fiaInfo.opName, "When page attention enable, the axis D of %s(%u) should be %u",
                                inputName.c_str(), tempD, compareD),
                        return ge::GRAPH_FAILED);
        }
    } else { // [blockNums, N, D1, blocksize, D0]
        tempN = tempShape.GetDim(DIM_NUM_1);
        tempD1 = tempShape.GetDim(DIM_NUM_2);
        tempBlockSize = tempShape.GetDim(DIM_NUM_3);
        tempD0 = tempShape.GetDim(DIM_NUM_4);
        const ge::DataType inputKvType = fiaInfo.inputKvType;
        uint32_t d0Size = NUM_16; // D0 = 16

        OP_CHECK_IF(tempN != fiaInfo.n2Size,
                    OP_LOGE(fiaInfo.opName, "When page attention enable, the axis N of %s(%u) should be %u",
                            inputName.c_str(), tempN, fiaInfo.n2Size),
                    return ge::GRAPH_FAILED);

        OP_CHECK_IF(tempBlockSize != fiaInfo.blockSize,
                    OP_LOGE(fiaInfo.opName, "When page attention enable, blocksize of %s(%u) should be %u",
                            inputName.c_str(), tempBlockSize, fiaInfo.blockSize),
                    return ge::GRAPH_FAILED);

        if (enableAntiQuant_) {
            d0Size = NUM_16;
            if (fiaInfo.inputKvType == ge::DT_INT4) {
                OP_CHECK_IF(tempD0 != d0Size,
                            OP_LOGE(fiaInfo.opName,
                                    "When PA_NZ enable, if input kv dataType is INT32, "
                                    "the last dim (D0) of kvCache(%u) should be %u; "
                                    "if input kv dataType is INT4, the last dim (D0) of %s(%u) should be %u",
                                    inputName.c_str(), tempD0 / NUM8, d0Size / NUM8, tempD0, d0Size),
                            return ge::GRAPH_FAILED);
            } else {
                OP_CHECK_IF(tempD0 != d0Size,
                            OP_LOGE(fiaInfo.opName, "When PA_NZ enable, the last dim (D0) of %s(%u) should be %u",
                                    inputName.c_str(), tempD0, d0Size),
                            return ge::GRAPH_FAILED);
            }
        } else {
            std::unordered_map<ge::DataType, float> typeSizeMap = {
                {ge::DT_FLOAT16, static_cast<float>(FLOAT16SIZE)},
                {ge::DT_BF16, static_cast<float>(BFLOAT16SIZE)},
                {ge::DT_INT8, static_cast<float>(INT8SIZE)},
                {ge::DT_HIFLOAT8, static_cast<float>(FLOAT8SIZE)},
                {ge::DT_FLOAT8_E4M3FN, static_cast<float>(FLOAT8SIZE)}};
            float dataTypeSizeValue = static_cast<float>(FLOAT16SIZE);
            auto inputTypeCheck = typeSizeMap.find(fiaInfo.inputKvType);
            if (inputTypeCheck != typeSizeMap.end()) {
                dataTypeSizeValue = inputTypeCheck->second;
            }

            if (enableFullQuant_ && inputName == "keyRope") {
                dataTypeSizeValue = static_cast<float>(BFLOAT16SIZE);
            }

            d0Size = BYTE_BLOCK / dataTypeSizeValue;
            OP_CHECK_IF(tempD0 != d0Size,
                        OP_LOGE(fiaInfo.opName, "When PA_NZ enable, the last dim (D0) of %s(%u) should be %u",
                                inputName.c_str(), tempD0, d0Size),
                        return ge::GRAPH_FAILED);
        }

        OP_CHECK_IF(tempD1 != compareD / d0Size,
                    OP_LOGE(fiaInfo.opName, "When PA_NZ enable, the third dim (D1) of %s(%u) should be %u",
                            inputName.c_str(), tempD1, compareD / d0Size),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// check input query dtype
ge::graphStatus PagedAttentionChecker::CheckQDtypeSupport(const FiaTilingInfo &fiaInfo)
{
    OP_CHECK_IF(
        fiaInfo.inputQType == ge::DT_INT8 && fiaInfo.ropeMode != RopeMode::ROPE_SPLIT,
        OP_LOGE(fiaInfo.opName,
                "When page attention enable, the datatype(%s) of query should not be INT8 except for MLA fullquant.",
                DataTypeToSerialString(fiaInfo.inputQType).c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckBlockTableShape(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.isMaxWorkspace) {
        return ge::GRAPH_SUCCESS;
    }

    if (fiaInfo.opParamInfo.actualSeqLengths.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    // 每个batch的 blockNum 小于 blocktable dim2
    int64_t maxBlockNumPerBatch = GetMaxBlockNumPerBatch(fiaInfo);
    const gert::Shape blockTableShape = fiaInfo.opParamInfo.blockTable.tensor->GetStorageShape();
    OP_CHECK_IF(((blockTableShape.GetDim(0) != fiaInfo.bSize) || (blockTableShape.GetDim(1) < maxBlockNumPerBatch)),
                OP_LOGE(fiaInfo.opName,
                        "When page attention enable, block table shape should be [%u, >=%ld], now is [%ld, %ld].",
                        fiaInfo.bSize, maxBlockNumPerBatch, blockTableShape.GetDim(0), blockTableShape.GetDim(1)),
                return ge::GRAPH_FAILED);

    // check key cache shape
    if (ge::GRAPH_SUCCESS != CheckPACacheShape(fiaInfo, fiaInfo.opParamInfo.key.shape->GetStorageShape(), "key") ||
        ge::GRAPH_SUCCESS != CheckPACacheShape(fiaInfo, fiaInfo.opParamInfo.value.shape->GetStorageShape(), "value")) {
        return ge::GRAPH_FAILED;
    }

    // check rope cache shape
    if (fiaInfo.ropeMode == RopeMode::ROPE_SPLIT) {
        if (ge::GRAPH_SUCCESS !=
            CheckPACacheShape(fiaInfo, fiaInfo.opParamInfo.keyRope.tensor->GetStorageShape(), "keyRope")) {
            return ge::GRAPH_FAILED;
        }
    }

    // warning: S2 <= 20M
    if (maxBlockNumPerBatch * fiaInfo.blockSize > S_LIMIT) {
        OP_LOGW(fiaInfo.opName, "When page attention enable, sequence length(%ld) of kv should <= 20M.",
                maxBlockNumPerBatch * fiaInfo.blockSize);
    }
    return ge::GRAPH_SUCCESS;
}

// check blocksize
ge::graphStatus PagedAttentionChecker::CheckBlockSizeSupport(const FiaTilingInfo &fiaInfo)
{
    if (enableNonQuant_) { // 非量化
        if (fiaInfo.socVersion == platform_ascendc::SocVersion::ASCEND910B) {
            if (fiaInfo.blockSize % BLOCK_SIZE_ALIGN_SIZE_16 != 0) {
                OP_LOGE(fiaInfo.opName, "In no quant scenario, blockSize(%d) should aligned to 16.", fiaInfo.blockSize);
                return ge::GRAPH_FAILED;
            }
            if (fiaInfo.blockSize > BLOCK_SIZE_MAX_FOR_NO_QUANT) {
                OP_LOGE(fiaInfo.opName, "In no quant scenario, blockSize(%d) should less equal than 1024.",
                        fiaInfo.blockSize);
                return ge::GRAPH_FAILED;
            }
            if (fiaInfo.kvStorageMode == KvStorageMode::PAGE_ATTENTION && fiaInfo.blockSize == 0) {
                OP_LOGE(fiaInfo.opName,
                        "In no quant scenario, when page attention enable, blockSize(%d) should not be 0.",
                        fiaInfo.blockSize);
                return ge::GRAPH_FAILED;
            }
        } else {
            if (fiaInfo.ropeMode != RopeMode::NO_ROPE) { // MLA场景 [16, 1024]且16对齐
                OP_CHECK_IF(
                    fiaInfo.blockSize > BLOCK_SIZE_MAX_FOR_NO_QUANT || fiaInfo.blockSize < BLOCK_SIZE_ALIGN_SIZE_16 ||
                        fiaInfo.blockSize % BLOCK_SIZE_ALIGN_SIZE_16 != 0,
                    OP_LOGE(
                        fiaInfo.opName,
                        "In no quant MLA scenario, when page attention enable, blockSize(%d) should be a multiple of "
                        "%u, and should be in range of [%u, %u].",
                        fiaInfo.blockSize, BLOCK_SIZE_ALIGN_SIZE_16, BLOCK_SIZE_ALIGN_SIZE_16,
                        BLOCK_SIZE_MAX_FOR_NO_QUANT),
                    return ge::GRAPH_FAILED);
            } else if (fiaInfo.qkHeadDim == NUM_64 ||
                       fiaInfo.qkHeadDim == NUM_128) { // GQA D =64/128场景 [16, 1024]且16对齐
                OP_CHECK_IF(fiaInfo.blockSize > BLOCK_SIZE_MAX_FOR_NO_QUANT ||
                                fiaInfo.blockSize < BLOCK_SIZE_ALIGN_SIZE_16 ||
                                fiaInfo.blockSize % BLOCK_SIZE_ALIGN_SIZE_16 != 0,
                            OP_LOGE(fiaInfo.opName,
                                    "In no quant GQA(D = %u) scenario, when page attention enable, blockSize(%d) "
                                    "should be a multiple "
                                    "of %u, and should be in range of [%u, %u].",
                                    fiaInfo.qkHeadDim, fiaInfo.blockSize, BLOCK_SIZE_ALIGN_SIZE_16,
                                    BLOCK_SIZE_ALIGN_SIZE_16, BLOCK_SIZE_MAX_FOR_NO_QUANT),
                            return ge::GRAPH_FAILED);
            } else {
                // GQA D != 64/128, QS > 1 [128, 512]且128对齐
                OP_CHECK_IF(
                    (fiaInfo.s1Size > NUM1) &&
                        (fiaInfo.blockSize > BLOCK_SIZE_MAX || fiaInfo.blockSize < BLOCK_SIZE_ALIGN_SIZE_128 ||
                         fiaInfo.blockSize % BLOCK_SIZE_ALIGN_SIZE_128 != 0),
                    OP_LOGE(fiaInfo.opName,
                            "In no quant GQA (QS > 1) scenario, when page attention enable, blockSize(%d) should be a "
                            "multiple "
                            "of %u, and should be in range of [%u, %u].",
                            fiaInfo.blockSize, BLOCK_SIZE_ALIGN_SIZE_128, BLOCK_SIZE_ALIGN_SIZE_128, BLOCK_SIZE_MAX),
                    return ge::GRAPH_FAILED);
                // GQA D != 64/128, QS = 1 [16, 512]且16对齐
                OP_CHECK_IF((fiaInfo.s1Size == NUM1) &&
                                (fiaInfo.blockSize > BLOCK_SIZE_MAX || fiaInfo.blockSize < BLOCK_SIZE_ALIGN_SIZE_16 ||
                                 fiaInfo.blockSize % BLOCK_SIZE_ALIGN_SIZE_16 != 0),
                            OP_LOGE(fiaInfo.opName,
                                    "In no quant GQA (QS = 1) scenario, when page attention enable, blockSize(%d) "
                                    "should be a multiple "
                                    "of %u, and should be in range of [%u, %u].",
                                    fiaInfo.s1Size, fiaInfo.blockSize, BLOCK_SIZE_ALIGN_SIZE_16,
                                    BLOCK_SIZE_ALIGN_SIZE_16, BLOCK_SIZE_MAX),
                            return ge::GRAPH_FAILED);
            }
        }
    } else if (enableAntiQuant_) {
        std::unordered_map<ge::DataType, float> typeSizeMap = {{ge::DT_FLOAT16, static_cast<float>(FLOAT16SIZE)},
                                                               {ge::DT_BF16, static_cast<float>(BFLOAT16SIZE)},
                                                               {ge::DT_INT8, static_cast<float>(INT8SIZE)},
                                                               {ge::DT_HIFLOAT8, static_cast<float>(FLOAT8SIZE)},
                                                               {ge::DT_FLOAT8_E4M3FN, static_cast<float>(FLOAT8SIZE)},
                                                               {ge::DT_INT4, INT4SIZE},
                                                               {ge::DT_FLOAT4_E2M1, FLOAT4SIZE}};
        float dataTypeSizeValue = static_cast<float>(FLOAT16SIZE);
        auto inputTypeCheck = typeSizeMap.find(fiaInfo.inputKvType);
        if (inputTypeCheck != typeSizeMap.end()) {
            dataTypeSizeValue = inputTypeCheck->second;
        }
        uint32_t blockSizeAlign = static_cast<uint32_t>(BYTE_BLOCK / dataTypeSizeValue);

        // 伪量化, 与Dtype相关
        OP_CHECK_IF((fiaInfo.blockSize > BLOCK_SIZE_MAX || fiaInfo.blockSize < blockSizeAlign ||
                     fiaInfo.blockSize % blockSizeAlign != 0),
                    OP_LOGE(fiaInfo.opName,
                            "In antiquant scenario, when page attention enable, if kvCache datatype is %s, "
                            "blockSize(%d) should be a multiple of %u, and should be in range of [%u, %u].",
                            DataTypeToSerialString(fiaInfo.inputKvType).c_str(), fiaInfo.blockSize, blockSizeAlign,
                            blockSizeAlign, BLOCK_SIZE_MAX),
                    return ge::GRAPH_FAILED);
    } else { // 全量化
        OP_CHECK_IF(fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D512 && fiaInfo.blockSize != NUM_128,
                    OP_LOGE(fiaInfo.opName,
                            "In MLA fullquant scenario, when page attention enable, blockSize(%d) should be 128.",
                            fiaInfo.blockSize),
                    return ge::GRAPH_FAILED);

        OP_CHECK_IF(
            fiaInfo.mlaMode == MlaMode::NO_MLA && fiaInfo.fullQuantMode != FiaFullQuantMode::QKV_MXFP8_FULL_QUANT &&
                (fiaInfo.blockSize > BLOCK_SIZE_MAX || fiaInfo.blockSize < NUM_32 || fiaInfo.blockSize % NUM_32 != 0),
            OP_LOGE(fiaInfo.opName,
                    "In per-tensor quant scenario, when page attention enable, "
                    "blockSize(%d) should be a multiple of %u, and should be in range of [%u, %u].",
                    fiaInfo.blockSize, NUM_32, NUM_32, BLOCK_SIZE_MAX),
            return ge::GRAPH_FAILED);

        // mxfp8 仅支持blocksize等于512或者1024
        OP_CHECK_IF(fiaInfo.fullQuantMode == FiaFullQuantMode::QKV_MXFP8_FULL_QUANT &&
                        (fiaInfo.blockSize != BLOCK_SIZE_FOR_MXFP8 && fiaInfo.blockSize != BLOCK_SIZE_1024_FOR_MXFP8),
                    OP_LOGE(fiaInfo.opName,
                            "In MXFP8 fullquant scenario, when page attention enable, "
                            "blockSize(%d) should be %u or %u.",
                            fiaInfo.blockSize, BLOCK_SIZE_FOR_MXFP8, BLOCK_SIZE_1024_FOR_MXFP8),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckKVLayout(const FiaTilingInfo &fiaInfo)
{
    if (!enableFullQuant_) {
        return ge::GRAPH_SUCCESS;
    }
    const string inputLayout = fiaInfo.opParamInfo.layOut;
    const uint32_t dimNum = fiaInfo.opParamInfo.key.shape->GetStorageShape().GetDimNum();
    if (fiaInfo.fullQuantMode == FiaFullQuantMode::QKV_MXFP8_FULL_QUANT) {
        OP_CHECK_IF(dimNum == 3, OP_LOGE(fiaInfo.opName, "In MXFP8 fullquant scenario, PA BnBsH is not supported."),
                    return ge::GRAPH_FAILED);
        if (fiaInfo.sparseMode == SPARSE_MODE_INIT_SWA) {
            OP_CHECK_IF(
                dimNum == 5,
                OP_LOGE(fiaInfo.opName, "In MXFP8 fullquant scenario, when sparse mode = 5, PA NZ is not supported."),
                return ge::GRAPH_FAILED);
        }
    } else if (inputLayout == "BSH" || inputLayout == "BSND" || inputLayout == "BSH_NBSD" ||
               inputLayout == "BSND_NBSD") {
        OP_CHECK_IF(dimNum == 4,
                    OP_LOGE(fiaInfo.opName,
                            "When page attention enable and input layout is %s, "
                            "PA BnNBsD is not supported.",
                            inputLayout.c_str()),
                    return ge::GRAPH_FAILED);
    }
    if (fiaInfo.socVersion == platform_ascendc::SocVersion::ASCEND910B) {
        OP_CHECK_IF(
            fiaInfo.kvLayout == FiaLayout::BnBsH &&
                (fiaInfo.qLayout != FiaLayout::BSH && fiaInfo.qLayout != FiaLayout::BSND &&
                 fiaInfo.qLayout != FiaLayout::BNSD && fiaInfo.qLayout != FiaLayout::TND &&
                 fiaInfo.qLayout != FiaLayout::NTD),
            OP_LOGE(fiaInfo.opName,
                    "In %s %s situation, the key/value's layout is BnBsH, query layout must be BSH, BSND, BNSD, TND "
                    "and NTD in page attention scene, but got %s",
                    QuantModeToSerialString(fiaInfo.quantMode).c_str(),
                    SituationToSerialString(fiaInfo.ropeMode).c_str(), LayoutToSerialString(fiaInfo.qLayout).c_str()),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(
            fiaInfo.kvLayout == FiaLayout::BnNBsD &&
                (fiaInfo.qLayout != FiaLayout::BSH && fiaInfo.qLayout != FiaLayout::BSND &&
                 fiaInfo.qLayout != FiaLayout::BNSD && fiaInfo.qLayout != FiaLayout::TND &&
                 fiaInfo.qLayout != FiaLayout::NTD),
            OP_LOGE(fiaInfo.opName,
                    "In %s %s situation, the key/value's layout is BnNBsD, query layout must be BSH, BSND, BNSD, TND "
                    "and NTD in page attention scene, but got %s",
                    QuantModeToSerialString(fiaInfo.quantMode).c_str(),
                    SituationToSerialString(fiaInfo.ropeMode).c_str(), LayoutToSerialString(fiaInfo.qLayout).c_str()),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(
            fiaInfo.kvLayout == FiaLayout::NZ &&
                (fiaInfo.qLayout != FiaLayout::BSH && fiaInfo.qLayout != FiaLayout::BSND &&
                 fiaInfo.qLayout != FiaLayout::BNSD && fiaInfo.qLayout != FiaLayout::TND &&
                 fiaInfo.qLayout != FiaLayout::NTD),
            OP_LOGE(fiaInfo.opName,
                    "In %s %s situation, the key/value's layout is NZ, query layout must be BSH, BSND, BNSD, TND and "
                    "NTD in page attention scene, but got %s",
                    QuantModeToSerialString(fiaInfo.quantMode).c_str(),
                    SituationToSerialString(fiaInfo.ropeMode).c_str(), LayoutToSerialString(fiaInfo.qLayout).c_str()),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckFeatureQueryS(const FiaTilingInfo &fiaInfo)
{
    // When antiquantMode is 0 or 1 and data type of key/value is int8 scenario, page attention is not supported.
    if (fiaInfo.s1Size > 1) {
        int64_t keyAntiquantMode = 0;
        if (fiaInfo.opParamInfo.keyAntiquantMode != nullptr) {
            keyAntiquantMode = *fiaInfo.opParamInfo.keyAntiquantMode;
        }
        OP_CHECK_IF((keyAntiquantMode == PER_CHANNEL_MODE || keyAntiquantMode == PER_TOKEN_MODE) &&
                        fiaInfo.inputKvType == ge::DT_INT8,
                    OP_LOGE(fiaInfo.opName,
                            "In keyAntiquant/valueAntiquant split mode and data type of key/value is int8 scenario, if "
                            "keyAntiquantMode/valueAntiquantMode is 0 or 1, page attention is not supported!"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckSinglePara(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.kvStorageMode != KvStorageMode::PAGE_ATTENTION) {
        return ge::GRAPH_SUCCESS;
    }

    if (ge::GRAPH_SUCCESS != CheckBlockTableDtype(fiaInfo) || ge::GRAPH_SUCCESS != CheckBlockTableShapeSize(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckBlockSize(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckParaExistence(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.kvStorageMode != KvStorageMode::PAGE_ATTENTION) {
        return ge::GRAPH_SUCCESS;
    }

    if (ge::GRAPH_SUCCESS != CheckBlockTableExistence(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckCrossFeature(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.kvStorageMode != KvStorageMode::PAGE_ATTENTION) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckSeqLengthKVExistence(fiaInfo) || ge::GRAPH_SUCCESS != CheckKVLayout(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckBlockSizeSupport(fiaInfo) || ge::GRAPH_SUCCESS != CheckBlockTableShape(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckMaskShape(fiaInfo) || ge::GRAPH_SUCCESS != CheckFeatureSupport(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckQDtypeSupport(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (enableAntiQuant_) {
        if (ge::GRAPH_SUCCESS != CheckFeatureQueryS(fiaInfo)) {
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckMultiParaConsistency(const FiaTilingInfo &fiaInfo)
{
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
