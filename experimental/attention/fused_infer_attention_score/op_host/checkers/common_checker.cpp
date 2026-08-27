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
 * \file common_checker.cpp
 * \brief
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../fused_infer_attention_score_tiling_constants.h"
#include "common_checker.h"

namespace optiling {
using std::map;
using std::string;
using std::pair;
using namespace ge;
using namespace AscendC;
using namespace arch35FIA;

// 公共校验函数
ge::graphStatus CommonChecker::CheckInputFormat(const FiaTilingInfo &fiaInfo)
{
    if (CheckFormatSupport(fiaInfo.opParamInfo.query.desc, "query") != ge::GRAPH_SUCCESS ||
        CheckFormatSupport(fiaInfo.opParamInfo.key.desc, "key") != ge::GRAPH_SUCCESS ||
        CheckFormatSupport(fiaInfo.opParamInfo.value.desc, "value") != ge::GRAPH_SUCCESS ||
        CheckFormatSupport(fiaInfo.opParamInfo.attenOut.desc, "attentionOut") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckParaExistenceImpl(const FiaTilingInfo &fiaInfo)
{
    OP_CHECK_IF(fiaInfo.opParamInfo.query.desc == nullptr || fiaInfo.opParamInfo.query.shape == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(fiaInfo.opName, "Query input is null pointer!"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(fiaInfo.opParamInfo.key.desc == nullptr || fiaInfo.opParamInfo.key.shape == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(fiaInfo.opName, "Key input is null pointer!"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(fiaInfo.opParamInfo.value.desc == nullptr || fiaInfo.opParamInfo.value.shape == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(fiaInfo.opName, "Value input is null pointer!"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(fiaInfo.opParamInfo.attenOut.desc == nullptr || fiaInfo.opParamInfo.attenOut.shape == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(fiaInfo.opName, "AttentionOut is null pointer!"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckDtypeCommon(const gert::CompileTimeTensorDesc *desc, const std::string &name,
                                                std::map<std::string, std::vector<ge::DataType>> dataMap)
{
    if (desc != nullptr) {
        const auto &it = dataMap.find(name);
        OP_CHECK_IF(it == dataMap.end(),
                    OP_LOGE("FIA", "%s datatype support list should be specify in map", name.c_str()),
                    return ge::GRAPH_FAILED);
        auto &expectDtypeList = it->second;
        OP_CHECK_IF(
            std::find(expectDtypeList.begin(), expectDtypeList.end(), desc->GetDataType()) == expectDtypeList.end(),
            "", // LogErrorDtypeSupport(expectDtypeList, desc->GetDataType(), name), //公共打印函数
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckPAKeyValue(const FiaTilingInfo &fiaInfo)
{
    const gert::StorageShape *keyShape = fiaInfo.opParamInfo.key.shape;
    const gert::StorageShape *valueShape = fiaInfo.opParamInfo.value.shape;
    uint32_t keyDimNum = keyShape->GetStorageShape().GetDimNum();
    uint32_t keyBlockNum = fiaInfo.totalBlockNum;
    uint32_t keyHeadNum = fiaInfo.n2Size;
    uint32_t keyBlockSize = fiaInfo.blockSize;
    uint32_t keyHeadDim = fiaInfo.qkHeadDim;
    uint32_t keyH = fiaInfo.qkHeadDim * fiaInfo.n2Size;
    uint32_t keyD0 = 16;
    uint32_t keyD1 = fiaInfo.qkHeadDim / 16; // 当前PA NZ场景D 16

    uint32_t valueDimNum = valueShape->GetStorageShape().GetDimNum();
    uint32_t valueBlockNum = fiaInfo.totalBlockNum;
    uint32_t valueHeadNum = fiaInfo.n2Size;
    uint32_t valueBlockSize = fiaInfo.blockSize;
    uint32_t valueHeadDim = fiaInfo.vHeadDim;
    uint32_t valueH = fiaInfo.vHeadDim * fiaInfo.n2Size;
    uint32_t valueD0 = 16;
    uint32_t valueD1 = fiaInfo.vHeadDim / 16; // 当前PA NZ场景D 16
    OP_CHECK_IF(keyDimNum != valueDimNum,
                OP_LOGE(fiaInfo.opName,
                        "The dim num of key should be consistent with the dim num of value,"
                        "but current dim num of key is %zu, dim num of value is %zu.",
                        keyDimNum, valueDimNum),
                return ge::GRAPH_FAILED);

    if (keyDimNum == 3) { // BBH
        keyBlockNum = keyShape->GetStorageShape().GetDim(0);
        keyBlockSize = keyShape->GetStorageShape().GetDim(1);
        keyH = keyShape->GetStorageShape().GetDim(2);
        valueBlockNum = valueShape->GetStorageShape().GetDim(0);
        valueBlockSize = valueShape->GetStorageShape().GetDim(1);
        valueH = valueShape->GetStorageShape().GetDim(2);
    } else if (keyDimNum == 4) { // BNBD
        keyBlockNum = keyShape->GetStorageShape().GetDim(0);
        keyHeadNum = keyShape->GetStorageShape().GetDim(1);
        keyBlockSize = keyShape->GetStorageShape().GetDim(2);
        keyHeadDim = keyShape->GetStorageShape().GetDim(3);
        valueBlockNum = valueShape->GetStorageShape().GetDim(0);
        valueHeadNum = valueShape->GetStorageShape().GetDim(1);
        valueBlockSize = valueShape->GetStorageShape().GetDim(2);
        valueHeadDim = valueShape->GetStorageShape().GetDim(3);
    } else if (keyDimNum == 5) { // BND1BD0
        keyBlockNum = keyShape->GetStorageShape().GetDim(0);
        keyHeadNum = keyShape->GetStorageShape().GetDim(1);
        keyD1 = keyShape->GetStorageShape().GetDim(2);
        keyBlockSize = keyShape->GetStorageShape().GetDim(3);
        keyD0 = keyShape->GetStorageShape().GetDim(4);

        valueBlockNum = valueShape->GetStorageShape().GetDim(0);
        valueHeadNum = valueShape->GetStorageShape().GetDim(1);
        valueD1 = valueShape->GetStorageShape().GetDim(2);
        valueBlockSize = valueShape->GetStorageShape().GetDim(3);
        valueD0 = valueShape->GetStorageShape().GetDim(4);
    } else {
        OP_LOGE(fiaInfo.opName, "The dim num of key only support [%u, %u], but current is %zu.",
                INPUT_KV_SHAPE_MIN_DIMS, INPUT_KV_SHAPE_MAX_DIMS, keyDimNum);
        return ge::GRAPH_FAILED;
    }

    if (fiaInfo.socVersion == platform_ascendc::SocVersion::ASCEND910B && fiaInfo.kvLayout == FiaLayout::NZ &&
        fiaInfo.ropeMode == RopeMode::NO_ROPE) {
        const std::vector<std::int32_t> nzNoRopeDSupportList = {64, 128};
        if (std::find(nzNoRopeDSupportList.begin(), nzNoRopeDSupportList.end(), keyHeadDim) ==
                nzNoRopeDSupportList.end() ||
            std::find(nzNoRopeDSupportList.begin(), nzNoRopeDSupportList.end(), valueHeadDim) ==
                nzNoRopeDSupportList.end()) {
            OP_LOGE(fiaInfo.opName,
                    "In %s %s situation, when the dim of key&value is 5, and the headDim shared by query and key is "
                    "equal to that of value. The headDim of query|key|value should be 64 | "
                    "128, but got valueHeadDim:%u, queryHeadDim and keyHeadDim:%u",
                    QuantModeToSerialString(fiaInfo.quantMode).c_str(),
                    SituationToSerialString(fiaInfo.ropeMode).c_str(), valueHeadDim, keyHeadDim);
            return ge::GRAPH_FAILED;
        }
    }

    OP_CHECK_IF((keyDimNum == 5) && ((keyBlockNum != valueBlockNum) || (keyHeadNum != valueHeadNum) ||
                                     ((keyD1 != valueD1) && (fiaInfo.mlaMode != MlaMode::ROPE_COMBINE_D128)) ||
                                     (keyBlockSize != valueBlockSize) ||
                                     ((keyD0 != valueD0) && (fiaInfo.mlaMode != MlaMode::ROPE_COMBINE_D128))),
                OPS_REPORT_VECTOR_INNER_ERR(
                    fiaInfo.opName,
                    "The dim num of key and value are inconsistent when PA enable. key shape [%ld, %ld, %ld, %ld, %ld],"
                    " value shape [%ld, %ld, %ld, %ld, %ld].",
                    keyBlockNum, keyHeadNum, keyD1, keyBlockSize, keyD0, valueBlockNum, valueHeadNum, valueD1,
                    valueBlockSize, valueD0),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((keyDimNum == 4) && ((keyBlockNum != valueBlockNum) || (keyHeadNum != valueHeadNum) ||
                                     (keyBlockSize != valueBlockSize) ||
                                     ((keyHeadDim != valueHeadDim) && (fiaInfo.mlaMode != MlaMode::ROPE_COMBINE_D128))),
                OPS_REPORT_VECTOR_INNER_ERR(
                    fiaInfo.opName,
                    "the dim num of key and value are inconsistent when PA enable. key shape [%ld,%ld,%ld,%ld],"
                    " value shape [%ld,%ld,%ld,%ld].",
                    keyBlockNum, keyHeadNum, keyBlockSize, keyHeadDim, valueBlockNum, valueHeadNum, valueBlockSize,
                    valueHeadDim),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((keyDimNum == 3) && ((keyBlockNum != valueBlockNum) || (keyBlockSize != valueBlockSize) ||
                                     ((keyH != valueH) && (fiaInfo.mlaMode != MlaMode::ROPE_COMBINE_D128))),
                OPS_REPORT_VECTOR_INNER_ERR(fiaInfo.opName,
                                            "the dim num of key and value are inconsistent when PA enable. key shape "
                                            "[%ld,%ld,%ld], value shape [%ld,%ld,%ld].",
                                            keyBlockNum, keyBlockSize, keyH, valueBlockNum, valueBlockSize, valueH),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(keyHeadDim != fiaInfo.qkHeadDim,
                OP_LOGE(fiaInfo.opName, "The headDim of key is %u, it should be %u.", keyHeadDim, fiaInfo.qkHeadDim),
                ge::GRAPH_FAILED);
    OP_CHECK_IF(valueHeadDim != fiaInfo.vHeadDim,
                OP_LOGE(fiaInfo.opName, "The headDim of value is %u, it should be %u.", valueHeadDim, fiaInfo.vHeadDim),
                ge::GRAPH_FAILED);
    OP_CHECK_IF(
        keyH != fiaInfo.qkHeadDim * fiaInfo.n2Size,
        OP_LOGE(fiaInfo.opName, "The H of key is %u, it should be %u.", keyH, fiaInfo.qkHeadDim * fiaInfo.n2Size),
        ge::GRAPH_FAILED);
    OP_CHECK_IF(
        valueH != fiaInfo.vHeadDim * fiaInfo.n2Size,
        OP_LOGE(fiaInfo.opName, "The H of value is %u, it should be %u.", valueH, fiaInfo.qkHeadDim * fiaInfo.n2Size),
        ge::GRAPH_FAILED);
    if (enableNonQuant_) {
        OP_CHECK_IF(keyD0 != 16 || valueD0 != 16,
                    OP_LOGE(fiaInfo.opName,
                            "The D0 of key is %u, the D0 of value is %u, "
                            "they both should be 16",
                            keyD0, valueD0),
                    ge::GRAPH_FAILED);
        OP_CHECK_IF(keyD1 != fiaInfo.qkHeadDim / 16,
                    OP_LOGE(fiaInfo.opName, "The D1 of key is %u, it should be %u.", keyD1, fiaInfo.qkHeadDim / 16),
                    ge::GRAPH_FAILED);
        OP_CHECK_IF(valueD1 != fiaInfo.vHeadDim / 16,
                    OP_LOGE(fiaInfo.opName, "The D1 of value is %u, it should be %u.", valueD1, fiaInfo.vHeadDim / 16),
                    ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

bool CommonChecker::CheckEmptyTensorList(const FiaTilingInfo &fiaInfo)
{
    for (int64_t tmpIdx = 0; tmpIdx < fiaInfo.kCache.size(); ++tmpIdx) {
        if (fiaInfo.kCache[tmpIdx]->GetStorageShape().GetShapeSize() != 0) {
            return false;
        }
        if (fiaInfo.vCache[tmpIdx]->GetStorageShape().GetShapeSize() != 0) {
            return false;
        }
    }
    return true;
}

bool CommonChecker::CheckNormalTensorList(const FiaTilingInfo &fiaInfo)
{
    std::string layoutStr(fiaInfo.opParamInfo.layOut);
    if (layoutStr == "BSH") { // check all H across batches and KVs are the same under BSH layout
        auto standardKH = fiaInfo.kCache[0]->GetStorageShape().GetDim(2);
        auto standardVH = fiaInfo.vCache[0]->GetStorageShape().GetDim(2);
        int64_t tmpNKv = (fiaInfo.n2Size != 0) ? fiaInfo.n2Size : fiaInfo.n1Size;
        int64_t keyRopeS = 0;
        if (fiaInfo.opParamInfo.keyRope.tensor != nullptr) {
            keyRopeS = fiaInfo.opParamInfo.keyRope.tensor->GetStorageShape().GetDim(1);
            OP_CHECK_IF(
                fiaInfo.opParamInfo.keyRope.tensor->GetStorageShape().GetDim(0) != fiaInfo.kCache.size(),
                OPS_REPORT_VECTOR_INNER_ERR(
                    fiaInfo.opName, "Batch of Key(%ld) do NOT equal to Batch of KeyRope(%ld) under tensorlist mode!",
                    fiaInfo.kCache.size(), fiaInfo.opParamInfo.keyRope.tensor->GetStorageShape().GetDim(0)),
                return false);
        }

        for (int64_t tmpIdx = 0; tmpIdx < fiaInfo.kCache.size(); ++tmpIdx) {
            // 2: The second dimension of the tensorlist represents n, in order to check whether all n in the tensorlist
            // are the same.
            if (fiaInfo.kCache[tmpIdx]->GetStorageShape().GetDim(2) != standardKH) {
                OP_LOGE(fiaInfo.opName,
                        "H of Key(%ld) in the %ld-th batch is different from H of Key(%ld) in the first batch under "
                        "tensorlist mode!",
                        fiaInfo.kCache[tmpIdx]->GetStorageShape().GetDim(2), tmpIdx + 1, standardKH);
                return false;
            }
            if (fiaInfo.vCache[tmpIdx]->GetStorageShape().GetDim(2) != standardVH) {
                OP_LOGE(fiaInfo.opName,
                        "H of Value(%ld) in the %ld-th batch is different from H of Value(%ld) in the first batch "
                        "under tensorlist mode!",
                        fiaInfo.vCache[tmpIdx]->GetStorageShape().GetDim(2), tmpIdx + 1, standardVH);
                return false;
            }
            if (fiaInfo.kCache[tmpIdx]->GetStorageShape().GetDim(1) !=
                fiaInfo.vCache[tmpIdx]->GetStorageShape().GetDim(1)) { // k_s != v_s
                OP_LOGE(fiaInfo.opName, "S for Key(%ld) and Value(%ld) does NOT equal!",
                        fiaInfo.kCache[tmpIdx]->GetStorageShape().GetDim(1),
                        fiaInfo.vCache[tmpIdx]->GetStorageShape().GetDim(1));
                return false;
            }
            if (fiaInfo.opParamInfo.keyRope.tensor != nullptr) {
                if (fiaInfo.kCache[tmpIdx]->GetStorageShape().GetDim(1) != keyRopeS) { // k_s != krope_s
                    OP_LOGE(fiaInfo.opName, "S for Key(%ld) and keyRope(%ld) does NOT equal but they should!",
                            fiaInfo.kCache[tmpIdx]->GetStorageShape().GetDim(1), keyRopeS);
                    return false;
                }
            }
        }
    } else if (layoutStr == "BNSD" || layoutStr == "BNSD_BSND") { // check N and D, respectively, are the same
        // across batches and KVs under BNSD/BNSD_BSND
        auto standardN = fiaInfo.kCache[0]->GetStorageShape().GetDim(1);
        auto standardKD = fiaInfo.kCache[0]->GetStorageShape().GetDim(3);
        auto standardVD = fiaInfo.vCache[0]->GetStorageShape().GetDim(3);
        int64_t tmpNKv = (fiaInfo.n2Size != 0) ? fiaInfo.n2Size : fiaInfo.n1Size;
        int64_t keyRopeS = 0;

        if (fiaInfo.opParamInfo.keyRope.tensor != nullptr) {
            keyRopeS = fiaInfo.opParamInfo.keyRope.tensor->GetStorageShape().GetDim(2);
            OP_CHECK_IF(
                fiaInfo.opParamInfo.keyRope.tensor->GetStorageShape().GetDim(0) != fiaInfo.kCache.size(),
                OPS_REPORT_VECTOR_INNER_ERR(
                    fiaInfo.opName, "Batch of Key(%ld) do NOT equal to Batch of KeyRope(%ld) under tensorlist mode!",
                    fiaInfo.kCache.size(), fiaInfo.opParamInfo.keyRope.tensor->GetStorageShape().GetDim(0)),
                return false);
        }

        if (tmpNKv != standardN) {
            OP_LOGE(fiaInfo.opName, "N of Key(%ld) in the first batch is different from numKeyValueHeads(%ld)!",
                    standardN, tmpNKv);
            return false;
        }

        for (int64_t tmpIdx = 0; tmpIdx < fiaInfo.kCache.size(); ++tmpIdx) {
            if ((fiaInfo.kCache[tmpIdx]->GetStorageShape().GetDim(1) != standardN) ||
                (fiaInfo.vCache[tmpIdx]->GetStorageShape().GetDim(1) != standardN)) {
                OP_LOGE(fiaInfo.opName,
                        "N of Key(%ld) and Value(%ld) in the %ld-th batch is different from numKeyValueHeads(%ld) "
                        "under tensorlist mode!",
                        fiaInfo.kCache[tmpIdx]->GetStorageShape().GetDim(1),
                        fiaInfo.vCache[tmpIdx]->GetStorageShape().GetDim(1), tmpIdx + 1, standardN);
                return false;
            }
            if (fiaInfo.kCache[tmpIdx]->GetStorageShape().GetDim(3) != standardKD) {
                OP_LOGE(fiaInfo.opName,
                        "D of Key(%ld) in the %ld-th batch is different from D of Key(%ld) in the first batch under "
                        "tensorlist mode!",
                        fiaInfo.kCache[tmpIdx]->GetStorageShape().GetDim(3), tmpIdx + 1, standardKD);
                return false;
            }
            if (fiaInfo.vCache[tmpIdx]->GetStorageShape().GetDim(3) != standardVD) {
                OP_LOGE(fiaInfo.opName,
                        "D of Value(%ld) in the %ld-th batch is different from D of Value(%ld) in the first batch "
                        "under tensorlist mode!",
                        fiaInfo.vCache[tmpIdx]->GetStorageShape().GetDim(3), tmpIdx + 1, standardVD);
                return false;
            }
            if (fiaInfo.kCache[tmpIdx]->GetStorageShape().GetDim(2) != // 2: Obtain the second dimension
                fiaInfo.vCache[tmpIdx]->GetStorageShape().GetDim(2)) { // 2: Obtain the second dimension
                OP_LOGE(fiaInfo.opName, "S from Key(%ld) and Value(%ld) does NOT equal but they should!",
                        fiaInfo.kCache[tmpIdx]->GetStorageShape().GetDim(2),
                        fiaInfo.vCache[tmpIdx]->GetStorageShape().GetDim(2));
                return false;
            }
            if (fiaInfo.opParamInfo.keyRope.tensor != nullptr) {
                if (fiaInfo.kCache[tmpIdx]->GetStorageShape().GetDim(2) != keyRopeS) { // k_s != krope_s
                    OP_LOGE(fiaInfo.opName, "S for Key(%ld) and keyRope(%ld) does NOT equal but they should!",
                            fiaInfo.kCache[tmpIdx]->GetStorageShape().GetDim(2), keyRopeS);
                    return false;
                }
            }
        }
    } else { // check N and D, respectively, are the same across batches and KVs under BSND
        auto standardN = fiaInfo.kCache[0]->GetStorageShape().GetDim(2);
        auto standardKD = fiaInfo.kCache[0]->GetStorageShape().GetDim(3);
        auto standardVD = fiaInfo.vCache[0]->GetStorageShape().GetDim(3);
        int64_t tmpNKv = (fiaInfo.n2Size != 0) ? fiaInfo.n2Size : fiaInfo.n1Size;
        int64_t keyRopeS = 0;

        if (fiaInfo.opParamInfo.keyRope.tensor != nullptr) {
            keyRopeS = fiaInfo.opParamInfo.keyRope.tensor->GetStorageShape().GetDim(1);
            OP_CHECK_IF(
                fiaInfo.opParamInfo.keyRope.tensor->GetStorageShape().GetDim(0) != fiaInfo.kCache.size(),
                OPS_REPORT_VECTOR_INNER_ERR(
                    fiaInfo.opName, "Batch of Key(%ld) do NOT equal to Batch of KeyRope(%ld) under tensorlist mode!",
                    fiaInfo.kCache.size(), fiaInfo.opParamInfo.keyRope.tensor->GetStorageShape().GetDim(0)),
                return false);
        }

        if (tmpNKv != standardN) {
            OP_LOGE(fiaInfo.opName, "N of Key(%ld) in the first batch is different from numKeyValueHeads(%ld)!",
                    standardN, tmpNKv);
            return false;
        }

        for (int64_t tmpIdx = 0; tmpIdx < fiaInfo.kCache.size(); ++tmpIdx) {
            if ((fiaInfo.kCache[tmpIdx]->GetStorageShape().GetDim(2) !=
                 standardN) || // 2: The second dimension of the tensorlist represents n, in order to check whether all
                               // n in the tensorlist are the same.
                (fiaInfo.vCache[tmpIdx]->GetStorageShape().GetDim(2) !=
                 standardN)) { // 2: The second dimension of the tensorlist represents n, in order to check whether all
                               // n in the tensorlist are the same.
                OP_LOGE(fiaInfo.opName,
                        "N of Key(%ld) and Value(%ld) in the %ld-th batch is different from numKeyValueHeads(%ld) "
                        "under tensorlist mode!",
                        fiaInfo.kCache[tmpIdx]->GetStorageShape().GetDim(2),
                        fiaInfo.vCache[tmpIdx]->GetStorageShape().GetDim(2), tmpIdx + 1, standardN);
                return false;
            }
            if (fiaInfo.kCache[tmpIdx]->GetStorageShape().GetDim(3) != standardKD) {
                OP_LOGE(fiaInfo.opName,
                        "D of Key(%ld) in the %ld-th batch is different from D of Key(%ld) in the first batch under "
                        "tensorlist mode!",
                        fiaInfo.kCache[tmpIdx]->GetStorageShape().GetDim(3), tmpIdx + 1, standardKD);
                return false;
            }
            if (fiaInfo.vCache[tmpIdx]->GetStorageShape().GetDim(3) != standardVD) {
                OP_LOGE(fiaInfo.opName,
                        "D of Value(%ld) in the %ld-th batch is different from D of Value(%ld) in the first batch "
                        "under tensorlist mode!",
                        fiaInfo.vCache[tmpIdx]->GetStorageShape().GetDim(3), tmpIdx + 1, standardVD);
                return false;
            }
            if (fiaInfo.kCache[tmpIdx]->GetStorageShape().GetDim(1) !=
                fiaInfo.vCache[tmpIdx]->GetStorageShape().GetDim(1)) {
                OP_LOGE(fiaInfo.opName, "S from Key(%ld) and Value(%ld) does NOT equal but they should!",
                        fiaInfo.kCache[tmpIdx]->GetStorageShape().GetDim(1),
                        fiaInfo.vCache[tmpIdx]->GetStorageShape().GetDim(1));
                return false;
            }
            if (fiaInfo.opParamInfo.keyRope.tensor != nullptr) {
                if (fiaInfo.kCache[tmpIdx]->GetStorageShape().GetDim(1) != keyRopeS) { // k_s != krope_s
                    OP_LOGE(fiaInfo.opName, "S for Key(%ld) and keyRope(%ld) does NOT equal but they should!",
                            fiaInfo.kCache[tmpIdx]->GetStorageShape().GetDim(1), keyRopeS);
                    return false;
                }
            }
        }
    }
    return true;
}

ge::graphStatus CommonChecker::CheckTensorList(const FiaTilingInfo &fiaInfo)
{
    std::string layoutStr(fiaInfo.opParamInfo.layOut);
    OP_CHECK_IF(
        (fiaInfo.opParamInfo.blockTable.tensor != nullptr),
        OPS_REPORT_VECTOR_INNER_ERR(fiaInfo.opName, "When tensorlist is used, page attention is not supported!"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((layoutStr == "TND" || layoutStr == "NTD" || layoutStr == "NTD_TND" || layoutStr == "TND_NTD"),
                OPS_REPORT_VECTOR_INNER_ERR(
                    fiaInfo.opName, "When tensorlist is used, layout TND/NTD/TND_NTD/NTD_TND is not supported!"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((fiaInfo.bSize > B_LIMIT),
                OPS_REPORT_VECTOR_INNER_ERR(
                    fiaInfo.opName, "Batch of Query(%ld) should NOT be larger than 65535 under tensorlist mode!",
                    fiaInfo.bSize),
                return ge::GRAPH_FAILED);
    if (CheckEmptyTensorList(fiaInfo)) {
        return ge::GRAPH_SUCCESS;
    }
    if (!CheckNormalTensorList(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if ((CheckQueryKeyTensorlistConsistency(fiaInfo)) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckMultiDtype(const FiaTilingInfo &fiaInfo)
{
    const std::map<std::string, std::vector<ge::DataType>> QKVD_Different_MAP = {
        {"query", {ge::DT_FLOAT16, ge::DT_BF16}},
        {"key", {ge::DT_FLOAT16, ge::DT_BF16}},
        {"value", {ge::DT_FLOAT16, ge::DT_BF16}},
        {"attentionOut", {ge::DT_FLOAT16, ge::DT_BF16}},
    };
    OP_CHECK_IF(fiaInfo.isQKVDDifferent &&
                    ge::GRAPH_SUCCESS != CheckDtypeCommon(fiaInfo.opParamInfo.query.desc, "query", QKVD_Different_MAP),
                OPS_REPORT_VECTOR_INNER_ERR(fiaInfo.opName, "Query data type must be float16 or bf16 when query and "
                                                            "key headdim is not equal to value headdim."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(fiaInfo.isQKVDDifferent && ge::GRAPH_SUCCESS != CheckDtypeCommon(fiaInfo.opParamInfo.attenOut.desc,
                                                                                 "attentionOut", QKVD_Different_MAP),
                OPS_REPORT_VECTOR_INNER_ERR(fiaInfo.opName, "Output data type must be float16 or bf16 when query and "
                                                            "key headdim is not equal to value headdim."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(fiaInfo.isQKVDDifferent &&
                    ge::GRAPH_SUCCESS != CheckDtypeCommon(fiaInfo.opParamInfo.key.desc, "key", QKVD_Different_MAP),
                OPS_REPORT_VECTOR_INNER_ERR(fiaInfo.opName, "Key data type must be float16 or bf16 when query and "
                                                            "key headdim is not equal to value headdim."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(fiaInfo.isQKVDDifferent &&
                    ge::GRAPH_SUCCESS != CheckDtypeCommon(fiaInfo.opParamInfo.value.desc, "value", QKVD_Different_MAP),
                OPS_REPORT_VECTOR_INNER_ERR(fiaInfo.opName, "Value data type must be float16 or bf16 when query and "
                                                            "key headdim is not equal to value headdim."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckAxis(const FiaTilingInfo &fiaInfo)
{
    OP_CHECK_IF(fiaInfo.bSize > B_LIMIT || fiaInfo.bSize <= 0,
                OP_LOGE(fiaInfo.opName, "The axis B only support (0, %u], the current is %u.", B_LIMIT, fiaInfo.bSize),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(fiaInfo.s1Size < 0,
                OP_LOGE(fiaInfo.opName, "The axis S of query only support >=0, the current is %ld.", fiaInfo.s1Size),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        fiaInfo.s2Size < 0,
        OP_LOGE(fiaInfo.opName, "The axis S of key and value only support >=0, the current is %ld.", fiaInfo.s2Size),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(fiaInfo.qkHeadDim > D_LIMIT || fiaInfo.qkHeadDim < 0,
                OP_LOGE(fiaInfo.opName, "The axis D of query and key only support (0, %u], the current is %u.", D_LIMIT,
                        fiaInfo.qkHeadDim),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(fiaInfo.vHeadDim > D_LIMIT || fiaInfo.vHeadDim < 0,
                OP_LOGE(fiaInfo.opName, "The axis D of value only support (0, %u], the current is %u.", D_LIMIT,
                        fiaInfo.vHeadDim),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(fiaInfo.isQKVDDifferent && (fiaInfo.qkHeadDim > 128),
                OPS_REPORT_VECTOR_INNER_ERR(fiaInfo.opName, "Query headdim must smaller than 128 when query and key "
                                                            "headdim is not equal to value headdim."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(fiaInfo.isQKVDDifferent && (fiaInfo.vHeadDim > 128),
                OPS_REPORT_VECTOR_INNER_ERR(fiaInfo.opName, "Value headdim must smaller than 128 when query and key "
                                                            "headdim is not equal to value headdim."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        fiaInfo.qLayout == FiaLayout::TND && fiaInfo.qTSize < 0,
        OP_LOGE(fiaInfo.opName, "The axis T of query only support >0 when input layout is TND, the current is %u",
                fiaInfo.qTSize),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        fiaInfo.kvLayout == FiaLayout::TND && fiaInfo.kTSize < 0,
        OP_LOGE(fiaInfo.opName, "The axis T of key only support >0 when input layout is TND, the current is %u.",
                fiaInfo.kTSize),
        return ge::GRAPH_FAILED);
    if (fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D512) {
        OP_CHECK_IF(fiaInfo.s1Size < 1,
                    OP_LOGE(fiaInfo.opName,
                            "input query's sequence length is %u, it should be "
                            ">=1 when enable ifa mla",
                            fiaInfo.s1Size),
                    return ge::GRAPH_FAILED);
        static const std::set<uint32_t> SUPPORT_G_IN_IFAMLA = {1U,  2U,  4U,  8U,
                                                               16U, 32U, 64U, 128U}; // ifa mla场景g轴支持范围
        OP_CHECK_IF((SUPPORT_G_IN_IFAMLA.find(fiaInfo.gSize) == SUPPORT_G_IN_IFAMLA.end()),
                    OPS_REPORT_VECTOR_INNER_ERR(fiaInfo.opName,
                                                "The axis G should be in range of "
                                                "{1, 2, 4, 8, 16, 32, 64, 128} when enable ifa mla, the current is %u.",
                                                fiaInfo.n1Size / fiaInfo.n2Size),
                    return ge::GRAPH_FAILED);
    }
    OP_LOGI(fiaInfo.opName, "The axis B(%u), qkD(%u), vD(%u), G(%u), qT(%u), kT(%u).", fiaInfo.bSize, fiaInfo.qkHeadDim,
            fiaInfo.vHeadDim, fiaInfo.gSize, fiaInfo.qTSize, fiaInfo.kTSize);
    return ge::GRAPH_SUCCESS;
}

void CommonChecker::GetQueryDimAndOutDim(const gert::StorageShape *queryShape, const gert::StorageShape *outShape,
                                         const std::string &layoutStr, int64_t &tmpQueryDim, int64_t &outDim,
                                         uint32_t i)
{
    if (layoutStr == "BNSD_BSND" || layoutStr == "BSND_BNSD") {
        if (i == 1) { // BNSD_BSND：query:N, output:S; BSND_BNSD：query:S, output:N
            tmpQueryDim = queryShape->GetStorageShape().GetDim(i);
            outDim = outShape->GetStorageShape().GetDim(i + 1);
        } else if (i == 2) { // BNSD_BSND：query:S, output:N; BSND_BNSD：query:N, output:S
            tmpQueryDim = queryShape->GetStorageShape().GetDim(i);
            outDim = outShape->GetStorageShape().GetDim(i - 1);
        }
    } else if (layoutStr == "BSH_BNSD") {
        if (i == 2) { // BSH_BNSD：query:H, output:ND
            tmpQueryDim = queryShape->GetStorageShape().GetDim(i);
            outDim = outShape->GetStorageShape().GetDim(i + 1) * outShape->GetStorageShape().GetDim(i - 1);
        }
    } else if (layoutStr == "BSND_NBSD") {
        if (i == 1) { // BSND_NBSD：query:S, output:B
            tmpQueryDim = queryShape->GetStorageShape().GetDim(i);
            outDim = outShape->GetStorageShape().GetDim(i + 1);
        } else if (i == 2) { // BSND_NBSD：query:N, output:S
            tmpQueryDim = queryShape->GetStorageShape().GetDim(i);
            outDim = outShape->GetStorageShape().GetDim(i - 2);
        }
    } else if (layoutStr == "BNSD_NBSD") {
        if (i == 1) { // BNSD_NBSD：query:N, output:B
            tmpQueryDim = queryShape->GetStorageShape().GetDim(i);
            outDim = outShape->GetStorageShape().GetDim(i - 1);
        } else if (i == 2) { // BNSD_NBSD：query:S, output:S
            tmpQueryDim = queryShape->GetStorageShape().GetDim(i);
            outDim = outShape->GetStorageShape().GetDim(i);
        }
    } else if (layoutStr == "BSH_NBSD") {
        if (i == 2) { // BSH_NBSD：query:H, output:ND
            tmpQueryDim = queryShape->GetStorageShape().GetDim(i);
            outDim = outShape->GetStorageShape().GetDim(i + 1) * outShape->GetStorageShape().GetDim(i - 2);
        }
    } else if (layoutStr == "NTD_TND" || layoutStr == "TND_NTD") {
        if (i == 0) { // query:N/T, output:T/N
            tmpQueryDim = queryShape->GetStorageShape().GetDim(i);
            outDim = outShape->GetStorageShape().GetDim(i + 1);
        } else if (i == 1) { // 2 for current queryDimNum; Q:T/N, Output:N/T
            tmpQueryDim = queryShape->GetStorageShape().GetDim(i);
            outDim = outShape->GetStorageShape().GetDim(i - 1);
        }
    } else {
        tmpQueryDim = queryShape->GetStorageShape().GetDim(i);
        outDim = outShape->GetStorageShape().GetDim(i);
    }
}

ge::graphStatus CommonChecker::CheckQueryOutConsistency(const FiaTilingInfo &fiaInfo)
{
    const gert::StorageShape *queryShape = fiaInfo.opParamInfo.query.shape;
    const gert::StorageShape *attentionOutShape = fiaInfo.opParamInfo.attenOut.shape;
    size_t dimNumQ = queryShape->GetStorageShape().GetDimNum();
    size_t dimNumOut = attentionOutShape->GetStorageShape().GetDimNum();
    int64_t tmpQueryDim = 0;
    int64_t tmpOutDim = 0;
    std::string layoutStr(fiaInfo.opParamInfo.layOut);
    bool isLayoutShapeSupport = layoutStr == "BSH_BNSD" || layoutStr == "BSH_NBSD";
    OP_CHECK_IF(dimNumQ != dimNumOut && !isLayoutShapeSupport,
                OP_LOGE(fiaInfo.opName,
                        "The dim num of query should be consistent with the dim num of attentionOut,"
                        "but current dim num of query is %zu, dim num of attentionOut is %zu.",
                        dimNumQ, dimNumOut),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(dimNumQ < INPUT_Q_SHAPE_MIN_DIMS || dimNumQ > INPUT_Q_SHAPE_MAX_DIMS,
                OP_LOGE(fiaInfo.opName, "The dim num of query only support [%u, %u], but current is %zu.",
                        INPUT_Q_SHAPE_MIN_DIMS, INPUT_Q_SHAPE_MAX_DIMS, dimNumQ),
                return ge::GRAPH_FAILED);

    for (uint32_t queryDimIdx = 0; queryDimIdx < dimNumQ; ++queryDimIdx) {
        if ((queryDimIdx == dimNumQ - 1) && fiaInfo.mlaMode == MlaMode::ROPE_COMBINE_D128) {
            continue;
        }
        GetQueryDimAndOutDim(queryShape, attentionOutShape, layoutStr, tmpQueryDim, tmpOutDim, queryDimIdx);
        OP_CHECK_IF(
            !fiaInfo.isQKVDDifferent && (tmpQueryDim != tmpOutDim),
            OPS_REPORT_VECTOR_INNER_ERR(
                fiaInfo.opName, "tensor query shape (%ld) do not equal to tensor output shape(%ld) in dim %u for %s.",
                tmpQueryDim, tmpOutDim, queryDimIdx, layoutStr.c_str()),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(
            fiaInfo.isQKVDDifferent && (queryDimIdx != dimNumQ - 1) && (tmpQueryDim != tmpOutDim),
            OPS_REPORT_VECTOR_INNER_ERR(
                fiaInfo.opName, "tensor query shape (%ld) do not equal to tensor output shape(%ld) in dim %u for %s.",
                tmpQueryDim, tmpOutDim, queryDimIdx, layoutStr.c_str()),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckKeyValueConsistency(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.kvStorageMode == KvStorageMode::PAGE_ATTENTION || fiaInfo.kvStorageMode == KvStorageMode::TENSOR_LIST) {
        return ge::GRAPH_SUCCESS;
    }
    const gert::StorageShape *keyShape = fiaInfo.opParamInfo.key.shape;
    const gert::StorageShape *valueShape = fiaInfo.opParamInfo.value.shape;
    ge::DataType keyDataType = fiaInfo.opParamInfo.key.desc->GetDataType();
    ge::DataType valueDataType = fiaInfo.opParamInfo.value.desc->GetDataType();

    const size_t keyDimNum = keyShape->GetStorageShape().GetDimNum();
    const size_t valueDimNum = valueShape->GetStorageShape().GetDimNum();
    OP_CHECK_IF((keyDataType != valueDataType),
                OP_LOGE(fiaInfo.opName, "The data type of key is %s,the data type of value is %s",
                        DataTypeToSerialString(keyDataType).c_str(), DataTypeToSerialString(valueDataType).c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(keyDimNum != valueDimNum,
                OP_LOGE(fiaInfo.opName,
                        "The dim num of key should be consistent with the dim num of value,"
                        "but current dim num of key is %zu, dim num of value is %zu.",
                        keyDimNum, valueDimNum),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(keyDimNum < INPUT_KV_SHAPE_MIN_DIMS || keyDimNum > INPUT_KV_SHAPE_MAX_DIMS,
                OP_LOGE(fiaInfo.opName, "The dim num of key only support [%u, %u], but current is %zu.",
                        INPUT_KV_SHAPE_MIN_DIMS, INPUT_KV_SHAPE_MAX_DIMS, keyDimNum),
                return ge::GRAPH_FAILED);

    for (uint32_t i = 0; i < keyDimNum; ++i) {
        if ((i == keyDimNum - 1) && fiaInfo.mlaMode == MlaMode::ROPE_COMBINE_D128) {
            continue;
        }
        int64_t tmpKeyDim = keyShape->GetStorageShape().GetDim(i);
        int64_t tmpValueDim = valueShape->GetStorageShape().GetDim(i);
        OP_CHECK_IF(!fiaInfo.isQKVDDifferent && (tmpKeyDim != tmpValueDim),
                    OPS_REPORT_VECTOR_INNER_ERR(
                        fiaInfo.opName, "tensor key shape(%ld) do not equal to tensor value shape(%ld) in dim %u.",
                        tmpKeyDim, tmpValueDim, i),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(fiaInfo.isQKVDDifferent && (i != keyDimNum - 1) && (tmpKeyDim != tmpValueDim),
                    OPS_REPORT_VECTOR_INNER_ERR(
                        fiaInfo.opName, "tensor key shape(%ld) do not equal to tensor value shape(%ld) in dim %u.",
                        tmpKeyDim, tmpValueDim, i),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckQueryShape(const FiaTilingInfo &fiaInfo)
{
    uint32_t attrN = fiaInfo.n1Size;
    const gert::StorageShape *queryShape = fiaInfo.opParamInfo.query.shape;
    uint32_t queryShapeHeadNum = attrN;
    size_t queryDim = queryShape->GetStorageShape().GetDimNum();
    int64_t queryH = 0;

    if (fiaInfo.qLayout == FiaLayout::BNSD || fiaInfo.qLayout == FiaLayout::TND) {
        queryShapeHeadNum = queryShape->GetStorageShape().GetDim(1);
    } else if (fiaInfo.qLayout == FiaLayout::BSND) {
        queryShapeHeadNum = queryShape->GetStorageShape().GetDim(2);
    } else if (fiaInfo.qLayout == FiaLayout::BSH) {
        queryH = queryShape->GetStorageShape().GetDim(2);
    } else if (fiaInfo.qLayout == FiaLayout::NTD) {
        queryShapeHeadNum = queryShape->GetStorageShape().GetDim(0);
    }
    OP_CHECK_IF(attrN != queryShapeHeadNum,
                OP_LOGE(fiaInfo.opName, "The attr numHeads is %u, the axis N of query is %u, they should be equal.",
                        attrN, queryShapeHeadNum),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(fiaInfo.qLayout == FiaLayout::BSH && queryH % attrN != 0,
                OP_LOGE(fiaInfo.opName,
                        "When layout is BSH, the axis H of query be an integer multiple of numHeads, "
                        "but current H is %ld numHeads is %u.",
                        queryH, attrN),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((fiaInfo.qLayout == FiaLayout::BNSD || fiaInfo.qLayout == FiaLayout::BSND) && queryDim != 4U,
                OP_LOGE(fiaInfo.opName, "When layout is %s, query dim num is  %u, it should be 4.",
                        fiaInfo.opParamInfo.layOut, queryDim),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((fiaInfo.qLayout == FiaLayout::BSH || fiaInfo.qLayout == FiaLayout::TND) && queryDim != 3U,
                OP_LOGE(fiaInfo.opName, "When layout is %s, query dim num is  %u, it should be 3.",
                        fiaInfo.opParamInfo.layOut, queryDim),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckKeyNHVaild(const FiaTilingInfo &fiaInfo, const gert::Shape &keyShape)
{
    uint32_t attrKvN = fiaInfo.n2Size;
    size_t keyDim = keyShape.GetDimNum();
    uint32_t keyShapeHeadNum = attrKvN;
    int64_t keyH = 0;

    if (fiaInfo.kvLayout == FiaLayout::BNSD || fiaInfo.kvLayout == FiaLayout::TND) {
        keyShapeHeadNum = keyShape.GetDim(1);
    } else if (fiaInfo.kvLayout == FiaLayout::BSND) {
        keyShapeHeadNum = keyShape.GetDim(2);
    } else if (fiaInfo.kvLayout == FiaLayout::BSH) {
        keyH = keyShape.GetDim(2);
    }
    OP_CHECK_IF(
        attrKvN != keyShapeHeadNum,
        OP_LOGE(fiaInfo.opName, "The attr numKeyValueHeads is %u, the axis N of key is %u, they should be equal.",
                attrKvN, keyShapeHeadNum),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(fiaInfo.kvLayout == FiaLayout::BSH && keyH % attrKvN != 0,
                OP_LOGE(fiaInfo.opName,
                        "When layout is BSH, the axis H of key be an integer multiple of numHeads, "
                        "but current H is %ld numKeyValueHeads is %u.",
                        keyH, attrKvN),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((fiaInfo.kvLayout == FiaLayout::BNSD || fiaInfo.kvLayout == FiaLayout::BSND) && keyDim != 4U,
                OP_LOGE(fiaInfo.opName, "When layout is %s, key dim num is  %u, it should be 4.",
                        fiaInfo.opParamInfo.layOut, keyDim),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((fiaInfo.kvLayout == FiaLayout::BSH || fiaInfo.kvLayout == FiaLayout::TND) && keyDim != 3U,
                OP_LOGE(fiaInfo.opName, "When layout is %s, key dim num is  %u, it should be 3.",
                        fiaInfo.opParamInfo.layOut, keyDim),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckKeyDVaild(const FiaTilingInfo &fiaInfo, const gert::Shape &keyShape)
{
    size_t keyDim = keyShape.GetDimNum();
    uint32_t keyHeadDim = 0;
    if (fiaInfo.kvLayout != FiaLayout::BSH) {
        keyHeadDim = keyShape.GetDim(keyDim - 1);
    } else {
        keyHeadDim = keyShape.GetDim(keyDim - 1) / fiaInfo.n2Size;
    }

    OP_CHECK_IF(keyHeadDim != fiaInfo.qkHeadDim,
                OP_LOGE(fiaInfo.opName, "The axis D of key is %u, the axis D of query is %u, they should be equal.",
                        keyHeadDim, fiaInfo.qkHeadDim),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckKeyShape(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.kvStorageMode == KvStorageMode::PAGE_ATTENTION) {
        return ge::GRAPH_SUCCESS;
    }
    if (fiaInfo.kvStorageMode == KvStorageMode::BATCH_CONTINUOUS) {
        return CheckKeyNHVaild(fiaInfo, fiaInfo.opParamInfo.key.shape->GetStorageShape());
    } else {
        for (uint32_t i = 0; i < fiaInfo.kCache.size(); i++) {
            auto keyShape = fiaInfo.kCache[i]->GetStorageShape();
            if (CheckKeyNHVaild(fiaInfo, keyShape) != ge::GRAPH_SUCCESS) {
                OP_LOGE(fiaInfo.opName, "When kv storage mode is tensorlist, the input %d of key shape is invalid", i);
                return ge::GRAPH_FAILED;
            }
        }
        return ge::GRAPH_SUCCESS;
    }
}

ge::graphStatus CommonChecker::CheckQueryKeyTensorlistConsistency(const FiaTilingInfo &fiaInfo)
{
    OP_CHECK_IF(fiaInfo.bSize != fiaInfo.kCache.size(),
                OP_LOGE(fiaInfo.opName,
                        "The axis B of query is %u, the number of key is %zu when tensor list, they should be equal",
                        fiaInfo.bSize, fiaInfo.kCache.size()),
                return ge::GRAPH_FAILED);
    for (uint32_t i = 0; i < fiaInfo.kCache.size(); i++) {
        auto keyShape = fiaInfo.kCache[i]->GetStorageShape();
        if (CheckKeyDVaild(fiaInfo, keyShape) != ge::GRAPH_SUCCESS) {
            OP_LOGE(fiaInfo.opName, "When kv storage mode is tensorlist, the input %d of key shape is invalid", i);
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckQueryKeyConsistency(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.kvStorageMode == KvStorageMode::PAGE_ATTENTION || fiaInfo.kvStorageMode == KvStorageMode::TENSOR_LIST) {
        return ge::GRAPH_SUCCESS;
    }
    ge::DataType queryDataType = fiaInfo.opParamInfo.query.desc->GetDataType();
    ge::DataType keyDataType = fiaInfo.opParamInfo.key.desc->GetDataType();
    if (enableNonQuant_) {
        OP_CHECK_IF((queryDataType != keyDataType),
                    OP_LOGE(fiaInfo.opName, "The data type of query is %s,the data type of key is %s",
                            DataTypeToSerialString(queryDataType).c_str(), DataTypeToSerialString(keyDataType).c_str()),
                    return ge::GRAPH_FAILED);
    }
    int64_t keyB = fiaInfo.opParamInfo.key.shape->GetStorageShape().GetDim(0);
    if (fiaInfo.kvStorageMode == KvStorageMode::BATCH_CONTINUOUS) {
        OP_CHECK_IF((fiaInfo.qLayout != FiaLayout::TND && fiaInfo.qLayout != FiaLayout::NTD) && fiaInfo.bSize != keyB,
                    OP_LOGE(fiaInfo.opName, "The axis B of query is %u, the axis B of key is %ld, they should be equal",
                            fiaInfo.bSize, keyB),
                    return ge::GRAPH_FAILED);
        return CheckKeyDVaild(fiaInfo, fiaInfo.opParamInfo.key.shape->GetStorageShape());
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckMultiAttr(const FiaTilingInfo &fiaInfo)
{
    OP_CHECK_IF(fiaInfo.kvStorageMode == KvStorageMode::PAGE_ATTENTION && fiaInfo.isQKVDDifferent,
                OPS_REPORT_VECTOR_INNER_ERR(fiaInfo.opName,
                                            "Not support PA when query and key headdim is not equal to value headdim."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(fiaInfo.pseShiftFlag && fiaInfo.isQKVDDifferent,
                OPS_REPORT_VECTOR_INNER_ERR(
                    fiaInfo.opName, "Not support pse shift when query and key headdim is not equal to value headdim."),
                return false);
    OP_CHECK_IF(fiaInfo.enableAlibiPse && fiaInfo.isQKVDDifferent,
                OPS_REPORT_VECTOR_INNER_ERR(
                    fiaInfo.opName, "Not support pse shift when query and key headdim is not equal to value headdim."),
                return false);

    if (fiaInfo.inputQType != ge::DT_FLOAT16) {
        OP_LOGW(fiaInfo.opName, "When query input is not fp16,innerPrecise will not take effect");
    }

    if (fiaInfo.qLayout == FiaLayout::TND) {
        OP_CHECK_IF(fiaInfo.kvStorageMode == KvStorageMode::TENSOR_LIST,
                    OP_LOGE(fiaInfo.opName, "When layout is TND, tensorlist is not supported!"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// enableNonQuant 相关校验函数
ge::graphStatus CommonChecker::CheckNonQuantDataType(const FiaTilingInfo &fiaInfo)
{
    const std::map<std::string, std::vector<ge::DataType>> NO_QUANT_MAP = {
        {"query", {ge::DT_FLOAT16, ge::DT_BF16}},
        {"key", {ge::DT_FLOAT16, ge::DT_BF16}},
        {"value", {ge::DT_FLOAT16, ge::DT_BF16}},
        {"attentionOut", {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, ge::DT_HIFLOAT8, ge::DT_FLOAT8_E4M3FN}},
    };
    if (ge::GRAPH_SUCCESS != CheckDtypeCommon(fiaInfo.opParamInfo.query.desc, "query", NO_QUANT_MAP) ||
        ge::GRAPH_SUCCESS != CheckDtypeCommon(fiaInfo.opParamInfo.key.desc, "key", NO_QUANT_MAP) ||
        ge::GRAPH_SUCCESS != CheckDtypeCommon(fiaInfo.opParamInfo.value.desc, "value", NO_QUANT_MAP) ||
        ge::GRAPH_SUCCESS != CheckDtypeCommon(fiaInfo.opParamInfo.attenOut.desc, "attentionOut", NO_QUANT_MAP)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckAttr(const FiaTilingInfo &fiaInfo)
{
    if (CheckHeadNum(fiaInfo) != ge::GRAPH_SUCCESS || CheckInputLayout(fiaInfo) != ge::GRAPH_SUCCESS ||
        CheckInnerPrecise(fiaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckHeadNum(const FiaTilingInfo &fiaInfo)
{
    if ((fiaInfo.n1Size < 0) || (fiaInfo.n2Size < 0)) {
        OP_LOGE(fiaInfo.opName, "numHeads(%d) or numKeyValueHeads(%d) is negative!", fiaInfo.n1Size, fiaInfo.n2Size);
        return ge::GRAPH_FAILED;
    }

    if (fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D512) { // ifamla
        static const std::set<uint32_t> SUPPORT_NUM_HEAD_IN_IFAMLA = {1U,  2U,  4U,  8U,
                                                                      16U, 32U, 64U, 128U}; // ifa mla场景qN支持范围
        OP_CHECK_IF((SUPPORT_NUM_HEAD_IN_IFAMLA.find(fiaInfo.n1Size) == SUPPORT_NUM_HEAD_IN_IFAMLA.end()),
                    OPS_REPORT_VECTOR_INNER_ERR(fiaInfo.opName,
                                                "Input query's heads num is %u, it should be in range of "
                                                "{1, 2, 4, 8, 16, 32, 64, 128} when enable ifa mla",
                                                fiaInfo.n1Size),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF((fiaInfo.n2Size != 1U),
                    OPS_REPORT_VECTOR_INNER_ERR(fiaInfo.opName,
                                                "Input key/value's heads num is %u, it should be 1 when enable "
                                                "ifa mla",
                                                fiaInfo.n2Size),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckInputLayout(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.opParamInfo.layOut == nullptr) {
        return ge::GRAPH_FAILED;
    }
    std::string inputLayout = fiaInfo.opParamInfo.layOut;
    if (fiaInfo.ropeMode == RopeMode::NO_ROPE) {
        if (fiaInfo.socVersion != platform_ascendc::SocVersion::ASCEND910B) {
            const std::vector<std::string> INPUT_LAYOUT_LIST = {"BSH",       "BSND",     "BNSD",    "TND",      "NTD",
                                                                "BSND_BNSD", "BSH_BNSD", "NTD_TND", "BNSD_BSND"};
            if (std::find(INPUT_LAYOUT_LIST.begin(), INPUT_LAYOUT_LIST.end(), inputLayout) == INPUT_LAYOUT_LIST.end()) {
                OP_LOGE(fiaInfo.opName,
                        "When gqa scenario is applied, the attr inputLayout only supports BSH, "
                        "BSND, BNSD, TND, NTD, BSND_BNSD, BSH_BNSD, NTD_TND, BNSD_BSND, but got %s",
                        inputLayout.c_str());
                return ge::GRAPH_FAILED;
            }
        } else {
            const std::vector<std::string> INPUT_LAYOUT_LIST = {
                "BSH",       "BSND",    "BNSD",    "TND",      "NTD",       "BSH_NBSD", "BSND_NBSD",
                "BNSD_NBSD", "TND_NTD", "NTD_TND", "BSH_BNSD", "BSND_BNSD", "BNSD_BSND"};
            OP_CHECK_IF(
                std::find(INPUT_LAYOUT_LIST.begin(), INPUT_LAYOUT_LIST.end(), inputLayout) == INPUT_LAYOUT_LIST.end(),
                OP_LOGE(
                    fiaInfo.opName,
                    "When gqa noquant scenario is applied, layout only supports BSH, BSND, BNSD, TND, "
                    "NTD, BSH_NBSD, BSND_NBSD, BNSD_NBSD, TND_NTD, NTD_TND, BSH_BNSD, BSND_BNSD, BNSD_BSND, but got %s",
                    inputLayout.c_str()),
                return ge::GRAPH_FAILED);
            const std::vector<std::string> noRopeLayoutSupportListA = {"BSH", "BSND", "BNSD"};
            const std::vector<std::string> noRopeLayoutSupportListB = {"BNSD_BSND"};
            const std::vector<std::string> noRopeLayoutSupportListC = {"NTD", "BSH_BNSD", "BSND_BNSD", "NTD_TND"};
            const std::vector<std::string> noRopeLayoutSupportListD = {"TND"};

            if (!fiaInfo.isLegacyIfa &&
                fiaInfo.vHeadDim % 16 != 0) { // 16: qkvD need 16 align when qs>1, in specific input layout
                OP_CHECK_IF(std::find(noRopeLayoutSupportListA.begin(), noRopeLayoutSupportListA.end(), inputLayout) !=
                                noRopeLayoutSupportListA.end(),
                            OP_LOGE(fiaInfo.opName,
                                    "In %s %s situation, when Qs>1 and input_layout is %s, headDim of query|key|value "
                                    "should be align to 16.",
                                    QuantModeToSerialString(fiaInfo.quantMode).c_str(),
                                    SituationToSerialString(fiaInfo.ropeMode).c_str(), inputLayout.c_str()),
                            return ge::GRAPH_FAILED);

                OP_CHECK_IF(std::find(noRopeLayoutSupportListB.begin(), noRopeLayoutSupportListB.end(), inputLayout) !=
                                noRopeLayoutSupportListB.end(),
                            OP_LOGE(fiaInfo.opName,
                                    "In %s %s situation, when Qs>1 and input_layout is %s, headDim of query|key|value "
                                    "should be align to 16.",
                                    QuantModeToSerialString(fiaInfo.quantMode).c_str(),
                                    SituationToSerialString(fiaInfo.ropeMode).c_str(), inputLayout.c_str()),
                            return ge::GRAPH_FAILED);
            }

            if (fiaInfo.isLegacyIfa &&
                (fiaInfo.vHeadDim != 64 &&
                 fiaInfo.vHeadDim != 128)) { // 64: qkvD need 64 128: qkvD need 128 in specific input layout
                OP_CHECK_IF(std::find(noRopeLayoutSupportListB.begin(), noRopeLayoutSupportListB.end(), inputLayout) !=
                                noRopeLayoutSupportListB.end(),
                            OP_LOGE(fiaInfo.opName,
                                    "In %s %s situation, when Qs=1 and input_layout is BNSD_BSND, only query|key|value "
                                    "headDim = 64/128 are supported, but got %u",
                                    QuantModeToSerialString(fiaInfo.quantMode).c_str(),
                                    SituationToSerialString(fiaInfo.ropeMode).c_str(), fiaInfo.vHeadDim),
                            return ge::GRAPH_FAILED);
            }
            if (std::find(noRopeLayoutSupportListC.begin(), noRopeLayoutSupportListC.end(), inputLayout) !=
                noRopeLayoutSupportListC.end()) {
                OP_CHECK_IF(fiaInfo.vHeadDim != 64 &&
                                fiaInfo.vHeadDim !=
                                    128, // 64: qkvD (optional) 64 128: qkvD (optional) 128 in specific input layout
                            OP_LOGE(fiaInfo.opName,
                                    "In %s %s situation, when input_layout is NTD, BSH_BNSD, BSND_BNSD, NTD_TND, only "
                                    "query|key|value headDim = 64/128 are supported, but got %u",
                                    QuantModeToSerialString(fiaInfo.quantMode).c_str(),
                                    SituationToSerialString(fiaInfo.ropeMode).c_str(), fiaInfo.vHeadDim),
                            return ge::GRAPH_FAILED);
            }
            if (std::find(noRopeLayoutSupportListD.begin(), noRopeLayoutSupportListD.end(), inputLayout) !=
                noRopeLayoutSupportListD.end()) {
                OP_CHECK_IF(fiaInfo.vHeadDim != 64 && fiaInfo.vHeadDim != 128 &&
                                fiaInfo.vHeadDim != 192, // 64: qkvD (optional) 64, 128: qkvD (optional) 128, 192: qkvD
                                                         // (optional) 192 when input_layout=TND
                            OP_LOGE(fiaInfo.opName,
                                    "In %s %s situation, when input_layout is TND, only query|key|value headDim = "
                                    "64/128/192 are supported, but got %u",
                                    QuantModeToSerialString(fiaInfo.quantMode).c_str(),
                                    SituationToSerialString(fiaInfo.ropeMode).c_str(), fiaInfo.vHeadDim),
                            return ge::GRAPH_FAILED);
            }
        }
    } else if (fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D512) { // decode mla
        const std::vector<std::string> INPUT_LAYOUT_LIST = {"BSH",       "BSND",      "BNSD",     "TND",
                                                            "BNSD_NBSD", "BSND_NBSD", "BSH_NBSD", "TND_NTD"};
        if (std::find(INPUT_LAYOUT_LIST.begin(), INPUT_LAYOUT_LIST.end(), inputLayout) == INPUT_LAYOUT_LIST.end()) {
            OP_LOGE(fiaInfo.opName,
                    "When decode mla scenario is applied, the attr inputLayout only supports BSH, BSND, BNSD, "
                    "TND ,BNSD_BSND, BSND_NBSD, BSH_NBSD, TND_NTD, but got %s",
                    inputLayout.c_str());
            return ge::GRAPH_FAILED;
        }
    } else { // prefill mla
        const std::vector<std::string> INPUT_LAYOUT_LIST = {"BSH",       "BSND",     "BNSD",    "TND",      "NTD",
                                                            "BSND_BNSD", "BSH_BNSD", "NTD_TND", "BNSD_BSND"};
        if (std::find(INPUT_LAYOUT_LIST.begin(), INPUT_LAYOUT_LIST.end(), inputLayout) == INPUT_LAYOUT_LIST.end()) {
            OP_LOGE(fiaInfo.opName,
                    "When prefill mla scenario is applied, the attr inputLayout only supports BSH, "
                    "BSND, BNSD, TND, NTD, BSND_BNSD, BSH_BNSD, NTD_TND, BNSD_BSND, but got %s",
                    inputLayout.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckInnerPrecise(const FiaTilingInfo &fiaInfo)
{
    OP_CHECK_IF(fiaInfo.innerPrecise > INNER_PRECISE_LIMIT || fiaInfo.innerPrecise < 0,
                OP_LOGE(fiaInfo.opName, "The attr innerPrecise only support [0, %u], the current is %u.",
                        INNER_PRECISE_LIMIT, fiaInfo.innerPrecise),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

bool CommonChecker::CheckTNDLayoutCrossover(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.inputLayout != TilingKeyLayout::TND) {
        return true;
    }

    std::string layoutStr(fiaInfo.opParamInfo.layOut);
    if (fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D512 && layoutStr == "TND_NTD") { // Decode MLA
        OP_CHECK_IF(fiaInfo.isOutQuantEnable,
                    OPS_REPORT_VECTOR_INNER_ERR(
                        fiaInfo.opName, "In Decode MLA scenario, when layout is TND_NTD, post quant is not supported!"),
                    return false);
    }

    OP_CHECK_IF(fiaInfo.kvStorageMode == KvStorageMode::TENSOR_LIST,
                OPS_REPORT_VECTOR_INNER_ERR(fiaInfo.opName, "When layout is TND, tensorlist is not supported!"),
                return false);

    OP_CHECK_IF(fiaInfo.pseShiftFlag,
                OPS_REPORT_VECTOR_INNER_ERR(fiaInfo.opName, "When layout is TND, pse is not supported!"), return false);

    return true;
}

bool CommonChecker::CheckNTDLayoutCrossover(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.inputLayout != TilingKeyLayout::NTD) {
        return true;
    }
    bool isGqa = enableNonQuant_ && !(fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D512) &&
                 !(fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D128) && !(fiaInfo.mlaMode == MlaMode::ROPE_COMBINE_D128);
    std::string layoutStr(fiaInfo.opParamInfo.layOut);
    if (isGqa) { // GQA
        OP_CHECK_IF((fiaInfo.qkHeadDim != 64 && fiaInfo.qkHeadDim != 128),
                    OPS_REPORT_VECTOR_INNER_ERR(
                        fiaInfo.opName,
                        "In GQA scenario, when layout is %s, d size of query must be 64 or 128, but got d = %d.",
                        layoutStr.c_str(), fiaInfo.qkHeadDim),
                    return false);
    }
    if (isGqa) { // GQA
        OP_CHECK_IF(
            fiaInfo.isQKVDDifferent,
            OPS_REPORT_VECTOR_INNER_ERR(
                fiaInfo.opName,
                "In GQA scenario, not support layout %s when query and key headdim is not equal to value headdim.",
                layoutStr.c_str()),
            return false);
    }

    OP_CHECK_IF(fiaInfo.kvStorageMode == KvStorageMode::TENSOR_LIST,
                OPS_REPORT_VECTOR_INNER_ERR(fiaInfo.opName, "When layout is %s, tensorlist is not supported!",
                                            layoutStr.c_str()),
                return false);

    OP_CHECK_IF(
        fiaInfo.pseShiftFlag,
        OPS_REPORT_VECTOR_INNER_ERR(fiaInfo.opName, "When layout is %s, pse is not supported!", layoutStr.c_str()),
        return false);

    return true;
}

bool CommonChecker::CheckTransposeLayoutCrossover(const FiaTilingInfo &fiaInfo)
{
    std::string layoutStr(fiaInfo.opParamInfo.layOut);
    if (layoutStr != "BSH_BNSD" && layoutStr != "BSND_BNSD" && layoutStr != "BNSD_BSND") {
        return true;
    }
    bool isGqa = enableNonQuant_ && !(fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D512) &&
                 !(fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D128) && !(fiaInfo.mlaMode == MlaMode::ROPE_COMBINE_D128);
    if (isGqa) { // GQA
        OP_CHECK_IF(
            fiaInfo.isQKVDDifferent,
            OPS_REPORT_VECTOR_INNER_ERR(
                fiaInfo.opName,
                "In GQA scenario, not support layout %s when query and key headdim is not equal to value headdim.",
                layoutStr.c_str()),
            return false);
    }
    if (layoutStr == "BSH_BNSD" || layoutStr == "BSND_BNSD") {
        if (isGqa) { // GQA
            OP_CHECK_IF((fiaInfo.qkHeadDim != 64 && fiaInfo.qkHeadDim != 128),
                        OPS_REPORT_VECTOR_INNER_ERR(
                            fiaInfo.opName,
                            "In GQA scenario, when layout is %s, d size of query must be 64 or 128, but got d = %d.",
                            layoutStr.c_str(), fiaInfo.qkHeadDim),
                        return false);
        }

        OP_CHECK_IF(fiaInfo.kvStorageMode == KvStorageMode::TENSOR_LIST,
                    OPS_REPORT_VECTOR_INNER_ERR(fiaInfo.opName, "When layout is %s, tensorlist is not supported!",
                                                layoutStr.c_str()),
                    return false);

        OP_CHECK_IF(
            fiaInfo.pseShiftFlag,
            OPS_REPORT_VECTOR_INNER_ERR(fiaInfo.opName, "When layout is %s, pse is not supported!", layoutStr.c_str()),
            return false);
    } else if (layoutStr == "BNSD_BSND") {
        if (isGqa) { // GQA
            OP_CHECK_IF((fiaInfo.outputType == ge::DT_INT8 && fiaInfo.qkHeadDim % 32 != 0),
                        OPS_REPORT_VECTOR_INNER_ERR(fiaInfo.opName,
                                                    "In GQA scenario, when layout is %s and output dtype is int8, d "
                                                    "size should be a multiple of %d, but got d = %d.",
                                                    layoutStr.c_str(), 32, fiaInfo.qkHeadDim),
                        return false);
            OP_CHECK_IF((fiaInfo.qkHeadDim % 16 != 0),
                        OPS_REPORT_VECTOR_INNER_ERR(
                            fiaInfo.opName,
                            "In GQA scenario, when layout is %s, d size should be a multiple of %d, but got d = %d.",
                            layoutStr.c_str(), 16, fiaInfo.qkHeadDim),
                        return false);
        }
    }
    return true;
}

ge::graphStatus CommonChecker::CheckSinglePara(const FiaTilingInfo &fiaInfo)
{
    if (enableNonQuant_) {
        if (CheckNonQuantDataType(fiaInfo) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    if (CheckInputFormat(fiaInfo) != ge::GRAPH_SUCCESS || CheckAttr(fiaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckParaExistence(const FiaTilingInfo &fiaInfo)
{
    if (CheckParaExistenceImpl(fiaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckCrossFeature(const FiaTilingInfo &fiaInfo)
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckMultiParaConsistency(const FiaTilingInfo &fiaInfo)
{
    if (enableNonQuant_) {
        if (!CheckTNDLayoutCrossover(fiaInfo) || !CheckNTDLayoutCrossover(fiaInfo) ||
            !CheckTransposeLayoutCrossover(fiaInfo)) {
            return ge::GRAPH_FAILED;
        }
        if (CheckMultiDtype(fiaInfo) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    if (fiaInfo.kvStorageMode == KvStorageMode::PAGE_ATTENTION) {
        if (CheckPAKeyValue(fiaInfo) != ge::GRAPH_SUCCESS) { // PA场景
            return ge::GRAPH_FAILED;
        }
    } else if (fiaInfo.kvStorageMode == KvStorageMode::TENSOR_LIST) {
        OP_CHECK_IF((CheckTensorList(fiaInfo)) != ge::GRAPH_SUCCESS,
                    OPS_REPORT_VECTOR_INNER_ERR(fiaInfo.opName, "Check Tensorlist failed!"), return ge::GRAPH_FAILED);
    }
    ge::DataType queryDataType = fiaInfo.opParamInfo.query.desc->GetDataType();
    ge::DataType attenOutDataType = fiaInfo.opParamInfo.attenOut.desc->GetDataType();
    if (!fiaInfo.isOutQuantEnable && !enableFullQuant_) {
        OP_CHECK_IF(
            (queryDataType != attenOutDataType),
            OP_LOGE(fiaInfo.opName, "The data type of query is %s,the data type of attentionOut is %s",
                    DataTypeToSerialString(queryDataType).c_str(), DataTypeToSerialString(attenOutDataType).c_str()),
            return ge::GRAPH_FAILED);
    }
    if (CheckAxis(fiaInfo) != ge::GRAPH_SUCCESS || CheckQueryOutConsistency(fiaInfo) != ge::GRAPH_SUCCESS ||
        CheckKeyValueConsistency(fiaInfo) != ge::GRAPH_SUCCESS || CheckQueryShape(fiaInfo) != ge::GRAPH_SUCCESS ||
        CheckKeyShape(fiaInfo) != ge::GRAPH_SUCCESS || CheckQueryKeyConsistency(fiaInfo) != ge::GRAPH_SUCCESS ||
        CheckMultiAttr(fiaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
