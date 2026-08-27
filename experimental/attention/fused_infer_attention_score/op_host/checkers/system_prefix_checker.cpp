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
 * \file system_prefix_checker.cpp
 * \brief
 */

#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../fused_infer_attention_score_tiling_constants.h"
#include "system_prefix_checker.h"

namespace optiling {
using std::string;
using namespace ge;
using namespace AscendC;
using namespace arch35FIA;

// singlepara
ge::graphStatus SystemPrefixChecker::CheckSharedPrefixDim(const FiaTilingInfo &fiaInfo)
{
    // 校验keySharedPrefix和valueSharedPrefix的的维度
    auto &keySharedPrefixTensor = fiaInfo.opParamInfo.keySharedPrefix.tensor;
    auto &valueSharedPrefixTensor = fiaInfo.opParamInfo.valueSharedPrefix.tensor;
    if (keySharedPrefixTensor == nullptr || valueSharedPrefixTensor == nullptr) {
        // 若keySharedPrefix或valueSharedPrefix的shape为空，则放弃后续校验
        return ge::GRAPH_SUCCESS;
    }
    // keySharedPrefix、valueSharedPrefix、key、value的维度相同
    uint32_t keySharedPrefixDimNum = keySharedPrefixTensor->GetStorageShape().GetDimNum();
    uint32_t valueSharedPrefixDimNum = valueSharedPrefixTensor->GetStorageShape().GetDimNum();
    uint32_t keyDimNum = fiaInfo.opParamInfo.key.shape->GetStorageShape().GetDimNum();
    OP_CHECK_IF((keySharedPrefixDimNum != keyDimNum || valueSharedPrefixDimNum != keyDimNum),
                OP_LOGE(fiaInfo.opName,
                        "The dimension of keySharedPrefix(%u), valueSharedPrefix(%u) and "
                        "key/value(%u) are different.",
                        keySharedPrefixDimNum, valueSharedPrefixDimNum, keyDimNum),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SystemPrefixChecker::CheckSharedPrefixDataType(const FiaTilingInfo &fiaInfo)
{
    // 校验keySharedPrefix和valueSharedPrefix的数据类型
    auto &keySharedPrefixDesc = fiaInfo.opParamInfo.keySharedPrefix.desc;
    auto &valueSharedPredixDesc = fiaInfo.opParamInfo.valueSharedPrefix.desc;
    if (keySharedPrefixDesc != nullptr && valueSharedPredixDesc != nullptr) {
        // keySharedPrefix和valueSharedPrefix的数据类型与key/value相同
        auto keySharedPrefixType = keySharedPrefixDesc->GetDataType();
        auto valueSharedPredixType = valueSharedPredixDesc->GetDataType();
        OP_CHECK_IF(
            (keySharedPrefixType != fiaInfo.inputKvType),
            OP_LOGE(
                fiaInfo.opName,
                "The datatype of keySharedPrefix(%s) and key/value(%s) are different. The datatype of keySharedPrefix "
                "and key/value must be the same",
                DataTypeToSerialString(keySharedPrefixType).c_str(),
                DataTypeToSerialString(fiaInfo.inputKvType).c_str()),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF((keySharedPrefixType != valueSharedPredixType),
                    OP_LOGE(fiaInfo.opName,
                            "The datatype of keySharedPrefix(%s) and valueSharedPrefix(%s) are different. The datatype "
                            "of keySharedPrefix and valueSharedPrefix must be the same.",
                            DataTypeToSerialString(keySharedPrefixType).c_str(),
                            DataTypeToSerialString(valueSharedPredixType).c_str()),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SystemPrefixChecker::CheckSharedPrefixShape(const FiaTilingInfo &fiaInfo)
{
    // 校验keySharedPrefix和valueSharedPrefix的shape合法性
    auto &keySharedPrefixTensor = fiaInfo.opParamInfo.keySharedPrefix.tensor;
    auto &valueSharedPrefixTensor = fiaInfo.opParamInfo.valueSharedPrefix.tensor;
    if (keySharedPrefixTensor != nullptr && valueSharedPrefixTensor != nullptr) {
        auto &keySharedPrefixShape = fiaInfo.opParamInfo.keySharedPrefix.tensor->GetStorageShape();
        auto &valueSharedPrefixShape = fiaInfo.opParamInfo.valueSharedPrefix.tensor->GetStorageShape();
        auto &keyShape = fiaInfo.opParamInfo.key.shape->GetStorageShape();
        // 第一维batch必须为1
        OP_CHECK_IF((keySharedPrefixShape.GetDim(DIM_NUM_0) != 1),
                    OP_LOGE(fiaInfo.opName, "The first dim(%ld) of keySharedPrefix is not 1.",
                            keySharedPrefixShape.GetDim(DIM_NUM_0)),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF((valueSharedPrefixShape.GetDim(DIM_NUM_0) != 1),
                    OP_LOGE(fiaInfo.opName, "The first dim(%ld) of valueSharedPrefix is not 1.",
                            valueSharedPrefixShape.GetDim(DIM_NUM_0)),
                    return ge::GRAPH_FAILED);
        // layout为BNSD和BSND情况下，N、D轴与key一致
        if (fiaInfo.kvLayout == FiaLayout::BNSD) {
            uint32_t keySharedPrefixN = keySharedPrefixShape.GetDim(DIM_NUM_1);
            uint32_t keySharedPrefixD = keySharedPrefixShape.GetDim(DIM_NUM_3);
            uint32_t valueSharedPrefixN = valueSharedPrefixShape.GetDim(DIM_NUM_1);
            uint32_t valueSharedPrefixD = valueSharedPrefixShape.GetDim(DIM_NUM_3);
            uint32_t keyN = keyShape.GetDim(DIM_NUM_1);
            uint32_t keyD = keyShape.GetDim(DIM_NUM_3);
            uint32_t keySharedPrefixS = keySharedPrefixShape.GetDim(DIM_NUM_2);
            uint32_t valueSharedPrefixS = valueSharedPrefixShape.GetDim(DIM_NUM_2);
            OP_CHECK_IF((keySharedPrefixN != keyN),
                        OP_LOGE(fiaInfo.opName,
                                "The N axis of keySharedPrefix(%ld) and key(%ld) are different. "
                                "The N axis of keySharedPrefix and key must be the same when "
                                "the layout of key is BNSD or BSND.",
                                keySharedPrefixN, keyN),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF((keySharedPrefixN != valueSharedPrefixN),
                        OP_LOGE(fiaInfo.opName,
                                "The N axis of keySharedPrefix(%ld) and valueSharedPrefix(%ld) are different. "
                                "The N axis of keySharedPrefix and valueSharedPrefix must be the same when "
                                "the layout of key is BNSD or BSND.",
                                keySharedPrefixN, valueSharedPrefixN),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF((keySharedPrefixD != keyD),
                        OP_LOGE(fiaInfo.opName,
                                "The D axis of keySharedPrefix(%ld) and key(%ld) are different. "
                                "The D axis of keySharedPrefix and key must be the same when "
                                "the layout of key is BNSD or BSND.",
                                keySharedPrefixD, keyD),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF((keySharedPrefixD != valueSharedPrefixD),
                        OP_LOGE(fiaInfo.opName,
                                "The D axis of keySharedPrefix(%ld) and valueSharedPrefix(%ld) are different. "
                                "The D axis of keySharedPrefix and valueSharedPrefix must be the same when "
                                "the layout of key is BNSD or BSND.",
                                keySharedPrefixD, valueSharedPrefixD),
                        return ge::GRAPH_FAILED);
            // keySharedPrefix和valueSharedPrefix的S应相等
            OP_CHECK_IF((keySharedPrefixS != valueSharedPrefixS),
                        OP_LOGE(fiaInfo.opName,
                                "The S axis of keySharedPrefix(%ld) and valueSharedPrefix(%ld) are different. "
                                "The S axis of keySharedPrefix and valueSharedPrefix must be "
                                "the same.",
                                keySharedPrefixS, valueSharedPrefixS),
                        return ge::GRAPH_FAILED);
        }
        if (fiaInfo.kvLayout == FiaLayout::BSND) {
            uint32_t keySharedPrefixN = keySharedPrefixShape.GetDim(DIM_NUM_2);
            uint32_t keySharedPrefixD = keySharedPrefixShape.GetDim(DIM_NUM_3);
            uint32_t valueSharedPrefixN = valueSharedPrefixShape.GetDim(DIM_NUM_2);
            uint32_t valueSharedPrefixD = valueSharedPrefixShape.GetDim(DIM_NUM_3);
            uint32_t keyN = keyShape.GetDim(DIM_NUM_2);
            uint32_t keyD = keyShape.GetDim(DIM_NUM_3);
            uint32_t keySharedPrefixS = keySharedPrefixShape.GetDim(DIM_NUM_1);
            uint32_t valueSharedPrefixS = valueSharedPrefixShape.GetDim(DIM_NUM_1);
            OP_CHECK_IF((keySharedPrefixN != keyN),
                        OP_LOGE(fiaInfo.opName,
                                "The N axis of keySharedPrefix(%ld) and key(%ld) are different. "
                                "The N axis of keySharedPrefix and key must be the same when "
                                "the layout of key is BNSD or BSND.",
                                keySharedPrefixN, keyN),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF((keySharedPrefixN != valueSharedPrefixN),
                        OP_LOGE(fiaInfo.opName,
                                "The N axis of keySharedPrefix(%ld) and valueSharedPrefix(%ld) are different. "
                                "The N axis of keySharedPrefix and valueSharedPrefix must be the same when "
                                "the layout of key is BNSD or BSND.",
                                keySharedPrefixN, valueSharedPrefixN),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF((keySharedPrefixD != keyD),
                        OP_LOGE(fiaInfo.opName,
                                "The D axis of keySharedPrefix(%ld) and key(%ld) are different. "
                                "The D axis of keySharedPrefix and key must be the same when "
                                "the layout of key is BNSD or BSND.",
                                keySharedPrefixD, keyD),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF((keySharedPrefixD != valueSharedPrefixD),
                        OP_LOGE(fiaInfo.opName,
                                "The D axis of keySharedPrefix(%ld) and valueSharedPrefix(%ld) are different. "
                                "The D axis of keySharedPrefix and valueSharedPrefix must be the same when "
                                "the layout of key is BNSD or BSND.",
                                keySharedPrefixD, valueSharedPrefixD),
                        return ge::GRAPH_FAILED);
            // keySharedPrefix和valueSharedPrefix的S应相等
            OP_CHECK_IF((keySharedPrefixS != valueSharedPrefixS),
                        OP_LOGE(fiaInfo.opName,
                                "The S axis of keySharedPrefix(%ld) and valueSharedPrefix(%ld) are different. "
                                "The S axis of keySharedPrefix and valueSharedPrefix must be "
                                "the same.",
                                keySharedPrefixS, valueSharedPrefixS),
                        return ge::GRAPH_FAILED);
        }
        // layout为BSH情况下H与key一致
        if (fiaInfo.kvLayout == FiaLayout::BSH) {
            uint32_t keySharedPrefixH = keySharedPrefixShape.GetDim(DIM_NUM_2);
            uint32_t valueSharedPrefixH = valueSharedPrefixShape.GetDim(DIM_NUM_2);
            uint32_t keySharedPrefixS = keySharedPrefixShape.GetDim(DIM_NUM_1);
            uint32_t valueSharedPrefixS = valueSharedPrefixShape.GetDim(DIM_NUM_1);
            uint32_t keyH = keyShape.GetDim(DIM_NUM_2);
            OP_CHECK_IF((keySharedPrefixH != keyH),
                        OP_LOGE(fiaInfo.opName,
                                "The H axis of keySharedPrefix(%ld) and key(%ld) are different. "
                                "The H axis of keySharedPrefix and key must be the same when "
                                "the layout of key is BSH.",
                                keySharedPrefixH, keyH),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF((keySharedPrefixH != valueSharedPrefixH),
                        OP_LOGE(fiaInfo.opName,
                                "The H axis of keySharedPrefix(%ld) and valueSharedPrefix(%ld) are different. "
                                "The H axis of keySharedPrefix and valueSharedPrefix must be the same when "
                                "the layout of key is BSH.",
                                keySharedPrefixH, valueSharedPrefixH),
                        return ge::GRAPH_FAILED);
            // keySharedPrefix和valueSharedPrefix的S应相等
            OP_CHECK_IF((keySharedPrefixS != valueSharedPrefixS),
                        OP_LOGE(fiaInfo.opName,
                                "The S axis of keySharedPrefix(%ld) and valueSharedPrefix(%ld) are different. "
                                "The S axis of keySharedPrefix and valueSharedPrefix must be "
                                "the same.",
                                keySharedPrefixS, valueSharedPrefixS),
                        return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SystemPrefixChecker::CheckActualSharedPrefixLenData(const FiaTilingInfo &fiaInfo)
{
    // 校验actualSharedPrefixLen的数值，其值不能大于keySharedPrefix和valueSharedPrefix的S
    auto &keySharedPrefixTensor = fiaInfo.opParamInfo.keySharedPrefix.tensor;
    auto &valueSharedPrefixTensor = fiaInfo.opParamInfo.valueSharedPrefix.tensor;
    auto &actualSharedPrefixLenTensor = fiaInfo.opParamInfo.actualSharedPrefixLen.tensor;
    if (keySharedPrefixTensor == nullptr || valueSharedPrefixTensor == nullptr) {
        // 若keySharedPrefix、valueSharedPrefix不存在，则放弃后续校验
        return ge::GRAPH_SUCCESS;
    }
    if (actualSharedPrefixLenTensor == nullptr || actualSharedPrefixLenTensor->GetData<int64_t>() == nullptr) {
        // 若没有传入actualSharedPrefixLen的shape或者data，则放弃后续校验
        return ge::GRAPH_SUCCESS;
    }
    uint32_t keySharedPrefixS = 0;
    uint32_t valueSharedPrefixS = 0;
    if (fiaInfo.kvLayout == FiaLayout::BNSD) {
        keySharedPrefixS = keySharedPrefixTensor->GetStorageShape().GetDim(DIM_NUM_2);
        valueSharedPrefixS = valueSharedPrefixTensor->GetStorageShape().GetDim(DIM_NUM_2);
    }
    if (fiaInfo.kvLayout == FiaLayout::BSND) {
        keySharedPrefixS = keySharedPrefixTensor->GetStorageShape().GetDim(DIM_NUM_1);
        valueSharedPrefixS = valueSharedPrefixTensor->GetStorageShape().GetDim(DIM_NUM_1);
    }
    if (fiaInfo.kvLayout == FiaLayout::BSH) {
        keySharedPrefixS = keySharedPrefixTensor->GetStorageShape().GetDim(DIM_NUM_1);
        valueSharedPrefixS = valueSharedPrefixTensor->GetStorageShape().GetDim(DIM_NUM_1);
    }
    uint32_t actualSharedPrefixLenData = actualSharedPrefixLenTensor->GetData<int64_t>()[0];
    OP_CHECK_IF((actualSharedPrefixLenData > keySharedPrefixS),
                OP_LOGE(fiaInfo.opName,
                        "actualSharedPrefixLen(%u) can not be greater than "
                        "keySharedPrefixS(%u).",
                        actualSharedPrefixLenData, keySharedPrefixS),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((actualSharedPrefixLenData > valueSharedPrefixS),
                OP_LOGE(fiaInfo.opName,
                        "actualSharedPrefixLen(%u) can not be greater than "
                        "valueSharedPrefixS(%u).",
                        actualSharedPrefixLenData, valueSharedPrefixS),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// existence
ge::graphStatus SystemPrefixChecker::CheckSharedPrefixExistence(const FiaTilingInfo &fiaInfo)
{
    // 校验keySharedPrefix和valueSharedPrefix的存在性
    auto &keySharedPrefixTensor = fiaInfo.opParamInfo.keySharedPrefix.tensor;
    auto &valueSharedPrefixTensor = fiaInfo.opParamInfo.valueSharedPrefix.tensor;
    // 两者要么都为空，要么都不为空
    OP_CHECK_IF((keySharedPrefixTensor != nullptr && valueSharedPrefixTensor == nullptr),
                OP_LOGE(fiaInfo.opName, "keySharedPrefix exists, but valueSharedPrefix does not exist. "
                                        "keySharedPrefix and valueSharedPrefix must exist simultaneously."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF((keySharedPrefixTensor == nullptr && valueSharedPrefixTensor != nullptr),
                OP_LOGE(fiaInfo.opName, "valueSharedPrefix exists, but keySharedPrefix does not exist. "
                                        "keySharedPrefix and valueSharedPrefix must exist simultaneously."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// feature
ge::graphStatus SystemPrefixChecker::CheckUnSupportFeature(const FiaTilingInfo &fiaInfo)
{
    if (!fiaInfo.sysPrefixFlag) {
        // 若不使能systemPrefix，则放弃后续校验
        return ge::GRAPH_SUCCESS;
    }
    std::string layoutStr(fiaInfo.opParamInfo.layOut);
    // 校验不支持带prefix的feautre
    bool enableIFAMLA = fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D512;
    bool enablePFARope = fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D128;
    bool enablePFAMLA = fiaInfo.mlaMode == MlaMode::ROPE_COMBINE_D128;
    // 不支持page attention场景
    OP_CHECK_IF((fiaInfo.kvStorageMode == KvStorageMode::PAGE_ATTENTION),
                OP_LOGE(fiaInfo.opName, "prefix is not supported when page attention is enabled."),
                return ge::GRAPH_FAILED);
    // 不支持tensorlist
    OP_CHECK_IF((fiaInfo.kvStorageMode == KvStorageMode::TENSOR_LIST),
                OP_LOGE(fiaInfo.opName, "prefix is not supported when tensorlist is enabled."),
                return ge::GRAPH_FAILED);
    // 不支持左padding场景
    OP_CHECK_IF((fiaInfo.qPaddingSizeFlag || fiaInfo.kvPaddingSizeFlag),
                OP_LOGE(fiaInfo.opName, "prefix is not supported when left padding is enabled."),
                return ge::GRAPH_FAILED);
    // 不支持alibi场景
    OP_CHECK_IF((fiaInfo.enableAlibiPse), OP_LOGE(fiaInfo.opName, "prefix is not supported when pseType is 2 or 3."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(layoutStr == "BSH_BNSD" || layoutStr == "BSND_BNSD" || layoutStr == "TND" || layoutStr == "NTD" ||
                    layoutStr == "NTD_TND" || layoutStr == "TND_NTD",
                OP_LOGE(fiaInfo.opName,
                        "prefix is not supported when the inputLayout is BSH_BNSD/BSND_BNSD/TND/NTD/TND_NTD/NTD_TND"),
                return ge::GRAPH_FAILED);
    // 不支持PFA MLA场景
    OP_CHECK_IF((enablePFAMLA || enablePFARope),
                OP_LOGE(fiaInfo.opName, "prefix is not supported in PFA MLA scenario."), return ge::GRAPH_FAILED);
    // 不支持IFA MLA场景
    OP_CHECK_IF((enableIFAMLA), OP_LOGE(fiaInfo.opName, "prefix is not supported in IFA MLA scenario."),
                return ge::GRAPH_FAILED);
    // 不支持qkv全部为INT8/FP8/HiF8(per-block/per-tensor全量化)的情况
    OP_CHECK_IF((fiaInfo.inputQType == ge::DT_INT8 && fiaInfo.inputKvType == ge::DT_INT8),
                OP_LOGE(fiaInfo.opName, "prefix is not supported when the datatype of query and key/value is INT8"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((fiaInfo.inputQType == ge::DT_FLOAT8_E4M3FN && fiaInfo.inputKvType == ge::DT_FLOAT8_E4M3FN),
                OP_LOGE(fiaInfo.opName, "prefix is not supported when the datatype of query and key/value is FP8"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((fiaInfo.inputQType == ge::DT_HIFLOAT8 && fiaInfo.inputKvType == ge::DT_HIFLOAT8),
                OP_LOGE(fiaInfo.opName, "prefix is not supported when the datatype of query and key/value is HiF8"),
                return ge::GRAPH_FAILED);
    // 后量化仅支持INT8
    if (fiaInfo.isOutQuantEnable) {
        OP_CHECK_IF((fiaInfo.outputType != ge::DT_INT8),
                    OP_LOGE(fiaInfo.opName, "Invalid output type. The datatype of output only support INT8 when "
                                            "prefix is enabled."),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SystemPrefixChecker::CheckFeatureAntiquant(const FiaTilingInfo &fiaInfo)
{
    // 校验prefix叠加伪量化的支持场景
    if (!fiaInfo.sysPrefixFlag) {
        // 若不使能systemPrefix，则放弃后续校验
        return ge::GRAPH_SUCCESS;
    }
    int64_t keyAntiquantMode = 0;
    int64_t valueAntiquantMode = 0;
    if (fiaInfo.opParamInfo.keyAntiquantMode != nullptr) {
        keyAntiquantMode = *fiaInfo.opParamInfo.keyAntiquantMode;
    }
    if (fiaInfo.opParamInfo.valueAntiquantMode != nullptr) {
        valueAntiquantMode = *fiaInfo.opParamInfo.valueAntiquantMode;
    }
    if (fiaInfo.s1Size > 1) {
        // Q_S > 1，per-channel(per-tensor)模式，key/value，仅支持INT8
        if (keyAntiquantMode == PER_CHANNEL_MODE && valueAntiquantMode == PER_CHANNEL_MODE) {
            OP_CHECK_IF(fiaInfo.inputKvType != ge::DT_INT8,
                        OP_LOGE(fiaInfo.opName,
                                "The datatype of key/value(%s) is not INT8. "
                                "The datatype of key/value only support INT8 when prefix is enabled, Q_S > 1 and "
                                "keyAntiquantMode is per-channel(per-tensor) mode and valueAntiquantMode is "
                                "per-channel(per-tensor) mode.",
                                DataTypeToSerialString(fiaInfo.inputKvType).c_str()),
                        return ge::GRAPH_FAILED);
        } else if (keyAntiquantMode == PER_TOKEN_MODE && valueAntiquantMode == PER_TOKEN_MODE) {
            OP_CHECK_IF(fiaInfo.inputKvType != ge::DT_INT8,
                        OP_LOGE(fiaInfo.opName,
                                "The datatype of key/value(%s) is not INT8. "
                                "The datatype of key/value only support INT8 when prefix is enabled, Q_S > 1, "
                                "keyAntiquantMode is per-token mode and valueAntiquantMode is per-token mode.",
                                DataTypeToSerialString(fiaInfo.inputKvType).c_str()),
                        return ge::GRAPH_FAILED);
        } else {
            // 其余量化模式在Q_S > 1时均为非法
            OP_LOGE(fiaInfo.opName,
                    "keyAntiquantMode(%ld) and valueAntiquantMode(%ld) is not supported when "
                    " prefix is enabled and Q_S > 1.",
                    keyAntiquantMode, valueAntiquantMode);
            return ge::GRAPH_FAILED;
        }
    } else if (fiaInfo.s1Size == 1) {
        // Q_S = 1
        auto &keyAntiquantScaleTensor = fiaInfo.opParamInfo.keyAntiquantScale.tensor;
        auto &valueAntiquantScaleTensor = fiaInfo.opParamInfo.valueAntiquantScale.tensor;
        if (keyAntiquantScaleTensor == nullptr || valueAntiquantScaleTensor == nullptr) {
            // 若不存在keyAntiquantScaleTensor和valueAntiquantScaleTensor，则放弃后续校验
            return ge::GRAPH_SUCCESS;
        }
        gert::Shape keyAntiquantScaleTensorShape = keyAntiquantScaleTensor->GetStorageShape();
        uint32_t keyAntiquantScaleTensorDimNum = keyAntiquantScaleTensorShape.GetDimNum();
        if (keyAntiquantMode == PER_CHANNEL_MODE && valueAntiquantMode == PER_CHANNEL_MODE) {
            if (keyAntiquantScaleTensorDimNum == DIM_NUM_1) {
                // per-tensor模式，仅支持INT8
                OP_CHECK_IF(fiaInfo.inputKvType != ge::DT_INT8,
                            OP_LOGE(fiaInfo.opName,
                                    "The datatype of key/value(%s) is not INT8. "
                                    "The datatype of key/value only support INT8 when prefix is enabled, Q_S = 1, "
                                    "keyAntiquantMode is per-tensor mode and valueAntiquantMode is per-tensor mode.",
                                    DataTypeToSerialString(fiaInfo.inputKvType).c_str()),
                            return ge::GRAPH_FAILED);
            } else {
                // per-channel模式，支持INT8或INT4(INT32)
                OP_CHECK_IF(fiaInfo.inputKvType != ge::DT_INT8 && fiaInfo.inputKvType != ge::DT_INT4,
                            OP_LOGE(fiaInfo.opName,
                                    "The datatype of key/value(%s) is not INT8 or INT4(INT32). "
                                    "The datatype of key/value only support INT8 or INT4(INT32) when "
                                    "prefix is enabled, Q_S = 1, "
                                    "keyAntiquantMode is per-channel mode and valueAntiquantMode is per-channel mode.",
                                    DataTypeToSerialString(fiaInfo.inputKvType).c_str()),
                            return ge::GRAPH_FAILED);
            }
        } else if (keyAntiquantMode == PER_TOKEN_MODE && valueAntiquantMode == PER_TOKEN_MODE) {
            // per-token模式，支持INT8或INT4(INT32)
            OP_CHECK_IF(fiaInfo.inputKvType != ge::DT_INT8 && fiaInfo.inputKvType != ge::DT_INT4,
                        OP_LOGE(fiaInfo.opName,
                                "The datatype of key/value(%s) is not INT8 or INT4(INT32). "
                                "The datatype of key/value only support INT8 or INT4(INT32) when "
                                "prefix is enabled, Q_S = 1, "
                                "keyAntiquantMode is per-token mode and valueAntiquantMode is per-token mode.",
                                DataTypeToSerialString(fiaInfo.inputKvType).c_str()),
                        return ge::GRAPH_FAILED);
            return ge::GRAPH_SUCCESS;
        } else if (keyAntiquantMode == PER_TENSOR_HEAD_MODE && valueAntiquantMode == PER_TENSOR_HEAD_MODE) {
            // per-tensor-head模式，仅支持INT8
            OP_CHECK_IF(fiaInfo.inputKvType != ge::DT_INT8,
                        OP_LOGE(fiaInfo.opName,
                                "The datatype of key/value(%s) is not INT8. "
                                "The datatype of key/value only support INT8 when prefix is enabled, Q_S = 1, "
                                "keyAntiquantMode is per-tensor-head mode and valueAntiquantMode is "
                                "per-tensor-head mode.",
                                DataTypeToSerialString(fiaInfo.inputKvType).c_str()),
                        return ge::GRAPH_FAILED);
            return ge::GRAPH_SUCCESS;
        } else if (keyAntiquantMode == PER_TOKEN_HEAD_MODE && valueAntiquantMode == PER_TOKEN_HEAD_MODE) {
            // per-token-head模式，支持INT8或INT4(INT32)
            OP_CHECK_IF(
                fiaInfo.inputKvType != ge::DT_INT8 && fiaInfo.inputKvType != ge::DT_INT4,
                OP_LOGE(fiaInfo.opName,
                        "The datatype of key/value(%s) is not INT8 or INT4(INT32). "
                        "The datatype of key/value only support INT8 or INT4(INT32) when "
                        "prefix is enabled, Q_S = 1, "
                        "keyAntiquantMode is per-token-head mode and valueAntiquantMode is per-token-head mode.",
                        DataTypeToSerialString(fiaInfo.inputKvType).c_str()),
                return ge::GRAPH_FAILED);
            return ge::GRAPH_SUCCESS;
        } else if (keyAntiquantMode == PER_TOKEN_PA_MODE && valueAntiquantMode == PER_TOKEN_PA_MODE) {
            // per-token模式使用page attention管理scale/offset，仅支持INT8
            OP_CHECK_IF(fiaInfo.inputKvType != ge::DT_INT8,
                        OP_LOGE(fiaInfo.opName,
                                "The datatype of key/value(%s) is not INT8. "
                                "The datatype of key/value only support INT8 when prefix is enabled, Q_S = 1, "
                                "keyAntiquantMode is per-tensor-PA mode and valueAntiquantMode is "
                                "per-tensor-PA mode.",
                                DataTypeToSerialString(fiaInfo.inputKvType).c_str()),
                        return ge::GRAPH_FAILED);
            return ge::GRAPH_SUCCESS;
        } else if (keyAntiquantMode == PER_TOKEN_HEAD_PA_MODE && valueAntiquantMode == PER_TOKEN_HEAD_PA_MODE) {
            // per-token叠加per-head模式并使用page attention管理scale/offset，仅支持INT8
            OP_CHECK_IF(fiaInfo.inputKvType != ge::DT_INT8,
                        OP_LOGE(fiaInfo.opName,
                                "The datatype of key/value(%s) is not INT8. "
                                "The datatype of key/value only support INT8 when prefix is enabled, Q_S = 1, "
                                "keyAntiquantMode is per-token-head-PA mode and valueAntiquantMode is "
                                "per-token-head-PA mode.",
                                DataTypeToSerialString(fiaInfo.inputKvType).c_str()),
                        return ge::GRAPH_FAILED);
            return ge::GRAPH_SUCCESS;
        } else if (keyAntiquantMode == PER_CHANNEL_MODE && valueAntiquantMode == PER_TOKEN_MODE) {
            // key支持per-channel叠加value支持per-token，支持INT8或INT4(INT32)
            OP_CHECK_IF(fiaInfo.inputKvType != ge::DT_INT8 && fiaInfo.inputKvType != ge::DT_INT4,
                        OP_LOGE(fiaInfo.opName,
                                "The datatype of key/value(%s) is not INT8 or INT4(INT32). "
                                "The datatype of key/value only support INT8 or INT4(INT32) when "
                                "prefix is enabled, Q_S = 1, "
                                "keyAntiquantMode is per-channel mode and valueAntiquantMode is "
                                "per-token mode.",
                                DataTypeToSerialString(fiaInfo.inputKvType).c_str()),
                        return ge::GRAPH_FAILED);
            return ge::GRAPH_SUCCESS;
        } else {
            // 其余量化模式在Q_S=1均为非法
            OP_LOGE(fiaInfo.opName,
                    "keyAntiquantMode(%ld) and valueAntiquantMode(%ld) is not supported when "
                    " prefix is enabled and Q_S = 1.",
                    keyAntiquantMode, valueAntiquantMode);
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

// multipara
ge::graphStatus SystemPrefixChecker::CheckSinglePara(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckSharedPrefixDim(fiaInfo) || ge::GRAPH_SUCCESS != CheckSharedPrefixDataType(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckSharedPrefixShape(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckActualSharedPrefixLenData(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SystemPrefixChecker::CheckParaExistence(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckSharedPrefixExistence(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SystemPrefixChecker::CheckCrossFeature(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckUnSupportFeature(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }

    if (enableAntiQuant_) {
        if (ge::GRAPH_SUCCESS != CheckFeatureAntiquant(fiaInfo)) {
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SystemPrefixChecker::CheckMultiParaConsistency(const FiaTilingInfo &fiaInfo)
{
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
