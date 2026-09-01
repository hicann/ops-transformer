/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QUANT_GMM_ALLTOALLV_HOST_UT_PARAM_H
#define QUANT_GMM_ALLTOALLV_HOST_UT_PARAM_H

#include <sstream>
#include <algorithm>
#include "op_host_csv_case_loader.h"

namespace QuantGmmAlltoAllvUT {

inline int64_t SafeStoll(const std::string &str, int64_t defaultValue = 0)
{
    try {
        return std::stoll(str);
    } catch (const std::exception &e) {
        std::cerr << "[WARN] SafeStoll failed for \"" << str << "\", using default: " << defaultValue << std::endl;
        return defaultValue;
    }
}

inline uint64_t SafeStoull(const std::string &str, uint64_t defaultValue = 0)
{
    try {
        return std::stoull(str);
    } catch (const std::exception &e) {
        std::cerr << "[WARN] SafeStoull failed for \"" << str << "\", using default: " << defaultValue << std::endl;
        return defaultValue;
    }
}

struct QuantGmmAlltoAllvTilingUtParam {
    std::string case_name;

    std::vector<int64_t> gmmXShape;
    ge::DataType gmmXDataType = ge::DT_UNDEFINED;
    ge::Format gmmXFormat = ge::FORMAT_ND;

    std::vector<int64_t> gmmWeightShape;
    ge::DataType gmmWeightDataType = ge::DT_UNDEFINED;
    ge::Format gmmWeightFormat = ge::FORMAT_ND;

    std::vector<int64_t> gmmXScaleShape;
    ge::DataType gmmXScaleDataType = ge::DT_UNDEFINED;
    ge::Format gmmXScaleFormat = ge::FORMAT_ND;

    std::vector<int64_t> gmmWeightScaleShape;
    ge::DataType gmmWeightScaleDataType = ge::DT_UNDEFINED;
    ge::Format gmmWeightScaleFormat = ge::FORMAT_ND;

    std::vector<int64_t> mmXShape;
    ge::DataType mmXDataType = ge::DT_UNDEFINED;
    ge::Format mmXFormat = ge::FORMAT_ND;

    std::vector<int64_t> mmWeightShape;
    ge::DataType mmWeightDataType = ge::DT_UNDEFINED;
    ge::Format mmWeightFormat = ge::FORMAT_ND;

    std::vector<int64_t> mmXScaleShape;
    ge::DataType mmXScaleDataType = ge::DT_UNDEFINED;
    ge::Format mmXScaleFormat = ge::FORMAT_ND;

    std::vector<int64_t> mmWeightScaleShape;
    ge::DataType mmWeightScaleDataType = ge::DT_UNDEFINED;
    ge::Format mmWeightScaleFormat = ge::FORMAT_ND;

    std::vector<int64_t> sendCounts;
    std::vector<int64_t> recvCounts;

    std::vector<int64_t> gmmYShape;
    ge::DataType gmmYDataType = ge::DT_UNDEFINED;
    ge::Format gmmYFormat = ge::FORMAT_ND;

    std::vector<int64_t> mmYShape;
    ge::DataType mmYDataType = ge::DT_UNDEFINED;
    ge::Format mmYFormat = ge::FORMAT_ND;

    int64_t gmmXQuantMode = 0;
    int64_t gmmWeightQuantMode = 0;
    int64_t mmXQuantMode = 0;
    int64_t mmWeightQuantMode = 0;

    int64_t groupSize = 0;
    std::string commMode = "default";

    bool transGmmWeight = false;
    bool transMmWeight = false;

    int64_t worldSize = 0;
    int64_t epWorldSize = 0;
    int64_t graphType = 0;

    ge::graphStatus expectResult = ge::GRAPH_FAILED;
    uint64_t expectTilingKey = 0;

    explicit QuantGmmAlltoAllvTilingUtParam(const csv_map &csvMap)
    {
        this->case_name = ReadMap(csvMap, "caseName");

        this->gmmXShape = GetShapeArr(ReadMap(csvMap, "gmmXShape"));
        this->gmmXDataType = Str2DTypeGE(ReadMap(csvMap, "gmmXDtype"));
        this->gmmXFormat = ReadMap(GE_FORMAT, ReadMap(csvMap, "gmmXFormat"), ge::FORMAT_ND);

        this->gmmWeightShape = GetShapeArr(ReadMap(csvMap, "gmmWeightShape"));
        this->gmmWeightDataType = Str2DTypeGE(ReadMap(csvMap, "gmmWeightDtype"));
        this->gmmWeightFormat = ReadMap(GE_FORMAT, ReadMap(csvMap, "gmmWeightFormat"), ge::FORMAT_ND);

        this->gmmXScaleShape = GetShapeArr(ReadMap(csvMap, "gmmXScaleShape"));
        this->gmmXScaleDataType = Str2DTypeGE(ReadMap(csvMap, "gmmXScaleDtype"));
        this->gmmXScaleFormat = ReadMap(GE_FORMAT, ReadMap(csvMap, "gmmXScaleFormat"), ge::FORMAT_ND);

        this->gmmWeightScaleShape = GetShapeArr(ReadMap(csvMap, "gmmWeightScaleShape"));
        this->gmmWeightScaleDataType = Str2DTypeGE(ReadMap(csvMap, "gmmWeightScaleDtype"));
        this->gmmWeightScaleFormat = ReadMap(GE_FORMAT, ReadMap(csvMap, "gmmWeightScaleFormat"), ge::FORMAT_ND);

        this->mmXShape = GetShapeArr(ReadMap(csvMap, "mmXShape"));
        this->mmXDataType = Str2DTypeGE(ReadMap(csvMap, "mmXDtype"));
        this->mmXFormat = ReadMap(GE_FORMAT, ReadMap(csvMap, "mmXFormat"), ge::FORMAT_ND);

        this->mmWeightShape = GetShapeArr(ReadMap(csvMap, "mmWeightShape"));
        this->mmWeightDataType = Str2DTypeGE(ReadMap(csvMap, "mmWeightDtype"));
        this->mmWeightFormat = ReadMap(GE_FORMAT, ReadMap(csvMap, "mmWeightFormat"), ge::FORMAT_ND);

        this->mmXScaleShape = GetShapeArr(ReadMap(csvMap, "mmXScaleShape"));
        this->mmXScaleDataType = Str2DTypeGE(ReadMap(csvMap, "mmXScaleDtype"));
        this->mmXScaleFormat = ReadMap(GE_FORMAT, ReadMap(csvMap, "mmXScaleFormat"), ge::FORMAT_ND);

        this->mmWeightScaleShape = GetShapeArr(ReadMap(csvMap, "mmWeightScaleShape"));
        this->mmWeightScaleDataType = Str2DTypeGE(ReadMap(csvMap, "mmWeightScaleDtype"));
        this->mmWeightScaleFormat = ReadMap(GE_FORMAT, ReadMap(csvMap, "mmWeightScaleFormat"), ge::FORMAT_ND);

        this->sendCounts = GetShapeArr(ReadMap(csvMap, "sendCounts"));
        this->recvCounts = GetShapeArr(ReadMap(csvMap, "recvCounts"));

        this->gmmYShape = GetShapeArr(ReadMap(csvMap, "gmmYShape"));
        this->gmmYDataType = Str2DTypeGE(ReadMap(csvMap, "gmmYDtype"));
        this->gmmYFormat = ReadMap(GE_FORMAT, ReadMap(csvMap, "gmmYFormat"), ge::FORMAT_ND);

        this->mmYShape = GetShapeArr(ReadMap(csvMap, "mmYShape"));
        this->mmYDataType = Str2DTypeGE(ReadMap(csvMap, "mmYDtype"));
        this->mmYFormat = ReadMap(GE_FORMAT, ReadMap(csvMap, "mmYFormat"), ge::FORMAT_ND);

        this->gmmXQuantMode = SafeStoll(ReadMap(csvMap, "gmmXQuantMode", "0"));
        this->gmmWeightQuantMode = SafeStoll(ReadMap(csvMap, "gmmWeightQuantMode", "0"));
        this->mmXQuantMode = SafeStoll(ReadMap(csvMap, "mmXQuantMode", "0"));
        this->mmWeightQuantMode = SafeStoll(ReadMap(csvMap, "mmWeightQuantMode", "0"));

        this->groupSize = SafeStoll(ReadMap(csvMap, "groupSize", "0"));
        this->commMode = ReadMap(csvMap, "commMode", "ccu");

        this->transGmmWeight = StrToBoolIgnoreCase(ReadMap(csvMap, "transGmmWeight", "false"));
        this->transMmWeight = StrToBoolIgnoreCase(ReadMap(csvMap, "transMmWeight", "false"));

        this->worldSize = SafeStoll(ReadMap(csvMap, "worldSize", "0"));
        this->epWorldSize = SafeStoll(ReadMap(csvMap, "epWorldSize", "0"));
        this->graphType = SafeStoll(ReadMap(csvMap, "graphType", "0"));

        this->expectResult = Str2StatusGE(ReadMap(csvMap, "expectStatus"));
        this->expectTilingKey = SafeStoull(ReadMap(csvMap, "expectTilingKey", "0"));
    }
};

inline std::ostream &operator<<(std::ostream &os, const QuantGmmAlltoAllvTilingUtParam &param)
{
    return os << param.case_name;
}

} // namespace QuantGmmAlltoAllvUT

#endif // QUANT_GMM_ALLTOALLV_HOST_UT_PARAM_H
