/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALLTO_ALL_MATMUL_HOST_UT_PARAM_H
#define ALLTO_ALL_MATMUL_HOST_UT_PARAM_H

#include <sstream>
#include "op_host_csv_case_loader.h"

namespace AlltoAllMatmulUT {
struct AlltoAllMatmulTilingUtParam {
    std::string case_name;
    gert::TilingContextPara::TensorDescription x1 = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription x2 = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription bias = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription x1Scale = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription x2Scale = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription commScale = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription x1Offset = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription x2Offset = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription y = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription all2allOut = TD_DEFAULT;
    std::string group;
    int64_t worldSize;
    int64_t all2allAxes;
    int64_t yDtypeAttr;
    int64_t x1QuantMode;
    int64_t x2QuantMode;
    int64_t commQuantMode;
    int64_t x1QuantDtype;
    int64_t commQuantDtype;
    bool transposeX1;
    bool transposeX2;
    int64_t groupSize;
    std::string commMode;
    bool alltoalloutFlag;
    std::string soc;
    ge::graphStatus expectResult;
    uint64_t expectTilingKey;
    std::string expectTilingData;
    std::vector<size_t> expectWorkspaces;
    uint64_t mc2TilingDataReservedLen;

    explicit AlltoAllMatmulTilingUtParam(const csv_map &csvMap)
    {
        this->case_name = ReadMap(csvMap, "caseName");
        GetTensorGE(csvMap, "x1Shape", "x1Dtype", "x1Format", this->x1);
        GetTensorGE(csvMap, "x2Shape", "x2Dtype", "x2Format", this->x2);
        GetTensorGE(csvMap, "biasShape", "biasDtype", "biasFormat", this->bias);
        GetTensorGE(csvMap, "x1ScaleShape", "x1ScaleDtype", "x1ScaleFormat", this->x1Scale);
        GetTensorGE(csvMap, "x2ScaleShape", "x2ScaleDtype", "x2ScaleFormat", this->x2Scale);
        GetTensorGE(csvMap, "commScaleShape", "commScaleDtype", "commScaleFormat", this->commScale);
        GetTensorGE(csvMap, "x1OffsetShape", "x1OffsetDtype", "x1OffsetFormat", this->x1Offset);
        GetTensorGE(csvMap, "x2OffsetShape", "x2OffsetDtype", "x2OffsetFormat", this->x2Offset);
        GetTensorGE(csvMap, "yShape", "yDtype", "yFormat", this->y);
        GetTensorGE(csvMap, "all2allOutShape", "outputDtype", "outputFormat", this->all2allOut);
        this->group = ReadMap(csvMap, "group");
        this->worldSize = stoll(ReadMap(csvMap, "worldSize", "0"));
        this->all2allAxes = stoll(ReadMap(csvMap, "all2allAxes", "0"));
        this->yDtypeAttr = stoll(ReadMap(csvMap, "yDtypeAttr", "0"));
        this->x1QuantMode = stoll(ReadMap(csvMap, "x1QuantMode", "0"));
        this->x2QuantMode = stoll(ReadMap(csvMap, "x2QuantMode", "0"));
        this->commQuantMode = stoll(ReadMap(csvMap, "commQuantMode", "0"));
        this->x1QuantDtype = stoll(ReadMap(csvMap, "x1QuantDtype", "0"));
        this->commQuantDtype = stoll(ReadMap(csvMap, "commQuantDtype", "0"));
        this->transposeX1 = StrToBoolIgnoreCase(ReadMap(csvMap, "transposeX1", "false"));
        this->transposeX2 = StrToBoolIgnoreCase(ReadMap(csvMap, "transposeX2", "false"));
        this->groupSize = stoll(ReadMap(csvMap, "groupSize", "0"));
        this->commMode = ReadMap(csvMap, "commMode", "");
        this->alltoalloutFlag = StrToBoolIgnoreCase(ReadMap(csvMap, "alltoalloutFlag", "false"));
        this->soc = ReadMap(csvMap, "socVersion");
        this->expectResult = Str2StatusGE(ReadMap(csvMap, "expectStatus"));
        std::string tilingKeyStr = ReadMap(csvMap, "expectTilingKey");
        if (!tilingKeyStr.empty()) {
            this->expectTilingKey = stoull(tilingKeyStr);
        } else {
            this->expectTilingKey = UINT64_MAX; // Skip validation marker
        }
        this->expectTilingData = ReadMap(csvMap, "expectTilingData");
        std::string wsStr = ReadMap(csvMap, "expectWorkspaces");
        if (!wsStr.empty()) {
            this->expectWorkspaces.push_back(static_cast<size_t>(stoull(wsStr)));
        }
        this->mc2TilingDataReservedLen = stoull(ReadMap(csvMap, "mc2TilingDataReservedLen", "0"));
    }
};

inline std::ostream &operator<<(std::ostream &os, const AlltoAllMatmulTilingUtParam &param)
{
    return os << param.case_name;
}

} // namespace AlltoAllMatmulUT

#endif // ALLTO_ALL_MATMUL_HOST_UT_PARAM_H
