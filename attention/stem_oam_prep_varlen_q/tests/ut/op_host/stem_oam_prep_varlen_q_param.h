/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef STEM_OAM_PREP_VARLEN_Q_PARAM_H
#define STEM_OAM_PREP_VARLEN_Q_PARAM_H

#include <string>
#include <vector>
#include <sstream>
#include "op_host_csv_case_loader.h"
#include "tiling_context_faker.h"
#include "infer_shape_context_faker.h"
#include "../../op_host/arch35/stem_oam_prep_varlen_q_tiling.h"

namespace StemOamPrepVarlenQUT {

using optiling::StemPrepQCompileInfo;

struct StemPrepQHostUtParamBase : public HostUtParamBase {
    int64_t stemBlockSize = 128;
    int64_t stem_stride = 16;

    StemPrepQHostUtParamBase(const csv_map &csvMap)
        : HostUtParamBase(csvMap)
    {
        stemBlockSize = std::stoll(ReadMap(csvMap, "stemBlockSize", "128"));
        stem_stride = std::stoll(ReadMap(csvMap, "stemStride", "16"));
    }
};

struct StemPrepQInferShapeUtParam : public StemPrepQHostUtParamBase {
    gert::InfershapeContextPara::TensorDescription q = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription qSeqLens = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription cuSeqLensQ = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription qScale = ID_DEFAULT;

    gert::InfershapeContextPara::TensorDescription qFlat = ID_DEFAULT;

    std::vector<std::vector<int64_t>> expectOutputShape;

    StemPrepQInferShapeUtParam(const csv_map &csvMap)
        : StemPrepQHostUtParamBase(csvMap)
    {
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "q_shape", "q_dtype", "q_format", this->q));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "qSeqLens_shape", "qSeqLens_dtype", "qSeqLens_format", this->qSeqLens));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "cuSeqLensQ_shape", "cuSeqLensQ_dtype", "cuSeqLensQ_format", this->cuSeqLensQ));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "qScale_shape", "qScale_dtype", "qScale_format", this->qScale));

        this->outputInstance.emplace_back(
            GetTensorGE(csvMap, "qFlat_shape", "qFlat_dtype", "qFlat_format", this->qFlat));

        if (this->expectResult == ge::GRAPH_SUCCESS) {
            std::string shapeStr = ReadMap(csvMap, "expect_qFlat_shape", "");
            if (!shapeStr.empty()) {
                std::vector<int64_t> shape;
                std::stringstream ss(shapeStr);
                int64_t dim;
                while (ss >> dim) {
                    shape.push_back(dim);
                }
                expectOutputShape.push_back(shape);
            }
        }
    }
};

struct StemPrepQInferDTypeUtParam : public StemPrepQHostUtParamBase {
    ge::DataType q_dtype = ge::DT_UNDEFINED;

    ge::DataType expect_qFlat_dtype = ge::DT_UNDEFINED;

    StemPrepQInferDTypeUtParam(const csv_map &csvMap)
        : StemPrepQHostUtParamBase(csvMap)
    {
        GetDataTypeGE(csvMap, "q_dtype", this->q_dtype);

        if (this->expectResult == ge::GRAPH_SUCCESS) {
            GetDataTypeGE(csvMap, "expect_qFlat_dtype", this->expect_qFlat_dtype);
        }
    }
};

struct StemPrepQTilingUtParam : public StemPrepQHostUtParamBase {
    std::vector<uint32_t> inputInstance;
    std::vector<uint32_t> outputInstance;

    gert::TilingContextPara::TensorDescription q = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription qSeqLens = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription cuSeqLensQ = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription qScale = TD_DEFAULT;

    gert::TilingContextPara::TensorDescription qFlat = TD_DEFAULT;

    uint64_t expectTilingKey = 0;
    std::string expectTilingDataHash;

    std::vector<int32_t> qSeqLensData;

    StemPrepQCompileInfo compileInfo = {64};

    StemPrepQTilingUtParam(const csv_map &csvMap)
        : StemPrepQHostUtParamBase(csvMap)
    {
        inputInstance.emplace_back(GetTensorGE(csvMap, "q_shape", "q_dtype", "q_format", this->q));
        inputInstance.emplace_back(
            GetTensorGE(csvMap, "qSeqLens_shape", "qSeqLens_dtype", "qSeqLens_format", this->qSeqLens));
        inputInstance.emplace_back(
            GetTensorGE(csvMap, "cuSeqLensQ_shape", "cuSeqLensQ_dtype", "cuSeqLensQ_format", this->cuSeqLensQ));
        inputInstance.emplace_back(GetTensorGE(csvMap, "qScale_shape", "qScale_dtype", "qScale_format", this->qScale));

        outputInstance.emplace_back(GetTensorGE(csvMap, "qFlat_shape", "qFlat_dtype", "qFlat_format", this->qFlat));

        std::string dataStr = ReadMap(csvMap, "qSeqLens_data", "");
        if (!dataStr.empty()) {
            std::istringstream iss(dataStr);
            std::string token;
            while (std::getline(iss, token, '|')) {
                qSeqLensData.push_back(std::stoi(token));
            }
            if (!qSeqLensData.empty()) {
                this->qSeqLens = gert::TilingContextPara::TensorDescription(
                    this->qSeqLens.shape_, this->qSeqLens.dtype_, this->qSeqLens.format_, true, qSeqLensData.data());
            }
        }

        if (this->expectResult == ge::GRAPH_SUCCESS) {
            expectTilingKey = std::stoull(ReadMap(csvMap, "expectTilingKey", "0"));
            expectTilingDataHash = ReadMap(csvMap, "expectTilingDataHash", "");
        }
    }
};

} // namespace StemOamPrepVarlenQUT

#endif // STEM_OAM_PREP_VARLEN_Q_PARAM_H
