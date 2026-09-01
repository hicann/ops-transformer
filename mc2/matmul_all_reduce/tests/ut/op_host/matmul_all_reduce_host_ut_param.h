/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATMUL_ALL_REDUCE_HOST_UT_PARAM_H
#define MATMUL_ALL_REDUCE_HOST_UT_PARAM_H

#include <sstream>
#include "op_host_csv_case_loader.h"

namespace MatmulAllReduceUT {

struct MatmulAllReduceHostUtParamBase {
    std::string case_name;
    std::vector<uint32_t> inputInstance;
    std::vector<uint32_t> outputInstance;
    std::string group;
    std::string reduce_op;
    bool is_trans_a;
    bool is_trans_b;
    int64_t comm_turn;
    int64_t antiquant_group_size;
    int64_t group_size;
    ge::DataType y_dtype;
    int64_t comm_quant_mode;
    std::string comm_mode;
    ge::graphStatus expectResult;

    MatmulAllReduceHostUtParamBase(const csv_map &csvMap)
    {
        this->case_name = ReadMap(csvMap, "case_name");
        this->group = ReadMap(csvMap, "group");
        this->reduce_op = ReadMap(csvMap, "reduce_op");
        this->is_trans_a = stoi(ReadMap(csvMap, "is_trans_a"));
        this->is_trans_b = stoi(ReadMap(csvMap, "is_trans_b"));
        this->comm_turn = stoll(ReadMap(csvMap, "comm_turn"));
        this->antiquant_group_size = stoll(ReadMap(csvMap, "antiquant_group_size"));
        this->group_size = stoll(ReadMap(csvMap, "group_size"));
        this->y_dtype = Str2DTypeGE(ReadMap(csvMap, "y_dtype"));
        this->comm_quant_mode = stoll(ReadMap(csvMap, "comm_quant_mode"));
        this->comm_mode = ReadMap(csvMap, "comm_mode", "ai_cpu");
        this->expectResult = Str2StatusGE(ReadMap(csvMap, "expectResult"));
    }
};

inline std::ostream &operator<<(std::ostream &os, const MatmulAllReduceHostUtParamBase &param)
{
    return os << param.case_name;
}

struct MatmulAllReduceTilingUtParam : public MatmulAllReduceHostUtParamBase {
    gert::TilingContextPara::TensorDescription x1 = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription x2 = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription bias = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription x3 = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription antiquant_scale = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription antiquant_offset = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription dequant_scale = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription pertoken_scale = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription comm_quant_scale_1 = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription comm_quant_scale_2 = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription y = TD_DEFAULT;
    std::string soc;
    uint64_t coreNum;
    uint64_t ubsize;
    uint64_t ranksize;
    uint64_t expectTilingKey;
    std::string expectTilingDataHash;

    MatmulAllReduceTilingUtParam(const csv_map &csvMap)
        : MatmulAllReduceHostUtParamBase(csvMap)
    {
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "x1_shape", "x1_dtype", "x1_format", this->x1));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "x2_shape", "x2_dtype", "x2_format", this->x2));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "bias_shape", "bias_dtype", "bias_format", this->bias));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "x3_shape", "x3_dtype", "x3_format", this->x3));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "antiquant_scale_shape", "antiquant_scale_dtype",
                                                     "antiquant_scale_format", this->antiquant_scale));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "antiquant_offset_shape", "antiquant_offset_dtype",
                                                     "antiquant_offset_format", this->antiquant_offset));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "dequant_scale_shape", "dequant_scale_dtype",
                                                     "dequant_scale_format", this->dequant_scale));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "pertoken_scale_shape", "pertoken_scale_dtype",
                                                     "pertoken_scale_format", this->pertoken_scale));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "comm_quant_scale_1_shape", "comm_quant_scale_1_dtype",
                                                     "comm_quant_scale_1_format", this->comm_quant_scale_1));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "comm_quant_scale_2_shape", "comm_quant_scale_2_dtype",
                                                     "comm_quant_scale_2_format", this->comm_quant_scale_2));

        this->outputInstance.emplace_back(
            GetTensorGE(csvMap, "output_y_shape", "output_y_dtype", "output_y_format", this->y));

        this->soc = ReadMap(csvMap, "soc");
        this->coreNum = stoull(ReadMap(csvMap, "core_num"));
        this->ubsize = stoull(ReadMap(csvMap, "ubsize"));
        this->ranksize = stoull(ReadMap(csvMap, "ranksize"));

        if (this->expectResult == ge::GRAPH_SUCCESS) {
            auto tilingKeyStr = ReadMap(csvMap, "expectTilingKey");
            this->expectTilingKey = tilingKeyStr.empty() ? UINT64_MAX : stoull(tilingKeyStr);
            this->expectTilingDataHash = ReadMap(csvMap, "expectTilingDataHash");
        }
    }
};

struct MatmulAllReduceInferShapeUtParam : public MatmulAllReduceHostUtParamBase {
    gert::InfershapeContextPara::TensorDescription x1 = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription x2 = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription bias = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription x3 = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription antiquant_scale = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription antiquant_offset = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription dequant_scale = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription pertoken_scale = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription comm_quant_scale_1 = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription comm_quant_scale_2 = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription y = ID_DEFAULT;
    uint64_t ranksize;
    std::vector<std::vector<int64_t>> expectOutputShape;

    MatmulAllReduceInferShapeUtParam(const csv_map &csvMap)
        : MatmulAllReduceHostUtParamBase(csvMap)
    {
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "x1_shape", "x1_dtype", "x1_format", this->x1));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "x2_shape", "x2_dtype", "x2_format", this->x2));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "bias_shape", "bias_dtype", "bias_format", this->bias));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "x3_shape", "x3_dtype", "x3_format", this->x3));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "antiquant_scale_shape", "antiquant_scale_dtype",
                                                     "antiquant_scale_format", this->antiquant_scale));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "antiquant_offset_shape", "antiquant_offset_dtype",
                                                     "antiquant_offset_format", this->antiquant_offset));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "dequant_scale_shape", "dequant_scale_dtype",
                                                     "dequant_scale_format", this->dequant_scale));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "pertoken_scale_shape", "pertoken_scale_dtype",
                                                     "pertoken_scale_format", this->pertoken_scale));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "comm_quant_scale_1_shape", "comm_quant_scale_1_dtype",
                                                     "comm_quant_scale_1_format", this->comm_quant_scale_1));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "comm_quant_scale_2_shape", "comm_quant_scale_2_dtype",
                                                     "comm_quant_scale_2_format", this->comm_quant_scale_2));

        this->outputInstance.emplace_back(GetTensorGE(csvMap, "out_y_shape", "out_y_dtype", "out_y_format", this->y));

        this->ranksize = stoull(ReadMap(csvMap, "ranksize"));
        if (this->expectResult == ge::GRAPH_SUCCESS) {
            this->expectOutputShape = {GetShapeArr(ReadMap(csvMap, "out_y_shape"))};
        }
    }
};

struct MatmulAllReduceInferDataTypeUtParam : public MatmulAllReduceHostUtParamBase {
    ge::DataType x1 = ge::DT_UNDEFINED;
    ge::DataType x2 = ge::DT_UNDEFINED;
    ge::DataType bias = ge::DT_UNDEFINED;
    ge::DataType x3 = ge::DT_UNDEFINED;
    ge::DataType antiquant_scale = ge::DT_UNDEFINED;
    ge::DataType antiquant_offset = ge::DT_UNDEFINED;
    ge::DataType dequant_scale = ge::DT_UNDEFINED;
    ge::DataType pertoken_scale = ge::DT_UNDEFINED;
    ge::DataType comm_quant_scale_1 = ge::DT_UNDEFINED;
    ge::DataType comm_quant_scale_2 = ge::DT_UNDEFINED;
    ge::DataType y = ge::DT_UNDEFINED;

    MatmulAllReduceInferDataTypeUtParam(const csv_map &csvMap)
        : MatmulAllReduceHostUtParamBase(csvMap)
    {
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "x1_dtype", this->x1));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "x2_dtype", this->x2));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "bias_dtype", this->bias));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "x3_dtype", this->x3));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "antiquant_scale_dtype", this->antiquant_scale));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "antiquant_offset_dtype", this->antiquant_offset));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "dequant_scale_dtype", this->dequant_scale));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "pertoken_scale_dtype", this->pertoken_scale));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "comm_quant_scale_1_dtype", this->comm_quant_scale_1));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "comm_quant_scale_2_dtype", this->comm_quant_scale_2));

        this->outputInstance.emplace_back(1);

        if (this->expectResult == ge::GRAPH_SUCCESS) {
            this->y = Str2DTypeGE(ReadMap(csvMap, "expect_y_dtype"));
        }
    }
};

} // namespace MatmulAllReduceUT

#endif // MATMUL_ALL_REDUCE_HOST_UT_PARAM_H
