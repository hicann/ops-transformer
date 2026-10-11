/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <cstdlib>
#include <iostream>
#include "../quant_gmm_alltoallv_host_ut_param.h"
#include "mc2_tiling_case_executor.h"
#include "../../../../op_host/op_tiling/arch35/tt_quant_grouped_mat_mul_allto_allv_tiling.h"
#include "../../../../op_host/op_tiling/arch35/mx_quant_grouped_mat_mul_allto_allv_tiling.h"

static std::string GetCsvPath(const char* file)
{
    const char* envPath = std::getenv("CSV_CASE_PATH");
    if (envPath != nullptr && strlen(envPath) > 0) {
        return std::string(envPath);
    }
    return ReplaceFileExtension2Csv(file);
}

namespace QuantGmmAlltoAllvUT {

static const std::string OP_NAME = "QuantGroupedMatMulAlltoAllv";

struct GroupedMatMulAlltoAllvCompileInfo {
} compileInfo;

class QuantGmmAlltoAllvTilingTest : public testing::TestWithParam<QuantGmmAlltoAllvTilingUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "QuantGmmAlltoAllvTilingTest SetUp." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "QuantGmmAlltoAllvTilingTest TearDown." << std::endl;
    }
};

TEST_P(QuantGmmAlltoAllvTilingTest, test_grouped_quant_mat_mul_allto_allv_tiling)
{
    auto param = GetParam();
    std::cout << "[TEST_CASE] " << param.case_name << std::endl;

    if (param.gmmXShape.size() < 2 || param.gmmWeightShape.size() < 3 || param.gmmYShape.size() < 2) {
        FAIL() << "Invalid shape dimensions in test case: " << param.case_name;
        return;
    }

    uint64_t coreNum = 36;
    uint64_t ubSize = 256 * 1024;
    uint64_t gmmXScaleDimMx = 3;
    uint64_t gmmWeightScaleDimMx = 4;
    uint64_t mmXScaleDimMx = 3;
    uint64_t mmWeightScaleDimMx = 3;
    size_t tilingDataSize = sizeof(QuantGmmA2avTilingData);

    gert::StorageShape mmXStorageShape;
    if (param.mmXShape.size() >= 2 && param.mmXShape[0] > 0) {
        mmXStorageShape = {{param.mmXShape[0], param.mmXShape[1]}, {param.mmXShape[0], param.mmXShape[1]}};
    } else {
        mmXStorageShape = {};
    }

    gert::StorageShape mmWeightStorageShape;
    if (param.mmWeightShape.size() >= 2 && param.mmWeightShape[0] > 0) {
        mmWeightStorageShape = {{param.mmWeightShape[0], param.mmWeightShape[1]},
                                {param.mmWeightShape[0], param.mmWeightShape[1]}};
    } else {
        mmWeightStorageShape = {};
    }

    gert::StorageShape mmYStorageShape;
    if (param.mmYShape.size() >= 2 && param.mmYShape[0] > 0) {
        mmYStorageShape = {{param.mmYShape[0], param.mmYShape[1]}, {param.mmYShape[0], param.mmYShape[1]}};
    } else {
        mmYStorageShape = {};
    }

    gert::StorageShape gmmXScaleStorageShape;
    if (param.gmmXScaleShape.size() > 0 && param.gmmXScaleShape[0] > 0) {
        if (param.gmmXScaleShape.size() == gmmXScaleDimMx) {
            gmmXScaleStorageShape = {{param.gmmXScaleShape[0], param.gmmXScaleShape[1], param.gmmXScaleShape[2]},
                                     {param.gmmXScaleShape[0], param.gmmXScaleShape[1], param.gmmXScaleShape[2]}};
        } else {
            std::cerr << "[WARN] gmmXScaleShape unexpected dim: " << param.gmmXScaleShape.size()
                      << " in case: " << param.case_name << std::endl;
            gmmXScaleStorageShape = {{param.gmmXScaleShape[0]}, {param.gmmXScaleShape[0]}};
        }
    } else {
        gmmXScaleStorageShape = {};
    }

    gert::StorageShape gmmWeightScaleStorageShape;
    if (param.gmmWeightScaleShape.size() > 0 && param.gmmWeightScaleShape[0] > 0) {
        if (param.gmmWeightScaleShape.size() == gmmWeightScaleDimMx) {
            gmmWeightScaleStorageShape = {{param.gmmWeightScaleShape[0], param.gmmWeightScaleShape[1],
                                           param.gmmWeightScaleShape[2], param.gmmWeightScaleShape[3]},
                                          {param.gmmWeightScaleShape[0], param.gmmWeightScaleShape[1],
                                           param.gmmWeightScaleShape[2], param.gmmWeightScaleShape[3]}};
        } else {
            std::cerr << "[WARN] gmmWeightScaleShape unexpected dim: " << param.gmmWeightScaleShape.size()
                      << " in case: " << param.case_name << std::endl;
            gmmWeightScaleStorageShape = {{param.gmmWeightScaleShape[0]}, {param.gmmWeightScaleShape[0]}};
        }
    } else {
        gmmWeightScaleStorageShape = {};
    }

    gert::StorageShape mmXScaleStorageShape;
    if (param.mmXScaleShape.size() > 0 && param.mmXScaleShape[0] > 0) {
        if (param.mmXScaleShape.size() == mmXScaleDimMx) {
            mmXScaleStorageShape = {{param.mmXScaleShape[0], param.mmXScaleShape[1], param.mmXScaleShape[2]},
                                    {param.mmXScaleShape[0], param.mmXScaleShape[1], param.mmXScaleShape[2]}};
        } else {
            std::cerr << "[WARN] mmXScaleShape unexpected dim: " << param.mmXScaleShape.size()
                      << " in case: " << param.case_name << std::endl;
            mmXScaleStorageShape = {{param.mmXScaleShape[0]}, {param.mmXScaleShape[0]}};
        }
    } else {
        mmXScaleStorageShape = {};
    }

    gert::StorageShape mmWeightScaleStorageShape;
    if (param.mmWeightScaleShape.size() > 0 && param.mmWeightScaleShape[0] > 0) {
        if (param.mmWeightScaleShape.size() == mmWeightScaleDimMx) {
            mmWeightScaleStorageShape = {
                {param.mmWeightScaleShape[0], param.mmWeightScaleShape[1], param.mmWeightScaleShape[2]},
                {param.mmWeightScaleShape[0], param.mmWeightScaleShape[1], param.mmWeightScaleShape[2]}};
        } else {
            std::cerr << "[WARN] mmWeightScaleShape unexpected dim: " << param.mmWeightScaleShape.size()
                      << " in case: " << param.case_name << std::endl;
            mmWeightScaleStorageShape = {{param.mmWeightScaleShape[0]}, {param.mmWeightScaleShape[0]}};
        }
    } else {
        mmWeightScaleStorageShape = {};
    }

    gert::TilingContextPara tilingContextPara(
        OP_NAME,
        {{{{param.gmmXShape[0], param.gmmXShape[1]}, {param.gmmXShape[0], param.gmmXShape[1]}},
          param.gmmXDataType,
          param.gmmXFormat},
         {{{param.gmmWeightShape[0], param.gmmWeightShape[1], param.gmmWeightShape[2]},
           {param.gmmWeightShape[0], param.gmmWeightShape[1], param.gmmWeightShape[2]}},
          param.gmmWeightDataType,
          param.gmmWeightFormat},
         {gmmXScaleStorageShape, param.gmmXScaleDataType, param.gmmXScaleFormat},
         {gmmWeightScaleStorageShape, param.gmmWeightScaleDataType, param.gmmWeightScaleFormat},
         {{}, ge::DT_INT64, ge::FORMAT_ND},
         {{}, ge::DT_INT64, ge::FORMAT_ND},
         {mmXStorageShape, param.mmXDataType, param.mmXFormat},
         {mmWeightStorageShape, param.mmWeightDataType, param.mmWeightFormat},
         {mmXScaleStorageShape, param.mmXScaleDataType, param.mmXScaleFormat},
         {mmWeightScaleStorageShape, param.mmWeightScaleDataType, param.mmWeightScaleFormat},
         {{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{{{param.gmmYShape[0], param.gmmYShape[1]}, {param.gmmYShape[0], param.gmmYShape[1]}},
          param.gmmYDataType,
          param.gmmYFormat},
         {mmYStorageShape, param.mmYDataType, param.mmYFormat}},
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.epWorldSize)},
            {"send_counts", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(param.sendCounts)},
            {"recv_counts", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(param.recvCounts)},
            {"gmm_x_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.gmmXQuantMode)},
            {"gmm_weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.gmmWeightQuantMode)},
            {"trans_gmm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(param.transGmmWeight)},
            {"trans_mm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(param.transMmWeight)},
            {"mm_x_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.mmXQuantMode)},
            {"mm_weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.mmWeightQuantMode)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.groupSize)},
            {"gmm_y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"mm_y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"comm_quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.commMode)},
        },
        &compileInfo, "3510", coreNum, ubSize, tilingDataSize);

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, param.expectResult, param.expectTilingKey);
}

INSTANTIATE_TEST_SUITE_P(QuantGmmAlltoAllvTilingUT, QuantGmmAlltoAllvTilingTest,
                         testing::ValuesIn(GetCasesFromCsv<QuantGmmAlltoAllvTilingUtParam>(GetCsvPath(__FILE__))),
                         PrintCaseInfoString<QuantGmmAlltoAllvTilingUtParam>);

} // namespace QuantGmmAlltoAllvUT
