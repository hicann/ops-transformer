/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>
#include <map>
#include <string>
#include "mc2/common/tests/ut/formal_error_assertions.h"

namespace {
using Mc2ErrorCodeUT::ExcelErrorReportScope;
using Mc2ErrorCodeUT::ExpectExcelFormalError;

void ReportSingleValue(const char* actual)
{
    ASSERT_EQ(ErrorManager::GetInstance().ReportErrMessage("EZ0024", {{"op_name", "ExactFieldsUT"},
                                                                      {"param_name", "world_size"},
                                                                      {"incorrect_value", actual},
                                                                      {"correct_value", "8"}}),
              0);
}

void ReportRankRelation(const char* actual)
{
    ASSERT_EQ(ErrorManager::GetInstance().ReportErrMessage("EZ0027",
                                                           {{"op_name", "ExactFieldsUT"},
                                                            {"param_names", "ep_rank_id, ep_world_size"},
                                                            {"incorrect_values", actual},
                                                            {"reason", "ep_rank_id must be in [0, ep_world_size)"}}),
              0);
}

TEST(FormalErrorAssertionTest, RejectsValueContainingExpectedDigits)
{
    ExcelErrorReportScope scope;
    ReportSingleValue("10");
    EXPECT_NONFATAL_FAILURE(ExpectExcelFormalError("EZ0024", "world_size", "0"), "fields.at(valueKey)");
}

TEST(FormalErrorAssertionTest, RejectsSwappedRelationValues)
{
    ExcelErrorReportScope scope;
    ReportRankRelation("ep_rank_id=0, ep_world_size=288");
    EXPECT_NONFATAL_FAILURE(
        ExpectExcelFormalError("EZ0027", "ep_rank_id, ep_world_size", "ep_rank_id=288, ep_world_size=288"),
        "fields.at(valueKey)");
}

TEST(FormalErrorAssertionTest, RejectsWrongCorrectValue)
{
    ExcelErrorReportScope scope;
    ReportSingleValue("0");
    EXPECT_NONFATAL_FAILURE(
        ExpectExcelFormalError("EZ0024", "world_size", "0", nullptr, nullptr, {{"correct_value", "16"}}),
        "expected.second");
}

TEST(FormalErrorAssertionTest, RejectsWrongOperatorName)
{
    ExcelErrorReportScope scope;
    ReportSingleValue("0");
    EXPECT_NONFATAL_FAILURE(
        ExpectExcelFormalError("EZ0024", "world_size", "0", nullptr, nullptr, {{"op_name", "WrongOperator"}}),
        "expected.second");
}

TEST(FormalErrorAssertionTest, RejectsPartialReason)
{
    ExcelErrorReportScope scope;
    ReportRankRelation("ep_rank_id=288, ep_world_size=288");
    EXPECT_NONFATAL_FAILURE(ExpectExcelFormalError("EZ0027", "ep_rank_id, ep_world_size",
                                                   "ep_rank_id=288, ep_world_size=288", "ep_rank_id"),
                            "expectedReason");
}

TEST(FormalErrorAssertionTest, AcceptsCompleteSingleAndRelationFields)
{
    ExcelErrorReportScope scope;
    ReportSingleValue("0");
    ExpectExcelFormalError("EZ0024", "world_size", "0", nullptr, nullptr,
                           {{"correct_value", "8"}, {"op_name", "ExactFieldsUT"}});
    ReportRankRelation("ep_rank_id=288, ep_world_size=288");
    ExpectExcelFormalError("EZ0027", "ep_rank_id, ep_world_size", "ep_rank_id=288, ep_world_size=288",
                           "ep_rank_id must be in [0, ep_world_size)", nullptr, {{"op_name", "ExactFieldsUT"}});
}
} // namespace
