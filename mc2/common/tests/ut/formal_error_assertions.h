/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MC2_UT_FORMAL_ERROR_ASSERTIONS_H
#define MC2_UT_FORMAL_ERROR_ASSERTIONS_H
#include <gtest/gtest.h>
#include <map>
#include <sstream>
#include <string>
#include "experiment/metadef/common/util/error_manager/error_manager.h"
namespace Mc2ErrorCodeUT {
class ExcelErrorReportScope {
public:
    ExcelErrorReportScope()
    {
        (void)ErrorManager::GetInstance().GetRawErrorMessages();
    }
    ~ExcelErrorReportScope()
    {
        (void)ErrorManager::GetInstance().GetRawErrorMessages();
    }
};
inline void ExpectExcelFormalError(const char* code, const char* parameter, const char* actual,
                                   const char* expectedReason = nullptr, const char* correctFormat = nullptr,
                                   const std::map<std::string, std::string>& expectedFields = {})
{
    const auto reports = ErrorManager::GetInstance().GetRawErrorMessages();
    std::ostringstream detail;
    for (const auto& report : reports) {
        detail << report.error_id << " ";
        for (const auto& field : report.args_map) {
            detail << field.first << "=" << field.second << "; ";
        }
    }
    SCOPED_TRACE(detail.str());
    ASSERT_EQ(reports.size(), 1U);
    ASSERT_EQ(reports[0].error_id, code);
    ASSERT_EQ(ErrorManager::GetInstance().SetRawErrorMessages(reports), 0);
    const auto formatted = ErrorManager::GetInstance().GetErrorMessage();
    EXPECT_NE(formatted.find(code), std::string::npos) << formatted;

    const auto& fields = reports[0].args_map;
    ASSERT_EQ(fields.count("op_name"), 1U);
    EXPECT_FALSE(fields.at("op_name").empty());
    for (const auto& expected : expectedFields) {
        ASSERT_EQ(fields.count(expected.first), 1U);
        EXPECT_EQ(fields.at(expected.first), expected.second);
    }
    if (std::string(code) == "EZ0004") {
        ASSERT_EQ(fields.count("param_name"), 1U);
        EXPECT_EQ(fields.at("param_name"), parameter);
        return;
    }
    if (std::string(code) == "EZ0017") {
        ASSERT_EQ(fields.count("param_name"), 1U);
        ASSERT_EQ(fields.count("incorrect_format"), 1U);
        ASSERT_EQ(fields.count("correct_format"), 1U);
        EXPECT_EQ(fields.at("param_name"), parameter);
        EXPECT_EQ(fields.at("incorrect_format"), actual);
        EXPECT_FALSE(fields.at("correct_format").empty());
        if (correctFormat != nullptr) {
            EXPECT_EQ(fields.at("correct_format"), correctFormat);
        }
        return;
    }
    if (std::string(code) == "EZ0019") {
        ASSERT_EQ(fields.count("param_name"), 1U);
        ASSERT_EQ(fields.count("incorrect_dtype"), 1U);
        ASSERT_EQ(fields.count("correct_dtype"), 1U);
        EXPECT_EQ(fields.at("param_name"), parameter);
        EXPECT_EQ(fields.at("incorrect_dtype"), actual);
        EXPECT_FALSE(fields.at("correct_dtype").empty());
        EXPECT_NE(formatted.find(fields.at("incorrect_dtype")), std::string::npos) << formatted;
        EXPECT_NE(formatted.find(fields.at("correct_dtype")), std::string::npos) << formatted;
        return;
    }
    if (std::string(code) != "EZ0024" && std::string(code) != "EZ0011") {
        ASSERT_EQ(fields.count("reason"), 1U);
        EXPECT_FALSE(fields.at("reason").empty());
    }
    if (expectedReason != nullptr) {
        ASSERT_EQ(fields.count("reason"), 1U);
        EXPECT_EQ(fields.at("reason"), expectedReason);
    }
    const bool relation = std::string(code) == "EZ0027";
    const char* nameKey = relation ? "param_names" : "param_name";
    const char* valueKey = relation ? "incorrect_values" :
                                      ((std::string(code) == "EZ0011" || std::string(code) == "EZ0012") ?
                                           "incorrect_dim" :
                                           (std::string(code) == "EZ0035" ? "incorrect_format" : "incorrect_value"));
    ASSERT_EQ(fields.count(nameKey), 1U);
    ASSERT_EQ(fields.count(valueKey), 1U);
    EXPECT_EQ(fields.at(nameKey), parameter);
    EXPECT_EQ(fields.at(valueKey), actual);
    EXPECT_NE(formatted.find(fields.at(nameKey)), std::string::npos) << formatted;
    EXPECT_NE(formatted.find(fields.at(valueKey)), std::string::npos) << formatted;
    if (std::string(code) == "EZ0011") {
        ASSERT_EQ(fields.count("correct_dim"), 1U);
        EXPECT_FALSE(fields.at("correct_dim").empty());
    }
    if (std::string(code) == "EZ0024") {
        ASSERT_EQ(fields.count("correct_value"), 1U);
        EXPECT_FALSE(fields.at("correct_value").empty());
    }
}
} // namespace Mc2ErrorCodeUT
#endif
