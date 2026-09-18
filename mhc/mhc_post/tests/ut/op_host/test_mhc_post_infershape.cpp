/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <iostream>

#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

#define private public
#include "platform/platform_info.h"

class MhcPost : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MhcPost Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MhcPost Proto Test TearDown" << std::endl;
    }
};

TEST_F(MhcPost, MhcPost_normal_dims4)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    optiCompilationInfo.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    gert::InfershapeContextPara infershapeContextPara(
        "MhcPost",
        {{{{512, 2, 4, 512}, {512, 2, 4, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{512, 2, 4, 4}, {512, 2, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{512, 2, 512}, {512, 2, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{512, 2, 4}, {512, 2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {});
    std::vector<std::vector<int64_t>> expectOutputShape = {{512, 2, 4, 512}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MhcPost, MhcPost_normal_dims3)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    optiCompilationInfo.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    gert::InfershapeContextPara infershapeContextPara(
        "MhcPost",
        {
            {{{1024, 4, 512}, {1024, 4, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024, 4, 4}, {1024, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1024, 512}, {1024, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1024, 4}, {1024, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {});
    std::vector<std::vector<int64_t>> expectOutputShape = {{1024, 4, 512}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MhcPost, MhcPost_unknowrank)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    optiCompilationInfo.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    gert::InfershapeContextPara infershapeContextPara(
        "MhcPost",
        {
            {{{-2}, {-2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512, 2, 4, 4}, {512, 2, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{512, 2, 512}, {512, 2, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512, 2, 4}, {512, 2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {});
    std::vector<std::vector<int64_t>> expectOutputShape = {{-2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MhcPost, MhcPost_unknowshape)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    optiCompilationInfo.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    gert::InfershapeContextPara infershapeContextPara(
        "MhcPost",
        {
            {{{-1, -1, -1, -1}, {-1, -1, -1, -1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512, 2, 4, 4}, {512, 2, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{512, 2, 512}, {512, 2, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512, 2, 4}, {512, 2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {});
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1, -1, -1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MhcPost, MhcPost_mixed_shape_01)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    optiCompilationInfo.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    gert::InfershapeContextPara infershapeContextPara(
        "MhcPost",
        {
            {{{-1, 2, 4, -1}, {-1, 2, 4, -1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512, 2, 4, 4}, {512, 2, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{512, 2, 512}, {512, 2, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512, 2, 4}, {512, 2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {});
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, 2, 4, -1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MhcPost, MhcPost_mixed_shape_02)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    optiCompilationInfo.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    gert::InfershapeContextPara infershapeContextPara("MhcPost",
                                                      {
                                                          {{{-1, 4, 512}, {-1, 4, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                                          {{{1024, 4, 4}, {1024, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{1024, 512}, {1024, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{1024, 4}, {1024, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                                      },
                                                      {});
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, 4, 512}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MhcPost, MhcPost_mixed_shape_03)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    optiCompilationInfo.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    gert::InfershapeContextPara infershapeContextPara(
        "MhcPost",
        {
            {{{-1, 2, 4, -1}, {-1, 2, 4, -1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{512, 2, 512}, {512, 2, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512, 2, 4}, {512, 2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {});
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, 2, 4, -1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MhcPost, MhcPost_xDims5)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    optiCompilationInfo.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    gert::InfershapeContextPara infershapeContextPara(
        "MhcPost",
        {
            {{{1, 512, 2, 4, 512}, {1, 512, 2, 4, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512, 2, 4, 4}, {512, 2, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{512, 2, 512}, {512, 2, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512, 2, 4}, {512, 2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{512, 2, 4, 512}, {512, 2, 4, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {});
    std::vector<std::vector<int64_t>> expectOutputShape = {{512, 2, 4, 512}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}
