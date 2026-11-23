/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../../op_host/op_tiling/mhc_post_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "experiment/metadef/common/util/error_manager/error_manager.h"

#include "op_host/tiling_templates_registry.h"
#include "platform/platform_infos_def.h"

using namespace std;
using namespace optiling;
class MhcPostTiling : public testing::Test {};

namespace {
void ExpectTemplateShapeFailure(int32_t priority)
{
    // Select an immutable registered factory: no priority changes or global registry mutation.
    const auto& templates = Ops::Transformer::OpTiling::TilingRegistryArch::GetInstance().GetTilingTemplates(
        "MhcPost", static_cast<int32_t>(NpuArch::DAV_3510));
    auto it = templates.find(priority);
    ASSERT_NE(it, templates.end());
    ASSERT_NE(it->second, nullptr);
    gert::StorageShape xShape = {{1024, 4, 5120}, {1024, 4, 5120}};
    gert::StorageShape resShape = {{1024, 4, 4}, {1024, 4, 4}};
    gert::StorageShape outShape = {{1024, 5121}, {1024, 5121}};
    gert::StorageShape postShape = {{1024, 4}, {1024, 4}};
    gert::StorageShape yShape = xShape;
    auto makeTensor = [](const gert::StorageShape& shape, ge::DataType dtype) {
        return gert::Tensor(shape, gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                            gert::TensorPlacement::kOnHost, dtype, nullptr);
    };
    auto x = makeTensor(xShape, ge::DT_FLOAT16);
    auto res = makeTensor(resShape, ge::DT_FLOAT);
    auto out = makeTensor(outShape, ge::DT_FLOAT16);
    auto post = makeTensor(postShape, ge::DT_FLOAT);
    auto y = makeTensor(yShape, ge::DT_FLOAT16);
    fe::PlatFormInfos platform;
    ASSERT_TRUE(platform.Init());
    std::map<std::string, std::string> specs = {
        {"ub_size", "262144"}, {"core_num", "64"}, {"vector_core_cnt", "64"}, {"ai_core_cnt", "64"}};
    platform.SetPlatformRes("AICoreSpec", specs);
    std::map<std::string, std::string> version = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    platform.SetPlatformRes("version", version);
    platform.SetCoreNumByCoreType("AICore");
    MhcPostCompileInfo compileInfo = {64, 64, platform_ascendc::SocVersion::ASCEND950, NpuArch::DAV_3510};
    auto data = gert::TilingData::CreateCap(4096);
    auto workspaces = gert::ContinuousVector::Create<size_t>(16);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("MhcPost")
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1}, {1})
                      .InputTensors({&x, &res, &out, &post})
                      .OutputTensors({&y})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(&platform)
                      .TilingData(data.get())
                      .Workspace(reinterpret_cast<gert::ContinuousVector*>(workspaces.get()))
                      .Build();
    auto tiler = it->second(holder.GetContext());
    ASSERT_NE(tiler, nullptr);
    auto& errors = ErrorManager::GetInstance();
    ASSERT_EQ(errors.Init(), 0);
    (void)errors.GetRawErrorMessages();
    EXPECT_EQ(tiler->DoTiling(), ge::GRAPH_FAILED);
    const auto reports = errors.GetRawErrorMessages();
    ASSERT_EQ(reports.size(), 1U);
    EXPECT_EQ(reports[0].error_id, "EZ0008");
    const std::map<std::string, std::string> expected = {{"param_name", "h_out"},
                                                         {"op_name", "MhcPost"},
                                                         {"incorrect_shape", "[1024, 5121]"},
                                                         {"correct_shape", "[1024, 5120]"}};
    EXPECT_EQ(reports[0].args_map, expected);
}
} // namespace

TEST_F(MhcPostTiling, base_template_independently_preserves_shape_error)
{
    ExpectTemplateShapeFailure(20);
}

TEST_F(MhcPostTiling, regbase_template_independently_preserves_shape_error)
{
    ExpectTemplateShapeFailure(10);
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
