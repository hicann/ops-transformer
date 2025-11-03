/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_cross_entropy_loss.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../op_host/rope_quant_kvcache_tiling.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include <nlohmann/json.hpp>
#include "platform/platform_infos_def.h"
#include "base/registry/op_impl_space_registry_v2.h"

using namespace std;

struct RopeQuantKvcacheCompileInfo {};
class RopeQuantKvcacheTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "RopeQuantKvcacheTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "RopeQuantKvcacheTiling TearDown" << std::endl;
    }
};

#define DO_TILING(tilingPara)                                                                                          \
    auto contextFaker = gert::TilingContextFaker();                                                                    \
    /* 1. input/output information */                                                                                  \
    size_t inputNum = tilingContextPara.inputTensorDesc_.size();                                                       \
    size_t outputNum = tilingContextPara.outputTensorDesc_.size();                                                     \
    contextFaker.NodeIoNum(inputNum, outputNum);                                                                       \
    std::vector<gert::Tensor *> inputTensors = {};                                                                     \
    std::vector<gert::Tensor *> outputTensors = {};                                                                    \
    for (size_t index = 0; index < inputNum; index++) {                                                                \
        contextFaker.NodeInputTd(index,                                                                                \
                                 tilingContextPara.inputTensorDesc_[index].dtype_,                                     \
                                 tilingContextPara.inputTensorDesc_[index].format_,                                    \
                                 tilingContextPara.inputTensorDesc_[index].format_);                                   \
        if (tilingContextPara.inputTensorDesc_[index].shape_.GetStorageShape().GetDimNum() == 0){                      \
            inputTensors.push_back(nullptr);                                                                           \
        } else {                                                                                                       \
            inputTensors.push_back((gert::Tensor *)&tilingContextPara.inputTensorDesc_[index].shape_);                 \
        }                                                                                                              \
    }                                                                                                                  \
    for (size_t index = 0; index < outputNum; index++) {                                                               \
        contextFaker.NodeOutputTd(index,                                                                               \
                                  tilingContextPara.outputTensorDesc_[index].dtype_,                                   \
                                  tilingContextPara.outputTensorDesc_[index].format_,                                  \
                                  tilingContextPara.outputTensorDesc_[index].format_);                                 \
        if (tilingContextPara.outputTensorDesc_[index].shape_.GetStorageShape().GetDimNum() == 0){                     \
            outputTensors.push_back(nullptr);                                                                          \
        } else {                                                                                                       \
            outputTensors.push_back((gert::Tensor *)&tilingContextPara.outputTensorDesc_[index].shape_);               \
        }                                                                                                              \
    }                                                                                                                  \
    contextFaker.InputTensors(inputTensors).OutputTensors(outputTensors);                                              \
    for (auto& attrInfo : tilingContextPara.attrs_) {                                                                  \
        switch (attrInfo.attr_.type_) {                                                                                \
            case Ops::Transformer::AnyValue::ValueType::VT_BOOL: {                                                            \
                contextFaker.Attr(attrInfo.attrName_, *reinterpret_cast<bool*>(attrInfo.attr_.valuePtr_.get()));       \
                break;}                                                                                                \
            case Ops::Transformer::AnyValue::ValueType::VT_INT: {                                                             \
                contextFaker.Attr(attrInfo.attrName_, *reinterpret_cast<int64_t*>(attrInfo.attr_.valuePtr_.get()));    \
                break;}                                                                                                \
            case Ops::Transformer::AnyValue::ValueType::VT_FLOAT: {                                                           \
                contextFaker.Attr(attrInfo.attrName_, *reinterpret_cast<float*>(attrInfo.attr_.valuePtr_.get()));      \
                break;}                                                                                                \
            case Ops::Transformer::AnyValue::ValueType::VT_STRING: {                                                          \
                contextFaker.Attr(attrInfo.attrName_, ge::AscendString(reinterpret_cast<std::string*>(attrInfo.attr_.valuePtr_.get())->c_str()));\
                break;}                                                                                                \
            case Ops::Transformer::AnyValue::ValueType::VT_LIST_BOOL: {                                                       \
                contextFaker.Attr(attrInfo.attrName_, *reinterpret_cast<std::vector<bool>*>(attrInfo.attr_.valuePtr_.get()));\
                break;}                                                                                                \
            case Ops::Transformer::AnyValue::ValueType::VT_LIST_INT: {                                                        \
                contextFaker.Attr(attrInfo.attrName_, *reinterpret_cast<std::vector<int64_t>*>(attrInfo.attr_.valuePtr_.get()));\
                break;}                                                                                                \
            case Ops::Transformer::AnyValue::ValueType::VT_LIST_LIST_INT: {                                                   \
                contextFaker.Attr(attrInfo.attrName_, *reinterpret_cast<std::vector<std::vector<int64_t>>*>(attrInfo.attr_.valuePtr_.get()));\
                break;}                                                                                                \
            case Ops::Transformer::AnyValue::ValueType::VT_LIST_FLOAT: {                                                      \
                contextFaker.Attr(attrInfo.attrName_, *reinterpret_cast<std::vector<float>*>(attrInfo.attr_.valuePtr_.get()));\
                break;}                                                                                                \
            default:                                                                                                   \
                std::cout << "[ERROR]" << __FILE__ << ":" << __LINE__ << "The ValueType " << attrInfo.attr_.type_ << "is not supported!" << std::endl;\
        }                                                                                                              \
    }                                                                                                                  \
    /* 2. base information */                                                                                          \
    fe::PlatFormInfos platformInfo;                                                                                    \
    platformInfo.Init();                                                                                               \
    auto tilingData = gert::TilingData::CreateCap(tilingContextPara.tilingDataSize_);                                  \
    gert::ContinuousVector workspace;                                                                                  \
    auto contextHolder = contextFaker.SetOpType(tilingContextPara.opName_.c_str())                                     \
                                     .CompileInfo(tilingContextPara.compileInfo_)                                      \
                                     .PlatformInfo(reinterpret_cast<char*>(&platformInfo))                             \
                                     .TilingData(tilingData.get())                                                     \
                                     .Workspace(&workspace)                                                            \
                                     .Build();                                                                         \
    string compileInfoStringPrefix = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1", "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false, "UB_SIZE": )";\
    string compileInfoStringMiddle = R"(, "L2_SIZE": 33554432, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": )";\
    string compileInfoStringSuffix = std::string(", \"socVersion\":\"") + tilingContextPara.socVersion_ + "\"} }";\
    string compileInfoString = compileInfoStringPrefix +                                                               \
                               std::to_string(tilingContextPara.ubSize_) +                                             \
                               compileInfoStringMiddle +                                                               \
                               std::to_string(tilingContextPara.coreNum_) +                                            \
                               compileInfoStringSuffix;                                                                \
    map<string, string> socInfos;                                                                                      \
    map<string, string> aicoreSpec;                                                                                    \
    map<string, string> intrinsics;                                                                                    \
    map<string, string> socversions = {{"Short_SoC_version", tilingContextPara.socVersion_}};                          \
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics);                                     \
    auto tilingContext = contextHolder.GetContext();                                                                   \
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);                                             \
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);                                        \
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");                                                  \
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);                           \
    tilingContext->GetPlatformInfo()->SetPlatformRes("version", socversions);                                          \
    /* 3. get tiling func */                                                                                           \
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();                         \
    auto tilingFunc = spaceRegistry->GetOpImpl(tilingContextPara.opName_.c_str())->tiling;                             \
    /* 4. check tiling func */                                                                                         \
    auto tilingRet = tilingFunc(tilingContext);

template <typename T>
static string to_string(void* buf, size_t size) {
    string result;
    const T* data = reinterpret_cast<const T*>(buf);
    size_t len = size / sizeof(T);
    for (size_t i = 0; i < len; i++) {
        result += std::to_string(data[i]);
        result += " ";
    }
    return result;
}

static void GetPlatFormInfos(const char* compileInfoStr, map<string, string>& socInfos, map<string, string>& aicoreSpec,
                             map<string, string>& intrinsics) {
    string default_hardward_info = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1", "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false, "UB_SIZE": 262144, "L2_SIZE": 33554432,
                          "L1_SIZE": 1048576, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144,
                          "CORE_NUM": 32}})";
    nlohmann::json compileInfoJson = nlohmann::json::parse(compileInfoStr);
    if (compileInfoJson.type() != nlohmann::json::value_t::object) {
        compileInfoJson = nlohmann::json::parse(default_hardward_info.c_str());
    }

    map<string, string> socInfoKeys = {{"ai_core_cnt", "CORE_NUM"},
                                       {"l2_size", "L2_SIZE"},
                                       {"cube_core_cnt", "cube_core_cnt"},
                                       {"vector_core_cnt", "vector_core_cnt"},
                                       {"core_type_list", "core_type_list"}};
    socInfos["core_type_list"] = "AICore";

    for (auto &t : socInfoKeys) {
        if (compileInfoJson.contains("hardware_info") && compileInfoJson["hardware_info"].contains(t.second)) {
            auto &objJson = compileInfoJson["hardware_info"][t.second];
            if (objJson.is_number_integer()) {
                socInfos[t.first] = to_string(compileInfoJson["hardware_info"][t.second].get<uint32_t>());
            } else if (objJson.is_string()) {
                socInfos[t.first] = objJson;
            }
        }
    }
    map<string, string> aicoreSpecKeys = {{"ub_size", "UB_SIZE"},
                                          {"l0_a_size", "L0A_SIZE"},
                                          {"l0_b_size", "L0B_SIZE"},
                                          {"l0_c_size", "L0C_SIZE"},
                                          {"l1_size", "L1_SIZE"},
                                          {"bt_size", "BT_SIZE"},
                                          {"load3d_constraints", "load3d_constraints"}};
    aicoreSpec["cube_freq"] = "cube_freq";
    for (auto &t : aicoreSpecKeys) {
        if (compileInfoJson.contains("hardware_info") && compileInfoJson["hardware_info"].contains(t.second)) {
            if (t.second == "load3d_constraints") {
                aicoreSpec[t.first] = compileInfoJson["hardware_info"][t.second].get<string>();
            } else {
                aicoreSpec[t.first] = to_string(compileInfoJson["hardware_info"][t.second].get<uint32_t>());
            }
        }
    }

    std::string intrinsicsKeys[] = {"Intrinsic_data_move_l12ub", "Intrinsic_data_move_l0c2ub",
                                    "Intrinsic_fix_pipe_l0c2out", "Intrinsic_data_move_out2l1_nd2nz",
                                    "Intrinsic_matmul_ub_to_ub", "Intrinsic_conv_ub_to_ub",
                                    "Intrinsic_data_move_l12bt"};
    for (string key : intrinsicsKeys) {
        if (compileInfoJson.contains("hardware_info") && compileInfoJson["hardware_info"].contains(key) &&
            compileInfoJson["hardware_info"][key].get<bool>()) {
            intrinsics[key] = "float16";
            if (key.find("Intrinsic_data_move_l12bt") != string::npos) {
                intrinsics[key] = "bf16";
            }
        }
    }
}

void ExecuteTestCaseWithoutTilingKey(const gert::TilingContextPara& tilingContextPara, 
                     ge::graphStatus                expectResult,
                     const string&                  expectTilingData,
                     const std::vector<size_t>&     expectWorkspaces,
                     uint64_t                       tilingDataReservedLen = 0)
{
    DO_TILING(tilingContextPara);

    // check tiling func
    EXPECT_EQ(tilingRet, expectResult);
    if (expectResult == ge::GRAPH_FAILED) {
        return;
    }

    // check workspace
    if (!expectWorkspaces.empty()) {
        size_t workspaceCount = tilingContext->GetWorkspaceNum();
        if (workspaceCount > 0) {
            auto workspaceSizes = tilingContext->GetWorkspaceSizes(workspaceCount);
            for (size_t i = 0; i < workspaceCount; i++) {
                ASSERT_EQ(workspaceSizes[i], expectWorkspaces[i]);
            }
        }
    }

    // check tiling data
    if (expectTilingData != "") {
        auto rawTilingData = tilingContext->GetRawTilingData();
        auto tilingDataReservedSize = tilingDataReservedLen * sizeof(uint64_t);
        auto tilingDataResult = to_string<int64_t>(rawTilingData->GetData() + tilingDataReservedSize,
                                                   rawTilingData->GetDataSize() - tilingDataReservedSize);
        EXPECT_EQ(tilingDataResult, expectTilingData);
    }
}

TEST_F(RopeQuantKvcacheTiling, test_tiling)
{
    RopeQuantKvcacheCompileInfo compile_info = {};
    std::vector<int64_t> splitsParams = {1024, 128, 128};
    gert::TilingContextPara tilingContextPara(
        "RopeQuantKvcache",
        {
            {{{4, 1, 1280}, {4, 1, 1280}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 1, 1, 128}, {4, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 1, 1, 128}, {4, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{128}, {128}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4, 2048, 1, 128}, {4, 2048, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{4, 2048, 1, 128}, {4, 2048, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{4, 1}, {4, 1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{4, 1, 8, 128}, {4, 1, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 1, 1, 128}, {4, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 1, 1, 128}, {4, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 2048, 1, 128}, {4, 2048, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{4, 2048, 1, 128}, {4, 2048, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        {
            {"size_splits", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(splitsParams)},
            {"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"kv_output", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compile_info);
    int64_t expectTilingKey = UINT64_MAX;
    string expectTilingData = "8 1 128 2048 1024 128 128 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCaseWithoutTilingKey(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingData, expectWorkspaces);
}
