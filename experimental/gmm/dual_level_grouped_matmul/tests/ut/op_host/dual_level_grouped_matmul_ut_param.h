// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#ifndef DUAL_LEVEL_GROUPED_MATMUL_UT_PARAM_H_
#define DUAL_LEVEL_GROUPED_MATMUL_UT_PARAM_H_

#include <cstdint>
#include <string>
#include <vector>
#include "op_host_csv_case_loader.h"
#include "tiling_context_faker.h"
#include "infer_shape_context_faker.h"
#include "../../../op_host/dual_level_grouped_matmul_tiling.h"
#include "../../../op_kernel/dual_level_grouped_matmul_tiling_data.h"

namespace DualLevelGmmUT {

using DualLevelGmmTilingData = DualLevelGroupedMatmul::DualLevelGroupedMatmulTilingData;

inline gert::Shape MakeGeShape(const std::vector<int64_t>& dims)
{
    gert::Shape shape;
    shape.SetDimNum(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        shape.SetDim(i, dims[i]);
    }
    return shape;
}

inline gert::StorageShape MakeGeStorageShape(const std::vector<int64_t>& originDims,
                                             const std::vector<int64_t>& storageDims)
{
    gert::StorageShape shape;
    shape.MutableOriginShape() = MakeGeShape(originDims);
    shape.MutableStorageShape() = MakeGeShape(storageDims.empty() ? originDims : storageDims);
    return shape;
}

// Absent optional input placeholder matching the optional-input contract in tiling.
template <typename T>
inline const T& GetDlgmmAbsentTensor()
{
    static const T absent(gert::StorageShape{}, ge::DT_FLOAT, ge::FORMAT_ND);
    return absent;
}

// Parses a tensor with a separate storage-shape column; storage defaults to the view shape.
template <typename T>
inline int GetTensorGEWithStorage(const csv_map& csvMap, const std::string& shapeKey, const std::string& storageKey,
                                  const std::string& dtypeKey, const std::string& formatKey, T& out)
{
    const std::string shapeStr = ReadMap(csvMap, shapeKey);
    const std::string dtypeStr = ReadMap(csvMap, dtypeKey);
    const std::string formatStr = ReadMap(csvMap, formatKey);
    if (shapeStr.empty() && dtypeStr.empty() && formatStr.empty()) {
        return 0;
    }
    std::vector<int64_t> originDims = GetShapeArr(shapeStr);
    std::vector<int64_t> storageDims = GetShapeArr(ReadMap(csvMap, storageKey));
    out = T(MakeGeStorageShape(originDims, storageDims), Str2DTypeGE(dtypeStr),
            ReadMap(GE_FORMAT, formatStr, ge::FORMAT_NULL));
    return 1;
}

struct DualLevelGmmHostUtParamBase : public HostUtParamBase {
    bool transposeWeight = false;
    int64_t groupListType = 0;
    int64_t outputDtype = ge::DT_FLOAT16;
    std::vector<int64_t> tuningConfig = {0};

    DualLevelGmmHostUtParamBase(const csv_map& csvMap)
        : HostUtParamBase(csvMap)
    {
        transposeWeight = StrToBoolIgnoreCase(ReadMap(csvMap, "transpose_weight", "false"));
        groupListType = std::stoll(ReadMap(csvMap, "group_list_type", "0"));
        outputDtype = Str2DTypeGE(ReadMap(csvMap, "dtype", "FLOAT16"));
        tuningConfig = GetShapeArr(ReadMap(csvMap, "tuning_config", "0"));
        if (tuningConfig.empty()) {
            tuningConfig = {0};
        }
    }

    // Attr order is fixed by the op impl: transpose_weight, group_list_type, tuning_config, dtype.
    template <typename T>
    std::vector<typename T::OpAttr> BuildAttrs() const
    {
        return {{"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transposeWeight)},
                {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)},
                {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(tuningConfig)},
                {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outputDtype)}};
    }

    std::vector<std::pair<std::string, Ops::Transformer::AnyValue>> BuildDtypeAttrs() const
    {
        return {{"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transposeWeight)},
                {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)},
                {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(tuningConfig)},
                {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outputDtype)}};
    }
};

struct DualLevelGmmTilingUtParam : public DualLevelGmmHostUtParamBase {
    gert::TilingContextPara::TensorDescription x = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription xLevel0Scale = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription perTokenScale = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription weight = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription weightLevel0Scale = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription scale = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription groupList = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription bias = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription y = TD_DEFAULT;

    bool hasBias = false;
    std::string socVersion = "Ascend950";
    bool shortTilingBuffer = false;
    uint64_t expectTilingKey = 0;
    uint64_t expectCoreNum = 32;
    uint32_t aicNum = 32;
    uint32_t aivNum = 64;
    std::string level0GroupSize;
    std::string level1GroupSize;

    DualLevelGmmTilingUtParam(const csv_map& csvMap)
        : DualLevelGmmHostUtParamBase(csvMap)
    {
        (void)GetTensorGE(csvMap, "x_shape", "x_dtype", "x_format", x);
        (void)GetTensorGE(csvMap, "x_level0_shape", "x_level0_dtype", "x_level0_format", xLevel0Scale);
        (void)GetTensorGE(csvMap, "per_token_shape", "per_token_dtype", "per_token_format", perTokenScale);
        (void)GetTensorGEWithStorage(csvMap, "weight_shape", "weight_storage_shape", "weight_dtype", "weight_format",
                                     weight);
        (void)GetTensorGE(csvMap, "weight_level0_shape", "weight_level0_dtype", "weight_level0_format",
                          weightLevel0Scale);
        (void)GetTensorGEWithStorage(csvMap, "scale_shape", "scale_storage_shape", "scale_dtype", "scale_format",
                                     scale);
        (void)GetTensorGE(csvMap, "group_list_shape", "group_list_dtype", "group_list_format", groupList);
        hasBias = GetTensorGE(csvMap, "bias_shape", "bias_dtype", "bias_format", bias) != 0;
        (void)GetTensorGE(csvMap, "y_shape", "y_dtype", "y_format", y);
        socVersion = ReadMap(csvMap, "soc_version", "Ascend950");
        aicNum = std::stoul(ReadMap(csvMap, "aic_num", "32"));
        aivNum = std::stoul(ReadMap(csvMap, "aiv_num", "64"));
        level0GroupSize = ReadMap(csvMap, "level0_group_size", "");
        level1GroupSize = ReadMap(csvMap, "level1_group_size", "");
        shortTilingBuffer = StrToBoolIgnoreCase(ReadMap(csvMap, "short_tiling_buffer", "false"));
        if (expectResult == ge::GRAPH_SUCCESS) {
            expectTilingKey = std::stoull(ReadMap(csvMap, "expect_tiling_key", "0"));
            expectCoreNum = std::stoull(ReadMap(csvMap, "expect_core_num", "32"));
        }
    }

    gert::TilingContextPara::TensorDescription BuildBias() const
    {
        return hasBias ? bias : GetDlgmmAbsentTensor<gert::TilingContextPara::TensorDescription>();
    }
};

struct DualLevelGmmInferShapeUtParam : public DualLevelGmmHostUtParamBase {
    gert::InfershapeContextPara::TensorDescription x = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription xLevel0Scale = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription perTokenScale = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription weight = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription weightLevel0Scale = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription scale = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription groupList = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription bias = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription y = ID_DEFAULT;

    std::string caseType = "shape";
    std::vector<int64_t> expectYShape;

    DualLevelGmmInferShapeUtParam(const csv_map& csvMap)
        : DualLevelGmmHostUtParamBase(csvMap)
    {
        caseType = ReadMap(csvMap, "case_type", "shape");
        (void)GetTensorGE(csvMap, "x_shape", "x_dtype", "x_format", x);
        (void)GetTensorGE(csvMap, "x_level0_shape", "x_level0_dtype", "x_level0_format", xLevel0Scale);
        (void)GetTensorGE(csvMap, "per_token_shape", "per_token_dtype", "per_token_format", perTokenScale);
        (void)GetTensorGE(csvMap, "weight_shape", "weight_dtype", "weight_format", weight);
        (void)GetTensorGE(csvMap, "weight_level0_shape", "weight_level0_dtype", "weight_level0_format",
                          weightLevel0Scale);
        (void)GetTensorGE(csvMap, "scale_shape", "scale_dtype", "scale_format", scale);
        (void)GetTensorGE(csvMap, "group_list_shape", "group_list_dtype", "group_list_format", groupList);
        (void)GetTensorGE(csvMap, "bias_shape", "bias_dtype", "bias_format", bias);
        (void)GetTensorGE(csvMap, "y_shape", "y_dtype", "y_format", y);
        expectYShape = GetShapeArr(ReadMap(csvMap, "expect_y_shape"));
    }

    bool RunInferDtypeOnly() const
    {
        return caseType == "dtype_only";
    }
};

inline const std::string& GetDlgmmSocInfoString()
{
    static const std::string socInfoString = R"({"hardware_info":{"CORE_NUM":32,"cube_core_cnt":32,"vector_core_cnt":64,
        "UB_SIZE":262144,"L1_SIZE":524288,"L0A_SIZE":65536,"L0B_SIZE":65536,"L0C_SIZE":262144,
        "L2_SIZE":33554432,"BT_SIZE":0,"load3d_constraints":"1"}})";
    return socInfoString;
}

} // namespace DualLevelGmmUT

#endif // DUAL_LEVEL_GROUPED_MATMUL_UT_PARAM_H_
