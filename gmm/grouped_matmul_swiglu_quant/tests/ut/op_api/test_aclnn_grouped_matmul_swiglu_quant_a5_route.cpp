/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "log/log.h"
#include <dlfcn.h>
#include <vector>
#include <new>
#include <string>
#include "aclnn_kernels/contiguous.h"
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "../../../op_api/grouped_matmul_swiglu_quant.h"
#include "../../../op_api/aclnn_grouped_matmul_swiglu_quant_weight_nz.h"
#include "../../../op_api/aclnn_grouped_matmul_swiglu_quant.h"
#include "../../../../grouped_matmul/op_api/grouped_matmul_util.h"
#include "../../../../grouped_matmul_swiglu_quant_v2/op_api/aclnn_grouped_matmul_swiglu_quant_weight_nz_v2.h"

#include <cstdio>
#include <utility>
#include "gtest/gtest.h"
#include "op_api_ut_common/tensor_desc.h"

namespace {
struct RouteProbe {
    int firstCalls = 0;
    int secondCalls = 0;
    int legacyRunCalls = 0;
    int createCalls = 0;
    int allocCalls = 0;
    int failAllocAt = 0;
    bool failCreate = false;
    aclnnStatus status = ACLNN_ERR_INNER;
    const aclTensor* x = nullptr;
    const aclTensor* weight = nullptr;
    const aclTensor* scale = nullptr;
    const aclTensor* bias = nullptr;
    const aclTensor* xScale = nullptr;
    const aclTensor* groups = nullptr;
    aclTensor* output = nullptr;
    aclTensor* outputScale = nullptr;
    void* workspace = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclrtStream stream = nullptr;
    std::string info;
    std::string error;
};
RouteProbe probe;
// Keep real executor/list ownership while injecting only the allocation outcomes.
auto RealRouteExecutor()
{
    using namespace op;
    return CREATE_EXECUTOR();
}
using RealExecutor = decltype(RealRouteExecutor());
struct ProbeExecutor {
    RealExecutor executor = RealRouteExecutor();
    aclOpExecutor* get()
    {
        return probe.failCreate ? nullptr : executor.get();
    }
    ProbeExecutor* operator->()
    {
        return this;
    }
    aclTensorList* AllocTensorList(const aclTensor* const* tensors, uint64_t size)
    {
        ++probe.allocCalls;
        if (probe.allocCalls == probe.failAllocAt) {
            return nullptr;
        }
        return executor->AllocTensorList(tensors, size);
    }
    uint64_t GetWorkspaceSize()
    {
        return executor->GetWorkspaceSize();
    }
    void ReleaseTo(aclOpExecutor** output)
    {
        executor.ReleaseTo(output);
    }
};
ProbeExecutor MakeProbeExecutor()
{
    ++probe.createCalls;
    return {};
}
void ProbeInfo(const char* message)
{
    probe.info += message;
}
template <typename... Args>
void ProbeError(aclnnStatus, const char* format, Args... args)
{
    char buffer[2048];
    std::snprintf(buffer, sizeof(buffer), format, args...);
    probe.error += buffer;
}
aclnnStatus ProbeLegacyRun(void*, uint64_t, aclOpExecutor*, aclrtStream)
{
    ++probe.legacyRunCalls;
    return ACLNN_SUCCESS;
}

aclnnStatus ProbeV2First(const aclTensor* x, const aclTensorList* weight, const aclTensorList* weightScale,
                         const aclTensorList* assistance, const aclTensor* bias, const aclTensor* xScale,
                         const aclTensor* smoothScale, const aclTensor* groups, int64_t dequantMode,
                         int64_t dequantDtype, int64_t quantMode, int64_t groupListType, const aclIntArray* tuning,
                         aclTensor* output, aclTensor* outputScale, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    ++probe.firstCalls;
    EXPECT_EQ(x, probe.x);
    EXPECT_EQ(weight->Size(), 1U);
    EXPECT_EQ(weightScale->Size(), 1U);
    EXPECT_EQ((*weight)[0], probe.weight);
    EXPECT_EQ((*weightScale)[0], probe.scale);
    EXPECT_EQ(bias, probe.bias);
    EXPECT_EQ(xScale, probe.xScale);
    EXPECT_EQ(groups, probe.groups);
    EXPECT_EQ(output, probe.output);
    EXPECT_EQ(outputScale, probe.outputScale);
    EXPECT_EQ(assistance, nullptr);
    EXPECT_EQ(smoothScale, nullptr);
    EXPECT_EQ(tuning, nullptr);
    EXPECT_EQ(dequantMode, 0);
    EXPECT_EQ(dequantDtype, 0);
    EXPECT_EQ(quantMode, 0);
    EXPECT_EQ(groupListType, 0);
    EXPECT_NE(workspaceSize, nullptr);
    EXPECT_NE(executor, nullptr);
    // A failed callee must not publish a successful workspace/executor.
    return probe.status;
}
aclnnStatus ProbeV2Second(void* workspace, uint64_t size, aclOpExecutor* executor, aclrtStream stream)
{
    ++probe.secondCalls;
    EXPECT_EQ(workspace, probe.workspace);
    EXPECT_EQ(size, probe.workspaceSize);
    EXPECT_EQ(executor, probe.executor);
    EXPECT_EQ(stream, probe.stream);
    return probe.status;
}
} // namespace

namespace l0op {
static const aclTensor* StcUnexpectedContiguous(const aclTensor*, aclOpExecutor*)
{
    ADD_FAILURE() << "Legacy kernel preparation must not be reached by these routing cases";
    return nullptr;
}
static aclTensor* StcUnexpectedViewCopy(const aclTensor*, aclTensor*, aclOpExecutor*)
{
    ADD_FAILURE() << "Legacy output copy must not be reached by these routing cases";
    return nullptr;
}
} // namespace l0op

// Compile the production body under private names. Dependency seams exist only
// in this TU; the public APIs and real V2 validation are tested in the other suites.
// DFX is excluded here so cache hits cannot bypass a routing assertion.
#undef CREATE_EXECUTOR
#define CREATE_EXECUTOR() MakeProbeExecutor()
#undef L2_DFX_PHASE_1
#define L2_DFX_PHASE_1(...)
#undef L2_DFX_PHASE_2
#define L2_DFX_PHASE_2(...)
#undef OP_LOGI
#define OP_LOGI(...) ProbeInfo(__VA_ARGS__)
#undef OP_LOGE
#define OP_LOGE(...) ProbeError(__VA_ARGS__)
#undef OP_LOGE_WITHOUT_REPORT
#define OP_LOGE_WITHOUT_REPORT(...) ProbeError(__VA_ARGS__)
// Structured logs bypass OP_LOGE. Capture the rendered message and the reported code.
#undef OP_LOGE_LIBOPAPI_REPORT
#define OP_LOGE_LIBOPAPI_REPORT(opName, ...) ProbeError(ACLNN_ERR_PARAM_INVALID, __VA_ARGS__)
#undef REPORT_PREDEFINED_ERR_MSG
#define REPORT_PREDEFINED_ERR_MSG(code, keys, values) (probe.error += code)
#undef OP_LOGW
#define OP_LOGW(...)
#undef OP_LOGD
#define OP_LOGD(...)
#define CommonOpExecutorRun ProbeLegacyRun
#define Contiguous StcUnexpectedContiguous
#define ViewCopy StcUnexpectedViewCopy
#define aclnnGroupedMatmulSwigluQuantGetWorkspaceSize StcNormalFirst
#define aclnnGroupedMatmulSwigluQuantWeightNZGetWorkspaceSize StcNzFirst
#define aclnnGroupedMatmulSwigluQuant StcNormalSecond
#define aclnnGroupedMatmulSwigluQuantWeightNZ StcNzSecond
#define aclnnGroupedMatmulSwigluQuantWeightNzV2GetWorkspaceSize ProbeV2First
#define aclnnGroupedMatmulSwigluQuantWeightNzV2 ProbeV2Second
#define isEnableWeightAssistanceMatrix stcWeightAssistance
#define dequantMode stcDequantMode
#include "../../../op_api/aclnn_grouped_matmul_swiglu_quant.cpp"
#undef aclnnGroupedMatmulSwigluQuantGetWorkspaceSize
#undef aclnnGroupedMatmulSwigluQuantWeightNZGetWorkspaceSize
#undef aclnnGroupedMatmulSwigluQuant
#undef aclnnGroupedMatmulSwigluQuantWeightNZ
#undef aclnnGroupedMatmulSwigluQuantWeightNzV2GetWorkspaceSize
#undef aclnnGroupedMatmulSwigluQuantWeightNzV2
#undef isEnableWeightAssistanceMatrix
#undef dequantMode
#undef CREATE_EXECUTOR
#undef OP_LOGI
#undef OP_LOGE
#undef OP_LOGE_WITHOUT_REPORT
#undef OP_LOGE_LIBOPAPI_REPORT
#undef REPORT_PREDEFINED_ERR_MSG
#undef OP_LOGW
#undef OP_LOGD
#undef CommonOpExecutorRun
#undef Contiguous
#undef ViewCopy
#undef L2_DFX_PHASE_1
#undef L2_DFX_PHASE_2

#include <cstdlib>
#include <fstream>
#include <optional>
#include "gmm_csv_acl_parse_utils.h"

namespace {
// CSV tensor specs use the shared shape#strides#storage syntax. NONE denotes
// nullptr and SCALAR denotes a rank-zero tensor, so those cases stay distinct.
using TensorPtr = std::unique_ptr<aclTensor, void (*)(aclTensor*)>;
constexpr uint64_t WORKSPACE_SENTINEL = 12345;
constexpr size_t ROUTE_CSV_COLUMNS = 40;

struct RouteCase {
    std::string caseName;
    std::string socVersion;
    std::string mode;
    std::string api;
    std::string xSpec;
    std::string xDtype;
    std::string weightSpec;
    std::string weightDtype;
    std::string weightFormat;
    std::string scaleSpec;
    std::string scaleDtype;
    std::string xScaleSpec;
    std::string xScaleDtype;
    std::string groupSpec;
    std::string groupDtype;
    std::string outputSpec;
    std::string outputDtype;
    std::string outputScaleSpec;
    std::string outputScaleDtype;
    std::string biasSpec;
    std::string biasDtype;
    std::string offsetSpec;
    std::string offsetDtype;
    std::string outputOffsetSpec;
    std::string outputOffsetDtype;
    std::string weightOffset;
    std::string expectRet;
    std::string compareV2;
    std::string expectedLog;
    std::string failCreate;
    std::string failAllocAt;
    std::string mockRet;
    std::string expectFirstCalls;
    std::string expectCreateCalls;
    std::string expectAllocCalls;
    std::string expectSecondCalls;
    std::string expectLegacyCalls;
    std::string nullWorkspace;
    std::string nullExecutor;
    std::string expectRoute;
};

aclnnStatus ParseStatus(const std::string& value)
{
    if (value == "ACLNN_ERR_INNER_CREATE_EXECUTOR") {
        return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    }
    return ops::ut::ParseAclnnStatus(value);
}

std::vector<RouteCase> LoadCases(const std::string& path)
{
    std::ifstream input(path);
    EXPECT_TRUE(input.is_open()) << "Failed to open CSV file: " << path;
    std::vector<RouteCase> cases;
    std::string line;
    std::getline(input, line);
    size_t lineNo = 1;
    while (std::getline(input, line)) {
        ++lineNo;
        if (ops::ut::Trim(line).empty()) {
            continue;
        }
        std::vector<std::string> cols;
        ops::ut::SplitStr2Vec(line, ",", cols);
        if (cols.size() != ROUTE_CSV_COLUMNS) {
            ADD_FAILURE() << path << ":" << lineNo << ": expected " << ROUTE_CSV_COLUMNS << " columns, got "
                          << cols.size();
            continue;
        }
        for (auto& col : cols) {
            col = ops::ut::Trim(col);
        }
        cases.push_back({cols[0],  cols[1],  cols[2],  cols[3],  cols[4],  cols[5],  cols[6],  cols[7],
                         cols[8],  cols[9],  cols[10], cols[11], cols[12], cols[13], cols[14], cols[15],
                         cols[16], cols[17], cols[18], cols[19], cols[20], cols[21], cols[22], cols[23],
                         cols[24], cols[25], cols[26], cols[27], cols[28], cols[29], cols[30], cols[31],
                         cols[32], cols[33], cols[34], cols[35], cols[36], cols[37], cols[38], cols[39]});
    }
    EXPECT_FALSE(cases.empty()) << "No cases parsed from CSV: " << path;
    return cases;
}

op::Shape ToShape(const std::vector<int64_t>& dims)
{
    op::Shape shape;
    for (const auto dim : dims) {
        shape.AppendDim(dim);
    }
    return shape;
}

TensorPtr MakeTensor(const std::string& value, const std::string& dtype, const std::string& format = "ND")
{
    if (value == "NONE") {
        return TensorPtr(nullptr, static_cast<void (*)(aclTensor*)>(Release));
    }
    auto spec = value == "SCALAR" ? ops::ut::AclTensorSpec{} : ops::ut::ParseAclTensorSpec(value);
    auto safeShape = spec.viewDims;
    for (auto& dim : safeShape) {
        if (dim < 0) {
            dim = 1;
        }
    }
    // Construct a valid object first. Invalid storage/stride/shape metadata from
    // negative CSV rows must reach the API checker rather than a constructor.
    auto tensor = TensorDesc(safeShape, ops::ut::ParseAclDtype(dtype), ops::ut::ParseAclFormat(format)).ToAclType();
    tensor->SetViewShape(ToShape(spec.viewDims));
    if (!spec.stride.empty()) {
        op::Strides strides;
        for (const auto stride : spec.stride) {
            strides.push_back(stride);
        }
        tensor->SetViewStrides(strides);
    }
    if (!spec.storageDims.empty()) {
        tensor->SetStorageShape(ToShape(spec.storageDims));
    }
    return tensor;
}

struct RouteInputs {
    TensorPtr x;
    TensorPtr weight;
    TensorPtr scale;
    TensorPtr xScale;
    TensorPtr groups;
    TensorPtr output;
    TensorPtr outputScale;
    TensorPtr bias;
    TensorPtr offset;
    TensorPtr outputOffset;

    explicit RouteInputs(const RouteCase& c)
        : x(MakeTensor(c.xSpec, c.xDtype)),
          weight(MakeTensor(c.weightSpec, c.weightDtype, c.weightFormat)),
          scale(MakeTensor(c.scaleSpec, c.scaleDtype)),
          xScale(MakeTensor(c.xScaleSpec, c.xScaleDtype)),
          groups(MakeTensor(c.groupSpec, c.groupDtype)),
          output(MakeTensor(c.outputSpec, c.outputDtype)),
          outputScale(MakeTensor(c.outputScaleSpec, c.outputScaleDtype)),
          bias(MakeTensor(c.biasSpec, c.biasDtype)),
          offset(MakeTensor(c.offsetSpec, c.offsetDtype)),
          outputOffset(MakeTensor(c.outputOffsetSpec, c.outputOffsetDtype))
    {
        if (weight != nullptr) {
            weight->SetViewOffset(std::stoll(c.weightOffset));
        }
    }
};

struct TensorMetadata {
    op::Shape view;
    op::Shape storage;
    op::Strides strides;
    int64_t offset;
    op::Format viewFormat;
    op::Format storageFormat;
    op::DataType dtype;

    explicit TensorMetadata(const aclTensor* tensor)
        : view(tensor->GetViewShape()),
          storage(tensor->GetStorageShape()),
          strides(tensor->GetViewStrides()),
          offset(tensor->GetViewOffset()),
          viewFormat(tensor->GetViewFormat()),
          storageFormat(tensor->GetStorageFormat()),
          dtype(tensor->GetDataType())
    {}

    void ExpectUnchanged(const aclTensor* tensor) const
    {
        EXPECT_EQ(tensor->GetViewShape(), view);
        EXPECT_EQ(tensor->GetStorageShape(), storage);
        EXPECT_EQ(tensor->GetViewStrides(), strides);
        EXPECT_EQ(tensor->GetViewOffset(), offset);
        EXPECT_EQ(tensor->GetViewFormat(), viewFormat);
        EXPECT_EQ(tensor->GetStorageFormat(), storageFormat);
        EXPECT_EQ(tensor->GetDataType(), dtype);
    }
};

void ExpectLogTokens(const std::string& log, const std::string& expected)
{
    if (expected == "NONE") {
        return;
    }
    std::vector<std::string> tokens;
    ops::ut::SplitStr2Vec(expected, "|", tokens);
    for (const auto& token : tokens) {
        EXPECT_NE(log.find(token), std::string::npos) << "Missing log token: " << token << "\n" << log;
    }
}

aclnnStatus CallFirst(const RouteCase& c, RouteInputs& inputs, uint64_t* workspace, aclOpExecutor** executor)
{
    auto function = c.api == "Normal" ? aclnnGroupedMatmulSwigluQuantGetWorkspaceSize :
                                        aclnnGroupedMatmulSwigluQuantWeightNZGetWorkspaceSize;
    if (c.mode == "MOCK_FIRST") {
        function = c.api == "Normal" ? StcNormalFirst : StcNzFirst;
    }
    return function(inputs.x.get(), inputs.weight.get(), inputs.bias.get(), inputs.offset.get(), inputs.scale.get(),
                    inputs.xScale.get(), inputs.groups.get(), inputs.output.get(), inputs.outputScale.get(),
                    inputs.outputOffset.get(), ops::ut::ParseBool(c.nullWorkspace) ? nullptr : workspace,
                    ops::ut::ParseBool(c.nullExecutor) ? nullptr : executor);
}

void ExpectV2Reference(const RouteCase& c, aclnnStatus actual, uint64_t workspace, const aclOpExecutor* executor)
{
    // V2 can modify format metadata. The reference must have independent tensors.
    RouteInputs reference(c);
    ASSERT_NE(reference.weight.get(), nullptr);
    ASSERT_NE(reference.scale.get(), nullptr);
    const aclTensor* weights[] = {reference.weight.get()};
    const aclTensor* scales[] = {reference.scale.get()};
    auto listExecutor = RealRouteExecutor();
    ASSERT_NE(listExecutor.get(), nullptr);
    auto weightList = listExecutor->AllocTensorList(weights, 1);
    auto scaleList = listExecutor->AllocTensorList(scales, 1);
    ASSERT_NE(weightList, nullptr);
    ASSERT_NE(scaleList, nullptr);
    uint64_t refWorkspace = WORKSPACE_SENTINEL;
    aclOpExecutor* refExecutor = nullptr;
    const auto ret = aclnnGroupedMatmulSwigluQuantWeightNzV2GetWorkspaceSize(
        reference.x.get(), weightList, scaleList, nullptr, reference.bias.get(), reference.xScale.get(), nullptr,
        reference.groups.get(), 0, 0, 0, 0, nullptr, reference.output.get(), reference.outputScale.get(), &refWorkspace,
        &refExecutor);
    EXPECT_EQ(actual, ret);
    EXPECT_EQ(workspace, refWorkspace);
    EXPECT_EQ(executor == nullptr, refExecutor == nullptr);
    if (refExecutor != nullptr) {
        aclDestroyAclOpExecutor(refExecutor);
    }
}

void ExpectProbeCounts(const RouteCase& c)
{
    EXPECT_EQ(probe.firstCalls, std::stoi(c.expectFirstCalls));
    EXPECT_EQ(probe.createCalls, std::stoi(c.expectCreateCalls));
    EXPECT_EQ(probe.allocCalls, std::stoi(c.expectAllocCalls));
    EXPECT_EQ(probe.secondCalls, std::stoi(c.expectSecondCalls));
    EXPECT_EQ(probe.legacyRunCalls, std::stoi(c.expectLegacyCalls));
}

class GmmsqA5RouteCsvTest : public testing::TestWithParam<RouteCase> {
protected:
    void SetUp() override
    {
        const auto& soc = GetParam().socVersion;
        if (soc == "Ascend950") {
            op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
        } else if (soc == "Ascend910B") {
            op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        } else if (soc == "Ascend910_93") {
            op::SetPlatformSocVersion(op::SocVersion::ASCEND910_93);
        } else {
            FAIL() << "Unknown socVersion: " << soc;
        }
        probe = {};
    }
    void TearDown() override
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
    }
};

TEST_P(GmmsqA5RouteCsvTest, RunCase)
{
    const auto& c = GetParam();
    ASSERT_TRUE(c.api == "WeightNZ" || c.api == "Normal");
    RouteInputs inputs(c);
    probe.x = inputs.x.get();
    probe.weight = inputs.weight.get();
    probe.scale = inputs.scale.get();
    probe.bias = inputs.bias.get();
    probe.xScale = inputs.xScale.get();
    probe.groups = inputs.groups.get();
    probe.output = inputs.output.get();
    probe.outputScale = inputs.outputScale.get();
    probe.status = ParseStatus(c.mockRet);
    probe.failCreate = ops::ut::ParseBool(c.failCreate);
    probe.failAllocAt = std::stoi(c.failAllocAt);

    if (c.mode == "PREDICATE") {
        EXPECT_EQ(ShouldRouteToSwigluQuantV2(inputs.x.get(), inputs.weight.get()), ops::ut::ParseBool(c.expectRoute));
        ExpectProbeCounts(c);
        return;
    }
    if (c.mode == "MOCK_SECOND") {
        uint8_t buffer[8] = {};
        probe.workspace = buffer;
        probe.workspaceSize = sizeof(buffer);
        EXPECT_EQ(StcNzSecond(buffer, sizeof(buffer), nullptr, nullptr), ParseStatus(c.expectRet));
        ExpectProbeCounts(c);
        return;
    }
    ASSERT_TRUE(c.mode == "REAL" || c.mode == "MOCK_FIRST") << c.mode;
    const bool mocked = c.mode == "MOCK_FIRST";
    const auto* stdoutLogging = std::getenv("ASCEND_SLOG_PRINT_TO_STDOUT");
    const bool checkLogs =
        !mocked && c.expectedLog != "NONE" && stdoutLogging != nullptr && std::string(stdoutLogging) == "1";
    RecordProperty("errorLogCheck", checkLogs ? "enabled" : "not_enabled");
    RecordProperty("mode", c.mode);
    std::optional<TensorMetadata> reservedMeta;
    std::optional<TensorMetadata> weightMeta;
    if (inputs.outputOffset != nullptr) {
        reservedMeta.emplace(inputs.outputOffset.get());
    }
    if (mocked && inputs.weight != nullptr) {
        weightMeta.emplace(inputs.weight.get());
    }
    uint64_t workspace = WORKSPACE_SENTINEL;
    aclOpExecutor* executor = nullptr;
    if (checkLogs) {
        std::fflush(stdout);
        testing::internal::CaptureStdout();
    }
    const auto ret = CallFirst(c, inputs, &workspace, &executor);
    if (checkLogs) {
        std::fflush(stdout);
        const auto log = testing::internal::GetCapturedStdout();
        ExpectLogTokens(log, c.expectedLog);
        std::fwrite(log.data(), 1, log.size(), stdout);
    }
    if (mocked) {
        ExpectLogTokens(probe.info + "\n" + probe.error, c.expectedLog);
        ExpectProbeCounts(c);
        if (c.expectFirstCalls == "0") {
            // The normal entry and nonmatching paths must not print a route message.
            if (c.api == "Normal" || ops::ut::ParseBool(c.nullWorkspace) || ops::ut::ParseBool(c.nullExecutor)) {
                EXPECT_TRUE(probe.info.empty());
            }
        }
    }
    if (reservedMeta) {
        reservedMeta->ExpectUnchanged(inputs.outputOffset.get());
    }
    if (weightMeta) {
        weightMeta->ExpectUnchanged(inputs.weight.get());
    }
    if (ops::ut::ParseBool(c.compareV2)) {
        ExpectV2Reference(c, ret, workspace, executor);
    }
    if (c.expectRet == "MATCH_V2") {
        // Exploratory metadata is checked for equivalence, not declared supported.
        ASSERT_TRUE(ops::ut::ParseBool(c.compareV2));
        RecordProperty("observedFirstPhaseStatus", std::to_string(ret));
    } else {
        EXPECT_EQ(ret, ParseStatus(c.expectRet));
    }
    if (ret == ACLNN_SUCCESS) {
        EXPECT_NE(executor, nullptr);
    } else {
        EXPECT_EQ(workspace, WORKSPACE_SENTINEL);
        EXPECT_EQ(executor, nullptr);
    }
    if (executor != nullptr) {
        aclDestroyAclOpExecutor(executor);
    }
}

INSTANTIATE_TEST_SUITE_P(
    GmmsqA5RouteCsv, GmmsqA5RouteCsvTest,
    testing::ValuesIn(LoadCases(ops::ut::ResolveCsvPath("test_aclnn_grouped_matmul_swiglu_quant_a5_route.csv",
                                                        "gmm/grouped_matmul_swiglu_quant/tests/ut/op_api", __FILE__))),
    [](const testing::TestParamInfo<RouteCase>& info) { return ops::ut::MakeSafeParamName(info.param.caseName); });
} // namespace
