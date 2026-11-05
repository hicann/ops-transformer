/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include <array>
#include <cstring>
#include <gtest/gtest.h>
#include "../quant_flash_attn_param.h"
#include "../../../../op_host/quant_flash_attn_tiling_common.h"
#include "../../../../op_kernel/arch92/quant_flash_attn_template_tiling_key_arch92.h"
#include "tiling_case_executor.h"

namespace QuantFlashAttnUT {
namespace {
using TD = gert::TilingContextPara::TensorDescription;
using Attr = gert::TilingContextPara::OpAttr;

constexpr uint64_t VTAIL_KEY_BIT = 1ULL << 27;
// FD 能力位位于原六字段编码的第 26 位；尾窗仍使用第 27 位，旧编码保持可解码。
constexpr uint64_t FD_KEY_BIT = 1ULL << 26;
// 固定保存原六字段编码的数值，避免依赖当前声明和宏来生成期望值。
constexpr uint64_t LEGACY_DN_KEY = 33817090ULL;
constexpr uint64_t LEGACY_ND_KEY = 34080258ULL;
constexpr uint64_t LEGACY_HIFP4_KEY = 2623489ULL;

TD Tensor(const std::string& shape, ge::DataType dtype)
{
    return TD(GetStorageShape(shape), dtype, ge::FORMAT_ND);
}

// 与 arch92 FlashAttentionScoreGrad 的切分单测配置一致：UB 为 384 KiB，L2 为 128 MiB。
const std::string SOC_9201 = R"({"hardware_info": {
    "BT_SIZE": 0, "load3d_constraints": "1", "Intrinsic_data_move_l12ub": true,
    "Intrinsic_data_move_l0c2ub": true, "UB_SIZE": 393216, "L2_SIZE": 134217728,
    "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144,
    "vector_core_cnt": 64, "cube_core_cnt": 64, "socVersion": "Ascend960DT"
}})";

struct VTailCase {
    optiling::quant_flash_attn::QuantFlashAttnCompileInfo compileInfo{};
    std::vector<int32_t> cuQ{0, 17, 34};
    std::vector<int32_t> seqQ{17, 17};
    std::vector<int32_t> seqKV{287, 128};
    std::vector<int32_t> tailLengths{31, 0};
    std::vector<TD> inputs;
    std::vector<TD> outputs;
    std::vector<Attr> attrs;
    bool decode;

    explicit VTailCase(bool isDecode, bool hasTail = true)
        : decode(isDecode)
    {
        inputs = {
            Tensor("34 4 128", ge::DT_FLOAT8_E4M3FN),    // q
            Tensor("8 2 128 128", ge::DT_FLOAT8_E4M3FN), // k
            Tensor("8 2 128 128", ge::DT_FLOAT8_E4M3FN), // v
            Tensor(isDecode ? "2 34 2 2 2" : "34 4 2 2", ge::DT_FLOAT8_E8M0),
            Tensor("8 2 128 2 2", ge::DT_FLOAT8_E8M0),
            Tensor("8 2 2 128 2", ge::DT_FLOAT8_E8M0),
            Tensor("2 3", ge::DT_INT32), // 主缓存块表
            Tensor("1", ge::DT_FLOAT),   // p_scale
            Tensor("3", ge::DT_INT32),   // cu_seqlens_q
            Tensor("", ge::DT_INT32),    // cu_seqlens_kv (PA)
            Tensor("2", ge::DT_INT32),   // seqused_q
            Tensor("2", ge::DT_INT32),   // seqused_kv
            Tensor("", ge::DT_FLOAT),    // sinks
            Tensor("", ge::DT_UINT8),    // mask
            Tensor("1", ge::DT_INT32),   // metadata
            Tensor(hasTail ? "2 2 128 128" : "", ge::DT_BF16),
            Tensor(hasTail ? "2 3" : "", ge::DT_INT32),
            Tensor(hasTail ? "2" : "", ge::DT_INT32),
        };
        inputs[8].isConst_ = true;
        inputs[8].constValue_ = cuQ.data();
        inputs[10].isConst_ = true;
        inputs[10].constValue_ = seqQ.data();
        inputs[11].isConst_ = true;
        inputs[11].constValue_ = seqKV.data();
        inputs[17].isConst_ = hasTail;
        inputs[17].constValue_ = hasTail ? tailLengths.data() : nullptr;
        outputs = {Tensor("34 4 128", ge::DT_BF16), Tensor("", ge::DT_FLOAT)};
        attrs = {
            {"quant_compute_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
            {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(17)},
            {"max_seqlen_kv", Ops::Transformer::AnyValue::CreateFrom<int64_t>(287)},
            {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"layout_q_descale", Ops::Transformer::AnyValue::CreateFrom<std::string>(isDecode ? "N2TGD" : "TND")},
            {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BNBD")},
            {"layout_out", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        };
    }

    gert::TilingContextPara Context()
    {
        return gert::TilingContextPara("QuantFlashAttn", inputs, outputs, attrs, &compileInfo, "9201", 64, 393216,
                                       16384, SOC_9201);
    }

    uint64_t ExpectedKey(bool hasTail) const
    {
        const auto config = decode ? Config_S1Aligned64_S2Aligned512_DAligned128_DVAligned128 :
                                     Config_S1Aligned128_S2Aligned256_DAligned128_DVAligned128;
        const auto mode = decode ? QFA_MXFP8_FP32_DECODE : QFA_MXFP8_FP32_PREFILL;
        return GET_TPL_TILING_KEY(InOutLayoutType_TND_TND, config, mode, false, KvLayoutType_PA_BNBD, true, hasTail);
    }
};

TEST(QuantFlashAttnArch92VTailTiling, TripletAndNoTailSerializeWithDistinctKeys)
{
    for (bool decode : {false, true}) {
        VTailCase withTail(decode);
        ::TilingInfo tailResult;
        ASSERT_TRUE(ExecuteTiling(withTail.Context(), tailResult));
        EXPECT_EQ(static_cast<uint64_t>(tailResult.tilingKey), withTail.ExpectedKey(true));
        ASSERT_EQ(tailResult.tilingDataSize, sizeof(optiling::QuantFlashAttnTilingData));
        optiling::QuantFlashAttnTilingData parsed{};
        std::memcpy(&parsed, tailResult.tilingData.get(), sizeof(parsed));
        const auto& tail = parsed.baseTiling.quantFlashAttnVTailParams;
        EXPECT_EQ(tail.hasVTail, 1);
        EXPECT_EQ(tail.seqUsedKvTailSize, 2U);
        EXPECT_EQ(tail.tailMaxBlockNum, 3U);
        EXPECT_EQ(tail.vTailStrides.bnStride, 0U);
        EXPECT_EQ(tail.vTailStrides.n2Stride, 0U);
        VTailCase ordinary(decode, false);
        ::TilingInfo ordinaryResult;
        ASSERT_TRUE(ExecuteTiling(ordinary.Context(), ordinaryResult));
        EXPECT_EQ(static_cast<uint64_t>(ordinaryResult.tilingKey), ordinary.ExpectedKey(false));
        EXPECT_EQ(static_cast<uint64_t>(ordinaryResult.tilingKey),
                  (decode ? LEGACY_ND_KEY : LEGACY_DN_KEY) | FD_KEY_BIT);
        EXPECT_EQ(static_cast<uint64_t>(tailResult.tilingKey),
                  (decode ? LEGACY_ND_KEY : LEGACY_DN_KEY) | FD_KEY_BIT | VTAIL_KEY_BIT);
        EXPECT_NE(tailResult.tilingKey, ordinaryResult.tilingKey);
        ASSERT_EQ(ordinaryResult.tilingDataSize, sizeof(parsed));
        std::memcpy(&parsed, ordinaryResult.tilingData.get(), sizeof(parsed));
        const auto& empty = parsed.baseTiling.quantFlashAttnVTailParams;
        EXPECT_EQ(empty.hasVTail, 0);
        EXPECT_EQ(empty.seqUsedKvTailSize, 0U);
        EXPECT_EQ(empty.tailMaxBlockNum, 0U);
        EXPECT_EQ(empty.vTailStrides.bnStride, 0U);
        EXPECT_EQ(empty.vTailStrides.n2Stride, 0U);
    }
}

TEST(QuantFlashAttnArch92FdLayoutTiling, FixedLayoutsSelectRouteAndReservePartials)
{
    // 两个批次、四个 Q 头、两个 KV 头；物理序列长度 17，沿用现有 PA 缓存用例。
    for (bool bnsd : {false, true}) {
        for (bool decode : {false, true}) {
            if (bnsd && decode) {
                continue; // BNSD 的 Q 系数没有 ND 合轴格式，必须保持 DN。
            }
            VTailCase params(decode, false);
            const std::string layout = bnsd ? "BNSD" : "BSND";
            params.inputs[0] = Tensor(bnsd ? "2 4 17 128" : "2 17 4 128", ge::DT_FLOAT8_E4M3FN);
            params.inputs[3] = Tensor(bnsd ? "2 4 17 2 2" : decode ? "2 2 17 2 2 2" : "2 17 4 2 2", ge::DT_FLOAT8_E8M0);
            params.inputs[8] = Tensor("", ge::DT_INT32);
            params.outputs[0] = Tensor(bnsd ? "2 4 17 128" : "2 17 4 128", ge::DT_BF16);
            params.attrs[7] = {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>(layout)};
            params.attrs[8] = {"layout_q_descale",
                               Ops::Transformer::AnyValue::CreateFrom<std::string>(decode ? "BN2SGD" : layout)};
            params.attrs[10] = {"layout_out", Ops::Transformer::AnyValue::CreateFrom<std::string>(layout)};
            ::TilingInfo result;
            ASSERT_TRUE(ExecuteTiling(params.Context(), result));
            const auto config = decode ? Config_S1Aligned64_S2Aligned512_DAligned128_DVAligned128 :
                                         Config_S1Aligned128_S2Aligned256_DAligned128_DVAligned128;
            const auto mode = decode ? QFA_MXFP8_FP32_DECODE : QFA_MXFP8_FP32_PREFILL;
            const auto layoutKey = bnsd ? InOutLayoutType_BNSD : InOutLayoutType_BSND;
            EXPECT_EQ(static_cast<uint64_t>(result.tilingKey),
                      GET_TPL_TILING_KEY(layoutKey, config, mode, false, KvLayoutType_PA_BNBD, true, false));
            optiling::QuantFlashAttnTilingData parsed{};
            ASSERT_EQ(result.tilingDataSize, sizeof(parsed));
            std::memcpy(&parsed, result.tilingData.get(), sizeof(parsed));
            const auto& workspace = parsed.baseTiling.quantFlashAttnWorkspaceParams;
            const uint32_t rows = decode ? 64U : 128U; // ND 和 DN 模板的 M 基本块。
            // 64 个 AIC，每核最多头/尾两个 partial；每个 sum/max 行保存 8 个重复 float。
            EXPECT_EQ(workspace.accumOutSize, 64U * 2U * rows * 128U);
            EXPECT_EQ(workspace.logSumExpSize, 64U * 2U * rows * 8U);
        }
    }
}

TEST(QuantFlashAttnArch92VTailTiling, ZeroTailTensorRetainsSpecialization)
{
    for (bool decode : {false, true}) {
        VTailCase params(decode);
        params.tailLengths = {0, 0};
        params.seqKV[0] = 256;
        params.inputs[17].constValue_ = params.tailLengths.data();
        ::TilingInfo result;
        ASSERT_TRUE(ExecuteTiling(params.Context(), result));
        EXPECT_EQ(static_cast<uint64_t>(result.tilingKey), params.ExpectedKey(true));
        optiling::QuantFlashAttnTilingData parsed{};
        ASSERT_EQ(result.tilingDataSize, sizeof(parsed));
        std::memcpy(&parsed, result.tilingData.get(), sizeof(parsed));
        EXPECT_EQ(parsed.baseTiling.quantFlashAttnVTailParams.hasVTail, 1);
        EXPECT_EQ(parsed.baseTiling.quantFlashAttnVTailParams.seqUsedKvTailSize, 2U);
    }
}

TEST(QuantFlashAttnArch92VTailTiling, FdAndTailBitsPreserveExistingKeyEncoding)
{
    // MXFP8 仅注册 IsFd=true；用受支持的 key 验证原编码位，避免调试版检查已删除的选择。
    EXPECT_EQ(GET_TPL_TILING_KEY(InOutLayoutType_TND_TND, Config_S1Aligned128_S2Aligned256_DAligned128_DVAligned128,
                                 QFA_MXFP8_FP32_PREFILL, false, KvLayoutType_PA_BNBD, true, false),
              LEGACY_DN_KEY | FD_KEY_BIT);
    EXPECT_EQ(GET_TPL_TILING_KEY(InOutLayoutType_TND_TND, Config_S1Aligned128_S2Aligned256_DAligned128_DVAligned128,
                                 QFA_MXFP8_FP32_PREFILL, false, KvLayoutType_PA_BNBD, true, true),
              LEGACY_DN_KEY | FD_KEY_BIT | VTAIL_KEY_BIT);
    EXPECT_EQ(GET_TPL_TILING_KEY(InOutLayoutType_TND_TND, Config_S1Aligned64_S2Aligned512_DAligned128_DVAligned128,
                                 QFA_MXFP8_FP32_DECODE, false, KvLayoutType_PA_BNBD, true, false),
              LEGACY_ND_KEY | FD_KEY_BIT);
    EXPECT_EQ(GET_TPL_TILING_KEY(InOutLayoutType_TND_TND, Config_S1Aligned64_S2Aligned512_DAligned128_DVAligned128,
                                 QFA_MXFP8_FP32_DECODE, false, KvLayoutType_PA_BNBD, true, true),
              LEGACY_ND_KEY | FD_KEY_BIT | VTAIL_KEY_BIT);
    EXPECT_EQ(GET_TPL_TILING_KEY(InOutLayoutType_BNSD_BNSD, Config_S1Aligned128_S2Aligned1024_DAligned128_DVAligned128,
                                 QFA_HIFP4_FP16_PREFILL, false, KvLayoutType_NO_PA, false, false),
              LEGACY_HIFP4_KEY);
}

TEST(QuantFlashAttnArch92VTailTiling, HiFP4HostKeepsTailDisabledAndLegacyKey)
{
    VTailCase params(false, false);
    params.inputs[0] = Tensor("2 4 17 128", Ops::Transformer::DT_HIFLOAT4);
    params.inputs[1] = Tensor("2 2 128 128", Ops::Transformer::DT_HIFLOAT4);
    params.inputs[2] = Tensor("2 2 128 128", Ops::Transformer::DT_HIFLOAT4);
    params.inputs[3] = Tensor("2 4 17 2", Ops::Transformer::DT_HIFLOAT4_SCALE);
    params.inputs[4] = Tensor("2 2 128 2", Ops::Transformer::DT_HIFLOAT4_SCALE);
    params.inputs[5] = Tensor("2 2 128 2", Ops::Transformer::DT_HIFLOAT4_SCALE);
    params.inputs[6] = Tensor("", ge::DT_INT32);
    params.inputs[8] = Tensor("", ge::DT_INT32);
    params.seqKV[0] = 128;
    params.outputs[0] = Tensor("2 4 17 128", ge::DT_BF16);
    params.attrs[0] = {"quant_compute_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(10)};
    params.attrs[6] = {"max_seqlen_kv", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)};
    for (size_t index : {7U, 8U, 9U, 10U}) {
        const std::string name = index == 7 ? "layout_q" :
                                 index == 8 ? "layout_q_descale" :
                                 index == 9 ? "layout_kv" :
                                              "layout_out";
        params.attrs[index] = {name, Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")};
    }
    ::TilingInfo result;
    ASSERT_TRUE(ExecuteTiling(params.Context(), result));
    EXPECT_EQ(static_cast<uint64_t>(result.tilingKey), LEGACY_HIFP4_KEY);
    EXPECT_EQ(static_cast<uint64_t>(result.tilingKey) & VTAIL_KEY_BIT, 0U);
}

std::vector<uint64_t> ParameterValues(const ParamStruct& param)
{
#ifdef ASCENDC_DEBUG
    // 调试版 SDK 元数据保留 UINT 的 UI_LIST/UI_RANGE 前缀，直到 EncodeTilingKey 解析其副本。
    if (param.paramType == ASCENDC_TPL_UINT) {
        if (param.vals.front() == ASCENDC_TPL_UI_LIST) {
            return {param.vals.begin() + 1, param.vals.end()};
        }
        std::vector<uint64_t> values;
        const size_t ranges = param.vals[1];
        for (size_t i = 0; i < ranges; ++i) {
            for (uint64_t value = param.vals[2 + i * 2]; value <= param.vals[3 + i * 2]; ++value) {
                values.push_back(value);
            }
        }
        if (param.vals.front() == ASCENDC_TPL_UI_MIX) {
            values.insert(values.end(), param.vals.begin() + 2 + ranges * 2, param.vals.end());
        }
        return values;
    }
#endif
    return param.vals;
}

TEST(QuantFlashAttnArch92VTailTiling, TemplateDeclarationHasSevenOrderedFields)
{
    const std::array<const char*, 7> names{"InOutLayoutType", "Config", "QuantMode", "HasAttenMask",
                                           "KvLayoutType",    "IsFd",   "HasVTail"};
    const std::array<uint8_t, 7> widths{8, 10, 5, 1, 2, 1, 1};
    const std::array<uint32_t, 7> types{ASCENDC_TPL_UINT, ASCENDC_TPL_UINT, ASCENDC_TPL_UINT, ASCENDC_TPL_BOOL,
                                        ASCENDC_TPL_UINT, ASCENDC_TPL_BOOL, ASCENDC_TPL_BOOL};
    ASSERT_EQ(g_tilingDeclareParams.size(), names.size());
    for (size_t i = 0; i < names.size(); ++i) {
        EXPECT_STREQ(g_tilingDeclareParams[i].name, names[i]);
        EXPECT_EQ(g_tilingDeclareParams[i].bitWidth, widths[i]);
        EXPECT_EQ(g_tilingDeclareParams[i].paramType, types[i]);
        const auto values = ParameterValues(g_tilingDeclareParams[i]);
        ASSERT_EQ(values.size(), 1ULL << widths[i]);
        for (size_t value = 0; value < values.size(); ++value) {
            EXPECT_EQ(values[value], value);
        }
    }
    EXPECT_EQ(g_tilingDeclareParams.back().paramType, ASCENDC_TPL_BOOL);
    EXPECT_EQ(g_tilingDeclareParams.back().vals, (std::vector<uint64_t>{0, 1}));
}

#ifdef ASCENDC_DEBUG
bool SelectionSupports(const std::array<uint64_t, 7>& args)
{
    return std::any_of(g_tilingSelectParams.begin(), g_tilingSelectParams.end(), [&](const auto& selection) {
        if (selection.size() != args.size()) {
            return false;
        }
        for (size_t i = 0; i < args.size(); ++i) {
            const auto values = ParameterValues(selection[i]);
            if (std::find(values.begin(), values.end(), args[i]) == values.end()) {
                return false;
            }
        }
        return true;
    });
}
#endif

TEST(QuantFlashAttnArch92VTailTiling, TemplateSelectionsKeepTailAndHiFP4Boundaries)
{
#ifdef ASCENDC_DEBUG
    ASSERT_EQ(g_tilingSelectParams.size(), 9U); // 八组 MXFP8 和一组 HiFP4。
    for (const auto& selection : g_tilingSelectParams) {
        ASSERT_EQ(selection.size(), 7U);
        for (size_t i = 0; i < selection.size(); ++i) {
            EXPECT_STREQ(selection[i].name, g_tilingDeclareParams[i].name);
        }
        const auto modes = ParameterValues(selection[2]);
        const bool mxfp8 = std::find(modes.begin(), modes.end(), QFA_MXFP8_FP32_PREFILL) != modes.end() ||
                           std::find(modes.begin(), modes.end(), QFA_MXFP8_FP32_DECODE) != modes.end();
        EXPECT_EQ(ParameterValues(selection[5]), (std::vector<uint64_t>{mxfp8 ? 1U : 0U}));
    }
    // 覆盖全部 Q 布局、DN/ND 配置及 FD 值，统计实际可选组合并排除不合法边界。
    size_t supportedMxfp8 = 0;
    for (uint64_t qLayout = 0; qLayout < 3; ++qLayout) {
        for (uint64_t config = 0; config < 8; ++config) {
            for (uint64_t mode : {1U, 2U}) {
                for (uint64_t mask : {0U, 1U}) {
                    for (uint64_t kvLayout = 0; kvLayout < 4; ++kvLayout) {
                        const bool d72Nz = (config == 1 || config == 5) && kvLayout == KvLayoutType_PA_NZ;
                        const bool matchingPath = (config < 4) == (mode == QFA_MXFP8_FP32_PREFILL);
                        const bool bnsdNd = qLayout == InOutLayoutType_BNSD && mode == QFA_MXFP8_FP32_DECODE;
                        for (uint64_t tail : {0U, 1U}) {
                            const bool expected =
                                matchingPath && !bnsdNd && !d72Nz && (tail == 0 || kvLayout != KvLayoutType_NO_PA);
                            const bool actual = SelectionSupports({qLayout, config, mode, mask, kvLayout, 1, tail});
                            EXPECT_EQ(actual, expected);
                            EXPECT_FALSE(SelectionSupports({qLayout, config, mode, mask, kvLayout, 0, tail}));
                            supportedMxfp8 += actual ? 1U : 0U;
                        }
                    }
                }
            }
        }
    }
    EXPECT_EQ(supportedMxfp8, 260U);
    EXPECT_TRUE(SelectionSupports({1, 8, 10, 0, 0, 0, 0}));
    EXPECT_FALSE(SelectionSupports({1, 8, 10, 0, 0, 1, 0}));
    EXPECT_FALSE(SelectionSupports({1, 8, 10, 0, 0, 0, 1}));
    EXPECT_FALSE(SelectionSupports({1, 8, 10, 1, 0, 0, 0}));
    EXPECT_FALSE(SelectionSupports({1, 8, 10, 0, 1, 0, 0}));
#else
    GTEST_SKIP() << "Selection metadata requires the SDK ASCENDC_DEBUG build; Host/key tests remain enabled.";
#endif
}

TEST(QuantFlashAttnArch92VTailTiling, DISABLED_IndependentTailViewStridesSurviveSerialization)
{
    for (bool decode : {false, true}) {
        VTailCase params(decode);
        params.inputs[15].hasStride_ = true;
        params.inputs[15].stride_ = MakeContiguousStride(params.inputs[15].shape_);
        params.inputs[15].stride_.SetStride(0, 65536);
        ::TilingInfo result;
        ASSERT_TRUE(ExecuteTiling(params.Context(), result));
        optiling::QuantFlashAttnTilingData parsed{};
        ASSERT_EQ(result.tilingDataSize, sizeof(parsed));
        std::memcpy(&parsed, result.tilingData.get(), sizeof(parsed));
        EXPECT_EQ(parsed.baseTiling.quantFlashAttnVTailParams.vTailStrides.bnStride, 65536U);
        EXPECT_EQ(parsed.baseTiling.quantFlashAttnVTailParams.vTailStrides.n2Stride, 16384U);
        EXPECT_EQ(parsed.baseTiling.quantFlashAttnBaseParams.valueStrides.bnStride, 0U);
    }
}

TEST(QuantFlashAttnArch92VTailTiling, MissingAnyTripletInputIsRejected)
{
    for (size_t missing : {15U, 16U, 17U}) {
        VTailCase params(false);
        params.inputs[missing] = Tensor("", ge::DT_INT32);
        ::TilingInfo result;
        EXPECT_FALSE(ExecuteTiling(params.Context(), result)) << missing;
    }
}

TEST(QuantFlashAttnArch92VTailTiling, DISABLED_InvalidTailDtypeShapeAndStridesAreRejected)
{
    for (size_t index : {15U, 16U, 17U}) {
        VTailCase params(false);
        params.inputs[index].dtype_ = ge::DT_FLOAT;
        ::TilingInfo result;
        EXPECT_FALSE(ExecuteTiling(params.Context(), result)) << index;
    }
    for (size_t index : {16U, 17U}) {
        VTailCase params(false);
        params.inputs[index] = Tensor(index == 16U ? "1 3" : "1", ge::DT_INT32);
        ::TilingInfo result;
        EXPECT_FALSE(ExecuteTiling(params.Context(), result)) << index;
    }
    for (const auto& shape : {"2 2 64 128", "2 2 128 64"}) {
        VTailCase params(false);
        params.inputs[15] = Tensor(shape, ge::DT_BF16);
        ::TilingInfo result;
        EXPECT_FALSE(ExecuteTiling(params.Context(), result)) << shape;
    }
    for (int64_t badStride : {0, -1}) {
        VTailCase params(false);
        params.inputs[15].hasStride_ = true;
        params.inputs[15].stride_ = MakeContiguousStride(params.inputs[15].shape_);
        params.inputs[15].stride_.SetStride(0, badStride);
        ::TilingInfo result;
        EXPECT_FALSE(ExecuteTiling(params.Context(), result)) << badStride;
    }
    VTailCase lastAxisView(false);
    lastAxisView.inputs[15].hasStride_ = true;
    lastAxisView.inputs[15].stride_ = MakeContiguousStride(lastAxisView.inputs[15].shape_);
    lastAxisView.inputs[15].stride_.SetStride(3, 2);
    ::TilingInfo result;
    EXPECT_FALSE(ExecuteTiling(lastAxisView.Context(), result));
}
} // namespace
} // namespace QuantFlashAttnUT
