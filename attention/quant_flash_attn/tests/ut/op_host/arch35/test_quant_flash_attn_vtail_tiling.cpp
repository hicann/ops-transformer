/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstring>
#include <gtest/gtest.h>
#include "../quant_flash_attn_param.h"
#include "../../../../op_host/quant_flash_attn_tiling_common.h"
#include "../../../../op_kernel/arch35/quant_flash_attn_template_tiling_key.h"
#include "tiling_case_executor.h"

namespace QuantFlashAttnUT {
namespace {
using TD = gert::TilingContextPara::TensorDescription;
using Attr = gert::TilingContextPara::OpAttr;

// 与 arch35 FlashAttentionScore 的切分单测配置一致：UB 192 KiB，L2 112 MiB。
const std::string SOC_950 = R"({"hardware_info": {
    "BT_SIZE": 0, "load3d_constraints": "1",
    "Intrinsic_fix_pipe_l0c2out": false,
    "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true,
    "Intrinsic_data_move_out2l1_nd2nz": false,
    "UB_SIZE": 196608, "L2_SIZE": 117440512, "L1_SIZE": 524288,
    "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 65536,
    "vector_core_cnt": 64, "cube_core_cnt": 32, "socVersion": "Ascend950"
}})";

TD Tensor(const std::string& shape, ge::DataType dtype)
{
    return TD(GetStorageShape(shape), dtype, ge::FORMAT_ND);
}

struct Arch35VTailCase {
    optiling::quant_flash_attn::QuantFlashAttnCompileInfo compileInfo{};
    std::vector<int32_t> cuQ{0, 17, 34};
    std::vector<int32_t> seqQ{17, 17};
    std::vector<int32_t> seqKV{287, 128};
    std::vector<int32_t> tailLengths{31, 0};
    std::vector<TD> inputs;
    std::vector<TD> outputs;
    std::vector<Attr> attrs;
    bool decode;

    explicit Arch35VTailCase(bool isDecode, bool hasTail = true)
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
        return gert::TilingContextPara("QuantFlashAttn", inputs, outputs, attrs, &compileInfo, "Ascend950", 64, 196608,
                                       16384, SOC_950);
    }
};

TEST(QuantFlashAttnArch35VTailTiling, VtailTripletEnabledAndDisabledSerializeTailFlag)
{
    optiling::QuantFlashAttnTilingData parsed{};
    for (bool decode : {false, true}) {
        Arch35VTailCase withTail(decode);
        ::TilingInfo tailResult;
        ASSERT_TRUE(ExecuteTiling(withTail.Context(), tailResult));
        ASSERT_EQ(tailResult.tilingDataSize, sizeof(parsed));
        std::memcpy(&parsed, tailResult.tilingData.get(), sizeof(parsed));
        const auto& tail = parsed.baseTiling.quantFlashAttnVTailParams;
        EXPECT_EQ(tail.hasVTail, 1);
        EXPECT_EQ(tail.seqUsedKvTailSize, 2U);
        EXPECT_EQ(tail.tailMaxBlockNum, 3U);

        Arch35VTailCase ordinary(decode, false);
        ::TilingInfo ordinaryResult;
        ASSERT_TRUE(ExecuteTiling(ordinary.Context(), ordinaryResult));
        ASSERT_EQ(ordinaryResult.tilingDataSize, sizeof(parsed));
        std::memcpy(&parsed, ordinaryResult.tilingData.get(), sizeof(parsed));
        const auto& empty = parsed.baseTiling.quantFlashAttnVTailParams;
        EXPECT_EQ(empty.hasVTail, 0);
        EXPECT_EQ(empty.seqUsedKvTailSize, 0U);
        EXPECT_EQ(empty.tailMaxBlockNum, 0U);
    }
}
} // namespace
} // namespace QuantFlashAttnUT
