// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

namespace {
using Type = at::ScalarType;
constexpr int64_t E = 4, M = 17, K = 128, PAIR = 2, MX_GROUP = 64, FP4_ATTR = 296;
constexpr int64_t SMALL_N = 128, MEDIUM_N = 256, LARGE_N = 384, TOO_MANY_GROUPS = 1025;
constexpr int SMOOTH_ID = 123, GROUP_ID = 456;
constexpr double CLAMP = 3.0, ALPHA = 0.5, BIAS = -1.0;

struct RouteCase {
    at::Tensor x{{M, K}, Type::Float8_e4m3fn};
    std::vector<at::Tensor> weight;
    std::vector<at::Tensor> scales;
    at::Tensor xs{{M, K / MX_GROUP, PAIR}, Type::Byte};
    at::Tensor groups{{E}, Type::Long, GROUP_ID};
    c10::optional<at::Tensor> smooth;
    c10::optional<int64_t> weightDtype;
    bool oldFp8;

    RouteCase(int64_t n, bool nk, Type weightType, bool withSmooth)
        : oldFp8(weightType == Type::Float8_e4m3fn)
    {
        const int64_t row = nk ? K / PAIR : K;
        const int64_t column = nk ? n : n / PAIR;
        weight = {oldFp8 ? at::Tensor({E, K, n}, weightType) :
                           at::Tensor({E, column, row}, weightType).transpose(1, PAIR)};
        const int64_t scaleRow = nk ? n : K / MX_GROUP;
        const int64_t scaleColumn = nk ? K / MX_GROUP : n;
        scales = {at::Tensor({E, scaleColumn, scaleRow, PAIR}, Type::Byte).transpose(1, PAIR)};
        if (withSmooth) {
            smooth = at::Tensor({E, n / PAIR}, Type::Float, SMOOTH_ID);
        }
        if (weightType == Type::Byte || weightType == Type::Char) {
            weightDtype = FP4_ATTR;
        }
    }

    std::tuple<at::Tensor, at::Tensor> Call() const
    {
        return op_api::grouped_matmul_swiglu_quant(x, weight, scales, xs, groups, smooth, std::nullopt, std::nullopt,
                                                   PAIR, static_cast<int64_t>(Type::Float), PAIR, ACL_FLOAT8_E4M3FN, 0,
                                                   std::nullopt, std::nullopt, weightDtype, std::nullopt, std::nullopt,
                                                   PAIR, CLAMP, ALPHA, BIAS, "rint", 0, 0.0);
    }
};

void CheckMockMetadata()
{
    const at::Tensor groups({E}, Type::Long);
    require(groups.numel() == E && groups.stride(0) == 1);
    const at::Tensor source({E, SMALL_N / PAIR, K}, Type::Byte);
    const auto view = source.transpose(1, PAIR);
    require(view.size(-1) == SMALL_N / PAIR && view.size(-PAIR) == K);
    require(view.stride(-PAIR) == 1 && view.stride(-1) == K);
    require(view.numel() == source.numel());
    const auto dense = view.contiguous();
    require(dense.shape == view.shape && dense.numel() == view.numel());
    require(dense.stride(-1) == 1 && dense.stride(-PAIR) == SMALL_N / PAIR);
    require(at::Tensor({0, K}, Type::Byte).numel() == 0);
}

void CheckForwarding(const RouteCase& input, int64_t n)
{
    record = {};
    const auto output = input.Call();
    require(record.calls == 1 && record.groups == GROUP_ID);
    require(record.x_scale_materialized == !input.oldFp8);
    require(record.smooth == (input.smooth.has_value() ? SMOOTH_ID : 0));
    require(record.dequant_dtype == ACL_FLOAT && record.dequant_mode == PAIR && record.quant_mode == PAIR);
    require(record.clamp == CLAMP && record.alpha == ALPHA && record.bias == BIAS);
    require(record.weight_type == (input.oldFp8 ? ACL_FLOAT8_E4M3FN : ACL_FLOAT4_E2M1));
    require(record.weight_shape == input.weight[0].shape);
    require(std::get<0>(output).shape == std::vector<int64_t>{input.x.size(0), n / PAIR});
    require(std::get<1>(output).shape ==
            std::vector<int64_t>{input.x.size(0), (n / PAIR + MX_GROUP - 1) / MX_GROUP, PAIR});
    // Mainline allocates both FP8 and A8W4 scales without a separate zero task.
    require(!std::get<1>(output).zeroed);
}

template <class Mutation>
void CheckRejected(const RouteCase& original, Mutation mutate)
{
    RouteCase input = original;
    mutate(input);
    record = {};
    bool rejected = false;
    try {
        input.Call();
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected && record.calls == 0);
}

void CheckInvalidMetadata(const RouteCase& input)
{
    CheckRejected(input, [](RouteCase& value) { value.groups.type = Type::Byte; });
    CheckRejected(input, [](RouteCase& value) { value.groups = at::Tensor({E, 1}, Type::Long); });
    CheckRejected(input, [](RouteCase& value) { value.groups = at::Tensor({0}, Type::Long); });
    CheckRejected(input, [](RouteCase& value) { value.groups = at::Tensor({TOO_MANY_GROUPS}, Type::Long); });
    if (input.weightDtype.has_value()) {
        CheckRejected(input, [](RouteCase& value) { value.weight[0].type = Type::Float; });
    }
    if (!input.oldFp8) {
        CheckRejected(input, [](RouteCase& value) { value.weight[0] = value.weight[0].contiguous(); });
        CheckRejected(input, [](RouteCase& value) { value.scales[0] = value.scales[0].contiguous(); });
        if (input.smooth.has_value()) {
            CheckRejected(input, [](RouteCase& value) { value.smooth->type = Type::Byte; });
            CheckRejected(input, [](RouteCase& value) { value.smooth->dev = 1; });
            CheckRejected(input, [](RouteCase& value) { value.smooth->shape[1] += 1; });
        }
    }
}
} // namespace

int main()
{
    CheckMockMetadata();
    for (int64_t n : {SMALL_N, MEDIUM_N, LARGE_N}) {
        for (bool nk : {false, true}) {
            for (Type weightType : {Type::Float4_e2m1fn_x2, Type::Byte, Type::Char, Type::Float8_e4m3fn}) {
                for (bool withSmooth : {false, true}) {
                    if (weightType == Type::Float8_e4m3fn && nk) {
                        continue;
                    }
                    const RouteCase input(n, nk, weightType, withSmooth);
                    CheckForwarding(input, n);
                    CheckInvalidMetadata(input);
                }
            }
        }
    }
    RouteCase empty(SMALL_N, false, Type::Float8_e4m3fn, false);
    empty.x = at::Tensor({0, K}, Type::Float8_e4m3fn);
    empty.xs = at::Tensor({0, K / MX_GROUP, PAIR}, Type::Byte);
    CheckForwarding(empty, SMALL_N);
    CheckInvalidMetadata(empty);
}
