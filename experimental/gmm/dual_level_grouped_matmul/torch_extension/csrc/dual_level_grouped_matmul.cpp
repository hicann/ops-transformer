// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#include <torch/extension.h>
#include <array>
#include <limits>
#include "aclnn_common.h"

namespace op_api {
namespace {
constexpr size_t SCALE_VIEW_LAST_MATRIX_AXIS = 2;
constexpr int64_t FP4_ELEMENTS_PER_BYTE = 2;

aclTensorList* MakeSingleTensorList(aclTensor* descriptor)
{
    static const auto createList = GET_OP_API_FUNC(aclCreateTensorList);
    TORCH_CHECK(descriptor != nullptr, "Cannot create ACL tensor descriptor.");
    const aclTensor* descriptors[] = {descriptor};
    auto* list = createList == nullptr ? nullptr : createList(descriptors, 1);
    if (list == nullptr) {
        Release(descriptor);
    }
    TORCH_CHECK(list != nullptr, "Cannot create ACL TensorList descriptor.");
    return list;
}
} // namespace
// Keep ND descriptor shapes explicit: the scalar ACLNN kernel validates storage dimensions.
// Reuse the common conversion for the NPU's private NZ descriptor.
struct DenseList {
    at::TensorList tensors;
    aclDataType dtype;
    bool transposedScale{false};
};

aclTensorList* ConvertType(const DenseList& arg)
{
    if (arg.tensors.empty()) {
        return nullptr;
    }
    static const auto createTensor = GET_OP_API_FUNC(aclCreateTensor);
    TORCH_CHECK(createTensor != nullptr, "ACL tensor descriptor API is unavailable.");
    const auto& tensor = arg.tensors[0];
    c10::SmallVector<int64_t, MAX_DIM_NUM> shape = op_infer::array_to_small_vector(tensor.sizes());
    c10::SmallVector<int64_t, MAX_DIM_NUM> strides = op_infer::array_to_small_vector(tensor.strides());
    if (Is4BitDtype(arg.dtype)) {
        if (tensor.numel() == 0) {
            shape.back() *= FP4_ELEMENTS_PER_BYTE;
        } else {
            CollectB4ShapeInfo(tensor, strides, shape);
        }
    }
    std::vector<int64_t> storage(shape.begin(), shape.end());
    if (arg.transposedScale) {
        std::swap(storage[1], storage[SCALE_VIEW_LAST_MATRIX_AXIS]);
    }
    auto* descriptor = createTensor(shape.data(), shape.size(), arg.dtype, strides.data(), 0, ACL_FORMAT_ND,
                                    storage.data(), storage.size(), const_cast<void*>(tensor.storage().data()));
    return MakeSingleTensorList(descriptor);
}

struct WeightList {
    at::TensorList tensors;
    int64_t k;
    int64_t n;
};

aclTensorList* ConvertType(const WeightList& arg)
{
    const auto& weight = arg.tensors[0];
    if (weight.numel() != 0) {
        return ConvertType(TensorListWrapper{arg.tensors, ACL_FLOAT4_E2M1});
    }
    // Empty packed views can have identical matrix strides. Match the logical dimensions instead.
    const bool packedNz = weight.size(1) == arg.k && weight.size(-1) == arg.n / FP4_ELEMENTS_PER_BYTE &&
                          arg.n % FP4_ELEMENTS_PER_BYTE == 0;
    const bool transposed = weight.size(1) == arg.k / FP4_ELEMENTS_PER_BYTE && weight.size(-1) == arg.n;
    TORCH_CHECK(packedNz || transposed, "Empty weight shape must match the packed K and N dimensions.");
    const std::array<int64_t, 3> shape{weight.size(0), arg.k, arg.n};
    const std::array<int64_t, 3> strides{0, transposed ? 1 : arg.n, transposed ? arg.k : 1};
    auto storage = at_npu::native::get_npu_storage_sizes(weight);
    TORCH_CHECK(!storage.empty(), "Missing NZ storage shape for empty weight.");
    storage.back() *= FP4_ELEMENTS_PER_BYTE;
    static const auto createTensor = GET_OP_API_FUNC(aclCreateTensor);
    TORCH_CHECK(createTensor != nullptr, "ACL tensor descriptor API is unavailable.");
    return MakeSingleTensorList(createTensor(shape.data(), shape.size(), ACL_FLOAT4_E2M1, strides.data(), 0,
                                             ACL_FORMAT_FRACTAL_NZ, storage.data(), storage.size(),
                                             const_cast<void*>(weight.storage().data())));
}

namespace {
constexpr int64_t FP4_E2M1_DTYPE_HINT = 296;
constexpr int64_t E8M0_DTYPE_HINT = 293;
constexpr int64_t SCALE_N_AXIS = 2;
constexpr int64_t PACKED_ROW_AXIS = -2;
constexpr int64_t X_RANK = 2;
constexpr int64_t WEIGHT_RANK = 3;
constexpr int64_t WEIGHT_SCALE_RANK = 4;

struct DescriptorInput {
    const char* name;
    const std::vector<at::Tensor>* tensors;
    aclDataType descriptorDtype;
};

void CheckDtypeHints(const c10::optional<int64_t>& x, const c10::optional<int64_t>& weight,
                     const c10::optional<int64_t>& scale, const c10::optional<int64_t>& perTokenScale)
{
    // TorchNPU semantic codes differ from aclDataType values.
    TORCH_CHECK(!x.has_value() || *x == FP4_E2M1_DTYPE_HINT, "x_dtype must be FP4 E2M1 (296).");
    TORCH_CHECK(!weight.has_value() || *weight == FP4_E2M1_DTYPE_HINT, "weight_dtype must be FP4 E2M1 (296).");
    TORCH_CHECK(!scale.has_value() || *scale == E8M0_DTYPE_HINT, "scale_dtype must be E8M0 (293).");
    TORCH_CHECK(!perTokenScale.has_value() || *perTokenScale == E8M0_DTYPE_HINT,
                "per_token_scale_dtype must be E8M0 (293).");
}

const at::Tensor& RequireSingleTensor(const DescriptorInput& input)
{
    TORCH_CHECK(input.tensors->size() == 1 && (*input.tensors)[0].defined(), input.name,
                " must contain one defined tensor in dual-level GMM.");
    return (*input.tensors)[0];
}

void CheckDescriptorInput(const DescriptorInput& input, const at::Tensor& tensor, c10::Device device,
                          bool weight = false)
{
    TORCH_CHECK(tensor.device() == device, input.name, " must be on ", device, ".");
    const auto format = at_npu::native::get_npu_format(tensor);
    if (weight) {
        TORCH_CHECK(format == ACL_FORMAT_FRACTAL_NZ, input.name, " must use FRACTAL_NZ format.");
    } else {
        TORCH_CHECK(format == ACL_FORMAT_ND, input.name, " has format=", format, "; expected ND in dual-level GMM.");
    }
    // Descriptor conversion uses the storage base and changes FP4 offsets to element units.
    TORCH_CHECK(tensor.storage_offset() == 0, input.name, " must have zero storage offset.");
    const auto dtype = tensor.scalar_type();
    const bool isFp4 = Is4BitDtype(input.descriptorDtype);
    if (isFp4 || input.descriptorDtype == ACL_FLOAT8_E8M0) {
        TORCH_CHECK(dtype == at::kChar || dtype == at::kByte, input.name, " requires int8 or uint8 storage.");
    } else {
        TORCH_CHECK(dtype == at::kFloat, input.name, " requires float32 storage.");
    }
    if (isFp4) {
        // Packed metadata is doubled when converted to logical FP4 elements.
        constexpr auto maxPacked = std::numeric_limits<int64_t>::max() / FP4_ELEMENTS_PER_BYTE;
        for (int64_t axis = 0; axis < tensor.dim(); ++axis) {
            TORCH_CHECK(tensor.size(axis) <= maxPacked && tensor.stride(axis) <= maxPacked, input.name,
                        " has unrepresentable packed FP4 metadata.");
        }
    }
}
} // namespace

std::vector<at::Tensor> dual_level_grouped_matmul(
    const std::vector<at::Tensor>& x, const std::vector<at::Tensor>& weight,
    const std::vector<at::Tensor>& x_level0_scale, const std::vector<at::Tensor>& x_level1_scale,
    const std::vector<at::Tensor>& weight_level0_scale, const std::vector<at::Tensor>& weight_level1_scale,
    int64_t level0_group_size, int64_t level1_group_size, const at::Tensor& group_list,
    const c10::optional<std::vector<at::Tensor>>& bias, int64_t split_item, int64_t group_type, int64_t group_list_type,
    const c10::optional<std::vector<int64_t>>& tuning_config, at::ScalarType output_dtype,
    const c10::optional<int64_t>& x_dtype, const c10::optional<int64_t>& weight_dtype,
    const c10::optional<int64_t>& scale_dtype, const c10::optional<int64_t>& per_token_scale_dtype)
{
    CheckDtypeHints(x_dtype, weight_dtype, scale_dtype, per_token_scale_dtype);
    const DescriptorInput xInput{"x", &x, ACL_FLOAT4_E2M1};
    const DescriptorInput weightInput{"weight", &weight, ACL_FLOAT4_E2M1};
    const auto& xTensor = RequireSingleTensor(xInput);
    const auto& weightTensor = RequireSingleTensor(weightInput);
    TORCH_CHECK(torch_npu::utils::is_npu(xTensor), "x[0] must be on NPU.");
    const auto device = xTensor.device();
    CheckDescriptorInput(xInput, xTensor, device);
    CheckDescriptorInput(weightInput, weightTensor, device, true);
    const DescriptorInput scaleInputs[] = {
        {"x_level0_scale", &x_level0_scale, ACL_FLOAT},
        {"x_level1_scale", &x_level1_scale, ACL_FLOAT8_E8M0},
        {"weight_level0_scale", &weight_level0_scale, ACL_FLOAT},
        {"weight_level1_scale", &weight_level1_scale, ACL_FLOAT8_E8M0},
    };
    for (const auto& input : scaleInputs) {
        CheckDescriptorInput(input, RequireSingleTensor(input), device);
    }
    if (bias.has_value()) {
        const DescriptorInput biasInput{"bias", &*bias, ACL_FLOAT};
        CheckDescriptorInput(biasInput, RequireSingleTensor(biasInput), device);
    }
    TORCH_CHECK(group_list.defined() && group_list.device() == device, "group_list must be a tensor on the same NPU.");
    TORCH_CHECK(at_npu::native::get_npu_format(group_list) == ACL_FORMAT_ND, "group_list must use ND format.");
    // Guard the dimensions used for allocation and packed descriptor conversion.
    const auto& weightScaleTensor = weight_level1_scale[0];
    TORCH_CHECK(
        xTensor.dim() == X_RANK && weightTensor.dim() == WEIGHT_RANK && weightScaleTensor.dim() == WEIGHT_SCALE_RANK,
        "x, weight and weight_level1_scale must have rank 2, 3 and 4.");
    const int64_t m = xTensor.size(0);
    const int64_t n = weightScaleTensor.size(SCALE_N_AXIS);
    TORCH_CHECK(output_dtype == at::kHalf || output_dtype == at::kBFloat16, "Invalid output dtype.");
    const c10::OptionalDeviceGuard guard(device);
    std::vector<at::Tensor> out{at::empty({m, n}, xTensor.options().dtype(output_dtype))};
    const std::vector<at::Tensor> empty;
    at::TensorList xr(x), wr(weight), sr(weight_level1_scale), ar(x_level1_scale);
    at::TensorList a0r(x_level0_scale), b0r(weight_level0_scale), yr(out);
    at::TensorList br(bias.has_value() ? *bias : empty);
    const bool transposed = weightTensor.stride(PACKED_ROW_AXIS) == 1 && weightTensor.stride(-1) > 1;
    DenseList xa{xr, ACL_FLOAT4_E2M1, false};
    DenseList sa{sr, ACL_FLOAT8_E8M0, transposed};
    DenseList pa{ar, ACL_FLOAT8_E8M0, false};
    DenseList a0{a0r, ACL_FLOAT, false};
    DenseList b0{b0r, ACL_FLOAT, false};
    DenseList ba{br, ACL_FLOAT, false};
    DenseList ya{yr, output_dtype == at::kHalf ? ACL_FLOAT16 : ACL_BF16, false};
    WeightList wa{wr, xTensor.size(1) * FP4_ELEMENTS_PER_BYTE, n};
    StorageShapeTensor groups{group_list};
    c10::optional<at::IntArrayRef> config = c10::nullopt;
    if (tuning_config.has_value()) {
        config = at::IntArrayRef(*tuning_config);
    }
    ACLNN_CMD(aclnnDualLevelGroupedMatmulWeightNz, xa, wa, ba, sa, pa, groups, a0, b0, level0_group_size,
              level1_group_size, split_item, group_type, group_list_type, config, ya);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("dual_level_grouped_matmul", &dual_level_grouped_matmul, "MXFP4 two-level grouped matmul");
}
} // namespace op_api
