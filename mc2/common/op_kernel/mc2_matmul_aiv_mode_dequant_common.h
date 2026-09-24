/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mc2_matmul_aiv_mode_dequant_common.h
 * \brief Common dequantization utilities for MC2 matmul AIV mode kernels.
 */

#ifndef MC2_MATMUL_AIV_MODE_DEQUANT_COMMON_H
#define MC2_MATMUL_AIV_MODE_DEQUANT_COMMON_H

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/tla_catlass.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/arch/tla_arch_resource.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/tla_coord.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/layout/tla_layout_layout.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/detail/tla_detail_callback.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/tla_gemm_coord.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/tla_matrix_coord.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/epilogue/block/tla_block_epilogue.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/tla_gemm_gemm_type.hpp"

namespace Mc2MatmulAivDequant {

template <class CType, class ScaleType, class PerTokenScaleType, class DType>
struct DataInfo {
    using ElementC = typename CType::Element;
    using LayoutC = typename CType::Layout;
    using ElementScale = typename ScaleType::Element;
    using LayoutScale = typename ScaleType::Layout;
    using ElementPerTokenScale = typename PerTokenScaleType::Element;
    using LayoutPerTokenScale = typename PerTokenScaleType::Layout;
    using ElementD = typename DType::Element;
    using LayoutD = typename DType::Layout;

    static constexpr bool IS_ELEMENT_TYPE_VALID =
        std::is_same_v<ElementC, int32_t> && (std::is_same_v<ElementD, half> || std::is_same_v<ElementD, bfloat16_t>) &&
        std::is_same_v<ElementScale, float> && std::is_same_v<ElementPerTokenScale, float>;
    static constexpr bool IS_LAYOUT_VALID = std::is_same_v<LayoutC, Catlass::layout::RowMajor> &&
                                            std::is_same_v<LayoutScale, Catlass::layout::VectorLayout> &&
                                            std::is_same_v<LayoutPerTokenScale, Catlass::layout::VectorLayout> &&
                                            std::is_same_v<LayoutD, Catlass::layout::RowMajor>;
};

template <class Resource>
class UbTensorAllocator {
public:
    CATLASS_DEVICE
    explicit UbTensorAllocator(const Resource &resource)
        : resource_(resource),
          offset_(0U)
    {}

    template <typename T>
    CATLASS_DEVICE AscendC::LocalTensor<T> Allocate(size_t elementCount)
    {
        return AllocateBytes<T>(elementCount * sizeof(T));
    }

    template <typename T>
    CATLASS_DEVICE AscendC::LocalTensor<T> AllocateBytes(size_t byteCount)
    {
        auto tensor = resource_.ubBuf.template GetBufferByByte<T>(offset_);
        offset_ += byteCount;
        return tensor;
    }

private:
    const Resource &resource_;
    size_t offset_;
};

class EventIdAllocator {
public:
    CATLASS_DEVICE
    EventIdAllocator()
        : eventVMte2_(0),
          eventMte2V_(0),
          eventMte3V_(0),
          eventVMte3_(0)
    {}

    CATLASS_DEVICE int32_t NextVMte2()
    {
        return eventVMte2_++;
    }

    CATLASS_DEVICE int32_t NextMte2V()
    {
        return eventMte2V_++;
    }

    CATLASS_DEVICE int32_t NextMte3V()
    {
        return eventMte3V_++;
    }

    CATLASS_DEVICE int32_t NextVMte3()
    {
        return eventVMte3_++;
    }

private:
    int32_t eventVMte2_;
    int32_t eventMte2V_;
    int32_t eventMte3V_;
    int32_t eventVMte3_;
};

} // namespace Mc2MatmulAivDequant

#endif // MC2_MATMUL_AIV_MODE_DEQUANT_COMMON_H
