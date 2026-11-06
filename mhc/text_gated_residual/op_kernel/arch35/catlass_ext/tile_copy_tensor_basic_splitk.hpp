/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or
 * modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS
 * SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT
 * NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of
 * the software repository for the full text of the License.
 */
/**
 * TileCopy for MmadAscend950BasicSplitK.
 * Named slots match PackedTileCopyTla so BlockMmadTla can substitute TileCopy_.
 * Internals use CANN tensor_api (AscendC::Te::Copy / MakeCopy).
 */
#ifndef CATLASS_GEMM_TILE_TILE_COPY_TENSOR_BASIC_SPLITK_HPP
#define CATLASS_GEMM_TILE_TILE_COPY_TENSOR_BASIC_SPLITK_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/detail/tag_to_layout.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"
#include "tensor_api/tensor.h"

namespace Catlass::Gemm::Tile {

struct TeCopyGmToL1 {
    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst& dst, TensorSrc const& src) const
    {
        auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
        AscendC::Te::Copy(copyGM2L1, dst, src);
    }
};

struct TeCopyL1ToL0A {
    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst& dst, TensorSrc const& src) const
    {
        auto copyL12L0A = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0A{});
        AscendC::Te::Copy(copyL12L0A, dst, src);
    }
};

struct TeCopyL1ToL0B {
    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst& dst, TensorSrc const& src) const
    {
        auto copyL12L0B = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0B{});
        AscendC::Te::Copy(copyL12L0B, dst, src);
    }
};

template <bool ENABLE_UNIT_FLAG>
struct TeCopyL0CToGm {
    static constexpr uint8_t kUnitFlagUpdate = 3;
    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst& dst, TensorSrc const& src) const
    {
        auto copyL0C2GM = AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2GM{});
        if constexpr (ENABLE_UNIT_FLAG) {
            AscendC::Te::FixpipeParams fixpParams{kUnitFlagUpdate};
            AscendC::Te::Copy(copyL0C2GM.with(fixpParams), dst, src);
        } else {
            AscendC::Te::Copy(copyL0C2GM, dst, src);
        }
    }
};

template <class ArchTag, class ElementA_, class LayoutTagA_, class ElementB_, class LayoutTagB_, class ElementC_,
          class LayoutTagC_, class ElementBias = void, bool ENABLE_UNIT_FLAG = true>
struct TileCopyTensorBasicSplitK {
    using ElementA = ElementA_;
    using ElementB = ElementB_;
    using LayoutTagA = LayoutTagA_;
    using LayoutTagB = LayoutTagB_;
    using LayoutTagC = LayoutTagC_;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
    static constexpr bool HAS_BIAS = !std::is_void_v<ElementBias>;
    static constexpr bool ReluEnable = false;

    using LayoutTagL1A = layout::zN;
    using LayoutTagL1B = layout::nZ;
    using LayoutTagL0A = layout::zN;
    using LayoutTagL0B = layout::nZ;
    using LayoutTagL0C = layout::L0C;

    using LayoutA = detail::TagToLayout_t<ElementA, LayoutTagA>;
    using LayoutB = detail::TagToLayout_t<ElementB, LayoutTagB>;
    using LayoutC = detail::TagToLayout_t<ElementC_, LayoutTagC>;

    using LayoutL1A = detail::TagToLayout_t<ElementA, LayoutTagL1A>;
    using LayoutL1B = detail::TagToLayout_t<ElementB, LayoutTagL1B>;
    using LayoutL0A = detail::TagToLayout_t<ElementA, LayoutTagL0A>;
    using LayoutL0B = detail::TagToLayout_t<ElementB, LayoutTagL0B>;
    using LayoutL0C = typename detail::LayoutL0C;

    template <class TensorA>
    using CopyGmToL1A = TeCopyGmToL1;
    template <class TensorB>
    using CopyGmToL1B = TeCopyGmToL1;
    using CopyL1ToL0A = TeCopyL1ToL0A;
    using CopyL1ToL0B = TeCopyL1ToL0B;
    template <class TensorC>
    using CopyL0CToDst = TeCopyL0CToGm<ENABLE_UNIT_FLAG>;
    template <class TensorC>
    using CopyL0CToGm = CopyL0CToDst<TensorC>;
};

} // namespace Catlass::Gemm::Tile

#endif
