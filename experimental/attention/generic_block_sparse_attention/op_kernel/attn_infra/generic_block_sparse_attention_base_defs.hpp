/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file generic_block_sparse_attention_base_defs.hpp
 * \brief
 */

#ifndef BASE_DEFS_HPP
#define BASE_DEFS_HPP

#include <cstdint>
#include <type_traits>

#include <kernel_operator.h>

#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/layout/matrix.hpp"
#include "catlass/layout/vector.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/helper.hpp"
#if (__CCE_AICORE__ == 220)
// The GBSA build adds the catlass include root plus -DCATLASS_ARCH=2201, so
// the catlass root dispatchers work; keep pulling in the atlasa2 leaf headers
// directly anyway (especially CopyL1ToBT, which the catlass TileCopy primary
// template references unconditionally) so every name the TileCopy primary
// template needs is defined even without CATLASS_ARCH. (Formerly the include
// block of gemm/tile_common/gbsa_tile_copy.hpp, merged here.)
#include "catlass/gemm/tile/atlasa2/copy_gm_to_l1.hpp"
#include "catlass/gemm/tile/atlasa2/copy_l0c_to_gm.hpp"
#include "catlass/gemm/tile/atlasa2/copy_l1_to_l0a.hpp"
#include "catlass/gemm/tile/atlasa2/copy_l1_to_l0b.hpp"
#include "catlass/gemm/tile/atlasa2/copy_l1_to_bt.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#endif

namespace NpuArch {

namespace Arch {
using Catlass::Arch::AtlasA2;

template <class ArchTag>
using Resource = Catlass::Arch::Resource<ArchTag>;
} // namespace Arch

using Catlass::Coord;
using Catlass::MakeCoord;
using Catlass::GemmShape;
using Catlass::GemmCoord;
using Catlass::MatrixShape;
using Catlass::MatrixCoord;

namespace layout {
using Catlass::layout::VectorLayout;
using Catlass::layout::RowMajor;
using Catlass::layout::ColumnMajor;
using Catlass::layout::nZ;
using Catlass::layout::zN;
using Catlass::layout::zZ;
using Catlass::layout::PaddingRowMajor;
using Catlass::layout::PaddingColumnMajor;
using Catlass::layout::nN;
} // namespace layout

namespace Gemm {
using Catlass::Gemm::GemmType;

template <bool PAGED_CACHE_FLAG_ = false, bool ENABLE_UNIT_FLAG_ = false>
using MmadAtlasA2SFAIQK = Catlass::Gemm::MmadAtlasA2FAIQK<PAGED_CACHE_FLAG_, ENABLE_UNIT_FLAG_>;

template <bool PAGED_CACHE_FLAG_ = false, bool ENABLE_UNIT_FLAG_ = false>
using MmadAtlasA2SFAIPV = Catlass::Gemm::MmadAtlasA2FAIPV<PAGED_CACHE_FLAG_, ENABLE_UNIT_FLAG_>;

namespace helper {
using Catlass::Gemm::helper::L1AlignHelper;
using Catlass::Gemm::helper::ElementAccumulatorSelector;
using Catlass::Gemm::helper::L1ATypeSelector;
using Catlass::Gemm::helper::L1BTypeSelector;
using Catlass::Gemm::helper::L1BiasTypeSelector;
} // namespace helper

#if (__CCE_AICORE__ == 220)
namespace Tile {
using Catlass::Gemm::Tile::TileCopy;
using Catlass::Gemm::Tile::TileMmad;
} // namespace Tile
#endif
} // namespace Gemm

constexpr uint32_t BYTE_PER_C0 = 32;
constexpr uint32_t BYTE_PER_C2 = 64;
constexpr uint32_t C0_NUM_PER_FRACTAL = 16;
constexpr uint32_t BYTE_PER_FRACTAL = BYTE_PER_C0 * C0_NUM_PER_FRACTAL;

constexpr uint32_t BYTE_PER_BLK = 32;
constexpr uint32_t BLK_NUM_PER_VECTOR_FRACTAL = 8;
constexpr uint32_t BYTE_PER_VECTOR_FRACTAL = BYTE_PER_BLK * BLK_NUM_PER_VECTOR_FRACTAL;

constexpr uint64_t L2_OFFSET = 0;
constexpr uint32_t STRIDE_LIMIT = 65536;

constexpr uint32_t BYTE_PER_BLK_FP = 128; /// datablock size of A1->C2PiPE2GM

constexpr uint32_t MX_SCALE_COPY_GROUP_NUM = 2;
constexpr uint32_t MX_SCALE_GROUP_NUM = 32;
constexpr uint32_t MX_BASEK_FACTOR = 64;

class EmptyClass {};

} // namespace NpuArch

#endif // HPP_HPP
