/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GEMM_TILE_COPY_L1_TO_L0B_A5_HPP
#define GEMM_TILE_COPY_L1_TO_L0B_A5_HPP

#include "../../../attn_infra/gbsa_base_defs.hpp"
#include "../../../attn_infra/arch/gbsa_arch.hpp"
#include "../../../attn_infra/gemm/tile_common/gbsa_tile_copy_tla.hpp"
#include "../../../tla/gbsa_tla_tensor.hpp"

namespace NpuArch::Gemm::Tile {

/// Partial specialization for CopyL1ToL0B, AtlasA5, not B8, zN in and nZ out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::AtlasA5, tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::A1>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::B2>,
    std::enable_if_t<!AscendC::Std::is_one_of_v<ElementSrc, int8_t, float8_e4m3_t, fp8_e4m3fn_t, float8_e5m2_t> &&
                     !AscendC::Std::is_one_of_v<ElementDst, int8_t, float8_e4m3_t, fp8_e4m3fn_t, float8_e5m2_t> &&
                     tla::detail::iszN<ElementSrc, LayoutSrc>::value &&
                     tla::detail::isnZ<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementSrc);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(ElementSrc);

    // Mehtods

    __aicore__ inline TileCopyTla(){};

    template <class TensorDst, class TensorSrc>
    __aicore__ inline void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor)
    {
        static_assert(
            !AscendC::Std::is_one_of_v<typename TensorSrc::Element, int8_t, float8_e4m3_t, fp8_e4m3fn_t,
                                       float8_e5m2_t> &&
                !AscendC::Std::is_one_of_v<typename TensorDst::Element, int8_t, float8_e4m3_t, fp8_e4m3fn_t,
                                           float8_e5m2_t> &&
                tla::detail::iszN<typename TensorSrc::Element, typename TensorSrc::Layout>::value &&
                tla::detail::isnZ<typename TensorDst::Element, typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::A1 && TensorDst::position == AscendC::TPosition::B2,
            "The input parameters do not match. TensorSrc must be L1 and zN, while TensorDst must be L0B and nZ");

        const uint32_t L0K = tla::get<0, 0>(dstTensor.shape()) * tla::get<0, 1>(dstTensor.shape());
        const uint32_t L0N = tla::get<1, 0>(dstTensor.shape()) * tla::get<1, 1>(dstTensor.shape());
        const uint32_t srcOuterStrideCol = tla::get<1, 1>(srcTensor.stride());
        const uint32_t dstOuterStrideRow = tla::get<0, 1>(dstTensor.stride());
        auto srcCoord = srcTensor.coord();

        AscendC::LoadData2DParamsV2 loadDataParams;
        loadDataParams.mStartPosition = CeilDiv<C0_NUM_PER_FRACTAL>(tla::get<0>(srcCoord));
        loadDataParams.kStartPosition = CeilDiv<ELE_NUM_PER_C0>(tla::get<1>(srcCoord));
        loadDataParams.mStep = CeilDiv<C0_NUM_PER_FRACTAL>(L0K);
        loadDataParams.kStep = CeilDiv<ELE_NUM_PER_C0>(L0N);
        loadDataParams.srcStride = CeilDiv<ELE_NUM_PER_FRACTAL>(srcOuterStrideCol);
        loadDataParams.dstStride = CeilDiv<ELE_NUM_PER_FRACTAL>(dstOuterStrideRow);
        loadDataParams.ifTranspose = true;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        AscendC::LoadData(dstTensor.data()[dstOffset], srcTensor.data(), loadDataParams);
    }

    template <class TensorDst, class TensorSrc>
    __aicore__ inline void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor, uint32_t l0Batch)
    {
        static_assert(
            !AscendC::Std::is_one_of_v<typename TensorSrc::Element, int8_t, float8_e4m3_t, fp8_e4m3fn_t,
                                       float8_e5m2_t> &&
                !AscendC::Std::is_one_of_v<typename TensorDst::Element, int8_t, float8_e4m3_t, fp8_e4m3fn_t,
                                           float8_e5m2_t> &&
                tla::detail::iszN<typename TensorSrc::Element, typename TensorSrc::Layout>::value &&
                tla::detail::isnZ<typename TensorDst::Element, typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::A1 && TensorDst::position == AscendC::TPosition::B2,
            "The input parameters do not match. TensorSrc must be L1 and zN, while TensorDst must be L0B and nZ");

        const uint32_t L1K = tla::get<0, 0>(srcTensor.shape()) * tla::get<0, 1>(srcTensor.shape());
        const uint32_t L1N = tla::get<1, 0>(srcTensor.shape()) * tla::get<1, 1>(srcTensor.shape());
        const uint32_t L0K = tla::get<0, 0>(dstTensor.shape()) * tla::get<0, 1>(dstTensor.shape());
        const uint32_t L0N = tla::get<1, 0>(dstTensor.shape()) * tla::get<1, 1>(dstTensor.shape());
        const uint32_t srcOuterStrideCol = tla::get<1, 1>(srcTensor.stride());
        const uint32_t dstOuterStrideRow = tla::get<0, 1>(dstTensor.stride());

        AscendC::LoadData2DParamsV2 loadDataParams;
        loadDataParams.mStartPosition = 0;
        loadDataParams.kStartPosition = 0;
        loadDataParams.mStep = CeilDiv<C0_NUM_PER_FRACTAL>(L0K);
        loadDataParams.kStep = CeilDiv<ELE_NUM_PER_C0>(L0N);
        loadDataParams.srcStride = CeilDiv<ELE_NUM_PER_FRACTAL>(srcOuterStrideCol);
        loadDataParams.dstStride = CeilDiv<ELE_NUM_PER_FRACTAL>(dstOuterStrideRow);
        loadDataParams.ifTranspose = true;

        for (uint32_t l0BatchIdx = 0; l0BatchIdx < l0Batch; l0BatchIdx++) {
            AscendC::LoadData(dstTensor.data()[l0BatchIdx * L0N * L0K], srcTensor.data()[l0BatchIdx * L1N * L1K],
                              loadDataParams);
        }
    }
};

/// Partial specialization for CopyL1ToL0B, AtlasA5, B8, zN in and nZ out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::AtlasA5, tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::A1>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::B2>,
    std::enable_if_t<AscendC::Std::is_one_of_v<ElementSrc, int8_t, float8_e4m3_t, fp8_e4m3fn_t, float8_e5m2_t> &&
                     AscendC::Std::is_one_of_v<ElementDst, int8_t, float8_e4m3_t, fp8_e4m3fn_t, float8_e5m2_t> &&
                     tla::detail::iszN<ElementSrc, LayoutSrc>::value &&
                     tla::detail::isnZ<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementSrc);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(ElementSrc);

    // Mehtods

    __aicore__ inline TileCopyTla(){};

    template <class TensorDst, class TensorSrc>
    __aicore__ inline void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor)
    {
        static_assert(
            AscendC::Std::is_one_of_v<typename TensorSrc::Element, int8_t, float8_e4m3_t, fp8_e4m3fn_t,
                                      float8_e5m2_t> &&
                AscendC::Std::is_one_of_v<typename TensorDst::Element, int8_t, float8_e4m3_t, fp8_e4m3fn_t,
                                          float8_e5m2_t> &&
                tla::detail::iszN<typename TensorSrc::Element, typename TensorSrc::Layout>::value &&
                tla::detail::isnZ<typename TensorDst::Element, typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::A1 && TensorDst::position == AscendC::TPosition::B2,
            "The input parameters do not match. TensorSrc must be L1 and zN, while TensorDst must be L0B and nZ");

        const uint32_t L0K = tla::get<0, 0>(dstTensor.shape()) * tla::get<0, 1>(dstTensor.shape());
        const uint32_t L0N = tla::get<1, 0>(dstTensor.shape()) * tla::get<1, 1>(dstTensor.shape());
        const uint32_t srcOuterStrideCol = tla::get<1, 1>(srcTensor.stride());
        const uint32_t dstOuterStrideRow = tla::get<0, 1>(dstTensor.stride());
        auto srcCoord = srcTensor.coord();

        AscendC::LoadData2DParamsV2 loadDataParams;
        if (L0N % ELE_NUM_PER_C0 == 0) {
            loadDataParams.mStartPosition = CeilDiv<C0_NUM_PER_FRACTAL>(tla::get<0>(srcCoord));
            loadDataParams.kStartPosition = CeilDiv<ELE_NUM_PER_C0>(tla::get<1>(srcCoord));
            loadDataParams.mStep = CeilDiv<C0_NUM_PER_FRACTAL>(L0K);
            loadDataParams.kStep = CeilDiv<ELE_NUM_PER_C0>(L0N);
            loadDataParams.srcStride = CeilDiv<ELE_NUM_PER_FRACTAL>(srcOuterStrideCol);
            loadDataParams.dstStride = CeilDiv<ELE_NUM_PER_FRACTAL>(dstOuterStrideRow);
            loadDataParams.ifTranspose = true;

            auto dstOffset = dstTensor.layout()(dstTensor.coord());
            AscendC::LoadData(dstTensor.data()[dstOffset], srcTensor.data(), loadDataParams);
        } else {
            for (uint32_t kIdx = 0; kIdx < L0K / ELE_NUM_PER_C0; kIdx++) {
                loadDataParams.mStartPosition = CeilDiv<C0_NUM_PER_FRACTAL>(tla::get<0>(srcCoord)) + kIdx * 2;
                loadDataParams.kStartPosition = CeilDiv<ELE_NUM_PER_C0>(tla::get<1>(srcCoord));
                loadDataParams.mStep = 2;
                loadDataParams.kStep = CeilDiv<ELE_NUM_PER_C0>(L0N);
                loadDataParams.srcStride = CeilDiv<ELE_NUM_PER_FRACTAL>(srcOuterStrideCol);
                loadDataParams.dstStride = CeilDiv<ELE_NUM_PER_FRACTAL>(dstOuterStrideRow);
                loadDataParams.ifTranspose = true;

                auto dstOffset = dstTensor.layout()(dstTensor.coord());
                AscendC::LoadData(dstTensor.data()[dstOffset + kIdx * L0N * ELE_NUM_PER_C0], srcTensor.data(),
                                  loadDataParams);
            }
        }
    }

    template <class TensorDst, class TensorSrc>
    __aicore__ inline void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor, uint32_t l0Batch)
    {
        static_assert(
            AscendC::Std::is_one_of_v<typename TensorSrc::Element, int8_t, float8_e4m3_t, fp8_e4m3fn_t,
                                      float8_e5m2_t> &&
                AscendC::Std::is_one_of_v<typename TensorDst::Element, int8_t, float8_e4m3_t, fp8_e4m3fn_t,
                                          float8_e5m2_t> &&
                tla::detail::iszN<typename TensorSrc::Element, typename TensorSrc::Layout>::value &&
                tla::detail::isnZ<typename TensorDst::Element, typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::A1 && TensorDst::position == AscendC::TPosition::B2,
            "The input parameters do not match. TensorSrc must be L1 and zN, while TensorDst must be L0B and nZ");

        const uint32_t L1K = tla::get<0, 0>(srcTensor.shape()) * tla::get<0, 1>(srcTensor.shape());
        const uint32_t L1N = tla::get<1, 0>(srcTensor.shape()) * tla::get<1, 1>(srcTensor.shape());
        const uint32_t L0K = tla::get<0, 0>(dstTensor.shape()) * tla::get<0, 1>(dstTensor.shape());
        const uint32_t L0N = tla::get<1, 0>(dstTensor.shape()) * tla::get<1, 1>(dstTensor.shape());
        const uint32_t srcOuterStrideCol = tla::get<1, 1>(srcTensor.stride());
        const uint32_t dstOuterStrideRow = tla::get<0, 1>(dstTensor.stride());

        AscendC::LoadData2DParamsV2 loadDataParams;
        if (L0N % ELE_NUM_PER_C0 == 0) {
            loadDataParams.mStartPosition = 0;
            loadDataParams.kStartPosition = 0;
            loadDataParams.mStep = CeilDiv<C0_NUM_PER_FRACTAL>(L0K);
            loadDataParams.kStep = CeilDiv<ELE_NUM_PER_C0>(L0N);
            loadDataParams.srcStride = CeilDiv<ELE_NUM_PER_FRACTAL>(srcOuterStrideCol);
            loadDataParams.dstStride = CeilDiv<ELE_NUM_PER_FRACTAL>(dstOuterStrideRow);
            loadDataParams.ifTranspose = true;

            for (uint32_t l0BatchIdx = 0; l0BatchIdx < l0Batch; l0BatchIdx++) {
                AscendC::LoadData(dstTensor.data()[l0BatchIdx * L0N * L0K], srcTensor.data()[l0BatchIdx * L1N * L1K],
                                  loadDataParams);
            }
        } else {
            loadDataParams.mStartPosition = 0;
            loadDataParams.kStartPosition = 0;
            loadDataParams.mStep = 2;
            loadDataParams.kStep = CeilDiv<ELE_NUM_PER_C0>(L0N);
            loadDataParams.srcStride = CeilDiv<ELE_NUM_PER_FRACTAL>(srcOuterStrideCol);
            loadDataParams.dstStride = CeilDiv<ELE_NUM_PER_FRACTAL>(dstOuterStrideRow);
            loadDataParams.ifTranspose = true;
            for (uint32_t l0BatchIdx = 0; l0BatchIdx < l0Batch; l0BatchIdx++) {
                for (uint32_t kIdx = 0; kIdx < L0K / ELE_NUM_PER_C0; kIdx++) {
                    AscendC::LoadData(dstTensor.data()[l0BatchIdx * L0N * L0K + kIdx * L0N * ELE_NUM_PER_C0],
                                      srcTensor.data()[l0BatchIdx * L1N * L1K + kIdx * ELE_NUM_PER_FRACTAL * 2],
                                      loadDataParams);
                }
            }
        }
    }
};

/// Partial specialization for CopyL1ToL0B, AtlasA5, nZ in and nZ out. (Transpose B)
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<Arch::AtlasA5,
                   tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::A1>,
                   tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::B2>,
                   std::enable_if_t<tla::detail::isnZ<ElementSrc, LayoutSrc>::value &&
                                    tla::detail::isnZ<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementSrc);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(ElementSrc);

    // Mehtods

    __aicore__ inline TileCopyTla(){};

    template <class TensorDst, class TensorSrc>
    __aicore__ inline void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor)
    {
        static_assert(
            tla::detail::isnZ<typename TensorSrc::Element, typename TensorSrc::Layout>::value &&
                tla::detail::isnZ<typename TensorDst::Element, typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::A1 && TensorDst::position == AscendC::TPosition::B2,
            "The input parameters do not match. TensorSrc must be L1 and nZ, while TensorDst must be L0B and nZ");

        const uint32_t dstOuterShapeRow = tla::get<0, 1>(dstTensor.shape());
        const uint32_t dstOuterShapeCol = tla::get<1, 1>(dstTensor.shape());
        const uint32_t srcOuterStrideRow = tla::get<0, 1>(srcTensor.stride());
        const uint32_t dstOuterStrideRow = tla::get<0, 1>(dstTensor.stride());
        auto srcCoord = srcTensor.coord();

        AscendC::LoadData2DParamsV2 loadDataParams;
        loadDataParams.mStartPosition = CeilDiv<C0_NUM_PER_FRACTAL>(tla::get<1>(srcCoord));
        loadDataParams.kStartPosition = CeilDiv<ELE_NUM_PER_C0>(tla::get<0>(srcCoord));
        loadDataParams.mStep = dstOuterShapeCol;
        loadDataParams.kStep = dstOuterShapeRow;
        loadDataParams.srcStride = CeilDiv<ELE_NUM_PER_FRACTAL>(srcOuterStrideRow);
        loadDataParams.dstStride = CeilDiv<ELE_NUM_PER_FRACTAL>(dstOuterStrideRow);
        loadDataParams.ifTranspose = false;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        AscendC::LoadData(dstTensor.data()[dstOffset], srcTensor.data(), loadDataParams);
    }

    template <class TensorDst, class TensorSrc>
    __aicore__ inline void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor, uint32_t l0Batch)
    {
        static_assert(
            tla::detail::isnZ<typename TensorSrc::Element, typename TensorSrc::Layout>::value &&
                tla::detail::isnZ<typename TensorDst::Element, typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::A1 && TensorDst::position == AscendC::TPosition::B2,
            "The input parameters do not match. TensorSrc must be L1 and nZ, while TensorDst must be L0B and nZ");

        const uint32_t dstOuterShapeRow = tla::get<0, 1>(dstTensor.shape());
        const uint32_t dstOuterShapeCol = tla::get<1, 1>(dstTensor.shape());
        const uint32_t srcOuterStrideRow = tla::get<0, 1>(srcTensor.stride());
        const uint32_t dstOuterStrideRow = tla::get<0, 1>(dstTensor.stride());

        AscendC::LoadData2DParamsV2 loadDataParams;
        loadDataParams.mStartPosition = 0;
        loadDataParams.kStartPosition = 0;
        loadDataParams.mStep = dstOuterShapeCol;
        loadDataParams.kStep = dstOuterShapeRow * l0Batch;
        loadDataParams.srcStride = CeilDiv<ELE_NUM_PER_FRACTAL>(srcOuterStrideRow);
        loadDataParams.dstStride = CeilDiv<ELE_NUM_PER_FRACTAL>(dstOuterStrideRow);
        loadDataParams.ifTranspose = false;

        AscendC::LoadData(dstTensor.data(), srcTensor.data(), loadDataParams);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace NpuArch::Gemm::Tile

#if (__CCE_AICORE__ == 310) && defined(ORIG_DTYPE_QUERY) && defined(ORIG_DTYPE_KEY) && \
    (ORIG_DTYPE_QUERY == DT_FLOAT4_E2M1) && (ORIG_DTYPE_KEY == DT_FLOAT4_E2M1)
namespace NpuArch::Gemm::Tile {

// Transpose a V submatrix into L0B. Channel and KV offsets address the
// original full-tile zN data and nN scale layouts independently.
struct CopyL1ToL0VMxfp4 {
    template <class DstTensor>
    __aicore__ inline void operator()(const DstTensor& dst, const AscendC::LocalTensor<uint8_t>& src,
                                      const AscendC::LocalTensor<uint8_t>& valueScale, uint32_t headDim,
                                      uint32_t kvRows, uint32_t destinationRows = 0, uint32_t sourceKvRows = 0,
                                      uint32_t kvStart = 0, uint32_t channelStart = 0) const
    {
        AscendC::LoadData2DParamsV2 data;
        data.mStartPosition = kvStart / 16;
        data.kStartPosition = channelStart / 64;
        data.mStep = kvRows / 16;
        data.kStep = headDim / 64;
        data.srcStride = (sourceKvRows == 0 ? kvRows : sourceKvRows) / 16;
        data.dstStride = (destinationRows == 0 ? headDim : destinationRows) / 16;
        data.ifTranspose = true;
        AscendC::LoadData2DMxParams mx;
        mx.xStartPosition = channelStart / 16;
        mx.yStartPosition = kvStart / 64;
        mx.xStep = (destinationRows == 0 ? headDim : destinationRows) / 16;
        mx.yStep = kvRows / 64;
        mx.srcStride = (sourceKvRows == 0 ? kvRows : sourceKvRows) / 64;
        mx.dstStride = kvRows / 64;
        AscendC::LoadData(dst, src.ReinterpretCast<fp4x2_e2m1_t>(), valueScale.ReinterpretCast<AscendC::fp8_e8m0_t>(),
                          data, mx);
    }
};

} // namespace NpuArch::Gemm::Tile
#endif // MXFP4 on AtlasA5

#endif // GEMM_TILE_COPY_L1_TO_L0B_A5_HPP
