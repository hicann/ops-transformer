/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fragment_tensor.h
 * \brief FragmentTensor：FragmentTensor：多个 GM fragment 的统一抽象，支持 Slice / Copy / Scatter。
 */

#pragma once

#include "tensor_api/tensor.h"
#include "blaze/gemm/tile/pad_mx_kl1.h"

namespace Apace {
namespace Basic {

using FragmentIdx = uint32_t;
using FragmentCount = uint32_t;

constexpr uint32_t MAX_FRAGMENT_COUNT = 32;

template <uint32_t Dims>
struct FragmentParam {
    uint64_t assembledShape[Dims];
    uint64_t fragmentSize;
    uint64_t realFragmentSize;
    uint32_t assembleAxis;
    uint32_t fragmentCnt;

    bool Validate() const
    {
        if (assembleAxis >= Dims) {
            return false;
        }
        if (assembledShape[assembleAxis] != fragmentSize * fragmentCnt) {
            return false;
        }
        if (fragmentSize == 0) {
            return false;
        }
        if (realFragmentSize == 0 || realFragmentSize > fragmentSize) {
            return false;
        }
        if (fragmentCnt == 0) {
            return false;
        }
        return true;
    }
};

template <uint32_t Dims, size_t... Is>
__aicore__ inline auto MakeCoordFromArrayImpl(const uint64_t (&arr)[Dims], AscendC::Std::index_sequence<Is...>)
{
    return AscendC::Te::MakeCoord(static_cast<int64_t>(arr[Is])...);
}

template <uint32_t Dims>
__aicore__ inline auto MakeCoordFromArray(const uint64_t (&arr)[Dims])
{
    return MakeCoordFromArrayImpl<Dims>(arr, AscendC::Std::make_index_sequence<Dims>{});
}

template <uint32_t Dims, size_t... Is>
__aicore__ inline auto MakeShapeFromArrayImpl(const uint64_t (&arr)[Dims], AscendC::Std::index_sequence<Is...>)
{
    return AscendC::Te::MakeShape(static_cast<int64_t>(arr[Is])...);
}

template <uint32_t Dims>
__aicore__ inline auto MakeShapeFromArray(const uint64_t (&arr)[Dims])
{
    return MakeShapeFromArrayImpl<Dims>(arr, AscendC::Std::make_index_sequence<Dims>{});
}

/*!
 * \brief FragmentTensor类（合并设计：fragment 定义 + 子区域视图统一）
 */
template <uint32_t Dims, uint32_t MaxFragments = MAX_FRAGMENT_COUNT, typename LayoutFactory = void,
          typename ElementType = uint8_t>
class FragmentTensor {
public:
    __aicore__ inline FragmentTensor() = default;

    /*!
     * \brief 构造函数
     * \note addrList 必须在 FragmentTensor 整个生命周期内保持有效，
     *       本类只保存指针，不拷贝数组内容。调用方须保证 addrList 非空。
     */
    __aicore__ inline FragmentTensor(const FragmentParam<Dims> &fragParam, GM_ADDR const *addrList)
        : fragParam_(fragParam)
    {
        if (addrList != nullptr) {
            addrList_ = addrList;
        }
        ArrayFill(sliceCoord_, 0, AscendC::Std::make_index_sequence<Dims>{});
        ArrayCopy(sliceShape_, fragParam_.assembledShape, AscendC::Std::make_index_sequence<Dims>{});
    }

    __aicore__ inline auto GetFragment(uint32_t idx) const
    {
        auto fragmentAddr = reinterpret_cast<__gm__ ElementType *>(addrList_[idx]);

        uint64_t fragmentShape[Dims];
        ArrayCopy(fragmentShape, fragParam_.assembledShape, AscendC::Std::make_index_sequence<Dims>{});
        fragmentShape[fragParam_.assembleAxis] = fragParam_.fragmentSize;

        auto memPtr = AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(fragmentAddr);
        return AscendC::Te::MakeTensor(memPtr, MakeLayout(fragmentShape));
    }

    __aicore__ inline uint64_t GetFragmentSize() const
    {
        return fragParam_.fragmentSize;
    }

    __aicore__ inline uint64_t GetRealFragmentSize() const
    {
        return fragParam_.realFragmentSize;
    }

    __aicore__ inline uint32_t GetFragmentCnt() const
    {
        return fragParam_.fragmentCnt;
    }

    __aicore__ inline uint32_t GetSplitAxis() const
    {
        return fragParam_.assembleAxis;
    }

    __aicore__ inline GM_ADDR GetFragmentAddr(uint32_t idx) const
    {
        return addrList_[idx];
    }

    __aicore__ inline void UpdateAddrList(GM_ADDR const *addrList)
    {
        if (addrList != nullptr) {
            addrList_ = addrList;
        }
    }

    __aicore__ inline const FragmentParam<Dims> &GetFragParam() const
    {
        return fragParam_;
    }

    __aicore__ inline uint64_t GetSliceCoord(uint32_t d) const
    {
        return sliceCoord_[d];
    }
    __aicore__ inline uint64_t GetSliceShape(uint32_t d) const
    {
        return sliceShape_[d];
    }

    template <typename CoordType, typename ShapeType>
    __aicore__ inline auto Slice(const CoordType &coord, const ShapeType &shape) const
    {
        FragmentTensor result(*this);
        SliceImpl(result.sliceCoord_, result.sliceShape_, coord, shape, AscendC::Std::make_index_sequence<Dims>{});
        return result;
    }

    /*!
     * \brief Fragment组成信息（动态推导）
     */
    struct FragmentComposition {
        uint64_t headSize;           // 头块大小
        uint64_t bodySize;           // 中间块大小
        uint64_t tailSize;           // 尾块大小
        uint64_t startRelativeCoord; // 头块在Fragment内的起始相对坐标

        uint32_t startFragmentIdx; // 起始Fragment索引
        uint32_t endFragmentIdx;   // 结束Fragment索引
        uint32_t totalFragmentCnt; // Fragment总数
        uint32_t bodyCnt;          // 中间块数量
    };

    /*!
     * \brief 获取子区域的Fragment组成（沿 assembleAxis 切分）
     */
    __aicore__ inline FragmentComposition ComputeFragmentComposition(uint64_t startCoord, uint64_t length) const
    {
        FragmentComposition comp;

        comp.startFragmentIdx = startCoord / fragParam_.fragmentSize;
        comp.startRelativeCoord = startCoord % fragParam_.fragmentSize;

        uint64_t endCoord = startCoord + length;
        comp.endFragmentIdx = endCoord / fragParam_.fragmentSize;

        comp.totalFragmentCnt = comp.endFragmentIdx - comp.startFragmentIdx + 1;

        if (comp.totalFragmentCnt == 1) {
            comp.headSize = length;
            comp.bodySize = fragParam_.fragmentSize;
            comp.bodyCnt = 0;
            comp.tailSize = length;
        } else {
            comp.headSize = fragParam_.fragmentSize - comp.startRelativeCoord;
            comp.bodySize = fragParam_.fragmentSize;
            comp.tailSize = endCoord % fragParam_.fragmentSize;

            if (comp.tailSize == 0) {
                comp.tailSize = fragParam_.fragmentSize;
                comp.endFragmentIdx--;
                comp.totalFragmentCnt--;
            }

            comp.bodyCnt = comp.totalFragmentCnt - 2;
        }

        return comp;
    }

    /*!
     * \brief Fragment信息
     *
     * - fragCoord[d]:   源侧（fragment 内相对坐标）
     * - localCoord[d]:   目的侧（子区域内坐标）
     * - copyShape[d]: 拷贝大小
     */
    struct FragmentInfo {
        GM_ADDR addr; // Fragment地址
        uint64_t fragCoord[Dims];
        uint64_t localCoord[Dims];
        uint64_t copyShape[Dims];
        uint32_t fragmentIdx; // Fragment索引（在父级addrList中）
    };

    /*!
     * \brief 获取Fragment信息（按槽位填充，this 的 sliceCoord/sliceShape 提供非 assembleAxis 值）
     */
    __aicore__ inline FragmentInfo GetFragmentInfo(uint32_t localIdx, const FragmentComposition &comp) const
    {
        FragmentInfo info;

        info.fragmentIdx = comp.startFragmentIdx + localIdx;
        info.addr = addrList_[info.fragmentIdx];

        // 非 assembleAxis 槽位：直接整体复制 slice 视图，assembleAxis 槽位下面覆盖
        ArrayCopy(info.fragCoord, sliceCoord_, AscendC::Std::make_index_sequence<Dims>{});
        ArrayFill(info.localCoord, 0, AscendC::Std::make_index_sequence<Dims>{});
        ArrayCopy(info.copyShape, sliceShape_, AscendC::Std::make_index_sequence<Dims>{});

        // assembleAxis 槽位：head/body/tail 三选一
        if (localIdx == 0) {
            info.copyShape[fragParam_.assembleAxis] = comp.headSize;
            info.localCoord[fragParam_.assembleAxis] = 0;
            info.fragCoord[fragParam_.assembleAxis] = comp.startRelativeCoord;
        } else if (localIdx < comp.totalFragmentCnt - 1) {
            info.copyShape[fragParam_.assembleAxis] = comp.bodySize;
            info.localCoord[fragParam_.assembleAxis] = comp.headSize + (localIdx - 1) * comp.bodySize;
            info.fragCoord[fragParam_.assembleAxis] = 0;
        } else {
            info.copyShape[fragParam_.assembleAxis] = comp.tailSize;
            info.localCoord[fragParam_.assembleAxis] = comp.headSize + comp.bodyCnt * comp.bodySize;
            info.fragCoord[fragParam_.assembleAxis] = 0;
        }

        return info;
    }

private:
    template <size_t... Is>
    static __aicore__ inline void ArrayCopy(uint64_t (&dst)[Dims], const uint64_t (&src)[Dims],
                                            AscendC::Std::index_sequence<Is...>)
    {
        ((dst[Is] = src[Is]), ...);
    }

    template <size_t... Is>
    static __aicore__ inline void ArrayFill(uint64_t (&dst)[Dims], uint64_t val, AscendC::Std::index_sequence<Is...>)
    {
        ((dst[Is] = val), ...);
    }

    template <typename CoordType, typename ShapeType, size_t... Is>
    __aicore__ inline void SliceImpl(uint64_t (&coord)[Dims], uint64_t (&shape)[Dims], const CoordType &inCoord,
                                     const ShapeType &inShape, AscendC::Std::index_sequence<Is...>) const
    {
        ((coord[Is] = sliceCoord_[Is] + AscendC::Te::Get<Is>(inCoord)), ...);
        ((shape[Is] = AscendC::Te::Get<Is>(inShape)), ...);
    }

    template <size_t... Is>
    __aicore__ inline auto MakeLayoutImpl(const uint64_t (&sizes)[Dims], AscendC::Std::index_sequence<Is...>) const
    {
        return LayoutFactory{}(sizes[Is]...);
    }

    __aicore__ inline auto MakeLayout(const uint64_t (&sizes)[Dims]) const
    {
        return MakeLayoutImpl(sizes, AscendC::Std::make_index_sequence<Dims>{});
    }

    FragmentParam<Dims> fragParam_;
    GM_ADDR const *addrList_{nullptr};
    uint64_t sliceCoord_[Dims]{};
    uint64_t sliceShape_[Dims]{};
};

template <bool isScatter, typename CopyHandle, typename TensorType, uint32_t Dims, uint32_t MaxF, typename GmLayoutF,
          typename ElemT>
__aicore__ inline void FragmentSliceCopy(CopyHandle copyHandle, TensorType &tensor,
                                         const FragmentTensor<Dims, MaxF, GmLayoutF, ElemT> &fragmentTensor)
{
    const auto assembleAxis = fragmentTensor.GetSplitAxis();
    auto composition = fragmentTensor.ComputeFragmentComposition(fragmentTensor.GetSliceCoord(assembleAxis),
                                                                 fragmentTensor.GetSliceShape(assembleAxis));

    for (uint32_t idx = 0; idx < composition.totalFragmentCnt; ++idx) {
        auto info = fragmentTensor.GetFragmentInfo(idx, composition);

        if constexpr (isScatter) {
            const uint64_t realFragmentSize = fragmentTensor.GetRealFragmentSize();
            if (info.fragCoord[assembleAxis] >= realFragmentSize) {
                continue;
            }
            uint64_t remain = realFragmentSize - info.fragCoord[assembleAxis];
            if (info.copyShape[assembleAxis] > remain) {
                info.copyShape[assembleAxis] = remain;
            }
        }

        auto fragment = fragmentTensor.GetFragment(info.fragmentIdx);
        auto fragSlice =
            fragment.Slice(MakeCoordFromArray<Dims>(info.fragCoord), MakeShapeFromArray<Dims>(info.copyShape));
        auto tensorSlice =
            tensor.Slice(MakeCoordFromArray<Dims>(info.localCoord), MakeShapeFromArray<Dims>(info.copyShape));
        if constexpr (isScatter) {
            AscendC::Te::Copy(copyHandle, fragSlice, tensorSlice);
        } else {
            AscendC::Te::Copy(copyHandle, tensorSlice, fragSlice);
        }
    }
}

// L1 tensor K 轴补零，复用 Blaze PadZero 底层 API。
template <typename TensorL1>
__aicore__ inline void PadMxKAL1Zero(TensorL1 &tensorL1, uint64_t kAxis)
{
    using type = typename TensorL1::elementType;
    auto layoutL1 = tensorL1.Layout();
    auto kAxisL1Align = AscendC::Std::get<0>(AscendC::Std::get<1>(layoutL1.Shape())) *
                        AscendC::Std::get<1>(AscendC::Std::get<1>(layoutL1.Shape()));

    if constexpr (AscendC::Te::IsSatisfiedPtnFormatV<TensorL1, AscendC::Te::NZLayoutPtn>) {
        if constexpr (Blaze::Gemm::Tile::PadMxKL1Base::IsMxFp4<type>()) {
            return;
        }
        if (kAxisL1Align - kAxis < AscendC::Te::C0_SIZE<type>) {
            return;
        }
        auto mAlign = AscendC::Std::get<0>(AscendC::Std::get<0>(layoutL1.Shape())) *
                      AscendC::Std::get<1>(AscendC::Std::get<0>(layoutL1.Shape()));
        auto kAxisND2NZAlign = AscendC::Std::ceil_align(kAxis, AscendC::Te::C0_SIZE<type>);
        auto sliceTensor = tensorL1.Slice(AscendC::Te::MakeCoord(0, kAxisND2NZAlign),
                                          AscendC::Te::MakeShape(mAlign, kAxisL1Align - kAxisND2NZAlign));
        Blaze::Gemm::Tile::PadMxKL1Base::PadZero(sliceTensor, 1, mAlign, 0);
    } else if constexpr (AscendC::Te::IsSatisfiedPtnFormatV<TensorL1, AscendC::Te::ZNLayoutPtn>) {
        if (kAxis == kAxisL1Align) {
            return;
        }
        auto m1 = AscendC::Std::get<1>(AscendC::Std::get<0>(layoutL1.Shape()));
        auto m0 = AscendC::Std::get<0>(AscendC::Std::get<0>(layoutL1.Shape()));
        auto dstRowStride = AscendC::Std::get<1>(AscendC::Std::get<0>(layoutL1.Stride()));
        auto dstGap = (dstRowStride / AscendC::Te::C0_ELEMENT<type>)-kAxisL1Align + kAxis;
        auto sliceTensor =
            tensorL1.Slice(AscendC::Te::MakeCoord(0, kAxis), AscendC::Te::MakeShape(m1 * m0, kAxisL1Align - kAxis));
        Blaze::Gemm::Tile::PadMxKL1Base::PadZero(sliceTensor, m1, kAxisL1Align - kAxis, dstGap);
    }
}

} // namespace Basic
} // namespace Apace
