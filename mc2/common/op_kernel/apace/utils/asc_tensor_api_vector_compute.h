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
 * \file asc_tensor_api_vector_compute.h
 * \brief
 */

#pragma once

#include "tensor_api/experimental/vector_ops.h"
#include "tensor_api/tensor.h"

// func for cast
template <typename DstT, typename SrcT>
__simd_callee__ inline asc::te::experimental::reg_tensor<DstT> cast_register(
    asc::te::experimental::reg_tensor<SrcT> src)
{
    return asc::te::experimental::cast<DstT>(src);
}

template <typename SrcTensor, typename DstTensor>
__simd_vf__ inline void cast_reg_loop(const SrcTensor src, DstTensor dst, uint32_t data_size)
{
    using src_type = typename SrcTensor::data_type;
    using dst_type = typename DstTensor::data_type;
    using wider_type = std::conditional_t<(sizeof(src_type) > sizeof(dst_type)), src_type, dst_type>;
    using wider_reg_type = typename asc::te::experimental::reg_tensor<wider_type>::reg_type;
    constexpr uint32_t one_repeat = sizeof(wider_reg_type) / sizeof(wider_type);
    const uint16_t repeat = static_cast<uint16_t>((data_size + one_repeat - 1) / one_repeat);
    uint32_t remain = data_size;

    for (uint16_t i = 0; i < repeat; ++i) {
        const uint32_t offset = i * one_repeat;
        const auto coord = asc::te::make_coord(offset);
        asc::te::experimental::reg_tensor<src_type> src_reg;

        if constexpr (sizeof(dst_type) == sizeof(src_type)) {
            src_reg = asc::te::experimental::load(src, coord);
        } else if constexpr (sizeof(dst_type) == 2 * sizeof(src_type)) {
            src_reg = asc::te::experimental::load<asc::te::experimental::load_sideband_mode::unpack>(src, coord);
        } else if constexpr (sizeof(dst_type) == 4 * sizeof(src_type)) {
            src_reg = asc::te::experimental::load<asc::te::experimental::load_sideband_mode::unpack4>(src, coord);
        } else {
            src_reg = asc::te::experimental::load(src, coord);
        }
        // update_mask 按引用自减 remain，先记录本轮有效元素个数（不能声明为 const）
        uint32_t store_remain = remain > one_repeat ? one_repeat : remain;
        src_reg = src_reg.with_mask(asc::te::experimental::update_mask<wider_type>(remain));

        auto dst_reg = cast_register<dst_type>(src_reg);
        if constexpr (sizeof(dst_type) == sizeof(src_type) || sizeof(dst_type) > sizeof(src_type)) {
            asc::te::experimental::store(dst, coord, dst_reg);
        } else if constexpr (sizeof(src_type) == 2 * sizeof(dst_type)) {
            auto compact = asc::te::experimental::deinterleave(dst_reg, dst_reg).first;
            compact = compact.with_mask(asc::te::experimental::update_mask<dst_type>(store_remain));
            asc::te::experimental::store(dst, coord, compact);
        } else {
            auto compact_2 = asc::te::experimental::deinterleave(dst_reg, dst_reg).first;
            auto compact_4 = asc::te::experimental::deinterleave(compact_2, compact_2).first;
            compact_4 = compact_4.with_mask(asc::te::experimental::update_mask<dst_type>(store_remain));
            asc::te::experimental::store(dst, coord, compact_4);
        }
    }
}

// func for add
template <typename InputTensor, typename OutputTensor>
__simd_vf__ inline void compute_add(InputTensor src0_tensor, InputTensor src1_tensor, OutputTensor dst_tensor,
                                    uint32_t data_size)
{
    using asc::te::experimental::operator+;

    using data_type = typename InputTensor::data_type;
    using reg_type = typename asc::te::experimental::reg_tensor<data_type>::reg_type;
    constexpr uint32_t one_repeat = sizeof(reg_type) / sizeof(data_type);
    const uint16_t repeat = static_cast<uint16_t>((data_size + one_repeat - 1) / one_repeat);
    uint32_t remain = data_size;

    for (uint16_t i = 0; i < repeat; ++i) {
        const uint32_t offset = i * one_repeat;
        const auto coord = asc::te::make_coord(offset);
        auto mask = asc::te::experimental::update_mask<data_type>(remain);
        auto src0 = asc::te::experimental::load(src0_tensor, coord).with_mask(mask);
        auto src1 = asc::te::experimental::load(src1_tensor, coord).with_mask(mask);
        auto dst = src0 + src1;
        asc::te::experimental::store(dst_tensor, coord, dst);
    }
}

// func for fill
template <typename T, typename U>
__simd_vf__ inline void compute_fill(T y_tensor, U scalar, uint32_t data_size)
{
    using data_type = typename T::data_type;
    constexpr uint32_t one_repeat_size = asc_get_vf_len() / sizeof(data_type);
    uint16_t repeat_times =
        static_cast<uint16_t>(AscendC::Std::ceil_align(data_size, one_repeat_size) / one_repeat_size);
    for (uint16_t i = 0; i < repeat_times; ++i) {
        auto mask = asc::te::experimental::update_mask<data_type>(data_size);
        uint32_t offset = i * one_repeat_size;
        const auto coord = asc::te::make_coord(offset);
        auto y_reg = asc::te::experimental::fill(scalar, mask);
        asc::te::experimental::store(y_tensor, coord, y_reg);
    }
}
