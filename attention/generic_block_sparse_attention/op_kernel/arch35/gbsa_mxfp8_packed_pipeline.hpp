/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GBSA_MXFP8_PACKED_PIPELINE_HPP
#define GBSA_MXFP8_PACKED_PIPELINE_HPP
#include "generic_block_sparse_attention_kernel_arch35_mxfp8_packed.h"
namespace GbsaKernelArch35 {
template <class QK, class Softmax, class PV, class Rescale, uint32_t ScaleMode, uint32_t PackedDim = 0U,
          Format QFormat = Format::BSND, Format KvFormat = QFormat, bool IsPaged = false>
struct GbsaMxfp8PackedPipelineSelector;
template <class BlockMmadQK, class EpilogueOnlineSoftmax, class BlockMmadPV, class EpilogueRescaleO, uint32_t PackedDim,
          Format QFormat, Format KvFormat, bool IsPaged>
struct GbsaMxfp8PackedPipelineSelector<BlockMmadQK, EpilogueOnlineSoftmax, BlockMmadPV, EpilogueRescaleO,
                                       GbsaMxScaleMode::PIPELINE, PackedDim, QFormat, KvFormat, IsPaged> {
    using Type = GbsaMxfp8PackedKernelArch35<BlockMmadQK, EpilogueOnlineSoftmax, BlockMmadPV, EpilogueRescaleO,
                                             PackedDim, QFormat, KvFormat, IsPaged>;
};

template <class QK, class Softmax, class PV, class Rescale, uint32_t PackedDim, Format QFormat, Format KvFormat,
          bool IsPaged>
struct GbsaMxfp8PackedPipelineSelector<QK, Softmax, PV, Rescale, GbsaMxScaleMode::COALESCED_PIPELINE, PackedDim,
                                       QFormat, KvFormat, IsPaged> {
    using Type = GbsaMxfp8PackedKernelArch35<QK, Softmax, PV, Rescale, PackedDim, QFormat, KvFormat, IsPaged>;
};

} // namespace GbsaKernelArch35
#endif
