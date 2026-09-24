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
 * \file all_gather_matmul_v2_padding.h
 * \brief
 */

#ifndef __ALL_GATHER_MATMUL_AIV_MODE_PADDING_H__
#define __ALL_GATHER_MATMUL_AIV_MODE_PADDING_H__

#pragma once

#include "../../common/op_kernel/mc2_matmul_aiv_padding_common.h"
#include "all_gather_matmul_aiv_mode_util.h"

using namespace AscendC;
using namespace Catlass;
#define PADDING_ARGS_FUN() \
    bool transA, bool transB, bool alignedA, bool alignedB, uint32_t matrixAM, uint32_t matrixAK, uint32_t matrixBK, \
        uint32_t matrixBN, uint32_t matrixAMAlign, uint32_t matrixAKAlign, uint32_t matrixBKAlign, \
        uint32_t matrixBNAlign, GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmAAlign, GM_ADDR gmBAlign
#define PADDING_ARGS_CALL() \
    transA, transB, alignedA, alignedB, matrixAM, matrixAK, matrixBK, matrixBN, matrixAMAlign, matrixAKAlign, \
        matrixBKAlign, matrixBNAlign, gmA, gmB, gmAAlign, gmBAlign
namespace Catlass::Gemm::Kernel {
template <class ArchTag_, class AType_, class BType_>
class TemplatePadder {
public:
    using ArchTag = ArchTag_;
    using LayoutA = typename AType_::Layout;
    using LayoutB = typename BType_::Layout;
    using Params = Mc2MatmulAivPadding::MatmulPaddingParams<LayoutA, LayoutB>;
    // Methods
    CATLASS_DEVICE
    TemplatePadder() {}
    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const &params);
    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const &params)
    {
        Mc2MatmulAivPadding::PadMatmulInputs<ArchTag, AType_, BType_>(resource, params);
        Mc2MatmulAivPadding::NotifyMatmulPaddingFinished(flagAivFinishPadding);
    }

private:
    static constexpr Arch::FlagID FLAG_AIV_FINISH_STORE = AIC_WAIT_AIV_FINISH_ALIGN_FLAG_ID;
    Arch::CrossCoreFlag flagAivFinishPadding{FLAG_AIV_FINISH_STORE};
    Arch::Resource<ArchTag> resource;
};
} // namespace Catlass::Gemm::Kernel
template <typename InputType, typename WeightType>
class PaddingRunner {
public:
    __aicore__ explicit PaddingRunner() = default;
    inline __aicore__ void Run(PADDING_ARGS_FUN())
    {
        using ArchTag = Arch::AtlasA2;
        using ElementA =
            typename std::conditional<std::is_same<InputType, AscendC::int4b_t>::value, int8_t, InputType>::type;
        using ElementB =
            typename std::conditional<std::is_same<WeightType, AscendC::int4b_t>::value, int8_t, WeightType>::type;
        if (!transA && !transB) {
            using LayoutA = layout::RowMajor;
            using LayoutB = layout::RowMajor;
            using AType = Gemm::GemmType<ElementA, LayoutA>;
            using BType = Gemm::GemmType<ElementB, LayoutB>;
            if constexpr (std::is_same_v<InputType, AscendC::int4b_t>) {
                matrixAK = matrixAK / 2;
                matrixBN = matrixBN / 2;
                matrixAKAlign = matrixAKAlign / 2;
                matrixBNAlign = matrixBNAlign / 2;
            }
            LayoutA layoutA{matrixAM, matrixAK};
            LayoutB layoutB{matrixBK, matrixBN};
            // 根据是否转置
            LayoutA layoutWA{matrixAM, matrixAKAlign};
            LayoutB layoutWB{matrixBK, matrixBNAlign};
            using TemplatePadder = Gemm::Kernel::TemplatePadder<ArchTag, AType, BType>;
            typename TemplatePadder::Params params{gmA,      layoutA,  gmB,      layoutB,  gmAAlign,
                                                   layoutWA, gmBAlign, layoutWB, alignedA, alignedB};
            TemplatePadder padder;
            padder(params);
        } else if (!transA && transB) {
            using LayoutA = layout::RowMajor;
            using LayoutB = layout::ColumnMajor;
            using AType = Gemm::GemmType<ElementA, LayoutA>;
            using BType = Gemm::GemmType<ElementB, LayoutB>;
            if constexpr (std::is_same_v<InputType, AscendC::int4b_t>) {
                matrixAK = matrixAK / 2;
                matrixBK = matrixBK / 2;
                matrixAKAlign = matrixAKAlign / 2;
                matrixBKAlign = matrixBKAlign / 2;
            }
            LayoutA layoutA{matrixAM, matrixAK};
            LayoutB layoutB{matrixBK, matrixBN};
            LayoutA layoutWA{matrixAM, matrixAKAlign};
            LayoutB layoutWB{matrixBKAlign, matrixBN};
            using TemplatePadder = Gemm::Kernel::TemplatePadder<ArchTag, AType, BType>;
            typename TemplatePadder::Params params{gmA,      layoutA,  gmB,      layoutB,  gmAAlign,
                                                   layoutWA, gmBAlign, layoutWB, alignedA, alignedB};
            TemplatePadder padder;
            padder(params);
        } else if (transA && !transB) {
            using LayoutA = layout::ColumnMajor;
            using LayoutB = layout::RowMajor;
            using AType = Gemm::GemmType<ElementA, LayoutA>;
            using BType = Gemm::GemmType<ElementB, LayoutB>;
            if constexpr (std::is_same_v<InputType, AscendC::int4b_t>) {
                matrixAM = matrixAM / 2;
                matrixBN = matrixBN / 2;
                matrixAMAlign = matrixAMAlign / 2;
                matrixBNAlign = matrixBNAlign / 2;
            }
            LayoutA layoutA{matrixAM, matrixAK};
            LayoutB layoutB{matrixBK, matrixBN};
            LayoutA layoutWA{matrixAMAlign, matrixAK};
            LayoutB layoutWB{matrixBK, matrixBNAlign};
            using TemplatePadder = Gemm::Kernel::TemplatePadder<ArchTag, AType, BType>;
            typename TemplatePadder::Params params{gmA,      layoutA,  gmB,      layoutB,  gmAAlign,
                                                   layoutWA, gmBAlign, layoutWB, alignedA, alignedB};
            TemplatePadder padder;
            padder(params);
        } else {
            using LayoutA = layout::ColumnMajor;
            using LayoutB = layout::ColumnMajor;
            using AType = Gemm::GemmType<ElementA, LayoutA>;
            using BType = Gemm::GemmType<ElementB, LayoutB>;
            if constexpr (std::is_same_v<InputType, AscendC::int4b_t>) {
                matrixAM = matrixAM / 2;
                matrixBK = matrixBK / 2;
                matrixAMAlign = matrixAMAlign / 2;
                matrixBKAlign = matrixBKAlign / 2;
            }
            LayoutA layoutA{matrixAM, matrixAK};
            LayoutB layoutB{matrixBK, matrixBN};
            LayoutA layoutWA{matrixAMAlign, matrixAK};
            LayoutB layoutWB{matrixBKAlign, matrixBN};
            using TemplatePadder = Gemm::Kernel::TemplatePadder<ArchTag, AType, BType>;
            typename TemplatePadder::Params params{gmA,      layoutA,  gmB,      layoutB,  gmAAlign,
                                                   layoutWA, gmBAlign, layoutWB, alignedA, alignedB};
            TemplatePadder padder;
            padder(params);
        }
    }
};

#endif //__ALL_GATHER_MATMUL_AIV_MODE_PADDING_H__
