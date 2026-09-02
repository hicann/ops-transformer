/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATMUL_REDUCE_SCATTER_AIV_MODE_PADDING_H
#define MATMUL_REDUCE_SCATTER_AIV_MODE_PADDING_H

#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/arch/tla_arch_resource.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/arch/tla_arch_cross_core_sync.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/kernel/tla_gemm_kernel_padding_matmul.hpp"
#include "matmul_reduce_scatter_aiv_mode_util.h"

using namespace AscendC;
using namespace Catlass;
using namespace matmulReduceScatterV2_util;

#define PADDING_ARGS_FUN() \
    bool transA, bool transB, bool alignedA, bool alignedB, uint32_t matrixAM, uint32_t matrixAK, uint32_t matrixBK, \
        uint32_t matrixBN, uint32_t matrixAMAlign, uint32_t matrixAKAlign, uint32_t matrixBKAlign, \
        uint32_t matrixBNAlign, GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmAAlign, GM_ADDR gmBAlign, bool castBias, \
        uint32_t biasLength, GM_ADDR gmBias, GM_ADDR gmBiasCast

namespace Catlass::Gemm::Kernel {
static constexpr uint32_t BIAS_CAST_TILE_ELEMENTS = 16 * 1024;

template <class ArchTag_>
class Bf16BiasCaster {
public:
    using ArchTag = ArchTag_;

    CATLASS_DEVICE
    explicit Bf16BiasCaster(Arch::Resource<ArchTag> &resource_)
        : resource(resource_)
    {}

    CATLASS_DEVICE
    void operator()(GM_ADDR ptrDst, GM_ADDR ptrSrc, uint32_t biasLength)
    {
        static_assert(BIAS_CAST_TILE_ELEMENTS * (sizeof(bfloat16_t) + sizeof(float)) <= ArchTag::UB_SIZE,
                      "Exceeding the UB space!");

        uint64_t aivNum = static_cast<uint64_t>(AscendC::GetBlockNum()) * AscendC::GetSubBlockNum();
        uint64_t aivId = AscendC::GetBlockIdx();
        uint64_t len = biasLength;
        uint64_t elementsPerAiv = len / aivNum + (len % aivNum != 0);
        uint64_t start = aivId * elementsPerAiv;
        uint64_t end = start + elementsPerAiv < len ? start + elementsPerAiv : len;

        AscendC::GlobalTensor<bfloat16_t> gmSrc;
        AscendC::GlobalTensor<float> gmDst;
        gmSrc.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ptrSrc));
        gmDst.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ptrDst));

        auto srcLocal = resource.ubBuf.template GetBufferByByte<bfloat16_t>(0);
        auto dstLocal = resource.ubBuf.template GetBufferByByte<float>(BIAS_CAST_TILE_ELEMENTS * sizeof(bfloat16_t));

        AscendC::TEventID eventSrc = EVENT_ID0;
        AscendC::TEventID eventDst = EVENT_ID1;
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventSrc);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventDst);

        for (uint64_t offset = start; offset < end; offset += BIAS_CAST_TILE_ELEMENTS) {
            uint32_t actual =
                static_cast<uint32_t>(end - offset < BIAS_CAST_TILE_ELEMENTS ? end - offset : BIAS_CAST_TILE_ELEMENTS);
            AscendC::DataCopyExtParams copyInParams(1, actual * sizeof(bfloat16_t), 0, 0, 0);
            AscendC::DataCopyPadExtParams<bfloat16_t> padParams;

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventSrc);
            AscendC::DataCopyPad(srcLocal, gmSrc[offset], copyInParams, padParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventSrc);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventSrc);

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventDst);
            AscendC::Cast(dstLocal, srcLocal, AscendC::RoundMode::CAST_NONE, actual);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventSrc);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventDst);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventDst);

            AscendC::DataCopyExtParams copyOutParams(1, actual * sizeof(float), 0, 0, 0);
            AscendC::DataCopyPad(gmDst[offset], dstLocal, copyOutParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventDst);
        }

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventSrc);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventDst);
    }

private:
    Arch::Resource<ArchTag> &resource;
};

template <class ArchTag_, class AType_, class BType_>
class TemplatePadder {
public:
    using ArchTag = ArchTag_;
    using ElementA = typename AType_::Element;
    using LayoutA = typename AType_::Layout;
    using ElementB = typename BType_::Element;
    using LayoutB = typename BType_::Layout;

    static const uint32_t COMPUTE_LENGTH_A = 96 * 1024 / sizeof(ElementA);
    using PaddingA = PaddingMatrix<ArchTag, ElementA, LayoutA, COMPUTE_LENGTH_A>;
    static const uint32_t COMPUTE_LENGTH_B = 96 * 1024 / sizeof(ElementB);
    using PaddingB = PaddingMatrix<ArchTag, ElementB, LayoutB, COMPUTE_LENGTH_B>;

    /// Parameters structure
    struct Params {
        // Data members
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrWA;    // A矩阵padding地址
        LayoutA layoutWA; // A矩阵padding布局
        GM_ADDR ptrWB;    // B矩阵padding地址
        LayoutB layoutWB; // B矩阵padding布局
        bool alignA;      // A矩阵是否padding
        bool alignB;      // B矩阵是否padding
        bool castBias;
        uint32_t biasLength;
        GM_ADDR ptrBias;
        GM_ADDR ptrBiasCast;

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GM_ADDR ptrA_, LayoutA layoutA_, GM_ADDR ptrB_, LayoutB layoutB_, GM_ADDR ptrWA_, LayoutA layoutWA_,
               GM_ADDR ptrWB_, LayoutB layoutWB_, bool alignA_, bool alignB_, bool castBias_, uint32_t biasLength_,
               GM_ADDR ptrBias_, GM_ADDR ptrBiasCast_)
            : ptrA(ptrA_),
              layoutA(layoutA_),
              ptrB(ptrB_),
              layoutB(layoutB_),
              ptrWA(ptrWA_),
              layoutWA(layoutWA_),
              ptrWB(ptrWB_),
              layoutWB(layoutWB_),
              alignA(alignA_),
              alignB(alignB_),
              castBias(castBias_),
              biasLength(biasLength_),
              ptrBias(ptrBias_),
              ptrBiasCast(ptrBiasCast_)
        {}
    };

    // Methods
    CATLASS_DEVICE
    TemplatePadder() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const &params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const &params)
    {
        if (params.alignA) {
            AscendC::GlobalTensor<ElementA> gmA;
            AscendC::GlobalTensor<ElementA> gmWA;
            gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrA));
            gmWA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrWA));
            PaddingA paddingA(resource);
            paddingA(gmWA, gmA, params.layoutWA, params.layoutA);
        }

        if (params.alignB) {
            AscendC::GlobalTensor<ElementB> gmB;
            AscendC::GlobalTensor<ElementB> gmWB;
            gmB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrB));
            gmWB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrWB));
            PaddingB paddingB(resource);
            paddingB(gmWB, gmB, params.layoutWB, params.layoutB);
        }
        if (params.castBias) {
            Bf16BiasCaster<ArchTag> caster(resource);
            caster(params.ptrBiasCast, params.ptrBias, params.biasLength);
        }
        // 0x0 synchronization control between AI Core
        Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishPadding);
    }

private:
    static constexpr Arch::FlagID FLAG_AIV_FINISH_STORE = AIC_WAIT_AIV_FINISH_ALIGN_FLAG_ID;
    Arch::CrossCoreFlag flagAivFinishPadding{FLAG_AIV_FINISH_STORE};
    Arch::Resource<ArchTag> resource;
};

} // namespace Catlass::Gemm::Kernel

namespace padding {
template <typename InputType, typename WeightType>
class PaddingRunner {
public:
    __aicore__ explicit PaddingRunner() = default;

    inline __aicore__ void Run(PADDING_ARGS_FUN())
    {
        using ArchTag = Arch::AtlasA2;
        using ElementA = InputType;
        using ElementB = WeightType;

        if (!transA && !transB) {
            using LayoutA = layout::RowMajor;
            using LayoutB = layout::RowMajor;
            using AType = Gemm::GemmType<ElementA, LayoutA>;
            using BType = Gemm::GemmType<ElementB, LayoutB>;
            LayoutA layoutA{matrixAM, matrixAK};
            LayoutB layoutB{matrixBK, matrixBN};
            // 根据是否转置
            LayoutA layoutWA{matrixAM, matrixAKAlign};
            LayoutB layoutWB{matrixBK, matrixBNAlign};

            using TemplatePadder = Gemm::Kernel::TemplatePadder<ArchTag, AType, BType>;
            typename TemplatePadder::Params params{gmA,      layoutA,    gmB,      layoutB,   gmAAlign,
                                                   layoutWA, gmBAlign,   layoutWB, alignedA,  alignedB,
                                                   castBias, biasLength, gmBias,   gmBiasCast};
            TemplatePadder padder;
            padder(params);
        } else if (!transA && transB) {
            using LayoutA = layout::RowMajor;
            using LayoutB = layout::ColumnMajor;
            using AType = Gemm::GemmType<ElementA, LayoutA>;
            using BType = Gemm::GemmType<ElementB, LayoutB>;
            LayoutA layoutA{matrixAM, matrixAK};
            LayoutB layoutB{matrixBK, matrixBN};
            LayoutA layoutWA{matrixAM, matrixAKAlign};
            LayoutB layoutWB{matrixBKAlign, matrixBN};

            using TemplatePadder = Gemm::Kernel::TemplatePadder<ArchTag, AType, BType>;
            typename TemplatePadder::Params params{gmA,      layoutA,    gmB,      layoutB,   gmAAlign,
                                                   layoutWA, gmBAlign,   layoutWB, alignedA,  alignedB,
                                                   castBias, biasLength, gmBias,   gmBiasCast};
            TemplatePadder padder;
            padder(params);
        } else if (transA && !transB) {
            using LayoutA = layout::ColumnMajor;
            using LayoutB = layout::RowMajor;
            using AType = Gemm::GemmType<ElementA, LayoutA>;
            using BType = Gemm::GemmType<ElementB, LayoutB>;
            LayoutA layoutA{matrixAM, matrixAK};
            LayoutB layoutB{matrixBK, matrixBN};
            LayoutA layoutWA{matrixAMAlign, matrixAK};
            LayoutB layoutWB{matrixBK, matrixBNAlign};

            using TemplatePadder = Gemm::Kernel::TemplatePadder<ArchTag, AType, BType>;
            typename TemplatePadder::Params params{gmA,      layoutA,    gmB,      layoutB,   gmAAlign,
                                                   layoutWA, gmBAlign,   layoutWB, alignedA,  alignedB,
                                                   castBias, biasLength, gmBias,   gmBiasCast};
            TemplatePadder padder;
            padder(params);
        } else {
            using LayoutA = layout::ColumnMajor;
            using LayoutB = layout::ColumnMajor;
            using AType = Gemm::GemmType<ElementA, LayoutA>;
            using BType = Gemm::GemmType<ElementB, LayoutB>;
            LayoutA layoutA{matrixAM, matrixAK};
            LayoutB layoutB{matrixBK, matrixBN};
            LayoutA layoutWA{matrixAMAlign, matrixAK};
            LayoutB layoutWB{matrixBKAlign, matrixBN};

            using TemplatePadder = Gemm::Kernel::TemplatePadder<ArchTag, AType, BType>;
            typename TemplatePadder::Params params{gmA,      layoutA,    gmB,      layoutB,   gmAAlign,
                                                   layoutWA, gmBAlign,   layoutWB, alignedA,  alignedB,
                                                   castBias, biasLength, gmBias,   gmBiasCast};
            TemplatePadder padder;
            padder(params);
        }
    }
};
} // namespace padding

#endif
