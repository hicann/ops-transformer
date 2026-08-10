/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_GMM_COMMON_H
#define MEGA_MOE_GMM_COMMON_H

#include "kernel_operator.h"
#include "tensor_api/tensor.h"
#include "mega_moe_constants.h"
#include "mega_moe_types.h"
#include "mega_moe_utils.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "../blaze/gemm/block/block_scheduler_swizzle.h"
#include "../blaze/gemm/block/block_mmad_mx_fp8fp4.h"
#include "../blaze/prologue/block_prologue_mx_fp8fp4.h"

namespace MegaMoeImpl {

using namespace AscendC;

namespace GmmKernel {

constexpr uint32_t L1_TILE_K = 256U;
constexpr uint32_t SCALE_K_L1_RATE = 2U;

using BlockScheduler = typename Blaze::Gemm::Block::BlockSchedulerSwizzle<3, 1>; // 3: SwizzleOffset

// 根据数据类型路径选择对应的 BlockMmad 实现。
template <bool IsA8W4, typename C>
struct BlockMmadSelector;

template <typename C>
struct BlockMmadSelector<false, C> {
    using type =
        Blaze::Gemm::Block::BlockMmad<typename C::DispatchPolicy, typename C::ElementAType, typename C::LayoutA,
                                      typename C::ElementBType, typename C::LayoutB, typename C::ElementCType,
                                      typename C::LayoutC, typename C::BiasType, typename C::LayoutBias>;
};

template <typename C>
struct BlockMmadSelector<true, C> {
    using type =
        Blaze::Gemm::Block::BlockMmad<typename C::DispatchPolicy,
                                      AscendC::Std::tuple<typename C::ElementAType, typename C::ElementMxScaleAType>,
                                      AscendC::Std::tuple<typename C::MakeLayoutA, typename C::MakeLayoutScaleA>,
                                      AscendC::Std::tuple<typename C::ElementBType, typename C::ElementMxScaleBType>,
                                      AscendC::Std::tuple<typename C::MakeLayoutB, typename C::MakeLayoutScaleB>,
                                      typename C::ElementCType, typename C::MakeLayoutC, void, void>;
};

// 汇总 GMM1/GMM2 在不同量化路径下共用的类型、shape 和 layout 配置。
template <bool IsA8W4, uint8_t CombineQuantMode, typename ElementA, typename ElementB, typename ElementC,
          typename ElementMxScaleA, typename ElementMxScaleB, bool IsWeightNZ = false,
          bool TopkWeightsPrefetch = false, bool IsShared = false,
          bool IsLayered = false, bool IsGmm1Interleaved = false, bool IsWaveFlagGrained = false>
struct Config {
    static constexpr bool IS_SHARED = IsShared;
    static constexpr bool TOPK_WEIGHTS_PREFETCH = TopkWeightsPrefetch;
    using ElementAType = ElementA;
    using ElementBType = ElementB;
    using ElementCType = ElementC;
    using ElementMxScaleAType = ElementMxScaleA;
    using ElementMxScaleBType = ElementMxScaleB;
    using BiasType = float;

    static constexpr uint32_t C0_SIZE_A = AuxGetC0Size<ElementA>();
    static constexpr uint32_t C0_SIZE_C = AuxGetC0Size<ElementC>();
    static constexpr uint32_t C0_SIZE_SCALE = 2U;
    static constexpr uint32_t C0_SIZE_B = IsA8W4 ? 32U : AuxGetC0Size<ElementB>();
    static constexpr uint32_t C0_SIZE_BIAS = AuxGetC0Size<BiasType>();

    using LayoutA = Te::NDExtLayoutPtn;
    using LayoutC = Te::NDExtLayoutPtn;
    using LayoutScaleA = Te::ScaleANDLayoutPtn;
    using LayoutScaleB = Te::ScaleBDNLayoutPtn;

    using LayoutBias = Te::NDExtLayoutPtn;
    using DispatchPolicy =
        Std::conditional_t<IsA8W4, Blaze::Gemm::MatmulMxFp8Fp4DynamicKL1TailResplit, Blaze::Gemm::MatmulWithScaleMx<>>;
    using LayoutB = Std::conditional_t<IsA8W4, Te::ZNLayoutPtn,
                                       Std::conditional_t<IsWeightNZ, Te::ZNLayoutPtn, Te::DNExtLayoutPtn>>;

    using MakeLayoutA = Te::FrameLayoutFormat<LayoutA, Std::Int<C0_SIZE_A>>;
    using MakeLayoutB = Te::FrameLayoutFormat<LayoutB, Std::Int<C0_SIZE_B>>;
    using MakeLayoutScaleA = Te::FrameLayoutFormat<LayoutScaleA, Std::Int<C0_SIZE_SCALE>>;
    using MakeLayoutScaleB = Te::FrameLayoutFormat<LayoutScaleB, Std::Int<C0_SIZE_SCALE>>;
    using MakeLayoutC = Te::FrameLayoutFormat<LayoutC, Std::Int<C0_SIZE_C>>;
    using MakeLayoutBias = Te::FrameLayoutFormat<LayoutBias, Std::Int<C0_SIZE_BIAS>>;

    using ProblemShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using LayoutAType = decltype(MakeLayoutA{}(uint32_t{}, uint32_t{}));
    using LayoutBType = decltype(MakeLayoutB{}(uint32_t{}, uint32_t{}));
    using LayoutScaleAType = decltype(MakeLayoutScaleA{}(uint32_t{}, uint32_t{}));
    using LayoutScaleBType = decltype(MakeLayoutScaleB{}(uint32_t{}, uint32_t{}));
    using LayoutCType = decltype(MakeLayoutC{}(uint32_t{}, uint32_t{}));
    using LayoutBiasType = decltype(MakeLayoutBias{}(uint32_t{}, uint32_t{}));

    using BlockMmad = typename BlockMmadSelector<IsA8W4, Config>::type;
    using BlockPrologue =
        Std::conditional_t<IsA8W4, Blaze::Gemm::Prologue::BlockPrologue<DispatchPolicy, ElementA, ElementB>, void>;

    struct ProblemConfig {
        using KernelConfig = Config;
        static constexpr bool SOURCE_GMM1_INTERLEAVED = IsGmm1Interleaved;
        static constexpr bool IS_WAVE_FLAG_GRAINED = IsWaveFlagGrained;

        static __aicore__ inline typename BlockMmad::L1Params DefaultL1Params()
        {
            if constexpr (IsA8W4) {
                return typename BlockMmad::L1Params{.kL1 = L1_TILE_K, .scaleKL1 = 4096};
            } else {
                return typename BlockMmad::L1Params{
                    .kL1 = L1_TILE_K, .scaleKL1 = L1_TILE_K * SCALE_K_L1_RATE, .l1BufNum = 2};
            }
        }

        uint32_t m = 0;
        uint32_t n = 0;
        uint32_t k = 0;
        uint32_t outputN = 0;
        uint32_t schedulerN = 0;
        uint32_t blockNum = 0;
        uint32_t blockIdx = 0;
        uint32_t scaleK = 0;
        uint32_t tileM = 0;
        uint32_t swigluTileM = L1_TILE_M_256;
        typename BlockMmad::L1Params l1Params = DefaultL1Params();
    };

    struct LayoutBundle {
        LayoutAType a;
        LayoutBType b;
        LayoutScaleAType scaleA;
        LayoutScaleBType scaleB;
        LayoutCType c;
        LayoutBiasType bias;
    };

    __aicore__ static inline void FinalizeProblemConfig(ProblemConfig &config, const BlockJobContext &blockJob,
                                                        uint32_t gmm1TileM)
    {
        config.blockNum = blockJob.totalJobs;
        config.blockIdx = blockJob.jobIndex;
        config.scaleK = CeilDiv(config.k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        config.swigluTileM = TopkWeightsPrefetch ? L1_TILE_M_128 : gmm1TileM;
    }

    __aicore__ static inline ProblemConfig BuildGmm1ProblemConfig(const ProblemShape &problemShape,
                                                                  const BlockJobContext &blockJob,
                                                                  uint32_t gmm1TileM)
    {
        ProblemConfig config;
        config.m = Get<M_VALUE>(problemShape);
        config.n = Get<N_VALUE>(problemShape);
        config.k = Get<K_VALUE>(problemShape);
        config.outputN = config.n / SWIGLU_N_HALF;
        config.schedulerN = IsGmm1Interleaved ? config.n : config.outputN;
        config.tileM = gmm1TileM;
        FinalizeProblemConfig(config, blockJob, gmm1TileM);
        return config;
    }

    __aicore__ static inline ProblemConfig BuildGmm2ProblemConfig(const ProblemShape &problemShape,
                                                                  const BlockJobContext &blockJob,
                                                                  uint32_t gmm1TileM)
    {
        ProblemConfig config;
        config.m = Get<M_VALUE>(problemShape);
        config.n = Get<K_VALUE>(problemShape);
        config.k = Get<N_VALUE>(problemShape) / SWIGLU_N_HALF;
        config.outputN = config.n;
        config.schedulerN = config.outputN;
        config.tileM = L1_TILE_M_256;
        FinalizeProblemConfig(config, blockJob, gmm1TileM);
        return config;
    }

    __aicore__ static inline LayoutBundle BuildLayouts(const ProblemConfig &config)
    {
        LayoutBundle layouts;
        layouts.a = MakeLayoutA{}(config.m, config.k);
        layouts.b = MakeLayoutB{}(config.k, config.n);
        layouts.scaleA = MakeLayoutScaleA{}(config.m, config.scaleK);
        layouts.scaleB = MakeLayoutScaleB{}(config.scaleK, config.n);
        layouts.c = MakeLayoutC{}(config.m, config.n);
        layouts.bias = MakeLayoutBias{}(1U, config.n);
        return layouts;
    }
};

// 保存 GMM 执行所需的全部 tensor；当 bias 或 C 无实际存储时，调用方传入零地址占位 tensor。
template <typename Scheduler, typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB,
          typename TensorBias, typename TensorC>
struct GroupMatmulWorkSet {
    Scheduler &scheduler;
    TensorA &gmA;
    TensorB &gmB;
    TensorScaleA &gmScaleA;
    TensorScaleB &gmScaleB;
    TensorBias &gmBias;
    TensorC &gmC;
};

} // namespace GmmKernel
} // namespace MegaMoeImpl

#endif // MEGA_MOE_GMM_COMMON_H
