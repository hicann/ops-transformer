/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

/*!
 * \file quant_matmul_reduce_scatter.cpp
 * \brief Kernel entry: QuantMatmul + ReduceScatter fusion
 *
 * Flow (INTRA per-sub-block sync, apace/utils/sync.h; flags: fwd=0, ack=1, vec1 domain = id+16):
 *   AIC: Matmul -> L0C->UB fixpipe (DualDst SplitM, FP32, no on-the-fly cast)
 *        -> Sync::NotifyVector(0); Sync::WaitForVector(1) before next tile's fixpipe
 *   AIV: Sync::WaitForCube(0) -> BlockEpilogueAlltoAll:
 *        cast FP32->BF16 (VF) -> Sync::NotifyCube(1 ack) -> AllToAll to peer windows
 *   AIV: SyncAll -> status flag barrier -> ReduceAdd (unchanged)
 *
 * TILING_KEY (TPL 模板实参 TRANSB，见 quant_matmul_reduce_scatter_tiling_key.h):
 *   false: x2 非转置 [K,N]（transB=false）
 *   true : x2 转置   [N,K]（transB=true）
 *   x2 format（ND/NZ）不进 tiling key：由 opc 按 def 的 Format 组合逐份编译注入 FORMAT_X2 宏，
 *   kernel 侧编译期 FORMAT_X2 == FORMAT_FRACTAL_NZ 判断
 *
 * DTYPE (dtype 泛化):
 *   TypeA/TypeB/TypeC 取框架按 def 的 DataType 组合逐份编译时注入的
 *   DTYPE_X1/DTYPE_X2/DTYPE_Y 宏（opc 机制，非 TPL）：
 *     DT_FLOAT8_E4M3FN -> fp8_e4m3fn_t, DT_FLOAT8_E5M2 -> fp8_e5m2_t,
 *     DT_FLOAT4_E2M1   -> fp4x2_e2m1_t (2 元素/字节打包), DT_BF16 -> bfloat16_t
 *   def 中 fp8 配对均可配 x2 ND/NZ，fp4+fp4 仅配 ND；y 固定 DT_BF16。
 */

#include "basic_api/kernel_basic_intf.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
// 注：不包含 blaze/gemm/kernel/kernel_universal.h——新版 ops-tensor 会经它引入
// stock kernel_qbmm_mx.h 的 GemmUniversal 特化，与 apace 的 Mc2QbmmMxUniversal 冲突
#include "apace/block/epilogue/block_epilogue_all_to_all.h"
#include "apace/kernel/fusions/quant_matmul_reduce_scatter/quant_matmul_reduce_scatter_impl.h"
#include "apace/tiling/matmul_tiling/quant_matmul_tiling_data.h"
#include "quant_matmul_reduce_scatter_tiling_key.h"

using namespace AscendC;
using namespace Blaze::Gemm;

template <bool TRANSB>
__aicore__ inline void QuantMatmulReduceScatterKernel(GM_ADDR context, GM_ADDR x1, GM_ADDR x2, GM_ADDR x1Scale,
                                                      GM_ADDR x2Scale, GM_ADDR y, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    using TypeA = DTYPE_X1;
    using TypeB = DTYPE_X2;
    using TypeC = DTYPE_Y;
    using MatmulOutType = float; // L0C->UB fixpipe 落地类型（UB 侧无随路 cast）
    using BiasType = float;

    using LayoutA = AscendC::Te::NDExtLayoutPtn;
    // 右矩阵布局: ND/NZ x 转置([N,K])/非转置([K,N]) 四种组合
    constexpr bool isX2Nz = (FORMAT_X2 == FORMAT_FRACTAL_NZ);
    using LayoutB = AscendC::Std::conditional_t<
        isX2Nz, AscendC::Std::conditional_t<TRANSB, AscendC::Te::ZNLayoutPtn, AscendC::Te::NZLayoutPtn>,
        AscendC::Std::conditional_t<TRANSB, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn>>;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;

    // L0C2UB_MODE_DUAL_DST_SPLIT_M: L0C 结果经 fixpipe 直写两个 AIV 子核的 UB（M 对半）
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<0, 0, Blaze::Gemm::KernelMmadWithScaleMx,
                                                          Blaze::Gemm::L0C2UB_MODE_DUAL_DST_SPLIT_M>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, TypeA, LayoutA, TypeB, LayoutB, MatmulOutType,
                                                    LayoutC, BiasType, LayoutC>;
    using BlockScheduler =
        Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>, 0,
                                                             LayoutA, LayoutB, TypeA>;
    using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueAlltoAll<TypeC, LayoutC>;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

    using Impl = Apace::QuantMatmulReduceScatterImpl<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using KernelImpl = typename Impl::KernelImpl;

    __gm__ QuantMatmulTilingData* tilingData = (__gm__ QuantMatmulTilingData*)tilingGM;

    const uint64_t M = tilingData->m;
    const uint64_t N = tilingData->n;
    const uint64_t K = tilingData->k;

    (void)workspaceGM; // L0C->UB，中间结果不再落 GM

    typename Impl::Params params;
    params.matmulKernelParams.problemShape =
        ProblemShape{static_cast<int64_t>(M), static_cast<int64_t>(N), static_cast<int64_t>(K), 1L};
    params.matmulKernelParams.mmadParams = typename BlockMmad::Params{x1, x2, nullptr, nullptr, x1Scale, x2Scale};
    params.matmulKernelParams.l1Params = typename BlockMmad::L1Params{
        static_cast<uint64_t>(tilingData->stepK) * tilingData->baseK, tilingData->scaleKL1, 2};
    params.matmulKernelParams.schParams = typename BlockScheduler::Params{tilingData->baseM,
                                                                          tilingData->baseN,
                                                                          tilingData->mTailTile,
                                                                          tilingData->nTailTile,
                                                                          tilingData->mBaseTailSplitCnt,
                                                                          tilingData->nBaseTailSplitCnt,
                                                                          tilingData->mTailMain,
                                                                          tilingData->nTailMain};
    params.matmulKernelParams.qbmmParams = typename KernelImpl::QBMMTiling{1,
                                                                           1,
                                                                           1,
                                                                           1,
                                                                           1,
                                                                           1,
                                                                           1,
                                                                           1,
                                                                           1,
                                                                           1,
                                                                           1,
                                                                           1,
                                                                           0,
                                                                           tilingData->baseM,
                                                                           tilingData->baseN,
                                                                           tilingData->baseK,
                                                                           0,
                                                                           tilingData->dbL0c};
    params.matmulKernelParams.epilogueParams = typename BlockEpilogue::Params{M, N, 0, context};

    params.yGM = y;
    params.ubmemContext = context;

    Impl impl;
    impl(params);
}

template <bool TRANSB>
__global__ __aicore__ void quant_matmul_reduce_scatter(GM_ADDR context, GM_ADDR x1, GM_ADDR x2, GM_ADDR x1Scale,
                                                       GM_ADDR x2Scale, GM_ADDR y, GM_ADDR workspaceGM,
                                                       GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(QuantMatmulTilingData);
    GET_TILING_DATA_WITH_STRUCT(QuantMatmulTilingData, tilingData, tilingGM);

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    QuantMatmulReduceScatterKernel<TRANSB>(context, x1, x2, x1Scale, x2Scale, y, workspaceGM, tilingGM);
}
