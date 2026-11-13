/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License for the specific language governing permissions and limitations under the License.
 */

/*!
 * \file all_gather_matmul_v3_apt.cpp
 * \brief Kernel entry for AllGatherMatmulV3 (ubmem/MTE window path only, MX-quant FP8/FP4)
 */

#include <kernel_operator.h>
#include "all_gather_matmul_v3_tiling_key.h"
#include "apace/tiling/matmul_tiling/quant_matmul_tiling_data.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#include "blaze/gemm/kernel/kernel_qbmm_mx.h"
#include "blaze/epilogue/block/block_epilogue_empty.h"
#include "apace/kernel/fusions/all_gather_quant_matmul/all_gather_quant_matmul_ubmem_impl.h"
#include "apace/block/mmad/qmm_mx_block_mmad_tile_k_wait_flag.h"

using namespace AscendC;
using namespace Apace;

template <bool TRANS_B>
__global__ __aicore__ void all_gather_matmul_v3(GM_ADDR context, GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR x1_scale,
                                                GM_ADDR x2_scale, GM_ADDR y, GM_ADDR gather_out, GM_ADDR amax_out,
                                                GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    (void)gather_out;
    (void)amax_out;
    (void)bias;
    (void)workspaceGM;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);

    REGISTER_TILING_DEFAULT(QuantMatmulTilingData);

    GET_TILING_DATA_WITH_STRUCT(QuantMatmulTilingData, tilingData, tilingGM);

    using TypeA = DTYPE_X1;
    using TypeB = DTYPE_X2;
    using TypeC = DTYPE_Y;
    using LayoutA = asc::te::nd_ext_layout_ptn;
    // 右矩阵布局: ND/NZ x 转置([N,K])/非转置([K,N]) 四种组合
    constexpr bool isX2Nz = (FORMAT_X2 == FORMAT_FRACTAL_NZ);
    using LayoutB = AscendC::Std::conditional_t<
        isX2Nz, AscendC::Std::conditional_t<TRANS_B, asc::te::zn_layout_ptn, asc::te::nz_layout_ptn>,
        AscendC::Std::conditional_t<TRANS_B, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>>;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    // apace路径下切k mmad模板(独立类, 每 AG_SYNC_INTERVAL 个 K 做一次跨核等待)
    using BlockMmad = Blaze::Gemm::Block::QmmMxBlockMmadTileKWaitFlag<0, false, TypeA, LayoutA, TypeB, LayoutB, TypeC,
                                                                      LayoutC, float, LayoutC>;
    using BlockScheduler =
        Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, 0, LayoutA, LayoutB, TypeA>;
    using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;
    using Impl = Apace::AllGatherQuantMatmulUbmemImpl<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;

    using KernelImpl = typename Impl::KernelImpl;

    typename Impl::Params params;
    params.matmulKernelParams.problemShape = ProblemShape{
        static_cast<int64_t>(tilingData.m), static_cast<int64_t>(tilingData.n), static_cast<int64_t>(tilingData.k), 1L};
    params.matmulKernelParams.mmadParams = typename BlockMmad::Params{nullptr, x2, y, nullptr, nullptr, x2_scale};
    params.matmulKernelParams.l1Params = typename BlockMmad::L1Params{
        static_cast<uint64_t>(tilingData.stepK) * tilingData.baseK, tilingData.scaleKL1, 2};
    params.matmulKernelParams.schParams = typename BlockScheduler::Params{tilingData.baseM,
                                                                          tilingData.baseN,
                                                                          tilingData.mTailTile,
                                                                          tilingData.nTailTile,
                                                                          tilingData.mBaseTailSplitCnt,
                                                                          tilingData.nBaseTailSplitCnt,
                                                                          tilingData.mTailMain,
                                                                          tilingData.nTailMain};
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
                                                                           tilingData.baseM,
                                                                           tilingData.baseN,
                                                                           tilingData.baseK,
                                                                           0,
                                                                           tilingData.dbL0c};
    params.hcclContext = context;
    params.tilingData = &tilingData;
    params.x1Addr = x1;
    params.x1ScaleAddr = x1_scale;

    Impl impl;
    impl(params);
}
