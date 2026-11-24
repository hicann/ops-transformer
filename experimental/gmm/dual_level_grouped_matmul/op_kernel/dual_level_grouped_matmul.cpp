// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "dual_level_grouped_matmul_tiling_data.h"
#include "blaze/gemm/kernel/kernel_qgmm_mx.h"
#include "blaze/gemm/block/block_mmad_qgmm_mx.h"
#include "blaze/epilogue/block/block_epilogue_dual_level_scale.h"

// The generated DFX metadata references this name at namespace scope.
using DualLevelGroupedMatmulTilingData = DualLevelGroupedMatmul::DualLevelGroupedMatmulTilingData;

namespace {
using namespace DualLevelGroupedMatmul;
template <class Output, bool TransposeWeight>
__aicore__ inline void RunDualLevelGroupedMatmul(GM_ADDR x, GM_ADDR a0, GM_ADDR a1, GM_ADDR weight, GM_ADDR b0,
                                                 GM_ADDR b1, GM_ADDR groups, GM_ADDR bias, GM_ADDR y,
                                                 const DualLevelGroupedMatmulTilingData& t)
{
    if (t.version != TILING_DATA_VERSION) {
        return;
    }
    using Nd = asc::te::nd_ext_layout_ptn;
    using WeightLayout = AscendC::Std::conditional_t<TransposeWeight, asc::te::zn_layout_ptn, asc::te::nz_layout_ptn>;
    using Policy = Blaze::Gemm::GroupedMatmulWithScaleMx<0, false, Blaze::Gemm::KernelGroupedMmadWithDualLevelScaleMx>;
    using Mmad =
        Blaze::Gemm::Block::BlockMmad<Policy, fp4x2_e2m1_t, Nd, fp4x2_e2m1_t, WeightLayout, Output, Nd, float, Nd>;
    using Epilogue = Blaze::Epilogue::Block::BlockEpilogueDualLevelScale<Output>;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<typename Mmad::ProblemShape, Mmad, Epilogue,
                                                      Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit>;
    typename Kernel::Params params{};
    params.problemShape = {t.m, t.n, t.k, 0};
    params.mmadParams.aGmAddr = x;
    params.mmadParams.bGmAddr = weight;
    params.mmadParams.cGmAddr = y;
    params.mmadParams.biasGmAddr = bias;
    params.mmadParams.scaleAGmAddr = a1;
    params.mmadParams.scaleBGmAddr = b1;
    params.epilogueParams.level0ScaleAGmAddr = a0;
    params.epilogueParams.level0ScaleBGmAddr = b0;
    params.groupListGmAddr = groups;
    auto& gmm = params.gmmParams;
    gmm.groupNum = t.experts;
    gmm.m = t.m;
    gmm.n = t.n;
    gmm.k = t.k;
    gmm.baseM = t.baseM;
    gmm.baseN = t.baseN;
    gmm.baseK = t.level0Size;
    gmm.kAL1 = t.level0Size;
    gmm.kBL1 = t.level0Size;
    gmm.scaleKAL1 = LEVEL1_SCALE_L1_WINDOW_K;
    gmm.scaleKBL1 = gmm.scaleKAL1;
    gmm.isBias = t.hasBias;
    gmm.groupType = 0;
    gmm.groupListType = t.groupListType;
    gmm.singleW = 1;
    Kernel kernel;
    kernel(params);
}
} // namespace

extern "C" __global__ __aicore__ void dual_level_grouped_matmul(GM_ADDR x, GM_ADDR a0, GM_ADDR a1, GM_ADDR weight,
                                                                GM_ADDR b0, GM_ADDR b1, GM_ADDR groups, GM_ADDR bias,
                                                                GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(DualLevelGroupedMatmulTilingData);
    DualLevelGroupedMatmulTilingData td;
    // Read the shared fixed-width POD directly; this entry also compiles outside
    // the generated op-build wrapper which supplies GET_TILING_DATA_WITH_STRUCT.
    auto* words = reinterpret_cast<uint64_t*>(&td);
    auto* gmWords = reinterpret_cast<__gm__ uint64_t*>(tiling);
    for (uint32_t i = 0; i < sizeof(DualLevelGroupedMatmulTilingData) / sizeof(uint64_t); ++i) {
        words[i] = gmWords[i];
    }
    AscendC::InitSocState();
#define RUN_DUAL_LEVEL(Output, Trans) \
    RunDualLevelGroupedMatmul<Output, Trans>(x, a0, a1, weight, b0, b1, groups, bias, y, td)
    if (TILING_KEY_IS(DLGMM_FP16_TILING_KEY)) {
        RUN_DUAL_LEVEL(half, false);
    }
    if (TILING_KEY_IS(DLGMM_FP16_TRANSPOSE_TILING_KEY)) {
        RUN_DUAL_LEVEL(half, true);
    }
    if (TILING_KEY_IS(DLGMM_BF16_TILING_KEY)) {
        RUN_DUAL_LEVEL(bfloat16_t, false);
    }
    if (TILING_KEY_IS(DLGMM_BF16_TRANSPOSE_TILING_KEY)) {
        RUN_DUAL_LEVEL(bfloat16_t, true);
    }
#undef RUN_DUAL_LEVEL
}
