/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "graph/types.h"

// formulaic 路径在 arch35 被 GetTilingResult override 绕过，且符号对 executable 不可见；
// 将依赖 TU 编入本用例对象以直接驱动 AllGatherPlusMMV2（与 op_kernel UT include .cpp 惯例一致）。
#include "../../../../../common/op_host/op_tiling/hccl_performance.cpp"
#include "../../../../../common/op_host/op_tiling/matmul_performance.cpp"
#include "../../../../../common/op_host/op_tiling/hccl_formulaic_tiling.cpp"
#include "../../../../op_host/op_tiling/all_gather_matmul_v2_formulaic_tiling.cpp"

namespace {

mc2tiling::TilingArgs MakeFormulaicArgs(uint64_t m, uint64_t k, uint64_t n, uint32_t rankDim, uint64_t aicCoreNum = 30,
                                        ge::DataType dtype = ge::DT_FLOAT16)
{
    mc2tiling::TilingArgs args{};
    args.cmdType = mc2tiling::AicpuComType::HCCL_CMD_ALLGATHER;
    args.rankDim = rankDim;
    args.mValue = m;
    args.kValue = k;
    args.nValue = n;
    args.orgMValue = m;
    args.orgKValue = k;
    args.orgNValue = n;
    args.inputDtypeSize = ge::GetSizeByDataType(dtype);
    args.outputDtypeSize = args.inputDtypeSize;
    args.aicCoreNum = aicCoreNum;
    args.geAType = dtype;
    args.geBType = dtype;
    args.geCType = dtype;
    args.aType = matmul_tiling::DataType::DT_FLOAT16;
    args.bType = matmul_tiling::DataType::DT_FLOAT16;
    args.cType = matmul_tiling::DataType::DT_FLOAT16;
    if (dtype == ge::DT_FLOAT8_E4M3FN) {
        args.aType = matmul_tiling::DataType::DT_FLOAT8_E4M3FN;
        args.bType = matmul_tiling::DataType::DT_FLOAT8_E4M3FN;
    }
    return args;
}

CutResult RunFormulaicTiling(const mc2tiling::TilingArgs &args, uint32_t rankDim, NpuArch npuArch = Ops::Base::DAV_2201,
                             SocVersion_2201 soc = SocVersion_2201::SOC910_B)
{
    AllGatherPlusMMV2 tileFormulate(args, rankDim, KernelType::ALL_GATHER, npuArch, soc);
    tileFormulate.GetTiling();
    return tileFormulate.tilingM_.cutRes;
}

TEST(AllGatherMatmulV2FormulaicTilingTest, Dav3510A5Basic)
{
    auto args = MakeFormulaicArgs(512, 8192, 1280, 8);
    auto cut = RunFormulaicTiling(args, 8, Ops::Base::DAV_3510);
    EXPECT_GT(cut.totalTileCnt, 0U);
}

TEST(AllGatherMatmulV2FormulaicTilingTest, Soc91093CommFactor)
{
    auto args = MakeFormulaicArgs(512, 8192, 1280, 8);
    auto cut = RunFormulaicTiling(args, 8, Ops::Base::DAV_2201, SocVersion_2201::SOC910_93);
    EXPECT_GT(cut.totalTileCnt, 0U);
}

TEST(AllGatherMatmulV2FormulaicTilingTest, Soc910BComputeBoundRank8)
{
    auto args = MakeFormulaicArgs(4096, 8192, 1280, 8);
    auto cut = RunFormulaicTiling(args, 8, Ops::Base::DAV_2201, SocVersion_2201::SOC910_B);
    EXPECT_GT(cut.totalTileCnt, 0U);
}

TEST(AllGatherMatmulV2FormulaicTilingTest, Soc910BCommBoundStrongTp)
{
    auto args = MakeFormulaicArgs(256, 16384, 6144, 8);
    auto cut = RunFormulaicTiling(args, 8, Ops::Base::DAV_2201, SocVersion_2201::SOC910_B);
    EXPECT_GT(cut.totalTileCnt, 0U);
}

TEST(AllGatherMatmulV2FormulaicTilingTest, Soc910BReduceAlignLenTinyM)
{
    auto args = MakeFormulaicArgs(256, 16384, 6144, 8);
    auto cut = RunFormulaicTiling(args, 8, Ops::Base::DAV_2201, SocVersion_2201::SOC910_B);
    EXPECT_GT(cut.longTileLen, 0U);
}

TEST(AllGatherMatmulV2FormulaicTilingTest, Soc910BNoCutSmallShape)
{
    auto args = MakeFormulaicArgs(64, 512, 512, 8);
    auto cut = RunFormulaicTiling(args, 8, Ops::Base::DAV_2201, SocVersion_2201::SOC910_B);
    EXPECT_EQ(cut.numLongTile, 1U);
}

TEST(AllGatherMatmulV2FormulaicTilingTest, Soc910BHugeKDisablesNoCut)
{
    auto args = MakeFormulaicArgs(128, 40000, 1280, 8);
    auto cut = RunFormulaicTiling(args, 8, Ops::Base::DAV_2201, SocVersion_2201::SOC910_B);
    EXPECT_GT(cut.totalTileCnt, 0U);
}

TEST(AllGatherMatmulV2FormulaicTilingTest, Soc910BComputeBoundRank2SmallFront)
{
    auto args = MakeFormulaicArgs(4096, 1024, 1024, 2);
    auto cut = RunFormulaicTiling(args, 2, Ops::Base::DAV_2201, SocVersion_2201::SOC910_B);
    EXPECT_GT(cut.totalTileCnt, 0U);
}

TEST(AllGatherMatmulV2FormulaicTilingTest, Soc910BBwGrowthByShapeLargeNK)
{
    auto args = MakeFormulaicArgs(512, 10000, 6000, 8);
    auto cut = RunFormulaicTiling(args, 8, Ops::Base::DAV_2201, SocVersion_2201::SOC910_B);
    EXPECT_GT(cut.totalTileCnt, 0U);
}

TEST(AllGatherMatmulV2FormulaicTilingTest, Soc910BMedianMCommGrowth)
{
    auto args = MakeFormulaicArgs(3000, 10000, 6000, 8);
    auto cut = RunFormulaicTiling(args, 8, Ops::Base::DAV_2201, SocVersion_2201::SOC910_B);
    EXPECT_GT(cut.totalTileCnt, 0U);
}

TEST(AllGatherMatmulV2FormulaicTilingTest, Soc910BBwGrowthByUtilLargeShape)
{
    auto args = MakeFormulaicArgs(8192, 8192, 8192, 8);
    auto cut = RunFormulaicTiling(args, 8, Ops::Base::DAV_2201, SocVersion_2201::SOC910_B);
    EXPECT_GT(cut.totalTileCnt, 0U);
}

TEST(AllGatherMatmulV2FormulaicTilingTest, Dav3510QuantDtype)
{
    auto args = MakeFormulaicArgs(512, 8192, 1280, 8, 32, ge::DT_FLOAT8_E4M3FN);
    auto cut = RunFormulaicTiling(args, 8, Ops::Base::DAV_3510);
    EXPECT_GT(cut.totalTileCnt, 0U);
}

TEST(AllGatherMatmulV2FormulaicTilingTest, Soc910BReduceAlignLenBranch)
{
    // m<=512 且 n>2048，同时 strongTpBound 触发 allowMoreCuts → SetAlignLength/2
    auto args = MakeFormulaicArgs(128, 16384, 6144, 8);
    auto cut = RunFormulaicTiling(args, 8, Ops::Base::DAV_2201, SocVersion_2201::SOC910_B);
    EXPECT_GT(cut.longTileLen, 0U);
}

TEST(AllGatherMatmulV2FormulaicTilingTest, Soc910BBwGrowthByShapeOnly)
{
    // cubeUtil>=0.85 时走 bwGrowthByShape 分支（k>=8192 且 n>5120）
    auto args = MakeFormulaicArgs(512, 9000, 6000, 8);
    RunFormulaicTiling(args, 8, Ops::Base::DAV_2201, SocVersion_2201::SOC910_B);
}

TEST(AllGatherMatmulV2FormulaicTilingTest, Soc910BRank2ShortAtEndBalancing)
{
    // rank=2 + compute bound → ShortAtEndCalcBoundBalancing
    auto args = MakeFormulaicArgs(8192, 4096, 4096, 2);
    auto cut = RunFormulaicTiling(args, 2, Ops::Base::DAV_2201, SocVersion_2201::SOC910_B);
    EXPECT_GT(cut.totalTileCnt, 0U);
}

TEST(AllGatherMatmulV2FormulaicTilingTest, Soc910BRank2SmallDimAlignUp)
{
    // rank=2 + shortTileAtBack + 小 N/K → smallDimAlignUp
    auto args = MakeFormulaicArgs(8192, 1024, 1024, 2);
    auto cut = RunFormulaicTiling(args, 2, Ops::Base::DAV_2201, SocVersion_2201::SOC910_B);
    EXPECT_GT(cut.totalTileCnt, 0U);
}

TEST(AllGatherMatmulV2FormulaicTilingTest, Soc910BRank2ComputeBoundSmallNK)
{
    // rank=2 + 强计算 bound + 小 K/N：ShortAtEndCalcBoundBalancing + smallDimAlignUp
    auto args = MakeFormulaicArgs(32768, 1024, 1024, 2);
    RunFormulaicTiling(args, 2, Ops::Base::DAV_2201, SocVersion_2201::SOC910_B);
}

TEST(AllGatherMatmulV2FormulaicTilingTest, Soc910BRank2MidMComputeBound)
{
    auto args = MakeFormulaicArgs(2048, 512, 512, 2);
    RunFormulaicTiling(args, 2, Ops::Base::DAV_2201, SocVersion_2201::SOC910_B);
}

} // namespace
