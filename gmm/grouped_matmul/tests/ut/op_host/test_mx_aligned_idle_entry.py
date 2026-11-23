# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Compile the real Aligned adapter with host stubs and exercise its lane guard."""

from pathlib import Path
import unittest
from mx_test_utils import run_cpp


class AlignedIdleEntryTest(unittest.TestCase):
    def test_entry_preserves_constant_and_global_tiling_sources(self):
        root = Path(__file__).resolve().parents[3]
        entry = (root / "op_kernel/grouped_matmul_apt.cpp").read_text(encoding="utf-8")
        branch = entry.split(
            "if constexpr (ALGORITHM_CATEGORY == WQGMM_MX_DYNAMIC_ALIGNED)", 1
        )[1]
        reads = branch[
            branch.index("#if defined(CONST_TILING)") : branch.index("// FixedTiles")
        ]
        utils = (root / "op_kernel/grouped_matmul_utils.h").read_text(encoding="utf-8")
        macros = utils[
            utils.index("#if defined(CONST_TILING)") : utils.index(
                "namespace GROUPED_MATMUL {"
            )
        ]
        adapter = (
            root / "op_kernel/arch35/quant_adaptive_sliding_window_templates/"
            "gmm_weight_quant_tensor_api_mx_kernel.h"
        ).read_text(encoding="utf-8")
        helper = adapter[
            adapter.index(
                "__aicore__ inline GroupedMatmulTilingData::GMMWeightQuantParam MakeFixedMxGroupParams("
            ) : adapter.index("namespace MxPipeline {")
        ]
        prelude = r"""
#include <cassert>
#include <cstdint>
#include <cstddef>
#define __aicore__
#define __gm__
#define V310_GMM_ANTI_QUANT
using std::size_t;
namespace GroupedMatmulTilingData {
struct GMMWeightQuantParam {uint64_t kSize,nSize,groupNum,groupListType,hasBias,coreNum,mainBlockCount;};
}
struct GMMWeightQuantTilingData {GroupedMatmulTilingData::GMMWeightQuantParam gmmWeightQuantParam;};
#define GET_TILING_DATA_MEMBER(type, member, name, addr) const auto name = reinterpret_cast<const type*>(addr)->member
namespace GROUPED_MATMUL {
"""
        body = (
            prelude
            + helper
            + "}\n"
            + macros
            + "\nauto read(uint8_t* tiling){\n"
            + reads
            + "return gp;}\n"
        )
        checks = r"""
int main(){
 GMMWeightQuantTilingData tiling{{7168,6144,14,1,0,32,24}};
 auto params=read(reinterpret_cast<uint8_t*>(&tiling));
 assert(params.kSize==7168&&params.nSize==6144&&params.groupNum==14);
 assert(params.groupListType==1&&params.hasBias==0);
}
"""
        for constant in [False, True]:
            with self.subTest(const_tiling=constant):
                run_cpp(("#define CONST_TILING\n" if constant else "") + body + checks)

    def test_fixed_group_projection_preserves_dynamic_dimensions(self):
        root = Path(__file__).resolve().parents[3]
        source = (
            root / "op_kernel/arch35/quant_adaptive_sliding_window_templates/"
            "gmm_weight_quant_tensor_api_mx_kernel.h"
        ).read_text(encoding="utf-8")
        helper = source[
            source.index(
                "__aicore__ inline GroupedMatmulTilingData::GMMWeightQuantParam MakeFixedMxGroupParams("
            ) : source.index("namespace MxPipeline {")
        ]
        run_cpp(
            r"""
#include <cassert>
#include <cstdint>
#include <initializer_list>
#define __aicore__
#define __gm__
using GM_ADDR = void*;
namespace GroupedMatmulTilingData {
struct GMMWeightQuantParam {
 uint64_t kSize,nSize,groupNum,groupListType,hasBias,coreNum,mainBlockCount;
 uint64_t groupType=0,singleX=0,singleWeight=0,singleY=0,mainBlockSize=0;
 uint64_t firstTailBlockCount=0,firstTailBlockSize=0,secondTailBlockCount=0,secondTailBlockSize=0,cubeNumBlocksN=0;
};}
"""
            + helper
            + r"""
int main(){
 for(uint64_t k:{256UL,512UL,7168UL})for(uint64_t n:{256UL,1024UL,6144UL})
 for(uint64_t e:{1UL,4UL,14UL,37UL}){
  GroupedMatmulTilingData::GMMWeightQuantParam raw{k,n,e,99,99,99,99};
  raw.groupType=2;raw.singleX=3;raw.singleWeight=4;raw.singleY=5;raw.mainBlockSize=128;
  raw.firstTailBlockCount=6;raw.firstTailBlockSize=64;raw.secondTailBlockCount=7;
  raw.secondTailBlockSize=32;raw.cubeNumBlocksN=31;
  const auto* address=reinterpret_cast<const int32_t*>(&raw);
  auto projected=MakeFixedMxGroupParams(address);
  assert(projected.kSize==k&&projected.nSize==n&&projected.groupNum==e);
  assert(projected.groupListType==1&&projected.hasBias==0);
  assert(projected.coreNum==0&&projected.mainBlockCount==0);
 }
}
"""
        )

    def test_fixed_entry_does_not_copy_unused_mmad_tiling(self):
        root = Path(__file__).resolve().parents[3]
        source = (root / "op_kernel/grouped_matmul_apt.cpp").read_text(encoding="utf-8")
        branch = source.split(
            "if constexpr (ALGORITHM_CATEGORY == WQGMM_MX_DYNAMIC_ALIGNED)", 1
        )[1]
        branch = branch.split("    if constexpr (W_TYPE", 1)[0]
        self.assertNotIn(
            "GET_TILING_DATA_MEMBER(GMMWeightQuantTilingData, mmTilingData", branch
        )
        self.assertIn(
            "mm.baseM = static_cast<uint32_t>(GROUPED_MATMUL::MxPipeline::ALIGNED_TILE_SIZE)",
            branch,
        )
        self.assertIn("DTYPE_Y, false, true>", branch)
        # The shared builder consumes no other TCubeTiling member.
        adapter = (
            root / "op_kernel/arch35/quant_adaptive_sliding_window_templates/"
            "gmm_weight_quant_tensor_api_mx_kernel.h"
        ).read_text(encoding="utf-8")
        builder = adapter.split("// Keep list interpretation", 1)[1].split(
            "template <MxBuffered::Route Pipeline", 1
        )[0]
        import re

        self.assertEqual(re.findall(r"\bmm->(\w+)", builder), ["baseM"])

    def test_inactive_lane_exits_before_parameter_builder(self):
        root = Path(__file__).resolve().parents[3]
        source = (
            root / "op_kernel/arch35/quant_adaptive_sliding_window_templates/"
            "gmm_weight_quant_tensor_api_mx_kernel.h"
        ).read_text(encoding="utf-8")
        adapter = source.split("namespace Aligned {", 1)[1].split(
            "} // namespace Aligned", 1
        )[0]
        self.assertLess(
            adapter.index("AscendC::GetSubBlockIdx()"),
            adapter.index("MakeMxKernelParams<"),
        )
        prelude = r"""
#include <cassert>
#include <cstdint>
#include <initializer_list>
#define __aicore__
#define ASCEND_IS_AIV (vectorCore)
using GM_ADDR = void*;
bool vectorCore;
unsigned lane, laneQueries, builderCalls, kernelCalls;
namespace AscendC {unsigned GetSubBlockIdx(){++laneQueries;return lane;}}
namespace Blaze {namespace Gemm {enum class MxPipelineMode{Aligned=2};}}
namespace GroupedMatmulTilingData {
struct GMMWeightQuantParam {uint64_t nSize=512;uint32_t groupListType=0,hasBias=0;};}
struct TCubeTiling {};
struct Scheduler {uint64_t a,b,c,d,e,f,g,h;int32_t i;uint64_t j;};
struct Params {Scheduler scheduler{};uint32_t groupListType=0,hasBias=0;};
struct Config {
 struct Consumer {static constexpr uint64_t L1_N=256,L1_M=256;};};
Params observed;
struct Kernel {void operator()(const Params& p){++kernelCalls;observed=p;}};
template<uint32_t,bool,class,class,class,class,class,bool>struct MxKernelTypes {
 using Kernel=::Kernel;using Config=::Config;};
template<class,bool,class... Args>Params MakeMxKernelParams(Args...){++builderCalls;return {};}
"""
        checks = r"""
template<bool Fixed>void check(bool vector,unsigned subBlock){
 vectorCore=vector;lane=subBlock;laneQueries=builderCalls=kernelCalls=0;
 GroupedMatmulTilingData::GMMWeightQuantParam base;TCubeTiling mm;
 GmmWeightQuantTensorApiMxKernel<int,int,int,int,int,false,Fixed>(
   nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,&base,&mm);
 bool active=!vector||subBlock==0;
 assert(builderCalls==unsigned(active)&&kernelCalls==unsigned(active));
 assert(laneQueries==unsigned(vector));
 if(active&&Fixed){assert(observed.groupListType==1&&observed.hasBias==0);
   assert(observed.scheduler.a==2&&observed.scheduler.g==32&&observed.scheduler.i==256);}
}
int main(){for(bool vector:{false,true})for(unsigned subBlock:{0U,1U}){
 check<true>(vector,subBlock);check<false>(vector,subBlock);}}
"""
        constants = root / "op_kernel/mx_a8w4_pipeline_common.h"
        run_cpp(
            f'#include "{constants.as_posix()}"\n'
            "using GROUPED_MATMUL::MxPipeline::ALIGNED_CORE_COUNT;\n"
            + prelude
            + adapter
            + checks
        )
