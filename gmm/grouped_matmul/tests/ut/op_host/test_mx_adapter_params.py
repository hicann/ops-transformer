# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check shared adapter wiring and compile its real parameter builder with host stubs."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


FOLDER = Path(__file__).resolve().parents[3] / (
    "op_kernel/arch35/quant_adaptive_sliding_window_templates"
)


class AdapterParamsTest(unittest.TestCase):
    def test_all_adapters_share_parameter_builder(self):
        source = (FOLDER / "gmm_weight_quant_tensor_api_mx_kernel.h").read_text(
            encoding="utf-8"
        )
        self.assertEqual(
            source.count("inline typename KernelImpl::Params MakeMxKernelParams("), 1
        )
        self.assertEqual(source.count("MakeMxKernelParams<KernelImpl,"), 2)
        adapters = source.split("template <MxBuffered::Route Pipeline", 1)[1].split(
            "// The original entry uses", 1
        )[0]
        self.assertNotIn("GetTensorAddr<", adapters)
        self.assertIn(
            "params.scheduler = {nb, Config::Consumer::L1_N, 0, 0, 0, 0, ALIGNED_CORE_COUNT,",
            " ".join(source.split()),
        )
        self.assertNotIn("RunSssUnfolded<true>", source)

    def test_shared_builder_has_no_pipeline_or_sync_decisions(self):
        source = (FOLDER / "gmm_weight_quant_tensor_api_mx_kernel.h").read_text(
            encoding="utf-8"
        )
        builder = source.split("// Keep list interpretation", 1)[1].split(
            "template <MxBuffered::Route Pipeline", 1
        )[0]
        for forbidden in [
            "WaitFlag",
            "SetFlag",
            "MxBuffered::Select",
            "FixedTiles",
            "GetSubBlockIdx",
        ]:
            self.assertNotIn(forbidden, builder)
        self.assertIn("if constexpr (!IsSingleMultiSingle)", builder)
        self.assertIn("if (base->hasBias != 0)", builder)

    def test_sss_sms_and_bias_addresses(self):
        compiler = shutil.which("g++") or shutil.which("clang++") or shutil.which("cl")
        if not compiler:
            self.skipTest("Requires a C++17 host compiler; no device SDK needed")
        source = (FOLDER / "gmm_weight_quant_tensor_api_mx_kernel.h").read_text(
            encoding="utf-8"
        )
        builder = (
            "template <class KernelImpl, bool IsSingleMultiSingle>"
            + source.split("template <class KernelImpl, bool IsSingleMultiSingle>", 1)[
                1
            ].split("template <MxBuffered::Route Pipeline", 1)[0]
        )
        prelude = r"""
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <tuple>
#define __aicore__
#define __gm__
using GM_ADDR = unsigned char*;
namespace asc {namespace te {using std::make_tuple;
template<class... T> auto make_shape(T... v){return std::make_tuple(v...);}}}
namespace GroupedMatmulTilingData {
struct GMMWeightQuantParam {
    uint64_t kSize=2048,nSize=576,groupNum=3;
    uint64_t mainBlockCount=2,mainBlockSize=256,firstTailBlockCount=1,firstTailBlockSize=64;
    uint64_t secondTailBlockCount=0,secondTailBlockSize=32,coreNum=32,cubeNumBlocksN=3;
    uint32_t groupListType=1,hasBias=0;
};}
struct TCubeTiling {int32_t baseM=256;};
struct List {GM_ADDR address; int reads=0;};
template<class T> T* GetTensorAddr(int,GM_ADDR p){
    assert(p); auto& list=*reinterpret_cast<List*>(p);++list.reads;
    return reinterpret_cast<T*>(list.address);
}
struct Kernel {
    using AType=uint8_t;using BType=uint8_t;using ScaleBType=uint8_t;
    using BiasType=uint16_t;using CType=uint16_t;
    struct Params {
        std::tuple<unsigned long,uint64_t,uint64_t,uint64_t> problemShape;
        struct {GM_ADDR aGmAddr,scaleAGmAddr,scaleBGmAddr,biasGmAddr,cGmAddr;}mmad;
        struct {uint64_t mainBlockCount,mainBlockSize,firstTailBlockCount,firstTailBlockSize;
            uint64_t secondTailBlockCount,secondTailBlockSize,coreNum,cubeNumBlocksN;
            int32_t baseM;uint64_t nSize;}scheduler;
        struct {uint8_t* ptrB;}prologue;
        GM_ADDR ptrGroupList;uint32_t groupListType,hasBias;
    };
};
"""
        main = r"""
template<bool Sms> void check(bool biasPresent,uint32_t groupType){
    unsigned char data[7][16]{};
    List x{data[0]},w{data[1]},scale{data[2]},bias{data[3]},y{data[4]};
    auto addr=[](List& p){return reinterpret_cast<GM_ADDR>(&p);};
    GroupedMatmulTilingData::GMMWeightQuantParam base;
    base.hasBias=biasPresent;base.groupListType=groupType;
    TCubeTiling mm;
    GM_ADDR biasArg=biasPresent?addr(bias):nullptr;
    auto p=MakeMxKernelParams<Kernel,Sms>(addr(x),addr(w),addr(scale),biasArg,
                                         data[5],data[6],addr(y),&base,&mm);
    assert(p.mmad.aGmAddr==data[0] && p.mmad.cGmAddr==data[4]);
    assert(p.mmad.scaleAGmAddr==data[6] && p.ptrGroupList==data[5]);
    assert(p.prologue.ptrB==(Sms?addr(w):data[1]));
    assert(p.mmad.scaleBGmAddr==(Sms?addr(scale):data[2]));
    assert(p.mmad.biasGmAddr==(biasPresent?(Sms?addr(bias):data[3]):nullptr));
    assert(x.reads==1 && y.reads==1 && w.reads==(Sms?0:1));
    assert(scale.reads==(Sms?0:1) && bias.reads==(!Sms && biasPresent?1:0));
    assert(p.groupListType==groupType && p.hasBias==biasPresent);
    assert(p.problemShape==std::make_tuple(0UL,base.kSize,base.nSize,base.groupNum));
    assert(p.scheduler.mainBlockCount==2 && p.scheduler.mainBlockSize==256);
    assert(p.scheduler.firstTailBlockCount==1 && p.scheduler.firstTailBlockSize==64);
    assert(p.scheduler.secondTailBlockCount==0 && p.scheduler.secondTailBlockSize==32);
    assert(p.scheduler.coreNum==32 && p.scheduler.cubeNumBlocksN==3);
    assert(p.scheduler.baseM==256 && p.scheduler.nSize==576);
}
int main(){
    for(bool bias:{false,true})for(uint32_t type:{0U,1U}){
        check<false>(bias,type);check<true>(bias,type);
    }
}
"""
        with tempfile.TemporaryDirectory() as tmp:
            cpp = Path(tmp) / "adapter.cpp"
            exe = Path(tmp) / "adapter.exe"
            cpp.write_text(prelude + builder + main, encoding="utf-8")
            if Path(compiler).stem.lower() == "cl":
                command = [
                    compiler,
                    "/nologo",
                    "/std:c++17",
                    "/EHsc",
                    str(cpp),
                    "/Fe:" + str(exe),
                    "/Fo:" + str(Path(tmp) / "adapter.obj"),
                ]
            else:
                command = [
                    compiler,
                    "-std=c++17",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    str(cpp),
                    "-o",
                    str(exe),
                ]
            compiled = subprocess.run(
                command, capture_output=True, text=True, errors="replace"
            )
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            subprocess.run(
                [str(exe)], check=True, capture_output=True, text=True, errors="replace"
            )


if __name__ == "__main__":
    unittest.main()
