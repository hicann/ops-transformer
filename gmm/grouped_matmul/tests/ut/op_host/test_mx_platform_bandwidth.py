# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Execute the production bandwidth calculation and count platform resource reads."""

from pathlib import Path
import unittest

from mx_test_utils import run_cpp


class PlatformBandwidthTest(unittest.TestCase):
    def test_merged_mainline_guard_uses_cached_bandwidth(self):
        folder = Path(__file__).resolve().parents[3] / "op_host/op_tiling/arch35"
        source = (folder / "grouped_weight_quant_batch_matmul_tiling.cpp").read_text(
            encoding="utf-8"
        )
        header = (folder / "grouped_weight_quant_batch_matmul_tiling.h").read_text(
            encoding="utf-8"
        )
        for name in ["GetHbmBW", "GetL2BW", "GetCoreFreq", "HasMxPipelineBandwidth"]:
            self.assertNotRegex(source + header, rf"\b{name}\s*\(")
        start = source.index(
            "bool GroupedWeightQuantBatchMatmulTiling::EnableBandWidthSelect("
        )
        method = source[
            start : source.index(
                "bool GroupedWeightQuantBatchMatmulTiling::CalcResplitTiling(", start
            )
        ]
        run_cpp(
            r"""
#include <cassert>
constexpr double BANDWIDTH_THRESHOLD=2.0;
struct GroupedWeightQuantBatchMatmulTiling {
 struct {double hbm=0.0;} platformBandwidth_;
 bool EnableBandWidthSelect() const;
};
"""
            + method
            + r"""
int main(){
 GroupedWeightQuantBatchMatmulTiling t;
 assert(!t.EnableBandWidthSelect());
 t.platformBandwidth_.hbm=1.99999;assert(!t.EnableBandWidthSelect());
 t.platformBandwidth_.hbm=2.0;assert(!t.EnableBandWidthSelect());
 t.platformBandwidth_.hbm=2.00001;assert(t.EnableBandWidthSelect());
}
"""
        )

    def test_shared_model_and_invalid_platform_resources(self):
        source = (
            Path(__file__).resolve().parents[3]
            / "op_host/op_tiling/arch35/grouped_weight_quant_batch_matmul_tiling.cpp"
        ).read_text(encoding="utf-8")
        parser = source[
            source.index("double StrToDoubleWithDefault(") : source.index(
                "} // namespace"
            )
        ]
        start = source.index("GroupedWeightQuantBatchMatmulTiling::PlatformBandwidth")
        method = source[
            start : source.index(
                "bool GroupedWeightQuantBatchMatmulTiling::CalcMxA8W4BandwidthResplit(",
                start,
            )
        ]
        prelude = r"""
#include <string>
#include <map>
#include <cmath>
#include <stdexcept>
#include <cassert>
namespace fe {struct PlatFormInfos {
 std::map<std::string,std::string> values;
 std::map<std::string,unsigned> reads;
 void GetPlatformRes(const char*,const char* key,std::string& value){++reads[key];value=values[key];}
};}
constexpr double DEFAULT_AICORE_FREQ_MHZ=1650.0,DEFAULT_AICORE_COUNT=32.0;
constexpr double DEFAULT_DDR_RATE=31.0,DEFAULT_L2_RATE=100.0,MHZ_PER_GHZ=1000.0,KB_SIZE=1024.0;
struct GroupedWeightQuantBatchMatmulTiling {
 struct PlatformBandwidth {double hbm=0.0,l2=0.0;bool supportsMxPipeline=false;};
 static PlatformBandwidth GetPlatformBandwidth(fe::PlatFormInfos*);
};
"""
        checks = r"""
int main(){
 using T=GroupedWeightQuantBatchMatmulTiling;
 assert(!T::GetPlatformBandwidth(nullptr).supportsMxPipeline);
 fe::PlatFormInfos p;
 auto result=T::GetPlatformBandwidth(&p);
 assert(!result.supportsMxPipeline);
 assert(result.hbm==1650.0*32.0*31.0/1000.0/1024.0);
 assert(result.l2==1650.0*32.0*100.0/1000.0/1024.0);
 p.values={{"cube_freq","1650"},{"ai_core_cnt","32"},{"ddr_rate","80"},{"l2_rate","100"}};
 p.reads.clear();result=T::GetPlatformBandwidth(&p);
 assert(result.supportsMxPipeline);
 for(auto key:{"cube_freq","ai_core_cnt","ddr_rate","l2_rate"})assert(p.reads[key]==1);
 p.values["cube_freq"]="1000";p.values["ddr_rate"]="64";
 assert(T::GetPlatformBandwidth(&p).supportsMxPipeline);
 p.values["ddr_rate"]="63.99999";assert(!T::GetPlatformBandwidth(&p).supportsMxPipeline);
 for(auto key:{"cube_freq","ai_core_cnt","ddr_rate"})for(auto bad:{"","0","-1","nan","inf","1e9999","abc","31junk"}){
  p.values={{"cube_freq","1650"},{"ai_core_cnt","32"},{"ddr_rate","80"}};p.values[key]=bad;
  assert(!T::GetPlatformBandwidth(&p).supportsMxPipeline);
 }
 // A subsequent platform must not reuse a process-wide first-device cache.
 fe::PlatFormInfos other;other.values={{"cube_freq","1000"},{"ai_core_cnt","32"},{"ddr_rate","32"}};
 assert(!T::GetPlatformBandwidth(&other).supportsMxPipeline);
 p.values={{"cube_freq","1e308"},{"ai_core_cnt","1e308"},{"ddr_rate","1e308"}};
 assert(!T::GetPlatformBandwidth(&p).supportsMxPipeline);
}
"""
        run_cpp(prelude + parser + method + checks)


if __name__ == "__main__":
    unittest.main()
