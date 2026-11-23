# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""The original route must preserve the original host partitions byte for byte."""

from pathlib import Path
import unittest
from mx_test_utils import run_cpp


class OriginalPartitionTest(unittest.TestCase):
    def test_original_route_preserves_partitions(self):
        root = Path(__file__).resolve().parents[3]
        source = (
            root
            / "op_host/op_tiling/arch35/grouped_weight_quant_batch_matmul_tiling.cpp"
        ).read_text(encoding="utf-8")
        start = source.index(
            "uint64_t GroupedWeightQuantBatchMatmulTiling::SelectMxPipelineKey("
        )
        method = source[
            start : source.index(
                "\nbool GroupedWeightQuantBatchMatmulTiling::SetCustomParam(", start
            )
        ]
        policy = root / "op_host/op_tiling/arch35/mx_a8w4_pipeline_policy.h"
        run_cpp(
            '#include "'
            + policy.as_posix()
            + '"\n'
            + r"""
#include <cassert>
#include <algorithm>
#include <cstring>
#include <initializer_list>
namespace ge {enum DataType {DT_FLOAT8_E4M3FN,DT_FLOAT4_E2M1,DT_FLOAT8_E8M0,DT_BF16};}
struct Desc {ge::DataType GetDataType()const{return ge::DT_BF16;}};
bool bandwidth_ok=true,platform_present=true;
namespace fe {struct PlatFormInfos {}; }
namespace gert {struct TilingContext {Desc desc;fe::PlatFormInfos platform;Desc* GetOutputDesc(int){return &desc;}fe::PlatFormInfos* GetPlatformInfo(){return platform_present?&platform:nullptr;}};}
constexpr bool GROUPED_MATMUL_ENABLE_MX_PIPELINE=true;
constexpr uint8_t DECIMAL=10;
struct GroupedWeightQuantBatchMatmulTiling {
 bool hasBias_=false;int64_t groupListType_=1,groupType_=0;
 uint64_t coreNum_=32,mSize_=0,groupNum_=0,kSize_=0,nSize_=0;
 ge::DataType xDType_=ge::DT_FLOAT8_E4M3FN,weightDtype_=ge::DT_FLOAT4_E2M1,antiquantScaleDtype_=ge::DT_FLOAT8_E8M0;
 struct Data {struct Groups {uint64_t mainBlockSize,mainBlockCount,firstTailBlockCount,secondTailBlockCount;
 uint64_t firstTailBlockSize,secondTailBlockSize,cubeNumBlocksN,reserved;}gmmWeightQuantParam;
 struct {uint64_t baseN;}mmTilingData;}tilingData_;
 struct Key {uint8_t algorithm=0;uint64_t GenTilingKey(){return 74530817ULL|(uint64_t(algorithm/10)<<29);}}tilingKeyConfig_;
 struct {bool supportsMxPipeline=false;}platformBandwidth_;
 uint64_t SelectMxPipelineKey(gert::TilingContext*,uint64_t);
};
struct TilingKeyConfigure {static uint64_t GetOriginalMxKey(){return 74530817ULL;}};
"""
            + method
            + r"""
int main(){
 gert::TilingContext ctx;
 // Rejected resource profiles must preserve all original tiling fields and key bits.
 for(bool available:{false,true})for(bool capable:{false,true})
 for(uint64_t cores:{28ULL,32ULL,36ULL})for(uint64_t originalKey:{74530817ULL,91308033ULL}) {
  platform_present=available;bandwidth_ok=capable;
  if(available&&capable&&cores==32&&originalKey==74530817ULL)continue;
  GroupedWeightQuantBatchMatmulTiling t;t.coreNum_=cores;
  t.platformBandwidth_.supportsMxPipeline=available&&capable;
  t.mSize_=32768;t.groupNum_=4;t.kSize_=7168;t.nSize_=6144;
  std::memset(&t.tilingData_,0x5A,sizeof(t.tilingData_));t.tilingData_.gmmWeightQuantParam.reserved=0;
  auto saved=t.tilingData_;auto selected=t.SelectMxPipelineKey(&ctx,originalKey);
  assert(selected==originalKey);assert(std::memcmp(&saved,&t.tilingData_,sizeof(saved))==0);
 }
 platform_present=true;bandwidth_ok=true;
 for(uint64_t m:{0ULL,1ULL,8ULL,16ULL,36ULL,48ULL,80ULL,128ULL,256ULL,432ULL,768ULL,32768ULL})
 for(uint64_t e:{0ULL,1ULL,3ULL,7ULL,12ULL,16ULL,36ULL,48ULL})
 for(uint64_t k:{0ULL,256ULL,512ULL,576ULL,1024ULL,3584ULL,5120ULL})
 for(uint64_t n:{0ULL,32ULL,128ULL,1536ULL,2048ULL,4096ULL,4608ULL,6144ULL}){
  GroupedWeightQuantBatchMatmulTiling t;t.mSize_=m;t.groupNum_=e;t.kSize_=k;t.nSize_=n;
  t.platformBandwidth_.supportsMxPipeline=true;
  std::memset(&t.tilingData_,0x5A,sizeof(t.tilingData_));auto saved=t.tilingData_;
  const auto route=GROUPED_MATMUL::MxPipeline::Select(m,e,k,n);
  auto key=t.SelectMxPipelineKey(&ctx,74530817ULL);
  assert(t.tilingData_.gmmWeightQuantParam.reserved == saved.gmmWeightQuantParam.reserved);
  assert(std::memcmp(&saved,&t.tilingData_,sizeof(saved))==0);
  if(route==GROUPED_MATMUL::MxPipeline::Route::Original)assert(key==74530817ULL);
  else assert(key!=74530817ULL);
 }
}
"""
        )


if __name__ == "__main__":
    unittest.main()
