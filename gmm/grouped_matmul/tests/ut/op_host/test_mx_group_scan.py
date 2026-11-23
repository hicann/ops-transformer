# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from pathlib import Path
import unittest

from mx_test_utils import run_cpp


class GroupScanTest(unittest.TestCase):
    def test_combined_bounds_and_single_read(self):
        header = (
            Path(__file__).resolve().parents[3]
            / "op_kernel/arch35/quant_adaptive_sliding_window_templates/gmm_weight_quant_mx_buffered_policy.h"
        )
        run_cpp(
            '#include "'
            + header.as_posix()
            + '"\n'
            + r"""
#include <array>
#include <cassert>
#include <initializer_list>
#include <limits>
#include <vector>
using namespace GROUPED_MATMUL::MxBuffered;

// Preserve the three previous passes as the reference, including their short circuits.
bool Before(const int64_t* counts, uint64_t groups, Route route, uint64_t maxN) {
    bool nonempty=false;
    for(uint64_t g=0;g<groups;++g) {
        if(counts[g]<0 || counts[g]>static_cast<int64_t>(MAX_GROUP_ROWS))return false;
        nonempty=nonempty || counts[g]!=0;
    }
    if(!nonempty)return false;
    if(route==Route::WeightQuad) {
        if(maxN==0 || maxN>MAX_N_WIDTH)return false;
        for(uint64_t g=0;g<groups;++g) {
            if(counts[g]<0 || counts[g]>static_cast<int64_t>(MAX_GROUP_ROWS))return false;
            const uint64_t aligned=(static_cast<uint64_t>(counts[g])+CUBE_ALIGNMENT-1)
                                    /CUBE_ALIGNMENT*CUBE_ALIGNMENT;
            if(aligned>L0C_REGION_BYTES/(maxN*sizeof(float)))return false;
        }
    }
    const uint64_t maxRows=route==Route::OperandTriple?MAX_GROUP_ROWS-1:MAX_GROUP_ROWS;
    for(uint64_t g=0;g<groups;++g) {
        const int64_t m=counts[g];
        if(m==0)continue;
        if(m<=static_cast<int64_t>(REUSE_MIN_ROWS) || m>static_cast<int64_t>(maxRows))return false;
    }
    return true;
}

struct CountedReads {
    const int64_t* values;
    unsigned* reads;
    int64_t operator[](uint64_t i)const {++reads[i];return values[i];}
};

void Check(const std::vector<int64_t>& counts, Route route, uint64_t maxN) {
    std::vector<unsigned> reads(counts.size(),0);
    const bool actual=FitsBufferedGroups(CountedReads{counts.data(),reads.data()},counts.size(),route,maxN);
    assert(actual==Before(counts.data(),counts.size(),route,maxN));
    for(unsigned n:reads) {assert(n<=1);if(actual)assert(n==1);}
}

int main() {
    for(auto route:{Route::OperandTriple,Route::WeightQuad}) {
        for(uint64_t width=0;width<=MAX_N_WIDTH+1;++width) {
            Check({},route,width);
            for(int64_t m=-2;m<=static_cast<int64_t>(MAX_GROUP_ROWS)+2;++m)
                Check({0,m,0},route,width);
        }
        for(uint64_t width:{0UL,1UL,31UL,32UL,127UL,128UL,129UL,160UL,192UL,224UL,255UL,256UL,257UL})
        for(int64_t a:{-1L,0L,128L,129L,144L,160L,192L,255L,256L,257L})
        for(int64_t b:{0L,128L,129L,160L,255L,256L})
        for(int64_t c:{0L,128L,129L,160L,255L,256L})Check({a,b,c},route,width);
        Check({std::numeric_limits<int64_t>::min(),129},route,128);
        Check({129,std::numeric_limits<int64_t>::max()},route,128);
        Check({129,0},route,std::numeric_limits<uint64_t>::max());
        Check(std::vector<int64_t>(1024,0),route,128);
        auto many=std::vector<int64_t>(1024,129);
        Check(many,route,128);
        many.back()=128;Check(many,route,128);
    }
}
"""
        )
