# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compile production planning helpers without the NPU runtime."""

from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[3]


def _run(tmp_path, body):
    compiler = shutil.which("g++")
    assert compiler, "g++ required for production helper behavior tests"
    header = (ROOT / "op_kernel/allto_allv_grouped_mat_mul_tiling.h").read_text()
    header = header.split("#pragma pack(push, 8)")[0]
    header = re.sub(r"^#include .*$", "", header, flags=re.M)
    source = (
        "#include <cstdint>\n#include <cassert>\n#include <array>\n"
        + header
        + "\n#endif\n"
    )
    source += (
        "\nusing namespace AlltoAllvGroupedMatMulAivMode;\nint main() {\n"
        + body
        + "\n}\n"
    )
    cpp = tmp_path / "behavior.cpp"
    exe = tmp_path / "behavior"
    cpp.write_text(source)
    subprocess.run(
        [compiler, "-std=c++17", "-O0", str(cpp), "-o", str(exe)], check=True
    )
    subprocess.run([str(exe)], check=True)


def test_large_experts_use_pipeline_for_sparse_small_work(tmp_path):
    _run(
        tmp_path,
        r"""
    for (uint32_t e : {129U,134U,135U,136U,256U,269U,270U,271U,405U,512U}) {
        std::array<int32_t,1024> prefix{};
        for (uint32_t i=0; i<2*e; ++i) prefix[i]=1;
        assert(IsExpertOverlapProtocolSafe(prefix.data(),2,e));
        assert(ShouldUseAutomaticExpertOverlap(prefix.data(),2,e,16,16));
    }
    std::array<int32_t,1024> invalid{};
    invalid[0]=1;
    assert(!ShouldUseAutomaticExpertOverlap(invalid.data(),2,512,64,64));
    assert(!ShouldUseAutomaticExpertOverlap(invalid.data(),2,513,64,64));
    """,
    )


def test_counted_notifications_are_drained_before_reuse(tmp_path):
    _run(
        tmp_path,
        r"""
    for (uint32_t experts : {128U,129U,134U,135U,136U,256U,269U,270U,271U,405U,512U}) {
        for (uint32_t invocation=0; invocation<4; ++invocation) {
            std::array<uint32_t,9> outstanding{};
            uint32_t drains=0;
            for (uint32_t e=0; e<experts; ++e) {
                const auto id=ExpertReadyFlagId(e);
                assert(id<9 && id!=kSharedReadyFlag && id!=kRoutedReadyFlag);
                assert(++outstanding[id]<=15); // producer may run ahead of all consumers
                if (NeedsExpertEventDrain(e+1)) {
                    for (auto &value:outstanding) value=0; // both sides reached barrier
                    ++drains;
                }
            }
            assert(drains==experts/135);
        }
    }
    assert(!NeedsExpertEventDrain(0));
    """,
    )


def test_copy_capacity_reserves_metadata_and_both_buffers(tmp_path):
    _run(
        tmp_path,
        r"""
    for (uint64_t ub=0; ub<300000; ++ub) {
        auto elements=CopyMoveCapacity(ub);
        assert(elements%16==0 && elements<=16384);
        if (elements) assert(uint64_t(elements)*2*2+kCopyFlagBufferBytes<=ub);
    }
    assert(CopyMoveCapacity(65568)==16384);
    assert(CopyMoveCapacity(95)==0);
    assert(CopyMoveCapacity(96)==16);
    """,
    )


def test_real_copy_helper_drains_and_overlaps_alternating_buffers(tmp_path):
    comm = (ROOT / "op_kernel/allto_allv_grouped_mat_mul_aiv_comm.h").read_text()
    start = comm.index(
        "template <typename T>\n__aicore__ inline void CopyPeerGmToLocalGm"
    )
    end = comm.index("__aicore__ inline __gm__ int32_t *GetFlagAddress", start)
    helper = comm[start:end]
    # A deliberately deferred MTE3 model detects UB overwrite before reuse,
    # incomplete final drain, and the absence of two-buffer overlap.
    stub = r"""
#include <cassert>
#include <cstdint>
#include <algorithm>
#include <functional>
#include <vector>
#define __aicore__
constexpr int EVENT_ID0=0, EVENT_ID1=1;
namespace AscendC {
enum class TPosition { VECCALC };
enum class HardEvent { MTE2_MTE3, MTE3_MTE2 };
using TEventID=int;
std::function<void()> pending[2];
int overlaps=0;
int sets=0;
template<class T> struct LocalTensor { T* p; int id; };
template<class T> struct GlobalTensor {
    T* p; GlobalTensor operator[](uint64_t n) { return {p+n}; }
};
template<TPosition P> struct TBuf {
    int32_t values[64]{}; int id;
    template<class T> LocalTensor<T> Get() { return {reinterpret_cast<T*>(values),id}; }
};
struct DataCopyExtParams {
    uint32_t bytes;
    DataCopyExtParams(uint32_t, uint32_t length, uint32_t, uint32_t, uint32_t):bytes(length) {}
};
template<class T> struct DataCopyPadExtParams {};
template<class T> void DataCopyPad(LocalTensor<T> local, GlobalTensor<T> src,
                                  DataCopyExtParams p, DataCopyPadExtParams<T>) {
    assert(!pending[local.id]);
    if (pending[1-local.id]) ++overlaps;
    std::copy_n(src.p,p.bytes/sizeof(T),local.p);
}
template<class T> void DataCopyPad(GlobalTensor<T> dst, LocalTensor<T> local, DataCopyExtParams p) {
    assert(!pending[local.id]);
    pending[local.id]=[=] {std::copy_n(local.p,p.bytes/sizeof(T),dst.p);};
}
template<HardEvent E> void SetFlag(int) { ++sets; }
template<HardEvent E> void WaitFlag(int id) {
    if constexpr(E==HardEvent::MTE3_MTE2) {
        if (pending[id]) { pending[id](); pending[id]={}; }
    }
}
}
"""
    main = r"""
int main() {
    AscendC::TBuf<AscendC::TPosition::VECCALC> b0{},b1{};
    b0.id=0; b1.id=1;
    for (int repeat=0; repeat<3; ++repeat) {
        for (uint32_t count : {0U,1U,8U,9U,16U,17U,24U,25U,40U,47U}) {
            std::vector<int32_t> src(count+2),dst(count+2,-1);
            for(uint32_t i=0;i<src.size();++i) src[i]=i+repeat*100;
            AscendC::overlaps=0; AscendC::sets=0;
            CopyPeerGmToLocalGm<int32_t>({dst.data()+1},{src.data()+1},count,8,b0,b1);
            assert(!AscendC::pending[0] && !AscendC::pending[1]);
            assert(dst.front()==-1 && dst.back()==-1);
            for(uint32_t i=1;i<=count;++i) assert(dst[i]==src[i]);
            if(count>8) assert(AscendC::overlaps>0);
            if(count>0 && count<=8) assert(AscendC::sets==2);
            CopyLocalGmToWindow<int32_t>({dst.data()+1},{src.data()+1},count,8,b0,b1);
            assert(!AscendC::pending[0] && !AscendC::pending[1]);
        }
    }
}
"""
    cpp = tmp_path / "copy.cpp"
    cpp.write_text(stub + helper + main)
    exe = tmp_path / "copy"
    subprocess.run(["g++", "-std=c++17", str(cpp), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
