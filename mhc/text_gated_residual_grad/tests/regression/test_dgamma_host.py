#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Host FP32 regression; hardware instructions, synchronization and DMA are mocked."""

from pathlib import Path
import subprocess
import re

import argparse
import tempfile

DEFAULT_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "op_kernel/arch35/text_gated_residual_grad_vector_arch35.h"
)

METHOD_NAMES = (
    "DgammaComp",
    "DgammaAccumAdd",
    "FloatBuf",
    "Pointer",
    "ReduceDgamma",
)


def method(source, name):
    start = re.search(r"    __aicore__ inline [^\n]*\b" + name + r"\(", source).start()
    pos = start
    opening = source.index("{", pos)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end].replace("__aicore__ ", "")


prefix = r"""
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include "text_gated_residual_grad_constants.h"
using namespace TextGatedResidualGrad;
using GM_ADDR=unsigned char*;
template<class T> struct LocalTensor {T* p; T& operator[](unsigned i)const{return p[i];}};
struct Buffer {std::vector<float> data=std::vector<float>(1024);template<class T> LocalTensor<T> GetWithOffset(unsigned,unsigned off){return {reinterpret_cast<T*>(reinterpret_cast<unsigned char*>(data.data())+off)};}};
void Add(LocalTensor<float>d,LocalTensor<float>a,LocalTensor<float>b,unsigned n){for(unsigned i=0;i<n;i++)d[i]=a[i]+b[i];}
void Sub(LocalTensor<float>d,LocalTensor<float>a,LocalTensor<float>b,unsigned n){for(unsigned i=0;i<n;i++)d[i]=a[i]-b[i];}
void Adds(LocalTensor<float>d,LocalTensor<float>a,float b,unsigned n){for(unsigned i=0;i<n;i++)d[i]=a[i]+b;}
void Muls(LocalTensor<float>d,LocalTensor<float>a,float b,unsigned n){for(unsigned i=0;i<n;i++)d[i]=a[i]*b;}
void Duplicate(LocalTensor<float>d,float b,unsigned n){for(unsigned i=0;i<n;i++)d[i]=b;}
struct Kernel {
 Buffer ub_;
 bool kahanDgamma_=true;
 unsigned h_=16,bufCount_=16,c_=1,aivNum_=2,aivId_=0,dgammaBytes_=64,ubBufBase_=128;
 struct {uint64_t dgammaPartial=0;} layout_;
 GM_ADDR workspace_=nullptr,gmDgamma_=nullptr;
 template<class T> void CopyIn(GM_ADDR src,uint64_t off,LocalTensor<T> dst,unsigned n){std::memcpy(dst.p,reinterpret_cast<T*>(src)+off,n*sizeof(T));}
 template<class T> void CopyOut(GM_ADDR dst,uint64_t off,LocalTensor<T> src,unsigned n){std::memcpy(reinterpret_cast<T*>(dst)+off,src.p,n*sizeof(T));}
"""
suffix = r"""
};
int main(){
 float partial[64]{},output[16]{};
 for(unsigned core=0;core<2;core++){
  Kernel k;auto val=k.FloatBuf(0,16);auto tmp=k.FloatBuf(1,16);auto tmp2=k.FloatBuf(2,16);
  Duplicate(val,core==0?16777216.0f:-16777216.0f,16);k.DgammaAccumAdd(val,16,0,tmp,tmp2);
  if(core==0){Duplicate(val,1.0f,16);k.DgammaAccumAdd(val,16,0,tmp,tmp2);}
  std::memcpy(partial+core*32,k.ub_.data.data(),128);
 }
 Kernel reducer;reducer.workspace_=reinterpret_cast<GM_ADDR>(partial);reducer.gmDgamma_=reinterpret_cast<GM_ADDR>(output);reducer.ReduceDgamma();
 std::cout<<output[0]<<'\n';
 for(float v:output)if(v!=1.0f)return 1;
}
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Execute the actual Kahan methods with host FP32 operations."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Compile without running the host regression.",
    )
    args = parser.parse_args()
    source = args.source.read_text()
    methods = []
    for name in METHOD_NAMES:
        methods.append(method(source, name))
    body = "\n".join(methods)
    with tempfile.TemporaryDirectory(prefix="tgrg-kahan-") as tmp:
        cpp = Path(tmp) / "kahan.cpp"
        exe = Path(tmp) / "kahan"
        cpp.write_text(prefix + body + suffix)
        subprocess.run(
            [
                "c++",
                "-std=c++17",
                "-O0",
                "-ffp-contract=off",
                "-I",
                str(DEFAULT_SOURCE.parent.parent),
                str(cpp),
                "-o",
                str(exe),
            ],
            check=True,
        )
        if not args.compile_only:
            subprocess.run([str(exe)], check=True)
    if args.compile_only:
        print("COMPILED: extracted DgammaAccumAdd/ReduceDgamma (not executed)")
    else:
        print("PASS: actual DgammaAccumAdd/ReduceDgamma preserve +1 after cancellation")
