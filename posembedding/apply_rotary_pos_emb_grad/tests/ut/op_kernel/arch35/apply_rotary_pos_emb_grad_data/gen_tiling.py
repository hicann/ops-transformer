#!/usr/bin/python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License).
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import numpy as np
import sys

# tiling.bin 为 int64 流, 内存布局与 op_kernel/arch35/apply_rotary_pos_emb_grad_tiling_data.h 的
# ApplyRopeGradTilingData 严格一致(共 101 槽 = 808B):
#   [reduceTiling 66 槽] [ropeGradParams 17 槽] [ropeGradABParams 17 槽] [dCosFlag|layout 合 1 槽]
#
# case0: B=2 S=64 nQ=nK=4 D=48 fp32, cos/sin=[2,64,1,48] 与 grad 的 B/S 轴一致(N 轴广播)
#        → 内部 layout=SBND → AB 模板(DxTilingKey=204) + dCosFlag=1
# reduce 视角: 部分积 [B*S, maxN, D]=[128,4,48], N 轴居中为 reduce 轴(ARA pattern=20),
#             32 核分核 A 轴(128 行 / ubFactorA=2 = 64 块, 每核 2 块)

_case0_reduce_tiling = (
    [
        2,  # factorACntPerCore: 32 核 x 2 块 = 64
        64,  # factorATotalCnt: A 轴(128 行) / ubFactorA(2)
        2,  # ubFactorA
        1,  # factorRCntPerCore: R 轴不分核
        1,  # factorRTotalCnt
        1,  # ubFactorR
        1,  # groupR
        6144,  # outSize: 128 * 48 (reduce 输出元素数)
        52736,  # basicBlock: preBuf 单块字节数
        8192,  # resultBlock: resBuf 字节数
        (32 << 32) | 48,  # coreNum(低32)=48, realCoreNum(高32)=32
        0,  # useNddma(低32)=0, meanVar(高32)=0
    ]
    + [128, 4, 48]
    + [0] * 6  # shape: [BS, N, D]
    + [192, 48, 1]
    + [0] * 6  # stride: 连续布局元素步长
    + [48, 48, 1]
    + [0] * 6  # dstStride: grad_cos 扁平 [128, 1, 48]
    + [1, 1, 1]
    + [0] * 6  # sliceNum
    + [128, 4, 48]
    + [0] * 6  # sliceShape
    + [192, 48, 1]
    + [0] * 6
)  # sliceStride

# BAB/A 模板共用参数, AB 用例仅填 shape 头, 分核字段置 0
_case0_rope_grad_params = [
    2,  # b
    64,  # s
    48,  # d
    4,  # nQ
    4,  # nK
] + [0] * 12

_case0_rope_grad_ab_params = [
    2,  # b
    64,  # s
    48,  # d
    4,  # nQ
    4,  # nK
    64,  # dAlign: 48 对齐至 64
    2,  # dSplitCoef: half 模式
    43,  # blockNumBS
    3,  # blockFactorBS
    2,  # blockTailBS
    1,  # blockNumN
    4,  # blockFactorN
    4,  # blockTailN
    3,  # ubFactorBS
    4,  # ubFactorN
    43,  # usedCoreNum = blockNumBS * blockNumN
    0,  # rotaryMode: HALF=0
]

case0_params = (
    _case0_reduce_tiling
    + _case0_rope_grad_params
    + _case0_rope_grad_ab_params
    + [(1 << 32) | 1]
)  # dCosFlag(低32)=1, layout(高32)=1(SBND)

# case1: 与 case0 同形态但 D=64 (天然 32B 对齐, 无 dAlign padding), 用于隔离 d=48 的
#        dAlign=64 padding 搬运/计算路径差异
_case1_reduce_tiling = list(_case0_reduce_tiling)
for i, v in enumerate(_case1_reduce_tiling):
    if v == 6144:  # outSize: 128 * 64
        _case1_reduce_tiling[i] = 8192
_case1_reduce_tiling[12:15] = [128, 4, 64]  # shape
_case1_reduce_tiling[21:24] = [256, 64, 1]  # stride
_case1_reduce_tiling[30:33] = [64, 64, 1]  # dstStride
_case1_reduce_tiling[48:51] = [128, 4, 64]  # sliceShape
_case1_reduce_tiling[57:60] = [256, 64, 1]  # sliceStride

_case1_rope_grad_params = [2, 64, 64, 4, 4] + [0] * 12  # b, s, d, nQ, nK
_case1_rope_grad_ab_params = list(_case0_rope_grad_ab_params)
_case1_rope_grad_ab_params[2] = 64  # d

case1_params = (
    _case1_reduce_tiling
    + _case1_rope_grad_params
    + _case1_rope_grad_ab_params
    + [(1 << 32) | 1]
)

params_info = {
    "case0": case0_params,
    "case1": case1_params,
}


def main():
    params_list = params_info[
        sys.argv[1]
    ]  # python gen_tiling.py case0  sys.argv[1]="case0"
    base_params = np.array(params_list, dtype=np.int64)
    tiling_file = open("tiling.bin", "wb")
    base_params.tofile(tiling_file)
    tiling_file.close()


if __name__ == "__main__":
    main()
