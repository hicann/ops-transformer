# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch

# =====================================================================================================================
# Compressor 位级一致性测试参数配置
#
# 测试目的:
#   1. 跨模板位级一致性: BSH + S<=4 且 B*S<=token_limit(arch22为128) 时走 FULL_LOAD 模板,
#      B 再增大即切换到 PERF(normal) 模板; 两次运行共享 batch 的输出必须位级一致;
#   2. normal(PERF)模板内部"悬崖机制"位级一致性: PERF 模板按 mSize=B*S 与 AIC 核数动态选择 D 分块大小
#      (dBaseSize 32 <-> 64), 穿过边界后同一输出列由不同核计算, hIdxStart=(核内D块idx)*K_L1_BASE 随之改变,
#      K 轴累加顺序变化可能带来位级差异;
#   3. batch 位置一致性: 同 B 下将"目标样本"(独立数据 + 独立 start_pos)换到首/中/尾位置, 其结果必须位级一致.
#
# 用例表(与需求对应):
#   | 用例                | 目的                                                        |
#   | B=32/33、S=4        | tokenSize=128/132, FULL_LOAD -> PERF 模板切换                |
#   | B=96/97、S=4        | 24 AIC核、coff=2、D=512 下, PERF 动态 D 分块变化(悬崖边界)    |
#   | B=31/32             | tokenSize=124/128, 同为 FULL_LOAD, 对照组                    |
#   | 同B, 目标样本首/中/尾 | batch 位置一致性                                             |
#
# 说明:
#   - "B=96/97"是 24 AIC 核设备(910B1等)的悬崖边界: mSize<=128*(AIC//8) 时 dBaseSize=32;
#     非该核数设备上由测试框架按当前设备 AIC 核数自动推导边界对(910B3=20核时为 B=64/65),
#     "dyn" 即表示该自适应边界对, 与固定用例重复时会自动去重.
#   - 所有用例固定 layout=BSH(FULL_LOAD 模板仅 BSH 触发)、随机种子固定, 保证跨 B/跨位置数据可复现.
#   - 参数均以列表书写(与 test_compressor_paramset.py 风格一致), 多值时做笛卡尔积展开.
# =====================================================================================================================

# 公共参数: C4A 组合(D=512, coff=2, cmp_ratio=4), BF16, 连续buffer
_COMMON = {
    "seq_len": [4],
    "head_dim": [512],
    "hidden_size": [4096],
    "block_size": [128],
    "cmp_ratio": [4],
    "coff": [2],
    "cache_mode": [1],
    "data_type": [torch.bfloat16],
    # start_pos 取值说明(与 block_size=128、cmp_ratio=4 配合):
    #   default_start_p=256: 对齐(p%4==0), 每batch引用2个物理块(读前一块, 写当前块), nb=3
    #   target_start_p=257: 未对齐, 跨压缩块边界(headHolder路径), 仅 position 用例的目标样本使用
    "default_start_p": [256],
    "target_start_p": [257],
    "datarange": [[-10, 10]],
    "seed": [2026],
}

BITLEVEL_TEST_PARAMS = {
    # 用例1: B=31/32, tokenSize=124/128, 均为 FULL_LOAD, 同模板对照组
    "FullLoadCtrl": {
        "case_type": ["pair"],
        "pair_b": [(31, 32)],
        **_COMMON,
    },
    # 用例2: B=32/33, tokenSize=128/132, FULL_LOAD -> PERF 模板切换(跨模板位级一致性)
    "FullLoadToPerf": {
        "case_type": ["pair"],
        "pair_b": [(32, 33)],
        **_COMMON,
    },
    # 用例3: B=64/65(20 AIC核悬崖边界, 固定保留) + "dyn"(按当前设备 AIC 核数自适应推导的悬崖边界对)
    #        PERF 动态 D 分块变化(dBaseSize 32<->64, hIdxStart 改变)位级一致性
    "PerfDsplitCliff": {
        "case_type": ["pair"],
        "pair_b": [(64, 65), "dyn"],
        **_COMMON,
    },
    # 用例4: 同 B, 目标样本(独立数据 + target_start_p)换到首/中/尾位置, batch 位置一致性
    #        B=32(FULL_LOAD)/33(PERF) 覆盖两种模板, B=96/97 覆盖大 B 悬崖两侧场景
    "PositionConsistency": {
        "case_type": ["position"],
        "batch_size": [32, 33, 96, 97],
        **_COMMON,
    },
    # 用例5(模板探针): B=2, S=6, BSH, start_pos=256
    #   S=6 > 4 不满足 FULL_LOAD 的 seqSize<=4 条件 -> 走 NORMAL(PERF)模板
    #   配合 op_host/arch22/compressor_tiling.cpp SetInnerSplitInfo else 分支的调试探针,
    #   验证 NORMAL 模板不设置 kBaseNum(K 轴不做核间切分)
    #   对照: 把 seq_len 改成 [6, 4] 可同时触发 B=2/S=4(tokenSize=8<=128)的 FULL_LOAD 分支打印
    "NormalTemplateProbe": {
        "case_type": ["single"],
        "batch_size": [2],
        **_COMMON,
        "seq_len": [6, 4],
    },
}

# 按需选择要启用的测试参数组: 注释掉对应行即可屏蔽该组用例
ENABLED_PARAMS = [
    # ---- 位级一致性用例(batch一致性) ----
    BITLEVEL_TEST_PARAMS["FullLoadCtrl"],  # B=31/32 同FULL_LOAD对照
    BITLEVEL_TEST_PARAMS["FullLoadToPerf"],  # B=32/33 FULL_LOAD->PERF模板切换
    BITLEVEL_TEST_PARAMS["PerfDsplitCliff"],  # B=96/97+dyn 动态D分块悬崖
    BITLEVEL_TEST_PARAMS["PositionConsistency"],  # 同B目标样本首/中/尾位置一致性
    # BITLEVEL_TEST_PARAMS["NormalTemplateProbe"],   # B=2, S=6 -> NORMAL模板(验证SetInnerSplitInfo else分支)
]
