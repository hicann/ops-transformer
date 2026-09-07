#!/usr/bin/python
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ------------------------------------------------------------------------------

import torch


####### 参数说明 ########
# 所有参数均使用 list；同一路径的相邻取值可在一个 TestCases 条目内展开。
# batch_size: BSND 的 B；TND 的 B 由 cu_seqlens_q 的长度确定。
# q_seq/k_seq: BSND 的 S1/S2；TND 场景下为标称长度。
# q_t_size/k_t_size: TND 的总 token 数，等于对应 cu_seqlens 的末值。
# q_head_num/k_head_num: N1/N2，当前实现要求 N2=1，G=N1/N2<=64。
# head_dim: query/key 的 D，当前实现要求 D=128。
# block_size/block_num: PA 物理块大小和数量；block_size 为 [16, 1024] 内 16 的倍数。
# cu_seqlens_q/k: TND 的 B+1 长度前缀和；非 TND 传 None。
# seqused_q/k: 每个 batch 的实际压缩后长度；优先级高于 cu_seqlens 和 shape。
# cmp_residual_k: mask3 下压缩前 K 长度的余数，actual_s2_orig=seqused_k*cmp_ratio+residual。
# max_seqlen_q: 控制 cube 的 S2 内部分块；[0, 4] 取 256，其余取 128。
# layout_query/layout_key: 合法组合为 BSND+BSND、TND+TND、BSND+PA、TND+PA。
# sparse_count: 每行 TopK 数量，范围 [1, 8192]。
# sparse_mode: 0 为无 mask，3 为下三角 sparse mask。
# cmp_ratio: K 压缩倍率，范围 [1, 128]。
# return_value: 0 仅返回 indices，1 同时返回 sparse_values。
# output_idx_offset: 可选索引偏移；BSND 为 [B,S1,N2]，TND 为 [T,N2]。
# _quant_profiles: 构造字段；将场景与合法 quant_mode/dtype 组合绑定。

BASE = {
    "batch_size": [1],
    "q_seq": [1],
    "k_seq": [128],
    "q_t_size": [None],
    "k_t_size": [None],
    "q_head_num": [1],
    "k_head_num": [1],
    "head_dim": [128],
    "block_size": [None],
    "block_num": [None],
    "weight_dtype": [torch.float32],
    "dequant_dtype": [torch.float8_e8m0fnu],
    "actual_seq_dtype": [torch.int32],
    "cu_seqlens_q": [None],
    "cu_seqlens_k": [None],
    "seqused_q": [None],
    "seqused_k": [None],
    "cmp_residual_k": [None],
    "max_seqlen_q": [1],
    "layout_query": ["BSND"],
    "layout_key": ["BSND"],
    "sparse_count": [64],
    "sparse_mode": [0],
    "query_datarange": [[-448, 448]],
    "key_datarange": [[-20, 20]],
    "weights_datarange": [[-123, 123]],
    # 避免零 scale 产生大量同分值，使 TopK 索引比较退化为未定义的 tie-breaking。
    "q_scale_datarange": [[1, 255]],
    "k_scale_datarange": [[1, 65504]],
    "cmp_ratio": [1],
    "return_value": [0],
    "output_idx_offset": [None],
    "run_mode": ["eager"],
}


####### 覆盖范围 ########
# dtype/layout：覆盖五种 quant_mode 与四种合法 layout 组合。
# S1/G：覆盖 S1 基本块 4/2 的首块、整块、多块和尾块，以及 G 的奇偶与上界。
# S2：覆盖 0、基本块 128 的左右边界、多块整除和多块尾块。
# TopK：覆盖所有 trunkLen 切换点、对齐、补齐、单轮及多轮 merge。
# metadata：覆盖 FD 开关、batch/row/block/ForceAssign、LD 分配及零 cost 游标。
# 其他：覆盖实际长度来源、mask、cmp_ratio、PA block、offset 和返回值分支。
TestCases = {
    # ------ dtype + layout
    # FP8，BSND+BSND。
    "BSND_01": {
        **BASE,
        "_quant_profiles": ["FP8"],
    },
    # FP8，BSND+PA。
    "PA_02": {
        **BASE,
        "block_size": [128],
        "block_num": [1],
        "seqused_k": [[128]],
        "layout_key": ["PA_BBND"],
        "_quant_profiles": ["FP8"],
    },
    # FP8，TND+TND。
    "TND_03": {
        **BASE,
        "q_t_size": [1],
        "k_t_size": [128],
        "cu_seqlens_q": [[0, 1]],
        "cu_seqlens_k": [[0, 128]],
        "layout_query": ["TND"],
        "layout_key": ["TND"],
        "_quant_profiles": ["FP8"],
    },
    # FP8，TND+PA。
    "PA_04": {
        **BASE,
        "q_t_size": [1],
        "block_size": [128],
        "block_num": [1],
        "cu_seqlens_q": [[0, 1]],
        "seqused_k": [[128]],
        "layout_query": ["TND"],
        "layout_key": ["PA_BBND"],
        "_quant_profiles": ["FP8"],
    },
    # INT8，BSND+BSND。
    "BSND_05": {
        **BASE,
        "_quant_profiles": ["INT8"],
    },
    # INT8，BSND+PA。
    "PA_06": {
        **BASE,
        "block_size": [128],
        "block_num": [1],
        "seqused_k": [[128]],
        "layout_key": ["PA_BBND"],
        "_quant_profiles": ["INT8"],
    },
    # INT8，TND+TND。
    "TND_07": {
        **BASE,
        "q_t_size": [1],
        "k_t_size": [128],
        "cu_seqlens_q": [[0, 1]],
        "cu_seqlens_k": [[0, 128]],
        "layout_query": ["TND"],
        "layout_key": ["TND"],
        "_quant_profiles": ["INT8"],
    },
    # INT8，TND+PA。
    "PA_08": {
        **BASE,
        "q_t_size": [1],
        "block_size": [128],
        "block_num": [1],
        "cu_seqlens_q": [[0, 1]],
        "seqused_k": [[128]],
        "layout_query": ["TND"],
        "layout_key": ["PA_BBND"],
        "_quant_profiles": ["INT8"],
    },
    # MXFP8，BSND+BSND。
    "BSND_09": {
        **BASE,
        "_quant_profiles": ["MXFP8"],
    },
    # MXFP8，BSND+PA。
    "PA_10": {
        **BASE,
        "block_size": [128],
        "block_num": [1],
        "seqused_k": [[128]],
        "layout_key": ["PA_BBND"],
        "_quant_profiles": ["MXFP8"],
    },
    # MXFP8，TND+TND。
    "TND_11": {
        **BASE,
        "q_t_size": [1],
        "k_t_size": [128],
        "cu_seqlens_q": [[0, 1]],
        "cu_seqlens_k": [[0, 128]],
        "layout_query": ["TND"],
        "layout_key": ["TND"],
        "_quant_profiles": ["MXFP8"],
    },
    # MXFP8，TND+PA。
    "PA_12": {
        **BASE,
        "q_t_size": [1],
        "block_size": [128],
        "block_num": [1],
        "cu_seqlens_q": [[0, 1]],
        "seqused_k": [[128]],
        "layout_query": ["TND"],
        "layout_key": ["PA_BBND"],
        "_quant_profiles": ["MXFP8"],
    },
    # HIFLOAT8，BSND+BSND。
    "BSND_13": {
        **BASE,
        "_quant_profiles": ["HIFLOAT8"],
    },
    # HIFLOAT8，BSND+PA。
    "PA_14": {
        **BASE,
        "block_size": [128],
        "block_num": [1],
        "seqused_k": [[128]],
        "layout_key": ["PA_BBND"],
        "_quant_profiles": ["HIFLOAT8"],
    },
    # HIFLOAT8，TND+TND。
    "TND_15": {
        **BASE,
        "q_t_size": [1],
        "k_t_size": [128],
        "cu_seqlens_q": [[0, 1]],
        "cu_seqlens_k": [[0, 128]],
        "layout_query": ["TND"],
        "layout_key": ["TND"],
        "_quant_profiles": ["HIFLOAT8"],
    },
    # HIFLOAT8，TND+PA。
    "PA_16": {
        **BASE,
        "q_t_size": [1],
        "block_size": [128],
        "block_num": [1],
        "cu_seqlens_q": [[0, 1]],
        "seqused_k": [[128]],
        "layout_query": ["TND"],
        "layout_key": ["PA_BBND"],
        "_quant_profiles": ["HIFLOAT8"],
    },
    # MXFP4，BSND+BSND，覆盖 D/2 的 packed stride。
    "BSND_17": {
        **BASE,
        "_quant_profiles": ["MXFP4"],
    },
    # MXFP4，BSND+PA，覆盖 packed PA key。
    "PA_18": {
        **BASE,
        "block_size": [128],
        "block_num": [1],
        "seqused_k": [[128]],
        "layout_key": ["PA_BBND"],
        "_quant_profiles": ["MXFP4"],
    },
    # MXFP4，TND+TND。
    "TND_19": {
        **BASE,
        "q_t_size": [1],
        "k_t_size": [128],
        "cu_seqlens_q": [[0, 1]],
        "cu_seqlens_k": [[0, 128]],
        "layout_query": ["TND"],
        "layout_key": ["TND"],
        "_quant_profiles": ["MXFP4"],
    },
    # MXFP4，TND+PA。
    "PA_20": {
        **BASE,
        "q_t_size": [1],
        "block_size": [128],
        "block_num": [1],
        "cu_seqlens_q": [[0, 1]],
        "seqused_k": [[128]],
        "layout_query": ["TND"],
        "layout_key": ["PA_BBND"],
        "_quant_profiles": ["MXFP4"],
    },
    # ------ S1 / G
    # TopK<=6144，S1<4，单个 S1 尾块。
    "S1_21": {
        **BASE,
        "q_seq": [1],
        "_quant_profiles": ["MXFP8"],
    },
    # TopK<=6144，S1=4，单个完整 S1 块。
    "S1_22": {
        **BASE,
        "q_seq": [4],
        "max_seqlen_q": [4],
        "_quant_profiles": ["MXFP8"],
    },
    # TopK<=6144，S1=8，多个完整 S1 块。
    "S1_23": {
        **BASE,
        "q_seq": [8],
        "max_seqlen_q": [8],
        "_quant_profiles": ["MXFP8"],
    },
    # TopK<=6144，S1=5，完整块后带 S1 尾块。
    "S1_24": {
        **BASE,
        "q_seq": [5],
        "max_seqlen_q": [5],
        "_quant_profiles": ["MXFP8"],
    },
    # G=2，覆盖偶数 G 向量规约。
    "G_25": {
        **BASE,
        "q_seq": [4],
        "q_head_num": [2],
        "max_seqlen_q": [4],
        "_quant_profiles": ["MXFP8"],
    },
    # G=3，覆盖奇数 G 向量规约。
    "G_26": {
        **BASE,
        "q_seq": [4],
        "q_head_num": [3],
        "max_seqlen_q": [4],
        "_quant_profiles": ["MXFP8"],
    },
    # G=64，覆盖 G 上界和 M=256 的满块。
    "G_27": {
        **BASE,
        "q_seq": [4],
        "q_head_num": [64],
        "max_seqlen_q": [4],
        "_quant_profiles": ["MXFP8"],
    },
    # TopK>6144，S1=2，覆盖缩小后的完整 S1 块。
    "S1_28": {
        **BASE,
        "q_seq": [2],
        "k_seq": [6145],
        "max_seqlen_q": [2],
        "sparse_count": [6145],
        "_quant_profiles": ["MXFP8"],
    },
    # TopK>6144，S1=3，覆盖缩小后的 S1 尾块。
    "S1_29": {
        **BASE,
        "q_seq": [3],
        "k_seq": [6145],
        "max_seqlen_q": [3],
        "sparse_count": [6145],
        "_quant_profiles": ["MXFP8"],
    },
    # ------ S2
    # actual S2=0，覆盖无计算任务及输出初始化。
    "S2_30": {
        **BASE,
        "k_seq": [1],
        "seqused_k": [[0]],
        "sparse_count": [1],
        "return_value": [1],
        "_quant_profiles": ["MXFP8"],
    },
    # actual S2=1，覆盖最短非零尾块。
    "S2_31": {
        **BASE,
        "k_seq": [1],
        "sparse_count": [1],
        "_quant_profiles": ["MXFP8"],
    },
    # actual S2=127，覆盖基本块左边界。
    "S2_32": {
        **BASE,
        "k_seq": [127],
        "sparse_count": [64],
        "_quant_profiles": ["MXFP8"],
    },
    # actual S2=128，覆盖单个完整基本块。
    "S2_33": {
        **BASE,
        "k_seq": [128],
        "sparse_count": [64],
        "_quant_profiles": ["MXFP8"],
    },
    # actual S2=129，覆盖基本块右边界。
    "S2_34": {
        **BASE,
        "k_seq": [129],
        "sparse_count": [64],
        "_quant_profiles": ["MXFP8"],
    },
    # actual S2=256，覆盖多个基本块且无尾块。
    "S2_35": {
        **BASE,
        "k_seq": [256],
        "sparse_count": [64],
        "_quant_profiles": ["MXFP8"],
    },
    # actual S2=257，覆盖多个基本块后的最短尾块。
    "S2_36": {
        **BASE,
        "k_seq": [257],
        "sparse_count": [64],
        "_quant_profiles": ["MXFP8"],
    },
    # ------ TopK
    # TopK=1，覆盖下界和非 16/256 对齐。
    "TOPK_37": {
        **BASE,
        "k_seq": [1],
        "sparse_count": [1],
        "_quant_profiles": ["MXFP8"],
    },
    # TopK=2048，trunkLen=16K 的右边界。
    "TOPK_38": {
        **BASE,
        "k_seq": [2048],
        "sparse_count": [2048],
        "_quant_profiles": ["MXFP8"],
    },
    # TopK=2049，trunkLen 切换为 12K。
    "TOPK_39": {
        **BASE,
        "k_seq": [2049],
        "sparse_count": [2049],
        "_quant_profiles": ["MXFP8"],
    },
    # TopK=3072，trunkLen=12K 的右边界。
    "TOPK_40": {
        **BASE,
        "k_seq": [3072],
        "sparse_count": [3072],
        "_quant_profiles": ["MXFP8"],
    },
    # TopK=3073，trunkLen 切换为 8K。
    "TOPK_41": {
        **BASE,
        "k_seq": [3073],
        "sparse_count": [3073],
        "_quant_profiles": ["MXFP8"],
    },
    # TopK=4096，trunkLen=8K 的右边界。
    "TOPK_42": {
        **BASE,
        "k_seq": [4096],
        "sparse_count": [4096],
        "_quant_profiles": ["MXFP8"],
    },
    # TopK=4097，trunkLen 切换为 4K，且 TopK 大于 trunkLen。
    "TOPK_43": {
        **BASE,
        "k_seq": [4097],
        "sparse_count": [4097],
        "_quant_profiles": ["MXFP8"],
    },
    # TopK=5120，trunkLen=4K 的右边界。
    "TOPK_44": {
        **BASE,
        "k_seq": [5120],
        "sparse_count": [5120],
        "_quant_profiles": ["MXFP8"],
    },
    # TopK=5121，trunkLen 切换为 2K。
    "TOPK_45": {
        **BASE,
        "k_seq": [5121],
        "sparse_count": [5121],
        "_quant_profiles": ["MXFP8"],
    },
    # TopK=6144，trunkLen=2K 的右边界。
    "TOPK_46": {
        **BASE,
        "k_seq": [6144],
        "sparse_count": [6144],
        "_quant_profiles": ["MXFP8"],
    },
    # TopK=6145，trunkLen 回到 12K，并切换 S1Base/MBaseMax。
    "TOPK_47": {
        **BASE,
        "k_seq": [6145],
        "sparse_count": [6145],
        "_quant_profiles": ["MXFP8"],
    },
    # TopK=8192，覆盖上界。
    "TOPK_48": {
        **BASE,
        "k_seq": [8192],
        "sparse_count": [8192],
        "_quant_profiles": ["MXFP8"],
    },
    # validS2<TopK，覆盖 indices/value 补齐。
    "TOPK_49": {
        **BASE,
        "k_seq": [65],
        "sparse_count": [129],
        "return_value": [1],
        "_quant_profiles": ["MXFP8"],
    },
    # TopK<=2K 且 S2>16K，覆盖多轮 TopK merge 和 LD 汇总；遍历两种返回模式，
    # 验证多轮 merge 后的 value 搬运、事件同步和仅 indices 路径。
    "TOPK_50": {
        **BASE,
        "k_seq": [16385],
        "sparse_count": [2048],
        "return_value": [0, 1],
        "_quant_profiles": ["MXFP8"],
    },
    # TopK>trunkLen 且 S2 超过 TopK 对齐长度，覆盖大 TopK 多轮 merge及两种返回模式。
    "TOPK_51": {
        **BASE,
        "k_seq": [5377],
        "sparse_count": [5121],
        "return_value": [0, 1],
        "_quant_profiles": ["MXFP8"],
    },
    # ------ mask / actual length / output
    # mask3，S1<=S2，覆盖有效下三角窗口。
    "MASK_52": {
        **BASE,
        "q_seq": [4],
        "k_seq": [128],
        "max_seqlen_q": [4],
        "sparse_mode": [3],
        "_quant_profiles": ["MXFP8"],
    },
    # mask3，S1>S2，覆盖前部 query 行无有效 K。
    "MASK_53": {
        **BASE,
        "q_seq": [5],
        "k_seq": [2],
        "max_seqlen_q": [5],
        "sparse_count": [2],
        "sparse_mode": [3],
        "return_value": [1],
        "_quant_profiles": ["MXFP8"],
    },
    # cmp_ratio=2 且 residual 非零，覆盖压缩长度与余数计算。
    "CMP_54": {
        **BASE,
        "q_seq": [4],
        "k_seq": [129],
        "seqused_k": [[129]],
        "cmp_residual_k": [[1]],
        "max_seqlen_q": [4],
        "sparse_mode": [3],
        "cmp_ratio": [2],
        "_quant_profiles": ["MXFP8"],
    },
    # cmp_ratio=128，覆盖压缩倍率上界。
    "CMP_55": {
        **BASE,
        "q_seq": [4],
        "k_seq": [129],
        "seqused_k": [[129]],
        "cmp_residual_k": [[127]],
        "max_seqlen_q": [4],
        "sparse_mode": [3],
        "cmp_ratio": [128],
        "_quant_profiles": ["MXFP8"],
    },
    # BSND seqused 覆盖 shape，并触发 actual S1 小于分配 S1 的输出清理。
    "ACTUAL_56": {
        **BASE,
        "q_seq": [5],
        "k_seq": [129],
        "seqused_q": [[3]],
        "seqused_k": [[127]],
        "max_seqlen_q": [5],
        "_quant_profiles": ["MXFP8"],
    },
    # TND 仅使用 cu_seqlens 差分得到实际长度。
    "ACTUAL_57": {
        **BASE,
        "batch_size": [2],
        "q_seq": [3],
        "k_seq": [129],
        "q_t_size": [3],
        "k_t_size": [129],
        "cu_seqlens_q": [[0, 1, 3]],
        "cu_seqlens_k": [[0, 1, 129]],
        "max_seqlen_q": [2],
        "layout_query": ["TND"],
        "layout_key": ["TND"],
        "_quant_profiles": ["MXFP8"],
    },
    # TND seqused 覆盖 cu_seqlens，并触发 TND 尾部输出清理。
    "ACTUAL_58": {
        **BASE,
        "batch_size": [2],
        "q_seq": [5],
        "k_seq": [256],
        "q_t_size": [5],
        "k_t_size": [256],
        "cu_seqlens_q": [[0, 2, 5]],
        "cu_seqlens_k": [[0, 128, 256]],
        "seqused_q": [[1, 2]],
        "seqused_k": [[127, 64]],
        "max_seqlen_q": [3],
        "layout_query": ["TND"],
        "layout_key": ["TND"],
        "_quant_profiles": ["MXFP8"],
    },
    # actual S1=0，覆盖单 batch 无 query 计算。
    "ACTUAL_59": {
        **BASE,
        "q_seq": [1],
        "seqused_q": [[0]],
        "return_value": [1],
        "_quant_profiles": ["MXFP8"],
    },
    # BSND output_idx_offset 非空。
    "OFFSET_60": {
        **BASE,
        "q_seq": [4],
        "output_idx_offset": [[0, 3, 0, 7]],
        "_quant_profiles": ["MXFP8"],
    },
    # TND output_idx_offset 非空并返回 values。
    "OFFSET_61": {
        **BASE,
        "batch_size": [2],
        "q_seq": [3],
        "k_seq": [128],
        "q_t_size": [3],
        "k_t_size": [128],
        "cu_seqlens_q": [[0, 1, 3]],
        "cu_seqlens_k": [[0, 64, 128]],
        "max_seqlen_q": [2],
        "layout_query": ["TND"],
        "layout_key": ["TND"],
        "return_value": [1],
        "output_idx_offset": [[1, 0, 5]],
        "_quant_profiles": ["MXFP8"],
    },
    # ------ max_seqlen_q / PA block
    # max_seqlen_q=-1，cube 的 S2 内部分块为 128。
    "MAXSEQ_62": {
        **BASE,
        "q_seq": [4],
        "max_seqlen_q": [-1],
        "_quant_profiles": ["MXFP8"],
    },
    # max_seqlen_q=4，cube 的 S2 内部分块为 256。
    "MAXSEQ_63": {
        **BASE,
        "q_seq": [4],
        "k_seq": [257],
        "max_seqlen_q": [4],
        "_quant_profiles": ["MXFP8"],
    },
    # max_seqlen_q=5，cube 的 S2 内部分块切回 128。
    "MAXSEQ_64": {
        **BASE,
        "q_seq": [5],
        "k_seq": [257],
        "max_seqlen_q": [5],
        "_quant_profiles": ["MXFP8"],
    },
    # PA block_size=16，覆盖下界及跨多个物理块。
    "PA_65": {
        **BASE,
        "k_seq": [129],
        "block_size": [16],
        "block_num": [9],
        "seqused_k": [[129]],
        "layout_key": ["PA_BBND"],
        "_quant_profiles": ["MXFP8"],
    },
    # PA block_size=96，覆盖 16 的倍数但非 2 次幂。
    "PA_66": {
        **BASE,
        "k_seq": [129],
        "block_size": [96],
        "block_num": [2],
        "seqused_k": [[129]],
        "layout_key": ["PA_BBND"],
        "_quant_profiles": ["MXFP8"],
    },
    # PA block_size=128，与 S2 基本块相等。
    "PA_67": {
        **BASE,
        "k_seq": [129],
        "block_size": [128],
        "block_num": [2],
        "seqused_k": [[129]],
        "layout_key": ["PA_BBND"],
        "_quant_profiles": ["MXFP8"],
    },
    # PA block_size=1024，覆盖上界及 S2 小于物理块。
    "PA_68": {
        **BASE,
        "k_seq": [129],
        "block_size": [1024],
        "block_num": [1],
        "seqused_k": [[129]],
        "layout_key": ["PA_BBND"],
        "_quant_profiles": ["MXFP8"],
    },
    # ------ metadata schedule
    # maxS2=640，命中 FD 开启阈值的左侧，按 batch/row 分配且不切 S2。
    "META_69": {
        **BASE,
        "q_seq": [4],
        "k_seq": [640],
        "q_head_num": [64],
        "max_seqlen_q": [4],
        "_quant_profiles": ["MXFP8"],
    },
    # maxS2=641，刚好开启 FD；遍历 return_value，并覆盖 LD 的索引偏移。
    # INT8 的 RV0 随机输入可能在 TopK 边界产生近邻值；RV0 无 value 输出时，
    # 现有比较器无法用 NPU value 消歧。INT8 的 FD 路径由下一条 RV1 用例覆盖，
    # RV0 分支则由其余 dtype 及其他 INT8 场景覆盖，避免把比较器限制误报成算子失败。
    "META_70": {
        **BASE,
        "q_seq": [4],
        "k_seq": [641],
        "q_head_num": [64],
        "max_seqlen_q": [4],
        "return_value": [0, 1],
        "output_idx_offset": [[0, 3, 0, 7]],
        "_quant_profiles": ["FP8", "MXFP8", "HIFLOAT8", "MXFP4"],
    },
    "META_70_INT8": {
        **BASE,
        "q_seq": [4],
        "k_seq": [641],
        "q_head_num": [64],
        "max_seqlen_q": [4],
        "return_value": [1],
        "output_idx_offset": [[0, 3, 0, 7]],
        "_quant_profiles": ["INT8"],
    },
    # maxS2>640 但 maxS2<TopK，关闭 FD 并覆盖 validS2<TopK。
    "META_71": {
        **BASE,
        "q_seq": [4],
        "k_seq": [641],
        "q_head_num": [64],
        "max_seqlen_q": [4],
        "sparse_count": [642],
        "return_value": [1],
        "_quant_profiles": ["MXFP8"],
    },
    # 所有 batch 的 S1 都为 0，覆盖 metadata 全空任务和主算子全局输出初始化。
    "META_72": {
        **BASE,
        "batch_size": [3],
        "q_seq": [4],
        "k_seq": [1],
        "seqused_q": [[0, 0, 0]],
        "seqused_k": [[1, 1, 1]],
        "max_seqlen_q": [4],
        "sparse_count": [1],
        "return_value": [1],
        "_quant_profiles": ["MXFP8"],
    },
    # PR10761 形态：首 batch 的 FD 任务结束后仍有连续的零 K cost batch。
    "META_73": {
        **BASE,
        "batch_size": [4],
        "q_seq": [4],
        "k_seq": [641],
        "q_head_num": [64],
        "seqused_k": [[641, 0, 0, 0]],
        "max_seqlen_q": [4],
        "return_value": [1],
        "_quant_profiles": ["MXFP8"],
    },
    # PR10761 对偶形态：首 batch 的 FD 任务结束后仍有连续的零 Q cost batch。
    "META_74": {
        **BASE,
        "batch_size": [4],
        "q_seq": [4],
        "k_seq": [641],
        "q_head_num": [64],
        "seqused_q": [[4, 0, 0, 0]],
        "seqused_k": [[641, 641, 641, 641]],
        "max_seqlen_q": [4],
        "return_value": [1],
        "_quant_profiles": ["MXFP8"],
    },
    # 前置零 cost batch，穿刺 AssignByBatch 跳过零负载后的游标和 cache 更新。
    "META_75": {
        **BASE,
        "batch_size": [4],
        "q_seq": [4],
        "k_seq": [641],
        "q_head_num": [64],
        "seqused_q": [[0, 0, 4, 4]],
        "seqused_k": [[641, 641, 641, 641]],
        "max_seqlen_q": [4],
        "_quant_profiles": ["MXFP8"],
    },
    # 非零/零 K cost 交替且尾部为零，穿刺跨 batch 游标及尾部范围补齐。
    "META_76": {
        **BASE,
        "batch_size": [4],
        "q_seq": [4],
        "k_seq": [641],
        "q_head_num": [64],
        "seqused_k": [[641, 0, 641, 0]],
        "max_seqlen_q": [4],
        "return_value": [1],
        "_quant_profiles": ["MXFP8"],
    },
    # TND 的 cu_seqlens_k 尾部重复，覆盖由前缀和产生的连续零 cost batch。
    "META_77": {
        **BASE,
        "batch_size": [4],
        "q_seq": [8],
        "k_seq": [641],
        "q_t_size": [8],
        "k_t_size": [641],
        "q_head_num": [64],
        "cu_seqlens_q": [[0, 2, 4, 6, 8]],
        "cu_seqlens_k": [[0, 641, 641, 641, 641]],
        "max_seqlen_q": [2],
        "layout_query": ["TND"],
        "layout_key": ["TND"],
        "return_value": [1],
        "_quant_profiles": ["FP8", "INT8", "MXFP8", "HIFLOAT8", "MXFP4"],
    },
    # TND seqused_k 覆盖非零 cu_seqlens_k，并在末尾制造零 cost batch。
    "META_78": {
        **BASE,
        "batch_size": [4],
        "q_seq": [16],
        "k_seq": [2564],
        "q_t_size": [16],
        "k_t_size": [2564],
        "q_head_num": [64],
        "cu_seqlens_q": [[0, 4, 8, 12, 16]],
        "cu_seqlens_k": [[0, 641, 1282, 1923, 2564]],
        "seqused_k": [[641, 641, 0, 0]],
        "max_seqlen_q": [4],
        "layout_query": ["TND"],
        "layout_key": ["TND"],
        "return_value": [1],
        "_quant_profiles": ["MXFP8"],
    },
    # S1 和 S2 同时为尾块，覆盖 metadata cost table 的 tail-tail 分支。
    "META_79": {
        **BASE,
        "q_seq": [5],
        "k_seq": [641],
        "q_head_num": [3],
        "max_seqlen_q": [5],
        "_quant_profiles": ["MXFP8"],
    },
    # 三个 S1 块配合长 S2，促使分核点依次落在 row 内和 row 边界。
    "META_80": {
        **BASE,
        "q_seq": [9],
        "k_seq": [1025],
        "q_head_num": [64],
        "max_seqlen_q": [9],
        "return_value": [1],
        "_quant_profiles": ["MXFP8"],
    },
    # mask3 下 batch 负载高度偏斜，包含短 K、S2 尾块和末尾零 cost。
    "META_81": {
        **BASE,
        "batch_size": [4],
        "q_seq": [8],
        "k_seq": [641],
        "q_head_num": [64],
        "seqused_q": [[1, 4, 8, 3]],
        "seqused_k": [[641, 2, 129, 0]],
        "max_seqlen_q": [8],
        "sparse_mode": [3],
        "return_value": [1],
        "_quant_profiles": ["MXFP8"],
    },
    # cmp_ratio/residual 令压缩前长度跨越极大，穿刺 mask cost 与压缩后分块不一致。
    "META_82": {
        **BASE,
        "batch_size": [3],
        "q_seq": [4],
        "k_seq": [641],
        "q_head_num": [64],
        "seqused_k": [[641, 129, 1]],
        "cmp_residual_k": [[127, 1, 0]],
        "max_seqlen_q": [4],
        "sparse_mode": [3],
        "cmp_ratio": [128],
        "return_value": [1],
        "_quant_profiles": ["MXFP8"],
    },
    # PA 最小物理块配合尾部零 cost，覆盖 FD workspace 与 block table 的组合路径。
    "META_83": {
        **BASE,
        "batch_size": [4],
        "q_seq": [4],
        "k_seq": [641],
        "q_head_num": [64],
        "block_size": [16],
        "block_num": [41],
        "seqused_k": [[641, 0, 0, 0]],
        "max_seqlen_q": [4],
        "layout_key": ["PA_BBND"],
        "return_value": [1],
        "_quant_profiles": ["FP8", "INT8", "MXFP8", "HIFLOAT8", "MXFP4"],
    },
    # TND+PA 同时包含不均匀 Q、零 Q 和不均匀 K，穿刺 FD 的 TND 输出偏移。
    # golden 会随机置换 9 个物理 block id，多 batch 共同消费 block table，覆盖 PA 非顺序物理映射。
    "META_84": {
        **BASE,
        "batch_size": [4],
        "q_seq": [10],
        "k_seq": [641],
        "q_t_size": [10],
        "q_head_num": [64],
        "block_size": [128],
        "block_num": [9],
        "cu_seqlens_q": [[0, 1, 5, 5, 10]],
        "seqused_k": [[641, 1, 0, 129]],
        "max_seqlen_q": [5],
        "layout_query": ["TND"],
        "layout_key": ["PA_BBND"],
        "return_value": [1],
        "_quant_profiles": ["FP8", "INT8", "MXFP8", "HIFLOAT8", "MXFP4"],
    },
    # TopK>6144 时 metadata 与主 kernel 同时切换 S1Base=2，并执行 FD/LD。
    "META_85": {
        **BASE,
        "q_seq": [3],
        "k_seq": [8192],
        "q_head_num": [64],
        "max_seqlen_q": [3],
        "sparse_count": [6145],
        "return_value": [1],
        "_quant_profiles": ["MXFP8"],
    },
    # FD 关闭时一个 batch 含两个 S1 行，穿刺 AssignByBatch 失败后由 AssignByRow 分配。
    "META_86": {
        **BASE,
        "q_seq": [8],
        "k_seq": [640],
        "q_head_num": [64],
        "max_seqlen_q": [8],
        "_quant_profiles": ["MXFP8"],
    },
    # FD 开启且一行含 14 个 S2 块，使核均摊负载足以进入 AssignByBlock。
    "META_87": {
        **BASE,
        "q_seq": [4],
        "k_seq": [1665],
        "q_head_num": [64],
        "max_seqlen_q": [4],
        "return_value": [1],
        "_quant_profiles": ["MXFP8"],
    },
    # 多 S1 行和长 S2 形成多组 FD 任务，穿刺 SplitFD 的多任务及多 vector 分配。
    "META_88": {
        **BASE,
        "q_seq": [101],
        "k_seq": [1665],
        "q_head_num": [64],
        "max_seqlen_q": [101],
        "return_value": [1],
        "_quant_profiles": ["MXFP8"],
    },
    # mask3+cmp_ratio 使前一个 S1 行压缩后无有效 S2、后一个 S1 行恢复为有效。
    "META_89": {
        **BASE,
        "q_seq": [8],
        "k_seq": [1],
        "q_head_num": [64],
        "seqused_k": [[1]],
        "max_seqlen_q": [8],
        "sparse_count": [1],
        "sparse_mode": [3],
        "cmp_ratio": [2],
        "cmp_residual_k": [[0]],
        "return_value": [1],
        "_quant_profiles": ["MXFP8"],
    },
    # ------ scale cache
    # topk>S2 关闭 FD，使同一核连续处理 17 个 S2 基本块；穿过 block 16 边界，
    # 验证非 MX 的 key scale cache 在环形缓冲区复用前重新从 GM 装载。
    # HIFLOAT8 使用 per-tensor scale，不经过该 per-token cache 分支。
    "SCALE_CACHE_90": {
        **BASE,
        "q_seq": [5],
        "k_seq": [2049],
        "max_seqlen_q": [5],
        "sparse_count": [2050],
        "return_value": [0],
        "_quant_profiles": ["FP8", "INT8"],
    },
}


# quant_mode 与输入 dtype 的合法组合。
QUANT_PROFILES = {
    "FP8": {
        "qk_dtype": [torch.float8_e4m3fn],
        "weight_dtype": [torch.float32],
        "dequant_dtype": [torch.float32],
        "quant_mode": [1],
        "q_scale_datarange": [[0, 1]],
        "k_scale_datarange": [[0, 1]],
    },
    "INT8": {
        "qk_dtype": [torch.int8],
        "weight_dtype": [torch.float16],
        "dequant_dtype": [torch.float16],
        "quant_mode": [2],
        "query_datarange": [[-100, 100]],
        "key_datarange": [[-100, 100]],
        "weights_datarange": [[-25, 25]],
    },
    "MXFP8": {
        "qk_dtype": [torch.float8_e4m3fn],
        "weight_dtype": [torch.float32],
        "dequant_dtype": [torch.float8_e8m0fnu],
        "quant_mode": [3],
    },
    "HIFLOAT8": {
        # 测试驱动先用 uint8 承载数据，再按 quant_mode=4 构造 HIFLOAT8 tensor。
        "qk_dtype": [torch.uint8],
        "weight_dtype": [torch.float32],
        "dequant_dtype": [torch.float32],
        "quant_mode": [4],
    },
    "MXFP4": {
        "qk_dtype": [torch.float4_e2m1fn_x2],
        "weight_dtype": [torch.float32],
        "dequant_dtype": [torch.float8_e8m0fnu],
        "quant_mode": [5],
    },
}


def _build_cases():
    """将每个场景与指定的 quant_mode/dtype 组合绑定。"""
    cases = {}
    for case_name, params in TestCases.items():
        case_template = dict(params)
        quant_names = case_template.pop("_quant_profiles")
        for quant_name in quant_names:
            case = dict(case_template)
            case.update(QUANT_PROFILES[quant_name])
            cases[f"{quant_name}_{case_name}"] = case
    return cases


TEST_PARAMS = _build_cases()

properties = torch.npu.get_device_properties()
ENABLED_PARAMSETS = []
if "Ascend950" in properties.name:
    ENABLED_PARAMSETS = [
        (name, TEST_PARAMS[name])
        for name in (
            # ------ dtype + layout
            "FP8_BSND_01",
            "FP8_PA_02",
            "FP8_TND_03",
            "FP8_PA_04",
            "INT8_BSND_05",
            "INT8_PA_06",
            "INT8_TND_07",
            "INT8_PA_08",
            "MXFP8_BSND_09",
            "MXFP8_PA_10",
            "MXFP8_TND_11",
            "MXFP8_PA_12",
            "HIFLOAT8_BSND_13",
            "HIFLOAT8_PA_14",
            "HIFLOAT8_TND_15",
            "HIFLOAT8_PA_16",
            "MXFP4_BSND_17",
            "MXFP4_PA_18",
            "MXFP4_TND_19",
            "MXFP4_PA_20",
            # ------ S1 / G
            "MXFP8_S1_21",
            "MXFP8_S1_22",
            "MXFP8_S1_23",
            "MXFP8_S1_24",
            "MXFP8_G_25",
            "MXFP8_G_26",
            "MXFP8_G_27",
            "MXFP8_S1_28",
            "MXFP8_S1_29",
            # ------ S2
            "MXFP8_S2_30",
            "MXFP8_S2_31",
            "MXFP8_S2_32",
            "MXFP8_S2_33",
            "MXFP8_S2_34",
            "MXFP8_S2_35",
            "MXFP8_S2_36",
            # ------ TopK
            "MXFP8_TOPK_37",
            "MXFP8_TOPK_38",
            "MXFP8_TOPK_39",
            "MXFP8_TOPK_40",
            "MXFP8_TOPK_41",
            "MXFP8_TOPK_42",
            "MXFP8_TOPK_43",
            "MXFP8_TOPK_44",
            "MXFP8_TOPK_45",
            "MXFP8_TOPK_46",
            "MXFP8_TOPK_47",
            "MXFP8_TOPK_48",
            "MXFP8_TOPK_49",
            "MXFP8_TOPK_50",
            "MXFP8_TOPK_51",
            # ------ mask / actual length / output
            "MXFP8_MASK_52",
            "MXFP8_MASK_53",
            "MXFP8_CMP_54",
            "MXFP8_CMP_55",
            "MXFP8_ACTUAL_56",
            "MXFP8_ACTUAL_57",
            "MXFP8_ACTUAL_58",
            "MXFP8_ACTUAL_59",
            "MXFP8_OFFSET_60",
            "MXFP8_OFFSET_61",
            # ------ max_seqlen_q / PA block
            "MXFP8_MAXSEQ_62",
            "MXFP8_MAXSEQ_63",
            "MXFP8_MAXSEQ_64",
            "MXFP8_PA_65",
            "MXFP8_PA_66",
            "MXFP8_PA_67",
            "MXFP8_PA_68",
            # ------ metadata schedule
            "FP8_META_70",
            "FP8_META_77",
            "FP8_META_83",
            "FP8_META_84",
            "INT8_META_70_INT8",
            "INT8_META_77",
            "INT8_META_83",
            "INT8_META_84",
            "HIFLOAT8_META_70",
            "HIFLOAT8_META_77",
            "HIFLOAT8_META_83",
            "HIFLOAT8_META_84",
            "MXFP4_META_70",
            "MXFP4_META_77",
            "MXFP4_META_83",
            "MXFP4_META_84",
            "MXFP8_META_69",
            "MXFP8_META_70",
            "MXFP8_META_71",
            "MXFP8_META_72",
            "MXFP8_META_73",
            "MXFP8_META_74",
            "MXFP8_META_75",
            "MXFP8_META_76",
            "MXFP8_META_77",
            "MXFP8_META_78",
            "MXFP8_META_79",
            "MXFP8_META_80",
            "MXFP8_META_81",
            "MXFP8_META_82",
            "MXFP8_META_83",
            "MXFP8_META_84",
            "MXFP8_META_85",
            "MXFP8_META_86",
            "MXFP8_META_87",
            "MXFP8_META_88",
            "MXFP8_META_89",
            # ------ scale cache
            "FP8_SCALE_CACHE_90",
            "INT8_SCALE_CACHE_90",
        )
    ]

ENABLED_PARAMS = [params for _, params in ENABLED_PARAMSETS]
