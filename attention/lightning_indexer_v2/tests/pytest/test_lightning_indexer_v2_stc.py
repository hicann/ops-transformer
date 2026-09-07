#!/usr/bin/python
# -*- coding: utf-8 -*-
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


####### 参数说明 ########
# 所有参数均使用 list；同一路径的相邻取值可在一个 TestCases 条目内展开。
# batch_size: BSND 的 B；TND 的 B 由 cu_seqlens_q 的长度确定。
# q_seq/k_seq: BSND 的 S1/S2；TND 下为标称长度。
# q_t_size/k_t_size: TND 的总 token 数，等于对应 cu_seqlens 的末值。
# q_head_num/k_head_num: N1/N2；当前实现要求 N2=1、G=N1/N2<=64。
# head_dim: query/key 的 D，当前实现要求 D=128。
# block_size/block_num: PA 物理块大小和数量；block_size 为 [16, 1024] 内 16 的倍数。
# cu_seqlens_q/k: TND 的 B+1 长度前缀和；非 TND 传 None。
# seqused_q/k: 每个 batch 的实际长度；优先级高于 cu_seqlens 和 shape。
# cmp_residual_k: mask3 且 cmp_ratio>1 时必传，取值范围为 [0, cmp_ratio)。
# layout_q/layout_k: 合法组合为 BSND+BSND、TND+TND、BSND+PA、TND+PA。
# topk: 每行 TopK 数量，Ascend 950 上范围为 [1, 8192]。
# mask_mode: 0 为无 mask，3 为下三角 sparse mask。
# cmp_ratio: K 压缩倍率，范围为 [1, 128]。
# return_value: 0 仅返回 indices，1 同时返回 sparse_values。
# output_idx_offset: 可选索引偏移；BSND 为 [B,S1,N2]，TND 为 [T,N2]。

BASE = {
    "batch_size": [1],
    "q_seq": [1],
    "k_seq": [128],
    "q_t_size": [1],
    "k_t_size": [128],
    "q_head_num": [1],
    "k_head_num": [1],
    "head_dim": [128],
    "block_size": [None],
    "block_num": [None],
    "qk_dtype": [torch.float16],
    "cu_seqlens_q": [None],
    "cu_seqlens_k": [None],
    "seqused_q": [None],
    "seqused_k": [None],
    "cmp_residual_k": [None],
    "output_idx_offset": [None],
    "layout_q": ["BSND"],
    "layout_k": ["BSND"],
    "topk": [64],
    "mask_mode": [0],
    "query_datarange": [[-2, 2]],
    "key_datarange": [[-2, 2]],
    "weights_datarange": [[-2, 2]],
    "cmp_ratio": [1],
    "return_value": [0],
    "max_seqlen_q": [1],
}


####### 覆盖范围 ########
# dtype/layout：覆盖 fp16、bf16 与四种合法 layout 组合。
# S1/G：覆盖 S1Base=4/2 的首块、整块、多块和尾块，以及 G 的奇偶与切换边界。
# S2/TopK：覆盖 128 基本块边界、三段 trunkLen、对齐补齐及单轮/多轮 merge。
# metadata：覆盖 batch/row/block/ForceAssign、FD/LD、零 cost 游标及尾部范围补齐。
# 其他：覆盖实际长度来源、mask、cmp_ratio、PA block、offset 和返回值分支。
TestCases = {
    # ------ dtype + layout
    "FP16_01": {
        **BASE,
        "qk_dtype": [torch.float16],
    },
    "BF16_02": {
        **BASE,
        "qk_dtype": [torch.bfloat16],
    },
    "PA_03": {
        **BASE,
        "block_size": [128],
        "block_num": [1],
        "seqused_k": [[128]],
        "layout_k": ["PA_BBND"],
    },
    "TND_04": {
        **BASE,
        "q_t_size": [1],
        "k_t_size": [128],
        "cu_seqlens_q": [[0, 1]],
        "cu_seqlens_k": [[0, 128]],
        "layout_q": ["TND"],
        "layout_k": ["TND"],
    },
    "TND_PA_05": {
        **BASE,
        "q_t_size": [1],
        "block_size": [128],
        "block_num": [1],
        "cu_seqlens_q": [[0, 1]],
        "seqused_k": [[128]],
        "layout_q": ["TND"],
        "layout_k": ["PA_BBND"],
    },
    # ------ S1 / G
    # S1Base=4：首个尾块、完整块、完整块后尾块及多块。
    "S1_06": {
        **BASE,
        "q_seq": [1],
        "max_seqlen_q": [1],
    },
    "S1_07": {
        **BASE,
        "q_seq": [4],
        "max_seqlen_q": [4],
    },
    "S1_08": {
        **BASE,
        "q_seq": [5],
        "max_seqlen_q": [5],
    },
    "S1_09": {
        **BASE,
        "q_seq": [8],
        "max_seqlen_q": [8],
    },
    # G=2/3 分别覆盖偶数和奇数向量规约。
    "G_10": {
        **BASE,
        "q_seq": [4],
        "q_head_num": [2, 3],
        "max_seqlen_q": [4],
    },
    # G=32/33 覆盖由 G>32 触发的 S1Base 4->2 切换。
    "G_11": {
        **BASE,
        "q_seq": [4],
        "q_head_num": [32],
        "max_seqlen_q": [4],
    },
    "G_12": {
        **BASE,
        "q_seq": [3],
        "q_head_num": [33],
        "max_seqlen_q": [3],
    },
    "G_13": {
        **BASE,
        "q_seq": [2],
        "q_head_num": [64],
        "max_seqlen_q": [2],
    },
    # G<=32 时，TopK=2048/2049 覆盖另一条 S1Base 4->2 切换条件。
    "S1_TOPK_14": {
        **BASE,
        "q_seq": [4],
        "k_seq": [2048],
        "topk": [2048],
        "max_seqlen_q": [4],
    },
    "S1_TOPK_15": {
        **BASE,
        "q_seq": [3],
        "k_seq": [2049],
        "topk": [2049],
        "max_seqlen_q": [3],
    },
    # ------ S2
    # 一组小尺寸覆盖零填充、128 基本块左右边界、整块和多块尾块。
    "S2_16": {
        **BASE,
        "k_seq": [1, 127, 128, 129, 256, 257],
        "topk": [64],
        "return_value": [1],
    },
    "S2_ZERO_17": {
        **BASE,
        "k_seq": [1],
        "seqused_k": [[0]],
        "topk": [1],
        "return_value": [1],
    },
    # ------ TopK
    "TOPK_18": {
        **BASE,
        "k_seq": [1],
        "topk": [1],
    },
    # 5120/5121 覆盖 trunkLen 8K->4K 的切换。
    "TOPK_19": {
        **BASE,
        "k_seq": [5121],
        "topk": [5120, 5121],
    },
    # 7168/7169 覆盖 trunkLen 4K->1K，并进入 TopK 大于 trunkLen 的路径。
    "TOPK_20": {
        **BASE,
        "k_seq": [7169],
        "topk": [7168, 7169],
    },
    "TOPK_21": {
        **BASE,
        "k_seq": [8192],
        "topk": [8192],
        "return_value": [1],
    },
    # validS2<TopK，覆盖 indices/value 的尾部补齐。
    "TOPK_22": {
        **BASE,
        "k_seq": [65],
        "topk": [129],
        "return_value": [1],
    },
    # S2 超过 8K，覆盖 TopK<=2K 的多轮选取和 LD 汇总；同时遍历是否返回 value，
    # 验证多轮 merge 后的 value 搬运和仅 indices 路径。
    "TOPK_23": {
        **BASE,
        "k_seq": [8193],
        "topk": [2048],
        "return_value": [0, 1],
    },
    # TopK 对齐长度大于 1K trunk，覆盖大 TopK 的分块 LD merge；两种返回模式均需经过该路径。
    "TOPK_24": {
        **BASE,
        "k_seq": [8192],
        "topk": [7169],
        "return_value": [0, 1],
    },
    # ------ mask / actual length / output
    "MASK_25": {
        **BASE,
        "q_seq": [4],
        "k_seq": [128],
        "mask_mode": [3],
        "max_seqlen_q": [4],
    },
    # S1>S2，覆盖下三角窗口前部 query 行无有效 K。
    "MASK_26": {
        **BASE,
        "q_seq": [5],
        "k_seq": [2],
        "topk": [2],
        "mask_mode": [3],
        "return_value": [1],
        "max_seqlen_q": [5],
    },
    "CMP_27": {
        **BASE,
        "q_seq": [4],
        "k_seq": [65],
        "seqused_k": [[65]],
        "cmp_residual_k": [[1]],
        "mask_mode": [3],
        "cmp_ratio": [2],
        "max_seqlen_q": [4],
    },
    "CMP_28": {
        **BASE,
        "q_seq": [4],
        "k_seq": [129],
        "seqused_k": [[129]],
        "cmp_residual_k": [[127]],
        "mask_mode": [3],
        "cmp_ratio": [128],
        "max_seqlen_q": [4],
    },
    # BSND seqused 覆盖 shape，并使实际 S1/S2 同时形成尾块。
    "ACTUAL_29": {
        **BASE,
        "q_seq": [5],
        "k_seq": [129],
        "seqused_q": [[3]],
        "seqused_k": [[127]],
        "max_seqlen_q": [5],
    },
    # TND 仅由 cu_seqlens 差分取得实际长度。
    "ACTUAL_30": {
        **BASE,
        "batch_size": [2],
        "q_seq": [3],
        "k_seq": [129],
        "q_t_size": [3],
        "k_t_size": [129],
        "cu_seqlens_q": [[0, 1, 3]],
        "cu_seqlens_k": [[0, 1, 129]],
        "layout_q": ["TND"],
        "layout_k": ["TND"],
        "max_seqlen_q": [2],
    },
    # TND seqused 覆盖前缀和长度，并包含不均匀 batch。
    "ACTUAL_31": {
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
        "layout_q": ["TND"],
        "layout_k": ["TND"],
        "max_seqlen_q": [3],
    },
    "ACTUAL_ZERO_32": {
        **BASE,
        "seqused_q": [[0]],
        "return_value": [1],
    },
    "OFFSET_33": {
        **BASE,
        "q_seq": [4],
        "output_idx_offset": [[0, 3, 0, 7]],
        "max_seqlen_q": [4],
    },
    "OFFSET_34": {
        **BASE,
        "batch_size": [2],
        "q_seq": [3],
        "k_seq": [128],
        "q_t_size": [3],
        "k_t_size": [128],
        "cu_seqlens_q": [[0, 1, 3]],
        "cu_seqlens_k": [[0, 64, 128]],
        "layout_q": ["TND"],
        "layout_k": ["TND"],
        "output_idx_offset": [[1, 0, 5]],
        "return_value": [1],
        "max_seqlen_q": [2],
    },
    # seqused_q 保证 metadata 实际长度稳定，遍历 attr 的合法边界和典型值。
    "MAXSEQ_35": {
        **BASE,
        "q_seq": [5],
        "seqused_q": [[4]],
        "max_seqlen_q": [4, 5],
    },
    # TND 不依赖 max_seqlen_q 描述物理 S1 轴，可覆盖 -1/0 两个合法 attr 分支。
    "MAXSEQ_TND_35": {
        **BASE,
        "q_seq": [5],
        "k_seq": [128],
        "q_t_size": [5],
        "k_t_size": [128],
        "cu_seqlens_q": [[0, 5]],
        "cu_seqlens_k": [[0, 128]],
        "layout_q": ["TND"],
        "layout_k": ["TND"],
        "max_seqlen_q": [-1, 0],
    },
    # ------ PA block
    "PA_36": {
        **BASE,
        "k_seq": [129],
        "block_size": [16],
        "block_num": [9],
        "seqused_k": [[129]],
        "layout_k": ["PA_BBND"],
    },
    # 96 是 16 的倍数但不是 2 的幂。
    "PA_37": {
        **BASE,
        "k_seq": [129],
        "block_size": [96],
        "block_num": [2],
        "seqused_k": [[129]],
        "layout_k": ["PA_BBND"],
    },
    "PA_38": {
        **BASE,
        "k_seq": [129],
        "block_size": [128],
        "block_num": [2],
        "seqused_k": [[129]],
        "layout_k": ["PA_BBND"],
    },
    "PA_39": {
        **BASE,
        "k_seq": [129],
        "block_size": [1024],
        "block_num": [1],
        "seqused_k": [[129]],
        "layout_k": ["PA_BBND"],
    },
    # ------ metadata schedule
    # 全部 S1 为零，覆盖 metadata 空任务和主算子输出初始化。
    "META_ZERO_40": {
        **BASE,
        "batch_size": [3],
        "q_seq": [4],
        "k_seq": [1],
        "seqused_q": [[0, 0, 0]],
        "seqused_k": [[1, 1, 1]],
        "topk": [1],
        "return_value": [1],
        "max_seqlen_q": [4],
    },
    # !10761 形态：首个非零任务完成后仍有连续的零 K cost batch。
    "META_TRAILING_K_ZERO_41": {
        **BASE,
        "batch_size": [4],
        "q_seq": [4],
        "k_seq": [641],
        "q_head_num": [64],
        "seqused_k": [[641, 0, 0, 0]],
        "return_value": [1],
        "max_seqlen_q": [4],
    },
    # !10761 对偶形态：首个非零任务完成后仍有连续的零 Q cost batch。
    "META_TRAILING_Q_ZERO_42": {
        **BASE,
        "batch_size": [4],
        "q_seq": [4],
        "k_seq": [641],
        "q_head_num": [64],
        "seqused_q": [[4, 0, 0, 0]],
        "seqused_k": [[641, 641, 641, 641]],
        "return_value": [1],
        "max_seqlen_q": [4],
    },
    # 前置零 cost，覆盖跳过空 batch 后的 cache 更新。
    "META_LEADING_ZERO_43": {
        **BASE,
        "batch_size": [4],
        "q_seq": [4],
        "k_seq": [641],
        "q_head_num": [64],
        "seqused_q": [[0, 0, 4, 4]],
        "seqused_k": [[641, 641, 641, 641]],
        "max_seqlen_q": [4],
    },
    # 非零/零 K cost 交替且尾部为零，覆盖跨 batch 游标和尾部范围补齐。
    "META_INTERLEAVED_ZERO_44": {
        **BASE,
        "batch_size": [4],
        "q_seq": [4],
        "k_seq": [641],
        "q_head_num": [64],
        "seqused_k": [[641, 0, 641, 0]],
        "return_value": [1],
        "max_seqlen_q": [4],
    },
    # TND 前缀和尾部重复，直接由 cu_seqlens_k 产生连续零 cost batch。
    "META_TND_ZERO_45": {
        **BASE,
        "batch_size": [4],
        "q_seq": [8],
        "k_seq": [641],
        "q_t_size": [8],
        "k_t_size": [641],
        "q_head_num": [64],
        "cu_seqlens_q": [[0, 2, 4, 6, 8]],
        "cu_seqlens_k": [[0, 641, 641, 641, 641]],
        "layout_q": ["TND"],
        "layout_k": ["TND"],
        "return_value": [1],
        "max_seqlen_q": [2],
    },
    # TND seqused_k 覆盖非零前缀和，并在末尾制造零 cost。
    "META_TND_OVERRIDE_46": {
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
        "layout_q": ["TND"],
        "layout_k": ["TND"],
        "return_value": [1],
        "max_seqlen_q": [4],
    },
    # S1 和 S2 同时为尾块，覆盖 cost table 的 tail-tail 分支。
    "META_TAIL_47": {
        **BASE,
        "q_seq": [5],
        "k_seq": [641],
        "q_head_num": [3],
        "max_seqlen_q": [5],
    },
    # 多个完整 batch，覆盖 AssignByBatch 连续吸收任务。
    "META_BATCH_48": {
        **BASE,
        "batch_size": [8],
        "q_seq": [1],
        "k_seq": [129],
        "seqused_q": [[1, 1, 1, 1, 1, 1, 1, 1]],
        "seqused_k": [[129, 129, 129, 129, 129, 129, 129, 129]],
    },
    # 单 batch 含多个 S1 行，覆盖 AssignByBatch 后的 AssignByRow。
    "META_ROW_49": {
        **BASE,
        "q_seq": [9],
        "k_seq": [1025],
        "q_head_num": [64],
        "return_value": [1],
        "max_seqlen_q": [9],
    },
    # 一行含多个 S2 块，覆盖 AssignByBlock 与空核上的 ForceAssign。
    "META_BLOCK_50": {
        **BASE,
        "q_seq": [4],
        "k_seq": [1665],
        "q_head_num": [64],
        "return_value": [1],
        "max_seqlen_q": [4],
    },
    # 多 S1 行和长 S2 形成多组 FD/LD 任务。
    "META_MULTI_FD_51": {
        **BASE,
        "q_seq": [9],
        "k_seq": [1665],
        "q_head_num": [64],
        "return_value": [1],
        "max_seqlen_q": [9],
    },
    # mask3+cmp_ratio 使前一行无有效 S2，后一行恢复有效。
    "META_MASK_CMP_52": {
        **BASE,
        "q_seq": [8],
        "k_seq": [1],
        "q_head_num": [64],
        "seqused_k": [[1]],
        "cmp_residual_k": [[0]],
        "topk": [1],
        "mask_mode": [3],
        "cmp_ratio": [2],
        "return_value": [1],
        "max_seqlen_q": [8],
    },
    # 长 S2 下遍历 dtype，保证 metadata/主 kernel 的两条类型实例化都经过 FD/LD。
    "META_BSND_53": {
        **BASE,
        "q_seq": [4],
        "k_seq": [641],
        "q_head_num": [64],
        "qk_dtype": [torch.float16, torch.bfloat16],
        "return_value": [1],
        "max_seqlen_q": [4],
    },
    "META_PA_54": {
        **BASE,
        "q_seq": [4],
        "k_seq": [641],
        "q_head_num": [64],
        "block_size": [128],
        "block_num": [6],
        "qk_dtype": [torch.float16, torch.bfloat16],
        "seqused_k": [[641]],
        "layout_k": ["PA_BBND"],
        "return_value": [1],
        "max_seqlen_q": [4],
    },
    "META_TND_55": {
        **BASE,
        "batch_size": [2],
        "q_seq": [8],
        "k_seq": [1282],
        "q_t_size": [8],
        "k_t_size": [1282],
        "q_head_num": [64],
        "qk_dtype": [torch.float16, torch.bfloat16],
        "cu_seqlens_q": [[0, 4, 8]],
        "cu_seqlens_k": [[0, 641, 1282]],
        "seqused_q": [[4, 4]],
        "seqused_k": [[641, 641]],
        "layout_q": ["TND"],
        "layout_k": ["TND"],
        "return_value": [1],
        "max_seqlen_q": [4],
    },
    # golden 会随机置换物理 block id；多 batch 共同消费 block table，覆盖 PA 非顺序物理映射。
    "META_TND_PA_56": {
        **BASE,
        "batch_size": [2],
        "q_seq": [8],
        "k_seq": [641],
        "q_t_size": [8],
        "q_head_num": [64],
        "block_size": [128],
        "block_num": [12],
        "qk_dtype": [torch.float16, torch.bfloat16],
        "cu_seqlens_q": [[0, 4, 8]],
        "seqused_q": [[4, 4]],
        "seqused_k": [[641, 641]],
        "layout_q": ["TND"],
        "layout_k": ["PA_BBND"],
        "return_value": [1],
        "max_seqlen_q": [4],
    },
    # TopK>2048 与长 S2 组合，覆盖 metadata 和主 kernel 的 S1Base=2 路径。
    "META_TOPK_57": {
        **BASE,
        "q_seq": [3],
        "k_seq": [4097],
        "q_head_num": [32],
        "topk": [2049],
        "return_value": [1],
        "max_seqlen_q": [3],
    },
}


properties = torch.npu.get_device_properties()
ENABLED_PARAMSETS = []
if "Ascend950" in properties.name:
    ENABLED_PARAMSETS = [
        (name, TestCases[name])
        for name in (
            # ------ dtype + layout
            "FP16_01",
            "BF16_02",
            "PA_03",
            "TND_04",
            "TND_PA_05",
            # ------ S1 / G
            "S1_06",
            "S1_07",
            "S1_08",
            "S1_09",
            "G_10",
            "G_11",
            "G_12",
            "G_13",
            "S1_TOPK_14",
            "S1_TOPK_15",
            # ------ S2 / TopK
            "S2_16",
            "S2_ZERO_17",
            "TOPK_18",
            "TOPK_19",
            "TOPK_20",
            "TOPK_21",
            "TOPK_22",
            "TOPK_23",
            "TOPK_24",
            # ------ mask / actual length / output
            "MASK_25",
            "MASK_26",
            "CMP_27",
            "CMP_28",
            "ACTUAL_29",
            "ACTUAL_30",
            "ACTUAL_31",
            "ACTUAL_ZERO_32",
            "OFFSET_33",
            "OFFSET_34",
            "MAXSEQ_35",
            "MAXSEQ_TND_35",
            # ------ PA block
            "PA_36",
            "PA_37",
            "PA_38",
            "PA_39",
            # ------ metadata schedule
            "META_ZERO_40",
            "META_TRAILING_K_ZERO_41",
            "META_TRAILING_Q_ZERO_42",
            "META_LEADING_ZERO_43",
            "META_INTERLEAVED_ZERO_44",
            "META_TND_ZERO_45",
            "META_TND_OVERRIDE_46",
            "META_TAIL_47",
            "META_BATCH_48",
            "META_ROW_49",
            "META_BLOCK_50",
            "META_MULTI_FD_51",
            "META_MASK_CMP_52",
            "META_BSND_53",
            "META_PA_54",
            "META_TND_55",
            "META_TND_PA_56",
            "META_TOPK_57",
        )
    ]

ENABLED_PARAMS = [params for _, params in ENABLED_PARAMSETS]
