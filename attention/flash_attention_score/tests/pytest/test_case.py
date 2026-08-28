#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import torch

####### 参数说明 ########
# B:必选; batch_size,TND格式下可选
# N1:必选; head_num
# N2:可选; kv's head_num,支持GQA/MHA/MQA
# Sq:必选; query's sequence length;TND格式下可选
# Skv:可选; key&value's sequence length
# D:必选; 表示query&key&value的head_dim
# DV:可选; value's head_dim;设置该参数,value的head_dim以DV为准
# DRope:可选; query_rope&key_rope's head_dim,支持D=128,DRope=64
# input_layout:必选; 输入tensor的格式, [BNSD, BSH, BSND, SBH, TND]
# dtype:必选; 数据类型, [torch.float16, torch.bfloat16, torch.float]
# scale:可选; 注意力得分缩放系数
# actual_seq_qlen:可选; TND下必选;query实际的序列长度
# actual_seq_kvlen:可选; TND下必选;key&value实际的序列长度

# keep_prob:可选; dropout的保留概率：keep_prob = 1 - dropout_p;dropout的cpu实现方式基于Ascend 950PR/Ascend 950DT
# seed:可选; 随机种子，用于随机数生成

# sparse_mode:可选; sparse模式, [0, 1, 2, 3, 4, 5, 6, 7, 8]
#     - sparse_mode=3:等价于gpu的"causal=true"
# pre_tokens:可选; 配合sparse_mode使用,详细资料可参考算子文档
# next_tokens:可选; 配合sparse_mode使用,详细资料可参考算子文档
# prefix:可选; list类型,sparse_mode=5/6时需要设置,满足len(prefix)==batch_size

# pse_type:可选; 位置编码, [0, 1, 2, 3]
#     - 0/1: pse_layout取值在[bnss, 1nss, bn1s, bnhs, 1nhs],TND格式下取值在[bnhs, 1nhs]
#     - 2: pse_layout取值在[bn, n],等价于gpu的"alibi=true"
#     - 3: pse_layout取值在[bn, n]
# pse_layout:可选; 用于生成golden,取值范围[bnss, 1nss, bn1s, bnhs, 1nhs, bn, n],约束如上
# q_start_idx:可选; query的位置起始偏移
# kv_start_idx:可选; key&value的位置起始偏移

TestCases = {
    "GQA_01": {
        "B": 4,
        "N1": 16,
        "N2": 8,
        "Sq": 256,
        "Skv": 256,
        "D": 128,
        "input_layout": "BSND",
        "dtype": torch.bfloat16,
        "sparse_mode": 3,  # causal=true
    },
    "MLA_02": {
        "N1": 16,
        "N2": 8,
        "actual_seq_qlen": [256],
        "D": 128,
        "DRope": 64,  # rope head-dim
        "input_layout": "TND",
        "dtype": torch.bfloat16,
        "sparse_mode": 3,
    },
    "MLA_03": {
        "B": 4,
        "N1": 16,
        "N2": 8,
        "Sq": 256,
        "D": 192,  # query&key's head-dim
        "DV": 128,  # value's head-dim
        "input_layout": "BSH",
        "dtype": torch.float16,
        "sparse_mode": 3,
    },
    "ALIBI_04": {
        "B": 4,
        "N1": 8,
        "Sq": 256,
        "D": 256,  # query&key's head-dim
        "input_layout": "BNSD",
        "dtype": torch.bfloat16,
        "sparse_mode": 3,
        "pse_type": 2,
        "pse_layout": "bn",  # for generate golden data
    },
    "ALIBI_05": {
        "B": 4,
        "N1": 8,
        "Sq": 256,
        "D": 128,  # query&key's head-dim
        "input_layout": "BSND",
        "dtype": torch.float16,
        "sparse_mode": 3,
        "pse_type": 3,  # special alibi
        "pse_layout": "n",  # for generate golden data
    },
    "SPARSE_06": {
        "B": 4,
        "N1": 8,
        "Sq": 256,
        "D": 768,  # query&key's head-dim
        "input_layout": "SBH",
        "dtype": torch.bfloat16,
        "sparse_mode": 0,
        "pre_tokens": 128,
        "next_tokens": 128,
        "pse_type": 3,  # special alibi
        "pse_layout": "n",  # for generate golden data
    },
    "SPARSE_07": {
        "B": 4,
        "N1": 8,
        "Sq": 256,
        "D": 128,  # query&key's head-dim
        "input_layout": "BSND",
        "dtype": torch.float32,
        "sparse_mode": 4,  # if sparse_mode=4, window_size=(pre_tokens, next_tokens)
        "pre_tokens": 128,
        "next_tokens": 128,
        "pse_type": 3,  # special alibi
        "pse_layout": "n",  # for generate golden data
    },
    "SPARSE_08": {
        "B": 4,
        "N1": 8,
        "Sq": 256,
        "D": 128,  # query&key's head-dim
        "input_layout": "BSND",
        "dtype": torch.bfloat16,
        "sparse_mode": 5,
        "prefix": [100, 128, 130, 150],
        "pse_type": 3,  # special alibi
        "pse_layout": "n",  # for generate golden data
    },
    "DROPOUT_09": {
        "B": 4,
        "N1": 8,
        "Sq": 256,
        "Skv": 512,
        "D": 128,  # query&key's head-dim
        "input_layout": "BSND",
        "dtype": torch.float32,
        "sparse_mode": 5,
        "prefix": [100, 128, 401, 300],
        "pse_type": 0,  # special alibi
        "pse_layout": "bnss",  # for generate golden data
        "keep_prob": 0.9,  # keep_prob = 1 - dropout_p
    },
    "TND_01": {
        "N1": 8,
        "N2": 4,
        "actual_seq_qlen": [128, 256, 512],
        "D": 128,  # query&key's head-dim
        "DRope": 64,  # rope head-dim
        "input_layout": "TND",
        "dtype": torch.bfloat16,
        "sparse_mode": 3,
    },
    "TND_02": {
        "N1": 8,
        "actual_seq_qlen": [2048],
        "D": 192,  # query&key's head-dim
        "input_layout": "TND",
        "dtype": torch.bfloat16,
        "sparse_mode": 6,
        "prefix": [1568],
    },
    "TND_03": {
        "N1": 8,
        "N2": 4,
        "actual_seq_qlen": [256, 512, 768],
        "actual_seq_kvlen": [256, 512, 768],
        "D": 128,  # query&key's head-dim
        "input_layout": "TND",
        "dtype": torch.bfloat16,
        "sparse_mode": 3,
        "pse_type": 3,  # special alibi
        "pse_layout": "n",  # for generate golden data
        "keep_prob": 0.9,  # keep_prob = 1 - dropout_p
    },
    "TND_04": {
        "N1": 8,
        "N2": 4,
        "actual_seq_qlen": [128, 256, 512],
        "actual_seq_kvlen": [256, 512, 768],
        "D": 128,  # query&key's head-dim
        "input_layout": "TND",
        "dtype": torch.bfloat16,
        "sparse_mode": 8,
        "next_tokens": 64,
        "q_start_idx": 64,
        "kv_start_idx": 32,
        "keep_prob": 0.9,  # keep_prob = 1 - dropout_p
    },
    "DYN_OPT_BNSD": {
        "B": 1,
        "N1": 1,
        "Sq": 30000,
        "D": 128,
        "input_layout": "BNSD",
        "dtype": torch.bfloat16,
    },
    "DYN_OPT_BSND": {
        "B": 1,
        "N1": 24,
        "Sq": 576,
        "D": 128,
        "input_layout": "BSND",
        "dtype": torch.bfloat16,
    },
}
