#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

####### 参数说明 ########
# nc_kv_dims: 0 -> dim0非连续; (0,1) -> dim0+dim1都非连续
# dim1-only物理上不可行(见data.py注释), 已全部移除
# 拦截用例(REJ): CheckKVContiguous 会拒绝的场景，预期在 NPU compute 时报错

# fmt: off
TestCases = {
    # ================================================================
    # 正常通过 — PA_BNBD（dim0 / dim01 基准）
    # ================================================================
    "NC_PA_BNBD_dim0": {"B": [2], "N1": [8], "N2": [1], "S1": [64], "S2": [1024],
                        "D": [128], "DV": [128], "Dtype": ["fp16"],
                        "layout_q": ["BNSD"], "layout_kv": ["PA_BNBD"], "layout_out": ["BNSD"],
                        "block_size": [128], "return_softmax_lse": [False],
                        "mask_mode": [0], "win_left": [-1], "win_right": [-1],
                        "q_range": [(-5.0, 5.0)], "k_range": [(-5.0, 5.0)], "v_range": [(-5.0, 5.0)],
                        "nc_kv_dims": [0]},
    "NC_PA_BNBD_dim01": {"B": [2], "N1": [8], "N2": [1], "S1": [64], "S2": [1024],
                         "D": [128], "DV": [128], "Dtype": ["fp16"],
                         "layout_q": ["BNSD"], "layout_kv": ["PA_BNBD"], "layout_out": ["BNSD"],
                         "block_size": [128], "return_softmax_lse": [False],
                         "mask_mode": [0], "win_left": [-1], "win_right": [-1],
                         "q_range": [(-5.0, 5.0)], "k_range": [(-5.0, 5.0)], "v_range": [(-5.0, 5.0)],
                         "nc_kv_dims": [(0, 1)]},

    # ================================================================
    # 正常通过 — PA_BBND（仅dim0合法）
    # ================================================================
    "NC_PA_BBND_dim0": {"B": [2], "N1": [8], "N2": [1], "S1": [64], "S2": [1024],
                        "D": [128], "DV": [128], "Dtype": ["fp16"],
                        "layout_q": ["BNSD"], "layout_kv": ["PA_BBND"], "layout_out": ["BNSD"],
                        "block_size": [128], "return_softmax_lse": [False],
                        "mask_mode": [0], "win_left": [-1], "win_right": [-1],
                        "q_range": [(-5.0, 5.0)], "k_range": [(-5.0, 5.0)], "v_range": [(-5.0, 5.0)],
                        "nc_kv_dims": [0]},

    # ================================================================
    # 正常通过 — PA_NZ（dim0 / dim01 基准）
    # ================================================================
    "NC_PA_NZ_dim0": {"B": [2], "N1": [8], "N2": [1], "S1": [64], "S2": [1024],
                      "D": [128], "DV": [128], "Dtype": ["fp16"],
                      "layout_q": ["BNSD"], "layout_kv": ["PA_NZ"], "layout_out": ["BNSD"],
                      "block_size": [128], "return_softmax_lse": [False],
                      "mask_mode": [0], "win_left": [-1], "win_right": [-1],
                      "q_range": [(-5.0, 5.0)], "k_range": [(-5.0, 5.0)], "v_range": [(-5.0, 5.0)],
                      "nc_kv_dims": [0]},
    "NC_PA_NZ_dim01": {"B": [2], "N1": [8], "N2": [1], "S1": [64], "S2": [1024],
                       "D": [128], "DV": [128], "Dtype": ["fp16"],
                       "layout_q": ["BNSD"], "layout_kv": ["PA_NZ"], "layout_out": ["BNSD"],
                       "block_size": [128], "return_softmax_lse": [False],
                       "mask_mode": [0], "win_left": [-1], "win_right": [-1],
                       "q_range": [(-5.0, 5.0)], "k_range": [(-5.0, 5.0)], "v_range": [(-5.0, 5.0)],
                       "nc_kv_dims": [(0, 1)]},

    # ================================================================
    # 正常通过 — 参数/Dtype 变体
    # ================================================================
    "NC_PA_BNBD_D64_dim0": {"B": [2], "N1": [8], "N2": [1], "S1": [64], "S2": [1024],
                            "D": [64], "DV": [64], "Dtype": ["fp16"],
                            "layout_q": ["BNSD"], "layout_kv": ["PA_BNBD"], "layout_out": ["BNSD"],
                            "block_size": [128], "return_softmax_lse": [False],
                            "mask_mode": [0], "win_left": [-1], "win_right": [-1],
                            "q_range": [(-5.0, 5.0)], "k_range": [(-5.0, 5.0)], "v_range": [(-5.0, 5.0)],
                            "nc_kv_dims": [0]},
    "NC_PA_BNBD_bf16_dim0": {"B": [2], "N1": [8], "N2": [1], "S1": [64], "S2": [1024],
                             "D": [128], "DV": [128], "Dtype": ["bf16"],
                             "layout_q": ["BNSD"], "layout_kv": ["PA_BNBD"], "layout_out": ["BNSD"],
                             "block_size": [128], "return_softmax_lse": [False],
                             "mask_mode": [0], "win_left": [-1], "win_right": [-1],
                             "q_range": [(-5.0, 5.0)], "k_range": [(-5.0, 5.0)], "v_range": [(-5.0, 5.0)],
                             "nc_kv_dims": [0]},
    "NC_PA_NZ_bs256_dim0": {"B": [2], "N1": [8], "N2": [1], "S1": [64], "S2": [1024],
                            "D": [128], "DV": [128], "Dtype": ["fp16"],
                            "layout_q": ["BNSD"], "layout_kv": ["PA_NZ"], "layout_out": ["BNSD"],
                            "block_size": [256], "return_softmax_lse": [False],
                            "mask_mode": [0], "win_left": [-1], "win_right": [-1],
                            "q_range": [(-5.0, 5.0)], "k_range": [(-5.0, 5.0)], "v_range": [(-5.0, 5.0)],
                            "nc_kv_dims": [0]},
    "NC_PA_BNBD_D64_dim01": {"B": [2], "N1": [8], "N2": [1], "S1": [64], "S2": [1024],
                             "D": [64], "DV": [64], "Dtype": ["fp16"],
                             "layout_q": ["BNSD"], "layout_kv": ["PA_BNBD"], "layout_out": ["BNSD"],
                             "block_size": [128], "return_softmax_lse": [False],
                             "mask_mode": [0], "win_left": [-1], "win_right": [-1],
                             "q_range": [(-5.0, 5.0)], "k_range": [(-5.0, 5.0)], "v_range": [(-5.0, 5.0)],
                             "nc_kv_dims": [(0, 1)]},
    "NC_PA_NZ_bf16_dim01": {"B": [2], "N1": [8], "N2": [1], "S1": [64], "S2": [1024],
                            "D": [128], "DV": [128], "Dtype": ["bf16"],
                            "layout_q": ["BNSD"], "layout_kv": ["PA_NZ"], "layout_out": ["BNSD"],
                            "block_size": [128], "return_softmax_lse": [False],
                            "mask_mode": [0], "win_left": [-1], "win_right": [-1],
                            "q_range": [(-5.0, 5.0)], "k_range": [(-5.0, 5.0)], "v_range": [(-5.0, 5.0)],
                             "nc_kv_dims": [(0, 1)]},
    # NZ + D=256
    "NC_PA_NZ_D256_dim0": {"B": [2], "N1": [8], "N2": [1], "S1": [64], "S2": [1024],
                           "D": [256], "DV": [256], "Dtype": ["fp16"],
                           "layout_q": ["BNSD"], "layout_kv": ["PA_NZ"], "layout_out": ["BNSD"],
                           "block_size": [128], "return_softmax_lse": [False],
                           "mask_mode": [0], "win_left": [-1], "win_right": [-1],
                           "q_range": [(-5.0, 5.0)], "k_range": [(-5.0, 5.0)], "v_range": [(-5.0, 5.0)],
                           "nc_kv_dims": [0]},
    # BBND + D64
    "NC_PA_BBND_D64_dim0": {"B": [2], "N1": [8], "N2": [1], "S1": [64], "S2": [1024],
                            "D": [64], "DV": [64], "Dtype": ["fp16"],
                            "layout_q": ["BNSD"], "layout_kv": ["PA_BBND"], "layout_out": ["BNSD"],
                            "block_size": [128], "return_softmax_lse": [False],
                            "mask_mode": [0], "win_left": [-1], "win_right": [-1],
                            "q_range": [(-5.0, 5.0)], "k_range": [(-5.0, 5.0)], "v_range": [(-5.0, 5.0)],
                            "nc_kv_dims": [0]},
    # BNBD + block_size=256 + dim01
    "NC_PA_BNBD_bs256_dim01": {"B": [2], "N1": [8], "N2": [1], "S1": [64], "S2": [1024],
                               "D": [128], "DV": [128], "Dtype": ["fp16"],
                               "layout_q": ["BNSD"], "layout_kv": ["PA_BNBD"], "layout_out": ["BNSD"],
                               "block_size": [256], "return_softmax_lse": [False],
                               "mask_mode": [0], "win_left": [-1], "win_right": [-1],
                               "q_range": [(-5.0, 5.0)], "k_range": [(-5.0, 5.0)], "v_range": [(-5.0, 5.0)],
                               "nc_kv_dims": [(0, 1)]},

    # ================================================================
    # 拦截用例 — CheckKVContiguous 拒绝
    # ================================================================
    # 非PA必须连续: BNSD dim0
    "NC_REJ_nonPA_BNSD_dim0": {"B": [2], "N1": [8], "N2": [1], "S1": [64], "S2": [512],
                               "D": [128], "DV": [128], "Dtype": ["fp16"],
                               "layout_q": ["BNSD"], "layout_kv": ["BNSD"], "layout_out": ["BNSD"],
                               "return_softmax_lse": [False],
                               "mask_mode": [0], "win_left": [-1], "win_right": [-1],
                               "q_range": [(-5.0, 5.0)], "k_range": [(-5.0, 5.0)], "v_range": [(-5.0, 5.0)],
                               "nc_kv_dims": [0]},
    # 非PA必须连续: BSND dim01（不同Q/KV layout）
    "NC_REJ_nonPA_BSND_dim01": {"B": [2], "N1": [8], "N2": [1], "S1": [64], "S2": [512],
                                "D": [128], "DV": [128], "Dtype": ["fp16"],
                                "layout_q": ["BSND"], "layout_kv": ["BSND"], "layout_out": ["BSND"],
                                "return_softmax_lse": [False],
                                "mask_mode": [0], "win_left": [-1], "win_right": [-1],
                                "q_range": [(-5.0, 5.0)], "k_range": [(-5.0, 5.0)], "v_range": [(-5.0, 5.0)],
                                "nc_kv_dims": [(0, 1)]},
    # BBND仅dim0: dim01
    "NC_REJ_PA_BBND_dim01": {"B": [2], "N1": [8], "N2": [1], "S1": [64], "S2": [1024],
                             "D": [128], "DV": [128], "Dtype": ["fp16"],
                             "layout_q": ["BNSD"], "layout_kv": ["PA_BBND"], "layout_out": ["BNSD"],
                             "block_size": [128], "return_softmax_lse": [False],
                             "mask_mode": [0], "win_left": [-1], "win_right": [-1],
                             "q_range": [(-5.0, 5.0)], "k_range": [(-5.0, 5.0)], "v_range": [(-5.0, 5.0)],
                             "nc_kv_dims": [(0, 1)]},
    # 非PA必须连续: TND dim0
    "NC_REJ_nonPA_TND_dim0": {"B": [2], "N1": [8], "N2": [1], "S1": [64], "S2": [512],
                              "D": [128], "DV": [128], "Dtype": ["fp16"],
                              "layout_q": ["TND"], "layout_kv": ["TND"], "layout_out": ["TND"],
                              "cu_seqlens_q": [[0, 64, 128]],
                              "cu_seqlens_kv": [[0, 512, 1024]],
                              "return_softmax_lse": [False],
                              "mask_mode": [0], "win_left": [-1], "win_right": [-1],
                              "q_range": [(-5.0, 5.0)], "k_range": [(-5.0, 5.0)], "v_range": [(-5.0, 5.0)],
                              "nc_kv_dims": [0]},
    # 非PA必须连续: TND dim01（序列总长度场景+dim01）
    "NC_REJ_nonPA_TND_dim01": {"B": [2], "N1": [8], "N2": [1], "S1": [64], "S2": [512],
                               "D": [128], "DV": [128], "Dtype": ["fp16"],
                               "layout_q": ["TND"], "layout_kv": ["TND"], "layout_out": ["TND"],
                               "cu_seqlens_q": [[0, 64, 128]],
                               "cu_seqlens_kv": [[0, 512, 1024]],
                               "return_softmax_lse": [False],
                               "mask_mode": [0], "win_left": [-1], "win_right": [-1],
                               "q_range": [(-5.0, 5.0)], "k_range": [(-5.0, 5.0)], "v_range": [(-5.0, 5.0)],
                               "nc_kv_dims": [(0, 1)]},
}
# fmt: on
