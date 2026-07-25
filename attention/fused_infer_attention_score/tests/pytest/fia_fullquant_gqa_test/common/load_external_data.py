#!/usr/bin/python3
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

import logging
import os

import torch

logger = logging.getLogger(__name__)


class ExternalData:
    """外部 NPU 排布 tensor 输入的容器"""

    def __init__(
        self,
        q_fp8,
        k_fp8,
        v_fp8,
        deq_q,
        deq_k,
        deq_v,
        block_table,
        B,
        N_q,
        N_kv,
        D,
        block_size,
        ACTUAL_SEQ_Q,
        ACTUAL_SEQ_KV,
        mask=None,
    ):
        self.q_fp8 = q_fp8  # NTD
        self.k_fp8 = k_fp8  # BNBD (KV cache)
        self.v_fp8 = v_fp8  # BNBD (KV cache)
        self.deq_q = deq_q  # NT
        self.deq_k = deq_k  # BNB
        self.deq_v = deq_v  # N
        self.p_scale = torch.tensor([1.0], dtype=torch.float32).cpu().contiguous()
        self.block_table = block_table  # (B, max_blocks) int32
        self.mask = mask  # 外部 mask，None 则脚本自行生成
        # ---- 推断出的配置参数 ----
        self.B = B
        self.N_q = N_q
        self.N_kv = N_kv
        self.D = D
        self.block_size = block_size
        self.ACTUAL_SEQ_Q = ACTUAL_SEQ_Q
        self.ACTUAL_SEQ_KV = ACTUAL_SEQ_KV


def load_data_from_dir(pt_dir):
    """从指定目录加载多个 pt 文件作为 tensor 输入（NPU 排布），替代 generate_data()

    目录结构:
        <pt_dir>/
            rank0_query_ntd.pt       -- NTD (N_q, T, D)
            rank0_key_cache.pt      -- BNBD (block_num, N_kv, block_size, D)  KV cache
            rank0_value_cache.pt    -- BNBD (block_num, N_kv, block_size, D)  KV cache
            rank0_q_scale.pt        -- NT (N_q, T)
            rank0_k_scale_cache.pt  -- BNB (block_num, N_kv, block_size)
            rank0_v_scale_cache.pt  -- N (N_kv)
            rank0_block_table.pt    -- (B, max_blocks) int32
            rank0_seq_lens.pt       -- list of int
            rank0_attn_mask.pt      -- [可选]
    """

    logger.info("Loading inputs from directory: %s", pt_dir)

    def _load_tensor(filename):
        filepath = os.path.join(pt_dir, filename)
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Required pt file not found: {filepath}")
        return torch.load(filepath, map_location="cpu", weights_only=False)

    def _load_tensor_optional(filename):
        filepath = os.path.join(pt_dir, filename)
        if not os.path.isfile(filepath):
            return None
        return torch.load(filepath, map_location="cpu", weights_only=False).cpu()

    # ---- 必须文件 (NPU 排布) ----
    q_fp8 = _load_tensor("rank0_query_ntd.pt")  # NTD: (N_q, T, D)
    k_fp8 = _load_tensor("rank0_key_cache.pt")  # BNBD: (Bn, N_kv, Bs, D)
    v_fp8 = _load_tensor("rank0_value_cache.pt")  # BNBD: (Bn, N_kv, Bs, D)
    deq_q = _load_tensor("rank0_q_scale.pt")  # NT: (N_q, T)
    deq_k = _load_tensor("rank0_k_scale_cache.pt")  # BNB: (Bn, N_kv, Bs)
    deq_v = _load_tensor("rank0_v_scale_cache.pt")  # N: (N_kv,)
    block_table = _load_tensor("rank0_block_table.pt")  # (B, max_blocks) int32
    seq_q = _load_tensor("rank0_seq_lens.pt")  # list of int
    seq_kv = _load_tensor("rank0_seq_lens.pt")  # list of int
    # ---- 可选文件 ----
    mask = _load_tensor_optional("rank0_attn_mask.pt")
    if mask is not None:
        logger.info("Loaded external mask, shape: %s", mask.shape)

    # 从 tensor shape 自动推导关键参数
    N_q, _, D = q_fp8.shape  # q: NTD → N_q, T, D
    Bn, N_kv, Bs, _ = k_fp8.shape  # k: BNBD
    BLOCK_SIZE = 128

    ACTUAL_SEQ_Q = seq_q.tolist() if hasattr(seq_q, "tolist") else list(seq_q)
    ACTUAL_SEQ_KV = seq_kv.tolist() if hasattr(seq_kv, "tolist") else list(seq_kv)
    B = len(ACTUAL_SEQ_Q)

    logger.info("q_fp8 shape: %s, dtype: %s", q_fp8.shape, q_fp8.dtype)
    logger.info("k_fp8 shape: %s, dtype: %s", k_fp8.shape, k_fp8.dtype)
    logger.info("v_fp8 shape: %s, dtype: %s", v_fp8.shape, v_fp8.dtype)
    logger.info("deq_q shape: %s", deq_q.shape)
    logger.info("deq_k shape: %s", deq_k.shape)
    logger.info("deq_v shape: %s", deq_v.shape)
    logger.info(
        "key is_contiguous: %s, value is_contiguous: %s",
        k_fp8.is_contiguous(),
        v_fp8.is_contiguous(),
    )
    logger.info("key stride: %s, value stride: %s", k_fp8.stride(), v_fp8.stride())
    logger.info("deq_k is_contiguous: %s", deq_k.is_contiguous())
    logger.info("deq_k stride: %s", deq_k.stride())
    logger.info("block_table shape: %s", block_table.shape)
    logger.info(
        "Inferred: B=%d, N_q=%d, N_kv=%d, D=%d, BLOCK_SIZE=%d",
        B,
        N_q,
        N_kv,
        D,
        BLOCK_SIZE,
    )
    logger.info("ACTUAL_SEQ_Q=%s, ACTUAL_SEQ_KV=%s", ACTUAL_SEQ_Q, ACTUAL_SEQ_KV)

    return ExternalData(
        q_fp8,
        k_fp8,
        v_fp8,
        deq_q,
        deq_k,
        deq_v,
        block_table,
        B=B,
        N_q=N_q,
        N_kv=N_kv,
        D=D,
        block_size=BLOCK_SIZE,
        ACTUAL_SEQ_Q=ACTUAL_SEQ_Q,
        ACTUAL_SEQ_KV=ACTUAL_SEQ_KV,
        mask=mask,
    )
