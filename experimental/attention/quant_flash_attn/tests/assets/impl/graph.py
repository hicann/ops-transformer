#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import logging
import os
import sys
from typing import List, Optional

import torch
import torch_npu

# 复用 common/ 里的 golden 模块 (generate_data + 配置区 + get_dtype + 算子 import)
_ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTS_DIR = os.path.join(_ASSETS_DIR, "..", "ttk", "qfa_mxfp4_test")
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
from common import qfa_mxfp4_golden as golden_mod
from cann_ops_transformer.ops import quant_flash_attn_metadata, quant_flash_attn

logger = logging.getLogger(__name__)


def _apply_golden_globals(attrs):
    for k, v in attrs.items():
        setattr(golden_mod, k, v)


def _to_npu(t):
    if t is None:
        return None
    return t.npu() if hasattr(t, "npu") and t.device.type == "cpu" else t


class QuantFlashAttnMxfp4AclGraph(torch.nn.Module):
    def __init__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_descale: torch.Tensor,
        k_descale: torch.Tensor,
        v_descale: torch.Tensor,
        block_table: Optional[torch.Tensor],
        *,
        B: int,
        N_q: int,
        N_kv: int,
        G: int,
        D: int,
        V_D: int,
        Rope_D: int,
        act_seq_lens_q: List[int],
        act_seq_lens_kv: List[int],
        input_layout: str,
        layout_q_descale: str,
        layout_kv: str,
        layout_out: str,
        kv_storage_mode: str,
        block_size: int,
        q_dtype: str,
        kv_dtype: str,
        out_dtype: str,
        q_quant_mode: int,
        mask_mode: int,
        pre_tokens: int,
        next_tokens: int,
        enable_mask: bool,
        enable_lse: bool,
        inner_precise: int,
        device_id: int,
        graph_path: int = 0,
        softmax_scale: Optional[float] = None,
        data_range_q: str = "1.0",
        data_range_k: str = "1.0",
        data_range_v: str = "1.0",
        max_seqlen_q: int = -1,
        max_seqlen_kv: int = -1,
        cu_seqlens_q: Optional[List[int]] = None,
        cu_seqlens_kv: Optional[List[int]] = None,
        # 可选 tensor 入参 (shape 为空 -> golden 传 None)
        block_table_shape: Optional[List[int]] = None,
        block_table_dtype: Optional[str] = None,
        p_scale_value: Optional[float] = None,
        p_scale_shape: Optional[List[int]] = None,
        p_scale_dtype: Optional[str] = None,
        p_scale_datarange: Optional[str] = None,
        sinks_shape: Optional[List[int]] = None,
        sinks_dtype: Optional[str] = None,
        sinks_datarange: Optional[str] = None,
        attn_mask_shape: Optional[List[int]] = None,
        attn_mask_dtype: Optional[str] = None,
        attn_mask_datarange: Optional[str] = None,
        # dtype 透传 (表格不传 -> None, golden 侧用默认值)
        q_descale_dtype: Optional[str] = None,
        k_descale_dtype: Optional[str] = None,
        v_descale_dtype: Optional[str] = None,
        seqused_q_dtype: Optional[str] = None,
        seqused_kv_dtype: Optional[str] = None,
        cu_seqlens_q_dtype: Optional[str] = None,
        cu_seqlens_kv_dtype: Optional[str] = None,
        softmax_lse_dtype: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        torch_npu.npu.set_device(int(device_id))

        _apply_golden_globals(
            {
                "B": B,
                "N_q": N_q,
                "N_kv": N_kv,
                "G": G,
                "D": D,
                "V_D": V_D if V_D is not None else D,
                "Rope_D": Rope_D,
                "INPUT_LAYOUT": input_layout,
                "LAYOUT_Q_DESCALE": layout_q_descale,
                "LAYOUT_KV": layout_kv,
                "LAYOUT_OUT": layout_out,
                "KV_STORAGE_MODE": kv_storage_mode,
                "BLOCK_SIZE": block_size,
                "Q_DTYPE": q_dtype,
                "KV_DTYPE": kv_dtype,
                "OUT_DTYPE": out_dtype,
                "Q_QUANT_MODE": q_quant_mode,
                "SPARSE_MODE": mask_mode,
                "PRE_TOKENS": pre_tokens,
                "NEXT_TOKENS": next_tokens,
                "ENABLE_MASK": enable_mask,
                "ENABLE_LSE": enable_lse,
                "INNER_PRECISE": inner_precise,
                "DEVICE_ID": device_id,
                "GRAPH_PATH": graph_path,
                "SOFTMAX_SCALE": softmax_scale,
                "DATA_RANGE_Q": data_range_q,
                "DATA_RANGE_K": data_range_k,
                "DATA_RANGE_V": data_range_v,
                "ACT_SEQ_LENS_Q": list(act_seq_lens_q) if act_seq_lens_q else [],
                "ACT_SEQ_LENS_KV": list(act_seq_lens_kv) if act_seq_lens_kv else [],
                "MAX_SEQLEN_Q": max_seqlen_q,
                "MAX_SEQLEN_KV": max_seqlen_kv,
                "CU_SEQLENS_Q": list(cu_seqlens_q) if cu_seqlens_q else [],
                "CU_SEQLENS_KV": list(cu_seqlens_kv) if cu_seqlens_kv else [],
                "BLOCK_TABLE_SHAPE": list(block_table_shape)
                if block_table_shape
                else [],
                "BLOCK_TABLE_DTYPE": block_table_dtype,
                "P_SCALE_VALUE": p_scale_value,
                "P_SCALE_SHAPE": list(p_scale_shape) if p_scale_shape else [],
                "P_SCALE_DTYPE": p_scale_dtype,
                "P_SCALE_DATARANGE": p_scale_datarange,
                "SINKS_SHAPE": list(sinks_shape) if sinks_shape else [],
                "SINKS_DTYPE": sinks_dtype,
                "SINKS_DATARANGE": sinks_datarange,
                "ATTN_MASK_SHAPE": list(attn_mask_shape) if attn_mask_shape else [],
                "ATTN_MASK_DTYPE": attn_mask_dtype,
                "ATTN_MASK_DATARANGE": attn_mask_datarange,
                "Q_DESCALE_DTYPE": q_descale_dtype,
                "K_DESCALE_DTYPE": k_descale_dtype,
                "V_DESCALE_DTYPE": v_descale_dtype,
                "SEQUSED_Q_DTYPE": seqused_q_dtype,
                "SEQUSED_KV_DTYPE": seqused_kv_dtype,
                "CU_SEQLENS_Q_DTYPE": cu_seqlens_q_dtype,
                "CU_SEQLENS_KV_DTYPE": cu_seqlens_kv_dtype,
                "SOFTMAX_LSE_DTYPE": softmax_lse_dtype,
            }
        )

        data_dict = golden_mod.generate_data()

        q_npu = _to_npu(data_dict["q"])
        k_npu = _to_npu(data_dict["k"])
        v_npu = _to_npu(data_dict["v"])
        q_descale_npu = _to_npu(data_dict["q_descale"])
        k_descale_npu = _to_npu(data_dict["k_descale"])
        v_descale_npu = _to_npu(data_dict["v_descale"])
        block_table_npu = _to_npu(data_dict.get("block_table"))
        p_scale_npu = _to_npu(data_dict.get("p_scale"))
        sinks_npu = _to_npu(data_dict.get("sinks"))
        attn_mask_npu = _to_npu(data_dict.get("attn_mask"))

        cu_seqlens_q_t = (
            torch.tensor(
                data_dict["cu_seqlens_q"],
                dtype=golden_mod.get_dtype(golden_mod.CU_SEQLENS_Q_DTYPE)
                or torch.int32,
            ).npu()
            if data_dict["cu_seqlens_q"] is not None
            else None
        )
        cu_seqlens_kv_t = (
            torch.tensor(
                data_dict["cu_seqlens_kv"],
                dtype=golden_mod.get_dtype(golden_mod.CU_SEQLENS_KV_DTYPE)
                or torch.int32,
            ).npu()
            if data_dict["cu_seqlens_kv"] is not None
            else None
        )
        seqused_q_t = (
            torch.tensor(
                data_dict["act_seq_lens_q"],
                dtype=golden_mod.get_dtype(golden_mod.SEQUSED_Q_DTYPE) or torch.int32,
            ).npu()
            if data_dict["act_seq_lens_q"]
            else None
        )
        seqused_kv_t = (
            torch.tensor(
                data_dict["act_seq_lens_kv"],
                dtype=golden_mod.get_dtype(golden_mod.SEQUSED_KV_DTYPE) or torch.int32,
            ).npu()
            if data_dict["act_seq_lens_kv"]
            else None
        )

        layout_q = data_dict["query_layout"]
        layout_kv = data_dict.get("kv_layout", layout_q)
        layout_out = data_dict["attn_out_layout"]

        torch.npu.synchronize()

        logger.info("[GRAPH] 构建 metadata (quant_flash_attn_metadata)")
        self.metadata = quant_flash_attn_metadata(
            num_heads_q=data_dict["num_heads"],
            num_heads_kv=data_dict["num_key_value_heads"],
            head_dim=D,
            quant_mode=q_quant_mode,
            cu_seqlens_q=cu_seqlens_q_t,
            cu_seqlens_kv=cu_seqlens_kv_t,
            seqused_q=seqused_q_t,
            seqused_kv=seqused_kv_t,
            v_descale=v_descale_npu,
            batch_size=B,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            mask_mode=mask_mode,
            win_left=pre_tokens,
            win_right=next_tokens,
            layout_q=layout_q,
            layout_q_descale=layout_q_descale,
            layout_kv=layout_kv,
            layout_out=layout_out,
        )

        if self.metadata.device != q_npu.device:
            self.metadata = self.metadata.to(q_npu.device)

        self.q = q_npu
        self.k = k_npu
        self.v = v_npu
        self.q_descale = q_descale_npu
        self.k_descale = k_descale_npu
        self.v_descale = v_descale_npu
        self.quant_mode = q_quant_mode
        self.block_table = block_table_npu
        self.p_scale = p_scale_npu
        self.cu_seqlens_q = cu_seqlens_q_t
        self.cu_seqlens_kv = cu_seqlens_kv_t
        self.seqused_q = seqused_q_t
        self.seqused_kv = seqused_kv_t
        self.sinks = sinks_npu
        self.attn_mask = attn_mask_npu
        self.softmax_scale = data_dict["softmax_scale"]
        self.mask_mode = mask_mode
        self.win_left = pre_tokens
        self.win_right = next_tokens
        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_kv = max_seqlen_kv
        self.layout_q = layout_q
        self.layout_q_descale = layout_q_descale
        self.layout_kv = layout_kv
        self.layout_out = layout_out
        self.return_softmax_lse = enable_lse

        logger.info(
            "[GRAPH] __init__ done: q=%s, k=%s, v=%s, layout_q=%s, layout_kv=%s, "
            "metadata=%s",
            self.q.shape,
            self.k.shape,
            self.v.shape,
            self.layout_q,
            self.layout_kv,
            self.metadata.shape,
        )

    def forward(self):
        atten_out, lse_out = quant_flash_attn(
            self.q,
            self.k,
            self.v,
            self.q_descale,
            self.k_descale,
            self.v_descale,
            self.quant_mode,
            block_table=self.block_table,
            p_scale=self.p_scale,
            cu_seqlens_q=self.cu_seqlens_q,
            cu_seqlens_kv=self.cu_seqlens_kv,
            seqused_q=self.seqused_q,
            seqused_kv=self.seqused_kv,
            sinks=self.sinks,
            attn_mask=self.attn_mask,
            metadata=self.metadata,
            softmax_scale=self.softmax_scale,
            mask_mode=self.mask_mode,
            win_left=self.win_left,
            win_right=self.win_right,
            max_seqlen_q=self.max_seqlen_q,
            max_seqlen_kv=self.max_seqlen_kv,
            layout_q=self.layout_q,
            layout_q_descale=self.layout_q_descale,
            layout_kv=self.layout_kv,
            layout_out=self.layout_out,
            return_softmax_lse=self.return_softmax_lse,
        )

        if not self.return_softmax_lse:
            lse_out = None
        elif isinstance(lse_out, torch.Tensor) and lse_out.ndim == 2:
            lse_out = lse_out.transpose(0, 1).contiguous()
        return atten_out, lse_out
